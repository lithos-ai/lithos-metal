// Four-token matrix attention: 8 query rows and either 32 or CH keys,
// selected uniformly within one dispatch at GQA_SMALL_CONTEXT.
// The compiler enables this measured variant for D=128, replication=2, T=4.
// Separate query/score arrays remove the union-reuse barrier after QK. Keep the
// final reuse barrier only when this threadgroup has another tile to process.
// Core and merge share pick_chunk(); workspace capacity covers 32-key chunks.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef MMA_SG
#define MMA_SG 8
#endif
#if D != 128 || CH != 64
#error "adaptive matrix attention requires D=128 and CH=64"
#endif
using GqaTG = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
using GqaFloatTG = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
template <uint QM, uint KN>
static inline void gqa_mma_tiles(device const ushort* qkvg, device ushort* k_cache, device ushort* v_cache,
                          device const ushort* cos_t, device const ushort* sin_t,
                          device const float* q_norm, device const float* k_norm,
                          device float* part_o, device float* part_md, constant GqaParams& p,
                          uint tgid, uint lane,
                          uint sgi, uint T, uint position, threadgroup bfloat* query, threadgroup float* score, threadgroup bfloat* kv_tile, threadgroup bfloat* prob) {
  constexpr auto score_desc = matmul2d_descriptor(QM, KN, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto value_desc = matmul2d_descriptor(QM, D, KN, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  const uint rows = T * (p.heads / p.kv_heads), rep = p.heads / p.kv_heads;
  const uint ctx = position + T, chunks = (ctx + KN - 1u) / KN;
  const uint groups = (rows + QM - 1u) / QM;
  for (uint block = tgid; block < p.kv_heads * chunks * groups; block += p.n_sg) {
    const uint j = block / (chunks * groups), c = (block / groups) % chunks, r0 = (block % groups) * QM;
    if (j >= p.kv_heads) return;
    for (uint r = sgi; r < QM; r += MMA_SG) {
      float q[DL];
      const uint row = r0 + r, t = row / rep, h = j * rep + row % rep;
      for (uint e = 0; e < DL; e++) q[e] = 0.0f;
      if (row < rows) {
        load_dl(qkvg + t * p.in_stride + p.q_off + h * D + lane * DL, q);
        norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
      }
      for (uint e = 0; e < DL; e++) query[r * D + lane * DL + e] = bfloat(q[e]);
    }
    for (uint kk = sgi; kk < KN; kk += MMA_SG) {
      const uint key = c * KN + kk;
      float k[DL];
      for (uint e = 0; e < DL; e++) k[e] = 0.0f;
      if (key < position) {
        load_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, k);
      } else if (key < ctx) {
        const uint tk = key - position;
        load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, k);
        norm_rope(k, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
        if (r0 == 0) {
          store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, k);
          copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL);
        }
      }
      for (uint e = 0; e < DL; e++) kv_tile[kk * D + lane * DL + e] = bfloat(k[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    matmul2d<score_desc, execution_simdgroups<MMA_SG>> score_op;
    GqaTG qt(query, dextents<int, 2>(D, QM));
    GqaTG kt(kv_tile, dextents<int, 2>(D, KN));
    auto scores = score_op.template get_destination_cooperative_tensor<GqaTG, GqaTG, float>();
    for (uint16_t i = 0; i < scores.get_capacity(); i++) if (scores.is_valid_element(i)) scores[i] = 0.0f;
    score_op.run(qt, kt, scores);
    GqaFloatTG score_tile(score, dextents<int, 2>(KN, QM));
    scores.store(score_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = sgi; r < QM; r += MMA_SG) {
      const uint row = r0 + r, t = row / rep;
      float sc[KN / 32];
      for (uint u = 0; u < KN / 32; u++) {
        const uint key = c * KN + lane + u * 32;
        sc[u] = row < rows && key < ctx && key <= position + t ? round_bf16(round_bf16(score[r * KN + lane + u * 32]) * p.scaling) : -INFINITY;
      }
      float local_max = -INFINITY;
      for (uint u = 0; u < KN / 32; u++) local_max = max(local_max, sc[u]);
      const float m = simd_max(local_max);
      float den = 0;
      for (uint u = 0; u < KN / 32; u++) {
        const float pr = sc[u] == -INFINITY ? 0.0f : exp(sc[u] - m);
        prob[r * KN + lane + u * 32] = bfloat(pr);
        den += pr;
      }
      den = simd_sum(den);
      if (lane == 0 && row < rows) {
        const uint base = ((j * p.n_chunks_max + c) * p.rows_max + row) * 2;
        part_md[base] = m; part_md[base + 1] = den;
      }
    }
    for (uint kk = sgi; kk < KN; kk += MMA_SG) {
      const uint key = c * KN + kk;
      float v[DL];
      for (uint e = 0; e < DL; e++) v[e] = 0.0f;
      if (key < position) load_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, v);
      else if (key < ctx) load_dl(qkvg + (key - position) * p.in_stride + p.v_off + j * D + lane * DL, v);
      for (uint e = 0; e < DL; e++) kv_tile[kk * D + lane * DL + e] = bfloat(v[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    matmul2d<value_desc, execution_simdgroups<MMA_SG>> value_op;
    GqaTG pt(prob, dextents<int, 2>(KN, QM));
    GqaTG vt(kv_tile, dextents<int, 2>(D, KN));
    auto out = value_op.template get_destination_cooperative_tensor<GqaTG, GqaTG, float>();
    for (uint16_t i = 0; i < out.get_capacity(); i++) if (out.is_valid_element(i)) out[i] = 0.0f;
    value_op.run(pt, vt, out);
    for (uint16_t i = 0; i < out.get_capacity(); i++) if (out.is_valid_element(i)) {
      auto idx = out.get_multidimensional_index(i);
      const uint row = r0 + idx[1], dim = idx[0];
      if (row < rows) part_o[((j * p.n_chunks_max + c) * p.rows_max + row) * D + dim] = out[i];
    }
    if (block + p.n_sg < p.kv_heads * chunks * groups) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void gqa_decode_mma(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                          device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                          device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                          device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat query[8 * D];
  threadgroup float score[8 * CH];
  threadgroup bfloat kv_tile[CH * D];
  threadgroup bfloat prob[8 * CH];
#if STEP_STATE
  if (st->done) return;
#if LM_MODE == 1
  const uint T = st->n_inject, position = st->position - st->n_inject;
#elif LM_MODE == 2
  const uint T = st->n_chain, position = st->position + CHAIN_I;
#elif LM_MODE == 3
  const uint T = st->n_inject + st->n_chain, position = st->position - st->n_inject;
#else
  const uint T = st->t_this_step, position = st->position;
#endif
#else
  const uint T = p.t_active, position = p.position;
#endif
  if (T == 0u) return;

  if (position + T <= GQA_SMALL_CONTEXT)
    gqa_mma_tiles<8, 32>(qkvg, k_cache, v_cache, cos_t, sin_t, q_norm, k_norm, part_o, part_md, p,
                         tgid, lane, sgi, T, position, query, score, kv_tile, prob);
  else
    gqa_mma_tiles<8, CH>(qkvg, k_cache, v_cache, cos_t, sin_t, q_norm, k_norm, part_o, part_md, p,
                          tgid, lane, sgi, T, position, query, score, kv_tile, prob);
}
