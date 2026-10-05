// Matrix-accelerator attention: one threadgroup per (KV head, key chunk,
// 16-query tile). Produces the existing deterministic gqa_merge partials.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef MMA_SG
#define MMA_SG 8
#endif
#ifndef DIRECT_KV
#define DIRECT_KV 0
#endif
#ifndef MMA_PROB_FP16
#define MMA_PROB_FP16 0
#endif
#define QM 16
#define KN CH
using GqaTG = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
#if MMA_PROB_FP16
// Probabilities are in [0, 1]. Keep the BF16 Q/K/V range while reducing
// rounding error in the unnormalized softmax partition used by the value MMA.
using GqaProbability = half;
#else
using GqaProbability = bfloat;
#endif
using GqaProbTG = tensor<threadgroup GqaProbability, dextents<int, 2>, tensor_inline>;
#if DIRECT_KV
using GqaDevice = tensor<device bfloat, dextents<int, 2>, tensor_inline>;
#endif
using GqaFloatTG = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
constexpr constant auto score_desc = matmul2d_descriptor(QM, KN, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
constexpr constant auto value_desc = matmul2d_descriptor(QM, D, KN, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
union GqaTileScratch { bfloat query[QM * D]; float score[QM * KN]; };

kernel void gqa_decode_mma(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                          device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                          device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                          device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
#if DRAFT
                          device const ushort* kvp [[buffer(11)]],
#endif
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup GqaTileScratch scratch;
#if !DIRECT_KV
  threadgroup bfloat kv_tile[KN * D];
#endif
  threadgroup GqaProbability prob[QM * KN];
#if STEP_STATE
  if (st->done) return;
#if DRAFT
  const uint T = p.t_active, position = st->drafter_ctx_len, n_new = st->n_inject;
#elif LM_MODE == 1
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
#if DRAFT
  const uint n_new = p.pad0;
#endif
#endif
#if DRAFT
  const uint qpos0 = position + n_new;
#else
  const uint qpos0 = position;
#endif
  if (T == 0u) return;
  const uint rows = T * (p.heads / p.kv_heads), rep = p.heads / p.kv_heads;
  const uint ctx = qpos0 + T, chunks = (ctx + KN - 1u) / KN;
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
#if !DIRECT_KV
#if DRAFT
        norm_rope(q, q_norm, cos_t + (qpos0 + t) * D, sin_t + (qpos0 + t) * D, p.eps, lane);
#else
        norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
#endif
#endif
      }
      for (uint e = 0; e < DL; e++) scratch.query[r * D + lane * DL + e] = bfloat(q[e]);
    }
#if !DIRECT_KV
    for (uint kk = sgi; kk < KN; kk += MMA_SG) {
      const uint key = c * KN + kk;
      float k[DL];
      for (uint e = 0; e < DL; e++) k[e] = 0.0f;
      if (key < position) {
        load_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, k);
#if DRAFT
      } else if (key < qpos0) {
        const uint tk = key - position;
        load_dl(kvp + tk * p.pad1 + j * D + lane * DL, k);
        norm_rope(k, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
        if (r0 == 0) {
          store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, k);
          copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL,
                  kvp + tk * p.pad1 + p.kv_heads * D + j * D + lane * DL);
        }
      } else if (key < ctx) {
        const uint tk = key - qpos0;
        load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, k);
        norm_rope(k, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
#else
      } else if (key < ctx) {
        const uint tk = key - position;
        load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, k);
        norm_rope(k, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
        if (r0 == 0) {
          store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, k);
          copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL);
        }
#endif
      }
      for (uint e = 0; e < DL; e++) kv_tile[kk * D + lane * DL + e] = bfloat(k[e]);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    matmul2d<score_desc, execution_simdgroups<MMA_SG>> score_op;
    GqaTG qt(scratch.query, dextents<int, 2>(D, QM));
#if DIRECT_KV
    GqaDevice kt_cache((device bfloat*)k_cache, dextents<int, 2>(p.kv_heads * D, p.ctx_max));
    auto kt = kt_cache.slice<D, KN>(j * D, c * KN);
#else
    GqaTG kt(kv_tile, dextents<int, 2>(D, KN));
#endif
    auto scores = score_op.get_destination_cooperative_tensor<GqaTG, decltype(kt), float>();
    for (uint16_t i = 0; i < scores.get_capacity(); i++) if (scores.is_valid_element(i)) scores[i] = 0.0f;
    score_op.run(qt, kt, scores);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GqaFloatTG score_tile(scratch.score, dextents<int, 2>(KN, QM));
    scores.store(score_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = sgi; r < QM; r += MMA_SG) {
      const uint row = r0 + r, t = row / rep;
      float sc[KN / 32];
      for (uint u = 0; u < KN / 32; u++) {
        const uint key = c * KN + lane + u * 32;
#if DRAFT
        sc[u] = row < rows && key < ctx ? round_bf16(round_bf16(scratch.score[r * KN + lane + u * 32]) * p.scaling) : -INFINITY;
#else
        sc[u] = row < rows && key < ctx && key <= position + t ? round_bf16(round_bf16(scratch.score[r * KN + lane + u * 32]) * p.scaling) : -INFINITY;
#endif
      }
      float local_max = -INFINITY;
      for (uint u = 0; u < KN / 32; u++) local_max = max(local_max, sc[u]);
      const float m = simd_max(local_max);
      float den = 0;
      for (uint u = 0; u < KN / 32; u++) {
        const float pr = sc[u] == -INFINITY ? 0.0f : exp(sc[u] - m);
        prob[r * KN + lane + u * 32] = GqaProbability(pr);
        den += pr;
      }
      den = simd_sum(den);
      if (lane == 0 && row < rows) {
        const uint base = ((j * p.n_chunks_max + c) * p.rows_max + row) * 2;
        part_md[base] = m; part_md[base + 1] = den;
      }
    }
#if !DIRECT_KV
    for (uint kk = sgi; kk < KN; kk += MMA_SG) {
      const uint key = c * KN + kk;
      float v[DL];
      for (uint e = 0; e < DL; e++) v[e] = 0.0f;
      if (key < position) load_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, v);
#if DRAFT
      else if (key < qpos0) load_dl(kvp + (key - position) * p.pad1 + p.kv_heads * D + j * D + lane * DL, v);
      else if (key < ctx) load_dl(qkvg + (key - qpos0) * p.in_stride + p.v_off + j * D + lane * DL, v);
#else
      else if (key < ctx) load_dl(qkvg + (key - position) * p.in_stride + p.v_off + j * D + lane * DL, v);
#endif
      for (uint e = 0; e < DL; e++) kv_tile[kk * D + lane * DL + e] = bfloat(v[e]);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    matmul2d<value_desc, execution_simdgroups<MMA_SG>> value_op;
    GqaProbTG pt(prob, dextents<int, 2>(KN, QM));
#if DIRECT_KV
    GqaDevice vt_cache((device bfloat*)v_cache, dextents<int, 2>(p.kv_heads * D, p.ctx_max));
    auto vt = vt_cache.slice<D, KN>(j * D, c * KN);
#else
    GqaTG vt(kv_tile, dextents<int, 2>(D, KN));
#endif
    auto out = value_op.get_destination_cooperative_tensor<GqaProbTG, decltype(vt), float>();
    for (uint16_t i = 0; i < out.get_capacity(); i++) if (out.is_valid_element(i)) out[i] = 0.0f;
    value_op.run(pt, vt, out);
    for (uint16_t i = 0; i < out.get_capacity(); i++) if (out.is_valid_element(i)) {
      auto idx = out.get_multidimensional_index(i);
      const uint row = r0 + idx[1], dim = idx[0];
      if (row < rows) part_o[((j * p.n_chunks_max + c) * p.rows_max + row) * D + dim] = out[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#if DIRECT_KV
// Prepare Q once per target row and write new K/V before direct matrix reads.
// QKV is dead after attention; the preceding projection overwrites it each step.
kernel void gqa_prepare_mma(device ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                           device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                           device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                           constant GqaParams& p [[buffer(9)]], device const StepState* st [[buffer(15)]],
                           uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  if (st->done) return;
  const uint idx = group * 4 + sg, T = st->t_this_step, position = st->position;
  if (idx < T * p.heads) {
    const uint t = idx / p.heads, h = idx % p.heads;
    float q[DL];
    device ushort* row = qkvg + t * p.in_stride + p.q_off + h * D + lane * DL;
    load_dl(row, q);
    norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
    store_dl(row, q);
  }
  if (idx < T * p.kv_heads) {
    const uint t = idx / p.kv_heads, h = idx % p.kv_heads;
    float k[DL];
    load_dl(qkvg + t * p.in_stride + p.k_off + h * D + lane * DL, k);
    norm_rope(k, k_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
    store_dl(k_cache + ((position + t) * p.kv_heads + h) * D + lane * DL, k);
    copy_dl(v_cache + ((position + t) * p.kv_heads + h) * D + lane * DL,
            qkvg + t * p.in_stride + p.v_off + h * D + lane * DL);
  }
}

#endif
