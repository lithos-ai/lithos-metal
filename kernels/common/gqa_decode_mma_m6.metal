// M6 target attention: 16 query/head rows, 32-key matrix tiles. Two
// tiles share one v1 64-key softmax partition and the original merge workspace.
// Keep v1's BF16 score/probability rounding and FP32 denominator/accumulation.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef MMA_SG
#define MMA_SG 8
#endif
#if K_CACHE_TYPE || V_CACHE_TYPE
// Quantized reconstruction and rotated Q are FP32. Do not introduce a BF16
// operand rounding that is absent from the v1 attention contract.
#define QM 8
#define KN 16
using M6Operand = float;
#else
#define QM 16
#define KN 32
using M6Operand = bfloat;
#endif
using M6Tile = tensor<threadgroup M6Operand, dextents<int, 2>, tensor_inline>;
using M6FloatTile = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
constexpr constant auto m6_score_desc = matmul2d_descriptor(QM, KN, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
constexpr constant auto m6_value_desc = matmul2d_descriptor(QM, D, KN, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);

kernel void gqa_decode_mma_m6(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
    device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
    device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
    device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
#if TURBO4_PREPARED_Q
    device const float* rotated_q [[buffer(13)]],
#endif
#if STEP_STATE
    device const StepState* st [[buffer(15)]],
#endif
    uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup M6Operand query[QM * D], kv_tile[KN * D], prob[QM * CH];
  threadgroup float score[QM * CH];
#if STEP_STATE
  if (st->done) return;
  const uint T = st->t_this_step, position = st->position;
#else
  const uint T = p.t_active, position = p.position;
#endif
  if (!T) return;
#if M6_MMA_DISPATCH
  if (T < 8u || position + T < 1024u) return;
#if M6_PV_MERGE
  if (T < M6_PV_MIN_ROWS) return;
#elif M6_PV_FALLBACK
  if (T >= M6_PV_MIN_ROWS) return;
#endif
#endif
  const uint rep = p.heads / p.kv_heads, rows = T * rep, ctx = position + T;
  const uint ch = pick_chunk(ctx, p.kv_heads, rep, p.n_sg, p.n_chunks_max);
  const uint chunks = (ctx + ch - 1) / ch, groups = (rows + QM - 1) / QM;
  matmul2d<m6_score_desc, execution_simdgroups<MMA_SG>> score_op;
  matmul2d<m6_value_desc, execution_simdgroups<MMA_SG>> value_op;
  for (uint block = tgid; block < p.kv_heads * chunks * groups; block += 48u) {
    const uint j = block / (chunks * groups), c = (block / groups) % chunks, r0 = (block % groups) * QM;
    for (uint r = sgi; r < QM; r += MMA_SG) {
      const uint row = r0 + r, t = row / rep, h = j * rep + row % rep;
      float q[DL];
      for (uint e = 0; e < DL; ++e) q[e] = 0;
      if (row < rows) {
#if TURBO4_PREPARED_Q
        load_part(rotated_q + (t * p.heads + h) * D + lane * DL, q);
#else
        load_dl(qkvg + t * p.in_stride + p.q_off + h * D + lane * DL, q);
        norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
#endif
      }
      for (uint e = 0; e < DL; ++e) query[r * D + lane * DL + e] = M6Operand(q[e]);
    }
    for (uint offset = 0; offset < ch; offset += KN) {
      for (uint kk = sgi; kk < KN; kk += MMA_SG) {
        const uint key = c * ch + offset + kk;
        float k[DL];
        for (uint e = 0; e < DL; ++e) k[e] = 0;
        if (offset + kk < ch && key < ctx) {
          if (key < position) kv_attention_load<K_CACHE_TYPE>(k_cache, key * p.kv_heads + j, k, lane);
          else {
            const uint tk = key - position;
            load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, k);
            norm_rope(k, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
            kv_attention_encode<K_CACHE_TYPE>(k_cache, key * p.kv_heads + j, k, lane, r0 == 0);
            if (r0 == 0) kv_copy<V_CACHE_TYPE>(v_cache, key * p.kv_heads + j, qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL, lane);
          }
        }
        for (uint e = 0; e < DL; ++e) kv_tile[kk * D + lane * DL + e] = M6Operand(k[e]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      M6Tile qt(query, dextents<int, 2>(D, QM)), kt(kv_tile, dextents<int, 2>(D, KN));
      auto scores = score_op.get_destination_cooperative_tensor<M6Tile, M6Tile, float>();
      for (uint16_t i = 0; i < scores.get_capacity(); ++i) if (scores.is_valid_element(i)) scores[i] = 0;
      score_op.run(qt, kt, scores);
      for (uint16_t i = 0; i < scores.get_capacity(); ++i) if (scores.is_valid_element(i)) {
        auto ix = scores.get_multidimensional_index(i);
        score[ix[1] * CH + offset + ix[0]] = scores[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = sgi; r < QM; r += MMA_SG) {
      const uint row = r0 + r, t = row / rep;
      float sc[CH / 32], local_max = -INFINITY;
      for (uint u = 0; u < CH / 32; ++u) {
        const uint kk = lane + u * 32, key = c * ch + kk;
        sc[u] = row < rows && kk < ch && key < ctx && key <= position + t ? round_bf16(round_bf16(score[r * CH + kk]) * p.scaling) : -INFINITY;
        local_max = max(local_max, sc[u]);
      }
      const float m = simd_max(local_max);
      float den = 0;
      for (uint u = 0; u < CH / 32; ++u) {
        const float pr = sc[u] == -INFINITY ? 0.f : exp(sc[u] - m);
        prob[r * CH + lane + u * 32] = M6Operand(round_bf16(pr));
        den += pr;
      }
      den = simd_sum(den);
      if (!lane && row < rows) {
        const uint base = ((j * p.n_chunks_max + c) * p.rows_max + row) * 2;
        part_md[base] = m; part_md[base + 1] = den;
      }
    }
#if M6_PV_MERGE
    // Reuse the FP32 partial allocation for the already-rounded probabilities.
    // The second stage reconstructs each PV partial and folds it in c order.
    for (uint r = sgi; r < QM; r += MMA_SG) {
      const uint row = r0 + r;
      if (row < rows) for (uint kk = lane; kk < CH; kk += 32u)
        reinterpret_cast<device ushort*>(part_o)[((j * p.n_chunks_max + c) * p.rows_max + row) * CH + kk] = bf16bits(float(prob[r * CH + kk]));
    }
#else
    auto out = value_op.get_destination_cooperative_tensor<M6Tile, M6Tile, float>();
    for (uint16_t i = 0; i < out.get_capacity(); ++i) if (out.is_valid_element(i)) out[i] = 0;
    for (uint offset = 0; offset < ch; offset += KN) {
      for (uint kk = sgi; kk < KN; kk += MMA_SG) {
        const uint key = c * ch + offset + kk;
        float v[DL];
        for (uint e = 0; e < DL; ++e) v[e] = 0;
        if (offset + kk < ch && key < ctx) {
          if (key < position) kv_attention_load<V_CACHE_TYPE>(v_cache, key * p.kv_heads + j, v, lane);
          else {
            load_dl(qkvg + (key - position) * p.in_stride + p.v_off + j * D + lane * DL, v);
            kv_attention_encode<V_CACHE_TYPE>(nullptr, 0, v, lane, false);
          }
        }
        for (uint e = 0; e < DL; ++e) kv_tile[kk * D + lane * DL + e] = M6Operand(v[e]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      M6Tile pt_all(prob, dextents<int, 2>(CH, QM)), vt(kv_tile, dextents<int, 2>(D, KN));
      auto pt = pt_all.slice<KN, QM>(offset, 0);
      value_op.run(pt, vt, out);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < out.get_capacity(); ++i) if (out.is_valid_element(i)) {
      auto ix = out.get_multidimensional_index(i);
      const uint row = r0 + ix[1];
      if (row < rows) part_o[((j * p.n_chunks_max + c) * p.rows_max + row) * D + ix[0]] = out[i];
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
