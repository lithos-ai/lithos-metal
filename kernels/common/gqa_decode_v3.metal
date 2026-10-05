// gqa_decode_v3: the attention core for few query rows as ONE dispatch — core and merge together (#113).
//
// Block = one query row (token t, q head h) of kv head j: one *threadgroup* of NSG3 SIMD-groups per block, the
// dispatch heads·T threadgroups. Lane ℓ owns dims [ℓ·D/32, (ℓ+1)·D/32) of every vector (v1's layout). SIMD-group s
// takes the keys s, s + NSG3, s + 2·NSG3, … of the context: for each it reads k (the cache, or the step's own new
// key normed + RoPE'd from the projection and, for the kv head's first row, appended to the caches — writer and
// reader are the same SIMD-group), scores it (q·k by one simd_sum, bf16(bf16(q·k) · scaling), causal inside the
// step), and folds it into its running online softmax (m, d, o) with exact FP32 rescaling, p̃ = bf16(exp(s − m))
// rounded like the reference's P. Cached pairs share their maximum and the rescaling of prior accumulators;
// an unpaired tail and new causal keys are folded individually. The NSG3 partials then meet in threadgroup memory
// and every SIMD-group folds one
// slice of D/NSG3 dims (lane ℓ = partial ℓ: one simd_max, one simd_sum per value) and writes it, normalized, rounded
// to BF16 and times bf16(σ(gate)) when the projection carries the gate — no partial workspace, no second dispatch.
//
// Why: at a short context v2 hands every SIMD-group a 32-key chunk to walk alone and 8 blocks to the whole GPU
// (the 0.6B: 12.6 µs of core + 4.6 of merge for 128 keys); here a head's keys are spread over 32 SIMD-groups
// (4 keys each at 128) and its rows over the cores (16 threadgroups for 16 heads at T = 1), the structure of MLX's
// decode attention. The work per core grows with rows · ctx, so this is the kernel for rep · T ≤ ~8 rows (T = 1:
// plain decode and an LM drafter's chain step); v2 keeps T > 1 and v1 the 32-row passes (decode-kernels.md §11).
//
// Macros: D (head dim, multiple of 32), NSG3 (SIMD-groups per threadgroup, D / (4·NSG3) slices per SIMD-group in the
// fold: 32 → 1024 threads), STEP_STATE, LM_MODE / CHAIN_I as v1, PERM_OUT (the consumer tile's order). Params as v1
// with n_sg = the threadgroups dispatched and pad1 = the gate rows' stride (0: out_stride). Buffers: 0 projection,
// 1–2 caches, 3–4 RoPE tables, 5–6 norms, 7 out, 9 params, 10 the gate value [T, heads·D] (read when has_gate),
// 15 StepState.
#ifndef NSG3
#define NSG3 32u
#endif
#ifndef SINGLE_BLOCK
#define SINGLE_BLOCK 0
#endif
#define MRG_STRIDE (D + 4u)          // floats per partial in the fold buffer: o[D], m, d, then padding to 16 bytes

kernel void gqa_decode_v3(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                          device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                          device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                          device ushort* out [[buffer(7)]], constant GqaParams& p [[buffer(9)]],
                          device const ushort* gate [[buffer(10)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup float mrg[NSG3 * MRG_STRIDE];
  threadgroup float query[D];
  const uint rep = p.heads / p.kv_heads;
#if STEP_STATE
  if (st->done) return;
#if LM_MODE == 1
  const uint T = st->n_inject, position = st->position - st->n_inject;             // an LM drafter's ingest: the committed rows it has not seen yet
#elif LM_MODE == 2
  const uint T = st->n_chain, position = st->position + CHAIN_I;                   // an LM drafter's chain step i: one row at position + i
#elif LM_MODE == 3
  const uint T = st->n_inject + st->n_chain, position = st->position - st->n_inject;   // its first step: the ingest rows, then the anchor
#else
  const uint T = st->t_this_step, position = st->position;
#endif
#else
  const uint T = p.t_active, position = p.position;
#endif
  if (T == 0u) return;
  const uint rows = rep * T;
  const uint ctx = position + T;
  const uint n_blocks = p.kv_heads * rows;                       // block = (kv head, row): the row's token t = r / rep, q head j·rep + r % rep
#if SINGLE_BLOCK
  const uint b = tgid;
  if (b < n_blocks) {
#else
  for (uint b = tgid; b < n_blocks; b += p.n_sg) {
#endif
    const uint j = b / rows, r = b % rows;
    const uint t = r / rep, h = j * rep + (r % rep);
    // Prologue: the row's query, normed and RoPE'd, DL dims per lane.
    float q[DL];
    // Every key group uses the same query. Normalize and rotate it once.
    if (sgi == 0u) {
      load_dl(qkvg + t * p.in_stride + p.q_off + h * D + lane * DL, q);
      norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
      for (uint e = 0; e < DL; e++) query[lane * DL + e] = q[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = 0; e < DL; e++) q[e] = query[lane * DL + e];
    // this SIMD-group's keys: s, s + NSG3, … — each scored and folded into the running (m, d, o)
    float m_run = -INFINITY, d_run = 0.0f, o_run[DL];
    for (uint e = 0; e < DL; e++) o_run[e] = 0.0f;
    uint key = sgi;

    // Two cached keys share one rescaling of the previous softmax state. Keep
    // score/probability BF16 boundaries; the tail and new causal keys follow below.
    for (; key + NSG3 < position; key += 2u * NSG3) {
      float scores[2u];
      float m_new = m_run;
#pragma clang loop unroll(full)
      for (uint pair = 0; pair < 2u; pair++) {
        float kf[DL];
        load_dl(k_cache + ((key + pair * NSG3) * p.kv_heads + j) * D + lane * DL, kf);
        float dot = 0.0f;
        for (uint e = 0; e < DL; e++) dot = fma(q[e], kf[e], dot);
        dot = simd_sum(dot);
        scores[pair] = round_bf16(round_bf16(dot) * p.scaling);
        m_new = max(m_new, scores[pair]);
      }
      const float a = m_run == -INFINITY ? 0.f : exp(m_run - m_new);
      d_run *= a;
      for (uint e = 0; e < DL; e++) o_run[e] *= a;
#pragma clang loop unroll(full)
      for (uint pair = 0; pair < 2u; pair++) {
        const float pr = exp(scores[pair] - m_new), pb = round_bf16(pr);
        float vf[DL];
        load_dl(v_cache + ((key + pair * NSG3) * p.kv_heads + j) * D + lane * DL, vf);
        d_run += pr;
        for (uint e = 0; e < DL; e++) o_run[e] = fma(pb, vf[e], o_run[e]);
      }
      m_run = m_new;
    }
    for (; key < position; key += NSG3) {
      float kf[DL], vf[DL];
      load_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
      load_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, vf);
      float dot = 0.0f;
      for (uint e = 0; e < DL; e++) dot = fma(q[e], kf[e], dot);
      dot = simd_sum(dot);
      const float sc = round_bf16(round_bf16(dot) * p.scaling);   // causal inside the step
      if (sc != -INFINITY) {
        const float m_new = max(m_run, sc);
        const float a = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float pr = exp(sc - m_new), pb = round_bf16(pr);
        d_run = fma(d_run, a, pr);
        for (uint e = 0; e < DL; e++) o_run[e] = fma(pb, vf[e], o_run[e] * a);
        m_run = m_new;
      }
    }
    for (; key < ctx; key += NSG3) {
      float kf[DL], vf[DL];
        const uint tk = key - position;
        load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, kf);
        norm_rope(kf, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
        load_dl(qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL, vf);
        if (r == 0u) {                                           // the kv head's first row appends k (normed, RoPE'd) and v once
          store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
          copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL);
        }
      float dot = 0.0f;
      for (uint e = 0; e < DL; e++) dot = fma(q[e], kf[e], dot);
      dot = simd_sum(dot);
      const float sc = (key > position + t) ? -INFINITY : round_bf16(round_bf16(dot) * p.scaling);   // causal inside the step
      if (sc != -INFINITY) {
        const float m_new = max(m_run, sc);
        const float a = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float pr = exp(sc - m_new), pb = round_bf16(pr);
        d_run = fma(d_run, a, pr);
        for (uint e = 0; e < DL; e++) o_run[e] = fma(pb, vf[e], o_run[e] * a);
        m_run = m_new;
      }
    }
    // the fold: the NSG3 partials meet in threadgroup memory …
    for (uint e = 0; e < DL; e++) mrg[sgi * MRG_STRIDE + lane * DL + e] = o_run[e];
    if (lane == 0) { mrg[sgi * MRG_STRIDE + D] = m_run; mrg[sgi * MRG_STRIDE + D + 1u] = d_run; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // … and SIMD-group s folds the dims [4·slice, 4·slice + 4) for slice = s, s + NSG3, …: lane ℓ holds partial ℓ
    const float m_l = (lane < NSG3) ? mrg[lane * MRG_STRIDE + D] : -INFINITY;
    const float m_g = simd_max(m_l);
    const float w = (m_l == -INFINITY) ? 0.0f : exp(m_l - m_g);
    const float d_l = (lane < NSG3) ? mrg[lane * MRG_STRIDE + D + 1u] : 0.0f;
    const float d_g = simd_sum(w * d_l);
    const float inv = d_g > 0.0f ? 1.0f / d_g : 0.0f;
    for (uint slice = sgi; slice < D / 4u; slice += NSG3) {
      float o4[4];
      for (uint e = 0; e < 4u; e++) o4[e] = simd_sum(w * ((lane < NSG3) ? mrg[lane * MRG_STRIDE + slice * 4u + e] : 0.0f));
      if (lane == 0) {
        for (uint e = 0; e < 4u; e++) {
          const uint dim = slice * 4u + e;
          float y = round_bf16(o4[e] * inv);
          if (p.has_gate) {
            const float g = bf16f(gate[t * (p.pad1 ? p.pad1 : p.out_stride) + p.gate_off + h * D + dim]);
            y = round_bf16(y * round_bf16(1.0f / (1.0f + exp(-g))));
          }
#if PERM_OUT
          out[t * PERM_K + perm_dest(h * D + dim)] = bf16bits(y);   // the consumer tile's x' (its K = heads · D)
#else
          out[t * p.out_stride + h * D + dim] = bf16bits(y);
#endif
        }
      }
    }
#if !SINGLE_BLOCK
    threadgroup_barrier(mem_flags::mem_threadgroup);              // the next block rewrites the fold buffer and query
#endif
  }
}
