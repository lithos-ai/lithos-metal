// gqa_decode_v2 + gqa_merge_v2: the attention core for long context and T > 1 (design §5.6 v2; issue #34).
//
// Block = (kv head j, batch of NSG chunks of CH keys), one *threadgroup* per block (the crew geometry: NSG = 12
// SIMD-groups per core). Per block the threadgroup first writes its R = rep·T query rows — normed, RoPE'd, BF16 —
// into threadgroup memory (R ≤ RMAX rows × D, 16 KB at 32 × 256) and appends the step's new keys that fall into the
// batch to the caches (one SIMD-group per key; a barrier makes them visible to the chunks that read them). SIMD-group
// s then takes chunk key0 + s·CH .. +CH, in passes of RG rows: for each sub-chunk of 32 keys every lane scores its
// own key against the pass's rows (the key read once from the cache, q broadcast from threadgroup memory: no
// cross-lane reduction per (key, row) — v1's `simd_sum` per pair was its cost), the sub-chunk's max, p̃ and sum per row
// come from one simd_max / simd_sum per row, and P·V runs lane-per-dim with p̃ broadcast by simd_shuffle, merged
// online into the chunk's running (m, d, o) for the pass with exact FP32 rescaling — the same fold gqa_merge applies
// to chunks, so the numbers equal v1's at chunk 32 folded hierarchically. One partial per (chunk, row); a chunk is
// 32 · {1, 2, 4} keys chosen from the context so every threadgroup keeps a block (chunk_of), and gqa_merge_v2
// folds ceil(ctx / CH) of them. K and V are read once per step from DRAM; rows beyond RG re-read the sub-chunk's
// K/V from cache. The DRAFT variant stays in v1.
//
// Macros: D, RMAX (query rows per block = rep · T_max, ≤ 32), RG (rows per pass), NSG, STEP_STATE. Params as v1
// with n_sg = the number of threadgroups and n_chunks_max = ceil(ctx_max / 32).
#ifndef RMAX
#define RMAX 32u
#endif
#ifndef RG
#define RG 4u
#endif

kernel void gqa_decode_v2(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                          device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                          device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                          device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort qtg[RMAX * D];
  const uint rep = p.heads / p.kv_heads;
#if STEP_STATE
  if (st->done) return;
  const uint T = st->t_this_step, position = st->position;
#else
  const uint T = p.t_active, position = p.position;
#endif
  const uint n_tg = p.n_sg;
  if (tgid >= n_tg) return;
  const uint rows = rep * T;
  const uint ctx = position + T;
  const uint ch = chunk_of(ctx, p.kv_heads, n_tg);
  const uint keys_per_block = NSG * ch;
  const uint n_batches = (ctx + keys_per_block - 1u) / keys_per_block;
  const uint n_blocks = p.kv_heads * n_batches;
  for (uint b = tgid; b < n_blocks; b += n_tg) {
    const uint j = b / n_batches, batch = b % n_batches;
    const uint key0 = batch * keys_per_block, key1 = min(key0 + keys_per_block, ctx);
    // phase 0: the block's query rows (row r = t·rep + i ↔ q head j·rep + i) into threadgroup memory
    for (uint r = sgi; r < rows; r += NSG) {
      const uint t = r / rep, h = j * rep + (r % rep);
      float q[DL];
      load_dl(qkvg + t * p.in_stride + p.q_off + h * D + lane * DL, q);
      norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);
      for (uint e = 0; e < DL; e++) qtg[r * D + lane * DL + e] = bf16bits(q[e]);
    }
    // phase 0b: the step's new keys inside this batch, normed + RoPE'd, appended (one SIMD-group per key)
    for (uint key = max(key0, position) + sgi; key < key1; key += NSG) {
      const uint tk = key - position;
      float kf[DL];
      load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, kf);
      norm_rope(kf, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
      store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
      copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    // phase 1: this SIMD-group's chunk, in passes of RG rows
    const uint c0 = key0 + sgi * ch, c1 = min(c0 + ch, ctx);
    const uint c = batch * NSG + sgi;                                // the chunk index the merge folds
    if (c0 < c1) {
      for (uint rg0 = 0; rg0 < rows; rg0 += RG) {
        const uint nr = min(RG, rows - rg0);
        float m_run[RG], d_run[RG], o_run[RG][DL];
        for (uint r = 0; r < RG; r++) { m_run[r] = -INFINITY; d_run[r] = 0.0f; for (uint e = 0; e < DL; e++) o_run[r][e] = 0.0f; }
        for (uint sub = 0; sub < ch; sub += 32u) {
          const uint key = c0 + sub + lane;
          const bool valid = key < c1;
          // scoring: the lane's key against the pass's rows, q broadcast from threadgroup memory
          float dot[RG];
          for (uint r = 0; r < RG; r++) dot[r] = 0.0f;
          if (valid) {
            device const uint4* kp = (device const uint4*)(k_cache + (key * p.kv_heads + j) * D);
            for (uint w = 0; w < D / 8u; w++) {
              const uint4 kw = kp[w];
              const float kf[8] = {bf16lo(kw.x), bf16hi(kw.x), bf16lo(kw.y), bf16hi(kw.y), bf16lo(kw.z), bf16hi(kw.z), bf16lo(kw.w), bf16hi(kw.w)};
              for (uint r = 0; r < RG; r++) {
                if (r < nr) {
                  const uint4 qw = ((threadgroup const uint4*)(qtg + (rg0 + r) * D))[w];
                  float acc = dot[r];
                  acc = fma(kf[0], bf16lo(qw.x), acc); acc = fma(kf[1], bf16hi(qw.x), acc);
                  acc = fma(kf[2], bf16lo(qw.y), acc); acc = fma(kf[3], bf16hi(qw.y), acc);
                  acc = fma(kf[4], bf16lo(qw.z), acc); acc = fma(kf[5], bf16hi(qw.z), acc);
                  acc = fma(kf[6], bf16lo(qw.w), acc); acc = fma(kf[7], bf16hi(qw.w), acc);
                  dot[r] = acc;
                }
              }
            }
          }
          // the sub-chunk's scores → max, p̃, sum per row (one reduction pair per row); the online merge into the pass's running state
          float pb[RG], bs[RG];
          for (uint r = 0; r < RG; r++) {
            const uint t = (rg0 + r) / rep;
            const float sc = (valid && !(key > position + t) && r < nr) ? round_bf16(round_bf16(dot[r]) * p.scaling) : -INFINITY;
            const float m_sub = simd_max(sc);
            const float pr = (sc == -INFINITY) ? 0.0f : exp(sc - m_sub);
            const float d_sub = simd_sum(pr);
            pb[r] = round_bf16(pr);
            const float m_new = max(m_run[r], m_sub);
            const float a = (m_run[r] == -INFINITY) ? 0.0f : exp(m_run[r] - m_new);
            bs[r] = (m_sub == -INFINITY) ? 0.0f : exp(m_sub - m_new);
            d_run[r] = fma(d_run[r], a, d_sub * bs[r]);
            for (uint e = 0; e < DL; e++) o_run[r][e] *= a;
            m_run[r] = m_new;
          }
          // P·V: lane-per-dim, the key's p̃ broadcast from its lane
          for (uint kk = 0; kk < 32u; kk++) {
            const uint kkey = c0 + sub + kk;
            if (kkey >= c1) break;
            float vf[DL];
            load_dl(v_cache + (kkey * p.kv_heads + j) * D + lane * DL, vf);
            for (uint r = 0; r < RG; r++) {
              if (r < nr) {
                const float pw = simd_shuffle(pb[r], ushort(kk)) * bs[r];
                for (uint e = 0; e < DL; e++) o_run[r][e] = fma(pw, vf[e], o_run[r][e]);
              }
            }
          }
        }
        for (uint r = 0; r < RG; r++) {
          if (r < nr) {
            const uint base = (j * p.n_chunks_max + c) * p.rows_max + rg0 + r;
            store_part(part_o + base * D + lane * DL, o_run[r]);
            if (lane == 0) { part_md[base * 2u] = m_run[r]; part_md[base * 2u + 1u] = d_run[r]; }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                 // the next block rewrites the query cache
  }
}

kernel void gqa_merge_v2(device const float* part_o [[buffer(0)]], device const float* part_md [[buffer(1)]], device const ushort* qkvg [[buffer(2)]],
                         device ushort* out [[buffer(3)]], constant GqaParams& p [[buffer(4)]],
#if STEP_STATE
                         device const StepState* st [[buffer(15)]],
#endif
                         uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / p.heads, h = sg % p.heads;
#if STEP_STATE
  if (st->done) return;
  const uint T = st->t_this_step, ctx = st->position + T;
#else
  const uint T = p.t_active, ctx = p.position + T;
#endif
  if (t >= T) return;
  const uint rep = p.heads / p.kv_heads;
  const uint j = h / rep, row = t * rep + (h % rep);
  const uint ch = chunk_of(ctx, p.kv_heads, p.n_sg);
  const uint n_chunks = (ctx + ch - 1u) / ch;
  float m_g = -INFINITY;
  for (uint c = 0; c < n_chunks; c++) m_g = max(m_g, part_md[((j * p.n_chunks_max + c) * p.rows_max + row) * 2u]);
  float d_g = 0.0f, o[DL];
  for (uint e = 0; e < DL; e++) o[e] = 0.0f;
  // MERGE_UNROLL chunks' (m, d, o) are requested before any is folded, and an empty chunk is skipped by a select, not a
  // `continue` — one load latency per chunk in a row was 9 µs over 33 chunks (1024 keys); the fold order is unchanged
  for (uint c0 = 0; c0 < n_chunks; c0 += MERGE_UNROLL) {
    float m_c[MERGE_UNROLL], d_c[MERGE_UNROLL], oc[MERGE_UNROLL][DL];
    for (uint u = 0; u < MERGE_UNROLL; u++) {
      const uint base = (j * p.n_chunks_max + min(c0 + u, n_chunks - 1u)) * p.rows_max + row;   // clamped: in bounds, not folded
      m_c[u] = part_md[base * 2u]; d_c[u] = part_md[base * 2u + 1u];
      load_part(part_o + base * D + lane * DL, oc[u]);
    }
    for (uint u = 0; u < MERGE_UNROLL; u++) {
      if (c0 + u >= n_chunks) break;
      const bool live = m_c[u] != -INFINITY;
      const float w = live ? exp(m_c[u] - m_g) : 0.0f;
      d_g = live ? fma(d_c[u], w, d_g) : d_g;
      for (uint e = 0; e < DL; e++) o[e] = live ? fma(w, oc[u][e], o[e]) : o[e];
    }
  }
  const float inv = d_g > 0.0f ? 1.0f / d_g : 0.0f;
  for (uint e = 0; e < DL; e++) {
    float y = round_bf16(o[e] * inv);
    if (p.has_gate) {
      const float g = bf16f(qkvg[t * p.in_stride + p.gate_off + h * D + lane * DL + e]);
      y = round_bf16(y * round_bf16(1.0f / (1.0f + exp(-g))));
    }
#if PERM_OUT
    out[t * PERM_K + perm_dest(h * D + lane * DL + e)] = bf16bits(y);
#else
    out[t * p.out_stride + h * D + lane * DL + e] = bf16bits(y);
#endif
  }
}
