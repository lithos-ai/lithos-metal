// gqa_decode + gqa_merge: the full-attention mixer for T new tokens (design §5.6; issue #21).
//
// Block = (kv head j, chunk c of ch key positions, row group of RBMAX query rows); a SIMD-group takes static
// slices of the kv_heads × n_chunks × n_row_groups blocks, n_chunks = ceil((position + T) / ch) computed in-kernel
// with ch = pick_chunk(...) ≤ CH (gqa_common.metal: smaller chunks while the blocks would leave the crew idle)
// so the work grows with the context under the fixed crew geometry. Lane ℓ owns dims [ℓ·D/32, (ℓ+1)·D/32) of
// every vector. For its block a SIMD-group:
//   1. prologue — its query rows of kv head j (q head h = j·rep + i, token t; row = t·rep + i):
//      per-head RMSNorm (FP32 Σq², rsqrt(mean + eps), · (1 + w) → BF16) then RoPE in the load-time
//      head-dim permutation (pairs (i, i + D/2) = lane ℓ and lane ℓ ^ 16; cos = 1 / sin = 0 outside the rotary
//      dims), each product and the sum rounded to BF16 like the reference's elementwise BF16 math;
//   2. keys — positions < position come from the cache; the T new positions are normed + RoPE'd from the
//      projection (the block owning the chunk that holds them also appends k and v to the caches: writer and
//      readers derive them from the same input, so no read-after-write inside the dispatch);
//      scores s = bf16(bf16(q·k) · scaling) (the reference's BF16 matmul and scaling), causal inside the step;
//   3. per chunk: m_c = max s, p̃ = bf16(exp(s − m_c)) rounded like the reference's P, d_c = Σ exp(s − m_c) in
//      FP32, o_c = Σ p̃·v in FP32; the (o_c, m_c, d_c) partials go to a workspace.
// gqa_merge (one SIMD-group per (token, q head)) folds the chunks in order — deterministic — normalizes, rounds to
// BF16, multiplies by bf16(σ(gate)) (the reference's `attn · sigmoid(gate)` in BF16) and writes [T, H·D].
//
// Macros: D (head dim, multiple of 32), CH (keys per chunk), RBMAX (query rows per pass; rows beyond re-stream the
// chunk). Params carry the projection's column offsets (q | gate | k | v), the strides, position and T.
//
// DRAFT=1 — the drafter's block attention (design §5.8, issue #24), the same blocks and passes with three key
// sources: the drafter's injected-context cache [0, position), n_new context positions whose k/v come from a second
// projection (buffer 11, the features' k/v; normed, RoPE'd and appended to the cache like new tokens), and the block's
// own T = γ rows (normed, RoPE'd, never appended). The queries are the block at positions position + n_new + t; no
// mask (the block is bidirectional). position = StepState.drafter_ctx_len and n_new = StepState.n_inject (or the
// params' position / pad0); T = the params' t_active (γ is static); pad1 = the second projection's row stride.
// (the params record and the helpers are in gqa_common.metal, concatenated ahead of this file)

kernel void gqa_decode(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                       device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                       device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                       device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
#if STEAL
                       device atomic_uint* cursors [[buffer(10)]],
#if STEAL_HITS
                       device atomic_uint* hits [[buffer(12)]],
#endif
#endif
#if DRAFT
                       device const ushort* kvp [[buffer(11)]],
#endif
#if STEP_STATE
                       device const StepState* st [[buffer(15)]],
#endif
                       uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint rep = p.heads / p.kv_heads;
#if DRAFT
#if STEP_STATE
  if (st->done) return;
  const uint position = st->drafter_ctx_len, n_new = st->n_inject;
#else
  const uint position = p.position, n_new = p.pad0;
#endif
  const uint T = p.t_active;                                   // the block size γ (static)
#elif STEP_STATE
  if (st->done) return;
#if LM_MODE == 1
  const uint T = st->n_inject, position = st->position - st->n_inject, n_new = 0u;   // an LM drafter's ingest: the committed rows it has not seen yet
#elif LM_MODE == 2
  const uint T = st->n_chain, position = st->position + CHAIN_I, n_new = 0u;         // an LM drafter's chain step i: one row at position + i
#elif LM_MODE == 3
  const uint T = st->n_inject + st->n_chain, position = st->position - st->n_inject, n_new = 0u;   // its first step: the ingest rows, then the anchor
#else
  const uint T = st->t_this_step, position = st->position, n_new = 0u;
#endif
#else
  const uint T = p.t_active, position = p.position, n_new = 0u;
#endif
  const uint qpos0 = position + n_new;                         // the first query position (DRAFT: after the new context)
  const uint rows = rep * T;
  const uint n_rg = (rows + RBMAX - 1u) / RBMAX;
  const uint ctx = qpos0 + T;                                  // keys: [0, position) cached, [position, qpos0) new, [qpos0, ctx) this step
  const uint ch = pick_chunk(ctx, p.kv_heads, rep, p.n_sg, p.n_chunks_max);   // keys per chunk this step (≤ CH)
  const uint n_chunks = (ctx + ch - 1u) / ch;
  const uint n_blocks = p.kv_heads * n_chunks * n_rg;          // block = (kv head, chunk, row group)
#if STEAL
  StealScan scan = steal_begin();                              // own slice first, then steal (kernels/common/steal.metal, #44)
  for (uint b = steal_next(cursors, n_blocks, p.nominal_sg, sg, lane, scan); b != STEAL_NONE; b = steal_next(cursors, n_blocks, p.nominal_sg, sg, lane, scan)) {
#if STEAL_HITS
    if (lane == 0) atomic_fetch_add_explicit(&hits[b], 1u, memory_order_relaxed);
#endif
#else
  for (uint b = sg; b < n_blocks; b += p.n_sg) {
#endif
    const uint rg = b % n_rg, c = (b / n_rg) % n_chunks, j = b / (n_rg * n_chunks);
    const uint k0 = c * ch, k1 = min(k0 + ch, ctx);
    const uint r0 = rg * RBMAX;
    const uint nr = min(RBMAX, rows - r0);
    // prologue: this row group's queries, normed and RoPE'd, DL dims per lane
    float q[RBMAX][DL];
    for (uint r = 0; r < RBMAX; r++) {
      if (r < nr) {
        const uint row = r0 + r, t = row / rep, h = j * rep + (row % rep);
        load_dl(qkvg + t * p.in_stride + p.q_off + h * D + lane * DL, q[r]);
        norm_rope(q[r], q_norm, cos_t + (qpos0 + t) * D, sin_t + (qpos0 + t) * D, p.eps, lane);
      } else {
        for (uint e = 0; e < DL; e++) q[r][e] = 0.0f;
      }
    }
    // pass 1: scores; key k0 + g*32 + kk lands on lane kk of group g (compile-time g, r)
    float s_keep[CH / 32u][RBMAX];
    float m_c[RBMAX];
    for (uint r = 0; r < RBMAX; r++) m_c[r] = -INFINITY;
    for (uint g = 0; g < CH / 32u; g++) {
      for (uint r = 0; r < RBMAX; r++) s_keep[g][r] = -INFINITY;
      for (uint kk = 0; kk < 32u; kk++) {
        const uint key = k0 + g * 32u + kk;
        if (key >= k1) break;
        float kf[DL];
        if (key < position) {
          load_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
#if DRAFT
        } else if (key < qpos0) {                          // a new context position: k/v from the features' projection, appended
          const uint tk = key - position;
          load_dl(kvp + tk * p.pad1 + j * D + lane * DL, kf);
          norm_rope(kf, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
          if (rg == 0) {
            store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
            copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, kvp + tk * p.pad1 + p.kv_heads * D + j * D + lane * DL);
          }
        } else {                                           // the block's own key: normed and RoPE'd, never appended
          const uint tk = key - qpos0;
          load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, kf);
          norm_rope(kf, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
        }
#else
        } else {
          const uint tk = key - position;
          load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, kf);
          norm_rope(kf, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
          if (rg == 0) {                                   // append k (normed, RoPE'd) and v to the caches once
            store_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
            copy_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, qkvg + tk * p.in_stride + p.v_off + j * D + lane * DL);
          }
        }
#endif
        for (uint r = 0; r < RBMAX; r++) {
          if (r < nr) {
            const uint t = (r0 + r) / rep;
            float dot = 0.0f;
            for (uint e = 0; e < DL; e++) dot = fma(q[r][e], kf[e], dot);
            dot = simd_sum(dot);
#if DRAFT
            const float sc = round_bf16(round_bf16(dot) * p.scaling);          // no mask: the block attends bidirectionally
#else
            const float sc = (key > position + t) ? -INFINITY : round_bf16(round_bf16(dot) * p.scaling);
#endif
            if (lane == kk) s_keep[g][r] = sc;
            m_c[r] = max(m_c[r], sc);
          }
        }
      }
    }
    // pass 2: p̃ = bf16(exp(s - m_c)), d in FP32 from the unrounded p, o = Σ p̃ v (lane-per-dim, p broadcast)
    float d_c[RBMAX], o[RBMAX][DL];
    for (uint r = 0; r < RBMAX; r++) { d_c[r] = 0.0f; for (uint e = 0; e < DL; e++) o[r][e] = 0.0f; }
    for (uint g = 0; g < CH / 32u; g++) {
      for (uint kk = 0; kk < 32u; kk += PV_UNROLL) {
        // the values of PV_UNROLL keys are requested before any is consumed: the loads overlap instead of each
        // waiting its turn (a short context at T = 1 is this pass's latency chain: 38 → 12 µs per layer on the M5 Pro)
        float vf[PV_UNROLL][DL];
        for (uint u = 0; u < PV_UNROLL; u++) {
          const uint key = k0 + g * 32u + kk + u;
          if (key >= k1) { for (uint e = 0; e < DL; e++) vf[u][e] = 0.0f; }
          else if (key < position) load_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, vf[u]);
#if DRAFT
          else if (key < qpos0) load_dl(kvp + (key - position) * p.pad1 + p.kv_heads * D + j * D + lane * DL, vf[u]);
          else load_dl(qkvg + (key - qpos0) * p.in_stride + p.v_off + j * D + lane * DL, vf[u]);
#else
          else load_dl(qkvg + (key - position) * p.in_stride + p.v_off + j * D + lane * DL, vf[u]);
#endif
        }
        for (uint u = 0; u < PV_UNROLL; u++) {
          const uint key = k0 + g * 32u + kk + u;
          if (key >= k1) break;
          for (uint r = 0; r < RBMAX; r++) {
            if (r < nr) {
              const float sc = simd_shuffle(s_keep[g][r], ushort(kk + u));
              const float pr = (sc == -INFINITY) ? 0.0f : exp(sc - m_c[r]);
              d_c[r] += pr;
              const float pb = round_bf16(pr);
              for (uint e = 0; e < DL; e++) o[r][e] = fma(pb, vf[u][e], o[r][e]);
            }
          }
        }
      }
    }
    for (uint r = 0; r < RBMAX; r++) {
      if (r < nr) {
        const uint base = (j * p.n_chunks_max + c) * p.rows_max + r0 + r;
        store_part(part_o + base * D + lane * DL, o[r]);
        if (lane == 0) { part_md[base * 2u] = m_c[r]; part_md[base * 2u + 1u] = d_c[r]; }
      }
    }
  }
}

kernel void gqa_merge(device const float* part_o [[buffer(0)]], device const float* part_md [[buffer(1)]], device const ushort* qkvg [[buffer(2)]],
                      device ushort* out [[buffer(3)]], constant GqaParams& p [[buffer(4)]],
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / p.heads, h = sg % p.heads;
#if DRAFT
#if STEP_STATE
  if (st->done) return;
  const uint T = p.t_active, ctx = st->drafter_ctx_len + st->n_inject + T;
#else
  const uint T = p.t_active, ctx = p.position + p.pad0 + T;
#endif
#elif STEP_STATE
  if (st->done) return;
#if LM_MODE == 1
  const uint T = st->n_inject, ctx = st->position;                                   // the ingest rows end at the new anchor's position
#elif LM_MODE == 2
  const uint T = st->n_chain, ctx = st->position + CHAIN_I + T;
#elif LM_MODE == 3
  const uint T = st->n_inject + st->n_chain, ctx = st->position + st->n_chain;
#else
  const uint T = st->t_this_step, ctx = st->position + T;
#endif
#else
  const uint T = p.t_active, ctx = p.position + T;
#endif
  if (t >= T) return;
  const uint rep = p.heads / p.kv_heads;
  const uint j = h / rep, row = t * rep + (h % rep);
  const uint ch = pick_chunk(ctx, p.kv_heads, rep, p.n_sg, p.n_chunks_max);
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
    out[t * PERM_K + perm_dest(h * D + lane * DL + e)] = bf16bits(y);   // the consumer tile's x' (its K = heads · D)
#else
    out[t * p.out_stride + h * D + lane * DL + e] = bf16bits(y);
#endif
  }
}
