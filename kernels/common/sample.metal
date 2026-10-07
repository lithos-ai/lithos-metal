// Stochastic sampling on the GPU (design D7; issue #23), appended to argmax.metal's helpers. Four dispatches:
//   sample_hist   — a histogram of the BF16 logits' monotone 16-bit keys per token (device atomics; 256 KB per token);
//   sample_select — one SIMD-group per token: scans the histogram from the top key down (lane ℓ owns a stripe of
//                   2048 keys, lane-prefix across stripes) for the logit thresholds of top-k (the k-th largest value:
//                   ties kept, like the HF warper), top-p (the bucket at which the descending cumulative softmax mass
//                   of logits / temperature first reaches p · Z) and min-p (max + temperature · log(min_p)); writes
//                   tau[t] = the most restrictive of the enabled ones and clears its stripe of the histogram;
//   sample_gumbel — argmax over the vocabulary of  logit / temperature + Gumbel noise  for logits ≥ tau[t] (Gumbel-max
//                   = a draw from the warped softmax), the noise from a counter-based hash keyed by (seed, step,
//                   token, index) so a run is reproducible bit for bit; partials as argmax_partial;
//   argmax_final  — as for greedy.
// Greedy (temperature 0) uses argmax_partial + argmax_final alone. Exact for BF16 logits: the thresholds are values
// of the 65536 possible logit codes, so no sort is needed.
struct SampleParams { uint vocab; uint t_active; uint n_sg; uint n_spans; uint top_k; float temperature; float top_p; float min_p;
                      uint seed_lo; uint seed_hi; uint step; uint flags; };   // flags: 1 top_k, 2 top_p, 4 min_p,
                                                                             // 8 top-p over the top-k set (the HF warpers' order)
#ifndef KEPT_STATS
#define KEPT_STATS 0                 // 1: sample_hist records each row's occupied key range and sample_select scans only it
#endif                               // and writes stats[t] = (max logit, softmax mass of the kept logits)
#define HIST_KEYS 65536u
#define LANE_KEYS (HIST_KEYS / 32u)

static inline uint key16(ushort bits) { return (bits & 0x8000u) ? uint((~bits) & 0xFFFFu) : uint(bits | 0x8000u); }
static inline float key_val(uint key) { ushort bits = (key & 0x8000u) ? ushort(key & 0x7FFFu) : ushort(~key & 0xFFFFu); return bf16f(bits); }

static inline float uniform01(uint seed_lo, uint seed_hi, uint step, uint t, uint i) {
  ulong x = (ulong(seed_hi) << 32) | ulong(seed_lo);
  x += ulong(step) * 0x9E3779B97F4A7C15ul + ulong(t) * 0xC2B2AE3D27D4EB4Ful + ulong(i) * 0x165667B19E3779F9ul;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ul; x ^= x >> 27; x *= 0x94D049BB133111EBul; x ^= x >> 31;   // splitmix64
  return (float((x >> 41) & 0x7FFFFFul) + 0.5f) * (1.0f / 8388608.0f);   // (0, 1): 23 bits, so u < 1 stays representable in float32 (24 bits would round to 1 → −log(−log 1) = ∞)
}
static inline float gumbel(uint seed_lo, uint seed_hi, uint step, uint t, uint i) { return -log(-log(uniform01(seed_lo, seed_hi, step, t, i))); }

kernel void sample_hist(device const ushort* logits [[buffer(0)]], device atomic_uint* hist [[buffer(1)]], constant SampleParams& p [[buffer(3)]],
#if KEPT_STATS
                        device atomic_uint* bounds [[buffer(2)]],   // per row: max key, 65535 - min key
#endif
#if STEP_STATE
                        device const StepState* st [[buffer(15)]],
#endif
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = st->t_this_step;
#else
  const uint T_act = p.t_active;
#endif
  for (uint t = 0; t < T_act; t++) {
    device const ushort* row = logits + (ulong)t * p.vocab;
    device atomic_uint* h = hist + (ulong)t * HIST_KEYS;
    uint kmax = 0u, kinv = 0u;
    for (uint s = sg; s < p.n_spans; s += p.n_sg) {
      const uint base = s * 256u + lane * 8u;
      for (uint e = 0; e < 8u; e++) {
        const uint i = base + e;
        if (i < p.vocab) {
          const uint key = key16(row[i]);
          atomic_fetch_add_explicit(h + key, 1u, memory_order_relaxed);
          kmax = max(kmax, key); kinv = max(kinv, 65535u - key);
        }
      }
    }
#if KEPT_STATS
    kmax = simd_max(kmax); kinv = simd_max(kinv);
    if (lane == 0) {
      atomic_fetch_max_explicit(bounds + 2u * t, kmax, memory_order_relaxed);
      atomic_fetch_max_explicit(bounds + 2u * t + 1u, kinv, memory_order_relaxed);
    }
#endif
  }
}

// the softmax mass (of logits / temperature, relative to vmax) of a stripe's keys whose value is >= th
static inline float stripe_mass(device atomic_uint* h, uint hi, uint nk, float th, float vmax, float inv_t) {
  float m = 0.0f;
  for (uint j = 0; j < nk; j++) {
    const uint k = hi - j;
    if (key_val(k) < th) break;                                      // keys descend within the stripe
    const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
    if (c != 0u) m += float(c) * exp((key_val(k) - vmax) * inv_t);
  }
  return m;
}

kernel void sample_select(device atomic_uint* hist [[buffer(1)]], device float* tau [[buffer(2)]], constant SampleParams& p [[buffer(3)]],
#if KEPT_STATS
                          device float* stats [[buffer(4)]], device atomic_uint* bounds [[buffer(5)]],
#endif
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
#if STEP_STATE
  if (st->done || t >= st->t_this_step) return;
#else
  if (t >= p.t_active) return;
#endif
  device atomic_uint* h = hist + (ulong)t * HIST_KEYS;
#if KEPT_STATS
  // the row's occupied keys [bot, top] only, in 32 descending stripes; top is the maximum logit's key
  const uint top = atomic_load_explicit(bounds + 2u * t, memory_order_relaxed);
  const uint span = top - (65535u - atomic_load_explicit(bounds + 2u * t + 1u, memory_order_relaxed));
  const uint sk = span / 32u + 1u, off = lane * sk;
  const uint hi = top - min(off, top), nk = (off > span) ? 0u : min(sk, span - off + 1u);
  const uint gmax = top;
#else
  const uint hi = HIST_KEYS - 1u - lane * LANE_KEYS, nk = LANE_KEYS;   // this lane's stripe: keys (hi - nk, hi], descending
  // pass a: the maximum logit (the highest non-empty key)
  uint kmax = 0u;
  for (uint j = 0; j < nk; j++) { const uint k = hi - j; if (atomic_load_explicit(h + k, memory_order_relaxed) != 0u) { kmax = k; break; } }
  const uint gmax = simd_max(kmax);
#endif
  const float inv_t = 1.0f / p.temperature;
  const float vmax = key_val(gmax);
  // pass b: per-stripe count and softmax mass (of logits / temperature, relative to the max)
  uint cnt = 0u;
  float mass = 0.0f;
  for (uint j = 0; j < nk; j++) {
    const uint k = hi - j;
    const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
    if (c != 0u) { cnt += c; mass += float(c) * exp((key_val(k) - vmax) * inv_t); }
  }
  const uint cnt_before = simd_prefix_exclusive_sum(cnt);
  const float mass_before = simd_prefix_exclusive_sum(mass);
  const float z = simd_sum(mass);
  // top-k: the bucket where the descending cumulative count reaches k
  float tau_k = -INFINITY, tau_p = -INFINITY, tau_m = -INFINITY;
  if (p.flags & 1u) {
    const uint k_target = p.top_k;
    if (cnt_before < k_target && cnt_before + cnt >= k_target) {
      uint run = cnt_before;
      for (uint j = 0; j < nk; j++) { const uint k = hi - j; run += atomic_load_explicit(h + k, memory_order_relaxed); if (run >= k_target) { tau_k = key_val(k); break; } }
    }
    tau_k = simd_max(tau_k);
  }
  if (p.flags & 2u) {
    const float target = p.top_p * (((p.flags & 9u) == 9u) ? simd_sum(stripe_mass(h, hi, nk, tau_k, vmax, inv_t)) : z);
    if (mass_before < target && mass_before + mass >= target) {
      float run = mass_before;
      for (uint j = 0; j < nk; j++) {
        const uint k = hi - j;
        const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
        if (c != 0u) { run += float(c) * exp((key_val(k) - vmax) * inv_t); if (run >= target) { tau_p = key_val(k); break; } }
      }
    }
    tau_p = simd_max(tau_p);
  }
  if (p.flags & 4u) tau_m = vmax + p.temperature * log(p.min_p);
  const float th = max(max(tau_k, tau_p), tau_m);
  if (lane == 0) tau[t] = th;
#if KEPT_STATS
  const float kept = simd_sum(stripe_mass(h, hi, nk, th, vmax, inv_t));
  if (lane == 0) {
    stats[2u * t] = vmax; stats[2u * t + 1u] = kept;
    atomic_store_explicit(bounds + 2u * t, 0u, memory_order_relaxed); atomic_store_explicit(bounds + 2u * t + 1u, 0u, memory_order_relaxed);
  }
#endif
  for (uint j = 0; j < nk; j++) atomic_store_explicit(h + (hi - j), 0u, memory_order_relaxed);   // clear for the next step
}

kernel void sample_gumbel(device const ushort* logits [[buffer(0)]], device const float* tau [[buffer(2)]], constant SampleParams& p [[buffer(3)]],
                          device float* part_val [[buffer(4)]], device uint* part_idx [[buffer(5)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = st->t_this_step, step = st->step;
  const uint seed_lo = st->rng_lo, seed_hi = st->rng_hi;
#else
  const uint T_act = p.t_active, step = p.step, seed_lo = p.seed_lo, seed_hi = p.seed_hi;
#endif
  const float inv_t = 1.0f / p.temperature;
  for (uint t = 0; t < T_act; t++) {
    const float th = tau[t];
    float best = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    device const ushort* row = logits + (ulong)t * p.vocab;
    for (uint s = sg; s < p.n_spans; s += p.n_sg) {
      const uint base = s * 256u + lane * 8u;
      for (uint e = 0; e < 8u; e++) {
        const uint i = base + e;
        if (i < p.vocab) {
          const float v = bf16f(row[i]);
          if (v >= th) better(best, bi, v * inv_t + gumbel(seed_lo, seed_hi, step, t, i), i);
        }
      }
    }
    simd_argmax(best, bi);
    if (lane == 0) { part_val[t * p.n_sg + sg] = best; part_idx[t * p.n_sg + sg] = bi; }
  }
}


// ---- speculative sampling with sampled drafts ------------------------------------------------------------------
// The drafter draws d_k ~ q_k = softmax(logits_k / temperature) (draft_q_*: Gumbel-max, noise stream 2000 + k) and keeps
// its logits row and log Σ exp(logits_k / temperature) in state. The verify step accepts draft r with probability
// min(1, p(d_r) / q_r(d_r)), p = the target's kept softmax of that row (sample_select's tau and stats; uniform stream
// 3000 + r), and replaces the first rejected one by a draw from norm(max(p − q_r, 0)) (Gumbel stream 4000): the
// committed tokens are distributed as the target's own samples. With every draft accepted the row after them keeps
// the target's sample. spec_q_accept rewrites token[] to [d_0 … d_{j−1}, correction], so accept_scan needs no change.
#ifndef DRAFT_K
#define DRAFT_K 0u                   // the draft position a draft_q_* kernel is compiled for
#endif

kernel void draft_q_partial(device const ushort* logits [[buffer(0)]], device float* part_val [[buffer(1)]], device uint* part_idx [[buffer(2)]],
                            constant SampleParams& p [[buffer(3)]], device float* part_m [[buffer(4)]], device float* part_s [[buffer(5)]],
                            device ushort* q_logits [[buffer(6)]],
#if STEP_STATE
                            device const StepState* st [[buffer(15)]],
#endif
                            uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint step = st->step, seed_lo = st->rng_lo, seed_hi = st->rng_hi;
#else
  const uint step = p.step, seed_lo = p.seed_lo, seed_hi = p.seed_hi;
#endif
  const float inv_t = 1.0f / p.temperature;
  device ushort* qrow = q_logits + (ulong)DRAFT_K * p.vocab;
  float best = -INFINITY, m = -INFINITY, sum = 0.0f;                 // Gumbel-max; running max and Σ exp(x − max)
  uint bi = 0xFFFFFFFFu;
  for (uint s = sg; s < p.n_spans; s += p.n_sg) {
    const uint base = s * 256u + lane * 8u;
    for (uint e = 0; e < 8u; e++) {
      const uint i = base + e;
      if (i < p.vocab) {
        const ushort bits = logits[i];
        qrow[i] = bits;
        const float x = bf16f(bits) * inv_t;
        better(best, bi, x + gumbel(seed_lo, seed_hi, step, 2000u + DRAFT_K, i), i);
        if (x > m) { sum = sum * exp(m - x) + 1.0f; m = x; } else { sum += exp(x - m); }
      }
    }
  }
  simd_argmax(best, bi);
  const float mm = simd_max(m);
  const float ss = simd_sum((m == -INFINITY) ? 0.0f : sum * exp(m - mm));
  if (lane == 0) { part_val[sg] = best; part_idx[sg] = bi; part_m[sg] = mm; part_s[sg] = ss; }
}

kernel void draft_q_final(device const float* part_val [[buffer(0)]], device const uint* part_idx [[buffer(1)]], device int* token [[buffer(2)]],
                          constant SampleParams& p [[buffer(3)]], device const float* part_m [[buffer(4)]], device const float* part_s [[buffer(5)]],
                          device float* q_lse [[buffer(6)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if (st->done) return;
#endif
  float best = -INFINITY, m = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint i = lane; i < p.n_sg; i += 32u) { better(best, bi, part_val[i], part_idx[i]); m = max(m, part_m[i]); }
  simd_argmax(best, bi);
  const float mm = simd_max(m);
  float s = 0.0f;
  for (uint i = lane; i < p.n_sg; i += 32u) if (part_m[i] > -INFINITY) s += part_s[i] * exp(part_m[i] - mm);
  s = simd_sum(s);
  if (lane == 0) { token[0] = int(bi); q_lse[DRAFT_K] = mm + log(s); }
}

#if STEP_STATE
// p(v) of verify row `row`: the kept softmax
static inline float spec_p(device const ushort* lrow, device const float* tau, device const float* stats, uint row, uint v, float inv_t) {
  const float z = bf16f(lrow[v]);
  return (z >= tau[row]) ? exp((z - stats[2u * row]) * inv_t) / stats[2u * row + 1u] : 0.0f;
}

// One SIMD-group: lane r tests draft r; lane 0 then rewrites token[] and sets flag[0] = the rejected row (or ~0).
kernel void spec_q_accept(device const ushort* logits [[buffer(0)]], device const float* tau [[buffer(1)]], device const float* stats [[buffer(2)]],
                          constant SampleParams& p [[buffer(3)]], device const ushort* q_logits [[buffer(4)]], device const float* q_lse [[buffer(5)]],
                          device int* token [[buffer(6)]], device uint* flag [[buffer(7)]],
                          device const StepState* st [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
  if (st->done) return;
  if (lane == 0) flag[0] = 0xFFFFFFFFu;
  if (st->prefill_left > 0u) return;
  const uint L = st->verify_len, t = st->t_this_step;
  if (L == 0u || t < L + 1u) return;
  const uint base = t - 1u - L;                                      // the row that verifies draft 0
  const float inv_t = 1.0f / p.temperature;
  bool reject = false;
  if (lane < L) {
    const uint d = uint(st->pending_tokens[base + lane + 1u]);
    const float pd = spec_p(logits + (ulong)(base + lane) * p.vocab, tau, stats, base + lane, d, inv_t);
    const float qd = exp(bf16f(q_logits[(ulong)lane * p.vocab + d]) * inv_t - q_lse[lane]);
    reject = uniform01(st->rng_lo, st->rng_hi, st->step, 3000u + lane, 0u) * qd >= pd;
  }
  const uint rejects = uint(ulong(simd_ballot(reject)));
  if (lane == 0) {
    const uint j = (rejects != 0u) ? ctz(rejects) : L;               // the first rejected draft (L: none)
    for (uint r = 0; r < j; r++) token[base + r] = st->pending_tokens[base + r + 1u];
    if (j < L) flag[0] = base + j;
  }
}

// The correction at the rejected row: Gumbel-max over log max(p − q, 0); partials as argmax_partial.
kernel void spec_residual_partial(device const ushort* logits [[buffer(0)]], device const float* tau [[buffer(1)]], device const float* stats [[buffer(2)]],
                                  constant SampleParams& p [[buffer(3)]], device const ushort* q_logits [[buffer(4)]], device const float* q_lse [[buffer(5)]],
                                  device const uint* flag [[buffer(7)]], device float* part_val [[buffer(8)]], device uint* part_idx [[buffer(9)]],
                                  device const StepState* st [[buffer(15)]],
                                  uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg || st->done) return;
  const uint row = flag[0];
  if (row == 0xFFFFFFFFu) return;
  const uint r = row - (st->t_this_step - 1u - st->verify_len);     // the draft position of that row
  const float inv_t = 1.0f / p.temperature;
  device const ushort* lrow = logits + (ulong)row * p.vocab;
  device const ushort* qrow = q_logits + (ulong)r * p.vocab;
  const float lse = q_lse[r];
  float best = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint s = sg; s < p.n_spans; s += p.n_sg) {
    const uint base = s * 256u + lane * 8u;
    for (uint e = 0; e < 8u; e++) {
      const uint i = base + e;
      if (i < p.vocab) {
        const float res = spec_p(lrow, tau, stats, row, i, inv_t) - exp(bf16f(qrow[i]) * inv_t - lse);
        if (res > 0.0f) better(best, bi, log(res) + gumbel(st->rng_lo, st->rng_hi, st->step, 4000u, i), i);
      }
    }
  }
  simd_argmax(best, bi);
  if (lane == 0) { part_val[sg] = best; part_idx[sg] = bi; }
}

kernel void spec_residual_final(device const float* part_val [[buffer(0)]], device const uint* part_idx [[buffer(1)]], device int* token [[buffer(2)]],
                                constant SampleParams& p [[buffer(3)]], device const uint* flag [[buffer(7)]],
                                device const StepState* st [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
  if (st->done) return;
  const uint row = flag[0];
  if (row == 0xFFFFFFFFu) return;
  float best = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint i = lane; i < p.n_sg; i += 32u) better(best, bi, part_val[i], part_idx[i]);
  simd_argmax(best, bi);
  if (lane == 0 && bi != 0xFFFFFFFFu) token[row] = int(bi);
}
#endif
