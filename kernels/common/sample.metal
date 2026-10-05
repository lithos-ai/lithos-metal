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
                      uint seed_lo; uint seed_hi; uint step; uint flags; };   // flags: 1 top_k, 2 top_p, 4 min_p
#define HIST_KEYS 65536u
#define LANE_KEYS (HIST_KEYS / 32u)

static inline uint key16(ushort bits) { return (bits & 0x8000u) ? uint((~bits) & 0xFFFFu) : uint(bits | 0x8000u); }
static inline float key_val(uint key) { ushort bits = (key & 0x8000u) ? ushort(key & 0x7FFFu) : ushort(~key & 0xFFFFu); return bf16f(bits); }

static inline float gumbel(uint seed_lo, uint seed_hi, uint step, uint t, uint i) {
  ulong x = (ulong(seed_hi) << 32) | ulong(seed_lo);
  x += ulong(step) * 0x9E3779B97F4A7C15ul + ulong(t) * 0xC2B2AE3D27D4EB4Ful + ulong(i) * 0x165667B19E3779F9ul;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ul; x ^= x >> 27; x *= 0x94D049BB133111EBul; x ^= x >> 31;   // splitmix64
  const float u = (float((x >> 41) & 0x7FFFFFul) + 0.5f) * (1.0f / 8388608.0f);   // (0, 1): 23 bits, so u < 1 stays representable in float32 (24 bits would round to 1 → −log(−log 1) = ∞)
  return -log(-log(u));
}

kernel void sample_hist(device const ushort* logits [[buffer(0)]], device atomic_uint* hist [[buffer(1)]], constant SampleParams& p [[buffer(3)]],
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
    for (uint s = sg; s < p.n_spans; s += p.n_sg) {
      const uint base = s * 256u + lane * 8u;
      for (uint e = 0; e < 8u; e++) {
        const uint i = base + e;
        if (i < p.vocab) atomic_fetch_add_explicit(h + key16(row[i]), 1u, memory_order_relaxed);
      }
    }
  }
}

kernel void sample_select(device atomic_uint* hist [[buffer(1)]], device float* tau [[buffer(2)]], constant SampleParams& p [[buffer(3)]],
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
  const uint hi = HIST_KEYS - 1u - lane * LANE_KEYS;                 // this lane's stripe: keys (hi - LANE_KEYS, hi], descending
  const float inv_t = 1.0f / p.temperature;
  // pass a: the maximum logit (the highest non-empty key)
  uint kmax = 0u;
  for (uint j = 0; j < LANE_KEYS; j++) { const uint k = hi - j; if (atomic_load_explicit(h + k, memory_order_relaxed) != 0u) { kmax = k; break; } }
  const uint gmax = simd_max(kmax);
  const float vmax = key_val(gmax);
  // pass b: per-stripe count and softmax mass (of logits / temperature, relative to the max)
  uint cnt = 0u;
  float mass = 0.0f;
  for (uint j = 0; j < LANE_KEYS; j++) {
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
      for (uint j = 0; j < LANE_KEYS; j++) { const uint k = hi - j; run += atomic_load_explicit(h + k, memory_order_relaxed); if (run >= k_target) { tau_k = key_val(k); break; } }
    }
    tau_k = simd_max(tau_k);
  }
  if (p.flags & 2u) {
    const float target = p.top_p * z;
    if (mass_before < target && mass_before + mass >= target) {
      float run = mass_before;
      for (uint j = 0; j < LANE_KEYS; j++) {
        const uint k = hi - j;
        const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
        if (c != 0u) { run += float(c) * exp((key_val(k) - vmax) * inv_t); if (run >= target) { tau_p = key_val(k); break; } }
      }
    }
    tau_p = simd_max(tau_p);
  }
  if (p.flags & 4u) tau_m = vmax + p.temperature * log(p.min_p);
  if (lane == 0) tau[t] = max(max(tau_k, tau_p), tau_m);
  for (uint j = 0; j < LANE_KEYS; j++) atomic_store_explicit(h + (hi - j), 0u, memory_order_relaxed);   // clear for the next step
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
