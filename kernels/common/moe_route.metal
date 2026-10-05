// moe_route: the router's top-k per token (design §5.11). One SIMD-group per token: softmax over the E expert logits
// (BF16 in, FP32 math), then k rounds of argmax over the probabilities — ties to the lowest expert index — with the
// chosen probability optionally renormalized over the k (RENORM); the weights are rounded to BF16 (the reference casts
// its routing weights to the hidden dtype) and stored as FP32. ids[t·k + j] = expert, weights[t·k + j] = weight.
// E <= 32·MAX_PER_LANE (256). With STEP_STATE the token count is t_this_step (buffer 15), else params.t_active.
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef MAX_PER_LANE
#define MAX_PER_LANE 8u                        // experts per lane: E <= 256
#endif
#ifndef RENORM
#define RENORM 0
#endif
struct RouteParams { uint n_experts; uint top_k; uint t_active; uint pad; };

static inline float route_bf16(uint u) { return as_type<float>(u << 16); }
static inline float route_round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }

kernel void moe_route(device const ushort* logits [[buffer(0)]], device int* ids [[buffer(1)]], device float* weights [[buffer(2)]],
                      constant RouteParams& p [[buffer(3)]],
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
  const uint E = p.n_experts;
  float v[MAX_PER_LANE];
  float m = -INFINITY;
  for (uint i = 0; i < MAX_PER_LANE; i++) {
    const uint e = lane + 32u * i;
    v[i] = (e < E) ? route_bf16(logits[t * E + e]) : -INFINITY;
    m = max(m, v[i]);
  }
  m = simd_max(m);
  float s = 0.0f;
  for (uint i = 0; i < MAX_PER_LANE; i++) { const uint e = lane + 32u * i; if (e < E) { v[i] = exp(v[i] - m); s += v[i]; } else v[i] = -1.0f; }
  s = simd_sum(s);
  for (uint i = 0; i < MAX_PER_LANE; i++) if (v[i] >= 0.0f) v[i] /= s;            // the probabilities; -1 marks an absent or chosen expert
  float chosen_w[16];
  uint chosen_e[16];
  float wsum = 0.0f;
  for (uint r = 0; r < p.top_k && r < 16u; r++) {
    float best = -1.0f; uint bi = 0xFFFFFFFFu;
    for (uint i = 0; i < MAX_PER_LANE; i++) { if (v[i] > best) { best = v[i]; bi = lane + 32u * i; } }
    const float gmax = simd_max(best);
    const uint cand = (best == gmax) ? bi : 0xFFFFFFFFu;                             // the lowest index among the lanes at the max
    const uint e = simd_min(cand);
    if (e == 0xFFFFFFFFu) break;                                                     // fewer experts than k
    const uint owner = e % 32u, slot = e / 32u;
    if (lane == owner) v[slot] = -1.0f;                                              // taken
    chosen_e[r] = e; chosen_w[r] = gmax; wsum += gmax;
  }
  for (uint r = 0; r < p.top_k && r < 16u; r++) {
    float w = chosen_w[r];
#if RENORM
    w = w / wsum;
#endif
    if (lane == 0) { ids[t * p.top_k + r] = int(chosen_e[r]); weights[t * p.top_k + r] = route_round_bf16(w); }
  }
}
