// moe_combine: the weighted sum of a token's k expert outputs (design §5.11), plus the shared expert gated by
// σ(shared_gate) when the model has one (HAS_SHARED), plus the residual (HAS_RESIDUAL); FP32 accumulation, one BF16
// rounding. h is [T, k·H] (slot j's H columns at j·H), weights [T, k] FP32 (BF16-valued), shared [T, H] BF16, shared_gate
// [T, 1] BF16 (the gate's logit), residual [T, H] BF16, out [T, H] BF16. One SIMD-group per token, lanes over columns.
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef HAS_SHARED
#define HAS_SHARED 0
#endif
#ifndef HAS_RESIDUAL
#define HAS_RESIDUAL 0
#endif
struct CombineParams { uint hidden; uint top_k; uint t_active; uint pad; };

static inline float comb_bf16(uint u) { return as_type<float>(u << 16); }
static inline ushort comb_to_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return ushort(u >> 16); }

kernel void moe_combine(device const ushort* h [[buffer(0)]], device const float* weights [[buffer(1)]],
                        device const ushort* shared [[buffer(2)]], device const ushort* shared_gate [[buffer(3)]],
                        device const ushort* residual [[buffer(4)]], device ushort* out [[buffer(5)]],
                        constant CombineParams& p [[buffer(6)]],
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
  const uint H = p.hidden, k = p.top_k;
#if HAS_SHARED
  const float g = 1.0f / (1.0f + exp(-comb_bf16(shared_gate[t])));
  const float gb = comb_bf16(comb_to_bf16(g));                                       // the reference's σ(gate) in BF16
#endif
  for (uint c = lane; c < H; c += 32u) {
    float acc = 0.0f;
    for (uint j = 0; j < k; j++) acc = fma(weights[t * k + j], comb_bf16(h[(t * k + j) * H + c]), acc);
#if HAS_SHARED
    acc = fma(gb, comb_bf16(shared[t * H + c]), acc);
#endif
#if HAS_RESIDUAL
    acc += comb_bf16(residual[t * H + c]);
#endif
    out[t * H + c] = comb_to_bf16(acc);
  }
}
