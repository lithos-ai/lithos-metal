// Optional BF16 rounding between RMS normalization and its learned scale.
// The default preserves the fused single-rounding convention.
#ifndef NORM_ROUND
#define NORM_ROUND 0
#endif
static inline float activation_bf16(float x) {
  uint u = as_type<uint>(x);
  return as_type<float>((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}
static inline float norm_scale(float x, float r, float w) {
  float n = x * r;
#if NORM_ROUND
  n = activation_bf16(n);
#endif
  return n * w;
}

#ifndef SILU_ROUND
#define SILU_ROUND 0
#endif
static inline float silu_mul(float gate, float up) {
#if SILU_ROUND
  gate = activation_bf16(gate);
  up = activation_bf16(up);
#endif
  float activated = gate / (1.0f + exp(-gate));
#if SILU_ROUND
  activated = activation_bf16(activated);
#endif
  return activated * up;
}
