// rmsnorm_stat: stat[t] = Σ_k h[t][k]² over a BF16 row (one SIMD-group per token; K % 8 == 0). The consumer
// (gemv_T with NORM=1, stat_parts = 1) turns it into r[t] = rsqrt(stat/K + eps). The standalone form of the
// statistic; the fuse pass replaces it by the producing GEMV's STAT_OUT partials wherever it can (design §5.1).
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef T_SRC
#define T_SRC 0                      // with STEP_STATE: 0 = t_this_step, 1 = n_inject, 2 = the T_STATIC_ROWS macro
#endif
#ifndef T_STATIC_ROWS
#define T_STATIC_ROWS 1u
#endif
struct StatParams { uint k; uint t_active; uint pad0; uint pad1; };

static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }

kernel void rmsnorm_stat(device const ushort* h [[buffer(0)]], device float* stat [[buffer(1)]], constant StatParams& p [[buffer(2)]],
#if STEP_STATE
                         device const StepState* st [[buffer(15)]],
#endif
                         uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
#if STEP_STATE
  if (st->done || t >= ((T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step))))) return;
#else
  if (t >= p.t_active) return;
#endif
  device const uint4* row = (device const uint4*)(h + (ulong)t * p.k);
  float s = 0.0f;
  for (uint j = lane; j < p.k / 8u; j += 32u) {
    uint4 q = row[j];
    float a = bf16lo(q.x), b = bf16hi(q.x), c = bf16lo(q.y), d = bf16hi(q.y);
    float e = bf16lo(q.z), f = bf16hi(q.z), g = bf16lo(q.w), k = bf16hi(q.w);
    s = fma(a, a, s); s = fma(b, b, s); s = fma(c, c, s); s = fma(d, d, s);
    s = fma(e, e, s); s = fma(f, f, s); s = fma(g, g, s); s = fma(k, k, s);
  }
  s = simd_sum(s);
  if (lane == 0) stat[t] = s;
}
