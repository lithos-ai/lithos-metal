#include <metal_stdlib>
using namespace metal;
// Decode GEMV / small-T GEMM: y[T][N] = x[T][K] . W[N][K]^T with W in FP8-E4M3 (per-tensor scale) or NVFP4 (E2M1 nibbles, E4M3 block
// scale per 16, tensor scale). The first real kernel of plan M1, written as a probe: is a properly engineered decode kernel bus-bound
// at T = 1 in the crew geometry, and at what T does verification turn ALU-bound on this chip?
//   block = R rows; the 32 lanes own 32 column stripes of K/32 = 160 columns (K = 5120); SIMD-group sg runs blocks sg, sg+n_sg, ...
//   weights: 16-byte loads from a block-lane-major pack in one of two lane orders (p12 showed the order decides bandwidth on the M5 Pro):
//     LAYOUT 0  lane-contiguous (design D8 as written): uint4 index (lane*R + r)*C4 + j   - lane's stripe of all R rows is one run
//     LAYOUT 1  lane-interleaved 16 B:                  uint4 index (r*C4 + j)*32 + lane  - one load instruction reads 512 contiguous bytes
//   FP8 pack: 10 uint4 per (lane,row). NVFP4 pack: 5 uint4 of nibbles (32 weights each, low nibble first) + 1 uint4 with the 10 E4M3
//   block scales (6 pad bytes) = 96 B per (lane,row), 90 useful. Activations: half, re-read (cached) per RG-row group; FP32 accumulate.
#ifndef FMT
#define FMT 0
#endif
#ifndef R
#define R 16
#endif
#ifndef T
#define T 1
#endif
#ifndef LAYOUT
#define LAYOUT 1
#endif
#ifndef RG
#define RG 4
#endif
#define K 5120u
#define KL 160u
#if FMT == 0
#define C4 10u
#else
#define C4 6u
#endif
#define BLK4 (R * 32u * C4)
struct P { uint n_sg; uint n_blocks; uint n_rows; uint pad; float scale; float pad1, pad2, pad3; };
inline float fp8_e4m3(uint q) { uint e = (q >> 3) & 15u, m = q & 7u; float v = as_type<float>(((q & 0x7Fu) << 20) + (120u << 23)); v = (e == 0u) ? float(m) * (1.0f / 512.0f) : v; return (q & 0x80u) ? -v : v; }
inline float fp4_e2m1(uint q) { uint e = (q >> 1) & 3u, m = q & 1u; float v = (e == 0u) ? float(m) * 0.5f : as_type<float>(((e + 126u) << 23) | (m << 22)); return (q & 8u) ? -v : v; }
inline uint widx(uint lane, uint r, uint j) {
#if LAYOUT == 0
  return (lane * R + r) * C4 + j;
#else
  return (r * C4 + j) * 32u + lane;
#endif
}
kernel void gemv(device const uint4* w [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]], constant P& p [[buffer(3)]],
                 uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw;
  for (uint b = sg; b < p.n_blocks; b += p.n_sg) {
    device const uint4* wb = w + (size_t)b * BLK4;
    for (uint r0 = 0; r0 < R; r0 += RG) {
      float acc[RG][T];
      for (uint i = 0; i < RG; i++) for (uint t = 0; t < T; t++) acc[i][t] = 0.0f;
#if FMT == 0
      for (uint j = 0; j < C4; j++) {                                       // 16 columns per step
        uint col = lane * KL + j * 16u; uint4 xq[T][2];
        for (uint t = 0; t < T; t++) { device const uint4* xp = (device const uint4*)(x + t * K + col); xq[t][0] = xp[0]; xq[t][1] = xp[1]; }
        for (uint i = 0; i < RG; i++) {
          uint4 q = wb[widx(lane, r0 + i, j)]; uint qw[4] = {q.x, q.y, q.z, q.w}; float wv[16];
          for (uint e = 0; e < 16; e++) wv[e] = fp8_e4m3((qw[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu);
          for (uint t = 0; t < T; t++) {
            uint xw[8] = {xq[t][0].x, xq[t][0].y, xq[t][0].z, xq[t][0].w, xq[t][1].x, xq[t][1].y, xq[t][1].z, xq[t][1].w}; float s = 0.0f;
            for (uint e = 0; e < 8; e++) { half2 h = as_type<half2>(xw[e]); s = fma(wv[2*e], float(h.x), s); s = fma(wv[2*e+1], float(h.y), s); }
            acc[i][t] += s; } } }
#else
      uint4 scq[RG]; for (uint i = 0; i < RG; i++) scq[i] = wb[widx(lane, r0 + i, 5u)];
      for (uint j = 0; j < 5u; j++) {                                       // 32 columns per step = 2 scale blocks
        uint col = lane * KL + j * 32u; uint4 xq[T][4];
        for (uint t = 0; t < T; t++) { device const uint4* xp = (device const uint4*)(x + t * K + col); xq[t][0] = xp[0]; xq[t][1] = xp[1]; xq[t][2] = xp[2]; xq[t][3] = xp[3]; }
        for (uint i = 0; i < RG; i++) {
          uint4 q = wb[widx(lane, r0 + i, j)]; uint sb[4] = {scq[i].x, scq[i].y, scq[i].z, scq[i].w};
          float s0 = fp8_e4m3((sb[(2u*j) >> 2] >> (((2u*j) & 3u) * 8u)) & 0xFFu), s1 = fp8_e4m3((sb[(2u*j+1u) >> 2] >> (((2u*j+1u) & 3u) * 8u)) & 0xFFu);
          uint qw[4] = {q.x, q.y, q.z, q.w}; float wv[32];
          for (uint e = 0; e < 32; e++) wv[e] = fp4_e2m1((qw[e >> 3] >> ((e & 7u) * 4u)) & 0xFu);
          for (uint t = 0; t < T; t++) {
            uint xw[16] = {xq[t][0].x, xq[t][0].y, xq[t][0].z, xq[t][0].w, xq[t][1].x, xq[t][1].y, xq[t][1].z, xq[t][1].w, xq[t][2].x, xq[t][2].y, xq[t][2].z, xq[t][2].w, xq[t][3].x, xq[t][3].y, xq[t][3].z, xq[t][3].w};
            float sa = 0.0f, sb2 = 0.0f;
            for (uint e = 0; e < 8; e++) { half2 h = as_type<half2>(xw[e]); sa = fma(wv[2*e], float(h.x), sa); sa = fma(wv[2*e+1], float(h.y), sa); }
            for (uint e = 8; e < 16; e++) { half2 h = as_type<half2>(xw[e]); sb2 = fma(wv[2*e], float(h.x), sb2); sb2 = fma(wv[2*e+1], float(h.y), sb2); }
            acc[i][t] = fma(sa, s0, fma(sb2, s1, acc[i][t])); } } }
#endif
      for (uint i = 0; i < RG; i++) for (uint t = 0; t < T; t++) { float v = simd_sum(acc[i][t]); if (lane == 0) y[t * p.n_rows + b * R + r0 + i] = v * p.scale; }
    }
  }
}
