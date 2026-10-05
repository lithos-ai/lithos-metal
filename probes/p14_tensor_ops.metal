#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// M5 neural accelerators for speculative-decode verification (design 5.6 / 5.12, plan M9): y[TM][N] = x[TM][K] . W[N][K]^T through
// mpp::tensor_ops::matmul2d, executed cooperatively by S SIMD-groups per threadgroup. Each threadgroup owns TN rows of W and loops
// over K in TK-wide tiles, accumulating into a cooperative (register) tensor that is stored once at the end.
//   gemm_fp8:  W is FP8 E4M3 in device memory; every tile is dequantized on the shader ALUs into a threadgroup-memory half tile
//              (the accelerator accepts half/bfloat/int8 operands only) and the accelerator multiplies from there.
//   gemm_half: W already half in device memory, multiplied straight from a device tensor (the accelerator's upper bound).
// Tensor extents are (innermost first): x is [TM x K] with K contiguous -> extents (K, TM); W is [N x K] with K contiguous -> the NT case,
// extents (K, N); y is [TM x N] with N contiguous -> extents (N, TM).
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 64
#endif
#ifndef TK
#define TK 64
#endif
#ifndef S
#define S 4
#endif
#define K 5120
struct P { uint n_rows; uint p0; uint p1; uint p2; float scale; float f1, f2, f3; };
inline float fp8_e4m3(uint q) { uint e = (q >> 3) & 15u, m = q & 7u; float v = as_type<float>(((q & 0x7Fu) << 20) + (120u << 23)); v = (e == 0u) ? float(m) * (1.0f / 512.0f) : v; return (q & 0x80u) ? -v : v; }
constexpr constant auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
using tA_t = tensor<device half, dextents<int, 2>, tensor_inline>;
using tW_t = tensor<device half, dextents<int, 2>, tensor_inline>;
using tB_t = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
using tC_t = tensor<device float, dextents<int, 2>, tensor_inline>;

kernel void gemm_fp8(device const uint4* w [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant P& p [[buffer(3)]],
                     threadgroup half* tgw [[threadgroup(0)]], uint tgid [[threadgroup_position_in_grid]], uint lt [[thread_position_in_threadgroup]], uint tpt [[threads_per_threadgroup]]) {
  matmul2d<desc, execution_simdgroups<S>> op;
  tA_t tA(x, dextents<int, 2>(K, TM));
  tB_t tB(tgw, dextents<int, 2>(TK, TN));
  auto cT = op.get_destination_cooperative_tensor<tA_t, tB_t, float>();
  for (uint16_t i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.0f;
  const uint n0 = tgid * TN;
  for (uint k0 = 0; k0 < K; k0 += TK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);                       // the previous run() is done with tgw
    for (uint i = lt; i < TN * TK / 16; i += tpt) { uint row = i / (TK / 16), c16 = i % (TK / 16);
      uint4 q = w[((size_t)(n0 + row) * K + k0 + c16 * 16) >> 4]; uint qw[4] = {q.x, q.y, q.z, q.w};
      for (uint e = 0; e < 16; e++) tgw[row * TK + c16 * 16 + e] = half(fp8_e4m3((qw[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto sA = tA.slice<TK, TM>(int(k0), 0);
    op.run(sA, tB, cT);
  }
  tC_t tC(y, dextents<int, 2>(int(p.n_rows), TM));
  auto sC = tC.slice<TN, TM>(int(n0), 0);
  for (uint16_t i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] *= p.scale;
  cT.store(sC);
}
inline float fp4_e2m1(uint q) { uint e = (q >> 1) & 3u, m = q & 1u; float v = (e == 0u) ? float(m) * 0.5f : as_type<float>(((e + 126u) << 23) | (m << 22)); return (q & 8u) ? -v : v; }
// NVFP4: nibbles [N x K/2] (low nibble first) and E4M3 block scales [N x K/16], both row-major; dequantized (x block scale) into the tile.
kernel void gemm_nvfp4(device const uint4* w [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant P& p [[buffer(3)]], device const uchar* sc [[buffer(4)]],
                       threadgroup half* tgw [[threadgroup(0)]], uint tgid [[threadgroup_position_in_grid]], uint lt [[thread_position_in_threadgroup]], uint tpt [[threads_per_threadgroup]]) {
  matmul2d<desc, execution_simdgroups<S>> op;
  tA_t tA(x, dextents<int, 2>(K, TM));
  tB_t tB(tgw, dextents<int, 2>(TK, TN));
  auto cT = op.get_destination_cooperative_tensor<tA_t, tB_t, float>();
  for (uint16_t i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.0f;
  const uint n0 = tgid * TN;
  for (uint k0 = 0; k0 < K; k0 += TK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lt; i < TN * TK / 32; i += tpt) { uint row = i / (TK / 32), c32 = i % (TK / 32);
      uint4 q = w[((size_t)(n0 + row) * (K / 2) + (k0 + c32 * 32) / 2) >> 4]; uint qw[4] = {q.x, q.y, q.z, q.w};
      size_t si = (size_t)(n0 + row) * (K / 16) + (k0 + c32 * 32) / 16; float s0 = fp8_e4m3(sc[si]), s1 = fp8_e4m3(sc[si + 1]);
      for (uint e = 0; e < 32; e++) tgw[row * TK + c32 * 32 + e] = half(fp4_e2m1((qw[e >> 3] >> ((e & 7u) * 4u)) & 0xFu) * (e < 16u ? s0 : s1)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto sA = tA.slice<TK, TM>(int(k0), 0);
    op.run(sA, tB, cT);
  }
  tC_t tC(y, dextents<int, 2>(int(p.n_rows), TM));
  auto sC = tC.slice<TN, TM>(int(n0), 0);
  for (uint16_t i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] *= p.scale;
  cT.store(sC);
}
kernel void gemm_half(device half* w [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant P& p [[buffer(3)]],
                      uint tgid [[threadgroup_position_in_grid]]) {
  matmul2d<desc, execution_simdgroups<S>> op;
  tA_t tA(x, dextents<int, 2>(K, TM));
  tW_t tW(w, dextents<int, 2>(K, int(p.n_rows)));
  auto cT = op.get_destination_cooperative_tensor<tA_t, tW_t, float>();
  for (uint16_t i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.0f;
  const uint n0 = tgid * TN;
  for (uint k0 = 0; k0 < K; k0 += TK) { auto sA = tA.slice<TK, TM>(int(k0), 0); auto sB = tW.slice<TK, TN>(int(k0), int(n0)); op.run(sA, sB, cT); }
  tC_t tC(y, dextents<int, 2>(int(p.n_rows), TM));
  auto sC = tC.slice<TN, TM>(int(n0), 0);
  cT.store(sC);
}
