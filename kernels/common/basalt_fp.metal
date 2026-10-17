// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu, include/basalt/kernels/dp4a.hpp and
// include/basalt/kernels/q8_1_finite.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// which adapt ggml-org/llama.cpp ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh} @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// The arithmetic helpers every basalt port kernel shares. Each kernel source starts with this file (and the pragma
// below), so a multiply feeding an add never fuses unless fma() is written: the fusion, flush-to-zero and division
// sequence nvcc compiles for basalt's sm_120 kernels are written out explicitly, as recorded per site in
// docs/research/basalt-nvcc/matvec.md.
//
// - ftz(x): a subnormal becomes the zero of its sign; every other value keeps its bits. Integer operations only, so
//   it means the same on any GPU, whatever the hardware does with subnormals.
// - add_ftz, mul_ftz, fma_ftz, max_ftz, abs_ftz: CUDA's FADD.FTZ, FMUL.FTZ, FFMA.FTZ, FMNMX.FTZ and the .FTZ absolute
//   value: the operation on flushed inputs, correctly rounded, then the result flushed.
// - rcp_ftz(x): stands where SASS has MUFU.RCP (an approximate reciprocal NVIDIA does not specify bit for bit): the
//   correctly rounded 1 / x of the flushed input, flushed.
// - warp_sum_ftz / warp_max_ftz: basalt's warp_sum and warp_max (native_mmvq.cu:130-144), XOR butterflies with offsets
//   16, 8, 4, 2, 1 on the value's bits (simd_sum documents no order and is never used).
// - load_int_b2, dp4a, byte_perm, vsubss4: CUDA's two 16-bit loads, __dp4a, __byte_perm (3-bit selectors) and
//   __vsubss4 as exact integer arithmetic.
#pragma STDC FP_CONTRACT OFF

static inline float ftz(float x) {
    const uint b = as_type<uint>(x);
    return (b & 0x7f800000u) == 0u ? as_type<float>(b & 0x80000000u) : x;
}
static inline float add_ftz(float a, float b) { return ftz(ftz(a) + ftz(b)); }
static inline float mul_ftz(float a, float b) { return ftz(ftz(a) * ftz(b)); }
static inline float fma_ftz(float a, float b, float c) { return ftz(fma(ftz(a), ftz(b), ftz(c))); }
static inline float max_ftz(float a, float b) { return fmax(ftz(a), ftz(b)); }
static inline float abs_ftz(float x) { return ftz(fabs(x)); }
static inline float rcp_ftz(float x) { return ftz(1.0f / ftz(x)); }

static inline float shfl_xor(float x, ushort o) { return as_type<float>(simd_shuffle_xor(as_type<uint>(x), o)); }

static inline float warp_sum_ftz(float x) {
    for (ushort o = 16; o > 0; o >>= 1) x = add_ftz(x, shfl_xor(x, o));
    return x;
}
static inline float warp_max_ftz(float x) {
    for (ushort o = 16; o > 0; o >>= 1) x = max_ftz(x, shfl_xor(x, o));
    return x;
}

// fp16 field to fp32 (exact): CUDA's __half2float / __low2float, HADD2.F32 in SASS.
static inline float h2f(ushort h) { return float(as_type<half>(h)); }

// native_mmvq.cu:353-358: a 32-bit word of a 2-byte aligned block, as two 16-bit loads combined little-endian.
static inline int load_int_b2(device const ushort* p, int i32) {
    return int(uint(p[2 * i32]) | (uint(p[2 * i32 + 1]) << 16));
}

// byte `byte` of a 2-byte aligned block, through a 16-bit load (little-endian)
static inline uchar w_u8(device const ushort* blk, uint byte) { return uchar(blk[byte >> 1] >> (8u * (byte & 1u))); }

// __dp4a(a, b, c): c plus the four products of sign-extended bytes (no sum here comes near 2^31).
static inline int dp4a(int a, int b, int c) {
    const char4 x = as_type<char4>(a), y = as_type<char4>(b);
    return c + int(x.x) * int(y.x) + int(x.y) * int(y.y) + int(x.z) * int(y.z) + int(x.w) * int(y.w);
}

// __byte_perm(x, y, s): byte n of the result is byte s<4n+2:4n> of {y:x}; the selectors' fourth bits and the upper
// 16 bits of s are not used (CUDA Math API).
static inline int byte_perm(int x, int y, int s) {
    uint r = 0u;
    for (uint n = 0u; n < 4u; ++n) {
        const uint sel = (uint(s) >> (4u * n)) & 7u;
        const uint src = sel < 4u ? uint(x) : uint(y);
        r |= ((src >> (8u * (sel & 3u))) & 0xffu) << (8u * n);
    }
    return int(r);
}

// __vsubss4(a, b): per signed byte a - b, saturated to [-128, 127].
static inline int vsubss4(int a, int b) {
    const int4 d = clamp(int4(as_type<char4>(a)) - int4(as_type<char4>(b)), int4(-128), int4(127));
    return as_type<int>(char4(d));
}
