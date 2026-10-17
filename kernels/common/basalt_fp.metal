// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu, src/kernels/cuda/iq_kernels.cu,
// include/basalt/kernels/dp4a.hpp and include/basalt/kernels/q8_1_finite.hpp
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
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
//   value: the operation on flushed inputs, correctly rounded (to nearest even on the float32 grid, subnormals
//   included: PTX's "rounded to single precision"), then a subnormal result flushed. This GPU's multiply and fma flush
//   on the exact value instead, before rounding: an exact result in [2^-126 - 2^-150, 2^-126), which rounds up to the
//   smallest normal 2^-126, comes back as the zero of its sign (tests/kernels/test_basalt_platform.py). So fma_ftz
//   checks a zero from nonzero factors against the exact value (fma_ftz_min_normal), and mul_ftz is fma_ftz(a, b, -0)
//   (a * b + -0 is a * b, zeros included). A sum below 2^-126 is exact, so add_ftz needs no check, and no reciprocal
//   of a float lies in that interval. max_ftz needs its explicit flush: this GPU's fmax compares a subnormal as zero
//   but returns it unflushed. On +0 against -0 it returns the second operand (CUDA's choice there is not checked); the
//   port takes max_ftz of absolute values only.
// - rcp_ftz(x): stands where SASS has MUFU.RCP (an approximate reciprocal NVIDIA does not specify bit for bit): the
//   correctly rounded 1 / x of the flushed input, flushed.
// - warp_sum_ftz / warp_max_ftz: basalt's warp_sum and warp_max (native_mmvq.cu:130-144), XOR butterflies with offsets
//   16, 8, 4, 2, 1 on the value's bits (simd_sum documents no order and is never used); warp_sum: the IEEE-built
//   files' butterfly (iq_kernels.cu:545-549), plain adds.
// - expf_neg: __expf(-g) as an IEEE-built file compiles it (docs/research/basalt-nvcc/experts.md).
// - load_int_b2, dp4a, byte_perm, vsubss4, vsub4: CUDA's two 16-bit loads, __dp4a, __byte_perm (3-bit selectors),
//   __vsubss4 and __vsub4 as exact integer arithmetic; iq4_table_lookup: the IQ4 codebook's two-stage byte lookup
//   (native_mmvq.cu:441-457; dense_tc.cu:79-84 computes the same bytes).
// - q8_d, q8_s, q8_w: a q8_1 block's scale, sum and code words; q2_0_load, q2_0_sumi: the Q2_0 decode and its exact
//   integer dot, shared by native_mmvq and the routed experts.
#pragma STDC FP_CONTRACT OFF

static inline float ftz(float x) {
    const uint b = as_type<uint>(x);
    return (b & 0x7f800000u) == 0u ? as_type<float>(b & 0x80000000u) : x;
}
static inline float add_ftz(float a, float b) { return ftz(ftz(a) + ftz(b)); }

// fma_ftz's check where the hardware fma of flushed operands, a and b nonzero, gave a zero r. The exact v = a * b + c
// in units of 2^-150: |a * b| = q * 2^k (q the 48-bit product of the significands), |c| = mc * 2^ec (ec its exponent
// field; 0 for a zero c). v rounds to +-2^-126 exactly when floor(|v|) is 2^24 - 1 units: from the tie 2^-126 - 2^-150
// (2^-126 is the even neighbour) up to 2^-126. With k > 0, v is a multiple of 2^-149; with ec > 25 (and k <= 0), |v| is
// above 2^-102: either way r is v's flushed value.
static inline float fma_ftz_min_normal(float a, float b, float c, float r) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b), uc = as_type<uint>(c);
    const int k = int((ua >> 23) & 0xffu) + int((ub >> 23) & 0xffu) - 150;
    const uint ec = (uc >> 23) & 0xffu;
    if (k > 0 || ec > 25u) return r;
    const ulong q = ulong((ua & 0x7fffffu) | 0x800000u) * ulong((ub & 0x7fffffu) | 0x800000u);
    const uint s = uint(-k);                                              // 0 <= s <= 148
    const ulong p = s < 64u ? q >> s : 0ul;                               // floor(|a * b|)
    const bool f = s != 0u && (s >= 64u || (q << (64u - s)) != 0ul);      // |a * b| has a fraction of a unit
    long n = ((ua ^ ub) >> 31) != 0u ? -long(p) - long(f) : long(p);      // floor(a * b)
    if (ec != 0u) {                                                       // c: whole units
        const long m = long((uc & 0x7fffffu) | 0x800000u) << ec;
        n += (uc >> 31) != 0u ? -m : m;
    }
    const long fl = n < 0 ? -n - long(f) : n;                             // floor(|v|); n < 0 exactly when v < 0
    return fl == long(0xffffff) ? as_type<float>((n < 0 ? 0x80000000u : 0u) | 0x00800000u) : r;
}
static inline float fma_ftz(float a, float b, float c) {
    a = ftz(a);
    b = ftz(b);
    c = ftz(c);
    const float r = ftz(fma(a, b, c));
    return r != 0.0f || a == 0.0f || b == 0.0f ? r : fma_ftz_min_normal(a, b, c, r);
}
static inline float mul_ftz(float a, float b) { return fma_ftz(a, b, -0.0f); }
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
// basalt's warp_sum in an IEEE-built file (iq_kernels.cu:545-549): the same butterfly, plain FADD. The port uses it
// only where no subnormal can reach the sum (docs/research/basalt-nvcc/experts.md), so the GPU's add is CUDA's.
static inline float warp_sum(float x) {
    for (ushort o = 16; o > 0; o >>= 1) x = x + shfl_xor(x, o);
    return x;
}

// CUDA's __expf(-g) in an IEEE-built file: ex2.approx.f32 of fl(g * -log2 e) (experts.md section 2). MUFU.EX2 has no
// documented Metal counterpart; the contract maps it to fast::exp2 of the same product.
static inline float expf_neg(float g) { return fast::exp2(g * as_type<float>(0xBFB8AA3Bu)); }

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

// __vsub4(a, b): per byte a - b, modulo 256.
static inline int vsub4(int a, int b) { return as_type<int>(as_type<uchar4>(a) - as_type<uchar4>(b)); }

// The IQ4 codebook as four words and its two-stage byte lookup: eight 4-bit codes q4 to their int8 values, .x the low
// nibbles', .y the high nibbles' (native_mmvq.cu:441-457; dense_tc.cu:56-57, 79-84 with the same table).
static inline int2 iq4_table_lookup(int q4) {
    const int t0 = int(0xbfad9881u), t1 = int(0xf6eaddcfu), t2 = int(0x26190d01u), t3 = int(0x71594535u);
    const int sel = int(0x32103210u | ((uint(q4) & 0x88888888u) >> 1));
    int tmp[2];
    for (int i = 0; i < 2; ++i) {
        const int shift = 16 * i;
        const int low = byte_perm(t0, t1, q4 >> shift);
        const int high = byte_perm(t2, t3, q4 >> shift);
        tmp[i] = byte_perm(low, high, int(uint(sel) >> shift));
    }
    return int2(byte_perm(tmp[0], tmp[1], 0x6420), byte_perm(tmp[0], tmp[1], 0x7531));
}

// q8_1 block b of an activation row (basalt_q8_1.metal's layout): 9 words, word 0 the fp16 pair (d, s), words 1..8
// the codes. P is a device or threadgroup pointer to the row's words.
template <typename P> static inline float q8_d(P x, int b) { return h2f(ushort(x[9 * b] & 0xffffu)); }
template <typename P> static inline float q8_s(P x, int b) { return h2f(ushort(x[9 * b] >> 16)); }
template <typename P> static inline int q8_w(P x, int b, int k) { return int(x[9 * b + 1 + k]); }

// Q2_0 (18-byte blocks: fp16 d, qs[16] of 2-bit codes, value = d * (code - 1)): basalt's Split<42> / Q20Traits
// (iq_kernels.cu:864-895, native_mmvq.cu:976-1007). q2_0_load decodes the 32-value half iqs of the block at blk
// (qs read as int16: the block is only 2-byte aligned), each code c as the byte c - 1 through __byte_perm;
// q2_0_sumi is the exact integer dot with q8_1 block b of x.
struct Q20W { int qx[4]; int qy[4]; float d2; };
static inline Q20W q2_0_load(device const ushort* blk, int iqs) {
    Q20W r;
    r.d2 = h2f(blk[0]);
    for (int j = 0; j < 4; ++j) {
        const int q = int(short(blk[1 + iqs * 4 + j]));
        const int qe = byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = byte_perm(0x020100ff, 0x020100ff, q >> 2);
        r.qx[j] = byte_perm(qe, qo, 0x5140);
        r.qy[j] = byte_perm(qe, qo, 0x7362);
    }
    return r;
}
template <typename P> static inline int q2_0_sumi(thread const Q20W& r, P x, int b) {
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        sumi = dp4a(q8_w(x, b, j * 2), r.qx[j], sumi);
        sumi = dp4a(q8_w(x, b, j * 2 + 1), r.qy[j], sumi);
    }
    return sumi;
}
