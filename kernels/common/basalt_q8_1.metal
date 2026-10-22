// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:146-177, src/kernels/cuda/iq_kernels.cu:1880-1900
// and include/basalt/kernels/q8_1_finite.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// which adapt ggml-org/llama.cpp ggml/src/ggml-cuda/quantize.cu @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT;
// third_party/NOTICE).
//
// basalt's q8_1 activation quantizers. A q8_1 block is 36 bytes: the fp16 scale d (bytes 0-1), the fp16 sum s of the
// block's 32 raw values (bytes 2-3), 32 int8 codes (bytes 4-35); column j's blocks follow column j - 1's, so a vector
// of ncols * n_in values (n_in % 32 == 0) is ncols columns. One thread per value, 256 per threadgroup, SIMD-group =
// block (whole SIMD-groups past the end return). Needs basalt_fp.metal ahead of it.
//
// - native_quantize_q8_1: the sequence nvcc compiles for native_mmvq.cu (--use_fast_math; matvec.md section 1):
//   flush-to-zero abs, max and add; d = fl(amax * fl(1/127)) (ptxas replaces the approximate division by 127 with
//   this multiply); the codes' xi / d as MUFU.RCP then FMUL.FTZ, here the correctly rounded reciprocal and the same
//   multiply; roundf; the +-127 clamp with NaN to 0.
// - native_swiglu_quantize_q8_1: the shared expert's SwiGLU value (native_mmvq.cu:170; matvec.md section 2), then the
//   native sequence. __expf is MUFU.EX2 of fl(g * -log2 e), here fast::exp2 of the same product.
// - quantize_q8_1: quantize_q8_1_kernel (iq_kernels.cu, nvcc's IEEE defaults; matvec.md section 3): correctly rounded
//   divisions and no flushing. This GPU flushes subnormal inputs and results of add and division, and its fmax
//   compares a subnormal as zero, returning the second operand of a tie rather than the larger
//   (tests/kernels/test_basalt_platform.py), so wherever a subnormal can occur the IEEE result is rebuilt from integer
//   operations: amax as the largest bit pattern, the block sum and the codes on operands scaled by 2^64, d = amax / 127
//   on the subnormal grid by integer division. Above 2^-100 every one of these is the plain operation.
#pragma STDC FP_CONTRACT OFF

struct Q81Params { uint n; uint pad0; uint pad1; uint pad2; };   // n: values (ncols * n_in), a multiple of 32

constant constexpr float Q8_1_HALF_MAX = 65504.0f;

static inline float q8_1_finite(float v) { return fabs(v) > Q8_1_HALF_MAX ? copysign(Q8_1_HALF_MAX, v) : v; }

// (int8_t) of the clamped roundf: q > 127 -> 127, q < -127 -> -127, NaN -> 0 (F2I.TRUNC.NTZ / cvt.rzi of NaN).
static inline char q8_1_code(float q) {
    return isnan(q) ? char(0) : char(int(q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q));
}

static inline void q8_1_put(device uchar* y, uint i, char code, float d, float sum) {
    y[36u * (i / 32u) + 4u + i % 32u] = as_type<uchar>(code);
    if (i % 32u == 0u) {
        *(device uint*) (y + 36u * (i / 32u)) = as_type<uint>(half2(half(q8_1_finite(d)), half(q8_1_finite(sum))));
    }
}

// The native sequence for value xi of block i / 32 (lane i % 32): the device function every native call site
// shares (native_quantize_q8_1, the SwiGLU quantizer, and the fused epilogues that reproduce it).
static inline void q8_1_store_native(float xi, device uchar* y, uint i) {
    const float amax = warp_max_ftz(abs_ftz(xi));
    const float sum = warp_sum_ftz(xi);
    const float d = q8_1_finite(mul_ftz(amax, as_type<float>(0x3C010204u)));   // fl(1/127)
    const char code = amax == 0.0f ? char(0) : q8_1_code(round(mul_ftz(xi, rcp_ftz(d))));
    q8_1_put(y, i, code, d, sum);
}

kernel void native_quantize_q8_1(device const float* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                                 constant Q81Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;                    // n is a multiple of 32: only whole SIMD-groups return
    q8_1_store_native(x[i], y, i);
}

kernel void native_swiglu_quantize_q8_1(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                                        device uchar* y [[buffer(2)]], constant Q81Params& p [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float g = gate[i];
    const float e = ftz(fast::exp2(mul_ftz(g, as_type<float>(0xBFB8AA3Bu))));   // __expf(-g): ex2.approx(g * -log2 e)
    const float s = mul_ftz(g, rcp_ftz(add_ftz(1.0f, e)));                      // __fdividef(g, __fadd_rn(1, e))
    q8_1_store_native(mul_ftz(s, up[i]), y, i);                                 // __fmul_rn(s, up)
}

// ---- IEEE operations where a subnormal can occur ---------------------------------------------------------------

constant constexpr uint IEEE_SMALL = 27u << 23;   // 2^-100: below it an operand or result may be subnormal

// x * 2^64, exactly, for |x| < 2^-100 (a subnormal's significand rebuilt as an integer: the hardware would flush it)
static inline float ieee_up64(float x) {
    const uint b = as_type<uint>(x);
    if ((b & 0x7f800000u) != 0u) return x * 0x1p64f;
    return copysign(float(b & 0x007fffffu) * 0x1p-85f, x);
}

// s * 2^-64 rounded as IEEE would: exact when the result is normal; a subnormal result is exact too (see ieee_add)
static inline float ieee_down64(float s) {
    if (fabs(s) >= 0x1p-62f || s == 0.0f) return s * 0x1p-64f;
    return as_type<float>((as_type<uint>(s) & 0x80000000u) | uint(fabs(s) * 0x1p85f));
}

// a + b with IEEE subnormals: below 2^-100 both operands are scaled by 2^64 (exactly), added, and scaled back; a sum
// whose exact value is subnormal is representable (Hauser), so the scaled sum is exact and the integer rebuild too.
static inline float ieee_add(float a, float b) {
    if ((as_type<uint>(a) & 0x7fffffffu) >= IEEE_SMALL || (as_type<uint>(b) & 0x7fffffffu) >= IEEE_SMALL) return a + b;
    return ieee_down64(ieee_up64(a) + ieee_up64(b));
}

// a * b with IEEE subnormals (CUDA's mul.rn.f32 in an IEEE-built kernel). Two operands with nonzero exponent fields
// (normal, infinite or NaN) whose product is at least 2^-125, infinite or NaN take the hardware's correctly rounded
// product. An infinity or NaN against a zero or a subnormal (which the hardware flushes, making a subnormal times inf
// 0 * inf) gives IEEE's result from the bits: NaN (CUDA's canonical 0x7fffffff) for a NaN or a zero operand, else the
// signed infinity. Otherwise (a subnormal operand, or a product the hardware may flush) the product is rebuilt from
// the exact integer product m of the significands: |a * b| = m * 2^e units of 2^-149, rounded to nearest even on the
// grid of its binade (k = 0 is the subnormal grid and binade -126), so that the float's bits are (k << 23) + r for r
// units of 2^(k - 149).
static inline float ieee_mul(float a, float b) {
    const float p = a * b;
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b);
    uint ea = (ab >> 23) & 0xffu, eb = (bb >> 23) & 0xffu, ma = ab & 0x7fffffu, mb = bb & 0x7fffffu;
    if (ea != 0u && eb != 0u && !(fabs(p) < 0x1p-125f)) return p;      // p infinite or NaN included
    const uint sign = (ab ^ bb) & 0x80000000u;
    const bool za = (ea | ma) == 0u, zb = (eb | mb) == 0u;             // zeros by their bits (a compare may flush)
    if (ea == 0xffu || eb == 0xffu)                                     // the other operand is a zero or a subnormal
        return as_type<float>((ea == 0xffu && ma != 0u) || (eb == 0xffu && mb != 0u) || za || zb
                              ? 0x7fffffffu : sign | 0x7f800000u);
    if (za || zb) return as_type<float>(sign);
    if (ea == 0u) ea = 1u; else ma |= 0x800000u;
    if (eb == 0u) eb = 1u; else mb |= 0x800000u;
    const ulong m = ulong(ma) * ulong(mb);
    const int e = int(ea + eb) - 151;                                   // |a * b| = m * 2^e units of 2^-149
    const int lead = 63 - int(clz(m));
    const int k = max(0, lead + e - 23);                                // the grid: 24 significant bits, or 2^-149
    const int s = k - e;                                                // m * 2^e / 2^k = m >> s
    ulong r;
    if (s <= 0) {
        r = m << uint(-s);
    } else if (s > 63) {
        r = 0;
    } else {
        r = m >> uint(s);
        const ulong rem = m & ((1ul << uint(s)) - 1ul), tie = 1ul << uint(s - 1);
        if (rem > tie || (rem == tie && (r & 1ul) != 0ul)) r += 1ul;
    }
    return as_type<float>(sign | (uint(k) << 23) + uint(r));            // r == 2^24 carries into the exponent
}

static inline void q8_1_store_ieee(float xi, device uchar* y, uint i) {
    int key = isnan(xi) ? -1 : int(as_type<uint>(fabs(xi)));        // fmaxf drops a NaN operand
    float sum = xi;
    for (ushort o = 16; o > 0; o >>= 1) {
        key = max(key, simd_shuffle_xor(key, o));
        sum = ieee_add(sum, shfl_xor(sum, o));
    }
    const float amax = key < 0 ? as_type<float>(0x7fffffffu) : as_type<float>(uint(key));
    const bool small = uint(key) < IEEE_SMALL;                        // NaN (key -1) is not small
    float d;                                                          // (amax compared by its bits: a compare may flush)
    if (!small || key == 0) {
        d = amax / 127.0f;
    } else if (key >= (8 << 23)) {                                    // amax >= 2^-119: d is normal, scaled division
        d = (ieee_up64(amax) / 127.0f) * 0x1p-64f;
    } else {                                                          // d on the subnormal grid: integer division
        const uint b = as_type<uint>(amax);
        const uint m = (b & 0x7f800000u) == 0u ? b : ((b & 0x007fffffu) | 0x00800000u) << ((b >> 23) - 1u);
        d = as_type<float>(m / 127u + (m % 127u >= 64u ? 1u : 0u));  // round to nearest (127 is odd: no ties)
    }
    d = q8_1_finite(d);
    char code = char(0);
    if (key != 0) code = q8_1_code(round(small ? ieee_up64(xi) / ieee_up64(d) : xi / d));
    q8_1_put(y, i, code, d, sum);
}

kernel void quantize_q8_1(device const float* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                          constant Q81Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    q8_1_store_ieee(x[i], y, i);
}
