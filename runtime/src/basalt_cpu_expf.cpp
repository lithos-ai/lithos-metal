// SPDX-License-Identifier: MIT
// Copyright (c) 2017-2025, Arm Limited (MIT OR Apache-2.0 WITH LLVM-exception; taken under MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from ARM-software/optimized-routines math/expf.c and math/exp2f_data.c @
// f67e7daa5457646fad9f75580263dba654c988aa (third_party/NOTICE, third_party/ARM-OPTIMIZED-ROUTINES-LICENSE), with
// the error returns of math/math_errf.c and the helpers of math/math_config.h at the same commit; and from basalt
// (jesdga95/basalt) src/kernels/cpu/pool.cpp:434 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
// third_party/NOTICE): the SwiGLU expression, in its order.
//
// WHICH BUILD OF ARM'S SOURCE THIS IS. The file has knobs; glibc picks these, and they are the ones transcribed.
//   * The `#else` of TOINT_INTRINSICS: `kd = z + SHIFT; ki = asuint64(kd); kd -= SHIFT`, which rounds to nearest
//     EVEN. glibc leaves TOINT_INTRINSICS at the 0 its own header defaults it to, and defines it nowhere else, so
//     every architecture's glibc compiles that branch [M, glibc 2.39's single sysdeps/ieee754/flt-32/math_config.h,
//     lines 40-44]. `round`/`lround`, which the other branch uses, take a tie away from zero instead, and that
//     branch would be the same function here anyway: over all 2^32 patterns on which the main path runs,
//     `InvLn2N * (double) x` is never exactly k + 0.5 [M, swept here], so no argument reaches a tie.
//   * WANT_ERRNO_UFLOW = 1, which glibc derives from WANT_ROUNDING && WANT_ERRNO: x in
//     [-0x1.9fe368p6, -0x1.9d1d9ep6) returns (0x1.4p-75f)^2 rather than taking the main path. Under round to
//     nearest that changes errno and not the result: the main path computes 0x00000001 at all 90,854 arguments of
//     the band [M, swept here], which is what the branch returns. It is transcribed because glibc has it.
//   * `poly_scaled` and `invln2_scaled`, the N = 32 rows; `poly` and `shift_scaled` belong to exp2f and powf.
// `eval_as_double` and `eval_as_float` are Arm's no-ops for FLT_EVAL_METHOD == 0, which is arm64 and x86-64 SSE2;
// the casts are written out instead, and `unlikely` on the special-case branch, a hint with no arithmetic in it,
// is dropped. errno is not set - ERANGE is all glibc adds there, nothing in this tree reads it, and the returned
// bits are the contract.
//
// WHICH VARIANT, AND WHAT FUSES IN IT. Arm writes every multiply-add with operators and leaves contraction to the
// compiler, so the source alone does not say what the function is. glibc's x86-64 multiarch holds __expf_fma,
// this same source compiled with -mfma -mavx2, and an ifunc picks it on any host with FMA and AVX2 - every host
// that can run basalt's AVX-512 expert kernels. That is the variant basalt runs and the one transcribed here. The
// contraction sites are the three of the polynomial - `C[0]*r + C[1]`, `C[2]*r + 1`, `z*r2 + y` - and, once
// `z = InvLn2N * xd` is substituted, `z + SHIFT` and `z - kd`; `r2 = r*r` and `y = y*s` have nothing to fuse into.
// Both orders were built here and swept over all 2^32 float inputs [M, Apple clang 21.0.0, -O2]:
//   * The three polynomial sites: `std::fma` at all three returns the identical bits at every one of the 2^32
//     inputs. The double carries 29 bits more than the float it is rounded back to, which is why fusing a double
//     rounding this far from the boundary cannot be seen. Written as Arm writes them, left to the build flag.
//   * `z + SHIFT` and `z - kd`: fusing those two changes 2 of the 2^32 - 0x4202422f (32.5646324), where unfused
//     gives 0x56fc9f1b and fused 0x56fc9f1c, and 0xc27c65d9 - because the fused form subtracts kd from the
//     unrounded product, where `z - kd` as written subtracts it from the rounded z and is itself exact. A
//     different r is a different function, and this is the one place the two glibc variants part: __expf_fma
//     fuses them and __expf_sse2 does not [M, Xeon Platinum 8375C, glibc 2.39-0ubuntu8.9, the ifunc intact and
//     again with GLIBC_TUNABLES disabling AVX2 and FMA]. They are written as explicit `std::fma` below so that
//     -ffp-contract cannot decide which function this is.
// The file is still built with -ffp-contract=off (runtime/CMakeLists.txt): that pins the polynomial, where Arm's
// operators stand, and the two explicit calls stand over it.
//
// WHERE NaN BITS COME FROM. x86 writes the floating-point indefinite 0xffc00000, sign bit set, where AArch64
// writes the default NaN 0x7fc00000 ([S] Intel SDM Vol. 1 4.8.3.7; [S] Arm ARM A1.5.3), and an SSE scalar
// operation that is handed a NaN returns the first of its sources that is one, quieted. expf itself needs no
// helper - its NaN return is `x + x`, which quiets the argument's payload and sign on both architectures - and
// neither does the SwiGLU's add, whose only invalid form is inf + -inf and whose `e` is never negative [M: swept
// here, expf_glibc returns no negative value at any of the 2^32 arguments]. The divide and the multiply do reach
// the indefinite from finite or infinite inputs. All three are written below as the x86 instructions basalt's
// codegen emits rather than as C operators, the add for fidelity to what basalt executes.

#include "basalt_cpu_expf.h"

#include "basalt_x86.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace monolith::basalt_cpu {

using namespace x86;
namespace {

constexpr int EXP2F_TABLE_BITS = 5;
constexpr int N = 1 << EXP2F_TABLE_BITS;

// exp2f_data.c, the N == 32 rows, copied. tab[i] = uint(2^(i/N)) - (i << 52-BITS), used for computing 2^(k/N) for
// an int |k| < 150 N as double(tab[k%N] + (k << 52-BITS)).
constexpr uint64_t T[N] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};
constexpr double InvLn2N = 0x1.71547652b82fep+0 * N;
constexpr double SHIFT = 0x1.8p+52;
constexpr double C[3] = {0x1.c6af84b912394p-5 / N / N / N, 0x1.ebfce50fac4f3p-3 / N / N, 0x1.62e42ff0c52d6p-1 / N};

inline uint32_t asuint(float f) { return std::bit_cast<uint32_t>(f); }
inline uint64_t asuint64(double f) { return std::bit_cast<uint64_t>(f); }
inline double asdouble(uint64_t i) { return std::bit_cast<double>(i); }
inline uint32_t top12(float x) { return asuint(x) >> 20; }

// math_config.h's barrier for a target without the aarch64 asm, and math_errf.c's xflowf: the value is the
// product, so the bits and the floating-point flag both come out of the arithmetic rather than a literal.
inline float opt_barrier_float(float x) {
    volatile float y = x;
    return y;
}
inline float xflowf(uint32_t sign, float y) { return opt_barrier_float(sign ? -y : y) * y; }
inline float math_oflowf(uint32_t sign) { return xflowf(sign, 0x1p97f); }          // -> +-inf
inline float math_uflowf(uint32_t sign) { return xflowf(sign, 0x1p-95f); }         // -> +-0
inline float math_may_uflowf(uint32_t sign) { return xflowf(sign, 0x1.4p-75f); }   // -> +-0x1p-149

}  // namespace

// expf.c, operation for operation. EXP2F_TABLE_BITS = 5, EXP2F_POLY_ORDER = 3; Arm records the worst case as
// 0.502 ulp (nearest rounding) and the relative error before rounding as 1.69 * 2^-34 in [-ln2/64, ln2/64].
float expf_glibc(float x) {
    uint32_t abstop;
    uint64_t ki, t;
    double kd, xd, z, r, r2, y, s;

    xd = x;
    abstop = top12(x) & 0x7ff;
    if (abstop >= top12(88.0f)) {
        /* |x| >= 88 or x is nan.  */
        if (asuint(x) == asuint(-std::numeric_limits<float>::infinity())) return 0.0f;
        if (abstop >= top12(std::numeric_limits<float>::infinity())) return x + x;   // +inf, or a NaN quieted
        if (x > 0x1.62e42ep6f)  /* x > log(0x1p128) ~= 88.72 */
            return math_oflowf(0);
        if (x < -0x1.9fe368p6f) /* x < log(0x1p-150) ~= -103.97 */
            return math_uflowf(0);
        if (x < -0x1.9d1d9ep6f) /* x < log(0x1p-149) ~= -103.28 */
            return math_may_uflowf(0);
    }

    /* x*N/Ln2 = k + r with r in [-1/2, 1/2] and int k.  */
    z = InvLn2N * xd;

    /* Round and convert z to int, the result is in [-150*N, 128*N] and
       ideally nearest int is used, otherwise the magnitude of r can be
       bigger which gives larger approximation error.  */
    // Written as explicit fma, not left to the compiler: glibc's x86-64 ifunc picks __expf_fma on any host with
    // FMA and AVX2, which is every host basalt's AVX-512 expert kernels run on, and its disassembly fuses these two
    // (libm+0x7dc40, glibc 2.39-0ubuntu8.9, measured). The two variants differ at exactly two of the 2^32 float32
    // arguments, 0x4202422f and 0xc27c65d9, where the fused rounding step rounds the other way; basalt holds the
    // fused bits, so the port computes them whatever -ffp-contract says.
    kd = std::fma(InvLn2N, xd, SHIFT);
    ki = asuint64(kd);
    kd -= SHIFT;
    r = std::fma(InvLn2N, xd, -kd);

    /* exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
    t = T[ki % N];
    t += ki << (52 - EXP2F_TABLE_BITS);
    s = asdouble(t);
    z = C[0] * r + C[1];
    r2 = r * r;
    y = C[2] * r + 1;
    y = z * r2 + y;
    y = y * s;
    return (float) y;
}

float swiglu(float g, float u) {
    // gcc 14.3 at -O3 compiles pool.cpp:434 to a sign flip, one scalar `call expf`, `addss` 1.0f, `divss`, `mulss`
    // and a store [M, docs/research/basalt-cpu-expf.md section 2]. The add takes `e` as SRC1 and the constant as
    // SRC2, which is the source's operands swapped and exact, since 1.f is never a NaN;
    // the divide takes `g`, which does not commute; the multiply takes the quotient, which is why a NaN `g` with
    // a NaN `u` returns the quotient's payload and not `u`'s.
    const float e = expf_glibc(-g);
    const float d = x86_add(e, 1.f);
    const float q = x86_div(g, d);
    return x86_mul(q, u);
}

}  // namespace monolith::basalt_cpu
