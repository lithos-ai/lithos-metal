// x86's scalar NaN rules, applied lane-exact on ARM. basalt's CPU tier runs on x86-64, and two of its pieces have
// to return the same bits here: the Q2_0 expert kernels (basalt_cpu.cpp) and the expert pool's own SwiGLU
// (basalt_cpu_expf.cpp). Both reach the same two rules, so they share them.
//
// An SSE or AVX arithmetic instruction handed a NaN returns the FIRST of its sources that is one, quieted, where
// an AArch64 instruction prefers a signalling NaN wherever it sits and otherwise takes its own order. A NaN the
// operation itself CREATES is the floating-point indefinite 0xffc00000, sign bit set, where AArch64 writes the
// default NaN 0x7fc00000 ([S] Intel SDM Vol. 1 4.8.3.7; [S] Arm ARM A1.5.3).
//
// Each helper therefore selects the NaN by an explicit bit test, in the operand order of the instruction GCC
// emits at that site, and no NEON NaN rule is consulted. The operand order is the compiler's rather than the
// source's, so every call site names its operands in the order its disassembly has them.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace monolith::basalt_cpu::x86 {

inline uint32_t to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}
inline float from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline bool is_nan(float x) { return (to_bits(x) & 0x7F800000u) == 0x7F800000u && (to_bits(x) & 0x007FFFFFu) != 0; }
inline float quiet(float x) { return from_bits(to_bits(x) | 0x00400000u); }   // an SNaN keeps its payload, bit 22 set

/// VMULPS, VADDPS, VADDSS, VSUBSS, DIVSS and MULSS: SRC1's NaN if it has one, else SRC2's, each quieted ([S] Intel
/// SDM Vol. 1 Table 4-8; [M] the recorded ground truth of both tiers); a NaN the operation CREATES (inf * 0,
/// inf - inf, inf / inf, 0 / 0) is the floating-point indefinite 0xffc00000 (4.8.3.7).
inline float x86_src1_first(float src1, float src2, float computed) {
    if (is_nan(src1)) return quiet(src1);
    if (is_nan(src2)) return quiet(src2);
    return is_nan(computed) ? from_bits(0xFFC00000u) : computed;
}
inline float x86_mul(float src1, float src2) { return x86_src1_first(src1, src2, src1 * src2); }
inline float x86_add(float src1, float src2) { return x86_src1_first(src1, src2, src1 + src2); }
inline float x86_sub(float src1, float src2) { return x86_src1_first(src1, src2, src1 - src2); }
inline float x86_div(float src1, float src2) { return x86_src1_first(src1, src2, src1 / src2); }

/// VFMADD231PS: the first NaN in the order first multiplicand, second multiplicand, addend, quieted ([S] Vol. 1
/// Table 14-17; [M] ground-truth:137-141). A QNaN addend therefore survives an inf * 0 product, where FMLA
/// discards it for the default NaN.
inline float x86_fma(float m1, float m2, float addend) {
    if (is_nan(m1)) return quiet(m1);
    if (is_nan(m2)) return quiet(m2);
    if (is_nan(addend)) return quiet(addend);
    const float r = std::fma(m1, m2, addend);
    return is_nan(r) ? from_bits(0xFFC00000u) : r;
}

}  // namespace monolith::basalt_cpu::x86
