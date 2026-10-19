// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cpu/expert.cpp:143-148, :336-370, :503-556 and
// include/basalt/kernels/cpu/expert.hpp:25-61 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// which adapt ggml-org/llama.cpp ggml/src/ggml-common.h (block_q2_0) and ggml/src/ggml-quants.c
// (dequantize_row_q2_0's code rule) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// ============================== the transcription contract ==============================
//
// Every output bit must equal what basalt's AVX-512 kernels produce for the same input. No exception is left: the
// one place where the NEON path writes different bits is the payload of a NaN res[t], and THE NaN PATH below
// recomputes that row and token on a scalar path that applies x86's rules instead. These are the steps where the
// obvious NEON intrinsic silently changes bits, and what stands in for each AVX-512 instruction:
//
//   _mm512_and_ps(x, 0x7fffffff)      -> vabsq_f32            identical: clears the sign bit, NaN stays NaN.
//   _mm512_max_ps, every reduce node  -> maxps() below        NOT vmaxq_f32 (FMAX keeps the NaN) and NOT
//                                                             vmaxnmq_f32 (FMAXNM keeps the number): MAXPS returns
//                                                             SRC2 on an unordered compare and on (+-0, -+0).
//   _mm512_reduce_{add,max}_ps        -> the trees below      NOT vaddvq_f32 / vmaxvq_f32: FADDP pairs
//                                                             (t0+t1)+(t2+t3) where x86 gives (t0+t2)+(t1+t3), and
//                                                             FMAXV has the wrong NaN rule.
//   _mm512_mul_ps then _mm512_add_ps  -> vmulq_f32+vaddq_f32  TWO roundings. GCC 12.2/13.2/14.3/15.2 and Clang 21
//   (and p = d * s, rounded before use)                       all emit a separate vmulps and vaddps for these, so
//                                                             this TU is built -ffp-contract=off (CMakeLists.txt):
//                                                             a contracted fma(x, inv, +-0.5) changes codes.
//   _mm512_fmadd_ps (acc, corr)       -> vfmaq_f32            one rounding; FMA is wanted here, which is why
//                                                             contraction is turned off rather than globally fused.
//   _mm512_cvttps_epi32               -> cvttps() below       FCVTZS SATURATES (NaN -> 0, +inf -> INT32_MAX), which
//                                                             after the clamp would give 0 and +127 where x86 gives
//                                                             -127 for both; x86 returns the integer indefinite.
//   _mm512_dpbusd_epi32               -> vdotq_s32            SDOT is signed x signed, VPDPBUSD unsigned x signed -
//                                                             equal here because the codes are 0..3. Vector j lane k
//                                                             covers weights 16j+4k .. 16j+4k+3, i.e. the x86 dword
//                                                             lane L = 4j + k, which holds weights 4L .. 4L+3.
//   _mm512_multishift_epi64_epi8      -> vqtbl1q_u8 + vshlq_u8 + vandq_u8, decoding the documented rule
//                                       code m = (qs[m/4] >> 2(m%4)) & 3 (ggml_quants.c:220-222), which is what
//                                       VPMULTISHIFTQB with basalt's (ctrl, lanes) argument order produces.
//   _mm256_cvtph_ps                   -> vcvt_f32_f16         both exact for every half, subnormals included, with
//                                                             MXCSR and FPCR at their defaults.
//   _mm512_cvtepi32_epi8, _mm512_reduce_add_epi32 -> vmovn_s32/vmovn_s16, vaddvq_s32: exact after the clamp, and
//                                                             the integer sum is order-independent (|sum| <= 4064).
//
// The 16 x86 lanes are four float32x4_t: lane L lives in vector L/4 at position L%4.
//
// THE NaN PATH. res[t] is the one output whose bits the NEON path gets wrong, in two ways: AArch64 writes its
// default NaN 0x7fc00000 where x86 writes the floating-point indefinite 0xffc00000, sign bit set ([S] Intel SDM
// Vol. 1 4.8.3.7), and FMLA prefers the addend's NaN where VFMADD231PS prefers the multiplicand's. NaN-ness itself
// never differs - both ISAs make a NaN at the same sites from the same inputs - so the vector result decides:
// where res[t] is a NaN, x86_row() below recomputes that (row, token) lane by lane under x86's rules, picking the
// NaN at every site by an explicit bit test rather than letting FMUL, FMLA or FADD pick. That is not pedantry:
// this port's own NaN payloads move with clang's optimisation level, and LLVM's LangRef entitles no expression
// written in C to a particular payload. The replay runs on no finite row, so it costs nothing.
//
// What the port then guarantees, from the AVX-512 ground truth (a Xeon Platinum 8488C running basalt's own
// kernels at basalt's own flags; .context/basalt-port/pr21-ground-truth/pr21-x86-ground-truth.md, cited below as
// ground-truth:<line>):
//   - A row whose fp16 block scales are all non-NaN, with activations from act_quant_q8_1 - which never writes a
//     NaN, since amax > 0 is an ordered compare and s = inf needs an infinite lane, whose sum <= -127 makes
//     hx = s * sum = -inf rather than inf * 0: every NaN in res[t] was GENERATED, every generated NaN on x86 is
//     the indefinite, and a propagated indefinite keeps its bits, so res[t] is exactly 0xffc00000. Measured over
//     2,506 such outputs under each of GCC 12.5, 13.4, 14.3 and 15.2 (ground-truth:375-378).
//   - A row a NaN ENTERS - a NaN fp16 block scale, or a NaN planted in scale or hx through the binding - has no
//     single x86 answer. The two halves of acc can end on different NaNs, and which one survives depends on the
//     token slot, on how many tokens share the job and on the GCC version (ground-truth:379-389), so basalt's own
//     x86 output there contradicts its promise that "Every token's output is bitwise the single-token job's"
//     (pool.hpp:54-55). This port pins one of those answers: GCC 14.3's NT = 1 operand order, for every token
//     slot, which is basalt 14.3's single-token job bit for bit and keeps the promise. That pin is a SECOND
//     build-dependent value beside amax below - a basalt built by another GCC, or run with nt > 1, may hold
//     different bits in this case alone.
//
// THE SUPPORTED INPUT RANGE, a decision of the engineering review. basalt's act_quant_q8_1 writes no NaN scale or
// hx for any float input (expert.cpp:352-367): a NaN amax fails amax > 0.f, so s = +0 (:353) and hx = +0 * sum is
// a zero (:367); a finite amax gives a finite s, so hx is finite or overflows to an infinity; and amax = +inf needs
// an infinite lane, whose t = inf * (1.f / s) = inf * 0 (:354, :356) is a NaN that VCVTTPS2DQ converts to INT_MIN
// and the clamp to -127, while every finite lane's t = +-0 gives 0 (:357-361), so sum <= -127 (:364) and
// hx = +inf * sum = -inf. Case A11 asserts the exact outputs for a single +inf, a single NaN and +inf with -inf.
// A NaN therefore enters a row only through a NaN fp16 block scale in the weights, which a valid model file does
// not hold: the pinned GCC 14.3 single-token rule above decides only rows of a corrupt weight file and test inputs,
// and every NaN generated from non-NaN inputs, an infinite activation's included, is 0xffc00000 under every build
// measured - basalt's under GCC 12.5, 13.4, 14.3 and 15.2 (ground-truth:375-378), this port's at -O0 and at the
// Release flags.
//
// THE TWO COMPILER-DEPENDENT VALUES. The first is amax. _mm512_reduce_max_ps is header-defined and basalt pins
// no compiler (README.md:35-43, CMakeLists.txt:45-46). GCC 14's gcc/config/i386/avx512fintrin.h:16339-16350 (used by
// _mm512_reduce_max_ps at :16359-16364) passes the HIGH half as SRC1 at the first two levels; Clang 21's
// avx512fintrin.h:9580 lowers __builtin_ia32_reduce_fmax_ps512 with the LOW half as SRC1. Levels 3 and 4 agree.
// MAXPS returns SRC2 on an unordered compare, so the two orders swallow different lanes and a chunk containing a
// NaN can get a different amax - sometimes a different FINITE amax. basalt's own binary therefore differs between
// its own builds and no bit-contract exists for a NaN activation. Clang's lowering also carries nnan, so there a
// NaN chunk's amax is undefined rather than merely differently ordered, and the low-half tree is the measured
// realisation of it [M]. This port writes GCC's order (basalt builds only on Linux - AGENTS.md:3, README.md:33 -
// where the default c++ is g++); flipping it is the operand order of the three maxps calls at the first two levels
// of reduce_max_ps. The second is the NaN path's pinned tree order above. _mm512_reduce_add_ps needs no choice for
// a NUMERIC result, but not because addition commutes: a reduction's bits follow its GROUPING, and both headers
// expand to the same four-level pairwise tree (GCC's avx512fintrin.h:16297-16306; Clang's reassoc
// llvm.vector.reduce.fadd, measured lowering [M]). Clang's reassoc licenses any association, so that agreement is
// a measurement to re-check, not a guarantee. Its per-node operand order is invisible to a numeric result and
// decides a NaN one, which is why the NaN path owns that order and the vector trees below do not.
//
// WHERE THE RESEARCH (docs/research/basalt-cpu-q2.md) AND basalt's SOURCE DISAGREE - the source governs:
//   - Step 6's "inv = +inf when s < 2^-128" (:155) is off by one value: 1.f / 2^-128 is already +inf, so the band
//     is s <= 2^-128. The source is a plain 1.f / s and that is what this file computes. A subnormal s ABOVE
//     2^-128 has a finite inv and ordinary codes.
//   - Step 11's clamp acts on both sides. The low side is the integer indefinite: NaN, +-inf and overflow give
//     -127. The high side needs a NaN in the chunk, because the reduce_max tree can then swallow the true peak:
//     x = 0.1 everywhere, x[4] = 0.11, x[16] = NaN gives amax = 0.1 (GCC's maxps drops lane 4 through the NaN),
//     so x[4] scales to r = 140.2 and clamps to +127 [M]. Without a NaN, |x| <= amax and the rounding is monotone,
//     so trunc(r) never exceeds 127 over any positive finite amax with a finite inv (the largest r is 127.50003,
//     at amax = 3.7322113794423347e-37) [M].
//   - A regime the specification does not name: amax > 0 with fl(amax / 127) == 0 makes s = 0, so the s > 0.f
//     guard gives inv = 0 and every code 0, with scale = 0 and hx = +0.0. That is amax <= 63 * 2^-149; from
//     64 * 2^-149 the quotient rounds up to the subnormal s = 2^-149, whose inv is +inf, so every code is -127 [M].
// basalt's act_quant_q8_1 opens with an --cpu-oracle-q8-0 branch (expert.cpp:337-340) that swaps the quantizer for
// quantize_oracle_q8_0, a debug path with an fp16 scale. It is not ported: the pack's sessions never set it, and
// the port reproduces the kernel this path runs, not basalt's debugging options.
#include "basalt_cpu.h"

#if !defined(__ARM_NEON)
#error "basalt_cpu.cpp is an ARM NEON transcription of basalt's AVX-512 CPU kernels; build it on Apple silicon."
#endif

#include <arm_neon.h>

#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace monolith::basalt_cpu {
namespace {

/// MAXPS(SRC1, SRC2) per lane, exactly: "IF ((SRC1 = 0.0) and (SRC2 = 0.0)) THEN DEST := SRC2; ELSE IF (SRC1 = NaN)
/// THEN DEST := SRC2; ... ELSE IF (SRC2 = NaN) THEN DEST := SRC2; ... ELSE IF (SRC1 > SRC2) THEN DEST := SRC1; ELSE
/// DEST := SRC2" (Intel SDM, MAXPS Operation) - an ordered greater-than select, which is false for either NaN and
/// for (+-0, -+0).
inline float32x4_t maxps(float32x4_t src1, float32x4_t src2) {
    return vbslq_f32(vcgtq_f32(src1, src2), src1, src2);
}

/// _mm512_reduce_max_ps's four-level tree over the 16 lanes held as v[0..3], in GCC's operand order (see the
/// contract above): u[j] = maxps(v[j+8], v[j]), t[j] = maxps(u[j+4], u[j]), p0 = maxps(t0, t2), p1 = maxps(t1, t3),
/// res = maxps(p0, p1).
inline float reduce_max_ps(const float32x4_t v[4]) {
    const float32x4_t u0 = maxps(v[2], v[0]);                     // SRC1 = the high half   (Clang: maxps(v[0], v[2]))
    const float32x4_t u1 = maxps(v[3], v[1]);                     //                        (Clang: maxps(v[1], v[3]))
    const float32x4_t t = maxps(u1, u0);                          // SRC1 = the high quarter (Clang: maxps(u0, u1))
    const float32x4_t p = maxps(t, vextq_f32(t, t, 2));           // p[0] = maxps(t0, t2), p[1] = maxps(t1, t3)
    return vgetq_lane_f32(maxps(p, vdupq_laneq_f32(p, 1)), 0);    // maxps(p0, p1)
}

/// _mm512_reduce_add_ps's four-level tree, same grouping. Written in GCC's operand order for symmetry; float
/// addition is bitwise commutative on non-NaN operands, so a numeric result does not depend on it, and a NaN one
/// is not taken from here at all (THE NaN PATH: x86_row() recomputes it).
inline float reduce_add_ps(const float32x4_t v[4]) {
    const float32x4_t u0 = vaddq_f32(v[2], v[0]);
    const float32x4_t u1 = vaddq_f32(v[3], v[1]);
    const float32x4_t t = vaddq_f32(u1, u0);
    const float32x4_t p = vaddq_f32(t, vextq_f32(t, t, 2));       // p[0] = t0 + t2, p[1] = t1 + t3
    return vgetq_lane_f32(p, 0) + vgetq_lane_f32(p, 1);
}

/// _mm512_cvttps_epi32: truncation toward zero, and the integer indefinite 0x80000000 for a NaN, an infinity or a
/// value outside int32 (Intel SDM Vol. 1 Table 11-1 and D.4.2.2, with the invalid exception masked). vcagtq_f32 is
/// false for a NaN and for |r| >= 2^31, so the one select covers all of them; r = +-2^31 exactly lands on
/// INT32_MIN, which is what x86 gives there too.
inline int32x4_t cvttps_epi32(float32x4_t r) {
    return vbslq_s32(vcagtq_f32(vdupq_n_f32(2147483648.0f), r), vcvtq_s32_f32(r), vdupq_n_s32(INT_MIN));
}

/// unpack64_q2_0 (expert.cpp:143-148): the 64 codes of one block in weight order, as four vectors - byte i of w[j]
/// is the code of weight 16j+i, which is (qs[(16j+i)/4] >> (2 * ((16j+i) % 4))) & 3.
inline void unpack64_q2_0(const uint8_t* codes, uint8x16_t w[4]) {
    static const uint8_t kByte[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};   // (16j+i)/4 - 4j
    static const int8_t kShift[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    const uint8x16_t qs = vld1q_u8(codes);
    const uint8x16_t byte = vld1q_u8(kByte);
    const int8x16_t shift = vld1q_s8(kShift);                     // USHL right-shifts where the count is negative
    const uint8x16_t three = vdupq_n_u8(3);
    for (int j = 0; j < 4; ++j)
        w[j] = vandq_u8(vshlq_u8(vqtbl1q_u8(qs, vaddq_u8(byte, vdupq_n_u8((uint8_t) (4 * j)))), shift), three);
}

/// The eight fp16 block scales of the group at b0, converted as _mm256_cvtph_ps does, with the tail left at
/// 0x0000 (= +0.0) exactly as basalt's sc[8] is (:523). vcvt_f32_f16 is VCVTPH2PS for every half, NaN payloads
/// included (case B6 checks all 65,536 against the x86 rule).
inline void group_scales(const uint8_t* row, int b0, int nb, float d[8]) {
    alignas(16) uint16_t sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < nb; ++i) std::memcpy(&sc[i], row + (size_t) (b0 + i) * BB, 2);
    vst1q_f32(d, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(sc))));
    vst1q_f32(d + 4, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(sc + 4))));
}

// ---- THE NaN PATH: x86's NaN rules, applied scalar and lane-exact ------------------------------------------------
//
// Each helper selects the NaN by an explicit bit test, in the operand order of the instruction GCC emits at that
// site, and returns the indefinite for a NaN the operation itself created. No NEON NaN rule is consulted.

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

/// VMULPS, VADDPS, VADDSS and VSUBSS: SRC1's NaN if it has one, else SRC2's, each quieted ([S] Intel SDM Vol. 1
/// Table 4-8; [M] ground-truth:134-136); a NaN the operation CREATES (inf * 0, inf - inf) is the floating-point
/// indefinite 0xffc00000 (4.8.3.7).
inline float x86_src1_first(float src1, float src2, float computed) {
    if (is_nan(src1)) return quiet(src1);
    if (is_nan(src2)) return quiet(src2);
    return is_nan(computed) ? from_bits(0xFFC00000u) : computed;
}
inline float x86_mul(float src1, float src2) { return x86_src1_first(src1, src2, src1 * src2); }
inline float x86_add(float src1, float src2) { return x86_src1_first(src1, src2, src1 + src2); }
inline float x86_sub(float src1, float src2) { return x86_src1_first(src1, src2, src1 - src2); }

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

/// _mm512_reduce_add_ps as GCC 14.3's build emits it for a ONE-token job: SRC1 is the high half at levels 1 and 2,
/// the {2,3,0,1} shuffle at level 3 and lane 0 at level 4 ("hhs0" for both trees, ground-truth:270; level 3 is the
/// shuffle in every slot of every version, :342). Pinning NT = 1 for every token slot is what keeps pool.hpp:54-55.
inline float reduce_add_x86(const float v[16]) {
    float u[8], t[4], w[2];
    for (int j = 0; j < 8; ++j) u[j] = x86_add(v[j + 8], v[j]);
    for (int j = 0; j < 4; ++j) t[j] = x86_add(u[j + 4], u[j]);
    for (int j = 0; j < 2; ++j) w[j] = x86_add(t[j + 2], t[j]);
    return x86_add(w[0], w[1]);
}

/// q2g_row_multi for ONE row and ONE token under the rules above, 16 lanes wide like the AVX-512 original: lane L
/// of the group at b0 takes block b0 + L/2's fp16 scale and the activation's chunk 2*b0 + L, and acc lanes 0-7 and
/// 8-15 take the block's first and second chunk. A single-accumulator loop would get the trees wrong.
/// The masked lanes (L >= 2 * nb) read nothing - they would read scale and hx past the width, which case B5
/// poisons - but still run sites 1 and 2 on the zeros the masked loads leave, as x86's unmasked vmulps and
/// vfmadd231ps do. fma(+0, +0, corr) is corr except where corr is -0.0, and there it is +0.0; corr can be -0.0,
/// because an underflowing product rounds to a signed zero (dd * hx = 2^-24 * -2^-149 = -2^-173 gives -0.0).
float x86_row(const uint8_t* row, const ActQ* a, int nblocks) {
    float acc[16] = {0.0f}, corr[16] = {0.0f};
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        float d[8], p[16];
        group_scales(row, b0, nb, d);                                         // d[nb..7] = +0.0: the masked dd
        for (int L = 0; L < 16; ++L) {
            const bool on = L < 2 * nb;
            p[L] = x86_mul(d[L / 2], on ? a->scale[2 * b0 + L] : 0.0f);       // site 1: vmulps, SRC1 = dd
            corr[L] = x86_fma(d[L / 2], on ? a->hx[2 * b0 + L] : 0.0f, corr[L]);   // site 2: dd, hx, corr
        }
        for (int i = 0; i < nb; ++i) {
            const uint8_t* qs = row + (size_t) (b0 + i) * BB + 2;
            const int8_t* q = a->q + (size_t) (b0 + i) * QK;
            for (int L = 0; L < 16; ++L) {
                int dot = 0;
                for (int k = 0; k < 4; ++k) {
                    const int m = 4 * L + k;
                    dot += ((qs[m / 4] >> (2 * (m % 4))) & 3) * q[m];
                }
                acc[L] = x86_fma(p[2 * i + (L >= 8)], (float) dot, acc[L]);   // site 3: P, D (never a NaN), acc
            }
        }
    }
    return x86_sub(reduce_add_x86(acc), reduce_add_x86(corr));                // sites 4 and 5: vsubss, SRC1 = A
}

/// q2g_row_multi (expert.cpp:503-533) for nt tokens. The port drops basalt's template<int NT> switch, which is a
/// register-allocation device: a numeric result does not depend on NT (acc[t], corr[t] and p[t] are per token),
/// and a NaN one comes from x86_row(), which pins NT = 1 for every token slot (THE NaN PATH).
inline void q2g_row_multi(const uint8_t* row, const ActQ* const* a, int nt, int nblocks, float* res) {
    float32x4_t acc[MAXT][4], corr[MAXT][4];
    for (int t = 0; t < nt; ++t)
        for (int v = 0; v < 4; ++v) acc[t][v] = corr[t][v] = vdupq_n_f32(0.0f);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        float d[8];
        group_scales(row, b0, nb, d);
        // dd = permutexvar(dup, d8) masked by m16 = (1 << 2*nb) - 1: dd[L] = d[L/2] where 2*b0+L is a real chunk,
        // +0.0 beyond - which the zero tail of sc already gives, since L >= 2*nb implies L/2 >= nb.
        float32x4_t dd[4];
        for (int v = 0; v < 4; ++v) {
            const float lane[4] = {d[(4 * v) / 2], d[(4 * v + 1) / 2], d[(4 * v + 2) / 2], d[(4 * v + 3) / 2]};
            dd[v] = vld1q_f32(lane);
        }
        float p[MAXT][8][2];
        for (int t = 0; t < nt; ++t) {
            // corr_L = fma(dd_L, hx_{2*b0+L}, corr_L). The masked lanes read nothing (_mm512_maskz_loadu_ps does
            // not touch them) but still run the FMA, as x86's does, with dd = hx = +0.0: fma(+0, +0, corr) is corr
            // except where corr is -0.0 - an underflowing product leaves one, as in x86_row() - and +0.0 there.
            float hx[16];
            for (int L = 0; L < 16; ++L) hx[L] = L < 2 * nb ? a[t]->hx[2 * b0 + L] : 0.0f;
            for (int v = 0; v < 4; ++v) corr[t][v] = vfmaq_f32(corr[t][v], dd[v], vld1q_f32(hx + 4 * v));
            // p[2i+h] = RN(d_{b0+i} * scale_{2(b0+i)+h}): the lane-wise _mm512_mul_ps of :518, rounded before use.
            // Only the lanes the permute of :528 can select are computed, which is also the whole read set.
            for (int i = 0; i < nb; ++i) {
                p[t][i][0] = d[i] * a[t]->scale[2 * (b0 + i)];
                p[t][i][1] = d[i] * a[t]->scale[2 * (b0 + i) + 1];
            }
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            uint8x16_t w[4];
            unpack64_q2_0(row + (size_t) b * BB + 2, w);
            for (int t = 0; t < nt; ++t) {
                const int8_t* q = a[t]->q + b * QK;
                for (int v = 0; v < 4; ++v) {
                    // Lanes 0-7 (vectors 0-1) carry the block's first activation chunk, 8-15 (vectors 2-3) the
                    // second: the base + 2i permute of :524-528.
                    const int32x4_t dot = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(w[v]), vld1q_s8(q + 16 * v));
                    acc[t][v] = vfmaq_f32(acc[t][v], vdupq_n_f32(p[t][i][v >> 1]), vcvtq_f32_s32(dot));
                }
            }
        }
    }
    for (int t = 0; t < nt; ++t) {
        res[t] = reduce_add_ps(acc[t]) - reduce_add_ps(corr[t]);
        // NaN-ness is the same on both ISAs, so the vector path decides WHETHER this row made a NaN and the
        // scalar replay decides WHICH one (THE NaN PATH).
        if (is_nan(res[t])) res[t] = x86_row(row, a[t], nblocks);
    }
}

}  // namespace

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    if (n <= 0 || n % QKA != 0 || n > H)       // the header's precondition: q[0, n) and the chunk arrays must fit
        throw std::invalid_argument("act_quant_q8_1: n must be a positive multiple of 32, at most 2560");
    a.nchunks = n / QKA;
    const float32x4_t half = vdupq_n_f32(0.5f), mhalf = vdupq_n_f32(-0.5f), zero = vdupq_n_f32(0.0f);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float32x4_t x0[4], x1[4], m[4];
        for (int v = 0; v < 4; ++v) {
            x0[v] = vld1q_f32(xb + 4 * v);
            x1[v] = vld1q_f32(xb + 16 + 4 * v);
            m[v] = maxps(vabsq_f32(x0[v]), vabsq_f32(x1[v]));     // SRC1 = |x0|, as _mm512_max_ps(abs(x0), abs(x1))
        }
        const float amax = reduce_max_ps(m);
        const float s = amax > 0.f ? amax / 127.f : 0.f;          // ordered: a NaN amax gives s = +0.0
        const float inv = s > 0.f ? 1.f / s : 0.f;                // s <= 2^-128 gives +inf; see the contract above
        const float32x4_t vinv = vdupq_n_f32(inv);
        int32x4_t v0[4], v1[4], sv = vdupq_n_s32(0);
        for (int v = 0; v < 4; ++v) {
            // r = t + (t >= 0 ? 0.5f : -0.5f), the ordered compare of :357-358: true for +0 and -0, false for a NaN.
            const float32x4_t t0 = vmulq_f32(x0[v], vinv), t1 = vmulq_f32(x1[v], vinv);
            const float32x4_t r0 = vaddq_f32(t0, vbslq_f32(vcgeq_f32(t0, zero), half, mhalf));
            const float32x4_t r1 = vaddq_f32(t1, vbslq_f32(vcgeq_f32(t1, zero), half, mhalf));
            v0[v] = vminq_s32(vmaxq_s32(cvttps_epi32(r0), lo), hi);
            v1[v] = vminq_s32(vmaxq_s32(cvttps_epi32(r1), lo), hi);
            sv = vaddq_s32(sv, vaddq_s32(v0[v], v1[v]));
        }
        int8_t* q = a.q + k * QKA;
        vst1q_s8(q, vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(v0[0]), vmovn_s32(v0[1]))),
                                vmovn_s16(vcombine_s16(vmovn_s32(v0[2]), vmovn_s32(v0[3])))));
        vst1q_s8(q + 16, vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(v1[0]), vmovn_s32(v1[1]))),
                                     vmovn_s16(vcombine_s16(vmovn_s32(v1[2]), vmovn_s32(v1[3])))));
        const int32_t sum = vaddvq_s32(sv);                       // |sum| <= 4064, so any order is exact
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    // The accumulators below are MAXT wide and the ActQ reads run to 64 * nblocks of an H-element array, so a
    // caller past either bound writes or reads off the end. basalt bounds nt by instantiating q2g_rows<8> for
    // every nt above 8 (expert.cpp:554); here it is a precondition the header states and this holds a C++ caller
    // to. Not an assert: the wheel builds Release, which defines NDEBUG (setup.py:30).
    if (nt < 1 || nt > MAXT || nblocks < 1 || nblocks > H / QK)
        throw std::invalid_argument("q2_0_gguf_rows_multi: need 1 <= nt <= 8 and 1 <= nblocks <= 40");
    float res[MAXT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi(w + (size_t) r * row_bytes, a, nt, nblocks, res);
        for (int t = 0; t < nt; ++t) out[t][r] = res[t];          // the ABSOLUTE row r, not r - r0
    }
}

}  // namespace monolith::basalt_cpu
