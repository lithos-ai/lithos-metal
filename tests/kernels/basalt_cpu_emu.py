# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/expert.cpp:143-148, :336-370, :503-556 and
# include/basalt/kernels/cpu/expert.hpp:25-61 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
# which adapt ggml-org/llama.cpp ggml/src/ggml-common.h (block_q2_0) and ggml/src/ggml-quants.c
# (dequantize_row_q2_0's code rule) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
"""numpy emulation of basalt's two CPU Q2_0 kernels as AVX-512 defines them, for the bit-for-bit tests of the ARM
NEON port (runtime/src/basalt_cpu.cpp).

A sibling of ``basalt_emu`` rather than part of it: that module is basalt's float32 arithmetic *as nvcc compiles it*
and its ``*_ftz`` forms are CUDA's, while these are AVX-512 semantics on a machine whose MXCSR (x86) and FPCR
(arm64) are at their defaults - round to nearest even, subnormals kept, every exception masked. Same primitives
(``F32``, ``fma32``), different source machine; ``h2f`` is this module's own, because a NaN half's conversion is
the instruction's rule and VCVTPH2PS is not what the GPU side converts with.

Every step has an exact numpy counterpart, so the infinity and subnormal-scale cases fall out rather than being
special-cased - but a NaN's *bits* do not: numpy inherits this machine's AArch64 rules, which rank a NaN FMA's
operands the other way round from x86's and write 0x7fc00000 where x86 writes the indefinite 0xffc00000. So every
site that can see or make a NaN goes through the ``x86_*`` helpers below, which select by explicit bit test:

- ``x86_mul``, ``x86_add``, ``x86_sub`` are VMULPS / VADDPS / VSUBSS: SRC1's NaN, else SRC2's, each quieted.
- ``x86_fma`` is VFMADD231PS: first multiplicand, second multiplicand, addend, so a QNaN addend survives an
  ``inf * 0`` product where AArch64's FMLA discards it.
- a NaN any of them CREATES is the floating-point indefinite 0xffc00000.
- ``reduce_add_ps`` is GCC 14.3's four-level tree with the operand order of a ONE-token job, which is what the
  port pins for every token slot (runtime/src/basalt_cpu.cpp, THE NaN PATH).
- MAXPS is an ordered ``a > b`` select at every node of the explicit reduction trees, which gives its NaN rule
  (unordered takes SRC2) and its (+-0, -+0) rule for free; it copies the operand, so payloads survive.
- VCVTTPS2DQ is the range test evaluated *before* any integer cast, so NaN, +-inf and out-of-range values all give
  the integer indefinite 0x80000000, which the clamp then turns into -127.
- ``h2f`` is VCVTPH2PS: exact for every finite half including subnormals, and a NaN half's payload shifts left 13
  with the quiet bit forced on.

``order`` selects the compiler-dependent ``amax``: ``_mm512_reduce_max_ps``'s first two levels pass the high half
as SRC1 under GCC and the low half under Clang, which matters only when a chunk holds a NaN. See
runtime/src/basalt_cpu.cpp's transcription contract.
"""

from __future__ import annotations

import numpy as np

from tests.kernels.basalt_emu import F32, fma32

QK, QKA, BB, H, MAXC, MAXT = 64, 32, 18, 2560, 80, 8
INDEF = -(2 ** 31)                     # the integer indefinite VCVTTPS2DQ returns (Intel SDM Vol. 1 D.4.2.2)
FP_INDEF = np.uint32(0xFFC00000).view(F32)     # the QNaN floating-point indefinite (Intel SDM Vol. 1 4.8.3.7)


def is_nan(x):
    """A NaN by its bits, so an SNaN counts and no float compare is involved."""
    b = np.ascontiguousarray(x, F32).view(np.uint32)
    return ((b & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & ((b & np.uint32(0x7FFFFF)) != 0)


def quiet(x):
    """x86 quieting: bit 22 set, payload and sign kept."""
    return (np.ascontiguousarray(x, F32).view(np.uint32) | np.uint32(0x400000)).view(F32)


def _x86_nan(computed, *operands):
    """``computed`` with x86's NaN selection applied: the first NaN operand in priority order, quieted; otherwise
    the indefinite if the operation created a NaN itself (``inf * 0``, ``inf - inf``), otherwise ``computed``."""
    out = np.where(is_nan(computed), FP_INDEF, np.asarray(computed, F32))
    for src in reversed(operands):                      # reversed, so the highest priority is applied last
        src = np.broadcast_to(np.asarray(src, F32), out.shape)
        out = np.where(is_nan(src), quiet(src), out)
    return np.ascontiguousarray(out, F32)


def x86_mul(src1, src2):
    """VMULPS: SRC1's NaN, else SRC2's ([S] Intel SDM Vol. 1 Table 4-8)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _x86_nan((np.asarray(src1, F32) * np.asarray(src2, F32)).astype(F32), src1, src2)


def x86_add(src1, src2):
    """VADDPS / VADDSS: SRC1's NaN, else SRC2's."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _x86_nan((np.asarray(src1, F32) + np.asarray(src2, F32)).astype(F32), src1, src2)


def x86_sub(src1, src2):
    """VSUBSS: SRC1's NaN, else SRC2's - a subtraction keeps its written order."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _x86_nan((np.asarray(src1, F32) - np.asarray(src2, F32)).astype(F32), src1, src2)


def x86_fma(m1, m2, addend):
    """VFMADD231PS: the first NaN in the order first multiplicand, second multiplicand, addend ([S] Intel SDM
    Vol. 1 Table 14-17), so a QNaN addend outranks an ``inf * 0`` product."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _x86_nan(fma32(m1, m2, addend), m1, m2, addend)


def h2f(bits):
    """VCVTPH2PS of the fp16 block scale. Exact for every finite half, subnormals included; a NaN half keeps its
    sign and payload shifted left 13, with the quiet bit forced on, so an SNaN half is quieted."""
    b = np.ascontiguousarray(bits, np.uint16)
    with np.errstate(invalid="ignore"):
        finite = b.view(np.float16).astype(F32)
    m = (b & np.uint16(0x3FF)).astype(np.uint32)
    nan = ((b & np.uint16(0x7C00)) == np.uint16(0x7C00)) & (m != 0)
    sign = (b >> np.uint16(15)).astype(np.uint32) << np.uint32(31)
    return np.where(nan, (sign | np.uint32(0x7FC00000) | (m << np.uint32(13))).view(F32), finite).astype(F32)


def maxps(src1, src2):
    """MAXPS(SRC1, SRC2) per lane: an ordered ``SRC1 > SRC2`` select, so either operand being NaN, and (+-0, -+0),
    give SRC2."""
    with np.errstate(invalid="ignore"):
        return np.where(src1 > src2, src1, src2).astype(F32)


def reduce_max_ps(v, order="gcc"):
    """``_mm512_reduce_max_ps`` over the last axis (16 lanes). The four-level tree of both compilers' headers; only
    the first two levels' operand order differs (GCC passes the high half as SRC1, Clang the low half). Clang's
    lowering carries ``nnan``, so ``order="clang"`` is the measured realisation of a value its compiler leaves
    undefined for a chunk holding a NaN, not a prediction of what every Clang build gives."""

    def node(hi, lo):
        return maxps(hi, lo) if order == "gcc" else maxps(lo, hi)

    u = node(v[..., 8:], v[..., :8])
    t = node(u[..., 4:], u[..., :4])
    p = maxps(t[..., :2], t[..., 2:])                       # p0 = maxps(t0, t2), p1 = maxps(t1, t3); both agree
    return maxps(p[..., 0], p[..., 1])


def reduce_add_ps(v):
    """``_mm512_reduce_add_ps`` over the last axis: the same four-level tree. The grouping is compiler-independent
    and a numeric result does not depend on the operand order, but a NaN one does, and GCC reassigns that order
    per token slot. This is the order of the real build's ONE-token job, which the port pins for every slot: SRC1
    is the high half at levels 1 and 2, the {2,3,0,1} shuffle at level 3 and lane 0 at level 4."""
    u = x86_add(v[..., 8:], v[..., :8])
    t = x86_add(u[..., 4:], u[..., :4])
    p = x86_add(t[..., 2:], t[..., :2])
    return x86_add(p[..., 0], p[..., 1])


def cvttps_epi32(r):
    """``_mm512_cvttps_epi32``: truncation toward zero, and the integer indefinite for a NaN, an infinity or a value
    outside int32. The comparison is false for a NaN, so the cast never sees one."""
    rr = np.asarray(r, F32).astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(rr) < 2.0 ** 31, np.trunc(rr), float(INDEF)).astype(np.int64)


def act_quant_q8_1(x, order="gcc"):
    """``act_quant_q8_1`` (expert.cpp:336-370, the AVX-512 block). Returns (q, scale, sum, hx, nchunks)."""
    x = np.ascontiguousarray(x, F32)
    nchunks = x.size // QKA
    xb = x.reshape(nchunks, QKA)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        a = np.abs(xb)                                      # _mm512_and_ps(x, 0x7fffffff): clears the sign bit
        amax = reduce_max_ps(maxps(a[:, :16], a[:, 16:]), order)      # SRC1 = |x0|
        # The two divisions are guarded by ordered compares that a NaN fails, so neither can see or make one.
        s = np.where(amax > 0, (amax / F32(127.0)).astype(F32), F32(0)).astype(F32)
        inv = np.where(s > 0, (F32(1.0) / s).astype(F32), F32(0)).astype(F32)
        t = x86_mul(xb, inv[:, None])                       # VMULPS, one rounding
        c = np.where(t >= 0, F32(0.5), F32(-0.5)).astype(F32)         # ordered: true for +0 and -0, false for NaN
        r = x86_add(t, c)                                   # VADDPS, a second rounding
        v = np.clip(cvttps_epi32(r), -127, 127).astype(np.int32)
        total = v.sum(axis=1, dtype=np.int64).astype(np.int32)        # |sum| <= 4064, exact in any order
        hx = x86_mul(s, total.astype(F32))                  # inert: a vcvtsi2ss result is never a NaN
    return v.astype(np.int8).reshape(-1), s, total, hx, nchunks


def unpack64_q2_0(block_codes):
    """The 64 codes of a block in weight order from its 16 code bytes (last axis): code m is
    ``(qs[m // 4] >> (2 * (m % 4))) & 3`` (ggml_quants.c:220-222)."""
    m = np.arange(64)
    return (np.asarray(block_codes, np.uint8)[..., m // 4] >> (2 * (m % 4)).astype(np.uint8)) & np.uint8(3)


def q2_0_rows(w, row_bytes, nblocks, q, scale, hx, rows):
    """``q2g_row_multi`` (expert.cpp:503-533) for one token over the given absolute row indices, vectorized over
    rows. Reads q[:64 * nblocks], scale[:2 * nblocks] and hx[:2 * nblocks] and nothing else."""
    raw = np.ascontiguousarray(w, np.uint8)
    q = np.ascontiguousarray(q, np.int8).astype(np.int32)
    scale = np.ascontiguousarray(scale, F32)
    hx = np.ascontiguousarray(hx, F32)
    base = np.asarray(rows, np.int64) * np.int64(row_bytes)
    nr = base.size
    acc = np.zeros((nr, 16), F32)
    corr = np.zeros((nr, 16), F32)
    lanes = np.arange(16)
    with np.errstate(invalid="ignore", over="ignore"):
        for b0 in range(0, nblocks, 8):
            nb = min(8, nblocks - b0)
            d = np.zeros((nr, 8), F32)                      # basalt's sc[8] keeps its zero tail -> h2f(0) = +0.0
            for i in range(nb):
                off = base + (b0 + i) * BB
                d[:, i] = h2f(raw[off].astype(np.uint16) | (raw[off + 1].astype(np.uint16) << 8))
            dd = d[:, lanes // 2]                           # _mm512_permutexvar_ps(dup, d8), masked beyond 2 * nb
            hxv = np.zeros(16, F32)
            hxv[: 2 * nb] = hx[2 * b0: 2 * b0 + 2 * nb]     # the masked lanes are never read
            corr = x86_fma(dd, hxv[None, :], corr)          # site 2: dd, hx, corr
            for i in range(nb):
                b = b0 + i
                off = base + b * BB + 2
                codes = unpack64_q2_0(raw[off[:, None] + np.arange(16)[None, :]]).astype(np.int32)
                dot = (codes.reshape(nr, 16, 4) * q[b * QK: (b + 1) * QK].reshape(1, 16, 4)).sum(-1)
                p0 = x86_mul(d[:, i], scale[2 * b])         # site 1: RN(d_b * s_{2b}), SRC1 = d, rounded before use
                p1 = x86_mul(d[:, i], scale[2 * b + 1])
                p = np.where(lanes[None, :] < 8, p0[:, None], p1[:, None]).astype(F32)
                acc = x86_fma(p, dot.astype(F32), acc)      # site 3: P, D (never a NaN), acc
        return x86_sub(reduce_add_ps(acc), reduce_add_ps(corr))       # sites 4 and 5: SRC1 = the acc sum


def q2_0_gguf_rows_multi(w, row_bytes, nblocks, acts, out, r0, r1):
    """``q2_0_gguf_rows_multi`` (expert.cpp:544-556): writes out[t][r] for r0 <= r < r1, at the ABSOLUTE row r, and
    leaves every other element of ``out`` alone. ``acts`` is a sequence of (q, scale, hx)."""
    if r1 <= r0:
        return out
    rows = np.arange(r0, r1)
    for t, (q, scale, hx) in enumerate(acts):
        out[t, rows] = q2_0_rows(w, row_bytes, nblocks, q, scale, hx, rows)
    return out
