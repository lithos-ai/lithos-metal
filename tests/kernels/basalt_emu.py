# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu, src/kernels/cuda/iq_kernels.cu,
# src/kernels/cuda/native_bf16.cu, src/kernels/cuda/dense_tc.cu, src/kernels/cuda/q5k_head.cu,
# src/kernels/cuda/head_proxy.cuh, src/kernels/cuda/quantize_act.cu, src/kernels/cuda/s2_expert_grouped.cu,
# include/basalt/kernels/q8_1_finite.hpp and include/basalt/kernels/f16_bits.hpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36, which adapt ggml-org/llama.cpp ggml/src/ggml-common.h (kvalues_iq4nl),
# ggml/src/ggml-quants.c (get_scale_min_k4), ggml/src/ggml-cuda/quantize.cu and ggml/src/ggml-cuda/mmvf.cu @
# 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
"""numpy emulation of basalt's float32 arithmetic as nvcc compiles it (docs/research/basalt-nvcc/matvec.md), for the
bit-for-bit tests of the Metal port. Every float operation rounds once to float32, to nearest even on the float32
grid, subnormals included (gradual underflow): numpy's float32 add, multiply and divide do; ``fma32`` does by rounding
the exact float64 sum to odd before the float32 cast. The ``*_ftz`` forms flush subnormal inputs, then flush the
rounded result if it is subnormal, as PTX defines ``.ftz`` (``fma.f32``: the exact value "rounded to single
precision"; ``.ftz`` flushes "subnormal inputs and results"): an exact result in [2^-126 - 2^-150, 2^-126) gives
2^-126, the tie 2^-126 - 2^-150 included (2^-126 is the even neighbour)."""

import numpy as np

F32 = np.float32


def ftz(x):
    """A subnormal becomes the zero of its sign; every other value keeps its bits."""
    x = np.asarray(x, F32)
    b = x.view(np.uint32)
    return np.where((b & 0x7F800000) == 0, (b & 0x80000000).view(F32), x).astype(F32)


def fma32(a, b, c):
    """a * b + c rounded once to float32 (to nearest even, on the grid with subnormals). a * b is exact in float64;
    the float64 sum is rounded to odd (TwoSum gives its exact error), which a float32 cast then rounds correctly
    (53 >= 24 + 2 bits)."""
    a, b, c = (np.asarray(v, F32).astype(np.float64) for v in (a, b, c))
    s = a * b
    t = s + c
    bp = t - s
    err = (s - (t - bp)) + (c - bp)
    even = (t.view(np.int64) & 1) == 0
    bump = (err != 0) & even & np.isfinite(t)
    t = np.where(bump, np.nextafter(t, np.where(err > 0, np.inf, -np.inf)), t)
    with np.errstate(over="ignore"):
        return t.astype(F32)


def add_ftz(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) + ftz(b))


def mul_ftz(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) * ftz(b))


def fma_ftz(a, b, c):
    """fma.rn.ftz.f32: rounded first, then flushed, so 2^-126 - 2^-151 exactly gives 2^-126 (module docstring)."""
    return ftz(fma32(ftz(a), ftz(b), ftz(c)))


def max_ftz(a, b):
    return np.fmax(ftz(a), ftz(b))


def rcp_ftz(x):
    """The correctly rounded reciprocal the port writes where SASS has MUFU.RCP."""
    with np.errstate(divide="ignore", over="ignore"):
        return ftz(F32(1.0) / ftz(x))


def roundf(x):
    """CUDA's roundf: half away from zero (trunc of |x| + 0.5 rounded toward zero, which is exact here)."""
    x = np.asarray(x, F32).astype(np.float64)
    return np.trunc(x + np.copysign(0.5, x)).astype(F32)


def half_bits(x):
    """float32 to fp16 bits, round to nearest even (CUDA's __float2half, Metal's half())."""
    with np.errstate(over="ignore"):
        return np.asarray(x, F32).astype(np.float16).view(np.uint16)


def h2f(bits):
    with np.errstate(invalid="ignore"):                      # NaN patterns stay NaN
        return np.asarray(bits, np.uint16).view(np.float16).astype(F32)


# ---- the q8_1 quantizers (matvec.md sections 1 and 3) -------------------------------------------------------------

RCP127 = np.uint32(0x3C010204).view(F32)      # fl(1/127): ptxas's multiply for the native amax / 127


def butterfly(v, op):
    """basalt's XOR butterfly over the 32 lanes of each row of ``v`` (offsets 16, 8, 4, 2, 1); every lane ends equal."""
    for o in (16, 8, 4, 2, 1):
        v = op(v, v[:, np.arange(32) ^ o])
    return v[:, 0]


def q8_1_finite(v):
    v = np.asarray(v, F32)
    return np.where(np.abs(v) > F32(65504.0), np.copysign(F32(65504.0), v), v).astype(F32)


def q8_1_codes(q, amax):
    with np.errstate(invalid="ignore"):
        c = np.where(np.isnan(q), 0, np.clip(q, -127, 127)).astype(np.int8)
    return np.where((amax == 0)[:, None], np.int8(0), c)


def q8_1_pack(d, s, codes):
    out = np.zeros((len(d), 36), np.uint8)
    out[:, 0:2] = half_bits(q8_1_finite(d)).view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = half_bits(q8_1_finite(s)).view(np.uint8).reshape(-1, 2)
    out[:, 4:] = codes.view(np.uint8)
    return out.reshape(-1)


def q8_1_native(x):
    """native_quantize_q8_1 as nvcc compiles it, with the correctly rounded reciprocal in place of MUFU.RCP."""
    v = np.asarray(x, F32).reshape(-1, 32)
    amax = butterfly(ftz(np.abs(v)), max_ftz)
    s = butterfly(v, add_ftz)
    d = q8_1_finite(mul_ftz(amax, RCP127))
    with np.errstate(invalid="ignore"):
        q = roundf(mul_ftz(v, rcp_ftz(d)[:, None]))
    return q8_1_pack(d, s, q8_1_codes(q, amax))


def q8_1_ieee(x):
    """quantize_q8_1_kernel (IEEE: correctly rounded divisions, subnormals kept), the host formula numpy computes."""
    v = np.asarray(x, F32).reshape(-1, 32)
    amax = np.fmax.reduce(np.abs(v), axis=1)                  # fmaxf drops a NaN operand
    s = butterfly(v, np.add)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        d = q8_1_finite(amax / F32(127.0))
        q = roundf(v / d[:, None])
    return q8_1_pack(d, s, q8_1_codes(q, amax))


# ---- native_mmvq (matvec.md sections 4-5) -------------------------------------------------------------------------
# The integer parts are computed from the decoded codes (each value's code, sub-block scale and min as ggml defines
# them), independently of basalt's byte tricks; the float parts follow the sequence nvcc compiles, per thread.

TRAITS = {2: (32, 2, 64, 18), 6: (32, 2, 64, 22), 8: (32, 4, 32, 34), 20: (32, 2, 64, 18), 11: (256, 16, 8, 110),
          12: (256, 16, 8, 144), 13: (256, 16, 8, 176), 14: (256, 32, 4, 210), 23: (256, 8, 16, 136), 42: (64, 2, 64, 18)}
KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int64)


def kqs_of(type_id, tid):
    return {2: 2 * (tid % 2), 6: 2 * (tid % 2), 20: 2 * (tid % 2), 8: 2 * (tid % 4), 11: tid % 16, 12: 2 * (tid % 16),
            13: 2 * (tid % 16), 14: tid % 32, 23: 4 * (tid % 8), 42: tid % 2}[type_id]


def _h(b, off):
    return h2f(b[..., off].astype(np.uint16) | (b[..., off + 1].astype(np.uint16) << 8))


def _scale_min_k4(sc, sb):
    """get_scale_min_k4 on the 12 scale bytes ``sc`` (..., 12) for sub-block indices ``sb`` (broadcast)."""
    g = lambda k: np.take_along_axis(sc, np.broadcast_to(k, sc.shape[:-1] + k.shape[-1:]), axis=-1).astype(np.int64)
    lo, hi = sb < 4, np.clip(sb, 4, 7)
    s = np.where(lo, g(np.clip(sb, 0, 3)) & 63, (g(hi + 4) & 0xF) | ((g(hi - 4) >> 6) << 4))
    m = np.where(lo, g(np.clip(sb, 0, 3) + 4) & 63, (g(hi + 4) >> 4) | ((g(hi) >> 6) << 4))
    return s, m


def _dot(codes, a):
    return (codes[None] * a).sum(axis=-1)


def _contrib(type_id, tmp, blk, kby, kqs, d8, s8, a):
    """One block step for every (column, row, thread): ``blk`` (rows, nt, bytes), ``kby`` and ``kqs`` (nt,), the q8_1
    scales d8 / sums s8 (cols, blocks) and codes a (cols, blocks, 32) of the activations."""
    nt = kqs.shape[0]
    t = np.arange(nt)
    xa = lambda b, pos: a[:, b[:, None], pos][:, None]                     # (cols, 1, nt, npos)
    if type_id in (2, 6, 20, 8):
        if type_id == 8:
            pos = 4 * kqs[:, None] + np.arange(8)
            codes = blk[..., 2:34].view(np.int8).astype(np.int64)
        else:
            pos = np.concatenate([4 * kqs[:, None] + np.arange(8), 16 + 4 * kqs[:, None] + np.arange(8)], axis=1)
            qs = blk[..., 6:22] if type_id == 6 else blk[..., 2:18]
            nib = np.concatenate([qs & 15, qs >> 4], axis=-1).astype(np.int64)
            if type_id == 6:
                qh = blk[..., 2:6].astype(np.int64)
                qh = qh[..., 0] | (qh[..., 1] << 8) | (qh[..., 2] << 16) | (qh[..., 3] << 24)
                nib = nib | (((qh[..., None] >> np.arange(32)) & 1) << 4)
            codes = KVALUES[nib] if type_id == 20 else nib
        c = np.take_along_axis(codes, np.broadcast_to(pos, codes.shape[:-1] + pos.shape[-1:]), axis=-1)
        sumi = _dot(c, xa(kby, pos)).astype(F32)
        d, dx, sx = _h(blk, 0)[None], d8[:, kby][:, None], s8[:, kby][:, None]
        if type_id in (8, 20):
            return fma_ftz(mul_ftz(d, dx), sumi, tmp)
        k = F32(-4.0) if type_id == 2 else F32(-8.0)
        return fma_ftz(d, fma_ftz(sumi, dx, mul_ftz(sx, k)), tmp)
    if type_id == 42:
        qs = blk[..., 2:18].astype(np.int64)
        codes = ((qs[..., np.arange(64) // 4] >> (2 * (np.arange(64) % 4))) & 3) - 1
        pos = 32 * kqs[:, None] + np.arange(32)
        c = np.take_along_axis(codes, np.broadcast_to(pos, codes.shape[:-1] + pos.shape[-1:]), axis=-1)
        b = kby + kqs
        sumi = _dot(c, xa(b, np.broadcast_to(np.arange(32), (nt, 32)))).astype(F32)
        return fma_ftz(mul_ftz(_h(blk, 0)[None], d8[:, b][:, None]), sumi, tmp)
    if type_id == 23:
        ib = kqs // 4
        qs = blk[..., 8:136].astype(np.int64)
        idx = 16 * ib[:, None] + np.arange(32) % 16
        nib = (np.take_along_axis(qs, np.broadcast_to(idx, qs.shape[:-1] + idx.shape[-1:]), axis=-1)
               >> (4 * (np.arange(32) // 16))) & 15
        b = kby + ib
        sumi = _dot(KVALUES[nib], xa(b, np.broadcast_to(np.arange(32), (nt, 32))))
        sl = np.take_along_axis(blk[..., 4:8].astype(np.int64), np.broadcast_to((ib // 2)[None, :, None], blk.shape[:-1] + (1,)), axis=-1)[..., 0]
        sh = blk[..., 2].astype(np.int64) | (blk[..., 3].astype(np.int64) << 8)
        ls = ((sl >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
        sumi = (sumi * (ls - 32)[None]).astype(F32)
        return fma_ftz(mul_ftz(_h(blk, 0)[None], d8[:, b][:, None]), sumi, tmp)
    if type_id in (12, 13):
        t16 = kqs // 2
        ch, mm = t16 // 4, t16 % 4
        pos = np.concatenate([4 * mm[:, None] + np.arange(4), 16 + 4 * mm[:, None] + np.arange(4)], axis=1)  # (nt, 8)
        qs0 = 48 if type_id == 13 else 16
        sc = blk[..., 4:16]
        sumf_d = sumf_m = np.zeros(tmp.shape, F32)
        for i in range(2):
            sb = 2 * ch + i
            qidx = qs0 + 32 * ch[:, None] + pos
            q = np.take_along_axis(blk, np.broadcast_to(qidx, blk.shape[:-1] + qidx.shape[-1:]), axis=-1).astype(np.int64)
            code = (q >> (4 * i)) & 15
            if type_id == 13:
                h = np.take_along_axis(blk, np.broadcast_to(16 + pos, blk.shape[:-1] + pos.shape[-1:]), axis=-1).astype(np.int64)
                code = code | (((h >> sb[:, None]) & 1) << 4)
            b = kby + sb
            av = xa(b, pos)
            dot1, dot2 = _dot(code, av), av.sum(axis=-1)
            s, m = _scale_min_k4(sc, np.broadcast_to(sb[None, :, None], sc.shape[:-1] + (1,)))
            dx = d8[:, b][:, None]
            sumf_d = fma_ftz(dx, (dot1 * s[None, ..., 0]).astype(F32), sumf_d)
            sumf_m = fma_ftz(dx, (dot2 * m[None, ..., 0]).astype(F32), sumf_m)
        return add_ftz(tmp, fma_ftz(_h(blk, 0)[None], sumf_d, -mul_ftz(_h(blk, 2)[None], sumf_m)))
    if type_id == 11:
        iqs = kqs
        h, pos = iqs // 8, 4 * (iqs % 8)[:, None] + np.arange(4)
        sumf = np.zeros(tmp.shape, F32)
        sc = blk[..., 96:108].astype(np.int64)
        for i in range(4):
            q = np.take_along_axis(blk, np.broadcast_to(32 + 32 * h[:, None] + pos, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            hm = np.take_along_axis(blk, np.broadcast_to(pos, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            code = ((q >> (2 * i)) & 3) - np.where((hm >> (4 * h[:, None] + i)) & 1, 0, 4)
            isc = 8 * h + 2 * i + (iqs % 8) // 4
            g = lambda k: np.take_along_axis(sc, np.broadcast_to(k[None, :, None], sc.shape[:-1] + (1,)), axis=-1)[..., 0]
            s = (((g(isc % 8) >> (4 * (isc // 8))) & 15) | (((g(8 + isc % 4) >> (2 * (isc // 4))) & 3) << 4)) - 32
            b = kby + 4 * h + i
            sumf = fma_ftz(d8[:, b][:, None], (_dot(code, xa(b, pos)) * s[None]).astype(F32), sumf)
        return fma_ftz(_h(blk, 108)[None], sumf, tmp)
    if type_id == 14:
        iqs = kqs
        h, l = iqs // 16, 4 * (iqs % 16)[:, None] + np.arange(4)
        sumf = np.zeros(tmp.shape, F32)
        for i in range(2):
            ql = np.take_along_axis(blk, np.broadcast_to(64 * h[:, None] + l, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            qh = np.take_along_axis(blk, np.broadcast_to(128 + 32 * h[:, None] + l % 32, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            code = (((ql >> (4 * i)) & 15) | (((qh >> (2 * (l // 32) + 4 * i)) & 3) << 4)) - 32
            sidx = 192 + 8 * h + 4 * i + (iqs % 16) // 4
            s = np.take_along_axis(blk, np.broadcast_to(sidx[None, :, None], blk.shape[:-1] + (1,)), axis=-1)[..., 0].view(np.int8).astype(np.int64)
            b = kby + 4 * h + 2 * i + (iqs % 16) // 8
            sumf = fma_ftz(d8[:, b][:, None], (_dot(code, xa(b, l % 32)) * s[None]).astype(F32), sumf)
        return fma_ftz(_h(blk, 208)[None], sumf, tmp)
    raise ValueError(type_id)


def mmvq(type_id, w, xq, n_in, n_out, ncols, nw=4, row_chunk=64):
    """native_mmvq's output (ncols, n_out) for GGUF rows ``w`` (n_out, row bytes) and q8_1 bytes ``xq``, in basalt's
    thread mapping with ``nw`` SIMD-groups (4: the one-column and exact layouts; 2: the upstream layout above 4
    columns): per thread the blocks kbx = tid / T + k * BPI', then SIMD-group order and the XOR butterfly."""
    div, T, bpi, bb = TRAITS[type_id]
    nb = n_in // div
    blocks = np.asarray(w, np.uint8).reshape(n_out, nb, bb)
    xb = np.asarray(xq, np.uint8).reshape(ncols, n_in // 32, 36)
    d8, s8, a = _h(xb, 0), _h(xb, 2), xb[..., 4:].view(np.int8).astype(np.int64)
    nt, bpi_l = 32 * nw, bpi * nw // 4
    tid = np.arange(nt)
    kqs, kby_per_block = kqs_of(type_id, tid), div // 32
    out = np.zeros((ncols, n_out), F32)
    for r0 in range(0, n_out, row_chunk):
        rows = blocks[r0: r0 + row_chunk]
        tmp = np.zeros((ncols, len(rows), nt), F32)
        for k in range(-(-nb // bpi_l) + 1):
            kbx = tid // T + k * bpi_l
            valid = kbx < nb
            if not valid.any():
                break
            kb = np.minimum(kbx, nb - 1)
            new = _contrib(type_id, tmp, rows[:, kb], kb * kby_per_block, kqs, d8, s8, a)
            tmp = np.where(valid[None, None], new, tmp)
        t = tmp[..., :32]
        for sgi in range(1, nw):
            t = add_ftz(t, tmp[..., 32 * sgi: 32 * sgi + 32])
        out[:, r0: r0 + row_chunk] = butterfly(t.reshape(-1, 32), add_ftz).reshape(ncols, len(rows))
    return out


# ---- the routed experts (docs/research/basalt-nvcc/experts.md) ----------------------------------------------------
# iq_kernels.cu builds with nvcc's IEEE defaults: no flush; numpy's float32 operations keep subnormals as CUDA does.

def q2_0_dots(rows, n, xq):
    """Q2_0 rows (R rows of ``n`` values, GGUF blocks) against q8_1 rows ``xq`` (E rows of n / 32 blocks): (E, R), in
    the order basalt's row_dot_multi sums (experts.md sections 1 and 3): call k (Q2_0 block k / 2, half k % 2, q8_1
    block k) on lane k % 32, s = fma(fl(d2 * d8), f(sumi), s) over the lane's calls in ascending k from +0, then the
    32-lane butterfly of plain adds. gu2's rows2 and the down kernel's 16-lane tree give the same bits."""
    rows = np.asarray(rows, np.uint8).reshape(-1, n // 64, 18)
    nr, nk = rows.shape[0], n // 32
    qs = rows[..., 2:18].astype(np.int64)
    codes = ((qs[..., np.arange(64) // 4] >> (2 * (np.arange(64) % 4))) & 3) - 1     # (R, n / 64, 64): code - 1
    xb = np.asarray(xq, np.uint8).reshape(-1, nk, 36)
    a = xb[..., 4:].view(np.int8).astype(np.float64)                                   # (E, K, 32)
    # exact integer dots (|sumi| <= 32 * 2 * 127), per call: (K, R, 32) @ (K, 32, E)
    sumi = np.matmul(codes.reshape(nr, nk, 32).transpose(1, 0, 2).astype(np.float64), a.transpose(1, 2, 0))
    d = _h(rows, 0)[:, np.arange(nk) // 2][None] * _h(xb, 0)[:, None, :]                # (E, R, K): exact products
    f = sumi.transpose(2, 1, 0).astype(F32)
    s = np.zeros(d.shape[:2] + (32,), F32)
    for k0 in range(0, nk, 32):
        ks = np.arange(k0, min(k0 + 32, nk))
        s[..., ks - k0] = fma32(d[..., ks], f[..., ks], s[..., ks - k0])
    for o in (16, 8, 4, 2, 1):
        s = s + s[..., np.arange(32) ^ o]
    return s[..., 0]


def swiglu_q8_1(gate, up, e):
    """The experts' SwiGLU + q8_1 (experts.md section 2): t = fl(1 + e), s = fl(g / t) correctly rounded,
    h = fl(s * up) with IEEE subnormals, then the IEEE quantizer. ``e`` is __expf(-g), taken from the GPU's expf_neg
    (MUFU.EX2 and fast::exp2 have no bit-exact host model)."""
    g, up, e = (np.asarray(v, F32) for v in (gate, up, e))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        h = (g / (F32(1.0) + e)) * up
    return q8_1_ieee(h)


# ---- the drafter's experts (docs/research/basalt-nvcc/s2_experts.md) ----------------------------------------------
# quantize_act.cu and s2_expert_grouped.cu build with nvcc's IEEE defaults: no flush, as numpy's float32 operations.

def f16_from_f32(x):
    """f16_bits.hpp:36-65's f16_from_f32, transcribed in integer operations: round to nearest even, a finite overflow
    to infinity, fp16 subnormals kept, a NaN to sign | 0x7E00."""
    xb = np.asarray(x, F32).view(np.uint32).astype(np.int64)
    sign, rawexp, man = (xb >> 16) & 0x8000, (xb >> 23) & 0xFF, xb & 0x7FFFFF
    exp = rawexp - 127 + 15
    sh = np.clip(14 - exp, 1, 24)
    mm = man | 0x800000
    h_sub = (mm >> sh) & 0x3FF
    rem, half = mm & ((1 << sh) - 1), 1 << (sh - 1)
    h_sub = h_sub + ((rem > half) | ((rem == half) & ((h_sub & 1) == 1)))
    h_norm = sign | (np.clip(exp, 0, 31) << 10) | (man >> 13)
    rem_n = man & 0x1FFF
    h_norm = h_norm + ((rem_n > 0x1000) | ((rem_n == 0x1000) & ((h_norm & 1) == 1)))
    out = np.where(exp <= 0, np.where(exp < -10, sign, sign | h_sub), h_norm)
    out = np.where(exp >= 31, sign | 0x7C00, out)
    out = np.where(rawexp == 0xFF, sign | 0x7C00 | np.where(man != 0, 0x200, 0), out)
    return out.astype(np.uint16)


def q8_0_scaled(x):
    """quantize_q8_0_scaled_warp_kernel (s2_experts.md section 1), the host transcription: per 32-value block amax =
    fmaxf over |x| (a NaN dropped), s = amax > 0 ? amax / 127 : 0, inv = s > 0 ? 1 / s : 0, t = x * inv,
    r = t + (t >= 0 ? 0.5 : -0.5), code (int) r clamped to +-127 (NaN to 0); d = f16_from_f32(s). Returns (34-byte
    blocks as bytes, fp32 scales)."""
    v = np.asarray(x, F32).reshape(-1, 32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        amax = np.fmax.reduce(np.abs(v), axis=1)
        s = np.where(amax > 0, amax / F32(127.0), F32(0.0)).astype(F32)
        inv = np.where(s > 0, F32(1.0) / s, F32(0.0)).astype(F32)
        t = v * inv[:, None]
        r = t + np.where(t >= 0, F32(0.5), F32(-0.5))
        code = np.where(np.isnan(r), 0, np.trunc(np.clip(r, -127, 127))).astype(np.int8)
    out = np.zeros((len(v), 34), np.uint8)
    out[:, :2] = f16_from_f32(s).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = code.view(np.uint8)
    return out.reshape(-1), s


def s2_dots(codes, scales, xq, xs):
    """Q2_0 rows of the drafter's blob against Q8_0 activation rows with fp32 scales, as gu_grouped_t_kernel and
    down_grouped_t_kernel sum them (s2_experts.md sections 2 and 5): (E, R) for code rows ``codes`` (R, n / 4 bytes),
    their fp16 scale bits ``scales`` (R, n / 64), the activation rows ``xq`` (E, n / 32 blocks of 34 bytes) and their
    scales ``xs`` (E, n / 32). Per (row, entry) lane L takes the chunks L, L + 32, ... in order, acc = fma(fl(dw * dx),
    f(s - hx), acc) from +0, s - hx the exact sum of (code - 1) * x; then the 32-lane butterfly of IEEE adds."""
    codes = np.asarray(codes, np.uint8)
    nr, nk = codes.shape[0], codes.shape[1] // 8
    w = ((codes[:, np.arange(32 * nk) // 4].astype(np.int64) >> (2 * (np.arange(32 * nk) % 4))) & 3) - 1
    xb = np.asarray(xq, np.uint8).reshape(-1, nk, 34)
    a = xb[..., 2:].view(np.int8).astype(np.float64)                                   # (E, K, 32)
    n = np.einsum("rkv,ekv->erk", w.reshape(nr, nk, 32).astype(np.float64), a).astype(F32)   # exact integers
    dw = h2f(np.asarray(scales, np.uint16))[:, np.arange(nk) // 2]                      # (R, K)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = dw[None] * np.asarray(xs, F32).reshape(-1, nk)[:, None, :]                  # (E, R, K): fl(dw * dx)
    acc = np.zeros(n.shape[:2] + (32,), F32)
    for k0 in range(0, nk, 32):
        ks = np.arange(k0, min(k0 + 32, nk))
        acc[..., ks - k0] = fma32(p[..., ks], n[..., ks], acc[..., ks - k0])
    with np.errstate(over="ignore", invalid="ignore"):
        for o in (16, 8, 4, 2, 1):
            acc = acc + acc[..., np.arange(32) ^ o]
    return acc[..., 0]


def s2_swiglu(g, u, e):
    """swiglu_kernel (s2_experts.md section 3): fl(fl(g / fl(1 + e)) * u), IEEE; ``e`` is __expf(-g), taken from the
    GPU's expf_neg (MUFU.EX2 and fast::exp2 have no bit-exact host model)."""
    g, u, e = (np.asarray(v, F32) for v in (g, u, e))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        return (g / (e + F32(1.0))) * u


# ---- the BF16 GEMVs on fp32 activations (docs/research/basalt-nvcc/bf16.md) -----------------------------------------
# native_bf16.cu builds with --use_fast_math: every FMA and add is .FTZ.

def bf16_bits(x):
    """float32 to bf16 bits, round to nearest even (bf16_bits.hpp's bf16_from_f32 for non-NaN values)."""
    b = np.asarray(x, F32).view(np.uint32).astype(np.uint64)
    return ((b + ((b >> 16) & 1) + 0x7FFF) >> 16).astype(np.uint16)


def mmvf(w, x, block, flush=True):
    """bf16_gemv_fp32_mmvf's outputs (n_tok, n_out) for BF16 bits ``w`` (n_out, n_in) and float32 rows ``x``
    (n_tok, n_in) with ``block`` threads (basalt's mmvf_block_size), as bf16.md records it: thread t takes the pairs
    t, t + block, ... in order, acc = fma(bf(w[2p]), x[2p], acc) then fma(bf(w[2p + 1]), x[2p + 1], acc) from +0,
    all .FTZ; the XOR butterfly per warp; for block > 32 the butterfly again over 32 slots, the warp sums and +0s.
    ``flush=False``: the same sequence in IEEE float32 (``fma32`` and plain adds), what the kernel would give without
    .FTZ."""
    fma, add = (fma_ftz, add_ftz) if flush else (fma32, np.add)
    w = np.asarray(w, np.uint16)
    n_out, n_in = w.shape
    x = np.asarray(x, F32).reshape(-1, n_in)
    wf = (w.astype(np.uint32) << 16).view(F32)
    acc = np.zeros((x.shape[0], n_out, block), F32)
    pairs = n_in // 2
    for p0 in range(0, pairs, block):
        p = np.arange(p0, min(p0 + block, pairs))
        a = fma(wf[None, :, 2 * p], x[:, None, 2 * p], acc[..., p - p0])
        acc[..., p - p0] = fma(wf[None, :, 2 * p + 1], x[:, None, 2 * p + 1], a)
    s = butterfly(acc.reshape(-1, 32), add).reshape(x.shape[0], n_out, block // 32)
    if block > 32:
        slots = np.zeros(s.shape[:2] + (32,), F32)
        slots[..., : block // 32] = s
        s = butterfly(slots.reshape(-1, 32), add).reshape(x.shape[0], n_out, 1)
    return s[..., 0]


# ---- dense_tc and dense_tc_pair (docs/research/basalt-nvcc/dense_tc.md) ----------------------------------------------
# dense_tc.cu builds with nvcc's IEEE defaults: no flush. The integer dots come from the decoded codes (each value's
# code and its block's scales as ggml defines the format), independently of basalt's byte tricks; the float steps
# follow dense_tc.md section 1 per (row, column) and section 2's warp chunks and trees.

DTC_SB = {11: 110, 12: 144, 13: 176, 14: 210, 23: 136, 42: 72}
DTC_WARPS = {"one": 24, "tile": 8, "pair": 8, "s1": 4, "s2": 8}


def _u16(b, off):
    return b[..., off].astype(np.int64) | (b[..., off + 1].astype(np.int64) << 8)


def dtc_decode(type_id, w, n_in, n_out):
    """The codes (n_out, n_in / 32, 32) of every 32-value block and its scale terms, as ggml's formats define them:
    Q4_K / Q5_K {"sc", "m"} per block and {"d", "dmin"} per super-block; Q6_K / Q3_K {"s16"} per 16 values and {"d"}
    per super-block; IQ4_XS {"ls"} (the scale minus 32) per block and {"d"} per super-block; Q2_0 {"d"} per 64-value
    block."""
    nk = n_in // 32
    if type_id == 42:
        blk = np.asarray(w, np.uint8).reshape(n_out, n_in // 64, 18)
        v = np.arange(64)
        codes = ((blk[..., 2 + v // 4].astype(np.int64) >> (2 * (v % 4))) & 3) - 1
        return codes.reshape(n_out, nk, 32), {"d": _h(blk, 0)}
    blk = np.asarray(w, np.uint8).reshape(n_out, n_in // 256, DTC_SB[type_id])
    v = np.arange(256)
    if type_id in (12, 13):
        qs = blk[..., 48:176] if type_id == 13 else blk[..., 16:144]
        q = (qs[..., 32 * (v // 64) + v % 32].astype(np.int64) >> (4 * ((v // 32) % 2))) & 15
        if type_id == 13:
            q = q | (((blk[..., 16 + v % 32].astype(np.int64) >> (v // 32)) & 1) << 4)
        sc = np.broadcast_to(blk[..., None, 4:16], blk.shape[:-1] + (8, 12))
        s, m = _scale_min_k4(sc, np.arange(8)[:, None])
        terms = {"sc": s[..., 0].reshape(n_out, nk), "m": m[..., 0].reshape(n_out, nk), "d": _h(blk, 0),
                 "dmin": _h(blk, 2)}
    elif type_id == 14:
        hh, l = v // 128, v % 128
        ql = (blk[..., 64 * hh + l % 64].astype(np.int64) >> (4 * (l // 64))) & 15
        qh = (blk[..., 128 + 32 * hh + l % 32].astype(np.int64) >> (2 * (l // 32))) & 3
        q = (ql | (qh << 4)) - 32
        terms = {"s16": blk[..., 192:208].view(np.int8).astype(np.int64).reshape(n_out, 2 * nk), "d": _h(blk, 208)}
    elif type_id == 11:
        hh, j, pos = v // 128, (v % 128) // 32, v % 32
        q = (blk[..., 32 + 32 * hh + pos].astype(np.int64) >> (2 * j)) & 3
        hm = (blk[..., pos].astype(np.int64) >> (4 * hh + j)) & 1
        q = q - np.where(hm == 1, 0, 4)
        sc = blk[..., 96:108].astype(np.int64)
        isc = np.arange(16)
        s16 = (((sc[..., isc % 8] >> (4 * (isc // 8))) & 15) | (((sc[..., 8 + isc % 4] >> (2 * (isc // 4))) & 3) << 4)) - 32
        terms = {"s16": s16.reshape(n_out, 2 * nk), "d": _h(blk, 108)}
    elif type_id == 23:
        b, pos = v // 32, v % 32
        q = KVALUES[(blk[..., 8 + 16 * b + pos % 16].astype(np.int64) >> (4 * (pos // 16))) & 15]
        ib = np.arange(8)
        sl = (blk[..., 4 + ib // 2].astype(np.int64) >> (4 * (ib % 2))) & 15
        sh = (_u16(blk, 2)[..., None] >> (2 * ib)) & 3
        terms = {"ls": ((sl | (sh << 4)) - 32).reshape(n_out, nk), "d": _h(blk, 0)}
    else:
        raise ValueError(type_id)
    return q.reshape(n_out, nk, 32), terms


def dense_tc(type_id, w, xq, n_in, n_out, ncols, order):
    """dense_tc's output (ncols, n_out) in summation order ``order`` ("one", "tile", "pair", "s1", "s2") for GGUF rows
    ``w`` and q8_1 bytes ``xq`` (at least ncols columns)."""
    codes, terms = dtc_decode(type_id, w, n_in, n_out)
    nk, nchunk = n_in // 32, n_in // 512
    xb = np.asarray(xq, np.uint8).reshape(-1, nk, 36)[:ncols]
    d8, a = _h(xb, 0), xb[..., 4:].view(np.int8).astype(np.int64)                     # (C, K), (C, K, 32)
    if type_id in (14, 11):                                                           # the 16-value halves' dots
        dots = [np.einsum("rkv,ckv->rck", codes[..., 16 * e: 16 * e + 16], a[..., 16 * e: 16 * e + 16]) for e in (0, 1)]
    else:
        dot = np.einsum("rkv,ckv->rck", codes, a)
    isum = a.sum(axis=-1)                                                             # (C, K): the code sums
    col = lambda v: np.asarray(v, F32)[:, None]                                      # a row term over the columns
    f32 = lambda n: np.asarray(n).astype(F32)
    zero = np.zeros((n_out, ncols), F32)

    def half_sb(acc, c, hb):
        sb, ks = 2 * c + (hb >> 1), [16 * c + 4 * hb + i for i in range(4)]
        if type_id in (12, 13):
            for k in ks:
                f1 = terms["d"][:, sb] * f32(terms["sc"][:, k])
                f2 = terms["dmin"][:, sb] * f32(terms["m"][:, k])
                inner = fma32(col(f1), f32(dot[:, :, k]), -(col(f2) * f32(isum[None, :, k])))
                acc = fma32(inner, d8[None, :, k], acc)
            return acc
        if type_id == 42:
            for b in (0, 1):
                part = zero
                for k in ks[2 * b: 2 * b + 2]:
                    part = fma32(f32(dot[:, :, k]), d8[None, :, k], part)
                acc = fma32(col(terms["d"][:, ks[2 * b] // 2]), part, acc)
            return acc
        part = zero
        for k in ks:
            if type_id == 23:
                ival = dot[:, :, k] * terms["ls"][:, k, None]
            else:
                ival = terms["s16"][:, 2 * k, None] * dots[0][:, :, k] + terms["s16"][:, 2 * k + 1, None] * dots[1][:, :, k]
            part = fma32(f32(ival), d8[None, :, k], part)
        return fma32(col(terms["d"][:, sb]), part, acc)

    nw = DTC_WARPS[order]
    ws = []
    for wp in range(nw):
        acc = zero
        for c in range(wp >> 2, nchunk, nw // 4):
            acc = half_sb(acc, c, wp & 3)
        ws.append(acc)
    if order == "one":
        h = []
        for q in range(4):
            v = ws[q]
            for j in range(1, 6):
                v = v + ws[4 * j + q]
            h.append(v)
        out = (h[0] + h[1]) + (h[2] + h[3])
    elif order in ("tile", "pair"):
        out = ((ws[0] + ws[4]) + (ws[1] + ws[5])) + ((ws[2] + ws[6]) + (ws[3] + ws[7]))
    elif order == "s1":
        out = ((ws[0] + ws[1]) + ws[2]) + ws[3]
    else:
        out = (((ws[0] + ws[1]) + ws[2]) + ws[3]) + (((ws[4] + ws[5]) + ws[6]) + ws[7])
    return out.T.copy()


# ---- the Q5_K head's top-1 (docs/research/basalt-nvcc/q5k_head.md section 2) -------------------------------------
# A Top is (m, idx, s) arrays. m and idx change only through comparisons of logits, so every __expf argument follows
# from the logits; ``expf`` evaluates the kernel's own expf_ftz on an array (MUFU.EX2 has no bit-exact model).

TOP_NONE = 0x7FFFFFFF


def top_take(t, l, row, on, expf):
    """Top::take of logit l at row ``row`` where ``on`` (q5k_head.cu:155-164)."""
    m, idx, s = t
    up = l > m
    with np.errstate(invalid="ignore"):
        e = expf(np.where(up, add_ftz(m, -l), add_ftz(l, -m)))
    s2 = np.where(up, fma_ftz(s, e, F32(1.0)), add_ftz(s, e))
    i2 = np.where(up | ((l == m) & (row < idx)), row, idx)
    return np.where(on, np.where(up, l, m), m).astype(F32), np.where(on, i2, idx), np.where(on, s2, s).astype(F32)


def top_merge(t, o, expf):
    """Top::merge of o into t (q5k_head.cu:165-175), elementwise."""
    (m, idx, s), (om, oidx, os_) = t, o
    on = oidx != TOP_NONE
    take = (idx == TOP_NONE) | (om > m)
    with np.errstate(invalid="ignore"):
        e = expf(np.where(on, np.where(take, add_ftz(m, -om), add_ftz(om, -m)), F32(0.0)))
    s2 = np.where(take, fma_ftz(s, e, os_), fma_ftz(os_, e, s))
    i2 = np.where(take | ((om == m) & (oidx < idx)), oidx, idx)
    return np.where(on & take, om, m).astype(F32), np.where(on, i2, idx), np.where(on, s2, s).astype(F32)


def q5k_top1(logits, mapped, blocks, expf):
    """q5k_head_top1 from the kernel's logits: SIMD-group k of threadgroup g takes rows 8g + k + 8 blocks i in order,
    unmapped rows skipped (:191-219); thread 0 merges SIMD-groups 0..7 (:297-299) into the partial; the merge: thread t
    merges partials t, t + 256, ... (:309-317), the XOR butterfly 16..1 over each SIMD-group (:318-324), thread 0 the
    SIMD-groups in order (:327-329). Returns (the partials (m, idx, s) per threadgroup, the row, the probability)."""
    n = len(logits)
    nw = 8 * blocks
    t = (np.full(nw, -np.inf, F32), np.full(nw, TOP_NONE, np.int64), np.zeros(nw, F32))
    for r0 in range(0, n, nw):
        row = r0 + np.arange(nw)
        ok = row < n
        rr = np.minimum(row, n - 1)
        t = top_take(t, np.asarray(logits, F32)[rr], row, ok & np.asarray(mapped)[rr], expf)
    w = [a.reshape(blocks, 8) for a in t]
    b = tuple(a[:, 0] for a in w)
    for k in range(1, 8):
        b = top_merge(b, tuple(a[:, k] for a in w), expf)
    part = b
    g = (np.full(256, -np.inf, F32), np.full(256, TOP_NONE, np.int64), np.zeros(256, F32))
    for i0 in range(0, blocks, 256):
        i = i0 + np.arange(256)
        ok = i < blocks
        ii = np.minimum(i, blocks - 1)
        o = (part[0][ii], np.where(ok, part[1][ii], TOP_NONE), part[2][ii])
        g = top_merge(g, o, expf)
    for off in (16, 8, 4, 2, 1):
        src = np.arange(256) ^ off
        g = top_merge(g, tuple(a[src] for a in g), expf)
    f = tuple(a[0:1] for a in g)
    for k in range(1, 8):
        f = top_merge(f, tuple(a[32 * k: 32 * k + 1] for a in g), expf)
    return part, int(f[1][0]), rcp_ftz(f[2])[0]


# ---- the 2-bit proxy (q5k_head.md section 4) ------------------------------------------------------------------------

PX_RCP15 = np.uint32(0x3F2AAAAB).view(F32)   # fl(1 / 1.5): ptxas's multiply for amax / 1.5f


def px_tiles(n):
    return (n + 31) // 32


def px_level(w, rs):
    """level(w, sh) as compiled: floor(FFMA.FTZ(w, rcp(sh), 2)) converted by F2I (NaN to 0, saturating), clamped."""
    with np.errstate(invalid="ignore"):
        f = np.floor(fma_ftz(w, rs, F32(2.0)))
        return np.where(np.isnan(f), 0, np.clip(f, 0, 3)).astype(np.int64)


def px_fit(w):
    """fit_group<false> on groups ``w`` (..., 32): (the fp16 scale bits, its value)."""
    amax = np.full(w.shape[:-1], F32(1e-9), F32)
    for l in range(32):
        amax = max_ftz(amax, ftz(np.abs(w[..., l])))
    a15 = mul_ftz(amax, PX_RCP15)
    best_s, best_e = a15, np.full(a15.shape, np.inf, F32)
    for t in range(6):
        s = mul_ftz(a15, fma_ftz(F32(t), F32(-0.1), F32(1.0)))
        sh = h2f(half_bits(s))
        lv = px_level(w, rcp_ftz(sh)[..., None])
        with np.errstate(invalid="ignore", over="ignore"):
            r = fma_ftz(add_ftz(lv.astype(F32), F32(-1.5)), sh[..., None], -w)
            e = np.zeros(a15.shape, F32)
            for l in range(32):
                e = fma_ftz(r[..., l], r[..., l], e)
        better = e < best_e
        best_e, best_s = np.where(better, e, best_e), np.where(better, s, best_s)
    hs = half_bits(best_s)
    return hs, h2f(hs)


def q5k_proxy_pack(q5, row0, n, n_cap, proxy=None):
    """q5k_proxy_pack_kernel with pack_group<false>: rows row0 .. row0 + n - 1 of the Q5_K rows ``q5`` (1760 bytes
    each) into the copy ``proxy`` (uint8 bytes; zeros when None). Returns the copy."""
    out = np.zeros(px_tiles(n_cap) * 40 * 32 * 20, np.uint8) if proxy is None else np.array(proxy, np.uint8)
    rows = row0 + np.arange(n)
    b = np.asarray(q5, np.uint8).reshape(-1, 1760)[rows].reshape(n, 10, 176)[:, np.arange(80) // 8]   # (n, 80, 176)
    j = np.arange(80) % 8
    scl = b[..., 4:16].astype(np.int64)
    at = lambda k: np.take_along_axis(scl, np.broadcast_to(k[None, :, None], (n, 80, 1)), axis=-1)[..., 0]
    lo = j < 4
    sci = np.where(lo, at(np.clip(j, 0, 3)) & 63, (at(np.clip(j, 4, 7) + 4) & 0xF) | ((at(np.clip(j, 4, 7) - 4) >> 6) << 4))
    mi = np.where(lo, at(np.clip(j, 0, 3) + 4) & 63, (at(np.clip(j, 4, 7) + 4) >> 4) | ((at(np.clip(j, 4, 7)) >> 6) << 4))
    d = mul_ftz(_h(b, 0), sci.astype(F32))
    mn = mul_ftz(_h(b, 2), mi.astype(F32))
    l = np.arange(32)
    qs = np.take_along_axis(b, np.broadcast_to((48 + 32 * (j // 2))[None, :, None] + l, (n, 80, 32)), axis=-1)
    qs = qs.astype(np.int64)
    nib = np.where((j & 1)[None, :, None] == 1, qs >> 4, qs & 15)
    q = nib | (((b[..., 16:48].astype(np.int64) >> j[None, :, None]) & 1) << 4)
    with np.errstate(invalid="ignore", over="ignore"):
        w = fma_ftz(d[..., None], q.astype(F32), -mn[..., None])
    hs, sh = px_fit(w)
    lv = px_level(w, rcp_ftz(sh)[..., None])                              # (n, 80, 32)
    sh_bits = (8 * (np.arange(16) % 4) + 2 * (np.arange(16) // 4)).astype(np.uint64)   # value 4 ii + bb: bit 8 bb + 2 ii
    words = [(lv[..., 16 * c: 16 * c + 16].astype(np.uint64) << sh_bits).sum(axis=-1).astype(np.uint32) for c in (0, 1)]
    g = np.arange(80)
    slot = ((rows[:, None] >> 5) * 40 + g[None] // 2) * 32 + (rows[:, None] & 31)
    h = (g & 1)[None]
    u32 = out[: px_tiles(n_cap) * 40 * 32 * 16].view(np.uint32)
    u32[slot * 4 + 2 * h] = words[0]
    u32[slot * 4 + 2 * h + 1] = words[1]
    out[px_tiles(n_cap) * 40 * 32 * 16:].view(np.uint16)[slot * 2 + h] = hs
    return out



def q5k_proxy_score(proxy, xq, n_rows, n_cap, map_, tiles, expf):
    """px::score_kernel on the tiles ``tiles`` of the copy ``proxy`` (``n_cap`` rows) against one q8_1 column ``xq``:
    (cand (tiles, 4, 2) float32 with the rows' bits in [..., 1], part (tiles, 2)). The exponent arguments follow from
    the scores; ``expf`` evaluates the kernel's own expf_ftz on them."""
    ntc, tiles = px_tiles(n_cap), np.asarray(tiles)
    nt_ = len(tiles)
    proxy = np.asarray(proxy, np.uint8)
    words = proxy[: ntc * 40 * 32 * 16].view(np.uint32).reshape(ntc, 40, 32, 4)[tiles]     # (tile, w4, lane, word)
    scw = proxy[ntc * 40 * 32 * 16:].view(np.uint32).reshape(ntc, 40, 32)[tiles]
    xb = np.asarray(xq, np.uint8).reshape(80, 36)
    d8 = h2f(xb[:, 0].astype(np.uint16) | (xb[:, 1].astype(np.uint16) << 8))
    codes = xb[:, 4:].view(np.int8).astype(np.int64)                                         # (80, 32)
    xs = mul_ftz(d8, codes.sum(axis=1).astype(F32))
    shifts = (8 * (np.arange(16) % 4) + 2 * (np.arange(16) // 4)).astype(np.uint32)        # value 4 ii + bb of a word
    lv = ((words[..., None] >> shifts) & 3).astype(np.int64).reshape(nt_, 40, 32, 2, 32)   # (.., h, 16 c + value)
    g = 2 * np.arange(40)[:, None] + np.arange(2)[None, :]                                  # group of (w4, h)
    P = (lv * codes[g][None, :, None]).sum(axis=-1)                                         # exact, as DP4A
    inner = fma_ftz(d8[g][None, :, None], P.astype(F32), -mul_ftz(xs[g], F32(1.5))[None, :, None])
    scale = h2f(np.stack([scw & 0xFFFF, scw >> 16], axis=-1).astype(np.uint16))
    red = []
    for k in range(4):                                     # SIMD-group k: w4 = 10 k + j for j = 0..9, then h = 0, 1
        acc = np.zeros((nt_, 32), F32)
        for j in range(10):
            for h in range(2):
                acc = fma_ftz(scale[:, 10 * k + j, :, h], inner[:, 10 * k + j, :, h], acc)
        red.append(acc)
    v = add_ftz(add_ftz(add_ftz(red[0], red[1]), red[2]), red[3])
    row = tiles[:, None] * 32 + np.arange(32)[None]
    masked = (row >= n_rows) | (np.asarray(map_, np.int64)[np.minimum(row, n_rows - 1)] < 0)
    v = np.where(masked, F32(-np.inf), v).astype(F32)
    m = butterfly(v, max_ftz)
    live = v > -np.inf
    with np.errstate(invalid="ignore"):
        arg = np.where(live, add_ftz(v, -m[:, None]), F32(0.0))
    e = np.where(live, expf(arg), F32(0.0)).astype(F32)
    part = np.stack([m, butterfly(e, add_ftz)], axis=-1)
    cand = np.zeros((nt_, 4, 2), F32)
    vv, at = v.copy(), np.arange(nt_)
    for k in range(4):
        bi = np.argmax(vv, axis=1)                         # the first maximum: ties to the lower row
        cand[:, k, 0] = vv[at, bi]
        cand[:, k, 1] = (tiles * 32 + bi).astype(np.int32).view(F32)
        vv[at, bi] = -np.inf
    return cand, part


def q5k_proxy_final(cand, part, exact, map_, expf):
    """q5k_proxy_final_kernel (collect, the exact rescoring, pick) from the score's ``cand`` (tiles, 4, 2) and ``part``
    (tiles, 2); ``exact``: the head's logits of every row (the kernel rescores the listed rows with the head's own row,
    native_mmvq's bits). The list in ascending candidate order, the port's fixed order (q5k_head.md section 4); ``expf``
    evaluates the kernel's expf_ftz. Returns (id, probability, the listed rows)."""
    c, pt = np.asarray(cand, F32).reshape(-1, 2), np.asarray(part, F32)
    m = np.max(c[:, 0])
    if m == -np.inf:                                        # every row is masked
        return (0 if map_[0] < 0 else int(map_[0])), F32(1.0), np.zeros(0, np.int64)
    sel = np.flatnonzero((c[:, 0] > -np.inf) & (c[:, 0] >= add_ftz(m, F32(-5.0))))[:64]
    rows, lp = c[sel, 1].view(np.int32).astype(np.int64), c[sel, 0]
    ex = np.asarray(exact, F32)[rows]
    bv = np.max(ex)
    bi = int(np.min(rows[ex == bv]))                        # the largest exact score, the lowest row on ties
    live = pt[:, 1] > 0
    et = expf(np.where(live, add_ftz(pt[:, 0], -bv), F32(0.0)))
    ed = add_ftz(expf(add_ftz(ex, -bv)), -expf(add_ftz(lp, -bv)))
    s = np.zeros(32, F32)                                   # lane L: tiles L, L + 32, ..., then rows L, L + 32
    for t0 in range(0, len(pt), 32):
        k = min(32, len(pt) - t0)
        s[:k] = np.where(live[t0:t0 + k], fma_ftz(pt[t0:t0 + k, 1], et[t0:t0 + k], s[:k]), s[:k])
    for c0 in range(0, len(rows), 32):
        k = min(32, len(rows) - c0)
        s[:k] = add_ftz(s[:k], ed[c0:c0 + k])
    total = max_ftz(butterfly(s[None], add_ftz)[0], F32(1.0 if bi in rows else 0.0))
    return int(np.asarray(map_)[bi]), rcp_ftz(total), rows
