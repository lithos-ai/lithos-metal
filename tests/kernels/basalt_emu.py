# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu, src/kernels/cuda/iq_kernels.cu and
# include/basalt/kernels/q8_1_finite.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36, which adapt ggml-org/llama.cpp
# ggml/src/ggml-common.h (kvalues_iq4nl), ggml/src/ggml-quants.c (get_scale_min_k4) and ggml/src/ggml-cuda/quantize.cu
# @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
"""numpy emulation of basalt's float32 arithmetic as nvcc compiles it (docs/research/basalt-nvcc/matvec.md), for the
bit-for-bit tests of the Metal port. Every float operation rounds once to float32, to nearest even: numpy's float32
add, multiply and divide do; ``fma32`` does by rounding the exact float64 sum to odd before the float32 cast. The
``*_ftz`` forms flush subnormal inputs and results to the zero of their sign, as SASS's ``.FTZ`` instructions do."""

import numpy as np

F32 = np.float32


def ftz(x):
    """A subnormal becomes the zero of its sign; every other value keeps its bits."""
    x = np.asarray(x, F32)
    b = x.view(np.uint32)
    return np.where((b & 0x7F800000) == 0, (b & 0x80000000).view(F32), x).astype(F32)


def fma32(a, b, c):
    """a * b + c rounded once to float32 (to nearest even). a * b is exact in float64; the float64 sum is rounded to
    odd (TwoSum gives its exact error), which a float32 cast then rounds correctly (53 >= 24 + 2 bits)."""
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
