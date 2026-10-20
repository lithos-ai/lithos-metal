# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1171-1197 (hc8_quant_kernel),
# src/core/model_file.cpp:487-513 (hc8_read) and src/kernels/hc8_parity.cpp:69-99 (hc8_parity's matrices) @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""Host references of basalt's hyper-connection unit for the port's tests: numpy only, no Metal, imported by every tier
like ``tests/moe_synth.py``. Each is a transcription of basalt's source written apart from the code it checks, so the
two cannot share a mistake.

* ``hc8_quant_scalar``: ``hc8_quant_kernel``, one group and one value at a time, a numpy float32 or float16 scalar at
  every step (the host keeps binary32 subnormals, as the IEEE-built kernel does); the independent side of
  ``monolith.packs.basalt_hc8.quantize``.
* ``dequant_copy``: a copy's exact values, ``q * f16(s)`` in float64, the weights the int8 dot products multiply by.
* ``hc8_parity_matrices``: hc8_parity's down, up and injection matrices, drawn with numpy.
* ``hc8_read_table``: ``hc8_read``, basalt's reader of a ``BASALTH8`` table, its checks transcribed.
"""

import struct
from pathlib import Path

import numpy as np

from monolith.formats.fp import f32_to_bf16


def hc8_quant_scalar(w, gpr, srow):                  # hc8_quant_kernel, hc_fused.cu:1171-1197, G8 = 32 (:343)
    """The copy of the BF16 bits ``w`` (any shape, flattened row-major; ``gpr`` scales per row, ``srow`` the row
    stride of the scales): ``(q, sc)``, the codes as ``int8 [rows * cols]`` and the scales' f16 bits as
    ``uint16 [rows * srow]``, padding scales 0."""
    w = np.asarray(w, np.uint16).reshape(-1)
    groups = w.size // 32
    q = np.zeros(w.size, np.int8)
    sc = np.zeros(groups // gpr * srow, np.uint16)   # cudaMemset(c.s, 0, rows * srow * 2) (:1223)
    for gi in range(groups):
        x = [np.uint32(int(w[gi * 32 + i]) << 16).view(np.float32) for i in range(32)]  # :1178
        amax = np.float32(0.0)
        for i in range(32):
            amax = np.fmax(amax, np.abs(x[i]))       # :1179 fmaxf(amax, fabsf(x[i]))
        with np.errstate(over="ignore"):             # above f16's range the scale is inf, as cvt.rn.f16.f32 gives
            h = np.float16(amax / np.float32(127.0))  # :1181 __float2half_rn(amax / 127.0f): div.rn, RNE
        s = np.float32(h)                            # :1182 __half2float(h)
        sc[gi // gpr * srow + gi % gpr] = h.view(np.uint16)                            # :1183
        for i in range(32):                          # :1185-1186
            v = np.fmin(np.fmax(np.rint(x[i] / s), np.float32(-127.0)), np.float32(127.0)) if s > 0 else np.float32(0.0)
            q[gi * 32 + i] = np.int8(int(v))
    return q, sc


def dequant_copy(q, s, cols):
    """The exact values of a copy: ``q * f16(s)`` in float64, ``[rows, cols]``, each scale broadcast over its group's 32
    columns. ``q`` holds the codes' int8 bit patterns (int8 or uint8, ``rows * cols`` of them), ``s`` the scales' f16
    bits or values (``uint16`` or ``float16``, ``[rows, srow]``). A code times a finite f16 scale has at most 18
    significant bits, so every such product is exact in double (an inf scale, a group whose absmax exceeds ``127 *
    65504``, has codes 0 and gives NaN): hc8_parity's "the copy's exact values in double", ``(double) q * (double)
    f16f(s)`` (``src/kernels/hc8_parity.cpp:254-259``), without the BF16 rounding of its part 1's dequantized copies."""
    q = np.asarray(q).view(np.int8).reshape(-1, cols)
    sc = np.asarray(s).view(np.float16).reshape(q.shape[0], -1).astype(np.float64)
    return q.astype(np.float64) * sc[:, np.arange(cols) // 32]


def hc8_parity_matrices(seed=7):
    """hc8_parity's three matrices as BF16 bits (``src/kernels/hc8_parity.cpp:69-99``): its shapes and distributions,
    with numpy's normals in place of its ``mt19937(7)`` ones (drawn in its order, down, up, injection; the values are
    numpy's, not basalt's): ``w_down [320][10240] = bf16(0.02 N)`` with the 64 outliers ``w_down[5 k][101 k] =
    bf16(0.6)`` that set their groups' scales (``:84``), ``w_up [10240][320] = bf16(0.05 N)`` and ``w_inject [4][10240]
    = bf16(0.02 N)``, each value the float32 product of the constant and an N(0, 1) draw (``0.02f * nd(rng)``), rounded
    to nearest even on the bits (basalt's ``bf16``, ``:36-40``)."""
    rng = np.random.default_rng(seed)
    w_down, w_up, w_inject = (f32_to_bf16(np.float32(sc) * rng.standard_normal(shape, dtype=np.float32))
                              for sc, shape in ((0.02, (320, 10240)), (0.05, (10240, 320)), (0.02, (4, 10240))))
    k = np.arange(64)
    w_down[5 * k, 101 * k] = f32_to_bf16(np.float32([0.6]))
    return w_down, w_up, w_inject


def hc8_read_table(path):                            # hc8_read, src/core/model_file.cpp:487-513
    """The ``BASALTH8`` table in the file ``path`` read as ``hc8_read`` reads it, with its checks: the magic and version
    1, at most 4096 entries, and each entry's ``bytes == rows * cols + rows * srow * 2`` (in uint64), its range inside
    the file, ``cols % 32 == 0`` and ``srow >= cols / 32``; ``ValueError`` with basalt's message otherwise.
    ``{name: (rows, cols, srow, offset, bytes, data)}`` in the table's order, ``data`` the entry's bytes, the name up
    to its first NUL (``strnlen``); a later entry of a name replaces the earlier one, as in basalt's map."""
    data = Path(path).read_bytes()
    if len(data) < 16:                               # read_range_at of the header (:490)
        raise ValueError(f"short read from {path}")
    magic, version, n = struct.unpack_from("<8sII", data)
    if magic != b"BASALTH8" or version != 1:
        raise ValueError(f"hc8: bad magic or version in {path}")
    if n > 4096:
        raise ValueError("hc8: implausible entry count")
    if len(data) < 16 + 160 * n:                     # read_range_at of the entries (:495)
        raise ValueError(f"short read from {path}")
    out = {}
    for i in range(n):
        raw, rows, cols, srow, _, offset, nbytes = struct.unpack_from("<128sIIIIQQ", data, 16 + 160 * i)
        name = raw.split(b"\0", 1)[0].decode()
        want = (rows * cols + rows * srow * 2) % 2 ** 64                                   # :502, uint64_t
        if nbytes != want or offset > len(data) or want > len(data) - offset or cols % 32 or srow < cols // 32:
            raise ValueError(f"hc8: entry {name} is malformed")
        out[name] = (rows, cols, srow, offset, nbytes, data[offset: offset + want])
    return out
