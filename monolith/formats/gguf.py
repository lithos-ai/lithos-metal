# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
"""The ggml block formats of GGUF files (llama.cpp), as the ``.basalt`` packs ship them (``basalt_reader.py``).

A GGUF tensor ``[ne0, ne1]`` is ``ne1`` rows of ``ne0`` values, ``ne0`` the contiguous axis: the ``[N, K]`` matrix with
``N = ne1``, ``K = ne0``. A row is ``K / block`` blocks back to back; a block is the struct of ggml-common.h, given here
as a numpy structured dtype field for field (``layout``). ``unpack({"weight": rows}, shape=(N, K))`` takes the rows'
raw bytes (an array of any dtype, viewed as bytes); ``dequantize`` is the oracle: the float32 values of ggml's
reference ``dequantize_row_*`` (ggml/src/ggml-quants.c of llama.cpp 3cf03257), bit for bit.

Exactness (an inference from the operand widths, checked by the contract tests): every product these decoders form is
exact in float32. A value is an fp16 scale (at most 11 significant bits) times at most two small integers, and the
widest product, Q6_K's ``d·sc·q`` or IQ4_XS's ``d·(ls − 32)·kvalue``, needs 23 bits, inside float32's normal range.
The one rounding step in all of them is the ``d·sc·q − dmin·m`` subtraction of Q4_K and Q5_K, and an FMA rounds it
the same way: the oracle's values do not depend on the evaluation order, on FP contraction or on the device.

Type ids are the GGUF tensor type numbers (ggml.h ``enum ggml_type``); ``GGML_TYPES`` maps the ones the format
understands. 42 is ``GGML_TYPE_Q2_0`` (ggml.h at 3cf03257: ``GGML_TYPE_Q1_0 = 41, GGML_TYPE_Q2_0 = 42``; basalt's
include/basalt/ggml/ggml.h carries the same id): 64 values per block, an fp16 scale ``d`` and sixteen bytes of 2-bit
codes, value ``(code − 1) · d``. BF16 rows are the ``bf16`` plugin's ``[N, K]`` uint16 matrix. F32 and F16 are
one-value blocks, ``f32`` and ``f16``: a dtype view, exact (in a pack they are norms, biases and convolution taps).

``pack`` / ``unpack_pack`` (the block-lane-major GEMV packs) and ``msl_decode`` arrive with the Metal decode kernels.
"""

# adapted from ggml-org/llama.cpp ggml/src/ggml-common.h and ggml/src/ggml-quants.c @ 3cf03257f219afbe7334045ff7c6a06ac68c627d
# (MIT, third_party/GGML-LICENSE): the block structs as dtypes and the dequantize_row_* rules, re-implemented in numpy

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .registry import FORMATS, register_format

QK_K = 256

# ggml type id -> (name, values per block, bytes per block); ggml.h and ggml-common.h of llama.cpp 3cf03257
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 6: ("q5_0", 32, 22), 8: ("q8_0", 32, 34),
    11: ("q3_k", QK_K, 110), 12: ("q4_k", QK_K, 144), 13: ("q5_k", QK_K, 176), 14: ("q6_k", QK_K, 210),
    20: ("iq4_nl", 32, 18), 23: ("iq4_xs", QK_K, 136), 30: ("bf16", 1, 2), 42: ("q2_0", 64, 18),
}

# kvalues_iq4nl, copied from ggml/src/ggml-common.h (the commit above)
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)

_E = np.arange(QK_K)                          # value index inside a 256-value super-block


def ggml_row_bytes(type_id: int, ne0: int) -> int:
    """Bytes of one row of ``ne0`` values (ggml_row_size)."""
    if type_id not in GGML_TYPES:
        raise ValueError(f"ggml type {type_id} has no format here")
    name, block, size = GGML_TYPES[type_id]
    if ne0 % block:
        raise ValueError(f"{name}: a row of {ne0} values is not whole blocks of {block}")
    return ne0 // block * size


def ggml_dequantize(type_id: int, rows: Any, shape: Tuple[int, int]) -> np.ndarray:
    """``N`` GGUF rows of type ``type_id`` (``shape = (N, K)``, their raw bytes in ``rows``) as float32 ``[N, K]``."""
    n, k = shape
    raw = np.ascontiguousarray(rows).reshape(-1).view(np.uint8)
    if raw.size != n * ggml_row_bytes(type_id, k):
        raise ValueError(f"ggml type {type_id}: {raw.size} bytes are not {n} rows of {k} values")
    name = GGML_TYPES[type_id][0]
    fmt = FORMATS.get(name)
    weight = raw.view("<u2").reshape(n, k) if name == "bf16" else raw
    return fmt.dequantize(fmt.unpack({"weight": weight}, shape=(n, k)))


def _f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)                # fp16 -> fp32 is exact


def _nibbles(qs: np.ndarray) -> np.ndarray:
    """``[nb, 16]`` bytes -> 32 codes: low nibbles of bytes 0 … 15, then their high nibbles (ggml's split halves)."""
    return np.concatenate([qs & 0xF, qs >> 4], axis=-1)


def _scale_min_k4(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The eight 6-bit (scale, min) pairs of a Q4_K / Q5_K block's 12 scale bytes (ggml-quants.c get_scale_min_k4)."""
    sc = np.concatenate([s[:, 0:4] & 63, (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)], axis=1)
    mn = np.concatenate([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], axis=1)
    return sc.astype(np.float32), mn.astype(np.float32)


class GGMLBlocks(Format):
    """A ggml block format: ``type_id`` and ``layout`` (one block); ``_values`` decodes ``[nb]`` blocks to ``[nb, block]``."""

    type_id: int = -1
    layout: np.dtype = np.dtype([])

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        n, k = shape
        raw = np.ascontiguousarray(tensors["weight"]).reshape(-1).view(np.uint8)
        if raw.size != n * ggml_row_bytes(self.type_id, k):
            raise ValueError(f"{self.name}: {raw.size} bytes are not {n} rows of {k} values")
        return DequantSpec(self.name, (n, k), {"weight": raw.reshape(n, -1)}, {"type_id": self.type_id})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        n, k = spec.shape
        blocks = np.ascontiguousarray(spec.tensors["weight"]).reshape(-1).view(self.layout)
        return self._values(blocks).reshape(n, k)

    def _values(self, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, Any]:
        raise NotImplementedError(f"{self.name}: no GEMV pack yet; it comes with the Metal decode kernels")

    def unpack_pack(self, data: bytes, info: Any) -> DequantSpec:
        raise NotImplementedError(f"{self.name}: no GEMV pack yet; it comes with the Metal decode kernels")


def _ggml(type_id: int, layout: list):
    """Register a block format under its ggml name, checking its layout against the type table."""
    name, block, size = GGML_TYPES[type_id]

    def deco(cls):
        cls.type_id, cls.layout, cls.bytes_per_weight = type_id, np.dtype(layout), size / block
        if cls.layout.itemsize != size:
            raise TypeError(f"{name}: the block layout is {cls.layout.itemsize} bytes, ggml's is {size}")
        return register_format(name)(cls)

    return deco


@_ggml(0, [("v", "<f4")])
class F32(GGMLBlocks):
    def _values(self, b):
        return b["v"].astype(np.float32)[:, None]


@_ggml(1, [("v", "<f2")])
class F16(GGMLBlocks):
    def _values(self, b):
        return _f32(b["v"])[:, None]


@_ggml(42, [("d", "<f2"), ("qs", "u1", 16)])
class Q2_0(GGMLBlocks):
    """value j = ((qs[j / 4] >> 2 (j % 4)) & 3) − 1, times d: the codes {0, 1, 2, 3} are {−1, 0, +1, +2}."""

    def _values(self, b):
        q = (b["qs"][:, :, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
        return (q.reshape(-1, 64).astype(np.float32) - 1) * _f32(b["d"])[:, None]


@_ggml(2, [("d", "<f2"), ("qs", "u1", 16)])
class Q4_0(GGMLBlocks):
    """value j < 16 is the low nibble of qs[j], value j + 16 its high nibble; minus 8, times d."""

    def _values(self, b):
        return (_nibbles(b["qs"]).astype(np.float32) - 8) * _f32(b["d"])[:, None]


@_ggml(6, [("d", "<f2"), ("qh", "u1", 4), ("qs", "u1", 16)])
class Q5_0(GGMLBlocks):
    """The fifth bit of value j is bit j of ``qh`` read as a little-endian uint32."""

    def _values(self, b):
        qh = np.ascontiguousarray(b["qh"]).view("<u4")
        hi = ((qh >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8)
        return ((_nibbles(b["qs"]) | (hi << 4)).astype(np.float32) - 16) * _f32(b["d"])[:, None]


@_ggml(8, [("d", "<f2"), ("qs", "i1", 32)])
class Q8_0(GGMLBlocks):
    def _values(self, b):
        return b["qs"].astype(np.float32) * _f32(b["d"])[:, None]


_Q3_H, _Q3_J, _Q3_SUB = _E // 128, (_E % 128) // 32, (_E % 32) // 16
_Q3_QS, _Q3_HM = 32 * _Q3_H + 16 * _Q3_SUB + _E % 16, 16 * _Q3_SUB + _E % 16
_Q3_SHIFT, _Q3_BIT = (2 * _Q3_J).astype(np.uint8), (4 * _Q3_H + _Q3_J).astype(np.uint8)


@_ggml(11, [("hmask", "u1", 32), ("qs", "u1", 64), ("scales", "u1", 12), ("d", "<f2")])
class Q3_K(GGMLBlocks):
    """Sixteen 6-bit scales re-packed from 12 bytes through ggml's kmask arithmetic, minus 32; value
    e = 128 h + 32 j + 16 sub + l has scale e / 16, its 2-bit code at bits 2 j of qs[32 h + 16 sub + l] and its high
    bit at bit 4 h + j of hmask[16 sub + l]; the code is lowered by 4 when the high bit is clear."""

    def _values(self, b):
        a = np.ascontiguousarray(b["scales"]).view("<u4")
        a0, a1, t = a[:, 0], a[:, 1], a[:, 2]
        k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
        aux = np.stack([(a0 & k2) | ((t & k1) << 4), (a1 & k2) | (((t >> 2) & k1) << 4),
                        ((a0 >> 4) & k2) | (((t >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((t >> 6) & k1) << 4)], axis=1)
        dl = _f32(b["d"])[:, None] * (np.ascontiguousarray(aux).view(np.int8).astype(np.float32) - 32)
        q = (b["qs"][:, _Q3_QS] >> _Q3_SHIFT) & 3
        high = (b["hmask"][:, _Q3_HM] >> _Q3_BIT) & 1
        return dl[:, _E // 16] * (q.astype(np.float32) - np.where(high, np.float32(0), np.float32(4)))


_K4_SB = _E // 32                              # Q4_K / Q5_K: sub-block of value e (its scale and min)
_K4_BYTE = (_K4_SB // 2) * 32 + _E % 32        # the byte holding its low four bits: low nibble for even sub-blocks
_K4_SHIFT, _K4_HBIT = (4 * (_K4_SB % 2)).astype(np.uint8), _K4_SB.astype(np.uint8)


@_ggml(12, [("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", 12), ("qs", "u1", 128)])
class Q4_K(GGMLBlocks):
    """Eight sub-blocks of 32, each with a 6-bit scale and min (get_scale_min_k4): value e of sub-block s = e / 32 is
    the low (even s) or high (odd s) nibble of qs[32 (s / 2) + e % 32], times d·sc[s], minus dmin·m[s]."""

    def _values(self, b):
        sc, mn = _scale_min_k4(b["scales"])
        d, m = _f32(b["d"])[:, None] * sc, _f32(b["dmin"])[:, None] * mn
        q = (b["qs"][:, _K4_BYTE] >> _K4_SHIFT) & 0xF
        return d[:, _K4_SB] * q.astype(np.float32) - m[:, _K4_SB]


@_ggml(13, [("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", 12), ("qh", "u1", 32), ("qs", "u1", 128)])
class Q5_K(GGMLBlocks):
    """Q4_K plus a fifth bit: bit ``sub-block`` of qh[e % 32]."""

    def _values(self, b):
        sc, mn = _scale_min_k4(b["scales"])
        d, m = _f32(b["d"])[:, None] * sc, _f32(b["dmin"])[:, None] * mn
        q = ((b["qs"][:, _K4_BYTE] >> _K4_SHIFT) & 0xF) | (((b["qh"][:, _E % 32] >> _K4_HBIT) & 1) << 4)
        return d[:, _K4_SB] * q.astype(np.float32) - m[:, _K4_SB]


_Q6_H, _Q6_K = _E // 128, (_E % 128) // 32
_Q6_QL, _Q6_QH = 64 * _Q6_H + 32 * (_Q6_K % 2) + _E % 32, 32 * _Q6_H + _E % 32
_Q6_LSHIFT, _Q6_HSHIFT = (4 * (_Q6_K // 2)).astype(np.uint8), (2 * _Q6_K).astype(np.uint8)


@_ggml(14, [("ql", "u1", 128), ("qh", "u1", 64), ("scales", "i1", 16), ("d", "<f2")])
class Q6_K(GGMLBlocks):
    """Value e = 128 h + 32 k + l: low four bits from ql[64 h + 32 (k % 2) + l] (the high nibble for k ≥ 2), two high
    bits at bits 2 k of qh[32 h + l], minus 32; scale e / 16."""

    def _values(self, b):
        lo = (b["ql"][:, _Q6_QL] >> _Q6_LSHIFT) & 0xF
        hi = (b["qh"][:, _Q6_QH] >> _Q6_HSHIFT) & 3
        d = _f32(b["d"])[:, None] * b["scales"].astype(np.float32)
        return d[:, _E // 16] * ((lo | (hi << 4)).astype(np.float32) - 32)


@_ggml(20, [("d", "<f2"), ("qs", "u1", 16)])
class IQ4_NL(GGMLBlocks):
    """Q4_0's nibble order through the non-linear codebook ``KVALUES_IQ4NL``."""

    def _values(self, b):
        return KVALUES_IQ4NL[_nibbles(b["qs"])] * _f32(b["d"])[:, None]


_IB = np.arange(8)


@_ggml(23, [("d", "<f2"), ("scales_h", "<u2"), ("scales_l", "u1", 4), ("qs", "u1", 128)])
class IQ4_XS(GGMLBlocks):
    """Eight groups of 32: the 6-bit group scale from a nibble of scales_l and two bits of scales_h, minus 32; the
    values IQ4_NL's codebook in IQ4_NL's nibble order."""

    def _values(self, b):
        ls = ((b["scales_l"][:, _IB // 2] >> (4 * (_IB % 2))) & 0xF) | (((b["scales_h"][:, None] >> (2 * _IB)) & 3) << 4)
        dl = _f32(b["d"])[:, None] * (ls.astype(np.float32) - 32)
        q = _nibbles(b["qs"].reshape(-1, 8, 16)).reshape(-1, QK_K)
        return dl[:, _E // 32] * KVALUES_IQ4NL[q]
