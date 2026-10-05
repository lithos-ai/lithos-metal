"""Exact float-code tables and conversions for the storage formats, in numpy (no torch).

* E4M3 (``float8_e4m3fn``): bias 7, no infinities, ``0x7F``/``0xFF`` are NaN, max finite 448. Used by FP8 weights and
  by NVFP4's per-16 block scales (stored non-negative, decoded as the signed type torch/ModelOpt use).
* E2M1 (NVFP4 codes): ``0 0.5 1 1.5 2 3 4 6`` with a sign bit; two codes per byte, **low nibble first** (element 2i in
  bits 0–3, element 2i+1 in bits 4–7 — the ModelOpt/vLLM packing convention).
* BF16: the top 16 bits of an IEEE float32; conversion rounds to nearest even.

The conversions are bit-exact by construction (tables), so they are the numerics reference the kernels' decode
snippets are checked against.
"""

from __future__ import annotations

import numpy as np

# ---- E4M3 ----------------------------------------------------------------------------------------------------------
def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint32)
    s = (codes >> 7) & 1
    e = (codes >> 3) & 0xF
    m = codes & 0x7
    val = np.where(e == 0, m.astype(np.float64) * 2.0 ** -9, (1.0 + m / 8.0) * 2.0 ** (e.astype(np.int64) - 7))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    val = np.where(s == 1, -val, val)
    return val.astype(np.float32)


E4M3_TABLE: np.ndarray = _e4m3_table()
E4M3_MAX = 448.0

_E4M3_POS_CODES = np.array([c for c in range(128) if c != 0x7F], dtype=np.uint8)      # finite non-negative codes
_E4M3_POS_VALS = E4M3_TABLE[_E4M3_POS_CODES]                                        # ascending with code


def e4m3_to_f32(codes: np.ndarray) -> np.ndarray:
    return E4M3_TABLE[np.asarray(codes, dtype=np.uint8)]


def f32_to_e4m3(x: np.ndarray, *, saturate: bool = True) -> np.ndarray:
    """Round-to-nearest-even to E4M3; magnitudes above 448 saturate (or raise); NaN → 0x7F."""
    x = np.asarray(x, dtype=np.float32)
    mag = np.abs(x).astype(np.float64)
    out = np.empty(x.shape, dtype=np.uint8)
    nan = np.isnan(x)
    over = mag > E4M3_MAX
    if over.any() and not saturate:
        raise ValueError("f32_to_e4m3: magnitude above 448")
    idx = np.searchsorted(_E4M3_POS_VALS, mag, side="left")            # first value >= mag
    idx = np.clip(idx, 0, len(_E4M3_POS_VALS) - 1)
    lo = np.clip(idx - 1, 0, len(_E4M3_POS_VALS) - 1)
    hi = idx
    dlo = mag - _E4M3_POS_VALS[lo]
    dhi = _E4M3_POS_VALS[hi] - mag
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (_E4M3_POS_CODES[hi] % 2 == 0))
    code = np.where(pick_hi, _E4M3_POS_CODES[hi], _E4M3_POS_CODES[lo]).astype(np.uint8)
    code = np.where(over, np.uint8(0x7E), code)
    code = np.where(np.signbit(x), code | np.uint8(0x80), code)
    code = np.where(nan, np.uint8(0x7F), code)
    out[...] = code
    return out


# ---- E2M1 (NVFP4 codes) --------------------------------------------------------------------------------------------
E2M1_TABLE: np.ndarray = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
E2M1_MAX = 6.0
_E2M1_POS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)


def e2m1_to_f32(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8)
    if c.size and c.max() > 15:
        raise ValueError("e2m1_to_f32: codes must be 4-bit")
    return E2M1_TABLE[c]


def f32_to_e2m1(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even to E2M1 (values are assumed pre-scaled into [-6, 6]; larger magnitudes saturate)."""
    x = np.asarray(x, dtype=np.float32)
    mag = np.minimum(np.abs(x).astype(np.float64), E2M1_MAX)
    idx = np.clip(np.searchsorted(_E2M1_POS, mag, side="left"), 0, 7)
    lo = np.clip(idx - 1, 0, 7)
    dlo, dhi = mag - _E2M1_POS[lo], _E2M1_POS[idx] - mag
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (idx % 2 == 0))
    code = np.where(pick_hi, idx, lo).astype(np.uint8)
    return np.where(np.signbit(x), code | np.uint8(8), code).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """``[..., K/2]`` uint8 → ``[..., K]`` codes, low nibble first."""
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = p & 0xF
    out[..., 1::2] = p >> 4
    return out


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8)
    if c.shape[-1] % 2:
        raise ValueError("pack_nibbles: last dimension must be even")
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


# ---- BF16 ----------------------------------------------------------------------------------------------------------
def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    u = np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even; NaN payloads are preserved with the quiet bit set."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(x)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    out = np.where(nan, (u >> 16) | 0x40, rounded)
    return out.astype(np.uint16)
