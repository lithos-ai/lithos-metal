"""Shared helpers for the kernel benches and the GPU tests: synthetic packed matrices with exact oracles, the BF16
ULP comparison of the numerics contract (leaf ops ≤ 2 ULP), device/profile lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .backends.metal import ChipConfig as Profile, config_for_device
from .formats import FORMATS, DequantSpec, PackLayout
from .formats.blm import PackInfo
from .formats.fp import E2M1_MAX, E4M3_MAX, bf16_to_f32, f32_to_bf16


def random_spec(fmt: str, n: int, k: int, rng: np.random.Generator) -> DequantSpec:
    """A random matrix *in the format's own codes* (no quantization pass), so large shapes are cheap to make."""
    if fmt == "nvfp4":
        codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
        scales = rng.integers(0x20, 0x48, size=(n, k // 16), dtype=np.uint8)            # E4M3 in [2^-3, 2^2)
        return DequantSpec(fmt, (n, k), {"weight": codes, "weight_scale": scales}, {"weight_scale_2": 0.02 / (E2M1_MAX * E4M3_MAX) * 64, "block": 16})
    if fmt == "fp8_e4m3":
        codes = (rng.integers(0, 256, size=(n, k), dtype=np.uint8) & 0xBF)             # exponent <= 7: |w| < 2
        return DequantSpec(fmt, (n, k), {"weight": codes}, {"weight_scale": 0.02})
    if fmt == "bf16":
        return DequantSpec(fmt, (n, k), {"weight": f32_to_bf16((rng.standard_normal((n, k)) * 0.02).astype(np.float32))})
    if fmt == "int8":
        codes = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
        scales = (rng.uniform(0.5, 2.0, size=(n, k // 32)) * 0.02 / 127).astype(np.float16)
        return DequantSpec(fmt, (n, k), {"weight": codes, "weight_scale": scales}, {"group": 32})
    if fmt == "int4_affine":
        codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
        scales = (rng.uniform(0.5, 2.0, size=(n, k // 64)) * 0.02 / 7.5).astype(np.float32)
        biases = (-7.5 * scales * rng.uniform(0.8, 1.2, size=scales.shape)).astype(np.float32)
        scales, biases = bf16_to_f32(f32_to_bf16(scales)), bf16_to_f32(f32_to_bf16(biases))   # the checkpoints' (and the pack's) BF16 pairs
        return DequantSpec(fmt, (n, k), {"weight": codes, "scales": scales, "biases": biases}, {"group": 64, "pair_dtype": "bf16"})
    raise KeyError(fmt)


def rows_of(spec: DequantSpec, rows: np.ndarray) -> DequantSpec:
    t = {k: (v[rows] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == spec.shape[0] else v) for k, v in spec.tensors.items()}
    return DequantSpec(spec.format, (len(rows), spec.shape[1]), t, dict(spec.params))


def pack_spec(spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo, np.ndarray]:
    """``(bytes, info, row_scales)`` with the uniform per-tensor scale as the row-scale table.

    The kernel computes ``codes × block scale × row_scale``; ``FORMATS[fmt].dequantize(spec)`` already includes the
    per-tensor scale, so it IS the oracle — never multiply it by ``row_scales`` again."""
    fmt = FORMATS.get(spec.format)
    data, info = fmt.pack(spec, layout)
    scale = float(spec.params.get("weight_scale_2", spec.params.get("weight_scale", 1.0)))
    return data, info, np.full(spec.shape[0], scale, dtype=np.float32)


def bf16_ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in BF16 ULPs after rounding both to BF16 (sign-magnitude codes made monotone)."""
    def key(x):
        u = f32_to_bf16(np.asarray(x, dtype=np.float32)).astype(np.int64)
        return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), u + 0x8000)
    return np.abs(key(a) - key(b))


@dataclass
class OracleCheck:
    """The leaf-op gate (design §5.9): every output within 2 BF16 ULPs *at the output's magnitude*.

    ``max_ulp_at_rms`` = max |y - y_ref| / ulp_bf16(rms(y_ref)); gating element-wise ULPs would fail on outputs that
    happen to be near zero, where float32 accumulation-order noise is many BF16 ULPs of the element itself but
    irrelevant once the BF16 residual stream rounds it. ``max_ulp_elementwise`` is reported for information.
    """

    max_ulp_at_rms: float
    max_ulp_elementwise: int
    max_rel_err: float          # max |y - y_ref| / max |y_ref| (float32 accumulation noise is ~1e-6)
    max_ulp_at_scale: float = 0.0   # max |y - y_ref| / ulp_bf16(max(|y_ref|, rms(y_ref))): the element's own ULP, floored at the RMS

    def ok(self) -> bool:
        return self.max_ulp_at_rms <= 2.0 and self.max_rel_err < 1e-4

    def ok_rounded(self) -> bool:
        """The gate for BF16 outputs that went through a chain of roundings (mixers, norms): every element within 2
        ULP of its own magnitude (floored at the RMS, so near-zero elements do not count their noise as ULPs)."""
        return self.max_ulp_at_scale <= 2.0 and self.max_rel_err < 1e-2


def bf16_ulp_of(magnitude: float) -> float:
    """The BF16 spacing at ``magnitude`` (8 significand bits incl. the hidden one)."""
    return float(2.0 ** (np.floor(np.log2(max(magnitude, 1e-30))) - 7))


def check_against_oracle(y: np.ndarray, y_ref: np.ndarray) -> OracleCheck:
    y = np.asarray(y, dtype=np.float64); y_ref = np.asarray(y_ref, dtype=np.float64)
    err = np.abs(y - y_ref)
    rms = float(np.sqrt(np.mean(y_ref ** 2)))
    denom = max(float(np.abs(y_ref).max()), 1e-30)
    mags = np.maximum(np.abs(y_ref), rms)
    ulps = 2.0 ** (np.floor(np.log2(np.maximum(mags, 1e-30))) - 7)
    return OracleCheck(float(err.max() / bf16_ulp_of(rms)), int(bf16_ulp_diff(y, y_ref).max()), float(err.max() / denom),
                       float((err / ulps).max()))


def profile_for_device(gpu_cores: int, apple_family: int, chip: Optional[str] = None) -> Optional[Profile]:
    """Compatibility wrapper; runtime callers also supply the exact chip name."""
    return config_for_device(gpu_cores, apple_family, chip)
