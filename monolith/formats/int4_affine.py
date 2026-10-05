"""Affine INT4 groups — the MLX / AWQ / GPTQ family (``mlx.core.quantize`` with ``bits = 4``, ``mode = "affine"``):
unsigned 4-bit codes, eight per ``uint32`` low nibble first, in ``weight`` ``U32 [N, K/8]``; a scale and a bias per
group of ``group_size`` columns (64 by default, 32 and 128 exist) in ``scales`` / ``biases`` ``F16`` or ``BF16``
``[N, K/g]``. Dequantization: ``w = scale · code + bias``. The plugin keeps the scales and biases as **16-bit pairs in
the checkpoint's own dtype** — BF16 (MLX's BF16 models) or F16 (AWQ / GPTQ, MLX's F16 models), recorded in the pack as
``scale_dtype`` and decoded by the kernels accordingly (``SCALE_F16``), so the dequantized weights equal the
checkpoint's to the bit; FP32 pairs are kept when they are exactly representable in one of the two and refused
otherwise (re-quantize at pack time instead); a matrix quantized at pack time rounds its FP32 pair to BF16 first —
mlx's quantizer does the same for a BF16 matrix. 4 bytes per group, the bytes mlx streams (#113: the FP32 pairs it
kept before were 12 % of the 0.6B's GEMV traffic); the GEMV's bias term is ``bias · Σ x`` over the group
(``SCALE_BIAS``).

Lane-row unit (K = 4096): 64 bytes of nibbles (4 words) + 8 bytes of scale/bias pairs → 72, padded to 80 inline or
kept as whole words with the pairs in the block's scale region (``scale_placement = "block"``); ``SCALE_GROUP = 64``.
A lane's stripe (K/32 columns) may be a fraction of a group (K = 1024: half a group per lane) or start inside one
(K = 3584: stripes of 112 columns — 3.5 words, a *ragged* stripe — start 0/48/32/16 columns into a group): the lane
carries the pairs of every group its stripe touches (the max over lanes, zero-padded — K = 3584: three pairs, 12
bytes after 56 of nibbles), and the kernels index them from the stripe's offset in its first group (``LANE_OFF`` /
``GROUP_SEG`` in gemv_T, the same arithmetic in embed). K must be a multiple of 256. A pack written with the FP32
pairs (before 2026-09-26) records ``scale_unit_bytes`` 8 and is refused by the decode kernels: re-pack it.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .blm import LANES, PackInfo, join_lanes, lane_groups, pack_blm, split_lanes, unpack_blm
from .fp import bf16_to_f32, f32_to_bf16, pack_nibbles, unpack_nibbles
from .registry import register_format

GROUP = 64


def _lane_groups(k: int, group: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per lane: the first group its stripe touches and how many (``blm.lane_groups``); K % 256 for the decode kernels."""
    if k % 256:
        raise ValueError(f"int4_affine: K must be a multiple of 256 (K={k})")
    return lane_groups(k, group)


@register_format("int4_affine")
class INT4Affine(Format):
    bytes_per_weight = 0.5 + 4.0 / GROUP
    weights_per_word = 32
    scale_group = GROUP
    pack_k_multiple = 256
    scale_unit_bytes = 4            # a 16-bit (scale, bias) pair per group: scale in the low half, bias in the high; BF16 or F16 (SCALE_F16)
    msl_decode = """
#define WEIGHTS_PER_WORD 32u
#define SCALE_GROUP 64u
#define SCALE_BIAS 1
#ifndef SCALE_F16
#define SCALE_F16 0                  // 1: the pairs are the checkpoint's F16 (AWQ / GPTQ / an F16 MLX model); 0: BF16
#endif
// 32 unsigned nibbles → the codes 0 … 15 as floats
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 32; e++) out[e] = float((w[e >> 3] >> ((e & 7u) * 4u)) & 0xFu);
}
// group g of this lane-row: the 16-bit pair (scale, bias) in uint g of the unit's scale region — scale low, bias high
#if SCALE_F16
static inline float decode_scale(thread const uint* sw, uint g) { return float(as_type<half>(ushort(sw[g] & 0xFFFFu))); }
static inline float decode_bias(thread const uint* sw, uint g) { return float(as_type<half>(ushort(sw[g] >> 16))); }
#else
static inline float decode_scale(thread const uint* sw, uint g) { return as_type<float>(sw[g] << 16); }
static inline float decode_bias(thread const uint* sw, uint g) { return as_type<float>(sw[g] & 0xFFFF0000u); }
#endif
"""

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        n, k = shape
        w = np.asarray(tensors["weight"])
        if w.dtype != np.uint32 or w.shape != (n, k // 8) or k % 8:
            raise ValueError(f"int4_affine: weight {w.shape} {w.dtype} does not match shape {shape}")
        sc, bi = _to_f32(tensors["scales"]), _to_f32(tensors["biases"])
        if sc.shape != bi.shape or sc.shape[0] != n or k % sc.shape[1]:
            raise ValueError(f"int4_affine: scales {sc.shape} / biases {bi.shape} do not match shape {shape}")
        group = k // sc.shape[1]
        return DequantSpec("int4_affine", (n, k), {"weight": np.ascontiguousarray(w.view(np.uint8).reshape(n, k // 2)),
                                                  "scales": sc, "biases": bi}, {"group": group, "pair_dtype": _pair_dtype(tensors["scales"], tensors["biases"], sc, bi)})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        n, k = spec.shape
        g = int(spec.params["group"])
        codes = unpack_nibbles(spec.tensors["weight"]).astype(np.float32).reshape(n, k // g, g)
        sc = spec.tensors["scales"].astype(np.float32).reshape(n, k // g, 1)
        bi = spec.tensors["biases"].astype(np.float32).reshape(n, k // g, 1)
        return (codes * sc + bi).reshape(n, k).astype(np.float32)

    def quantize(self, w: np.ndarray, group: int = GROUP) -> DequantSpec:
        """``mlx.core.quantize``'s affine rule, bit-exact in FP32 (mlx 0.32, ``affine_quantize``): per group
        ``w_max = max(max w, 0)``, ``scale₀ = max((w_max − w_min) / 15, 1e-7)``; the endpoint of larger magnitude is
        the ``edge`` (the min when ``|w_min| > |w_max|``, else the max with a negative scale), and the scale is snapped
        so the edge lands on the integer code ``q₀ = round(edge / scale₀)``: ``scale = edge / q₀``, ``bias = edge``
        (``q₀ = 0`` — a near-zero group — keeps ``scale₀`` and a zero bias); ``code = min(round((w − bias) / scale), 15)``,
        halves away from zero. Not a fixed point under re-quantization: the snapped grid of ``dequantize(quantize(w))``
        re-snaps in some groups, so the contract test bounds the drift by one code step instead of asking for identity.
        Scales and biases stay FP32 here; an MLX checkpoint rounds them to its dtype before ``unpack`` sees them."""
        w = np.asarray(w, dtype=np.float32)
        n, k = w.shape
        if k % group:
            raise ValueError(f"int4_affine: K must be a multiple of the group ({group})")
        blocks = w.reshape(n, k // group, group)
        w_min = blocks.min(axis=-1)
        w_max = np.maximum(blocks.max(axis=-1), np.float32(0))
        scale = np.maximum((w_max - w_min) / np.float32(15), np.float32(1e-7)).astype(np.float32)
        side = np.abs(w_min) > np.abs(w_max)
        scale = np.where(side, scale, -scale).astype(np.float32)
        edge = np.where(side, w_min, w_max).astype(np.float32)
        q0 = _round_away(edge / scale)
        at_zero = q0 == 0
        scale = np.where(at_zero, scale, edge / np.where(at_zero, np.float32(1), q0)).astype(np.float32)
        bias = np.where(at_zero, np.float32(0), edge).astype(np.float32)
        codes = np.clip(_round_away((blocks - bias[..., None]) / scale[..., None]), 0, 15).astype(np.uint8).reshape(n, k)
        scale, bias = bf16_to_f32(f32_to_bf16(scale)), bf16_to_f32(f32_to_bf16(bias))          # the pack's (and mlx's) BF16 pairs
        return DequantSpec("int4_affine", (n, k), {"weight": pack_nibbles(codes), "scales": scale, "biases": bias}, {"group": group, "pair_dtype": "bf16"})

    def _lanes(self, spec: DequantSpec) -> Tuple[np.ndarray, np.ndarray]:
        n, k = spec.shape
        g = int(spec.params["group"])
        payload = split_lanes(spec.tensors["weight"], k, 1, 2)                          # [N, 32, K/64] (a ragged stripe: not whole words)
        first, count, gpl = _lane_groups(k, g)
        pairs = np.zeros((n, LANES, gpl, 2), np.float32)                                # the (scale, bias) of every group a lane touches
        for lane in range(LANES):
            gs = slice(int(first[lane]), int(first[lane] + count[lane]))
            pairs[:, lane, : count[lane], 0] = spec.tensors["scales"][:, gs]
            pairs[:, lane, : count[lane], 1] = spec.tensors["biases"][:, gs]
        if _spec_pair_dtype(spec) == "f16":
            scales = np.ascontiguousarray(pairs.astype(np.float16)).view(np.uint8).reshape(n, LANES, gpl * 4)   # F16 pairs: scale, bias
        else:
            scales = np.ascontiguousarray(f32_to_bf16(pairs)).view(np.uint8).reshape(n, LANES, gpl * 4)      # BF16 pairs: scale, bias
        return payload, scales

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo]:
        n, k = spec.shape
        g = int(spec.params["group"])
        if g != GROUP:
            raise ValueError(f"int4_affine: the kernel decode is compiled for groups of {GROUP}, the checkpoint uses {g}")
        payload, scales = self._lanes(spec)
        return pack_blm(payload, scales, layout, format="int4_affine", k=k, scale_group=GROUP, scale_dtype=_spec_pair_dtype(spec))

    def unpack_pack(self, data: bytes, info: PackInfo) -> DequantSpec:
        payload, scales = unpack_blm(data, info)
        n, k = info.n, info.k
        first, count, gpl = _lane_groups(k, GROUP)
        if info.scale_unit_bytes and info.scale_unit_bytes != self.scale_unit_bytes:
            raise ValueError(f"int4_affine: the pack keeps {info.scale_unit_bytes}-byte scale pairs (FP32, before 2026-09-26): re-pack it")
        dtype = info.scale_dtype or "bf16"
        raw = np.ascontiguousarray(scales).view(np.uint16)
        pairs = (raw.view(np.float16).astype(np.float32) if dtype == "f16" else bf16_to_f32(raw)).reshape(n, LANES, gpl, 2)
        sc, bi = np.zeros((n, k // GROUP), np.float32), np.zeros((n, k // GROUP), np.float32)
        for lane in range(LANES):                                                       # a group shared by two lanes is stored twice, identically
            gs = slice(int(first[lane]), int(first[lane] + count[lane]))
            sc[:, gs] = pairs[:, lane, : count[lane], 0]
            bi[:, gs] = pairs[:, lane, : count[lane], 1]
        return DequantSpec("int4_affine", (n, k), {"weight": join_lanes(payload), "scales": sc, "biases": bi}, {"group": GROUP, "pair_dtype": dtype})


def _spec_pair_dtype(spec: DequantSpec) -> str:
    return str(spec.params.get("pair_dtype", "bf16"))


def _exact_in(values: np.ndarray, dtype: str) -> bool:
    """Every value survives the round trip through ``dtype`` (bf16 or f16) unchanged."""
    v = np.asarray(values, np.float32)
    back = bf16_to_f32(f32_to_bf16(v)) if dtype == "bf16" else v.astype(np.float16).astype(np.float32)
    return bool(np.array_equal(back, v))


def _pair_dtype(raw_scales: Any, raw_biases: Any, sc: np.ndarray, bi: np.ndarray) -> str:
    """The 16-bit dtype the pack keeps the (scale, bias) pairs in: the checkpoint's own where it is 16-bit (BF16 as
    uint16 bit patterns from the reader, F16 as float16); FP32 pairs whichever of the two holds them exactly, BF16
    first; refused when neither does — the pack must dequantize to the checkpoint's reference to the bit."""
    ds, db = np.asarray(raw_scales).dtype, np.asarray(raw_biases).dtype
    if ds == np.float16 and db == np.float16:
        return "f16"
    if ds == np.uint16 and db == np.uint16:
        return "bf16"
    for dtype in ("bf16", "f16"):
        if _exact_in(sc, dtype) and _exact_in(bi, dtype):
            return dtype
    raise ValueError("int4_affine: the checkpoint's FP32 scales / biases are representable in neither BF16 nor F16 — the pack keeps 16-bit "
                     "pairs; re-quantize the matrix at pack time (pack_weights.py --quantize int4_affine) or convert the checkpoint")


def _round_away(x: np.ndarray) -> np.ndarray:
    """Metal's ``round``: to nearest, halves away from zero (numpy's ``rint`` takes halves to even)."""
    x = np.asarray(x, dtype=np.float32)
    t = np.trunc(x)
    half = np.abs(x - t) == np.float32(0.5)                      # exact: x − trunc(x) is exact below 2^23
    return np.where(half, t + np.sign(x), np.rint(x)).astype(np.float32)


def _to_f32(a: Any) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.uint16:                                  # a BF16 bit pattern from the safetensors reader
        return bf16_to_f32(a)
    return a.astype(np.float32)
