"""INT8 with a ``half`` scale per group of 32 (symmetric, absmax): the load-time re-quantization for BF16 drafters
and Markov heads (llama.cpp's ``Q8_0`` shape). ``weight`` ``I8 [N, K]``, ``weight_scale`` ``F16 [N, K/32]``;
``w = code · scale``. Lane-row unit (K = 5120): 160 bytes + 10 scale bytes → 176; ``SCALE_GROUP = 32``.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .blm import LANES, PackInfo, join_lanes, lane_groups, pack_blm, split_lanes, unpack_blm
from .registry import register_format

GROUP = 32


@register_format("int8")
class INT8(Format):
    bytes_per_weight = 1.0 + 2.0 / GROUP
    weights_per_word = 16
    scale_group = GROUP
    pack_k_multiple = 256           # a stripe narrower than a group shares the group's scale (see nvfp4)
    scale_unit_bytes = 2
    msl_decode = """
#define WEIGHTS_PER_WORD 16u
#define SCALE_GROUP 32u
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 16; e++) out[e] = float(int(w[e >> 2] << (24u - (e & 3u) * 8u)) >> 24);
}
// scale of group g: half g of the unit's scale region
static inline float decode_scale(thread const uint* sw, uint g) { return float(as_type<half>(ushort((sw[g >> 1] >> ((g & 1u) * 16u)) & 0xFFFFu))); }
"""

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        n, k = shape
        w = np.asarray(tensors["weight"], dtype=np.int8)
        sc = np.asarray(tensors["weight_scale"], dtype=np.float16)
        if w.shape != (n, k) or sc.shape != (n, k // GROUP) or k % GROUP:
            raise ValueError(f"int8: weight {w.shape}, scales {sc.shape} do not match shape {shape}")
        return DequantSpec("int8", (n, k), {"weight": w, "weight_scale": sc}, {"group": GROUP})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        n, k = spec.shape
        w = spec.tensors["weight"].astype(np.float32).reshape(n, k // GROUP, GROUP)
        s = spec.tensors["weight_scale"].astype(np.float32).reshape(n, k // GROUP, 1)
        return (w * s).reshape(n, k).astype(np.float32)

    def quantize(self, w: np.ndarray) -> DequantSpec:
        w = np.asarray(w, dtype=np.float32)
        n, k = w.shape
        if k % GROUP:
            raise ValueError("int8: K must be a multiple of 32")
        g = w.reshape(n, k // GROUP, GROUP)
        sc = (np.abs(g).max(axis=-1) / 127.0).astype(np.float16)
        scf = sc.astype(np.float32)[..., None]
        codes = np.where(scf > 0, np.rint(g / np.where(scf > 0, scf, 1.0)), 0.0)
        codes = np.clip(codes, -127, 127).astype(np.int8).reshape(n, k)
        return DequantSpec("int8", (n, k), {"weight": codes, "weight_scale": sc}, {"group": GROUP})

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo]:
        n, k = spec.shape
        if k % 256:
            raise ValueError(f"int8: K must be a multiple of 256 for the decode kernels (K={k})")
        payload = split_lanes(spec.tensors["weight"].view(np.uint8), k, 1, 1)
        if (k // 32) % GROUP == 0:
            scales = split_lanes(spec.tensors["weight_scale"].view(np.uint8).reshape(n, 2 * (k // GROUP)), k, 2, GROUP)
        else:                                                                           # narrow stripes: the group(s) a lane touches
            first, count, gpl = lane_groups(k, GROUP)
            sc = np.zeros((n, LANES, gpl), np.float16)
            for lane in range(LANES):
                sc[:, lane, : count[lane]] = spec.tensors["weight_scale"][:, first[lane]: first[lane] + count[lane]]
            scales = np.ascontiguousarray(sc).view(np.uint8).reshape(n, LANES, 2 * gpl)
        return pack_blm(payload, scales, layout, format="int8", k=k, scale_group=GROUP)

    def unpack_pack(self, data: bytes, info: PackInfo) -> DequantSpec:
        payload, scales = unpack_blm(data, info)
        n, k = info.n, info.k
        if (k // 32) % GROUP == 0:
            ws = join_lanes(scales).view(np.float16).reshape(n, k // GROUP)
        else:
            first, count, gpl = lane_groups(k, GROUP)
            lanes16 = np.ascontiguousarray(scales).view(np.float16).reshape(n, LANES, gpl)
            ws = np.zeros((n, k // GROUP), np.float16)
            for lane in range(LANES):
                ws[:, first[lane]: first[lane] + count[lane]] = lanes16[:, lane, : count[lane]]
        return DequantSpec("int8", (n, k),
                           {"weight": join_lanes(payload).view(np.int8),
                            "weight_scale": ws},
                           {"group": GROUP})
