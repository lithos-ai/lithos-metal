"""BF16 weights: the target's embeddings, norms and small GDN projections; the CI model; BF16 drafters.
``weight`` ``BF16 [N, K]`` (uint16 view). Lane-row unit (K = 5120): 320 bytes = 20 words.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .blm import PackInfo, join_lanes, pack_blm, split_lanes, unpack_blm
from .fp import bf16_to_f32, f32_to_bf16
from .registry import register_format


@register_format("bf16")
class BF16(Format):
    bytes_per_weight = 2.0
    weights_per_word = 8
    scale_group = 0
    msl_decode = """
#define WEIGHTS_PER_WORD 8u
#define SCALE_GROUP 0u
#define BF16_STORAGE 1
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint i = 0; i < 4; i++) { out[2 * i] = as_type<float>(w[i] << 16); out[2 * i + 1] = as_type<float>(w[i] & 0xFFFF0000u); }
}
static inline float decode_scale(thread const uint* sw, uint g) { return 1.0f; }
"""

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        w = np.asarray(tensors["weight"], dtype=np.uint16)
        if w.shape != tuple(shape):
            raise ValueError(f"bf16: weight {w.shape} does not match shape {shape}")
        return DequantSpec("bf16", tuple(shape), {"weight": w})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        return bf16_to_f32(spec.tensors["weight"]).astype(np.float32)

    def quantize(self, w: np.ndarray) -> DequantSpec:
        w = np.asarray(w, dtype=np.float32)
        return DequantSpec("bf16", w.shape, {"weight": f32_to_bf16(w)})

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo]:
        n, k = spec.shape
        payload = split_lanes(spec.tensors["weight"].view(np.uint8).reshape(n, 2 * k), k, 2, 1)
        return pack_blm(payload, None, layout, format="bf16", k=k)

    def unpack_pack(self, data: bytes, info: PackInfo) -> DequantSpec:
        payload, _ = unpack_blm(data, info)
        return DequantSpec("bf16", (info.n, info.k), {"weight": join_lanes(payload).view(np.uint16).reshape(info.n, info.k)})
