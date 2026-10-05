"""FP8 E4M3 with one FP32 per-tensor scale (ModelOpt ``FP8``: attention and GDN projections of the target):
``weight`` ``F8_E4M3 [N, K]``, ``weight_scale`` ``F32 []``; ``w = e4m3(code) · weight_scale``. ``input_scale`` ignored.

Lane-row unit (K = 5120): 160 bytes = 10 words, no block scales.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .blm import PackInfo, join_lanes, pack_blm, split_lanes, unpack_blm
from .fp import E4M3_MAX, e4m3_to_f32, f32_to_e4m3
from .registry import register_format


@register_format("fp8_e4m3")
class FP8E4M3(Format):
    bytes_per_weight = 1.0
    weights_per_word = 16
    scale_group = 0
    msl_decode = """
#define WEIGHTS_PER_WORD 16u
#define SCALE_GROUP 0u
#ifndef FP8_DECODE
#define FP8_DECODE 0
#endif
static inline float fp8_e4m3(uint q) {
#if FP8_DECODE
  // Place the sign and magnitude in a half, including subnormals, then correct
  // the exponent bias in float. Opted in by the measured 512-row prefill policy.
  half h = as_type<half>(ushort(((q & 127u) << 7) | ((q & 128u) << 8)));
  return float(h) * 256.0f;
#else
  uint e = (q >> 3) & 15u, m = q & 7u;
  float v = as_type<float>(((q & 0x7Fu) << 20) + (120u << 23));
  v = (e == 0u) ? float(m) * (1.0f / 512.0f) : v;
  return (q & 0x80u) ? -v : v;
#endif
}
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 16; e++) out[e] = fp8_e4m3((w[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu);
}
static inline float decode_scale(thread const uint* sw, uint g) { return 1.0f; }
"""

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        n, k = shape
        w = np.asarray(tensors["weight"], dtype=np.uint8)
        s = float(np.asarray(tensors["weight_scale"], dtype=np.float32).reshape(()))
        if w.shape != (n, k):
            raise ValueError(f"fp8_e4m3: weight {w.shape} does not match shape {shape}")
        return DequantSpec("fp8_e4m3", (n, k), {"weight": w}, {"weight_scale": s})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        return (e4m3_to_f32(spec.tensors["weight"]) * np.float32(spec.params["weight_scale"])).astype(np.float32)

    def quantize(self, w: np.ndarray) -> DequantSpec:
        w = np.asarray(w, dtype=np.float32)
        amax = float(np.abs(w).max()) or 1.0
        s = np.float32(amax / E4M3_MAX)
        return DequantSpec("fp8_e4m3", w.shape, {"weight": f32_to_e4m3(w / s)}, {"weight_scale": float(s)})

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo]:
        n, k = spec.shape
        payload = split_lanes(spec.tensors["weight"], k, 1, 1)
        return pack_blm(payload, None, layout, format="fp8_e4m3", k=k, tensor_scale=spec.params["weight_scale"])

    def unpack_pack(self, data: bytes, info: PackInfo) -> DequantSpec:
        payload, _ = unpack_blm(data, info)
        return DequantSpec("fp8_e4m3", (info.n, info.k), {"weight": join_lanes(payload)}, {"weight_scale": info.tensor_scale})
