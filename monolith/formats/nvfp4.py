"""NVFP4 as ModelOpt stores it (``nvidia/Qwen3.8-27B-NVFP4``): E2M1 codes two per byte (low nibble first) in
``weight`` ``U8 [N, K/2]``, an E4M3 block scale per 16 columns in ``weight_scale`` ``F8_E4M3 [N, K/16]``, and one FP32
tensor scale ``weight_scale_2``. Dequantization: ``w = e2m1(code) · e4m3(block_scale) · weight_scale_2``. MLX's
``nvfp4`` mode (``mlx_lm.convert -q --mode nvfp4``) stores the same codes eight per ``U32`` (little-endian, so the
byte view is the ModelOpt layout) in ``weight`` ``U32 [N, K/8]`` and the E4M3 block scales as ``scales`` ``U8
[N, K/16]``, with no tensor scale: ``unpack`` reads both (a tensor scale of 1 for MLX), verified bit-exact against
``mx.dequantize``.
(the ModelOpt/vLLM convention; ``input_scale`` is an activation-quantization parameter and is ignored — design D11).

Lane-row unit (K = 5120): 80 bytes of nibbles (5 words) + 10 scale bytes, padded to 96; ``SCALE_GROUP = 16``.

Decode variants (``NVFP4_DECODE``): 0 = per-nibble float bit construction (the reference, ~14 ALU ops per weight),
1 = nibble pairs to ``half2`` with packed 16-bit integer arithmetic, 2 = the eight magnitudes as small integers in one
32-bit constant with an int→float conversion, the ×0.5 folded into the block scale, 3 = MLX's ``fp4.h`` decode (the
default): the three magnitude bits placed into a half's exponent field, the 2¹⁴ folded into the block scale —
bit-identical to 2 and 2–25 % faster per shape on the M5 Pro (docs/research/gemv-kernel-study.md §3, §3e).

Convention checked on the real checkpoint (layer 0 ``down_proj``, 2026-09-24): the stored block-scale codes are all
non-negative and top out at exactly 0x7E (448), i.e. ``weight_scale_2 = amax / (6 · 448)``; the dequantized matrix has
RMS 0.011 and absmax 0.98, while reading the tensor scale as a divisor or dropping it gives 7.9e4 or 29.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .blm import LANES, PackInfo, join_lanes, lane_groups, pack_blm, split_lanes, unpack_blm
from .fp import (E2M1_MAX, E4M3_MAX, e2m1_to_f32, e4m3_to_f32, f32_to_e2m1, f32_to_e4m3, pack_nibbles,
                 unpack_nibbles)
from .registry import register_format

BLOCK = 16


@register_format("nvfp4")
class NVFP4(Format):
    bytes_per_weight = 0.5 + 1.0 / BLOCK
    weights_per_word = 32
    scale_group = BLOCK
    pack_k_multiple = 256           # the decode kernels' stripe granularity; a stripe narrower than a group shares the group's byte
    msl_decode = """
#define WEIGHTS_PER_WORD 32u
#define SCALE_GROUP 16u
#ifndef NVFP4_DECODE
#define NVFP4_DECODE 3      // measured on the M5 Pro (gemv-kernel-study.md §3e, 2026-09-26): V3 > V2 > V1 > V0 on every shape
#endif
#if NVFP4_DECODE == 0
// V0: per nibble, float bit construction (reference; ~14 ALU ops per weight)
static inline float fp4_e2m1(uint q) {
  uint e = (q >> 1) & 3u, m = q & 1u;
  float v = (e == 0u) ? float(m) * 0.5f : as_type<float>(((e + 126u) << 23) | (m << 22));
  return (q & 8u) ? -v : v;
}
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 32; e++) out[e] = fp4_e2m1((w[e >> 3] >> ((e & 7u) * 4u)) & 0xFu);
}
#elif NVFP4_DECODE == 1
// V1: nibble PAIRS -> half2 bits with packed 16-bit integer arithmetic (both halves of one uint at once), then float2.
//   magnitude code m3 -> half bits: 0 -> 0, 1 -> 0x3800 (0.5), m3 >= 2 -> 0x3C00 + (m3-2)*0x200  (= 1, 1.5, 2, 3, 4, 6)
//   written branch-free as nz*0x3600 + m3*0x200 + ge2*0x200; sign bit 3 -> bit 15.
static inline uint fp4pair_half2_bits(uint c) {           // c = byte: nibble a in bits 0-3, nibble b in bits 4-7
  uint u = (c & 0xFu) | ((c & 0xF0u) << 12);              // a at [0,4), b at [16,20)
  uint m3 = u & 0x00070007u;
  uint nz = (m3 | (m3 >> 1) | (m3 >> 2)) & 0x00010001u;
  uint ge2 = ((m3 >> 1) | (m3 >> 2)) & 0x00010001u;
  return nz * 0x3600u + m3 * 0x200u + ge2 * 0x200u + ((u & 0x00080008u) << 12);
}
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint i = 0; i < 4; i++) for (uint b = 0; b < 4; b++) {
    float2 f = float2(as_type<half2>(fp4pair_half2_bits((w[i] >> (b * 8u)) & 0xFFu)));
    out[i * 8 + b * 2] = f.x; out[i * 8 + b * 2 + 1] = f.y;
  }
}
#elif NVFP4_DECODE == 2
// V2: the 8 magnitudes as small integers (value*2 = 0,1,2,3,4,6,8,12) packed in ONE 32-bit constant, 4 bits each;
//   int -> float conversion, sign by select; the *0.5 folds into the block scale via decode_scale.
#define NVFP4_LUT2 0xC8643210u
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 32; e++) {
    uint c = (w[e >> 3] >> ((e & 7u) * 4u)) & 0xFu;
    int k = int((NVFP4_LUT2 >> ((c & 7u) << 2)) & 0xFu);
    out[e] = float((c & 8u) ? -k : k);
  }
}
#elif NVFP4_DECODE == 3
// V3: MLX's fp4.h decode (ml-explore/mlx, mlx/backend/metal/kernels/fp4.h, v0.32, MIT — third_party/NOTICE): the three magnitude bits placed straight into a half's
//   exponent field — as_type<half>((c & 7) << 9) is the E2M1 value times 2^-14 exactly (e = 0 lands in the subnormals:
//   m · 2^-15) — the sign a select, then half -> float; the 2^14 folds into the block scale (decode_scale). The same
//   products and sums as V2 up to an exact power of two: bit-identical outputs.
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 32; e++) {
    uint c = (w[e >> 3] >> ((e & 7u) * 4u)) & 0xFu;
    half h = as_type<half>(ushort((c & 7u) << 9));
    out[e] = float((c & 8u) ? -h : h);
  }
}
#endif
static inline float fp8_e4m3_scale(uint q) {
  // Adapted from MLX fp8.h @ 1f8e74e3f12f31365464a6867c6579f0e9b29d85
  // (MIT; third_party/NOTICE). Half conversion handles E4M3 subnormals too;
  // moving the exact power-of-two multiply to float preserves every scale bit.
  half h = as_type<half>(ushort((q & 127u) << 7));
  return float((q & 128u) ? -h : h) * 256.0f;
}
// scale of group g of this lane-row: byte g of the unit's scale region (held in registers as uints)
#if NVFP4_DECODE == 2
static inline float decode_scale(thread const uint* sw, uint g) { return 0.5f * fp8_e4m3_scale((sw[g >> 2] >> ((g & 3u) * 8u)) & 0xFFu); }
#elif NVFP4_DECODE == 3
static inline float decode_scale(thread const uint* sw, uint g) { return 16384.0f * fp8_e4m3_scale((sw[g >> 2] >> ((g & 3u) * 8u)) & 0xFFu); }
#else
static inline float decode_scale(thread const uint* sw, uint g) { return fp8_e4m3_scale((sw[g >> 2] >> ((g & 3u) * 8u)) & 0xFFu); }
#endif
"""

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        n, k = shape
        if "weight_scale_2" in tensors:                                          # ModelOpt: U8 codes, F8_E4M3 scales, a tensor scale
            w = np.asarray(tensors["weight"], dtype=np.uint8)
            sc = np.asarray(tensors["weight_scale"], dtype=np.uint8)
            s2 = float(np.asarray(tensors["weight_scale_2"], dtype=np.float32).reshape(()))
        elif "scales" in tensors:                                                # MLX: U32 words of 8 codes, U8 E4M3 scales, no tensor scale
            w32 = np.ascontiguousarray(np.asarray(tensors["weight"], dtype=np.uint32))
            w = w32.view(np.uint8).reshape(w32.shape[0], w32.shape[1] * 4)
            sc = np.asarray(tensors["scales"], dtype=np.uint8)
            s2 = 1.0
        else:
            raise ValueError("nvfp4: expected weight_scale + weight_scale_2 (ModelOpt) or scales (MLX) beside the weight")
        if w.shape != (n, k // 2) or sc.shape != (n, k // BLOCK) or k % BLOCK:
            raise ValueError(f"nvfp4: weight {w.shape}, scales {sc.shape} do not match shape {shape}")
        return DequantSpec("nvfp4", (n, k), {"weight": w, "weight_scale": sc}, {"weight_scale_2": s2, "block": BLOCK})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        n, k = spec.shape
        codes = unpack_nibbles(spec.tensors["weight"])                                   # [N, K]
        vals = e2m1_to_f32(codes).reshape(n, k // BLOCK, BLOCK)
        scales = e4m3_to_f32(spec.tensors["weight_scale"]).reshape(n, k // BLOCK, 1)
        return (vals * scales * np.float32(spec.params["weight_scale_2"])).reshape(n, k).astype(np.float32)

    def quantize(self, w: np.ndarray) -> DequantSpec:
        w = np.asarray(w, dtype=np.float32)
        n, k = w.shape
        if k % BLOCK:
            raise ValueError("nvfp4: K must be a multiple of 16")
        amax = float(np.abs(w).max()) or 1.0
        s2 = np.float32(amax / (E2M1_MAX * E4M3_MAX))
        blocks = w.reshape(n, k // BLOCK, BLOCK)
        bmax = np.abs(blocks).max(axis=-1)                                                # [N, K/16]
        sc = f32_to_e4m3(np.where(bmax > 0, bmax / E2M1_MAX / s2, 0.0))
        scf = e4m3_to_f32(sc)
        denom = scf[..., None] * s2
        scaled = np.divide(blocks, denom, out=np.zeros_like(blocks), where=denom > 0)
        codes = f32_to_e2m1(scaled.reshape(n, k))
        return DequantSpec("nvfp4", (n, k), {"weight": pack_nibbles(codes), "weight_scale": sc},
                           {"weight_scale_2": float(s2), "block": BLOCK})

    def _lanes(self, spec: DequantSpec) -> Tuple[np.ndarray, np.ndarray]:
        n, k = spec.shape
        payload = split_lanes(spec.tensors["weight"], k, 1, 2)                          # [N, 32, K/64]
        if (k // 32) % BLOCK == 0:
            return payload, split_lanes(spec.tensors["weight_scale"], k, 1, BLOCK)      # [N, 32, K/512]: whole groups per stripe
        first, count, gpl = lane_groups(k, BLOCK)                                       # narrow stripes: the group(s) a lane touches
        sc = np.zeros((n, LANES, gpl), np.uint8)
        for lane in range(LANES):
            sc[:, lane, : count[lane]] = spec.tensors["weight_scale"][:, first[lane]: first[lane] + count[lane]]
        return payload, sc

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo]:
        n, k = spec.shape
        if k % 256:
            raise ValueError(f"nvfp4: K must be a multiple of 256 for the decode kernels (K={k})")
        payload, scales = self._lanes(spec)
        return pack_blm(payload, scales, layout, format="nvfp4", k=k, tensor_scale=spec.params["weight_scale_2"],
                        scale_group=BLOCK)

    def unpack_pack(self, data: bytes, info: PackInfo) -> DequantSpec:
        payload, scales = unpack_blm(data, info)
        n, k = info.n, info.k
        if (k // 32) % BLOCK == 0:
            sc = join_lanes(scales)
        else:
            first, count, gpl = lane_groups(k, BLOCK)
            sc = np.zeros((n, k // BLOCK), np.uint8)
            for lane in range(LANES):                                                   # a group shared by lanes is stored by each, identically
                sc[:, first[lane]: first[lane] + count[lane]] = scales[:, lane, : count[lane]]
        return DequantSpec("nvfp4", (n, k), {"weight": join_lanes(payload), "weight_scale": sc},
                           {"weight_scale_2": info.tensor_scale, "block": BLOCK})
