# Copyright (c) 2023-2026 The ggml authors (MIT; see third_party/TURBOQUANT-LICENSE).
# adapted from TheTom/llama-cpp-turboquant ggml/src/ggml-turbo-quant.c,
# ggml/src/ggml-quants.c @ 6e09de1b44ae328f641d1367bb4c875a8ffab43f
"""Position-major packed KV storage and NumPy reference codecs (no weight-format changes)."""
import numpy as np

KV_CACHE_DTYPES = ("bf16", "q8_0", "turbo4", "q4_0")
CENTROIDS = np.array([-0.241529, -0.182877, -0.143016, -0.111036,
    -0.083292, -0.058050, -0.034299, -0.011349,
     0.011349, 0.034299, 0.058050, 0.083292,
     0.111036, 0.143016, 0.182877, 0.241529], dtype=np.float32)
THRESHOLDS = np.array([-0.212203, -0.162947, -0.127026, -0.097164,
    -0.070671, -0.046174, -0.022824, 0, 0.022824, 0.046174,
    0.070671, 0.097164, 0.127026, 0.162947, 0.212203], dtype=np.float32)
SIGNS1 = np.array([-1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
SIGNS2 = np.array([1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float32)


def cache_row_bytes(head_dim, dtype):
    if dtype not in KV_CACHE_DTYPES:
        raise ValueError(f"Unknown KV cache dtype {dtype!r}; choose {KV_CACHE_DTYPES}")
    block = 128 if dtype == "turbo4" else 32
    if head_dim <= 0 or (dtype != "bf16" and head_dim % block):
        raise ValueError(f"{dtype} KV cache requires head_dim divisible by {block}, got {head_dim}")
    return head_dim * 2 if dtype == "bf16" else head_dim // block * {"q8_0": 34, "turbo4": 66, "q4_0": 18}[dtype]


def resolve_kv_cache_dtypes(kv_cache_dtype="bf16", k_cache_dtype=None, v_cache_dtype=None):
    """Per-side overrides take precedence over the shared default, independent of CLI order."""
    for dtype in (kv_cache_dtype, k_cache_dtype, v_cache_dtype):
        if dtype is not None:
            cache_row_bytes(128, dtype)
    return (k_cache_dtype if k_cache_dtype is not None else kv_cache_dtype,
            v_cache_dtype if v_cache_dtype is not None else kv_cache_dtype)


def configure_kv_cache(model, dtype="bf16", *, k_cache_dtype=None, v_cache_dtype=None):
    """Configure target GQA before allocation; validate both sides before mutating any module."""
    from .attention import GQAAttention
    k_dtype, v_dtype = resolve_kv_cache_dtypes(dtype, k_cache_dtype, v_cache_dtype)
    modules = [m for _, m in model.named_modules() if isinstance(m, GQAAttention)]
    for m in modules:
        cache_row_bytes(m.head_dim, k_dtype)
        cache_row_bytes(m.head_dim, v_dtype)
    for m in modules:
        m.kv_cache_dtype = dtype
        m.k_cache_dtype, m.v_cache_dtype = k_dtype, v_dtype
    return k_dtype, v_dtype


def rotate(x, inverse=False):
    y = np.asarray(x, dtype=np.float32).copy()
    y *= SIGNS2 if inverse else SIGNS1
    for h in (1, 2, 4, 8, 16, 32, 64):
        z = y.reshape(*y.shape[:-1], 128 // (2*h), 2*h)
        a, b = z[..., :h].copy(), z[..., h:].copy()
        z[..., :h], z[..., h:] = a+b, a-b
    return y * np.float32(128**-.5) * (SIGNS1 if inverse else SIGNS2)


def quantize(x, dtype):
    x = np.asarray(x, dtype=np.float32)
    width = cache_row_bytes(x.shape[-1], dtype)
    if dtype == "bf16":
        from ..formats.fp import f32_to_bf16
        return f32_to_bf16(x).view(np.uint8).reshape(*x.shape[:-1], width)
    block, size = {"q8_0": (32, 34), "turbo4": (128, 66), "q4_0": (32, 18)}[dtype]
    y = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    out = np.empty((*y.shape[:-1], size), np.uint8)
    if dtype == "q8_0":
        scale = np.max(np.abs(y), axis=-1) / np.float32(127)
        inv = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
        z = y * inv[..., None]
        codes = (np.sign(z) * np.floor(np.abs(z) + .5)).clip(-127, 127).astype(np.int8)
        out[..., 2:] = codes.view(np.uint8)
    elif dtype == "q4_0":
        # ggml chooses the first signed extremum, not the positive magnitude.
        index = np.argmax(np.abs(y), axis=-1)[..., None]
        peak = np.take_along_axis(y, index, axis=-1)[..., 0]
        peak = np.where(peak == 0, np.float32(0), peak)
        scale = peak / np.float32(-8)
        inv = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
        codes = (y * inv[..., None] + np.float32(8.5)).astype(np.int32).clip(0, 15).astype(np.uint8)
        # Low nibbles hold elements 0..15; high nibbles hold 16..31.
        out[..., 2:] = codes[..., :16] | (codes[..., 16:] << 4)
    else:
        norm = np.sqrt(np.sum(y*y, axis=-1))
        inv = np.divide(1, norm, out=np.zeros_like(norm), where=norm > 1e-10)
        codes = np.searchsorted(THRESHOLDS, rotate(y * inv[..., None]), side="right").astype(np.uint8)
        scale = norm / np.sqrt(np.sum(CENTROIDS[codes]**2, axis=-1))
        out[..., 2:] = codes[..., ::2] | (codes[..., 1::2] << 4)
    out[..., :2] = scale.astype('<f2')[..., None].view(np.uint8)
    return out.reshape(*x.shape[:-1], width)


def dequantize(packed, dtype, head_dim):
    packed = np.asarray(packed, dtype=np.uint8)
    width = cache_row_bytes(head_dim, dtype)
    if packed.shape[-1] != width:
        raise ValueError("Packed KV row has the wrong byte width")
    if dtype == "bf16":
        from ..formats.fp import bf16_to_f32
        return bf16_to_f32(packed.copy().view('<u2'))
    block, size = {"q8_0": (32, 34), "turbo4": (128, 66), "q4_0": (32, 18)}[dtype]
    b = packed.reshape(*packed.shape[:-1], head_dim // block, size)
    scale = b[..., :2].copy().view('<f2').astype(np.float32)
    if dtype == "q8_0":
        y = b[..., 2:].view(np.int8).astype(np.float32)
    elif dtype == "q4_0":
        codes = np.concatenate((b[..., 2:] & 15, b[..., 2:] >> 4), axis=-1)
        y = codes.astype(np.float32) - np.float32(8)
    else:
        codes = np.empty((*b.shape[:-1], block), np.uint8)
        codes[..., ::2], codes[..., 1::2] = b[..., 2:] & 15, b[..., 2:] >> 4
        y = rotate(CENTROIDS[codes], inverse=True)
    return (y * scale).reshape(*packed.shape[:-1], head_dim)
