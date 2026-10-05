"""RoPE tables for the pack (numpy, torch-free). Two layouts:

* ``hf``: ``[max_pos, rotary_dim]`` ``cat(freqs, freqs)`` — what the reference ``rotate_half`` consumes;
* ``permuted``: ``[max_pos, head_dim]`` in the load-time head-dim permutation (``packs.transforms.rope_head_perm``):
  the rotary frequencies sit at ``[0, rotary/2)`` and ``[D/2, D/2 + rotary/2)``, every other slot is the identity
  (cos = 1, sin = 0), so the kernel applies full-width ``(i, i + D/2)`` pairs and partial RoPE costs nothing.

adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/modeling.py ``build_rope_tables`` @ 5beaed8 (Apache-2.0).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def inv_freq(theta: float, rotary_dim: int) -> np.ndarray:
    return (1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))).astype(np.float32)


def scaled_inv_freq(theta: float, rotary_dim: int, parameters: Optional[dict] = None) -> Tuple[np.ndarray, float]:
    """Default or YaRN frequencies, including its cosine/sine amplitude.

    YaRN equations follow Hugging Face transformers 5.17's
    modeling_rope_utils._compute_yarn_parameters (Apache-2.0; third_party/NOTICE).
    Computed at packing time, with no runtime dependency on transformers.
    """
    p = parameters or {}
    kind = p.get("rope_type", "default")
    if kind == "default":
        return inv_freq(theta, rotary_dim), 1.0
    if kind != "yarn":
        raise ValueError(f"unsupported rope_type {kind!r}")
    factor = float(p.get("factor", 0))
    original = int(p.get("original_max_position_embeddings", 0))
    beta_fast, beta_slow = float(p.get("beta_fast", 32)), float(p.get("beta_slow", 1))
    if (not all(math.isfinite(x) for x in (theta, factor, beta_fast, beta_slow))
            or theta <= 1 or factor < 1 or original < 1 or not beta_fast >= beta_slow > 0
            or rotary_dim < 2 or rotary_dim % 2):
        raise ValueError("YaRN needs valid factor, original context, theta, beta range and even rotary dimension")
    bounds = [rotary_dim * math.log(original / (2 * math.pi * beta)) / (2 * math.log(theta))
              for beta in (beta_fast, beta_slow)]
    if p.get("truncate", True):
        bounds = [math.floor(bounds[0]), math.ceil(bounds[1])]
    low, high = max(bounds[0], 0), min(bounds[1], rotary_dim - 1)
    width = high - low if high != low else 0.001
    ramp = np.clip((np.arange(rotary_dim // 2, dtype=np.float32) - low) / width, 0, 1)
    frequencies = theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    extrapolation = 1 - ramp
    scaled = (1 / (factor * frequencies)) * (1 - extrapolation) + (1 / frequencies) * extrapolation
    amplitude = p.get("attention_factor")
    if amplitude is None:
        mscale = lambda x: 1.0 if factor <= 1 else 1 + 0.1 * float(x) * math.log(factor)
        amplitude = (mscale(p["mscale"]) / mscale(p["mscale_all_dim"])
                     if p.get("mscale") and p.get("mscale_all_dim") else mscale(1))
    if not math.isfinite(float(amplitude)) or float(amplitude) <= 0:
        raise ValueError("YaRN attention_factor must be finite and positive")
    return scaled.astype(np.float32), float(amplitude)


def rope_tables_hf(theta: float, rotary_dim: int, max_pos: int, *, parameters: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.arange(max_pos, dtype=np.float32)
    frequencies, amplitude = scaled_inv_freq(theta, rotary_dim, parameters)
    freqs = np.outer(pos, frequencies).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return (np.cos(emb) * amplitude).astype(np.float32), (np.sin(emb) * amplitude).astype(np.float32)


def rope_tables_permuted(theta: float, head_dim: int, rotary_dim: int, max_pos: int, *, parameters: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    pos = np.arange(max_pos, dtype=np.float32)
    frequencies, amplitude = scaled_inv_freq(theta, rotary_dim, parameters)
    freqs = np.outer(pos, frequencies).astype(np.float32)      # [P, half]
    full = np.zeros((max_pos, head_dim), dtype=np.float32)
    full[:, :half] = freqs
    full[:, head_dim // 2: head_dim // 2 + half] = freqs
    cos, sin = np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)
    if amplitude != 1:
        for table in (cos, sin):
            table[:, :half] *= amplitude
            table[:, head_dim // 2:head_dim // 2 + half] *= amplitude
    return cos, sin
