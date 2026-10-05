"""The stochastic sampler's numpy reference (design D7): the same thresholds and the same counter-based Gumbel
noise as ``kernels/common/sample.metal``, so a kernel draw can be checked for exact equality, not just in distribution.

Logits are BF16 values; thresholds are logit values. ``top_k`` keeps every logit ≥ the k-th largest (ties kept, like
the HF warper); ``top_p`` keeps every logit ≥ the value at which the descending cumulative softmax mass of
``logits / temperature`` first reaches ``p``; ``min_p`` keeps logits ≥ ``max + temperature · log(min_p)``. The draw
is ``argmax (logit / temperature + g_i)`` over the kept logits, ``g_i = −log(−log(u_i))`` with ``u_i`` from a
splitmix64 hash of ``(seed, step, token position, index)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def thresholds(logits: np.ndarray, *, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, min_p: float = 0.0) -> float:
    """The logit threshold τ (−inf if nothing is enabled) for one position's BF16-valued ``logits``."""
    v = np.asarray(logits, dtype=np.float32)
    taus = []
    if top_k > 0:
        taus.append(float(np.sort(v)[::-1][min(top_k, v.size) - 1]))
    if 0.0 < top_p < 1.0:
        vals = np.unique(v)[::-1]                                       # distinct values, descending
        counts = np.array([(v == x).sum() for x in vals], dtype=np.float64)
        w = counts * np.exp((vals.astype(np.float64) - vals[0]) / temperature)
        cum = np.cumsum(w)
        idx = int(np.argmax(cum >= top_p * cum[-1]))
        taus.append(float(vals[idx]))
    if min_p > 0.0:
        taus.append(float(v.max() + temperature * np.log(min_p)))
    return max(taus) if taus else -np.inf


def gumbel_noise(seed: int, step: int, t: int, idx: np.ndarray) -> np.ndarray:
    """``−log(−log(u))`` for every index in ``idx``, ``u`` from the kernel's splitmix64 hash (float32 arithmetic)."""
    with np.errstate(over="ignore"):
        idx = np.asarray(idx, dtype=np.uint64)
        x = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
        x = (x + np.uint64(step) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(t) * np.uint64(0xC2B2AE3D27D4EB4F) + idx * np.uint64(0x165667B19E3779F9)) & M64
        x ^= x >> np.uint64(30); x = (x * np.uint64(0xBF58476D1CE4E5B9)) & M64
        x ^= x >> np.uint64(27); x = (x * np.uint64(0x94D049BB133111EB)) & M64
        x ^= x >> np.uint64(31)
    u = ((x >> np.uint64(41)) & np.uint64(0x7FFFFF)).astype(np.float32)          # 23 bits: u < 1 representable in float32
    u = (u + np.float32(0.5)) * np.float32(1.0 / 8388608.0)
    return -np.log(-np.log(u)).astype(np.float32)


def sample(logits: np.ndarray, *, seed: int, step: int, t: int = 0, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
           min_p: float = 0.0) -> int:
    """The token the kernel draws for one position (exact up to float32 summation of logit/T + noise)."""
    v = np.asarray(logits, dtype=np.float32)
    tau = thresholds(v, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
    keys = v * np.float32(1.0 / temperature) + gumbel_noise(seed, step, t, np.arange(v.size))
    keys = np.where(v >= tau, keys, -np.inf).astype(np.float32)
    return int(np.argmax(keys))


def warped_probabilities(logits: np.ndarray, *, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, min_p: float = 0.0) -> np.ndarray:
    """The distribution the draws follow: softmax of ``logits / temperature`` over the kept logits."""
    v = np.asarray(logits, dtype=np.float64)
    tau = thresholds(v.astype(np.float32), temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
    z = np.where(v >= tau, (v - v.max()) / temperature, -np.inf)
    p = np.exp(z)
    return p / p.sum()
