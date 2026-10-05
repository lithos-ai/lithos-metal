"""Load-time weight transforms, all expressed as row index arrays (``new_row -> old_row``) or small elementwise maps.
Model packages choose which to apply; nothing here knows a model.

``head_dim_perm`` is adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/modeling.py ``rope_head_perm``
@ 5beaed8 (Apache-2.0): the permutation that makes HF's partial-rotary ``rotate_half`` pairs land on a kernel's
full-width pairs, so partial RoPE costs nothing at run time.
"""

from __future__ import annotations

import numpy as np

from ..formats.fp import bf16_to_f32


def interleave_chunks(n_a: int, n_b: int, chunk: int) -> np.ndarray:
    """Rows of two stacked matrices ``A`` (rows ``0..n_a``) and ``B`` (rows ``n_a..n_a+n_b``) reordered so every
    ``2·chunk`` rows hold ``chunk`` rows of A followed by the same ``chunk`` rows of B — the ``gate/up`` layout that
    lets one block compute ``silu(gate)·up`` locally."""
    if n_a != n_b or n_a % chunk:
        raise ValueError(f"interleave_chunks: need n_a == n_b and a multiple of chunk, got {n_a}, {n_b}, {chunk}")
    a = np.arange(n_a).reshape(-1, chunk)
    b = np.arange(n_a, n_a + n_b).reshape(-1, chunk)
    return np.stack([a, b], axis=1).reshape(-1)


def rope_head_perm(head_dim: int, rotary_dim: int) -> np.ndarray:
    """Permutation of one head's dims (``perm[new] = old``) so the rotary pairs ``(i, i + rotary/2)`` land on the
    kernel's full-width pairs ``(i, i + head_dim/2)``; the non-rotary dims fill the remaining slots."""
    half, d2 = rotary_dim // 2, head_dim // 2
    if rotary_dim % 2 or rotary_dim > head_dim or head_dim % 2:
        raise ValueError("rope_head_perm: rotary_dim must be even and <= head_dim (even)")
    perm = list(range(0, half))
    perm += list(range(rotary_dim, rotary_dim + (d2 - half)))
    perm += list(range(half, rotary_dim))
    perm += list(range(rotary_dim + (d2 - half), head_dim))
    assert sorted(perm) == list(range(head_dim))
    return np.array(perm, dtype=np.int64)


def head_dim_perm(n_heads: int, head_dim: int, rotary_dim: int, *, stride: int | None = None, offset: int = 0) -> np.ndarray:
    """The rope permutation applied to every head of a ``[n_heads × stride, K]`` matrix whose head ``h`` occupies
    rows ``h·stride + offset .. + head_dim`` (``stride`` defaults to ``head_dim``; use ``2·head_dim`` for a
    ``[q | gate]``-interleaved projection). Returns a full row index array over ``n_heads × stride`` rows."""
    stride = stride or head_dim
    base = rope_head_perm(head_dim, rotary_dim)
    rows = np.arange(n_heads * stride)
    for h in range(n_heads):
        lo = h * stride + offset
        rows[lo: lo + head_dim] = lo + base
    return rows


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``compose(p, q)[new] = q[p[new]]``: apply ``q`` first, then ``p`` (both ``perm[new] = old``)."""
    return np.asarray(inner)[np.asarray(outer)]


def one_plus(w: np.ndarray, *, dtype_in: str = "BF16") -> np.ndarray:
    """Gemma/Qwen3.5 RMSNorm weight ``(1 + w)`` in float32 (the kernels fold it into the following GEMV), from the
    BF16-valued parameter (F32 inputs are rounded to BF16 first, as the reference model holds them)."""
    return (1.0 + bf16_round_f32(w, dtype_in=dtype_in)).astype(np.float32)


def bf16_round_f32(w: np.ndarray, *, dtype_in: str = "BF16") -> np.ndarray:
    """The value a BF16-loaded reference model holds, widened to float32: BF16 inputs exactly, F32 inputs rounded to
    BF16 first (the HF reference casts every floating parameter to its ``dtype``)."""
    from ..formats.fp import f32_to_bf16

    if dtype_in == "BF16":
        return bf16_to_f32(w).astype(np.float32)
    return bf16_to_f32(f32_to_bf16(np.asarray(w, dtype=np.float32))).astype(np.float32)


def neg_exp(w: np.ndarray, *, dtype_in: str = "BF16") -> np.ndarray:
    """``−exp(A_log)`` in float32 from the BF16-valued parameter: the Gated-DeltaNet decay coefficient the mixer
    multiplies by ``softplus(a + dt_bias)``."""
    return (-np.exp(bf16_round_f32(w, dtype_in=dtype_in))).astype(np.float32)
