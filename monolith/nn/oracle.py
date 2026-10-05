"""Torch helpers shared by the layer oracles (``Module.forward``). Imported lazily from inside ``forward`` so the
package stays torch-free at import time.

The rounding order follows the HF reference (transformers' ``modeling_qwen3_5.py``, Apache-2.0) wherever the fused
kernel is meant to reproduce it: BF16 at op boundaries, FP32 inside a fused op, FP32 recurrent state (design D11,
§5.9). Matmuls accumulate in FP32 and round once.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
F32 = torch.float32


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ wᵀ`` with FP32 accumulation and one rounding to BF16 (the GEMV kernel's semantics)."""
    return (x.to(F32) @ w.to(F32).t()).to(BF16)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, *, one_plus: bool = True, round_before_scale: bool = False) -> torch.Tensor:
    """RMSNorm with optional zero-centered weights and an intermediate input-dtype rounding."""
    xf = x.to(F32)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if round_before_scale:
        y = y.to(x.dtype).to(F32)
    scale = (1.0 + w.to(F32)) if one_plus else w.to(F32)
    return (y * scale).to(x.dtype)


def gated_rms_norm(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``RMSNormGated`` in the reference's rounding order: FP32 norm → BF16, ``w · n`` in BF16, ``· silu(z)`` in FP32,
    → BF16. ``o`` and ``z`` are BF16 ``[..., D]``; ``w`` is the stored (BF16-valued) weight."""
    of = o.to(F32)
    n = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)).to(o.dtype)
    n = w.to(o.dtype) * n
    return (n * F.silu(z.to(F32))).to(o.dtype)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) · up``: both BF16, computed in FP32 and rounded once (the gate|up epilogue)."""
    return (F.silu(gate.to(F32)) * up.to(F32)).to(gate.dtype)


def rope_tables(theta: float, rotary_dim: int, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin`` ``[len(positions), rotary_dim]`` in FP32 (``cat(freqs, freqs)`` layout, as ``rotate_half``
    expects); the caller casts to the activation dtype like the reference does."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=F32) / rotary_dim))
    freqs = positions.to(F32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial RoPE on the leading ``rotary_dim`` dims of ``x [T, heads, D]`` with ``cos/sin [T, rotary_dim]``
    (elementwise in the tensor's dtype, as the reference does)."""
    r = cos.shape[-1]
    c, s = cos[:, None, :], sin[:, None, :]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * c + rotate_half(xr) * s, xp], dim=-1)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def causal_conv1d(x: torch.Tensor, conv_state: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv over time with SiLU: ``x [T, C]`` (BF16), ``conv_state [C, W-1]`` (the last W-1 inputs),
    ``w [C, W]``. Returns ``(y [T, C] BF16, new_state [C, W-1])``. FP32 accumulate, one rounding, SiLU in FP32."""
    t, c = x.shape
    width = w.shape[-1]
    xs = torch.cat([conv_state.to(F32), x.to(F32).t()], dim=-1)             # [C, W-1+T]
    wf = w.to(F32)
    y = torch.zeros(c, t, dtype=F32)
    for j in range(width):
        y += wf[:, j: j + 1] * xs[:, j: j + t]
    y = y.to(x.dtype)                                                        # conv output rounded (reference: conv1d in bf16)
    y = F.silu(y.to(F32)).to(x.dtype)
    return y.t().contiguous(), xs[:, -(width - 1):].to(conv_state.dtype).contiguous()


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                     state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The recurrent gated delta rule (reference ``torch_recurrent_gated_delta_rule`` with ``use_qk_l2norm_in_kernel``).

    ``q, k [T, Hv, dk]`` (already broadcast to the value heads), ``v [T, Hv, dv]``, ``g, beta [T, Hv]`` (FP32 decay
    log and FP32 β), ``state [Hv, dk, dv]`` FP32. Returns ``(o [T, Hv, dv] in v's dtype, new state)``.
    """
    qf, kf, vf = q.to(F32), k.to(F32), v.to(F32)
    qf = l2norm(qf) / (qf.shape[-1] ** 0.5)
    kf = l2norm(kf)
    s = state.to(F32).clone()
    outs = []
    for i in range(qf.shape[0]):
        s = s * g[i].exp()[:, None, None]
        kv_mem = (s * kf[i][:, :, None]).sum(dim=-2)                          # [Hv, dv]
        delta = (vf[i] - kv_mem) * beta[i][:, None]
        s = s + kf[i][:, :, None] * delta[:, None, :]
        outs.append((s * qf[i][:, :, None]).sum(dim=-2))
    return torch.stack(outs).to(v.dtype), s


def attention_step(q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, k_cache: torch.Tensor,
                   v_cache: torch.Tensor, pos: int, scaling: float) -> torch.Tensor:
    """GQA attention of ``T`` new queries against the cache plus the ``T`` new keys (causal inside the step), in the
    reference's rounding order: scores BF16·scaling, softmax FP32 → BF16, ``P·V`` FP32 → BF16.

    ``q [T, H, D]``, ``k_new/v_new [T, Hkv, D]`` (BF16); the caches ``[ctx_max, Hkv, D]`` are updated in place at
    ``pos .. pos+T``. Returns ``[T, H, D]`` BF16.
    """
    t, h, d = q.shape
    hkv = k_new.shape[1]
    k_cache[pos: pos + t] = k_new
    v_cache[pos: pos + t] = v_new
    n = pos + t
    rep = h // hkv
    ks = k_cache[:n].to(F32).repeat_interleave(rep, dim=1)                     # [n, H, D]
    vs = v_cache[:n].to(F32).repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,nhd->htn", q.to(F32), ks).to(q.dtype) * scaling # BF16 like the reference matmul
    if t > 1:
        causal = torch.arange(n)[None, :] > (pos + torch.arange(t))[:, None]   # [t, n] keys after the query
        scores = scores.masked_fill(causal[None], float("-inf"))
    p = torch.softmax(scores.to(F32), dim=-1).to(q.dtype)
    out = torch.einsum("htn,nhd->thd", p.to(F32), vs).to(q.dtype)
    return out


@torch.no_grad()
def greedy_decode(model, prompt_ids, n_new: int, *, state=None, device=None):
    """Prefill the prompt in one step, then decode ``n_new`` greedy tokens one at a time through the model's oracle.
    Returns ``(generated ids, prefill hidden states, prefill logits)``."""
    ids = torch.as_tensor(prompt_ids, dtype=torch.int64, device=device)
    state = model.init_state(device=device) if state is None else state
    logits, hidden, state = model.forward(ids, state, 0)
    pos = ids.shape[0]
    out = []
    nxt = int(torch.argmax(logits[-1].to(F32)))
    for _ in range(n_new):
        out.append(nxt)
        step_logits, _, state = model.forward(torch.tensor([nxt], dtype=torch.int64, device=device), state, pos)
        pos += 1
        nxt = int(torch.argmax(step_logits[-1].to(F32)))
    return out, hidden, logits
