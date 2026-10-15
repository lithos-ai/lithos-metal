"""Gated GQA attention with per-head q/k RMSNorm and partial RoPE — stages 1–3 of a full-attention layer: the
``q|gate|k|v`` GEMV (input norm fused; q/k rows in the load-time head-dim permutation so the kernel applies
full-width rotary pairs), the ``gqa_decode`` mixer (norms, RoPE, KV append, online-softmax attention, ``σ(gate)``),
and ``o_proj`` with the residual add.

Semantics follow transformers' ``modeling_qwen3_5.py`` attention (Apache-2.0): ``q_proj`` produces ``[heads, 2, D]``
rows (query | gate per head), q/k norms scale by ``(1 + w)`` over the head dim, RoPE covers the first
``rotary_dim`` dims with ``rotate_half`` pairing, ``scaling = D^-0.5``, output ``attn · σ(gate)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from ..kernels import gqa_chunks_max
from ..packs.transforms import rope_head_perm

ATTN_CHUNK = 64          # keys per chunk of the attention core (the partial workspace is sized by it)
from .linear import Linear, Part
from .module import LowerContext, Module, StateEntry, WeightSpec


class GQAAttention(Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, rotary_dim: int, rope_theta: float,
                 eps: float, *, hf_prefix: str, prefix: str = "", max_context: int, gate: bool = True,
                 norm_one_plus: bool = True, qk_norm: bool = True, kv_cache_dtype: str = "bf16",
                 k_cache_dtype: str | None = None, v_cache_dtype: str | None = None) -> None:
        """``gate``: the ``[q | gate]`` projection and the ``σ(gate)`` output gate (the Qwen3.5 hybrid); ``norm_one_plus``:
        the per-head q/k RMSNorm scales by ``1 + w`` (Gemma-style) or by ``w`` (the standard RMSNorm of Qwen3)."""
        super().__init__(prefix=prefix)
        if heads % kv_heads:
            raise ValueError("GQAAttention: heads must be a multiple of kv_heads")
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.rotary_dim, self.rope_theta, self.eps, self.gate = rotary_dim, rope_theta, eps, gate
        self.norm_one_plus = norm_one_plus
        self.qk_norm = qk_norm
        from .kv_quant import cache_row_bytes, resolve_kv_cache_dtypes
        self.k_cache_dtype, self.v_cache_dtype = resolve_kv_cache_dtypes(kv_cache_dtype, k_cache_dtype, v_cache_dtype)
        for dtype in (self.k_cache_dtype, self.v_cache_dtype):
            cache_row_bytes(head_dim, dtype)
        self.kv_cache_dtype = kv_cache_dtype
        self.max_context = max_context
        self.hf_prefix = hf_prefix
        hd, kd = heads * head_dim, kv_heads * head_dim
        q_rows = heads * (2 if gate else 1) * head_dim
        self.qkv = Linear(hidden, [Part("q_proj", f"{hf_prefix}q_proj.weight", q_rows),
                                   Part("k_proj", f"{hf_prefix}k_proj.weight", kd),
                                   Part("v_proj", f"{hf_prefix}v_proj.weight", kd)],
                          prefix=f"{prefix}qkv.", row_perm=self._row_perm())
        self.o_proj = Linear(hd, [Part("o_proj", f"{hf_prefix}o_proj.weight", hidden)], prefix=f"{prefix}o_proj.",
                             epilogue="residual")

    # ---- layout ---------------------------------------------------------------------------------------------------
    def _row_perm(self) -> np.ndarray:
        """Stacked rows ``[q_proj | k_proj | v_proj]`` → ``[q (head-permuted) | k (head-permuted) | v | gate]``: the
        gate rows last, so the mixer core's rows and the gate rows are two block-aligned ranges of one slab."""
        d, hp = self.head_dim, rope_head_perm(self.head_dim, self.rotary_dim)
        stride = 2 * d if self.gate else d
        q = np.concatenate([h * stride + hp for h in range(self.heads)])
        base = self.heads * stride
        k = base + np.concatenate([h * d + hp for h in range(self.kv_heads)])
        v = base + self.kv_heads * d + np.arange(self.kv_heads * d)
        if self.gate:
            gate = np.concatenate([h * stride + d + np.arange(d) for h in range(self.heads)])
            return np.concatenate([q, k, v, gate])
        return np.concatenate([q, k, v])

    def kernel_segments(self) -> List[tuple]:
        """Row ranges of the packed ``qkv`` slab: ``(name, offset, rows)`` for q, k, v (and the gate)."""
        hd, kd = self.heads * self.head_dim, self.kv_heads * self.head_dim
        segs = [("q", 0, hd), ("k", hd, kd), ("v", hd + kd, kd)]
        if self.gate:
            segs.append(("gate", hd + 2 * kd, hd))
        return segs

    @property
    def core_rows(self) -> int:
        """Rows of the projection the attention core reads: q | k | v (the gate follows)."""
        return (self.heads + 2 * self.kv_heads) * self.head_dim

    def weight_map(self) -> Dict[str, WeightSpec]:
        if not self.qk_norm:
            return {}
        perm = rope_head_perm(self.head_dim, self.rotary_dim)
        tf = "one_plus" if self.norm_one_plus else "bf16_f32"          # the kernel multiplies by the stored scale as is
        return {"q_norm": WeightSpec(f"{self.hf_prefix}q_norm.weight", (self.head_dim,), "f32", transform=tf, aux=True, perm=perm),
                "k_norm": WeightSpec(f"{self.hf_prefix}k_norm.weight", (self.head_dim,), "f32", transform=tf, aux=True, perm=perm)}

    def state_entries(self, checkpoints: int = 1) -> List[StateEntry]:
        from .kv_quant import cache_row_bytes
        entries = []
        for name, dtype in (("k_cache", self.k_cache_dtype), ("v_cache", self.v_cache_dtype)):
            packed = dtype != "bf16"
            width = cache_row_bytes(self.head_dim, dtype) if packed else self.head_dim
            entries.append(StateEntry(f"{self.prefix}{name}", (self.max_context, self.kv_heads, width),
                                      DType.U8 if packed else DType.BF16))
        return entries


    # ---- oracle -----------------------------------------------------------------------------------------------
    def forward(self, x: Any, residual: Any, state: Dict[str, Any], pos: int) -> Any:
        return self.o_proj.forward(self.mix(self.qkv.forward(x), state, pos), residual)

    def mix(self, proj: Any, state: Dict[str, Any], pos: int) -> Any:
        """The mixer alone (what ``gqa_decode`` + ``gqa_merge`` compute): ``proj [T, N1]`` in checkpoint column
        order → the gated attention output ``[T, heads·D]`` BF16, caches advanced in place."""
        import torch

        from . import oracle

        t = proj.shape[0]
        d, hd, kd = self.head_dim, self.heads * self.head_dim, self.kv_heads * self.head_dim
        q_rows = self.heads * (2 if self.gate else 1) * d
        if self.gate:
            qg = proj[:, :q_rows].reshape(t, self.heads, 2 * d)
            q, gate = qg[..., :d], qg[..., d:]
        else:
            q, gate = proj[:, :q_rows].reshape(t, self.heads, d), None
        k = proj[:, q_rows: q_rows + kd].reshape(t, self.kv_heads, d)
        v = proj[:, q_rows + kd:].reshape(t, self.kv_heads, d)
        if self.qk_norm:
            q = oracle.rms_norm(q, self.param("q_norm"), self.eps, one_plus=self.norm_one_plus)
            k = oracle.rms_norm(k, self.param("k_norm"), self.eps, one_plus=self.norm_one_plus)
        cos, sin = self.rotary_tables(torch.arange(pos, pos + t))
        cos, sin = cos.to(proj.dtype), sin.to(proj.dtype)
        q = oracle.apply_partial_rope(q, cos, sin)
        k = oracle.apply_partial_rope(k, cos, sin)
        if self.k_cache_dtype == self.v_cache_dtype == "bf16":
            o = oracle.attention_step(q, k, v, state[f"{self.prefix}k_cache"], state[f"{self.prefix}v_cache"], pos, d ** -0.5)
        else:
            from .kv_quant import quantize, dequantize
            perm = rope_head_perm(d, self.rotary_dim)
            inv = np.argsort(perm)
            def side(value, name, dtype, is_key):
                cache = state[f"{self.prefix}{name}"]
                if dtype == "bf16":
                    history, fresh, stored = cache[:pos].float(), value.float(), value
                else:
                    old = dequantize(cache[:pos].cpu().numpy(), dtype, d)
                    raw = value.float().cpu().numpy()
                    packed = quantize(raw[..., perm] if is_key else raw, dtype)
                    decoded = dequantize(packed, dtype, d)
                    history = torch.from_numpy((old[..., inv] if is_key else old).copy()).to(proj.device)
                    fresh = torch.from_numpy((decoded[..., inv] if is_key else decoded).copy()).to(proj.device)
                    stored = torch.from_numpy(packed).to(cache.device)
                full = torch.zeros((pos+t, self.kv_heads, d), dtype=torch.float32, device=proj.device)
                full[:pos] = history
                cache[pos:pos+t] = stored
                return full, fresh
            fk, fresh_k = side(k, "k_cache", self.k_cache_dtype, True)
            fv, fresh_v = side(v, "v_cache", self.v_cache_dtype, False)
            o = oracle.attention_step(q, fresh_k, fresh_v, fk, fv, pos, d ** -0.5)
        o = o.to(proj.dtype)                                                # the reference's BF16 attention output
        if gate is not None:
            o = o * torch.sigmoid(gate)                                     # BF16 · bf16(σ(gate)) → BF16, as the reference
        return o.reshape(t, hd)

    def rotary_tables(self, positions):
        """Oracle tables; packages may override positional scaling to match their packed tables."""
        from . import oracle
        return oracle.rope_tables(self.rope_theta, self.rotary_dim, positions)

    # ---- IR ---------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext) -> Value:
        """q|k|v rows → the attention core (partials per KV chunk) ‖ the gate rows as an un-barriered sibling → the
        merge (design §5.12: the ALU-bound core is encoded first, the bus-bound gate GEMV hides under it)."""
        hd, kd, chunk = self.heads * self.head_dim, self.kv_heads * self.head_dim, ATTN_CHUNK
        proj = self.qkv.lower(g, h, norm=norm, rows=(0, self.core_rows)).value
        kc, vc = ctx.states[f"{self.prefix}k_cache"], ctx.states[f"{self.prefix}v_cache"]
        cos, sin = ctx.consts["rope_cos"], ctx.consts["rope_sin"]
        # Keep the binding ABI; disabled norm pointers are never read by the kernel.
        qn = self.const_value(g, f"{self.prefix}q_norm", (self.head_dim,), DType.F32) if self.qk_norm else cos
        kn = self.const_value(g, f"{self.prefix}k_norm", (self.head_dim,), DType.F32) if self.qk_norm else cos
        rep, n_chunks_max = self.heads // self.kv_heads, gqa_chunks_max(self.max_context, self.kv_heads, chunk)
        t = h.shape[0]
        part_o = g.value(f"{self.prefix}part_o", (t, self.kv_heads * n_chunks_max * rep * self.head_dim), DType.F32)
        part_md = g.value(f"{self.prefix}part_md", (t, self.kv_heads * n_chunks_max * rep * 2), DType.F32)
        g.op("gqa_decode", [proj, kc, vc, cos, sin, qn, kn], [part_o, part_md], domain=BlockDomain("heads", self.heads),
             klass=OpClass.MAP, updates=[kc.name, vc.name], heads=self.heads, kv_heads=self.kv_heads,
             head_dim=self.head_dim, rotary_dim=self.rotary_dim, eps=self.eps, scaling=self.head_dim ** -0.5,
             qk_norm=self.qk_norm, kv_cache_dtype=self.kv_cache_dtype, k_cache_dtype=self.k_cache_dtype, v_cache_dtype=self.v_cache_dtype, segments=[s for s in self.kernel_segments() if s[0] != "gate"], rope="permuted", chunk=chunk, **ctx.mixer_attrs)
        ins = [part_o, part_md]
        if self.gate:
            ins.append(self.qkv.lower(g, h, norm=norm, rows=(self.core_rows, hd), sibling=True).value)
        o = g.value(f"{self.prefix}attn", (t, hd), DType.BF16)
        g.op("gqa_merge", ins, [o], domain=BlockDomain("heads", self.heads), klass=OpClass.MAP, heads=self.heads,
             kv_heads=self.kv_heads, head_dim=self.head_dim, gate=self.gate, qk_norm=self.qk_norm, kv_cache_dtype=self.kv_cache_dtype, k_cache_dtype=self.k_cache_dtype, v_cache_dtype=self.v_cache_dtype, chunk=chunk, **ctx.mixer_attrs)
        return self.o_proj.lower(g, o, residual=h, name=f"{self.prefix}h").value
