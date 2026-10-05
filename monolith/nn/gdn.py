"""Gated DeltaNet (linear attention) — stages 1–3 of a linear-attention layer: the ``in_proj_qkv|z|a|b`` GEMV
(input norm fused; one op per format group), the ``gdn_mixer`` (conv + SiLU, L2-norm, gates, delta rule, gated
RMSNorm — one block per value head) and ``out_proj`` with the residual add.

Semantics follow transformers' ``modeling_qwen3_5.py`` (Apache-2.0) in its rounding order: the projections are
BF16; the conv runs in BF16 with SiLU; ``β = σ(b)`` in BF16; ``g = −exp(A_log)·softplus(a + dt_bias)`` in FP32;
the recurrence (``torch_recurrent_gated_delta_rule`` with in-kernel q/k L2-norm and ``q / √dk``) keeps an FP32
state; the read-out is rounded to BF16 before the gated norm ``w · norm(o) · silu(z)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .linear import Linear, Part
from .module import LowerContext, Module, StateEntry, WeightSpec


class GatedDeltaNet(Module):
    def __init__(self, hidden: int, k_heads: int, v_heads: int, dk: int, dv: int, conv_width: int, eps: float, *,
                 hf_prefix: str, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        if v_heads % k_heads:
            raise ValueError("GatedDeltaNet: v_heads must be a multiple of k_heads")
        self.hidden, self.k_heads, self.v_heads, self.dk, self.dv = hidden, k_heads, v_heads, dk, dv
        self.conv_width, self.eps, self.hf_prefix = conv_width, eps, hf_prefix
        self.key_dim, self.value_dim = k_heads * dk, v_heads * dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # z first: the gate projection is a block-aligned row range at the head of the slab, emitted as the mixer
        # core's un-barriered sibling (design §5.12); the core reads the rows after it
        self.in_proj = Linear(hidden, [Part("in_proj_z", f"{hf_prefix}in_proj_z.weight", self.value_dim),
                                       Part("in_proj_qkv", f"{hf_prefix}in_proj_qkv.weight", self.conv_dim),
                                       Part("in_proj_a", f"{hf_prefix}in_proj_a.weight", v_heads),
                                       Part("in_proj_b", f"{hf_prefix}in_proj_b.weight", v_heads)],
                              prefix=f"{prefix}in_proj.")
        self.out_proj = Linear(self.value_dim, [Part("out_proj", f"{hf_prefix}out_proj.weight", hidden)],
                               prefix=f"{prefix}out_proj.", epilogue="residual")

    def weight_map(self) -> Dict[str, WeightSpec]:
        p = self.hf_prefix
        return {"conv_w": WeightSpec(f"{p}conv1d.weight", (self.conv_dim, 1, self.conv_width), "bf16", aux=True),
                "a_log": WeightSpec(f"{p}A_log", (self.v_heads,), "f32", transform="neg_exp", aux=True),
                "dt_bias": WeightSpec(f"{p}dt_bias", (self.v_heads,), "f32", transform="bf16_f32", aux=True),
                "norm_w": WeightSpec(f"{p}norm.weight", (self.dv,), "f32", transform="bf16_f32", aux=True)}

    def kernel_segments(self) -> List[tuple]:
        """Column ranges of the (stacked) projection in part order: ``(name, offset, cols)`` for z, q, k, v, a, b."""
        kd, vd, hv = self.key_dim, self.value_dim, self.v_heads
        return [("z", 0, vd), ("q", vd, kd), ("k", vd + kd, kd), ("v", vd + 2 * kd, vd),
                ("a", vd + self.conv_dim, hv), ("b", vd + self.conv_dim + hv, hv)]

    def state_entries(self, checkpoints: int = 2) -> List[StateEntry]:
        """Two slots by step parity (the step's pass reads one and writes the other; the speculative commit pass
        rewrites the written one with the accepted prefix)."""
        return [StateEntry(f"{self.prefix}conv_state", (self.conv_dim, self.conv_width - 1), DType.BF16, checkpoints),
                StateEntry(f"{self.prefix}rec_state", (self.v_heads, self.dk, self.dv), DType.F32, checkpoints)]

    # ---- oracle -----------------------------------------------------------------------------------------------
    def forward(self, x: Any, residual: Any, state: Dict[str, Any], pos: int = 0) -> Any:
        return self.out_proj.forward(self.mix(self.in_proj.forward(x), state), residual)

    def mix(self, proj: Any, state: Dict[str, Any]) -> Any:
        """The mixer alone (what ``gdn_mixer`` + ``gdn_norm`` compute): ``proj [T, N1]`` in the projection's part
        order ``z | qkv | a | b`` → the gated, normalized output ``[T, v_heads·dv]`` BF16; conv and recurrent states
        advanced in place."""
        import torch
        import torch.nn.functional as F

        from . import oracle

        t = proj.shape[0]
        kd, vd, hv = self.key_dim, self.value_dim, self.v_heads
        z, qkv = proj[:, :vd], proj[:, vd: vd + self.conv_dim]
        a, b = proj[:, vd + self.conv_dim: vd + self.conv_dim + hv], proj[:, vd + self.conv_dim + hv:]
        conv_name, rec_name = f"{self.prefix}conv_state", f"{self.prefix}rec_state"
        qkv, new_conv = oracle.causal_conv1d(qkv, state[conv_name], self.param("conv_w").reshape(self.conv_dim, self.conv_width))
        q = qkv[:, :kd].reshape(t, self.k_heads, self.dk)
        k = qkv[:, kd: 2 * kd].reshape(t, self.k_heads, self.dk)
        v = qkv[:, 2 * kd:].reshape(t, hv, self.dv)
        beta = torch.sigmoid(b).to(torch.float32)                                     # σ in BF16 like the reference
        # the pack's aux ``a_log`` holds −exp(A_log) (transform neg_exp); the oracle computes it from the raw parameter
        neg_exp_a = -torch.exp(self.param("a_log").to(torch.float32))
        g = neg_exp_a * F.softplus(a.to(torch.float32) + self.param("dt_bias").to(torch.float32))
        rep = hv // self.k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        o, new_rec = oracle.gated_delta_rule(q, k, v, g, beta, state[rec_name])       # o BF16 [T, Hv, dv]
        o = oracle.gated_rms_norm(o.reshape(t * hv, self.dv), z.reshape(t * hv, self.dv), self.param("norm_w"), self.eps)
        state[conv_name], state[rec_name] = new_conv, new_rec
        return o.reshape(t, vd)

    # ---- IR ---------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext) -> Value:
        """qkv|a|b rows → the mixer core (the FP32 read-out) ‖ the z rows as an un-barriered sibling → the gated norm
        (design §5.12: the ALU-bound core is encoded first, the bus-bound gate GEMV hides under it)."""
        if ctx.mixer_attrs:
            raise NotImplementedError(f"GatedDeltaNet {self.prefix}: the GDN kernels have no LM-drafter modes ({sorted(ctx.mixer_attrs)}); "
                                      "an LM drafter must be an attention-only model")
        vd = self.value_dim
        n0 = self.in_proj.slab_groups()[0].rows                          # z | qkv (| a | b when one format)
        main = self.in_proj.lower(g, h, norm=norm, rows=(vd, n0 - vd))
        cs, rs = ctx.states[f"{self.prefix}conv_state"], ctx.states[f"{self.prefix}rec_state"]
        conv_w = self.const_value(g, f"{self.prefix}conv_w", (self.conv_dim, self.conv_width), DType.BF16)
        nea = self.const_value(g, f"{self.prefix}a_log", (self.v_heads,), DType.F32)      # holds −exp(A_log)
        dtb = self.const_value(g, f"{self.prefix}dt_bias", (self.v_heads,), DType.F32)
        t = h.shape[0]
        o_part = g.value(f"{self.prefix}o_part", (t, vd), DType.F32)
        g.op("gdn_mixer", [*main.values, cs, rs, conv_w, nea, dtb], [o_part], domain=BlockDomain("heads", self.v_heads),
             klass=OpClass.MAP, updates=[cs.name, rs.name], k_heads=self.k_heads, v_heads=self.v_heads, dk=self.dk,
             dv=self.dv, conv_width=self.conv_width, eps=self.eps, segments=self.kernel_segments(),
             proj_segments=main.segments, commit_kind="gdn_commit")
        z = self.in_proj.lower(g, h, norm=norm, rows=(0, vd), groups=[0], sibling=True).value
        nw = self.const_value(g, f"{self.prefix}norm_w", (self.dv,), DType.F32)
        o = g.value(f"{self.prefix}mix", (t, vd), DType.BF16)
        g.op("gdn_norm", [o_part, z, nw], [o], domain=BlockDomain("heads", self.v_heads), klass=OpClass.MAP,
             v_heads=self.v_heads, dv=self.dv, eps=self.eps)
        return self.out_proj.lower(g, o, residual=h, name=f"{self.prefix}h").value
