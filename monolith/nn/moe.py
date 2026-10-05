"""A sparse mixture-of-experts MLP (design §5.11): a router GEMV, the top-k select, the experts' gate|up and down
projections as two row-stacked slabs streamed through ``moe_gemv`` (the GEMV kernel's pairs mode — the work items
are (token, slot, block) and the slab block comes from the router's ids, so a step's expert traffic is exactly the
chosen experts' bytes, no re-encode, no CPU), and the weighted sum with an optional shared expert. The oracle
mirrors the reference block: softmax over the experts in FP32, top-k (optionally renormalized), weights cast to the
hidden dtype, each expert a BF16 linear chain; the sum is accumulated in FP32 and rounded once (the reference
accumulates BF16 products in BF16 — the fused-epilogue deviation of design §5.9)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from ..packs.transforms import interleave_chunks
from .linear import Linear, Part
from .module import LowerContext, Module


class Experts(Module):
    """``n_experts`` copies of one projection row-stacked in a single slab: expert ``e``'s rows are
    ``[e · rows_per_expert, (e + 1) · rows_per_expert)``; with ``epilogue="silu_mul"`` each expert's gate|up rows are
    chunk-interleaved like ``GatedMLP``'s. ``parts`` = ``(local, hf_suffix, rows)`` per expert, ``hf_template`` the
    checkpoint prefix with ``{e}`` for the expert index (``…mlp.experts.{e}.``). The epilogue is applied by
    ``moe_gemv``, not by the slab's ``Linear``."""

    def __init__(self, in_features: int, n_experts: int, parts: Sequence[Tuple[str, str, int]], *, hf_template: str, prefix: str = "",
                 epilogue: Optional[str] = None, chunk: Optional[int] = None) -> None:
        super().__init__(prefix=prefix)
        self.n_experts, self.k = n_experts, in_features
        self.part_specs = list(parts)
        self.rows_per_expert = sum(r for _, _, r in parts)
        self.epilogue, self.chunk = epilogue, chunk
        perm = None
        if epilogue == "silu_mul":
            if len(parts) != 2 or parts[0][2] != parts[1][2] or not chunk:
                raise ValueError("Experts: silu_mul needs exactly two equal parts (gate, up) per expert and a chunk size")
            inter = parts[0][2]
            one = np.asarray(interleave_chunks(inter, inter, chunk), dtype=np.int64)
            perm = np.concatenate([one + e * 2 * inter for e in range(n_experts)])
        elif epilogue is not None:
            raise ValueError(f"Experts: unsupported epilogue {epilogue!r}")
        self.slab = Linear(in_features, [Part(f"e{e}.{local}", hf_template.format(e=e) + suffix, rows) for e in range(n_experts) for local, suffix, rows in parts],
                           prefix=prefix, row_perm=perm)

    @property
    def n_out(self) -> int:
        return self.rows_per_expert // 2 if self.epilogue == "silu_mul" else self.rows_per_expert

    def forward_expert(self, e: int, x: Any) -> Any:
        """Expert ``e`` on ``x [n, K]`` as the reference computes it (a BF16 linear per part, FP32 accumulation)."""
        import torch

        w = torch.cat([self.slab.param(f"e{e}.{local}") for local, _, _ in self.part_specs], dim=0)
        acc = x.to(torch.float32) @ w.to(torch.float32).t()
        if self.epilogue == "silu_mul":
            half = self.rows_per_expert // 2
            gate, up = acc[:, :half].to(x.dtype), acc[:, half:].to(x.dtype)                 # two BF16 linears, then silu·mul
            return (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(x.dtype)
        return acc.to(x.dtype)

    def weight_value(self, g: Graph, *_: Any) -> Value:  # type: ignore[override]
        grp = self.slab.slab_groups()
        if len(grp) != 1:
            raise ValueError(f"Experts {self.prefix}: the experts must share one storage format")
        return self.slab.weight_value(g, grp[0].name, (grp[0].rows, grp[0].k), grp[0].format)

    def lower(self, g: Graph, x: Value, ids: Value, top_k: int, *, x_per_slot: bool, name: str) -> Value:
        grp = self.slab.slab_groups()[0]
        w = self.weight_value(g)
        t = ids.shape[0]
        y = g.value(name, (t, top_k * self.n_out), DType.BF16)
        attrs: Dict[str, Any] = dict(top_k=top_k, expert_rows=self.rows_per_expert, epilogue=self.epilogue, format=grp.format, x_per_slot=x_per_slot)
        if self.epilogue == "silu_mul":
            attrs["chunk"] = self.chunk
        g.op("moe_gemv", [x, w, ids], [y], domain=BlockDomain("rows", t), klass=OpClass.MAP, **attrs)
        return y


class SparseMoE(Module):
    """The sparse MLP: ``router`` (``[E, hidden]`` BF16) → ``moe_route`` → the experts' ``gate|up`` (silu·mul) and
    ``down`` slabs → ``moe_combine`` (+ the shared expert ``σ(shared_gate(x)) · down(silu(gate(x))·up(x))`` when
    ``shared_intermediate`` is given) (+ the residual). ``renorm`` = the reference's ``norm_topk_prob``."""

    def __init__(self, hidden: int, n_experts: int, top_k: int, intermediate: int, *, hf_prefix: str, prefix: str = "", chunk: int = 8,
                 renorm: bool = True, shared_intermediate: Optional[int] = None) -> None:
        super().__init__(prefix=prefix)
        if not 1 <= top_k <= min(16, n_experts):
            raise ValueError(f"SparseMoE: top_k must be in 1..min(16, {n_experts}), got {top_k}")
        self.hidden, self.n_experts, self.top_k, self.intermediate, self.renorm = hidden, n_experts, top_k, intermediate, bool(renorm)
        self.router = Linear(hidden, [Part("gate", f"{hf_prefix}gate.weight", n_experts)], prefix=f"{prefix}router.")
        self.gate_up = Experts(hidden, n_experts, [("gate_proj", "gate_proj.weight", intermediate), ("up_proj", "up_proj.weight", intermediate)],
                               hf_template=f"{hf_prefix}experts.{{e}}.", prefix=f"{prefix}experts_gate_up.", epilogue="silu_mul", chunk=chunk)
        self.down = Experts(intermediate, n_experts, [("down_proj", "down_proj.weight", hidden)], hf_template=f"{hf_prefix}experts.{{e}}.",
                            prefix=f"{prefix}experts_down.")
        self.shared_intermediate = shared_intermediate
        if shared_intermediate:
            si = shared_intermediate
            self.shared_gate_up = Linear(hidden, [Part("gate_proj", f"{hf_prefix}shared_expert.gate_proj.weight", si),
                                                  Part("up_proj", f"{hf_prefix}shared_expert.up_proj.weight", si)],
                                         prefix=f"{prefix}shared_gate_up.", row_perm=interleave_chunks(si, si, chunk), epilogue="silu_mul", chunk=chunk)
            self.shared_down = Linear(si, [Part("down_proj", f"{hf_prefix}shared_expert.down_proj.weight", hidden)], prefix=f"{prefix}shared_down.")
            self.shared_gate = Linear(hidden, [Part("shared_expert_gate", f"{hf_prefix}shared_expert_gate.weight", 1)], prefix=f"{prefix}shared_gate.")

    # ---- oracle -------------------------------------------------------------------------------------------------
    def route(self, x: Any) -> Tuple[Any, Any]:
        """``(ids [T, k] int64, weights [T, k] in x's dtype)`` as the reference routes: softmax in FP32 over the BF16
        logits, top-k (ties to the lowest index), optionally renormalized, cast to the hidden dtype."""
        import torch

        logits = self.router.forward(x)                                                  # BF16 [T, E]
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
        if self.renorm:
            top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        return top_i, top_v.to(x.dtype)

    def forward(self, x: Any, residual: Any) -> Any:
        import torch

        ids, w = self.route(x)
        out = torch.zeros(x.shape[0], self.hidden, dtype=torch.float32)
        for t in range(x.shape[0]):
            for j in range(self.top_k):
                e = int(ids[t, j])
                a = self.gate_up.forward_expert(e, x[t: t + 1])
                d = self.down.forward_expert(e, a)
                out[t] += w[t, j].to(torch.float32) * d[0].to(torch.float32)
        if self.shared_intermediate:
            s = self.shared_down.forward(self.shared_gate_up.forward(x)).to(torch.float32)
            gate = torch.sigmoid(self.shared_gate.forward(x).to(torch.float32)).to(x.dtype).to(torch.float32)   # σ(gate) in the hidden dtype
            out = out + gate * s
        return (out + residual.to(torch.float32)).to(x.dtype)

    # ---- IR ---------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext) -> Value:
        t = h.shape[0]
        p = self.prefix
        logits = self.router.lower(g, h, norm=norm).value                              # the norm fused on the router's input
        xn = g.value(f"{p}xn", h.shape, DType.BF16)                                      # the normalized input the experts read
        g.op("norm_apply", [h, norm.stat, norm.weight], [xn], domain=BlockDomain("span", self.hidden), klass=OpClass.MAP, eps=norm.eps)
        ids = g.value(f"{p}ids", (t, self.top_k), DType.I32)
        weights = g.value(f"{p}weights", (t, self.top_k), DType.F32)
        g.op("moe_route", [logits], [ids, weights], domain=BlockDomain("rows", t), klass=OpClass.MAP, n_experts=self.n_experts, top_k=self.top_k,
             renorm=self.renorm)
        act = self.gate_up.lower(g, xn, ids, self.top_k, x_per_slot=False, name=f"{p}act")
        hk = self.down.lower(g, act, ids, self.top_k, x_per_slot=True, name=f"{p}hk")
        ins: List[Value] = [hk, weights]
        attrs: Dict[str, Any] = dict(top_k=self.top_k, has_shared=False, has_residual=True)
        if self.shared_intermediate:
            s_act = self.shared_gate_up.lower(g, h, norm=norm).value
            s_out = self.shared_down.lower(g, s_act, name=f"{p}shared").value
            s_gate = self.shared_gate.lower(g, h, norm=norm, name=f"{p}shared_gate").value
            ins += [s_out, s_gate]
            attrs["has_shared"] = True
        ins.append(h)
        out = g.value(f"{p}h", (t, self.hidden), DType.BF16)
        g.op("moe_combine", ins, [out], domain=BlockDomain("rows", t), klass=OpClass.MAP, **attrs)
        return out
