"""Gated MLP ``down(silu(gate(x)) · up(x))``: stages 4 and 5 of a layer — the ``gate|up`` GEMV with the post-attention
norm fused on its input and ``silu·mul`` in its epilogue, then ``down`` with the residual add."""

from __future__ import annotations

from typing import Any

from ..core.ir import Graph, Value
from ..packs.transforms import interleave_chunks
from .linear import Linear, Part
from .module import LowerContext, Module


class GatedMLP(Module):
    def __init__(self, hidden: int, intermediate: int, *, hf_prefix: str, prefix: str = "", chunk: int = 8) -> None:
        super().__init__(prefix=prefix)
        self.hidden, self.intermediate = hidden, intermediate
        self.gate_up = Linear(hidden, [Part("gate_proj", f"{hf_prefix}gate_proj.weight", intermediate),
                                       Part("up_proj", f"{hf_prefix}up_proj.weight", intermediate)],
                              prefix=f"{prefix}gate_up.", row_perm=interleave_chunks(intermediate, intermediate, chunk),
                              epilogue="silu_mul", chunk=chunk)
        self.down = Linear(intermediate, [Part("down_proj", f"{hf_prefix}down_proj.weight", hidden)],
                           prefix=f"{prefix}down.", epilogue="residual")

    def forward(self, x: Any, residual: Any, state=None, pos: int = 0) -> Any:
        return self.down.forward(self.gate_up.forward(x), residual)

    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext) -> Value:
        act = self.gate_up.lower(g, h, norm=norm).value
        return self.down.lower(g, act, residual=h, name=f"{self.prefix}h").value
