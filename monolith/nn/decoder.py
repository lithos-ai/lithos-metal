"""A pre-norm residual decoder layer: ``h += mixer(norm1(h))``, ``h += mlp(norm2(h))``. The mixer is any module
with the ``(x, residual, state, pos)`` oracle and ``(g, h, norm, ctx)`` lowering signature (attention or GDN);
each norm lowers to a statistic the following GEMV fuses."""

from __future__ import annotations

from typing import Any, Dict

from ..core.ir import Graph, Value
from .module import LowerContext, Module
from .norm import RMSNorm


class DecoderLayer(Module):
    def __init__(self, index: int, input_norm: RMSNorm, mixer: Module, post_norm: RMSNorm, mlp: Module, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.index = index
        self.input_norm, self.mixer, self.post_norm, self.mlp = input_norm, mixer, post_norm, mlp

    def forward(self, h: Any, state: Dict[str, Any], pos: int) -> Any:
        h = self.mixer.forward(self.input_norm.forward(h), h, state, pos)
        return self.mlp.forward(self.post_norm.forward(h), h)

    def lower(self, g: Graph, h: Value, ctx: LowerContext) -> Value:
        h = self.mixer.lower(g, h, self.input_norm.lower(g, h), ctx)
        return self.mlp.lower(g, h, self.post_norm.lower(g, h), ctx)
