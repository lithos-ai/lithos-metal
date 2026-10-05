"""Token embedding: a ``[vocab, hidden]`` table gathered by token id. When the model ties it with ``lm_head`` the
table is packed as a slab (the GEMV streams it) and the ``embed`` op gathers rows from the pack layout."""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .linear import SlabGroup
from .module import LowerContext, Module, WeightSpec


class Embedding(Module):
    def __init__(self, vocab: int, hidden: int, hf_name: str, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.vocab, self.hidden, self.hf_name = vocab, hidden, hf_name

    @property
    def slab_name(self) -> str:
        return f"{self.prefix}weight"

    def weight_map(self) -> Dict[str, WeightSpec]:
        return {"weight": WeightSpec(self.hf_name, (self.vocab, self.hidden), self.format_of("weight"), slab=self.slab_name)}

    def slab_groups(self) -> List[SlabGroup]:
        spec = self.weight_map()["weight"]
        return [SlabGroup(self.slab_name, spec.format, [("weight", spec)], None, self.vocab, self.hidden)]

    def forward(self, ids: Any) -> Any:
        return self.param("weight")[ids]

    def lower(self, g: Graph, tokens: Value, ctx: LowerContext) -> Value:
        w = self.weight_value(g, self.slab_name, (self.vocab, self.hidden), self.format_of("weight"))
        h = g.value(f"{self.prefix}h", (ctx.t, self.hidden), DType.BF16)
        g.op("embed", [tokens, w], [h], domain=BlockDomain("rows", ctx.t), klass=OpClass.MAP, packed=True)
        return h
