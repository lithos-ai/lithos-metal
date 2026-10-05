"""``lm_head``: the vocabulary GEMV with the final norm fused on its input; output BF16 logits (the reference's
``lm_head`` dtype, so the argmax sees the same roundings). Tied models reuse the embedding slab."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .embedding import Embedding
from .linear import SlabGroup
from .module import LowerContext, Module, WeightSpec


class LMHead(Module):
    def __init__(self, hidden: int, vocab: int, *, hf_name: Optional[str] = None, tied: Optional[Embedding] = None,
                 prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        if (hf_name is None) == (tied is None):
            raise ValueError("LMHead: give either hf_name (own weight) or tied (an Embedding)")
        self.hidden, self.vocab, self.hf_name = hidden, vocab, hf_name
        self._tied = tied          # underscore: not a child (its weight belongs to the embedding)

    @property
    def tied(self) -> Optional[Embedding]:
        return self._tied

    @property
    def slab_name(self) -> str:
        return self._tied.slab_name if self._tied is not None else f"{self.prefix}weight"

    def weight_map(self) -> Dict[str, WeightSpec]:
        if self._tied is not None:
            return {}
        return {"weight": WeightSpec(self.hf_name, (self.vocab, self.hidden), self.format_of("weight"), slab=self.slab_name)}

    def slab_groups(self) -> List[SlabGroup]:
        if self._tied is not None:
            return []
        spec = self.weight_map()["weight"]
        return [SlabGroup(self.slab_name, spec.format, [("weight", spec)], None, self.vocab, self.hidden)]

    def weight(self) -> Any:
        return self._tied.param("weight") if self._tied is not None else self.param("weight")

    def forward(self, x: Any) -> Any:
        from . import oracle

        return oracle.linear(x, self.weight())

    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext, *, name: str = "logits") -> Value:
        """``norm`` = the final norm fused on the input, or None when ``h`` is already normalized (a draft block)."""
        fmt = self._tied.format_of("weight") if self._tied is not None else self.format_of("weight")
        w = self.weight_value(g, self.slab_name, (self.vocab, self.hidden), fmt)
        logits = g.value(name, (h.shape[0], self.vocab), DType.BF16)
        ins = [h, w] + ([norm.stat, norm.weight] if norm is not None else [])
        attrs = dict(norm=norm is not None, out="bf16", format=fmt)
        if norm is not None:
            attrs["eps"] = norm.eps
        g.op("lm_head", ins, [logits], domain=BlockDomain("rows", self.vocab), klass=OpClass.MAP, **attrs)
        return logits
