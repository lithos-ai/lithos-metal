"""RMSNorm with the Gemma-style ``(1 + w)`` scale. In the step program the norm is never a dispatch of its own: the
statistic ``r = rsqrt(mean(h²) + eps)`` is a ``rmsnorm_stat`` op the fuse pass hoists into the producer's epilogue,
and the scaling ``x = h · r · (1 + w)`` is applied by the consuming GEMV on load (design §5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .module import Module, WeightSpec


@dataclass(frozen=True)
class NormInput:
    """What a GEMV needs to fuse an input norm: the per-token statistic and the ``(1 + w)`` weight constant."""

    stat: Value
    weight: Value
    eps: float


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float, hf_name: str, *, prefix: str = "", one_plus: bool = True, round_before_scale: bool = False) -> None:
        """``round_before_scale`` preserves separate BF16 normalization and weight-multiply operations."""
        super().__init__(prefix=prefix)
        self.round_before_scale = round_before_scale
        self.dim, self.eps, self.hf_name, self.one_plus = dim, eps, hf_name, one_plus

    def weight_map(self) -> Dict[str, WeightSpec]:
        return {"weight": WeightSpec(self.hf_name, (self.dim,), "f32", transform="one_plus" if self.one_plus else "bf16_f32", aux=True)}

    def forward(self, x: Any) -> Any:
        from . import oracle

        return oracle.rms_norm(x, self.param("weight"), self.eps, one_plus=self.one_plus, round_before_scale=self.round_before_scale)

    def lower(self, g: Graph, h: Value) -> NormInput:
        stat = g.value(f"{self.prefix}stat", (h.shape[0],), DType.F32)
        g.op("rmsnorm_stat", [h], [stat], domain=BlockDomain("span", self.dim), klass=OpClass.REDUCE, eps=self.eps,
             round_before_scale=self.round_before_scale)
        w = self.const_value(g, f"{self.prefix}weight", (self.dim,), DType.F32)
        return NormInput(stat, w, self.eps)
