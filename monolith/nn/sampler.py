"""Sampling on the GPU (design D7). ``GreedySampler``: argmax of the BF16 logits widened to FP32, ties → lowest
index, one token per step position. ``StochasticSampler``: temperature, top-k, top-p and min-p thresholds and a
Gumbel-max draw with a counter-based RNG keyed by (seed, step, position, index) — reproducible bit for bit, no CPU
sync; its numpy reference is :mod:`monolith.nn.sampling_ref`."""

from __future__ import annotations

from typing import Any

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .module import LowerContext, Module


class GreedySampler(Module):
    def forward(self, logits: Any) -> Any:
        import torch

        return torch.argmax(logits.to(torch.float32), dim=-1)

    def lower(self, g: Graph, logits: Value, ctx: LowerContext) -> Value:
        token = g.value("token", (logits.shape[0],), DType.I32)
        g.op("argmax", [logits], [token], domain=BlockDomain("span", logits.shape[1]), klass=OpClass.REDUCE)
        return token


class StochasticSampler(Module):
    def __init__(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, min_p: float = 0.0, seed: int = 0, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        if temperature <= 0:
            raise ValueError("StochasticSampler: temperature must be positive; use GreedySampler for greedy decoding")
        self.temperature, self.top_k, self.top_p, self.min_p, self.seed = float(temperature), int(top_k), float(top_p), float(min_p), int(seed)

    def forward(self, logits: Any, step: int = 0) -> Any:
        """The draws for every position of ``logits [T, V]`` (the exact kernel reference, on the CPU)."""
        import numpy as np
        import torch

        from . import sampling_ref

        lg = logits.to(torch.float32).cpu().numpy()
        out = [sampling_ref.sample(lg[t], seed=self.seed, step=step, t=t, temperature=self.temperature, top_k=self.top_k,
                                   top_p=self.top_p, min_p=self.min_p) for t in range(lg.shape[0])]
        return torch.tensor(out, dtype=torch.int64)

    def lower(self, g: Graph, logits: Value, ctx: LowerContext) -> Value:
        token = g.value("token", (logits.shape[0],), DType.I32)
        g.op("sample", [logits], [token], domain=BlockDomain("span", logits.shape[1]), klass=OpClass.REDUCE,
             temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, min_p=self.min_p, seed=self.seed)
        return token
