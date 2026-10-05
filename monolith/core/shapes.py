"""Shapes symbolic in the per-step token count ``T`` and the context length ``ctx``.

The step program is compiled once for ``T_max`` and replayed with the actual ``T`` read from ``StepState``
(design §5.7), so a shape may carry a symbol; the memory planner binds it to its maximum, kernels bind it at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple, Union


@dataclass(frozen=True)
class Sym:
    """A named symbolic dimension."""

    name: str

    def __repr__(self) -> str:
        return self.name


Dim = Union[int, Sym]
Shape = Tuple[Dim, ...]

T = Sym("T")        # tokens in this step: 1 for plain decode, 1 + L when verifying L drafts, ≤ T_max
N_INJ = Sym("n_inject")   # rows the drafter injects this step (the committed positions: accepted + 1, or a prefill chunk), ≤ T_max
N_CHAIN = Sym("n_chain")  # an LM drafter's chain rows: 1 in a step that drafts, 0 in a prefill chunk
N_FIRST = Sym("n_first")  # an LM drafter's first chain step: the ingest rows and the anchor (n_inject + n_chain), ≤ T_max + 1
CTX = Sym("ctx")    # committed context length (KV length)


def step_bindings(t_max: int) -> "dict":
    """The memory planner's bindings: every per-step row symbol at its maximum."""
    return {T: t_max, N_INJ: t_max, N_CHAIN: 1, N_FIRST: t_max + 1}


def is_static(shape: Iterable[Dim]) -> bool:
    return all(isinstance(d, int) for d in shape)


def bind(shape: Iterable[Dim], bindings: Mapping[Sym, int]) -> Tuple[int, ...]:
    """Replace every symbol by its binding; raises ``KeyError`` for an unbound symbol."""
    out = []
    for d in shape:
        if isinstance(d, Sym):
            if d not in bindings:
                raise KeyError(f"unbound symbolic dimension {d.name}")
            out.append(int(bindings[d]))
        else:
            out.append(int(d))
    return tuple(out)


def numel(shape: Iterable[Dim], bindings: Mapping[Sym, int] | None = None) -> int:
    n = 1
    for d in bind(shape, bindings or {}):
        n *= d
    return n
