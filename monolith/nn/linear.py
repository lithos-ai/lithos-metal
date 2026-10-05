"""``Linear``: one or more checkpoint matrices row-stacked into the slabs one input feeds — ``q|gate|k|v``,
``in_proj_qkv|z|a|b``, ``gate|up`` — with the GEMV fusions of design §5.6 as epilogues.

A stacked projection is one ``gemv`` op per *format group*: consecutive parts that share a storage format form one
slab (the 27B's FP8 ``qkv|z`` and BF16 ``a|b`` become two ops that read the same normalized input, un-barriered
siblings). Row permutations (head-dim permutation for RoPE, ``gate/up`` chunk interleaving) apply to a slab's
stacked rows and therefore require a single format group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .module import Module, WeightSpec

EPILOGUES = (None, "residual", "silu_mul")


@dataclass(frozen=True)
class Part:
    """One checkpoint matrix ``[rows, K]`` inside a stacked projection."""

    local: str
    hf_name: str
    rows: int


@dataclass
class SlabGroup:
    """A slab the packer writes and a ``gemv`` streams: consecutive parts of one format, with the row permutation."""

    name: str
    format: str
    parts: List[Tuple[str, WeightSpec]]
    row_perm: Optional[np.ndarray]
    rows: int
    k: int

    @property
    def segments(self) -> List[Tuple[str, int]]:
        return [(local, spec.shape[0]) for local, spec in self.parts]


@dataclass
class Projection:
    """The output(s) of a stacked projection: one value per slab group and where each part's columns live."""

    values: List[Value]
    segments: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)   # local -> (value index, column offset, columns)

    @property
    def value(self) -> Value:
        if len(self.values) != 1:
            raise ValueError("projection has several outputs (mixed-format parts); use .values")
        return self.values[0]


class Linear(Module):
    def __init__(self, in_features: int, parts: Sequence[Part], *, prefix: str = "", row_perm: Optional[np.ndarray] = None,
                 epilogue: Optional[str] = None, chunk: Optional[int] = None, round_residual: bool = False, round_silu: bool = False) -> None:
        """``round_residual``: with the residual epilogue, round the product to BF16 before the add (the reference's
        separate linear + BF16 add, two roundings) instead of the fused single rounding.
        ``round_silu`` rounds both projections and SiLU before the gate/up multiply."""
        super().__init__(prefix=prefix)
        if epilogue not in EPILOGUES:
            raise ValueError(f"Linear: epilogue must be one of {EPILOGUES}, got {epilogue!r}")
        if round_residual and epilogue != "residual":
            raise ValueError("Linear: round_residual needs the residual epilogue")
        if round_silu and epilogue != "silu_mul":
            raise ValueError("Linear: round_silu needs the silu_mul epilogue")
        self.round_silu = bool(round_silu)
        self.round_residual = bool(round_residual)
        self.k = in_features
        self.parts = tuple(parts)
        self.n = sum(p.rows for p in self.parts)
        self.row_perm = None if row_perm is None else np.asarray(row_perm, dtype=np.int64)
        if self.row_perm is not None and sorted(self.row_perm.tolist()) != list(range(self.n)):
            raise ValueError(f"Linear {prefix}: row_perm is not a permutation of {self.n} rows")
        self.epilogue = epilogue
        self.chunk = chunk
        if epilogue == "silu_mul" and (len(self.parts) != 2 or self.parts[0].rows != self.parts[1].rows or not chunk):
            raise ValueError("Linear: silu_mul needs exactly two equal parts (gate, up) and a chunk size")

    # ---- structure ----------------------------------------------------------------------------------------------
    def slab_groups(self) -> List[SlabGroup]:
        groups: List[SlabGroup] = []
        for p in self.parts:
            fmt = self.format_of(p.local)
            spec = WeightSpec(p.hf_name, (p.rows, self.k), fmt)
            if groups and groups[-1].format == fmt:
                groups[-1].parts.append((p.local, spec))
                groups[-1].rows += p.rows
            else:
                groups.append(SlabGroup("", fmt, [(p.local, spec)], None, p.rows, self.k))
        for grp in groups:
            grp.name = self.prefix + "+".join(local for local, _ in grp.parts)
            grp.parts = [(local, WeightSpec(s.hf_name, s.shape, s.format, slab=grp.name)) for local, s in grp.parts]
        if len(groups) > 1 and self.row_perm is not None:
            raise ValueError(f"Linear {self.prefix}: a row permutation cannot span parts of different formats "
                             f"({[g.format for g in groups]})")
        if groups and self.row_perm is not None:
            groups[0].row_perm = self.row_perm
        return groups

    def weight_map(self) -> Dict[str, WeightSpec]:
        return {local: spec for grp in self.slab_groups() for local, spec in grp.parts}

    # ---- oracle -------------------------------------------------------------------------------------------------
    def forward(self, x: Any, residual: Any = None) -> Any:
        """``[T, K] → [T, N]`` (or ``[T, N/2]`` for ``silu_mul``) with FP32 accumulation and one rounding; the residual
        epilogue adds before that rounding."""
        import torch

        w = torch.cat([self.param(p.local) for p in self.parts], dim=0)
        acc = x.to(torch.float32) @ w.to(torch.float32).t()
        if self.epilogue == "residual":
            if residual is None:
                raise ValueError("Linear with a residual epilogue needs the residual")
            if self.round_residual:
                acc = acc.to(x.dtype).to(torch.float32)
            acc = acc + residual.to(torch.float32)
        elif self.epilogue == "silu_mul":
            half = self.n // 2
            if self.round_silu:
                acc = acc.to(x.dtype).to(torch.float32)
            activated = torch.nn.functional.silu(acc[:, :half])
            if self.round_silu:
                activated = activated.to(x.dtype).to(torch.float32)
            acc = activated * acc[:, half:]
        return acc.to(x.dtype)

    # ---- IR -----------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, x: Value, *, norm=None, residual: Optional[Value] = None, name: Optional[str] = None,
              rows: Optional[Tuple[int, int]] = None, groups: Optional[Sequence[int]] = None, sibling: bool = False) -> Projection:
        """One ``gemv`` op per format group. ``rows = (start, count)`` restricts the first group to that row range
        (its own dispatch over the slab — a mixer emits its q|k|v rows, its core, then the gate rows as an
        un-barriered ``sibling``, design §5.12); ``groups`` selects which format groups to emit (all by default)."""
        if (self.epilogue == "residual") != (residual is not None):
            raise ValueError(f"Linear {self.prefix}: residual epilogue and residual input must go together")
        if rows is not None and self.epilogue == "silu_mul":
            raise ValueError(f"Linear {self.prefix}: a row range cannot cut a silu_mul projection")
        t = x.shape[0]
        proj = Projection([])
        all_groups = self.slab_groups()
        which = list(range(len(all_groups))) if groups is None else list(groups)
        for i, gi in enumerate(which):
            grp = all_groups[gi]
            w = self.weight_value(g, grp.name, (grp.rows, grp.k), grp.format)
            ins = [x, w]
            if norm is not None:
                ins += [norm.stat, norm.weight]
            if residual is not None:
                ins.append(residual)
            ranged = rows is not None and gi == 0
            start, count = (int(rows[0]), int(rows[1])) if ranged else (0, grp.rows)
            if start < 0 or count <= 0 or start + count > grp.rows:
                raise ValueError(f"Linear {self.prefix}: row range {rows} outside the slab's {grp.rows} rows")
            n_out = count // 2 if self.epilogue == "silu_mul" else count
            out_name = name if (name and len(which) == 1) else f"{grp.name}.y" + (f".r{start}" if ranged else "")
            y = g.value(out_name, (t, n_out), DType.BF16)
            attrs: Dict[str, Any] = dict(norm=norm is not None, epilogue=self.epilogue, format=grp.format)
            if self.round_silu:
                attrs["round_silu"] = True
            if self.round_residual:
                attrs["round_residual"] = True
            if norm is not None:
                attrs["eps"] = norm.eps
            if self.epilogue == "silu_mul":
                attrs["chunk"] = self.chunk
            if ranged:
                attrs["row_range"] = (start, count)
            if sibling:
                attrs["sibling"] = True
            segs = []
            col = 0
            for local, nrows in grp.segments:
                lo, hi = col, col + nrows
                if hi > start and lo < start + count:
                    off = max(lo, start) - start
                    segs.append((local, min(hi, start + count) - max(lo, start)))
                    proj.segments[local] = (i, off, min(hi, start + count) - max(lo, start))
                col += nrows
            attrs["segments"] = segs
            g.op("gemv", ins, [y], domain=BlockDomain("rows", count), klass=OpClass.MAP, **attrs)
            proj.values.append(y)
        return proj
