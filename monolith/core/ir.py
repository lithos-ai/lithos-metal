"""The compiler's IR: a typed tensor graph whose ops carry a block domain, an op class and a cost.

An ``Op`` is one *fused op* of the step program — it becomes one Metal dispatch with the crew geometry (design
§5.1–5.2). Its ``domain`` says how many blocks the op is partitioned into (one SIMD-group runs several blocks under
static slices), its ``klass`` says whether blocks write disjoint outputs (``MAP``), emit partials combined by a
follow-up (``REDUCE``), or run on one SIMD-group (``SERIAL``). Dependencies are edges through ``Value``s; the
barrier-placement pass turns every producer→consumer edge between ops into an ICB barrier only where blocks of the
consumer read outputs of *other* blocks of the producer.

The IR knows nothing about Metal, kernels or models: ops are named by ``kind`` and bound to kernels through the op
registry (``monolith.ops``) per chip profile — the coverage guard (``monolith.compiler.coverage``) fails a build for an
op without a binding.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .dtypes import DType
from .shapes import Dim, Shape, Sym


class OpClass(Enum):
    MAP = "MAP"          # disjoint outputs per block
    REDUCE = "REDUCE"    # per-block partials, combined in block order by a follow-up dispatch (deterministic)
    SERIAL = "SERIAL"    # one SIMD-group: sampling finalize, accept scan, verify-length select, state advance


@dataclass(frozen=True)
class BlockDomain:
    """How an op is cut into blocks: ``n_blocks`` blocks of kind ``kind`` (``rows``, ``heads``, ``span``, …)."""

    kind: str
    n_blocks: Dim

    def __post_init__(self) -> None:
        if isinstance(self.n_blocks, int) and self.n_blocks <= 0:
            raise ValueError(f"BlockDomain({self.kind}): n_blocks must be positive, got {self.n_blocks}")


@dataclass(eq=False)
class Value:
    """A tensor in the graph. Weights carry a storage ``format`` (a format-plugin name) instead of a float dtype."""

    name: str
    shape: Shape
    dtype: DType
    format: Optional[str] = None
    producer: Optional["Op"] = None
    consumers: List["Op"] = field(default_factory=list)
    is_input: bool = False
    is_weight: bool = False
    is_state: bool = False       # persistent per-sequence buffer (KV cache, recurrent state): read and updated in place
    is_const: bool = False       # a constant table or aux tensor from the pack (RoPE tables, norm weights, A_log …)
    view_of: Optional[Tuple[str, int]] = None   # (base value, first row): this value aliases rows of another value's buffer

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_view(self) -> bool:
        return self.view_of is not None

    @property
    def is_source(self) -> bool:
        """Inputs, weights, states and constants are graph sources: nothing produces them."""
        return self.is_input or self.is_weight or self.is_state or self.is_const

    def __repr__(self) -> str:
        fmt = f", format={self.format}" if self.format else ""
        return f"Value({self.name}: {list(self.shape)} {self.dtype}{fmt})"


@dataclass(eq=False)
class Op:
    """One fused op = one dispatch of the step program."""

    kind: str
    inputs: List[Value]
    outputs: List[Value]
    domain: BlockDomain
    klass: OpClass
    attrs: Dict[str, Any] = field(default_factory=dict)
    id: int = -1

    def __repr__(self) -> str:
        return f"Op#{self.id}({self.kind}, {self.klass.value}, {self.domain.kind}×{self.domain.n_blocks})"


class Graph:
    """Ops in program order plus the values that connect them."""

    def __init__(self, name: str = "step") -> None:
        self.name = name
        self.ops: List[Op] = []
        self.values: Dict[str, Value] = {}
        self.views: Dict[str, List[Value]] = {}       # base value name -> its views
        self.symbols: Set[Sym] = set()
        self._scope = ""

    # ---- values -------------------------------------------------------------------------------------------
    @contextmanager
    def scope(self, prefix: str) -> Iterator[None]:
        """Activation values created inside take ``prefix`` in front of their names. A module tree lowered several
        times into one program (an LM drafter's ingest pass and each of its chain steps, design §5.8) keeps its
        slabs, states and constants — sources, looked up by their own names — while every pass's activations stay
        distinct. Scopes nest."""
        saved, self._scope = self._scope, self._scope + prefix
        try:
            yield
        finally:
            self._scope = saved

    def value(self, name: str, shape: Sequence[Dim], dtype: DType, *, format: Optional[str] = None, scoped: bool = True) -> Value:
        if scoped:
            name = self._scope + name
        if name in self.values:
            raise ValueError(f"graph {self.name}: value name {name!r} is already taken")
        v = Value(name, tuple(shape), dtype, format)
        self.values[name] = v
        self.symbols.update(d for d in v.shape if isinstance(d, Sym))
        return v

    def input(self, name: str, shape: Sequence[Dim], dtype: DType) -> Value:
        v = self.value(name, shape, dtype, scoped=False)
        v.is_input = True
        return v

    def weight(self, name: str, shape: Sequence[int], format: str) -> Value:
        """A packed weight slab: ``format`` names the format plugin that decodes it; the dtype is the storage byte."""
        v = self.value(name, shape, DType.U8, format=format, scoped=False)
        v.is_weight = True
        return v

    def state(self, name: str, shape: Sequence[Dim], dtype: DType) -> Value:
        """A persistent per-sequence buffer. Ops read it as an input and declare in-place writes through the
        ``updates=[...]`` attr (a list of state value names), which the barrier pass treats as a write."""
        v = self.value(name, shape, dtype, scoped=False)
        v.is_state = True
        return v

    def const(self, name: str, shape: Sequence[Dim], dtype: DType) -> Value:
        """A constant tensor from the pack's aux section (tables, norm weights, small parameters)."""
        v = self.value(name, shape, dtype, scoped=False)
        v.is_const = True
        return v

    def view(self, name: str, base: Value, row: int, rows: int = 1) -> Value:
        """A value aliasing rows ``[row, row + rows)`` of ``base``: an op reads or writes a slice of a buffer (the
        Markov chain's per-position logits and tokens, design §5.8). The base is an ordinary value with a static row
        count that no op produces as a whole; :meth:`check` counts the rows written through its views."""
        if base.name not in self.values or self.values[base.name] is not base:
            raise ValueError(f"graph {self.name}: {base!r} does not belong to this graph")
        if base.is_source or base.is_view:
            raise ValueError(f"graph {self.name}: cannot view {base!r} (a source or a view)")
        n = base.shape[0] if base.shape else 0
        if not isinstance(n, int) or row < 0 or rows < 1 or row + rows > n:
            raise ValueError(f"graph {self.name}: view rows [{row}, {row + rows}) outside {base!r} (static rows only)")
        v = self.value(name, (rows,) + tuple(base.shape[1:]), base.dtype)
        v.view_of = (base.name, row)
        self.views.setdefault(base.name, []).append(v)
        return v

    # ---- ops ----------------------------------------------------------------------------------------------
    def op(
        self,
        kind: str,
        inputs: Iterable[Value],
        outputs: Iterable[Value],
        *,
        domain: BlockDomain,
        klass: OpClass,
        **attrs: Any,
    ) -> Op:
        ins, outs = list(inputs), list(outputs)
        for v in ins:
            if v.name not in self.values or self.values[v.name] is not v:
                raise ValueError(f"graph {self.name}: input {v!r} does not belong to this graph")
        for v in outs:
            if v.name not in self.values or self.values[v.name] is not v:
                raise ValueError(f"graph {self.name}: output {v!r} does not belong to this graph")
            if v.producer is not None:
                raise ValueError(f"graph {self.name}: {v!r} already has a producer {v.producer!r}")
            if v.is_source:
                raise ValueError(f"graph {self.name}: {v!r} is a graph source and cannot be produced by an op")
        for name in attrs.get("updates", ()):
            sv = self.values.get(name)
            if sv is None or not sv.is_state:
                raise ValueError(f"graph {self.name}: op {kind} updates {name!r}, which is not a state of this graph")
            if sv not in ins:
                raise ValueError(f"graph {self.name}: op {kind} updates {name!r} without reading it (list it as an input)")
        op = Op(kind, ins, outs, domain, klass, dict(attrs), id=len(self.ops))
        for v in outs:
            v.producer = op
        for v in ins:
            v.consumers.append(op)
        self.ops.append(op)
        return op

    # ---- structure ----------------------------------------------------------------------------------------
    def producers(self, op: Op) -> List[Op]:
        """Ops this op depends on, in program order (deduplicated)."""
        seen: Dict[int, Op] = {}
        for v in op.inputs:
            if v.producer is not None:
                seen[v.producer.id] = v.producer
        return [seen[k] for k in sorted(seen)]

    def states(self) -> List[Value]:
        return [v for v in self.values.values() if v.is_state]

    def check(self) -> None:
        """Every consumed value is a source or produced by an earlier op (a value written through views: every row
        read was written by an earlier op); every op has ≥ 1 output."""
        for op in self.ops:
            if not op.outputs:
                raise ValueError(f"graph {self.name}: {op!r} has no outputs")
            for v in op.inputs:
                if v.is_source:
                    continue
                if v.producer is not None:
                    if v.producer.id >= op.id:
                        raise ValueError(f"graph {self.name}: {op!r} reads {v!r} before its producer {v.producer!r}")
                    continue
                base_name, row0 = v.view_of if v.view_of is not None else (v.name, 0)
                base = self.values[base_name]
                rows = v.shape[0] if v.shape and isinstance(v.shape[0], int) else None
                if rows is None or (base.producer is None and base_name not in self.views):
                    raise ValueError(f"graph {self.name}: {op!r} reads {v!r}, which nothing produces")
                covered: Set[int] = set()
                if base.producer is not None and base.producer.id < op.id:
                    covered.update(range(int(base.shape[0])))
                for w in self.views.get(base_name, ()):
                    if w.producer is not None and w.producer.id < op.id:
                        covered.update(range(w.view_of[1], w.view_of[1] + int(w.shape[0])))
                missing = sorted(set(range(row0, row0 + rows)) - covered)
                if missing:
                    raise ValueError(f"graph {self.name}: {op!r} reads rows {missing[:4]} of {base!r}, which nothing produces before it")

    def __len__(self) -> int:
        return len(self.ops)

    def __repr__(self) -> str:
        return f"Graph({self.name}: {len(self.ops)} ops, {len(self.values)} values)"
