"""The step program as the runtime sees it (design §5.1, §5.4): kernels compiled once, buffers allocated once, ops
encoded once into an ICB and replayed; every parameter lives in a buffer (ICBs have no setBytes). This is the
``program.json`` contract the compiler will emit (plan M4); the toy program of the M2 gate uses it directly."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..core.step_state import StepStateLayout


@dataclass
class KernelSpec:
    source: str                          # MSL source (already assembled: prelude + snippets + template)
    function: str
    macros: Dict[str, str] = field(default_factory=dict)
    language_version: int = 0            # 0 = the compiler's default; the tensor-ops kernels need MSL 4.0 (4 << 16)


@dataclass
class BufferSpec:
    nbytes: int
    init: Optional[bytes] = None         # initial contents (zeros if None)
    role: str = "arena"                  # "weights" | "state" | "arena" | "step_state" | "ring" | "params" | "host"
    file: Optional[str] = None           # weights: a page-aligned window of this file, mapped without a copy
    file_offset: int = 0


@dataclass
class OpSpec:
    kernel: str                          # key into Program.kernels
    bindings: List[Tuple[int, str, int]] # (buffer index, buffer name, byte offset)
    grid: Tuple[int, int, int]
    threadgroup: Tuple[int, int, int]
    barrier_before: bool = True          # this op waits for every op before it; false only where the barrier pass proved it independent (§5.12)
    threadgroup_memory: List[Tuple[int, int]] = field(default_factory=list)
    name: str = ""
    meta: Dict[str, object] = field(default_factory=dict)   # informational: op kind, bytes streamed, layer … (JSON)

    def access(self) -> Tuple[Set[str], Set[str]]:
        """(reads, writes) as buffer names. ``meta["writes"]`` lists the binding slots the op writes; an op without it
        writes every buffer it binds. ``meta["indirect_reads"]`` and ``meta["indirect_writes"]`` name buffers the op
        reaches through GPU addresses held in another buffer rather than through a binding."""
        w_idx = self.meta.get("writes")
        bound = {b[1] for b in self.bindings}
        writes = bound if w_idx is None else {b[1] for b in self.bindings if b[0] in w_idx}
        writes |= set(self.meta.get("indirect_writes", ()))
        return (bound | set(self.meta.get("indirect_reads", ()))) - writes, writes


@dataclass
class HostPoint:
    """CPU work inside a step (design: Runtime program). Boundary b sits before op b of the step. The task starts once
    ops [0, publish) of the step have completed (publish 0: once the previous step completed), and the GPU work from
    op ``wait`` on starts after it returns (wait = len(ops): the next step's). publish == wait makes a sync point.
    ``writes`` and ``reads`` name the buffers the task touches; the ops that may run beside it, [publish, wait), must
    not race them (Program.check_host_points)."""
    task: str                            # the name the session's task is supplied under (Engine(host_tasks=...))
    publish: int
    wait: int
    writes: List[str] = field(default_factory=list)
    reads: List[str] = field(default_factory=list)


@dataclass
class Program:
    kernels: Dict[str, KernelSpec]
    buffers: Dict[str, BufferSpec]
    ops: List[OpSpec]
    step_state: str = "step_state"       # buffer name holding StepState
    ring: str = "ring"                   # buffer name of the token ring: 8-byte slots, (sequence << 32) | token
    ring_capacity: int = 4096
    layout: StepStateLayout = field(default_factory=StepStateLayout)
    context_capacity: int = 0            # positions a sequence may occupy (0 = unbounded): the serial ops stop the program (error 2) beyond it
    backend_id: str = "common"
    config_digest: str = ""
    host_points: List[HostPoint] = field(default_factory=list)   # in program order: publish boundaries never decrease

    def check_host_points(self, same: Optional[Callable[[str, str], bool]] = None) -> None:
        """Raises ValueError unless every host point is well formed and its task cannot race the GPU: no op of
        [publish, wait), which may run while the task runs, writes a buffer the task reads or writes, or reads a buffer
        the task writes. ``same(a, b)`` says whether two buffer names reach the same memory; by default only equal
        names do (the Engine compares GPU address ranges)."""
        same = same or (lambda a, b: a == b)
        n, prev = len(self.ops), 0
        for i, p in enumerate(self.host_points):
            where = f"host point {i} ({p.task!r})"
            if not p.task:
                raise ValueError(f"host point {i}: needs a task name")
            if not 0 <= p.publish <= p.wait <= n:
                raise ValueError(f"{where}: need 0 <= publish ({p.publish}) <= wait ({p.wait}) <= {n}, the op count")
            if p.publish < prev:
                raise ValueError(f"{where}: publish boundaries must not decrease in program order")
            prev = p.publish
            for name in [*p.writes, *p.reads]:
                if name not in self.buffers:
                    raise ValueError(f"{where}: unknown buffer {name!r}")
            for name in p.writes:
                if self.buffers[name].role in ("weights", "params"):
                    raise ValueError(f"{where}: writes {name!r}, a read-only {self.buffers[name].role} buffer")
                hit = next((b for b, spec in self.buffers.items() if spec.role in ("weights", "params") and same(name, b)), None)
                if hit is not None:
                    raise ValueError(f"{where}: writes {name!r}, which reaches the read-only {self.buffers[hit].role} buffer {hit!r}")
            for j in range(p.publish, p.wait):
                reads, writes = self.ops[j].access()
                op = f"op {j} ({self.ops[j].name or self.ops[j].kernel})"
                for t in p.writes:
                    hit = next((b for b in sorted(reads | writes) if same(t, b)), None)
                    if hit is not None:
                        raise ValueError(f"{where}: {op} accesses {hit!r} while the task writes {t!r}")
                for t in p.reads:
                    hit = next((b for b in sorted(writes) if same(t, b)), None)
                    if hit is not None:
                        raise ValueError(f"{where}: {op} writes {hit!r} while the task reads {t!r}")

    def to_json(self) -> str:
        d = {"version": 1, "step_state": self.step_state, "ring": self.ring, "ring_capacity": self.ring_capacity,
             "context_capacity": self.context_capacity, "backend_id": self.backend_id, "config_digest": self.config_digest,
             "layout": {"t_max": self.layout.t_max, "gamma_max": self.layout.gamma_max},
             "kernels": {k: {"function": v.function, "macros": v.macros, "source": v.source, "language_version": v.language_version}
                         for k, v in self.kernels.items()},
             "buffers": {k: {"nbytes": v.nbytes, "role": v.role, "init_hex": v.init.hex() if v.init is not None else None,
                             "file": v.file, "file_offset": v.file_offset} for k, v in self.buffers.items()},
             "ops": [{"kernel": o.kernel, "bindings": o.bindings, "grid": o.grid, "threadgroup": o.threadgroup,
                      "barrier_before": o.barrier_before, "threadgroup_memory": o.threadgroup_memory, "name": o.name, "meta": o.meta} for o in self.ops],
             "host_points": [{"task": h.task, "publish": h.publish, "wait": h.wait, "writes": h.writes, "reads": h.reads}
                             for h in self.host_points]}
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Program":
        d = json.loads(text)
        if d.get("version") != 1:
            raise ValueError("unsupported program version")
        return cls(
            kernels={k: KernelSpec(v["source"], v["function"], dict(v.get("macros", {})), int(v.get("language_version", 0))) for k, v in d["kernels"].items()},
            buffers={k: BufferSpec(v["nbytes"], bytes.fromhex(v["init_hex"]) if v.get("init_hex") else None, v.get("role", "arena"),
                                   v.get("file"), int(v.get("file_offset", 0))) for k, v in d["buffers"].items()},
            ops=[OpSpec(o["kernel"], [tuple(b) for b in o["bindings"]], tuple(o["grid"]), tuple(o["threadgroup"]), o.get("barrier_before", o.get("barrier_after", True)),
                        [tuple(t) for t in o.get("threadgroup_memory", [])], o.get("name", ""), dict(o.get("meta", {}))) for o in d["ops"]],
            step_state=d["step_state"], ring=d["ring"], ring_capacity=d["ring_capacity"], context_capacity=int(d.get("context_capacity", 0)),
            layout=StepStateLayout(d["layout"]["t_max"], d["layout"]["gamma_max"]),
            backend_id=d.get("backend_id", "common"), config_digest=d.get("config_digest", ""),
            host_points=[HostPoint(h["task"], int(h["publish"]), int(h["wait"]), list(h.get("writes", [])), list(h.get("reads", [])))
                         for h in d.get("host_points", [])])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Program":
        return cls.from_json(Path(path).read_text())
