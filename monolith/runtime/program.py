"""The step program as the runtime sees it (design §5.1, §5.4): kernels compiled once, buffers allocated once, ops
encoded once into an ICB and replayed; every parameter lives in a buffer (ICBs have no setBytes). This is the
``program.json`` contract the compiler will emit (plan M4); the toy program of the M2 gate uses it directly."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    role: str = "arena"                  # "weights" | "state" | "arena" | "step_state" | "ring" | "params"
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

    def to_json(self) -> str:
        d = {"version": 1, "step_state": self.step_state, "ring": self.ring, "ring_capacity": self.ring_capacity,
             "context_capacity": self.context_capacity, "backend_id": self.backend_id, "config_digest": self.config_digest,
             "layout": {"t_max": self.layout.t_max, "gamma_max": self.layout.gamma_max},
             "kernels": {k: {"function": v.function, "macros": v.macros, "source": v.source, "language_version": v.language_version}
                         for k, v in self.kernels.items()},
             "buffers": {k: {"nbytes": v.nbytes, "role": v.role, "init_hex": v.init.hex() if v.init is not None else None,
                             "file": v.file, "file_offset": v.file_offset} for k, v in self.buffers.items()},
             "ops": [{"kernel": o.kernel, "bindings": o.bindings, "grid": o.grid, "threadgroup": o.threadgroup,
                      "barrier_before": o.barrier_before, "threadgroup_memory": o.threadgroup_memory, "name": o.name, "meta": o.meta} for o in self.ops]}
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
            backend_id=d.get("backend_id", "common"), config_digest=d.get("config_digest", ""))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Program":
        return cls.from_json(Path(path).read_text())
