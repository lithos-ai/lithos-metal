from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..core.ir import OpClass
from ..registry import Registry

WILDCARD = "*"


@dataclass(frozen=True)
class KernelBinding:
    """Which MSL kernel implements an op on a profile, with its function constants and threadgroups per core."""

    kernel: str
    function_constants: Mapping[str, Any] = field(default_factory=dict)
    threadgroups_per_core: int = 1


@dataclass(frozen=True)
class CostModel:
    """Per-block cost the compiler and the autotuner reason with (bytes streamed, ALU work, ALU-bound flag)."""

    bytes_per_block: int = 0
    flops_per_block: int = 0
    alu_bound: bool = False


@dataclass
class OpDef:
    """An op kind: its class, the kind of block domain it partitions over, its cost, and its kernel bindings."""

    name: str
    klass: OpClass
    domain_kind: str
    cost: CostModel = field(default_factory=CostModel)
    kernels: Dict[str, KernelBinding] = field(default_factory=dict)   # profile key -> binding, "*" = any

    def binding_for(self, profile_key: str) -> Optional[KernelBinding]:
        return self.kernels.get(profile_key) or self.kernels.get(WILDCARD)

    def bind(self, profile_key: str, binding: KernelBinding) -> "OpDef":
        if profile_key in self.kernels and self.kernels[profile_key] != binding:
            raise ValueError(f"op {self.name}: profile {profile_key!r} already bound to {self.kernels[profile_key]}")
        self.kernels[profile_key] = binding
        return self


OPS: Registry[OpDef] = Registry("op")


def register_op(opdef: OpDef) -> OpDef:
    return OPS.register(opdef.name, opdef)
