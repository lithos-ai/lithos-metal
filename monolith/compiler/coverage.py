"""The coverage guard (design §5.7, §5.14): a program whose op has no kernel binding for the target profile does not
build. The check runs on the IR before any emission, so a model that lowers to an unimplemented op fails at compile
time with the list of what is missing, never at run time."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..core.ir import Graph, Op
from ..backends.metal.config import ChipConfig as Profile
from ..ops.registry import OPS, OpDef


class CoverageError(RuntimeError):
    def __init__(self, profile: Profile, missing: List[Tuple[Op, Optional[OpDef]]]) -> None:
        self.profile, self.missing = profile, missing
        lines = [f"{len(missing)} op(s) have no kernel for profile {profile.name} (key {profile.key!r}):"]
        for op, od in missing:
            why = "unknown op kind (not registered)" if od is None else f"bound only for {sorted(od.kernels) or 'nothing'}"
            lines.append(f"  {op!r}: {why}")
        super().__init__("\n".join(lines))


def check_coverage(graph: Graph, profile: Profile) -> None:
    """Raise :class:`CoverageError` unless every op of ``graph`` has a kernel binding for ``profile``."""
    missing: List[Tuple[Op, Optional[OpDef]]] = []
    for op in graph.ops:
        od = OPS.resolve(op.kind)
        if od is None or od.binding_for(profile.key) is None:
            missing.append((op, od))
    if missing:
        raise CoverageError(profile, missing)
