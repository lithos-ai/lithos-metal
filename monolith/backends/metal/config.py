"""Chip configurations: the measured, per-chip values the compiler and the autotuner consume (design §5.7).

A chip configuration is stored beside its backend under ``monolith/backends/metal/`` (hand-derived from the probe results today; written by the autotuner
later). The free-form measurement blocks are kept as ``raw``; the ``engine`` block is the normalized part this class
exposes: GPU family (the kernel-binding key), lane order of the weight pack, threadgroups per core, the encode-order
rule for sibling overlap, the command-buffer length, and the ``cost(T)`` tables the verify-length rule optimizes
against (design §5.8).
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

COST_FORMAT = {"fp8_e4m3": "fp8"}   # pack format -> the profile's cost_T key (the others use their own name)


@dataclass
class ChipConfig:
    name: str
    chip: str
    family: str                       # "Apple9", "Apple10", …
    gpu_cores: int
    nominal_gbps: float
    lane_order: str                   # "contiguous" | "interleaved16"
    scale_placement: str = "inline"   # "inline" | "block": where a pack keeps its block scales (blm.py, #101)
    threadgroups_per_core: int = 1
    sibling_order: str = "either"     # "alu_first" | "bus_first" | "either"
    max_cb_ms: float = 16.0
    attention: str = "v1"             # v1 / v2 / v3 / mma / mma-direct / auto
    attention_rows: int = 4           # v1's query rows per pass over a chunk (RBMAX): more rows stream the chunk fewer times, at register cost
    attention_v2_threadgroups: int = 2  # v2's threadgroups per core (its blocks are threadgroups: two per core hide the latency of one)
    accelerator: str = "off"          # "on": T > 1 GEMVs run on the tensor-ops tile (gemm_tile, #50/#51) above accelerator_min_t
    accelerator_min_t: Dict[str, int] = field(default_factory=dict)     # cost_T format key -> the smallest T the tile covers (default 2)
    cost_t: Dict[str, Dict[int, float]] = field(default_factory=dict)   # format -> {T: cost relative to T = 1}
    gdn_mixer_fusion: Dict[str, Any] = field(default_factory=dict)      # measured fixed-eight-row shape and worker configuration
    backend: str = "common"
    validation: str = "unmeasured"
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- kernel-binding key -------------------------------------------------------------------------------------
    @property
    def key(self) -> str:
        """Profile key used by op kernel bindings: the GPU family, lower-case (``apple10``)."""
        return self.family.lower()

    @property
    def crew_threads(self) -> int:
        return 384

    @property
    def fingerprint(self) -> str:
        """Keep measured choices and exact core variants out of each other's caches."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    # ---- cost tables --------------------------------------------------------------------------------------------
    def cost(self, fmt: str, t: int) -> float:
        """Cost of a ``t``-token pass over weights of format ``fmt``, in units of a T = 1 pass.

        Exact at measured points, linear between them; raises ``KeyError`` for an unmeasured format and
        ``ValueError`` outside the measured range (the rule must not extrapolate).
        """
        table = self.cost_t.get(fmt)
        if not table:
            raise KeyError(f"profile {self.name}: no cost table for format {fmt!r}")
        if t in table:
            return table[t]
        ts = sorted(table)
        if t < ts[0] or t > ts[-1]:
            raise ValueError(f"profile {self.name}: T={t} outside the measured range {ts[0]}..{ts[-1]} for {fmt!r}")
        lo = max(x for x in ts if x < t)
        hi = min(x for x in ts if x > t)
        w = (t - lo) / (hi - lo)
        return table[lo] * (1 - w) + table[hi] * w

    # ---- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "ChipConfig":
        eng = d.get("engine")
        if not isinstance(eng, Mapping):
            raise ValueError(f"profile {name}: missing the 'engine' block")
        for k in ("family", "lane_order"):
            if k not in eng:
                raise ValueError(f"profile {name}: engine.{k} is required")
        if eng["lane_order"] not in ("contiguous", "interleaved16"):
            raise ValueError(f"profile {name}: engine.lane_order must be 'contiguous' or 'interleaved16'")
        if eng.get("scale_placement", "inline") not in ("inline", "block"):
            raise ValueError(f"profile {name}: engine.scale_placement must be 'inline' or 'block'")
        if eng.get("attention", "v1") not in ("v1", "v2", "v3", "mma", "mma-direct", "auto"):
            raise ValueError(f"profile {name}: engine.attention must be 'v1', 'v2', 'v3', 'mma', 'mma-direct' or 'auto'")
        if int(eng.get("attention_rows", 4)) not in (1, 2, 4, 8, 16):
            raise ValueError(f"profile {name}: engine.attention_rows must be 1, 2, 4, 8 or 16")
        if int(eng.get("attention_v2_threadgroups", 2)) not in (1, 2, 3, 4):
            raise ValueError(f"profile {name}: engine.attention_v2_threadgroups must be 1..4")
        if eng.get("accelerator", "off") not in ("on", "off"):
            raise ValueError(f"profile {name}: engine.accelerator must be 'on' or 'off'")
        min_t = {str(f): int(t) for f, t in (eng.get("accelerator_min_t") or {}).items()}
        if any(t < 1 for t in min_t.values()):
            raise ValueError(f"profile {name}: engine.accelerator_min_t entries must be >= 1")
        cost_t = {f: {int(t): float(c) for t, c in tbl.items()} for f, tbl in (eng.get("cost_T") or {}).items()}
        fusion = dict(eng.get("gdn_mixer_fusion") or {})
        for f, tbl in cost_t.items():
            if tbl.get(1, 1.0) != 1.0:
                raise ValueError(f"profile {name}: cost_T[{f!r}][1] must be 1.0 (costs are relative to T = 1)")
        config = cls(
            name=name,
            chip=str(d.get("chip", name)),
            family=str(eng["family"]),
            gpu_cores=int(d["gpu_cores"]),
            nominal_gbps=float(d["nominal_gbps"]),
            lane_order=str(eng["lane_order"]),
            scale_placement=str(eng.get("scale_placement", "inline")),
            threadgroups_per_core=int(eng.get("threadgroups_per_core", 1)),
            sibling_order=str(eng.get("sibling_order", "either")),
            max_cb_ms=float(eng.get("max_cb_ms", 16.0)),
            attention=str(eng.get("attention", "v1")),
            attention_rows=int(eng.get("attention_rows", 4)),
            attention_v2_threadgroups=int(eng.get("attention_v2_threadgroups", 2)),
            accelerator=str(eng.get("accelerator", "off")),
            accelerator_min_t=min_t,
            cost_t=cost_t,
            gdn_mixer_fusion=fusion,
            backend=str(d.get("backend", "common")),
            validation=str(d.get("validation", "unmeasured")),
            raw=dict(d),
        )
        from .registry import validate_config
        validate_config(config)
        return config


def load_config(path: str | Path) -> ChipConfig:
    p = Path(path)
    with p.open() as f:
        doc = json.load(f)
    return ChipConfig.from_dict(doc.get("name", p.stem), doc)


def load_configs(directory: Optional[str | Path] = None) -> Dict[str, ChipConfig]:
    from .registry import CONFIG_PATHS
    paths = sorted(Path(directory).glob("*.json")) if directory is not None else CONFIG_PATHS.values()
    configs = [load_config(p) for p in paths]
    if len({p.name for p in configs}) != len(configs):
        raise ValueError("duplicate backend configuration name")
    return {p.name: p for p in configs}
