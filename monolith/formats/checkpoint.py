"""Checkpoint-side helpers: group a safetensors name list into matrices with their scale tensors, and detect the
storage format of each group from the safetensors dtypes (the ModelOpt conventions of the target checkpoint)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

SIDE_KEYS = ("weight_scale", "weight_scale_2", "input_scale", "scales", "biases")


@dataclass
class TensorGroup:
    base: str                                   # e.g. "model.language_model.layers.0.mlp.gate_proj"
    weight: str                                 # the "<base>.weight" key
    sides: Dict[str, str] = field(default_factory=dict)   # side key -> full name
    format: str = ""                            # "nvfp4" | "fp8_e4m3" | "int4_affine" | "bf16" | "f32" | ""


def group_tensors(names: Iterable[str], dtypes: Mapping[str, str]) -> Dict[str, TensorGroup]:
    """``{base: TensorGroup}`` for every ``<base>.weight`` with its ``weight_scale*`` / ``input_scale`` siblings."""
    names = list(names)
    groups: Dict[str, TensorGroup] = {}
    for n in names:
        if n.endswith(".weight"):
            base = n[: -len(".weight")]
            groups[base] = TensorGroup(base, n)
    for n in names:
        for side in SIDE_KEYS:
            if n.endswith("." + side):
                base = n[: -len("." + side)]
                if base in groups:
                    groups[base].sides[side] = n
    for g in groups.values():
        g.format = detect_format(g, dtypes)
    return groups


def detect_format(g: TensorGroup, dtypes: Mapping[str, str]) -> str:
    wd = dtypes.get(g.weight, "")
    if wd == "U8" and "weight_scale" in g.sides and dtypes.get(g.sides["weight_scale"]) == "F8_E4M3" and "weight_scale_2" in g.sides:
        return "nvfp4"
    if wd == "F8_E4M3" and "weight_scale" in g.sides and dtypes.get(g.sides["weight_scale"]) == "F32":
        return "fp8_e4m3"
    if wd == "U32" and "scales" in g.sides and "biases" in g.sides and dtypes.get(g.sides["scales"]) in ("F16", "BF16"):
        return "int4_affine"                      # MLX / AWQ affine 4-bit groups (the group size comes from the shapes)
    if wd == "U32" and "scales" in g.sides and "biases" not in g.sides and dtypes.get(g.sides["scales"]) == "U8":
        return "nvfp4"                            # MLX's nvfp4 mode: the same codes and E4M3 block scales, no tensor scale (see nvfp4.py)
    if wd == "BF16":
        return "bf16"
    if wd == "F32":
        return "f32"
    return ""


def logical_shape(g: TensorGroup, shapes: Mapping[str, Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    """The ``[N, K]`` (or other) shape of the dequantized tensor."""
    sh = tuple(shapes[g.weight])
    if g.format == "nvfp4":
        return sh[:-1] + (sh[-1] * (8 if "weight_scale_2" not in g.sides else 2),)   # MLX packs 8 codes per U32, ModelOpt 2 per U8
    if g.format == "int4_affine":
        return sh[:-1] + (sh[-1] * 8,)
    return sh
