"""Exact dequantizer: rewrite a quantized checkpoint as BF16 safetensors the HF reference can load (design D11,
§5.9 — the reference is the HF model on *these* weights).

    python -m monolith.formats.dequant --model <dir> --out <dir> [--keep-vision] [--shard-gb 4]

Quantized groups (NVFP4, FP8, affine INT4 …) are dequantized with the format oracles; BF16/F32 tensors are copied; ``input_scale`` tensors
(activation quantization) are dropped; ``config.json`` is copied without its ``quantization_config``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from . import FORMATS
from .checkpoint import group_tensors, logical_shape
from .fp import f32_to_bf16
from .safetensors_reader import SafetensorsDir, write_safetensors


def dequantize_dir(model: Path, out: Path, *, keep_vision: bool = False, shard_bytes: int = 4 << 30) -> Dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    st = SafetensorsDir(model)
    dtypes = {n: st.info(n).dtype for n in st.names()}
    shapes = {n: st.info(n).shape for n in st.names()}
    groups = group_tensors(st.names(), dtypes)
    side_names = {s for g in groups.values() for s in g.sides.values()}
    weight_map: Dict[str, str] = {}
    pending: Dict[str, Tuple[str, np.ndarray]] = {}
    pending_bytes, shard_idx = 0, 0

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_idx
        if not pending:
            return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}.safetensors"
        write_safetensors(out / fn, pending, {"format": "pt"})
        for n in pending:
            weight_map[n] = fn
        pending, pending_bytes = {}, 0

    for name in st.names():
        if name in side_names:
            continue
        if name.startswith("model.visual.") and not keep_vision:
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        g = groups.get(base) if base else None
        if g is not None and g.format not in ("", "bf16", "f32"):          # every quantized format with a plugin
            fmt = FORMATS.get(g.format)
            shape = logical_shape(g, shapes)
            tensors = {"weight": st.get(g.weight)}
            for side, full in g.sides.items():
                tensors[side] = st.get(full)
            w = fmt.dequantize(fmt.unpack(tensors, shape=shape))
            arr, dt = f32_to_bf16(w), "BF16"
        else:
            info = st.info(name)
            arr, dt = np.ascontiguousarray(st.get(name)), info.dtype
        pending[name] = (dt, arr)
        pending_bytes += arr.nbytes
        if pending_bytes >= shard_bytes:
            flush()
    flush()
    # rename shards to the HF convention now that the count is known
    total = shard_idx
    renamed: Dict[str, str] = {}
    for i in range(1, total + 1):
        old, new = f"model-{i:05d}.safetensors", f"model-{i:05d}-of-{total:05d}.safetensors"
        (out / old).rename(out / new)
        renamed[old] = new
    weight_map = {n: renamed[f] for n, f in weight_map.items()}
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    for extra in model.glob("*.json"):
        if extra.name in ("model.safetensors.index.json",):
            continue
        if extra.name == "config.json":
            with open(extra) as f:
                cfg = json.load(f)
            cfg.pop("quantization_config", None)
            with open(out / "config.json", "w") as f:
                json.dump(cfg, f, indent=1)
        else:
            shutil.copy(extra, out / extra.name)
    for extra in list(model.glob("*.txt")) + list(model.glob("*.model")) + list(model.glob("*.jinja")):
        shutil.copy(extra, out / extra.name)
    return weight_map


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-vision", action="store_true")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args(argv)
    wm = dequantize_dir(Path(a.model), Path(a.out), keep_vision=a.keep_vision, shard_bytes=int(a.shard_gb * (1 << 30)))
    print(f"wrote {len(wm)} tensors in {len(set(wm.values()))} shard(s) to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
