"""A minimal, torch-free safetensors reader/writer (header JSON + memory-mapped data).

The standard ``safetensors`` numpy API cannot express BF16 or F8_E4M3; here those come back as ``uint16`` / ``uint8``
views with the safetensors dtype string kept alongside, which is exactly what the format plugins want.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

import numpy as np

_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F8_E4M3": np.uint8, "U8": np.uint8, "I8": np.int8,
       "I32": np.int32, "I64": np.int64, "U32": np.uint32, "BOOL": np.bool_, "F64": np.float64}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int
    file: Path


def read_header(path: str | Path) -> Tuple[Dict[str, TensorInfo], Mapping[str, str]]:
    p = Path(path)
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {}) or {}
    infos = {}
    for name, v in header.items():
        s, e = v["data_offsets"]
        infos[name] = TensorInfo(name, v["dtype"], tuple(v["shape"]), 8 + n + s, 8 + n + e, p)
    return infos, meta


class SafetensorsDir:
    """All shards of a checkpoint directory (or one file); tensors are memory-mapped on access. ``rename`` maps
    the stored names to the names the model package expects (a checkpoint written by another tool); ``adapt``
    (``(name, array, info) → array``, applied by ``get`` under the package's names) maps stored values to the
    package's convention (a tool that folds a constant into a tensor), keeping the stored dtype."""

    def __init__(self, path: str | Path, rename=None, adapt=None) -> None:
        p = Path(path)
        files: List[Path]
        if p.is_file():
            files = [p]
        else:
            idx = p / "model.safetensors.index.json"
            if idx.exists():
                with open(idx) as f:
                    files = sorted({p / fn for fn in json.load(f)["weight_map"].values()})
            else:
                files = sorted(p.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors under {p}")
        self.infos: Dict[str, TensorInfo] = {}
        for f in files:
            infos, _ = read_header(f)
            if rename is not None:
                infos = {rename(k): v for k, v in infos.items()}
            self.infos.update(infos)
        self._maps: Dict[Path, np.memmap] = {}
        self._adapt = adapt

    def names(self) -> List[str]:
        return sorted(self.infos)

    def info(self, name: str) -> TensorInfo:
        return self.infos[name]

    def _mm(self, file: Path) -> np.memmap:
        if file not in self._maps:
            self._maps[file] = np.memmap(file, dtype=np.uint8, mode="r")
        return self._maps[file]

    def get(self, name: str) -> np.ndarray:
        """The tensor as a numpy view of the file (BF16 → uint16, F8_E4M3 → uint8)."""
        t = self.infos[name]
        raw = self._mm(t.file)[t.start: t.end]
        arr = raw.view(_NP[t.dtype]).reshape(t.shape)
        return arr if self._adapt is None else self._adapt(name, arr, t)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for name in self.names():
            yield name, self.get(name)

    def close(self) -> None:
        for m in self._maps.values():
            del m
        self._maps.clear()


def write_safetensors(path: str | Path, tensors: Mapping[str, Tuple[str, np.ndarray]], metadata: Mapping[str, str] | None = None) -> None:
    """``tensors``: ``{name: (safetensors dtype string, array)}``; arrays are written as their raw bytes."""
    header: Dict[str, object] = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        dt, arr = tensors[name]
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    if metadata:
        header["__metadata__"] = dict(metadata)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
