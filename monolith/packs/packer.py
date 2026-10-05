"""The packer and the pack reader. File layout::

    weights.pack   [slab 0 payload][row scales f32][pad to 16 KB][slab 1 …][aux 0][pad][aux 1 …]
    manifest.json  everything the runtime needs to bind a slab without reading the payload

Slabs are processed one at a time (memory = the largest slab), so a 21 GB checkpoint packs on a 36 GB machine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..formats import FORMATS, PackLayout
from ..formats.blm import PackInfo
from ..formats.checkpoint import group_tensors, logical_shape
from ..formats.fp import bf16_to_f32
from ..formats.safetensors_reader import SafetensorsDir
from . import transforms

ALIGN = 16384
MANIFEST_VERSION = 3  # payload-ordered NVFP4 scales; versions 1 and 2 remain readable
_NP_OF = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I32": np.int32, "I64": np.int64, "U8": np.uint8}


@dataclass(frozen=True)
class Segment:
    """Rows of one checkpoint matrix (``source`` = ``<base>`` of ``<base>.weight``), optionally a subset in a given
    order (``rows[new] = old``)."""

    source: str
    rows: Optional[np.ndarray] = None

    def describe(self) -> Dict[str, Any]:
        return {"source": self.source, "row_select": None if self.rows is None else int(len(self.rows))}


@dataclass
class SlabRequest:
    name: str
    format: str
    segments: Sequence[Segment]
    layout: PackLayout = field(default_factory=PackLayout)
    row_perm: Optional[np.ndarray] = None        # applied to the stacked rows, perm[new] = old


AUX_TRANSFORMS = (None, "f32", "bf16_f32", "one_plus", "neg_exp")


@dataclass
class AuxRequest:
    """A small tensor stored raw. ``transform``: None (bytes as stored), ``"f32"`` (widen exactly), ``"bf16_f32"``
    (the BF16-valued parameter a BF16 reference model holds, widened), ``"one_plus"`` (``1 + w`` as float32),
    ``"neg_exp"`` (``−exp(w)`` as float32). ``perm`` (``perm[new] = old``) reorders the leading axis afterwards."""

    name: str
    source: str                                  # full checkpoint key
    transform: Optional[str] = None
    perm: Optional[np.ndarray] = None


@dataclass
class TableRequest:
    """A computed constant (RoPE tables …) stored raw: ``array`` with the safetensors dtype name it is stored as
    (``"F32"``, ``"BF16"`` — a BF16 table is passed as its uint16 bit pattern)."""

    name: str
    array: np.ndarray
    dtype: str = "F32"


class Packer:
    def __init__(self, checkpoint: str | Path, out_dir: str | Path, rename=None, adapt=None) -> None:
        self.ckpt = SafetensorsDir(checkpoint, rename=rename, adapt=adapt)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dtypes = {n: self.ckpt.info(n).dtype for n in self.ckpt.names()}
        self.shapes = {n: self.ckpt.info(n).shape for n in self.ckpt.names()}
        self.groups = group_tensors(self.ckpt.names(), self.dtypes)
        self._slabs: List[Dict[str, Any]] = []
        self._aux: List[Dict[str, Any]] = []
        self._fh = open(self.out / "weights.pack", "wb")
        self._pos = 0

    # ---- internals -------------------------------------------------------------------------------------------
    def _align(self) -> None:
        pad = (-self._pos) % ALIGN
        if pad:
            self._fh.write(b"\0" * pad)
            self._pos += pad

    def _write(self, data: bytes) -> int:
        off = self._pos
        self._fh.write(data)
        self._pos += len(data)
        return off

    def _segment_spec(self, seg: Segment, fmt_name: str):
        g = self.groups.get(seg.source)
        if g is None:
            raise KeyError(f"packer: no tensor group {seg.source!r} in the checkpoint")
        requantize = g.format != fmt_name
        if requantize and g.format not in ("bf16", "f32"):
            raise ValueError(f"packer: {seg.source} is {g.format!r}, slab wants {fmt_name!r} (only a BF16 / F32 matrix is re-quantized at pack time)")
        shape = logical_shape(g, self.shapes)
        tensors = {"weight": self.ckpt.get(g.weight)}
        for side, full in g.sides.items():
            tensors[side] = self.ckpt.get(full)
        if seg.rows is not None:
            rows = np.asarray(seg.rows)
            tensors = {k: (v[rows] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == shape[0] else v) for k, v in tensors.items()}
            shape = (int(len(rows)),) + tuple(shape[1:])
        if len(shape) != 2:
            raise ValueError(f"packer: {seg.source} is not a matrix ({shape})")
        if requantize:
            # re-quantization at pack time (a BF16 drafter packed as NVFP4 / INT8 / …): the format's own quantizer on the
            # float matrix the reference holds; the module tree is bound to the pack's format when a session loads it
            w = tensors["weight"]
            w32 = bf16_to_f32(w) if self.dtypes.get(g.weight) == "BF16" else np.asarray(w, dtype=np.float32)
            return FORMATS.get(fmt_name).quantize(np.ascontiguousarray(w32.reshape(int(shape[0]), int(shape[1]))))
        return FORMATS.get(fmt_name).unpack(tensors, shape=(int(shape[0]), int(shape[1])))

    # ---- public --------------------------------------------------------------------------------------------------
    def add_slab(self, req: SlabRequest) -> Dict[str, Any]:
        fmt = FORMATS.get(req.format)
        specs = [self._segment_spec(s, req.format) for s in req.segments]
        k = specs[0].shape[1]
        if any(s.shape[1] != k for s in specs):
            raise ValueError(f"slab {req.name}: segments have different K")
        r = req.layout.rows
        seg_rows = [s.shape[0] for s in specs]
        # stack the raw tensors along N
        stacked = {key: np.concatenate([s.tensors[key] for s in specs], axis=0) for key in specs[0].tensors}
        n = sum(seg_rows)
        seg_of_row = np.repeat(np.arange(len(specs)), seg_rows)
        if req.row_perm is not None:
            perm = np.asarray(req.row_perm)
            if sorted(perm.tolist()) != list(range(n)):
                raise ValueError(f"slab {req.name}: row_perm is not a permutation of {n} rows")
            stacked = {key: v[perm] for key, v in stacked.items()}
            seg_of_row = seg_of_row[perm]
        params = dict(specs[0].params)
        spec = type(specs[0])(req.format, (n, k), stacked, params)
        data, info = fmt.pack(spec, req.layout)
        # per-row tensor scales: each row keeps the per-tensor scale of the segment it came from, whatever the
        # row permutation did (gate/up interleaving mixes two NVFP4 matrices with different scales in one block)
        scale_of_seg = np.array([_tensor_scale(s) for s in specs], dtype=np.float32)
        row_scales = scale_of_seg[seg_of_row]
        self._align()
        off = self._write(data)
        rs_off = self._write(row_scales.astype(np.float32).tobytes())
        entry = {
            "name": req.name, "format": req.format, "offset": off, "nbytes": len(data), "n": n, "k": k,
            "rows": r, "unit_bytes": info.unit_bytes, "payload_bytes": info.payload_bytes, "scale_bytes": info.scale_bytes,
            "lane_order": info.lane_order, "scale_group": info.scale_group, "n_blocks": info.n_blocks, "scale_placement": info.scale_placement,
            "scale_unit_bytes": info.scale_unit_bytes, "scale_dtype": info.scale_dtype,
            "scale_lane_divisor": info.scale_lane_divisor, "scale_order": info.scale_order,
            "row_scales_offset": rs_off, "row_perm": req.row_perm is not None,
            "segments": [dict(s.describe(), rows=int(nr), tensor_scale=float(sc)) for s, nr, sc in zip(req.segments, seg_rows, scale_of_seg)],
        }
        self._slabs.append(entry)
        return entry

    def add_aux(self, req: AuxRequest) -> Dict[str, Any]:
        info = self.ckpt.info(req.source)
        arr = self.ckpt.get(req.source)
        if req.transform is None:
            out, dtype = np.ascontiguousarray(arr), info.dtype
        elif req.transform == "f32":
            out = bf16_to_f32(arr) if info.dtype == "BF16" else np.asarray(arr, dtype=np.float32)
            dtype = "F32"
        elif req.transform == "one_plus":
            out, dtype = transforms.one_plus(arr, dtype_in=info.dtype), "F32"
        elif req.transform == "bf16_f32":
            out, dtype = transforms.bf16_round_f32(arr, dtype_in=info.dtype), "F32"
        elif req.transform == "neg_exp":
            out, dtype = transforms.neg_exp(arr, dtype_in=info.dtype), "F32"
        else:
            raise ValueError(f"aux {req.name}: unknown transform {req.transform!r} (one of {AUX_TRANSFORMS})")
        if req.perm is not None:
            perm = np.asarray(req.perm)
            if sorted(perm.tolist()) != list(range(out.shape[0])):
                raise ValueError(f"aux {req.name}: perm is not a permutation of the leading axis ({out.shape[0]})")
            out = np.ascontiguousarray(out[perm])
        self._align()
        off = self._write(out.astype(out.dtype).tobytes())
        entry = {"name": req.name, "source": req.source, "offset": off, "nbytes": out.nbytes, "dtype": dtype,
                 "shape": [int(x) for x in out.shape], "transform": req.transform}
        self._aux.append(entry)
        return entry

    def add_table(self, req: TableRequest) -> Dict[str, Any]:
        if req.dtype not in _NP_OF:
            raise ValueError(f"table {req.name}: unsupported dtype {req.dtype!r}")
        out = np.ascontiguousarray(np.asarray(req.array).astype(_NP_OF[req.dtype], copy=False))
        self._align()
        off = self._write(out.tobytes())
        entry = {"name": req.name, "source": None, "offset": off, "nbytes": out.nbytes, "dtype": req.dtype,
                 "shape": [int(x) for x in out.shape], "transform": "table"}
        self._aux.append(entry)
        return entry

    def write(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._align()
        self._fh.close()
        version = MANIFEST_VERSION if any(s.get("scale_order") == "payload" for s in self._slabs) else 2
        manifest = {"version": version, "alignment": ALIGN, "pack": "weights.pack", "nbytes": self._pos,
                    "slabs": self._slabs, "aux": self._aux, **(extra or {})}
        with open(self.out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest


def _tensor_scale(spec) -> float:
    return float(spec.params.get("weight_scale_2", spec.params.get("weight_scale", 1.0)))


class PackFile:
    """Read side of a pack directory (the same manifest the C++ runtime parses)."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        with open(self.dir / "manifest.json") as f:
            self.manifest = json.load(f)
        if self.manifest.get("version") not in (1, 2, MANIFEST_VERSION):
            raise ValueError(f"unsupported manifest version {self.manifest.get('version')}")
        if self.manifest["version"] == 1 and any(s.get("scale_lane_divisor", 1) != 1 for s in self.manifest["slabs"]):
            raise ValueError("shared scale runs require manifest version 2")
        if self.manifest["version"] < 3 and any(s.get("scale_order", "lane") != "lane" for s in self.manifest["slabs"]):
            raise ValueError("payload scale order requires manifest version 3")
        self._mm = np.memmap(self.dir / self.manifest["pack"], dtype=np.uint8, mode="r")
        self.slabs = {s["name"]: s for s in self.manifest["slabs"]}
        self.aux = {a["name"]: a for a in self.manifest["aux"]}

    def slab_info(self, name: str) -> PackInfo:
        s = self.slabs[name]
        return PackInfo(s["format"], s["n"], s["k"], s["rows"], s["unit_bytes"], s["payload_bytes"], s["scale_bytes"],
                        s["lane_order"], s["n_blocks"], 1.0, s["scale_group"], s.get("scale_placement", "inline"),
                        int(s.get("scale_unit_bytes", 8 if (s["format"] == "int4_affine" and s["scale_bytes"]) else 0)),   # older INT4 packs: FP32 pairs
                        s.get("scale_dtype", "bf16" if (s["format"] == "int4_affine" and s["scale_bytes"] and int(s.get("scale_unit_bytes", 8)) == 4) else ""),   # 4-byte pairs before scale_dtype: BF16
                        int(s.get("scale_lane_divisor", 1)), s.get("scale_order", "lane"))

    def slab_bytes(self, name: str) -> np.ndarray:
        s = self.slabs[name]
        return self._mm[s["offset"]: s["offset"] + s["nbytes"]]

    def row_scales(self, name: str) -> np.ndarray:
        s = self.slabs[name]
        return np.frombuffer(self._mm[s["row_scales_offset"]: s["row_scales_offset"] + 4 * s["n"]], dtype=np.float32)

    def uniform_row_scale_bits(self, name: str) -> Optional[int]:
        """An immutable slab's common finite scale, or None when loads are needed.

        Compare representations so mixed signed zeros cannot become one constant.
        Read the actual scale table; no checkpoint- or format-specific assumption.
        """
        bits = self.row_scales(name).view(np.uint32)
        if not bits.size:
            return None
        first = int(bits[0])
        return first if first & 0x7F800000 != 0x7F800000 and np.all(bits == first) else None

    def aux_array(self, name: str) -> np.ndarray:
        a = self.aux[name]
        raw = self._mm[a["offset"]: a["offset"] + a["nbytes"]]
        return raw.view(_NP_OF[a["dtype"]]).reshape(a["shape"])

    def dequantize_slab(self, name: str) -> np.ndarray:
        """The float32 matrix a slab encodes (rows in pack order), row scales applied — the kernel's semantics."""
        info = self.slab_info(name)
        fmt = FORMATS.get(info.format)
        spec = fmt.unpack_pack(self.slab_bytes(name).tobytes(), info)
        return (fmt.dequantize(spec) * self.row_scales(name)[:, None]).astype(np.float32)
