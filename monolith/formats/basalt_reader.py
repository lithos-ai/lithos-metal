# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
"""Reader for basalt's ``.basalt`` model file (github.com/jesdga95/basalt, MIT): the layout of
include/basalt/core/model_format.hpp, whose Python side is basalt's tools/model_file.py, with the checks basalt's own
readers make (src/core/model_file.cpp, src/core/weights.cpp, src/kernels/cpu/expert_layout.cpp,
src/core/expert_source.cpp, src/core/mtp.cpp).
Where a looser parse could still decode a plausible weight from a wrong offset, it is stricter and also refuses:

* a section named twice (basalt takes the first);
* a negative offset or size (basalt reads a negative ``meta`` number as its magnitude, and an index's as an unsigned
  one that can wrap past its bounds checks);
* a ``meta`` tensor without ``bytes`` (basalt's converter always writes it; basalt's reader treats it as optional);
* a ``dense.index`` that is not v3 (basalt does not check the version), a row of more than 19 fields (basalt refuses
  fewer and ignores the extra ones), or a header whose tensor count is not the table's;
* an ``experts.layout`` layer listed twice (basalt keeps the last) or a header total the layers do not fill;
* an ``mtp/dense.index`` line of more than 6 fields (basalt refuses fewer and ignores the extra ones), or one whose
  size disagrees with its kind and shape.

Little-endian. A 64 KiB header — ``FileHeader`` (magic ``BASALTMF``, version 1, section count, file size) and one
128-byte ``SectionEntry`` per section (NUL-padded name, offset, size, SHA-256) — then the sections, each at a multiple of
2 MiB. Sections are found by name; this module parses:

* ``meta`` — JSON. ``meta["tensors"][name]`` is every non-expert tensor of the source GGUF files: its ggml type id, its
  GGUF shape (``ne0`` first, the contiguous axis), its size and its offset inside the ``gguf`` section (``tensor``).
* ``dense.index`` v3 — the ``dense`` pool's table: ``# align A pool P tensors T``, then 19 fields per tensor
  (``DenseRow``). A row with ``code_bits != 0`` and no bytes (``in_gguf``) holds a shape only: the tensor is served
  from ``gguf`` and its placeholder fields (``8 0 32``) are not its format; ``meta`` has the type.
* ``experts.layout`` v5 — ``layer gu_type d_type offset blob_bytes`` per layer: each layer's ``n_expert`` routed expert
  blobs start at ``offset`` in ``experts`` and are ``blob_bytes`` each, an expert's GGUF rows ``[gate | up | down]``
  back to back (``ExpertsLayout.matrices``). The routed experts are not in ``meta``.
* ``mtp/dense.index`` — the drafter's dense tensors: ``name kind rows cols offset bytes`` (``kind`` ``f32``, ``bf16``
  or ``q8_0``; offsets into ``mtp/dense``).

The other sections (``tokenizer/*``, ``gguf``, ``dense``, ``experts``, ``mtp/experts``, ``profile``, ``vision/*``) are
bytes by name. ``mtp/experts`` is not GGUF blocks: basalt writes its Q2_0 experts planar (all codes, then all fp16
scales; gate and up rows interleaved), see basalt's include/basalt/kernels/cpu/expert.hpp.
"""

# adapted from jesdga95/basalt tools/model_file.py @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT,
# third_party/BASALT-LICENSE): the layout constants and the FileHeader / SectionEntry structs

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections import namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gguf import GGML_TYPES, ggml_row_bytes

MAGIC = b"BASALTMF"
VERSION = 1
SECTION_ALIGN = 2 << 20
HEADER_BYTES = 64 << 10
HEADER = struct.Struct("<8sIIQ40s")           # magic, version, n_sections, file_bytes, reserved
ENTRY = struct.Struct("<64sQQ32s16s")         # name, offset, bytes, sha256, reserved
MAX_SECTIONS = (HEADER_BYTES - HEADER.size) // ENTRY.size
EXPERTS_LAYOUT_VERSION = 5                     # the newest basalt writes and reads (kExpertLayoutVersion)
MTP_KINDS = {"f32": 0, "bf16": 30, "q8_0": 8}  # mtp/dense.index kind -> ggml type id

Section = namedtuple("Section", "name offset nbytes sha256")
GgufTensor = namedtuple("GgufTensor", "name type_id ne offset nbytes")      # offset: from the start of the file
DenseIndex = namedtuple("DenseIndex", "version align pool rows")
ExpertLayer = namedtuple("ExpertLayer", "layer gu_type d_type offset blob_bytes")
MtpTensor = namedtuple("MtpTensor", "name kind rows cols offset nbytes")


def parse_header(head: bytes) -> Tuple[int, Dict[str, Section]]:
    """``(file_bytes, sections by name)`` from the file's first bytes (the header and its table at least)."""
    if len(head) < HEADER.size:
        raise ValueError("not a .basalt model file (too short)")
    magic, version, n, file_bytes, _ = HEADER.unpack_from(head)
    if magic != MAGIC:
        raise ValueError("not a .basalt model file (bad magic)")
    if version != VERSION:
        raise ValueError(f"model file version {version}, this reader reads {VERSION}")
    if n > MAX_SECTIONS or len(head) < HEADER.size + n * ENTRY.size:
        raise ValueError("the section table is too long or cut short")
    sections: Dict[str, Section] = {}
    for i in range(n):
        raw, off, size, digest, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
        name = raw.split(b"\0", 1)[0].decode()
        if off % SECTION_ALIGN or off < HEADER_BYTES or off > file_bytes or size > file_bytes - off:
            raise ValueError(f"section {name!r} lies outside the file or is misaligned")
        if name in sections:
            raise ValueError(f"section {name!r} appears twice")
        sections[name] = Section(name, off, size, digest)
    return file_bytes, sections


class DenseRow(namedtuple("DenseRow", "name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias "
                                      "group_elems codebook has_offset codes_bytes scales_bytes offset_bytes "
                                      "scales_fp16 act_kind")):
    """One ``dense.index`` row, basalt's ``IndexRow`` field for field. ``kind``: 0 verbatim, 1 BF16 promoted to F32,
    2 F32, 3 F16 promoted to F32, 4 BF16 and 5 F16 as stored; offsets are into ``dense``."""

    @property
    def in_gguf(self) -> bool:
        """A shape-only row: the tensor's bytes are in ``gguf`` (``meta`` has its type)."""
        return self.code_bits != 0 and self.dst_bytes == 0


def _check_loaded_row(r: DenseRow, dense_bytes: int) -> None:
    """weights.cpp's ``WeightTable::load`` on a row it copies out of ``dense`` (lines 284-386)."""
    if r.file != 0:
        raise ValueError(f"dense.index: {r.name} names file {r.file}; a .basalt pack has file 0 (dense) only")
    if r.codes_bytes:                                        # codes, scales (fp16 ones widen to f32), then offsets
        if r.scales_fp16 and r.scales_bytes % 4:
            raise ValueError(f"dense.index: {r.name}: an fp16 scale plane of an odd byte count")
        scales = r.scales_bytes // 2 if r.scales_fp16 else r.scales_bytes
        planes = (r.codes_bytes + scales + r.offset_bytes, r.codes_bytes + r.scales_bytes + r.offset_bytes)
    elif r.kind in (1, 3):                                   # BF16 or F16 promoted to F32: 4 bytes in, 2 out
        planes = (4 * r.ne0 * max(r.ne1, 1), 2 * r.ne0 * max(r.ne1, 1))
    else:
        planes = (r.src_bytes, r.dst_bytes)
    if planes != (r.src_bytes, r.dst_bytes):
        raise ValueError(f"dense.index: {r.name} is {r.src_bytes}/{r.dst_bytes} bytes in/out, its planes make "
                         f"{planes[0]}/{planes[1]}")
    if r.src_bytes and not 0 <= r.src_off <= r.src_off + r.src_bytes <= dense_bytes:     # an empty row reads nothing
        raise ValueError(f"dense.index: {r.name} runs past the end of the dense section")


def parse_dense_index(text: str, dense_bytes: int) -> DenseIndex:
    """``dense.index`` v3 over a ``dense`` section of ``dense_bytes``. A row that is not exactly 19 fields is refused
    (basalt refuses fewer): a misparsed row is a wrong byte offset that decodes to a plausible weight. A row basalt
    loads (any but an ``in_gguf`` one) must name file 0, have planes that add up to its sizes and lie in ``dense``."""
    version = header = None
    rows: List[DenseRow] = []
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.match(r"# dense index v(\d+)", line)
            version = int(m.group(1)) if m else version
            m = re.match(r"# align (\d+) pool (\d+) tensors (\d+)", line)
            header = tuple(int(g) for g in m.groups()) if m else header
            continue
        f = line.split()
        if len(f) != 19:
            raise ValueError(f"dense.index: a row of {len(f)} of 19 fields: {line!r}")
        rows.append(DenseRow(f[0], *(int(x) for x in f[1:])))
    if version != 3:
        raise ValueError(f"dense.index: version {version}, this reader reads v3")
    if header is None or not header[1] or not rows:
        raise ValueError("dense.index: no '# align … pool …' header or no rows")
    if header[2] != len(rows):
        raise ValueError(f"dense.index: the header counts {header[2]} tensors, the table has {len(rows)}")
    for r in rows:
        if not r.in_gguf:
            _check_loaded_row(r, dense_bytes)
    return DenseIndex(version, header[0], header[1], tuple(rows))


class ExpertsLayout(namedtuple("ExpertsLayout", "version n_expert total layers")):
    """``experts.layout``: ``layers[l]`` is layer ``l``'s :data:`ExpertLayer`; offsets are into ``experts``."""

    def _layer(self, layer: int) -> ExpertLayer:
        if not 0 <= layer < len(self.layers):
            raise IndexError(f"layer {layer} of {len(self.layers)}")
        return self.layers[layer]

    def blob_offset(self, layer: int, expert: int) -> int:
        if not 0 <= expert < self.n_expert:
            raise IndexError(f"expert {expert} of {self.n_expert}")
        e = self._layer(layer)
        return e.offset + expert * e.blob_bytes

    def matrices(self, layer: int, n_embd: int, n_ff: int) -> Tuple[Tuple[int, int, int, int], ...]:
        """``(type id, byte offset in the blob, rows, values per row)`` of the blob's gate, up and down matrices —
        ``[n_ff, n_embd]``, ``[n_ff, n_embd]``, ``[n_embd, n_ff]`` as GGUF rows (basalt's native_fmt). The model's
        sizes are the caller's; a blob they do not fill exactly is refused."""
        e = self._layer(layer)
        gu_row, d_row = ggml_row_bytes(e.gu_type, n_embd), ggml_row_bytes(e.d_type, n_ff)
        up = gu_row * n_ff
        if 2 * up + d_row * n_embd != e.blob_bytes:
            raise ValueError(f"experts.layout: layer {layer}'s {e.blob_bytes}-byte blob is not gate/up/down of "
                             f"{n_embd} x {n_ff} in types {e.gu_type}/{e.d_type}")
        return (e.gu_type, 0, n_ff, n_embd), (e.gu_type, up, n_ff, n_embd), (e.d_type, 2 * up, n_embd, n_ff)


def parse_experts_layout(text: str, experts_bytes: int, n_expert: Optional[int] = None) -> ExpertsLayout:
    """``experts.layout`` up to v5 over an ``experts`` section of ``experts_bytes``; the header's ``n_expert`` wins
    over the argument (a pruned pack ships fewer). Columns past the fifth are ignored, as basalt ignores them. The
    layers must tile the section from offset 0 and fill it, or fill all of it but one blob of padding: basalt's
    default (arena) expert source takes either, its ``--mmap-experts`` one the first only (expert_source.cpp)."""
    version, total, layers = 0, None, {}
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.match(r"# experts v(\d+)", line)
            if m:
                version = int(m.group(1))
                if version > EXPERTS_LAYOUT_VERSION:
                    raise ValueError(f"experts.layout v{version}: this reader reads up to v{EXPERTS_LAYOUT_VERSION}")
            m = re.search(r"\(n_expert (\d+)", line)
            n_expert = int(m.group(1)) if m else n_expert
            m = re.search(r"total (\d+)\)", line)
            total = int(m.group(1)) if m else total
            continue
        f = line.split()
        if not f:
            continue
        try:
            layer = ExpertLayer(*(int(x) for x in f[:5]))
        except (TypeError, ValueError):
            raise ValueError(f"experts.layout: a malformed line: {line!r}") from None
        if min(layer) < 0 or layer.layer in layers:
            raise ValueError(f"experts.layout: a malformed line: {line!r}")
        layers[layer.layer] = layer
    if not n_expert or not layers:
        raise ValueError("experts.layout: no expert count or no layers")
    at = 0
    for i in range(len(layers)):
        if i not in layers or layers[i].offset != at:
            raise ValueError(f"experts.layout: layer {i} is missing or not contiguous")
        at += layers[i].blob_bytes * n_expert
    if total is not None and total != at:
        raise ValueError(f"experts.layout: the layers fill {at} bytes, the header says {total}")
    if experts_bytes not in (at, at + max(e.blob_bytes for e in layers.values())):
        raise ValueError(f"experts.layout: the layers fill {at} bytes, the experts section is {experts_bytes}")
    return ExpertsLayout(version, n_expert, at, tuple(layers[i] for i in range(len(layers))))


def parse_mtp_dense_index(text: str, dense_bytes: int) -> Tuple[MtpTensor, ...]:
    """``mtp/dense.index`` over an ``mtp/dense`` section of ``dense_bytes``: six fields per line, the size of a known
    kind matching its rows and columns, every tensor inside the section (mtp.cpp's ``MtpDrafter::load``)."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        try:
            if len(f) != 6:
                raise ValueError
            t = MtpTensor(f[0], f[1], *(int(x) for x in f[2:]))
        except ValueError:
            raise ValueError(f"mtp/dense.index: a malformed line: {line!r}") from None
        if t.kind in MTP_KINDS and t.nbytes != t.rows * ggml_row_bytes(MTP_KINDS[t.kind], t.cols):
            raise ValueError(f"mtp/dense.index: {t.name} is {t.nbytes} bytes, not {t.rows} x {t.cols} {t.kind}")
        if not 0 <= t.offset <= t.offset + t.nbytes <= dense_bytes:
            raise ValueError(f"mtp/dense.index: {t.name} names bytes past the mtp/dense section")
        out.append(t)
    return tuple(out)


class BasaltFile:
    """A ``.basalt`` file. ``BasaltFile(path)`` maps it; any buffer with a length that slices to bytes (``bytes``, a
    range reader over HTTP) works the same way. ``data`` is that buffer."""

    def __init__(self, source: Any) -> None:
        self.data = np.memmap(source, dtype=np.uint8, mode="r") if isinstance(source, (str, os.PathLike)) else source
        if len(self.data) < HEADER_BYTES:
            raise ValueError("not a .basalt model file (too short)")
        self.file_bytes, self.sections = parse_header(bytes(self.data[:HEADER_BYTES]))
        if len(self.data) < self.file_bytes:
            raise ValueError(f".basalt file truncated ({len(self.data)} of {self.file_bytes} bytes)")
        self._meta: Optional[Dict[str, Any]] = None

    def section(self, name: str) -> Section:
        if name not in self.sections:
            raise KeyError(f"the .basalt file has no {name!r} section")
        return self.sections[name]

    def view(self, name: str) -> Sequence[int]:
        s = self.section(name)
        return self.data[s.offset: s.offset + s.nbytes]

    def read(self, name: str) -> bytes:
        return bytes(self.view(name))

    def verify(self, name: str, chunk: int = 64 << 20) -> bool:
        """The section's bytes against the SHA-256 in the table (basalt's engine never checks it)."""
        s, h = self.section(name), hashlib.sha256()
        for at in range(s.offset, s.offset + s.nbytes, chunk):
            h.update(bytes(self.data[at: min(at + chunk, s.offset + s.nbytes)]))
        return h.digest() == s.sha256

    def meta(self) -> Dict[str, Any]:
        if self._meta is None:
            self._meta = json.loads(self.read("meta"))
        return self._meta

    def tensor(self, name: str) -> GgufTensor:
        """A ``gguf``-section tensor, its offset from the start of the file; the size must match its type and shape."""
        t, g = self.meta()["tensors"][name], self.section("gguf")
        try:
            ne, off, nbytes = tuple(int(x) for x in t["shape"]), int(t["offset"]), int(t["bytes"])
            type_id = int(t["type_id"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"meta.tensors[{name!r}] is malformed (a field is missing or not an integer)") from None
        if not 1 <= len(ne) <= 4:
            raise ValueError(f"meta.tensors[{name!r}] is malformed (a shape of rank {len(ne)})")
        if not 0 <= off <= off + nbytes <= g.nbytes:
            raise ValueError(f"meta.tensors[{name!r}] runs past the gguf section")
        if type_id in GGML_TYPES and nbytes != ggml_row_bytes(type_id, ne[0]) * int(np.prod(ne[1:], dtype=np.int64)):
            raise ValueError(f"meta.tensors[{name!r}]: {nbytes} bytes do not match type {type_id} and shape {list(ne)}")
        return GgufTensor(name, type_id, ne, g.offset + off, nbytes)

    def dense_index(self) -> DenseIndex:
        return parse_dense_index(self.read("dense.index").decode(), self.section("dense").nbytes)

    def experts_layout(self) -> ExpertsLayout:
        return parse_experts_layout(self.read("experts.layout").decode(), self.section("experts").nbytes)

    def mtp_dense_index(self) -> Tuple[MtpTensor, ...]:
        return parse_mtp_dense_index(self.read("mtp/dense.index").decode(), self.section("mtp/dense").nbytes)
