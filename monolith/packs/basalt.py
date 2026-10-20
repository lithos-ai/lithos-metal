"""A ``.basalt`` model file as a weight source beside :class:`~monolith.packs.packer.PackFile` (OI-013): the tensors
basalt's kernels read as shipped, each an absolute byte range of the file, by IR name. ``emit_program`` takes a source
wherever it takes a pack and binds each weight or constant value the source holds through a read-only window of the
file (``compiler/emit.py``, ``basalt_windows``), without a copy, as packs bind their slabs. Basalt copies these bytes
into CUDA allocations of its own; the bytes, and so every kernel's arithmetic, are the same.

IR names and what each entry holds (``BasaltEntry``: absolute ``offset``, ``nbytes``, ggml ``type_id``, ``shape``):

* ``gguf/<name>``: a ``meta["tensors"]`` tensor (``BasaltFile.tensor``), its shape ``ne`` reversed (row-major:
  ``(N, K)`` for a ``[K, N]`` GGUF matrix, ``(K,)`` for a vector);
* ``dense/<name>``: a ``dense.index`` row basalt copies to its GPU arena byte for byte, kinds 2 (F32), 4 (BF16) and
  5 (F16) (basalt's weights.cpp), shape ``(ne1, ne0)``, or ``(ne0,)`` when ``ne1`` is 0;
* ``mtp/<name>``: an ``mtp/dense.index`` tensor (``f32``, ``bf16``, ``q8_0``), shape ``(rows, cols)``;
* ``<section>``: a whole section (``mtp/experts``, ``experts`` …) as bytes, ``type_id`` None, shape ``(nbytes,)``.

A value binds an entry only when it declares it exactly: a weight's format, or a constant's dtype, is the entry's
ggml type name (``u8`` for a section), and its shape is the entry's. Refused when a value binds it, with the reason: a
``dense`` row with no ggml type a value can declare (an ``in_gguf`` row, which ``gguf/<name>`` serves; kind 0, basalt's
verbatim planes; kinds 1 and 3, which basalt converts at load) or with source bytes that do not fill its arena slot
exactly (basalt copies every source byte into the slot), a hyper-connection down/up matrix (``held_as_hc8``: basalt's
kernels read only its 8-bit copy, which the load quantizes from these bytes), a type the format plugins lack, a size
its type and shape do not give, an entry of no bytes. Two entries of one name make the file unreadable.

``copy=True`` is a test mode: each bound tensor becomes a buffer of exactly its own bytes, so shader validation bounds
every access by its tensor, not by a page-rounded window that holds its neighbours too. The bytes are the same.
"""

from __future__ import annotations

import math
import os
from collections import namedtuple
from typing import Dict, Optional, Sequence

from ..formats.basalt_reader import MTP_KINDS, BasaltFile
from ..formats.gguf import GGML_TYPES, ggml_row_bytes

BasaltEntry = namedtuple("BasaltEntry", "name offset nbytes type_id shape")
DENSE_KINDS = {2: 0, 4: 30, 5: 1}                 # dense.index kinds copied as stored -> ggml type id (F32, BF16, F16)
HC8 = ("basalt's kernels read this hyper-connection matrix only through its 8-bit copy, which the load makes from the "
       "stored form and then drops it (generate.cpp:2233-2244, :2319-2368 for the target; mtp.cpp:215-218, :254-260 "
       "for the drafter; hc8_only() is true at hc_fused.cu:1250): only a port of that load-time quantizer may read it")


def held_as_hc8(name: str, mtp: Optional[tuple] = None) -> bool:
    """Whether basalt leaves this matrix out of its arena on one GPU and computes with its 8-bit hyper-connection copy
    instead: the target's ``blk.*.hc_{attn,ffn}_{down,up}.weight`` and ``output_hc_{down,up}.weight``
    (``generate.cpp:2235-2243``), or, for a drafter tensor ``mtp = (kind, rows, cols, nbytes)``, a bf16
    ``hyper_connection`` ``input_mix_weight_`` matrix of exactly ``rows * cols * 2`` bytes (``mtp.cpp:215-218``)."""
    if mtp is not None:
        kind, rows, cols, nbytes = mtp
        return (kind == "bf16" and "hyper_connection" in name and "input_mix_weight_" in name
                and rows * cols * 2 == nbytes)
    return ((name.startswith("blk.") and name.endswith((".hc_attn_down.weight", ".hc_attn_up.weight",
                                                         ".hc_ffn_down.weight", ".hc_ffn_up.weight")))
            or name in ("output_hc_down.weight", "output_hc_up.weight"))


class BasaltSource:
    """The bindable tensors of a ``.basalt`` file. ``file`` is the opened file (``BasaltFile(path)`` by default; any
    ``BasaltFile`` over the same bytes works, such as a range reader); ``path`` is the file a program reads: the
    windows map it, and the copy mode reads each tensor from it, so both modes serve the same bytes."""

    slabs: Dict[str, dict] = {}                    # no block-lane-major slabs and no pack aux: the emitter's pack paths
    aux: Dict[str, dict] = {}                      # find nothing here, and only its window planner reads a source

    def __init__(self, path: str | os.PathLike, file: Optional[BasaltFile] = None, *, copy: bool = False) -> None:
        self.path = os.path.realpath(path)
        self.file = f = file if file is not None else BasaltFile(self.path)
        self.file_bytes = f.file_bytes
        self.copy = copy
        self.entries: Dict[str, BasaltEntry] = {}
        self.refused: Dict[str, str] = {}          # IR name -> why a value cannot bind it
        if "meta" in f.sections and "gguf" in f.sections:
            for name in f.meta().get("tensors", {}):
                try:
                    t = f.tensor(name)
                except ValueError as err:                          # a tensor the reader refuses is refused alone
                    self.refused[f"gguf/{name}"] = str(err)
                    continue
                self._add(f"gguf/{name}", t.offset, t.nbytes, t.type_id, tuple(reversed(t.ne)),
                          HC8 if held_as_hc8(name) else None)
        if "dense.index" in f.sections:
            base = f.section("dense").offset
            for r in f.dense_index().rows:
                why = (HC8 if held_as_hc8(r.name) else
                       f"its bytes are in gguf: bind gguf/{r.name}" if r.in_gguf else
                       f"kind {r.kind} has no ggml type to declare (kind 0 is basalt's verbatim planes; it converts "
                       "kinds 1 and 3 at load)" if r.kind not in DENSE_KINDS else
                       f"its row gives {r.src_bytes} source bytes but a {r.dst_bytes}-byte slot (basalt's loader "
                       "copies every source byte into the slot)" if r.src_bytes != r.dst_bytes else None)
                self._add(f"dense/{r.name}", base + r.src_off, r.src_bytes, DENSE_KINDS.get(r.kind),
                          (r.ne1, r.ne0) if r.ne1 else (r.ne0,), why)
        if "mtp/dense.index" in f.sections:
            base = f.section("mtp/dense").offset
            for t in f.mtp_dense_index():
                self._add(f"mtp/{t.name}", base + t.offset, t.nbytes, MTP_KINDS.get(t.kind), (t.rows, t.cols),
                          HC8 if held_as_hc8(t.name, (t.kind, t.rows, t.cols, t.nbytes)) else
                          None if t.kind in MTP_KINDS else f"mtp kind {t.kind!r} has no format here")
        for s in f.sections.values():
            self._add(s.name, s.offset, s.nbytes, None, (s.nbytes,))

    def _add(self, name: str, offset: int, nbytes: int, type_id: Optional[int], shape: Sequence[int],
             why: Optional[str] = None) -> None:
        if name in self:
            raise ValueError(f"{self.path}: two entries are named {name!r}")
        if why is None and type_id is not None:                              # a section (no type) is bytes
            if type_id not in GGML_TYPES:
                why = f"ggml type {type_id} has no format here"
            elif nbytes != ggml_row_bytes(type_id, shape[-1]) * math.prod(shape[:-1]):    # exact: no int64 wrap
                why = f"{nbytes} bytes are not {GGML_TYPES[type_id][0]} {list(shape)}"
        if why is None and not nbytes:
            why = "it holds no bytes"
        if why is None:
            self.entries[name] = BasaltEntry(name, offset, nbytes, type_id, tuple(shape))
        else:
            self.refused[name] = why

    def __contains__(self, name: str) -> bool:
        return name in self.entries or name in self.refused

    def bind(self, name: str, kind: str, shape: Sequence[int]) -> BasaltEntry:
        """The entry a value named ``name`` binds: ``kind`` is a weight's format or a constant's dtype (``u8`` for a
        section), ``shape`` the value's static shape. Refuses an entry that cannot be bound and a value that does not
        declare its entry exactly."""
        if name in self.refused:
            raise ValueError(f"{self.path}: {name!r} cannot be bound: {self.refused[name]}")
        e = self.entries[name]
        want = "u8" if e.type_id is None else GGML_TYPES[e.type_id][0]
        if (kind, tuple(shape)) != (want, e.shape):
            raise ValueError(f"{self.path}: {name!r} is {want} {list(e.shape)}; the graph declares {kind} {list(shape)}")
        return e

    def read(self, e: BasaltEntry) -> bytes:
        """``e``'s bytes from ``path``; refuses a file that ends inside them."""
        with open(self.path, "rb") as fh:
            fh.seek(e.offset)
            data = fh.read(e.nbytes)
        if len(data) != e.nbytes:
            raise ValueError(f"{self.path} ends inside {e.name!r} ({len(data)} of its {e.nbytes} bytes)")
        return data
