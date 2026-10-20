# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1171-1227 (hc8_quant_kernel, hc8_make),
# include/basalt/core/model_file.hpp:126-149 and src/core/model_file.cpp:438-485 (the BASALTH8 table),
# src/core/verify.cpp:566-581 and src/core/mtp.cpp:215-218, :488-503 (the copied matrices), src/core/verify.cpp:638-640
# and src/core/mtp.cpp:516-521 (their order in hc8_dump's table) @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
# third_party/NOTICE).
"""Basalt's int8 copies of the hyper-connection matrices, made on the host.

A copy of a BF16 matrix ``w [rows][cols]`` (``cols % 32 == 0``) is one block of bytes, the allocation ``hc8_make``
fills (``src/kernels/cuda/hc_fused.cu:1210-1227``): the codes ``q [rows][cols]`` as int8, row-major, then the scales
``s [rows][srow]`` as f16, ``srow = (cols / 32 + 7) & ~7``. Scale ``j`` of row ``r`` covers ``w[r][32 j .. 32 j + 31]``:
``s = f16_rne(absmax / 127)`` and ``q = clamp(rint(w / s), -127, 127)``, every code of the group 0 where ``s`` is 0, and
the ``srow - cols / 32`` padding scales of each row are 0. ``hc8_quant_kernel`` (``hc_fused.cu:1171-1197``) computes it
in IEEE binary32: subnormals kept, both divisions correctly rounded, both roundings to nearest even. ``quantize`` is
that arithmetic in numpy, operation for operation; ``docs/research/basalt-nvcc/hc_fused.md`` section 1 records what
nvcc compiles it to.

On one GPU basalt quantizes every hyper-connection down, up and injection matrix of the target and of the drafter at
load, from the BF16 bytes its model file stores (``src/program/generate.cpp:2233-2365``, ``src/core/mtp.cpp:254-282``,
``:488-503``). It holds the down and up matrices only as their copies, while the injections stay BF16 beside theirs
(``generate.cpp:2360``, ``mtp.cpp:215-218``). Every hyper-connection read then takes the copies (decode's fused read,
``src/kernels/cuda/hc_fused.cu:1430-1431``, and ``hc8_gr_read_multi``, ``src/kernels/cuda/hc8_gr.cu:250``, ``:279``)
except the target prompt path's injection product, which multiplies the BF16 matrix (``src/prefill/prefill.cpp:1572``;
``src/prefill/gemm.cu:136`` takes a copy only when it is ``owned``, the matrix's only form, as the down and up copies
are: ``hc_fused.cu:1274``, ``:1302``). The port makes the copies on the host, once per ``.basalt`` file:

* Exact without a flush argument: the host CPU computes binary32 with subnormals kept, the kernel's own arithmetic,
  where a Metal pass would flush them.
* No new runtime mechanism: the repository has no run-once setup program, and packs prepared on the host exist already.
* Memory: the GPU binds only the copies, never the target's 1,271,398,400 bytes of BF16 down and up matrices beside the
  decode program.

The copies pack is a :class:`~monolith.packs.packer.PackFile` directory. Its data file ``weights.pack`` is basalt's
``BASALTH8`` table as ``hc8_write`` lays it out (``include/basalt/core/model_file.hpp:126-149``,
``src/core/model_file.cpp:438-485``): a 16-byte header (``b"BASALTH8"``, version 1 and the entry count as ``uint32``),
160-byte entries (the name NUL-padded to 128 bytes; rows, cols, srow and 0 as ``uint32``; offset and bytes as
``uint64``), each copy's bytes at a 256-aligned offset with zero fill between, and the file zero-padded to a multiple
of 16384 bytes, because the runtime maps whole pages only. Its manifest lists two aux rows per copy, under basalt's
table name (``blk.L.hc_attn_down.weight`` and the like; ``mtp.`` then the tensor's name for the drafter's):
``hc8/<name>/q``, ``U8 [rows, cols]`` at the entry's offset, and ``hc8/<name>/s``, ``F16 [rows, srow]`` at that
offset plus ``rows * cols``.
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter, namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np

from ..formats.basalt_reader import BasaltFile
from .packer import ALIGN, PackFile

G8 = 32                                           # values per scale group (hc_fused.cu:343)
Hc8Entry = namedtuple("Hc8Entry", "name src offset rows cols")   # offset: of the BF16 bytes, from the file's start
TARGET = re.compile(r"blk\.(\d+)\.hc_(attn|ffn)_(down|up|inject)\.weight")       # hc8_tensors (verify.cpp:566-581)
OUTPUT = ("output_hc_down.weight", "output_hc_up.weight")                        # its last two (:578-579)
DRAFTER = tuple(p + m for p in ("attn_hyper_connection.", "mlp_hyper_connection.", "hyper_connection_mixer.")
                for m in ("input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight"))


def quantize(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The copy of the BF16 matrix whose bits are ``w`` (``uint16 [rows, cols]``): ``(q, s)``, ``q`` the codes' int8
    bit patterns as ``uint8 [rows, cols]`` and ``s`` the scales' f16 bits as ``uint16 [rows, srow]``, padding scales
    0, so that ``q.tobytes() + s.tobytes()`` is basalt's allocation. Raises ``ValueError`` unless ``w`` is a 2-D
    ``uint16`` array with ``rows`` and ``cols`` positive, ``cols % 32 == 0`` and every weight finite. A finite weight
    above ``127 * 65504`` is basalt's case too: its group's scale is inf and every code of the group 0."""
    if not (isinstance(w, np.ndarray) and w.dtype == np.uint16 and w.ndim == 2):
        raise ValueError(f"quantize: the weights must be a 2-D uint16 array of BF16 bits, not "
                         f"{type(w).__name__} {getattr(w, 'dtype', '')} of shape {getattr(w, 'shape', '')}")
    rows, cols = w.shape
    if rows == 0 or cols == 0 or cols % G8:
        raise ValueError(f"quantize: a {rows} x {cols} matrix; rows and cols must be positive and cols a multiple of "
                         f"{G8}, so that a group of {G8} never straddles a row")
    bad = (w & 0x7F80) == 0x7F80
    if bad.any():
        r, c = np.argwhere(bad)[0]
        raise ValueError(f"quantize: {int(bad.sum())} non-finite weights, the first at row {r}, column {c} "
                         f"(bits 0x{int(w[r, c]):04X}); a valid pack's hyper-connection matrices have none")
    gpr = cols // G8
    x = (w.astype(np.uint32) << 16).view(np.float32).reshape(rows * gpr, G8)    # :1178 the BF16 bits as fp32
    amax = np.abs(x).max(axis=1)                  # :1179 the fmaxf chain from 0: exact on finite weights
    with np.errstate(over="ignore"):              # above f16's range: inf, basalt's scale too
        h = (amax / np.float32(127)).astype(np.float16)                          # :1181 div.rn.f32, cvt.rn.f16.f32
    s = h.astype(np.float32)                      # :1182 exact
    with np.errstate(divide="ignore", invalid="ignore"):                         # s == 0: the quotients are unused
        v = np.where(s[:, None] > 0, np.clip(np.rint(x / s[:, None]), -127, 127), 0)  # :1185 div.rn, rni, max, min
    q = v.astype(np.int8).view(np.uint8).reshape(rows, cols)                    # :1186 integral, in [-127, 127]
    sc = np.zeros((rows, (gpr + 7) & ~7), np.uint16)                            # hc8_make's memset (:1223)
    sc[:, :gpr] = h.view(np.uint16).reshape(rows, gpr)                          # :1183
    return q, sc


def copy_list(f: BasaltFile) -> Tuple[Hc8Entry, ...]:
    """The matrices basalt copies on one GPU, in the order of the table ``hc8_dump`` writes
    (``src/core/verify.cpp:619-645``), from the file's index sections alone (no weight byte is read). ``name`` is
    basalt's table name, ``src`` the IR name of the BF16 row (for messages), ``offset`` its byte offset in the file,
    ``rows`` x ``cols`` the copy's shape. The target's come first, in ``hc8_tensors``' order (``verify.cpp:566-581``,
    ``:638-639``): per layer, ascending, ``blk.L.hc_attn_{down,up,inject}.weight`` then
    ``blk.L.hc_ffn_{down,up,inject}.weight``, then ``output_hc_{down,up}.weight``, each a ``dense/`` row of ``ne1`` x
    ``ne0`` (``generate.cpp:2326``, ``:2350``). Then the drafter's (``verify.cpp:640``), named ``"mtp." + name``: the
    ``input_mix_weight_down``, ``input_mix_weight_up`` and ``block_inject_weight`` tensors of its attention's, MLP's and
    final mixer's hyper-connections (``mtp.cpp:488-503``), in ``mtp/dense.index``'s line order, the order in which
    ``hc8_keys`` lists them (``mtp.cpp:516-521``) and the load copies the down and up matrices (``:257-267``). A matrix
    the file lacks is left out, as basalt's ``prepare_hc8`` leaves it (``verify.cpp:604``); a graph that reads its copy
    is refused by ``emit_program``, where basalt's read ends the process (``hc_fused.cu:1442-1445``).

    Raises ``ValueError`` naming the row or cause for a file with an ``hc8`` section, a target row that is in ``gguf``,
    not kind 4 (BF16 as stored) or not ``ne0 * ne1 * 2`` bytes, a drafter row not ``bf16`` of ``rows * cols * 2`` bytes,
    a copied matrix that two rows of its index name (``BasaltSource`` refuses two entries of one name,
    ``monolith/packs/basalt.py:106``; basalt would copy the first of two injections, ``mtp.cpp:189-193``, ``:495``, and
    both of two down or up matrices, ``:257-267``), an empty matrix (``rows`` or ``cols`` 0, of which basalt makes no
    copy: ``hc_fused.cu:1253``, ``:1270``), ``cols % 32 != 0``, and a table name of 128 bytes or more (``hc8_write``'s
    limit, ``src/core/model_file.cpp:443``)."""
    if "hc8" in f.sections:
        raise ValueError("copy_list: the file has an 'hc8' section, a table of copies basalt would adopt as it stands: "
                         "this port builds its own copies from the BF16 rows and does not read a precomputed hc8 table")
    out = []

    def add(name, src, offset, rows, cols, why):
        why = why or (f"{rows} x {cols}: an empty matrix" if not rows or not cols else
                      f"{rows} x {cols}: its columns are not whole groups of {G8}" if cols % G8 else
                      f"a table name of {len(name.encode())} bytes; hc8_write's hold at most 127"
                      if len(name.encode()) >= 128 else None)
        if why:
            raise ValueError(f"copy_list: {src}: {why}")
        out.append(Hc8Entry(name, src, offset, rows, cols))

    dense = f.dense_index().rows
    rows, named, base = {r.name: r for r in dense}, Counter(r.name for r in dense), f.section("dense").offset
    hc = sorted((int(m[1]), ("attn", "ffn").index(m[2]), ("down", "up", "inject").index(m[3]), n)
                for n in rows if (m := TARGET.fullmatch(n)))
    for n in [n for *_, n in hc] + [n for n in OUTPUT if n in rows]:
        r = rows[n]
        add(n, f"dense/{n}", base + r.src_off, r.ne1, r.ne0,
            f"{named[n]} rows of dense.index are named {n!r}" if named[n] > 1 else
            "its bytes are in gguf" if r.in_gguf else
            f"kind {r.kind}, not 4 (BF16 as stored)" if r.kind != 4 else
            f"{r.src_bytes} bytes, not {r.ne1} x {r.ne0} BF16" if r.src_bytes != r.ne0 * r.ne1 * 2 else None)
    mtp, base = (f.mtp_dense_index(), f.section("mtp/dense").offset) \
        if "mtp/dense.index" in f.sections else ((), 0)                          # a file with no drafter
    named = Counter(t.name for t in mtp)
    for t in mtp:                                 # the index's line order, hc8_keys' (mtp.cpp:516-521)
        if t.name in DRAFTER:
            add("mtp." + t.name, f"mtp/{t.name}", base + t.offset, t.rows, t.cols,
                f"{named[t.name]} rows of mtp/dense.index are named {t.name!r}" if named[t.name] > 1 else
                f"kind {t.kind!r}, not 'bf16'" if t.kind != "bf16" else
                f"{t.nbytes} bytes, not {t.rows} x {t.cols} BF16" if t.nbytes != t.rows * t.cols * 2 else None)
    return tuple(out)


def write_pack(f: BasaltFile, directory) -> PackFile:
    """The copies pack of the ``.basalt`` file ``f``, written into the new directory ``directory`` (an existing one is
    refused): every matrix of ``copy_list(f)`` read from its BF16 bytes and quantized, one at a time. ``weights.pack``
    is basalt's ``BASALTH8`` table as ``hc8_write`` lays it out (``src/core/model_file.cpp:438-485``), zero-padded to
    a multiple of 16384 bytes; ``manifest.json``, written last, lists each copy's two aux rows (the module's
    docstring). ``ValueError`` as ``copy_list`` and ``quantize`` raise it."""
    es = copy_list(f)
    out = Path(directory)
    out.mkdir(parents=True)                       # FileExistsError if it exists: a pack is never written over
    table, aux, at = [], [], (16 + 160 * len(es) + 255) & ~255                   # the first payload (:440)
    for e in es:
        srow, codes = (e.cols // G8 + 7) & ~7, e.rows * e.cols
        nbytes = codes + e.rows * srow * 2                                       # :444
        table.append(struct.pack("<128sIIIIQQ", e.name.encode(), e.rows, e.cols, srow, 0, at, nbytes))  # :446-452
        aux += [dict(name=f"hc8/{e.name}/q", offset=at, nbytes=codes, dtype="U8", shape=[e.rows, e.cols]),
                dict(name=f"hc8/{e.name}/s", offset=at + codes, nbytes=nbytes - codes, dtype="F16",
                     shape=[e.rows, srow])]
        at = (at + nbytes + 255) & ~255                                          # :453
    with open(out / "weights.pack", "wb") as fh:
        fh.write(struct.pack("<8sII", b"BASALTH8", 1, len(es)) + b"".join(table))           # :455-463
        for e, row in zip(es, aux[::2]):
            fh.write(bytes(row["offset"] - fh.tell()))                          # the zero fill (:465-474)
            w = np.frombuffer(f.data[e.offset: e.offset + 2 * e.rows * e.cols], np.uint16)  # little-endian, as the host
            q, s = quantize(w.reshape(e.rows, e.cols))
            fh.write(q.tobytes() + s.tobytes())
        size = -(-fh.tell() // ALIGN) * ALIGN    # the runtime maps whole pages only; hc8_read reads only each entry
        fh.write(bytes(size - fh.tell()))
    (out / "manifest.json").write_text(json.dumps({"version": 2, "alignment": ALIGN, "pack": "weights.pack",
                                                   "nbytes": size, "slabs": [], "aux": aux}, indent=1))
    return PackFile(out)
