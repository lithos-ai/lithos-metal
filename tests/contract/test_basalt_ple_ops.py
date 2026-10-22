# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) tests/core/native_dense_ple_key_test.cpp and src/kernels/ple_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): native_dense_ple_key_test's key rows (:62-78)
# and its three cases (:116-171: a BF16 key row, a shape-only one, the index's edge cases) as the key's binding, and
# ple_parity's prelaunch refusals (:215-220, :715-739, :869-891) as the emitter's.
"""basalt's n-gram block on the decode path (monolith/ops/basalt_ple.py) as compiler/emit_basalt.py emits it: the window
op ``basalt_ple_block`` and the commit op ``basalt_ple_commit``, over the pack's six tensors declared as the op contract
gives them, from a synthetic .basalt file that holds them under the pack's names and types (``write_file``). The GPU
side is tests/kernels/test_basalt_ple_program.py.

- REG ``test_registered_and_covered``: both kinds are ``OpClass.MAP``, each bound to the kernel of its own name, and
  the coverage guard passes on the window graph and on the commit graph.
- DISP ``test_dispatches_are_the_builders``: at T = 1 .. 8 the window is 11 T dispatches in the plan's order
  (``STAGES``), each with its builder's kernel, grid, threads and params record (the records' fields pinned,
  ``LAUNCHES``), its slots and ``writes=``, row r's row bindings at r times their row bytes; no StepState. The commit
  is one dispatch, ``basalt_ple_restore`` over the history, the snapshots and the record, no params record and no
  StepState, the same at every width as at 8.
- X-PROJ ``test_projections_are_main_ops``: every row's q8_1, key and value dispatches are what main's handlers emit for
  a one-column graph on the same tensors, the row's input at r times its row bytes.
- BUF ``test_buffers``: ``hist`` and ``snap`` are states of one size at every width and in the commit's program, the
  row I/O T rows, the seven scratch buffers of the plan's sizes, and the six named values in buffers of their own; the
  record ``commit`` is a 40-byte host-written buffer of the commit's program only.
- BAR ``test_barriers``: every two dispatches in conflict at buffer granularity, write-after-read included, have a
  barrier between them.
- X-REF ``test_refusals``: one case for each refusal of both kinds (``REFUSALS``): a valid op, then the same op with
  exactly one defect, refused by that defect's own check (its message). No graph reaches the others, so they have no
  case: a misaligned ``hist``, ``snap`` or commit record (states and inputs bind at byte 0 of their own buffers, a
  StepState field at a whole word), ``hist`` or ``snap`` sharing a buffer (a state cannot be viewed), and basalt's
  overlap (H2) and scratch cases (C2, C7): the emitter binds the history and owns the scratch buffers.
- X-BIND-1 ``test_key_binds_gguf_type42``: the key's shape-only dense row is refused (its bytes are in gguf), and the
  block's key binds ``gguf/blk.1.ple_key.weight`` as type 42 ``[10240, 2560]`` at the tensor's bytes, on the
  synthetic file and on the pack's own dense.index (its row 46) with the key's gguf record.
- X-BIND-2 ``test_bf16_key_row_is_refused``: with a BF16 dense key row, the gguf key is refused (basalt serves the key
  from that row) and so is the row as the key (the port builds no BF16 key arm).
- X-BIND-3 ``test_missing_key_is_refused``: no key in either name space, or a key from a file with no dense.index,
  fails naming the key; a dense.index without the key's row keeps the gguf key."""

import json
import math
import re
import struct

import pytest

from monolith import kernels
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import basalt_windows
from monolith.compiler.emit_basalt import GGML_IDS
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.core.shapes import T
from monolith.formats.gguf import ggml_row_bytes
from monolith.ops import OPS
from monolith.packs.basalt import BasaltSource
from tests.basalt_ple_ref import D, HIST, KERN, N, T_MAX, TENSORS, committed
from tests.contract.test_basalt_ops import emit, gemv, op, pack, q8_1, spliced, weight
from tests.contract.test_basalt_reader import write_basalt
from tests.contract.test_basalt_windows import PROFILE, real_source

F32 = DType.F32
KEY, VALUE, TAPS = "gguf/blk.1.ple_key.weight", "dense/blk.1.ple_value.weight", "dense/blk.1.ple_conv1d.weight"
NORMS = tuple(f"dense/blk.1.ple_norm_{n}.weight" for n in ("key", "query", "conv"))
DENSE_KEY = "dense/blk.1.ple_key.weight"


def write_file(path, data, key_row="shape", index=True):
    """A .basalt file holding the pack's six n-gram tensors as the pack ships them (tests.basalt_ple_ref.TENSORS; the
    pack's dense.index rows 45-50): all six in the gguf section with their ggml types, and in dense.index the key's
    shape-only row and the other five as dense rows of their kinds (5 F16, 2 F32, 4 BF16) of the same bytes, each
    tensor at the next multiple of 64 of its section. ``data`` maps each tensor's name to its bytes; a tensor without
    them is in neither section. The key's other forms (X-BIND): ``key_row="bf16"`` writes its row as zero BF16 bytes of
    its shape (kind 4, code_bits 0, basalt's unquantized key row: native_dense_ple_key_test.cpp:62-78), ``None`` no
    row; ``index=False`` writes no dense.index."""
    gguf, tensors, dense, rows = bytearray(), {}, bytearray(), []
    for name, t in TENSORS.items():
        ne0, ne1 = (*t.ne, 0)[:2]
        blob = data.get(name)
        if blob is not None:
            assert len(blob) == t.nbytes, name
            gguf += bytes(-len(gguf) % 64)
            tensors[name] = {"type_id": t.type_id, "shape": list(t.ne), "offset": len(gguf), "bytes": t.nbytes}
            gguf += blob
        kind = t.kind
        if t.dense is None:                                                  # the key: its bytes are in gguf
            if key_row == "shape":                                           # and a shape only in dense.index
                rows.append(f"{name} 0 0 0 0 {len(dense)} 0 {ne0} {ne1} 8 0 32 0 0 0 0 0 0 0")
            if key_row != "bf16":
                continue
            kind, blob = 4, bytes(2 * ne0 * ne1)
        if blob is None:
            continue
        dense += bytes(-len(dense) % 64)
        at = len(dense)
        rows.append(f"{name} 0 {kind} {at} {len(blob)} {at} {len(blob)} {ne0} {ne1} 0 0 1 0 0 0 0 0 0 0")
        dense += blob
    meta = {"format": "basalt", "version": 1, "tensors": tensors}
    sections = {"meta": json.dumps(meta).encode(), "gguf": bytes(gguf), "dense": bytes(dense)}
    if index:
        sections["dense.index"] = (f"# dense index v3\n# align 64 pool {len(dense)} tensors {len(rows)}\n"
                                   + "\n".join(rows) + "\n").encode()
    path.write_bytes(write_basalt(sections))
    return path


def zeros():
    """The six tensors as zero bytes (the planner and the emitter never read them)."""
    return {name: bytes(t.nbytes) for name, t in TENSORS.items()}


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return BasaltSource(write_file(tmp_path_factory.mktemp("ple") / "ple.basalt", zeros()))


# ---- graph builders ----------------------------------------------------------------------------------------------

SPECS = {  # role: (graph method, name, shape, dtype or format), each value as the op contract declares it
    "rows": ("input", "ngram.rows", (T, N), F32), "hidden": ("input", "ngram.hidden", (T, D), F32),
    "key": ("weight", KEY, (D, N), "q2_0"), "value": ("const", VALUE, (N, N), DType.BF16),
    **{f"norm_{n}": ("const", name, (D,), F32) for n, name in zip(("key", "query", "conv"), NORMS)},
    "taps": ("const", TAPS, (D, KERN), DType.F16),
    "hist": ("state", "ngram.hist", (D, HIST), F32), "snap": ("state", "ngram.snap", (T_MAX, D, HIST), F32),
    "result": ("value", "ngram.result", (T, D), F32), "normalized": ("value", "ngram.normalized", (T, D), F32),
    "commit": ("input", "commit", (2 + T_MAX,), DType.I32),
}
BLOCK_IN = ("rows", "hidden", "key", "value", "norm_key", "norm_query", "norm_conv", "taps", "hist", "snap")


def declare(g, role, **change):
    """``role``'s value as ``SPECS`` declares it, with ``change``'s ``method``, ``name``, ``shape`` or ``dtype``."""
    spec = dict(zip(("method", "name", "shape", "dtype"), SPECS[role]), **change)
    return getattr(g, spec["method"])(spec["name"], spec["shape"], spec["dtype"])


def block(g, **over):
    """The window: ``basalt_ple_block`` over the T rows of ``ngram.rows`` and ``ngram.hidden``, with the key, the
    value, the three norms and the taps declared as they bind (``gguf/`` a weight, ``dense/`` a constant), and
    ``hist`` and ``snap`` one size in every program, the windows' and the commit's, so they are shared by name.
    ``over`` replaces a role's value or the ``updates`` (the refusals)."""
    ins = [over[r] if r in over else declare(g, r) for r in BLOCK_IN]
    outs = [over[r] if r in over else declare(g, r) for r in ("result", "normalized")]
    return op(g, "basalt_ple_block", ins, outs, updates=over.get("updates", [ins[8].name, ins[9].name]))


def commit(g, **over):
    """The commit: ``basalt_ple_commit`` restores ``hist`` from the snapshot that word 1 of the record ``commit``
    names, into a placeholder output; ``over`` as ``block``'s."""
    ins = [over[r] if r in over else declare(g, r) for r in ("hist", "snap", "commit")]
    return op(g, "basalt_ple_commit", ins, [g.value("ngram.commit", (1,), DType.U32)],
              updates=over.get("updates", [ins[0].name]))


def window(src, t):
    """The window's graph and its program at T = t."""
    g = Graph("step")
    block(g)
    return g, emit(g, src, t)


def commit_program(src, t):
    """The commit's program at T = t."""
    g = Graph("step")
    commit(g)
    return emit(g, src, t)


# ---- registration ------------------------------------------------------------------------------------------------

def test_registered_and_covered():
    for build in (block, commit):
        g = Graph("step")
        kind = build(g).kind
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
        assert [o.kind for o in g.ops] == [kind]
        check_coverage(g, PROFILE)


# ---- the window's dispatches -------------------------------------------------------------------------------------

STAGES = [  # the plan's eleven dispatches a row (§3): stage, function, the values in slots 0, 1, ..., writes=
    ("q8_1", "native_quantize_q8_1", ("rows", "ple.q8_1"), [1]),
    ("key", "basalt_mmvq", (KEY, "ple.q8_1", "ple.key"), [2]),
    ("value", "basalt_mmvf", ("rows", VALUE, "ple.value"), [2]),
    ("norm_key", "basalt_gr_norm", ("ple.key", NORMS[0], "ple.key_norm"), [2]),
    ("norm_query", "basalt_gr_norm", ("hidden", NORMS[1], "normalized"), [2]),
    ("gate", "basalt_ple_gate", ("ple.key_norm", "normalized", "ple.gate"), [2]),
    ("broadcast", "basalt_ple_broadcast", ("ple.value", "ple.gate", "ple.gated"), [2]),
    ("norm_conv", "basalt_gr_norm", ("ple.gated", NORMS[2], "normalized"), [2]),
    ("conv", "basalt_ple_conv", ("hist", "normalized", TAPS, "hidden", "ple.gated", "ple.conv", "result"), [5, 6]),
    ("advance", "basalt_ple_advance", ("hist", "normalized"), [0]),
    ("snapshot", "basalt_ple_snapshot", ("snap", "hist"), [0]),
]
LAUNCHES = {  # the plan's launches at every width (§3): macros, (threadgroups, threads), params fields or None
    "native_quantize_q8_1": ({}, (10, 256), ("<4I", (2560, 0, 0, 0))),
    "basalt_mmvq": ({"MMVQ_TYPE": "42", "NCOLS": "1", "ROWS": "4", "NW": "4"}, (2560, 128),
                    ("<4I", (2560, 10240, 0, 0))),
    "basalt_mmvf": ({"BLOCK": "256", "NT": "1"}, (2560, 256), ("<8I", (2560, 2560, 1, 2560, 2560, 0, 0, 0))),
    "basalt_gr_norm": ({"NORM_BLOCK": "1024"}, (4, 1024), ("<4I", (2560, 4, 0x358637BD, 0))),    # eps f32(1e-6)
    "basalt_ple_gate": ({}, (4, 512), ("<4I", (0x3CA1E89B, 0, 0, 0))),                          # 1 / sqrt(2560)
    "basalt_ple_broadcast": ({}, (40, 256), None),
    "basalt_ple_conv": ({}, (40, 256), None),
    "basalt_ple_advance": ({}, (40, 256), None),
    "basalt_ple_snapshot": ({}, (64, 256), None),
    "basalt_ple_restore": ({}, (64, 256), None),                                                # the commit's
}
ROW_BYTES = {"rows": 10240, "hidden": 40960, "result": 40960, "normalized": 40960, "snap": 368640}
SCRATCH = {"ple.q8_1": 2880, "ple.key": 40960, "ple.key_norm": 40960, "ple.value": 10240, "ple.gate": 16,
           "ple.gated": 40960, "ple.conv": 40960}
OWN = ("rows", "hidden", "result", "normalized", "hist", "snap")      # the values that must not share a buffer


def builders():
    """Each kernel's (source, macros, (threadgroups, threads), params record or None) from monolith/kernels.py's
    builders: the quantizer, the key and the value at one column whatever the width, the norms over 4 rows."""
    mq, mf = kernels.basalt_mmvq_macros(42, N, 1), kernels.basalt_mmvf_macros(N, N, 1)
    mn, ple = kernels.basalt_gr_norm_macros(N), kernels.basalt_ple_source()
    return {
        "native_quantize_q8_1": (kernels.basalt_q8_1_source(), {}, kernels.basalt_q8_1_grid(N),
                                 kernels.basalt_q8_1_params(N)),
        "basalt_mmvq": (kernels.basalt_mmvq_source(), mq, kernels.basalt_mmvq_grid(mq, D),
                        kernels.basalt_mmvq_params(N, D)),
        "basalt_mmvf": (kernels.basalt_mmvf_source(), mf, kernels.basalt_mmvf_grid(mf, N),
                        kernels.basalt_mmvf_params(N, N, 1)),
        "basalt_gr_norm": (kernels.basalt_gr_norm_source(), mn, kernels.basalt_gr_norm_grid(4, mn),
                           kernels.basalt_gr_norm_params(N, 4, kernels.PLE_RMS_EPS)),
        "basalt_ple_gate": (ple, {}, kernels.PLE_GRIDS["gate"], kernels.basalt_ple_gate_params()),
        **{f"basalt_ple_{s}": (ple, {}, kernels.PLE_GRIDS[s], None)
           for s in ("broadcast", "conv", "advance", "snapshot", "restore")},
    }


def rows_of(prog, t):
    """``(r, STAGES entry)`` of each dispatch in order, after checking the names: 11 T, row by row in STAGES' order."""
    assert [o.name for o in prog.ops] == [f"basalt_ple_block:{r}:{s[0]}" for r in range(t) for s in STAGES]
    return [(r, s) for r in range(t) for s in STAGES]


def scratch_of(prog, t):
    """The program's scratch buffers by role: ``ws.T<t>.ple.<role>.<n>`` as ``ple.<role>``."""
    return {"ple." + b.split(".")[3]: b for b in prog.buffers if b.startswith(f"ws.T{t}.ple.")}


@pytest.mark.parametrize("t", range(1, 9))
def test_dispatches_are_the_builders(src, t):
    """DISP, the window: 11 T dispatches in STAGES' order, each with its builder's kernel (function, macros, source),
    grid, threads and params record (made once: five records, nine pipelines), the builders' values the plan's
    (``LAUNCHES``, the records' fields by struct.unpack_from), each slot bound to its value (a weight at its window, a
    scratch buffer at 0, row r of a row binding at r times its row bytes), the plan's ``writes=``; no StepState and
    no threadgroup memory. The commit: one dispatch, ``basalt_ple_restore`` (one pipeline, no params record) bound to
    hist, snap and the record at byte 0, writing hist, the same op and pipeline at every width as at 8."""
    g, prog = window(src, t)
    _, bound = basalt_windows(src, g.values.values(), t)
    built = builders()
    assert set(built) == set(LAUNCHES)
    for fn, (macros, grid, fields) in LAUNCHES.items():
        _, m, gr, prm = built[fn]
        assert (m, gr, prm is None) == (macros, grid, fields is None), fn
        assert fields is None or struct.unpack_from(fields[0], prm) == fields[1], fn
    scratch = scratch_of(prog, t)

    def at(v, r):
        return (scratch[v], 0) if v in scratch else bound[v] if v in bound else (f"ngram.{v}", r * ROW_BYTES.get(v, 0))

    for o, (r, (stage, fn, values, writes)) in zip(prog.ops, rows_of(prog, t)):
        source, macros, (groups, threads), prm = built[fn]
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, macros, spliced(source)), o.name
        assert (o.grid, o.threadgroup, o.threadgroup_memory) == ((groups, 1, 1), (threads, 1, 1), []), o.name
        want = [(i, *at(v, r)) for i, v in enumerate(values)]
        if prm is not None:
            slot, record, off = o.bindings[-1]
            assert (slot, off, prog.buffers[record].role, prog.buffers[record].init) == (len(values), 0, "params",
                                                                                         prm), o.name
            want.append(o.bindings[-1])
        assert o.bindings == want, o.name
        assert o.meta["writes"] == writes, o.name
    assert prog.step_state not in {b for o in prog.ops for _, b, _ in o.bindings}
    assert sum(b.role == "params" for b in prog.buffers.values()) == 5 and len(prog.kernels) == 9
    c, c8 = commit_program(src, t), commit_program(src, 8)                          # the commit: one dispatch
    (o,), (o8,) = c.ops, c8.ops
    source, macros, (groups, threads), _ = built["basalt_ple_restore"]
    k = c.kernels[o.kernel]
    assert (o.name, k.function, k.macros, k.source) == ("basalt_ple_commit:restore", "basalt_ple_restore", macros,
                                                        spliced(source))
    assert (o.grid, o.threadgroup, o.threadgroup_memory) == ((groups, 1, 1), (threads, 1, 1), [])
    assert o.bindings == [(0, "ngram.hist", 0), (1, "ngram.snap", 0), (2, "commit", 0)] and o.meta["writes"] == [0]
    assert len(c.kernels) == 1 and not any(b.role == "params" for b in c.buffers.values())
    assert (o, k) == (o8, c8.kernels[o8.kernel])                                    # the same at every width


@pytest.mark.parametrize("t", range(1, 9))
def test_projections_are_main_ops(src, t):
    """X-PROJ: every row's q8_1, key and value dispatches are the dispatches main's ``_q8_1``, ``_mmvq`` and ``_mmvf``
    emit for a one-column graph of the same three ops on the same tensors (basalt's per-row call): the same pipeline
    (its key, source, macros and function), grid, threads, written slots and params bytes, each weight at the same
    bytes of the file, the row's input at r times its row bytes where main binds row 0, every other slot a buffer of
    the same role and size at 0."""
    _, prog = window(src, t)
    g = Graph("step")
    x = g.input("ngram.rows", (T, N), F32)
    gemv(g, "basalt_mmvq", "key", q8_1(g, "xq", [x]), weight(g, KEY, src))
    gemv(g, "basalt_mmvf", "value", x, weight(g, VALUE, src))
    main = emit(g, src, 1)
    mains = dict(zip(("q8_1", "key", "value"), main.ops))
    assert [o.name.split(":")[0] for o in main.ops] == ["basalt_q8_1", "basalt_mmvq", "basalt_mmvf"]

    def reaches(p, b, off):     # a binding as what it reads or writes: a window's bytes of the file, or a buffer
        spec = p.buffers[b]
        return (spec.role, spec.file, spec.file_offset + off) if spec.role == "weights" else (spec.role, spec.nbytes,
                                                                                              spec.init, off)

    ops = {o.name: o for o in prog.ops}
    for r in range(t):
        for stage, m in mains.items():
            o = ops[f"basalt_ple_block:{r}:{stage}"]
            assert (o.kernel, prog.kernels[o.kernel]) == (m.kernel, main.kernels[m.kernel]), o.name
            assert (o.grid, o.threadgroup, o.threadgroup_memory, o.meta["writes"]) == (
                m.grid, m.threadgroup, m.threadgroup_memory, m.meta["writes"]), o.name
            assert [i for i, _, _ in o.bindings] == [i for i, _, _ in m.bindings], o.name
            for (_, b, off), (_, mb, moff) in zip(o.bindings, m.bindings):
                if mb == "ngram.rows":
                    assert (b, off) == (mb, moff + 4 * N * r), o.name
                else:
                    assert reaches(prog, b, off) == reaches(main, mb, moff), (o.name, b, mb)


@pytest.mark.parametrize("t", range(1, 9))
def test_buffers(src, t):
    """BUF, the window: ``hist`` (368,640 B) and ``snap`` (2,949,120 B) are states of one size at every width, so the
    programs share them by name; the rows are T rows of 10,240 B and the hidden rows and the outputs T rows of 40,960 B,
    host- or GPU-written arenas; the scratch is the plan's seven buffers at its sizes, one each, which every row reuses;
    and rows, hidden, result, normalized, hist and snap are bound to six buffers of their own. The commit's program
    holds hist and snap at the window's sizes, and the record ``commit`` as a 40-byte host-written arena that no window
    program has."""
    _, prog = window(src, t)
    sizes = {n: (b.role, b.nbytes) for n, b in prog.buffers.items()}
    assert {v: sizes[f"ngram.{v}"] for v in OWN} == {
        "rows": ("arena", t * 10240), "hidden": ("arena", t * 40960), "result": ("arena", t * 40960),
        "normalized": ("arena", t * 40960), "hist": ("state", 368640), "snap": ("state", 2949120)}
    assert {v: sizes[b] for v, b in scratch_of(prog, t).items()} == {v: ("arena", n) for v, n in SCRATCH.items()}
    assert len([b for b in prog.buffers if b.startswith("ws.")]) == len(SCRATCH)
    bound = {v: set() for v in OWN}
    for o, (_, (_, _, values, _)) in zip(prog.ops, rows_of(prog, t)):
        for (_, b, _), v in zip(o.bindings, values):
            if v in bound:
                bound[v].add(b)
    assert all(len(b) == 1 for b in bound.values()) and len(set().union(*bound.values())) == len(OWN), bound
    c = commit_program(src, t)                         # the commit's program: the same states, and the record its own
    assert {n: (b.role, b.nbytes) for n, b in c.buffers.items() if n in ("ngram.hist", "ngram.snap", "commit")} == {
        "ngram.hist": sizes["ngram.hist"], "ngram.snap": sizes["ngram.snap"], "commit": ("arena", 40)}
    assert "commit" not in prog.buffers


@pytest.mark.parametrize("t", range(1, 9))
def test_barriers(src, t):
    """BAR: two dispatches conflict when one writes a buffer the other reads or writes (buffer granularity, the reads
    and writes the plan's ``writes=`` give each dispatch's bindings), write-after-read included; every such pair has a
    barrier between them, on the later one or on a dispatch between them (a barrier makes its dispatch, and every one
    after it, wait for every one before it: monolith/compiler/barriers.py)."""
    _, prog = window(src, t)
    access = []
    for o, (_, (_, _, _, writes)) in zip(prog.ops, rows_of(prog, t)):
        w = {b for i, b, _ in o.bindings if i in writes}
        access.append(({b for _, b, _ in o.bindings} - w, w))
    last, war = -1, 0                                                   # the latest barrier at or before dispatch j
    for j, (rj, wj) in enumerate(access):
        last = j if prog.ops[j].barrier_before else last
        for i, (ri, wi) in enumerate(access[:j]):
            if wj & (ri | wi) or rj & wi:
                assert last > i, (prog.ops[i].name, prog.ops[j].name)
                war += not (wj & wi or rj & wi)                         # only a write after a read
    assert prog.ops[0].barrier_before and war, war


# ---- the refusals (X-REF) ----------------------------------------------------------------------------------------

def nbytes(v):
    """A pack entry's bytes for ``v``: a weight's rows as stored, a constant's values."""
    if v.is_weight:
        return v.shape[0] * ggml_row_bytes(GGML_IDS[v.format], v.shape[1])
    return math.prod(v.shape) * v.dtype.itemsize


def _dynamic(build):
    def case(g, src, bad, tmp):
        build(g)
        return {"dynamic_t": bad}
    return case


def _bad(build, make):
    """``build`` (``block`` or ``commit``) with the values or ``updates`` ``make(g)`` gives, in the bad case only."""
    def case(g, src, bad, tmp):
        build(g, **(make(g) if bad else {}))
        return {}
    return case


def _entry(build, role, at, slab=(False, False), **change):
    """``build`` with ``role`` declared as ``p`` (its ``SPECS`` declaration with ``change``), read from a pack entry of
    exactly its bytes at byte ``at[bad]`` (test_basalt_ops.pack: the only route to a misaligned binding), a
    block-lane-major slab where ``slab[bad]``; where ``at[bad]`` is None, ``role`` as ``SPECS`` declares it."""
    def case(g, src, bad, tmp):
        if at[bad] is None:
            build(g)
            return {}
        v = declare(g, role, name="p", **change)
        build(g, **{role: v})
        return {"packs": [pack(tmp, {"p": (at[bad], nbytes(v))}, slab=slab[bad])]}
    return case


def _nine_rows(g, src, bad, tmp):
    """The window's four row values at 9 static rows, or at basalt's most, 8."""
    block(g, **{r: declare(g, r, shape=(9 if bad else 8, SPECS[r][2][1]))
                for r in ("rows", "hidden", "result", "normalized")})
    return {}


def _views(g):
    """``result`` and ``normalized`` as two views of one value."""
    base = g.value("ngram.out", (4, D), F32)
    return {"result": g.view("ngram.result", base, 0, 2), "normalized": g.view("ngram.normalized", base, 2, 2)}


def _one_pack(g, src, bad, tmp):
    """``rows`` and ``hidden`` read from pack entries ``p`` and ``q``, of one pack (one window) or of a pack each."""
    block(g, rows=declare(g, "rows", method="const", name="p", shape=(2, N)),
          hidden=declare(g, "hidden", method="const", name="q", shape=(2, D)))
    entries = {"p": (0, 2 * N * 4), "q": (2 * N * 4, 2 * D * 4)}
    if bad:
        return {"packs": [pack(tmp, entries)]}
    return {"packs": [pack(tmp.with_name(f"{tmp.name}.{n}"), {n: e}) for n, e in entries.items()]}


B, C = "basalt_ple_block: ", "basalt_ple_commit: "
DYNAMIC = ("basalt's kernels take the window's width as a compiled count, one static program per width; a dynamic-T "
           "program cannot run them")
REFUSALS = [  # (id, build, the reason's regex): monolith/ops/basalt_ple.py's refusals of both kinds, in its order
    ("window dynamic-T", _dynamic(block), B + DYNAMIC),
    ("window of 9 rows", _nine_rows, B + r"9 rows; basalt's windows run 1 to 8 rows \(verify\.cpp:685-688\)"),
    ("key Q3_K (C7's type 11)", _entry(block, "key", (None, 0), dtype="q3_k"),
     B + r"the weight 'p' \(q3_k \[10240, 2560\]\) is not a weight or constant \[n_out, n_in\] of ggml type \['q2_0'\]"),
    ("key of 2560 rows", _entry(block, "key", (None, 0), shape=(N, N)),
     B + r"the key 'p' is q2_0 \[2560, 2560\], not \[10240, 2560\]"),
    ("key a block-lane-major slab", _entry(block, "key", (0, 0), (False, True)),
     B + r"the weight 'p' is a block-lane-major pack slab, not rows as stored"),
    ("value Q2_0 (the key's matrix)", _bad(block, lambda g: dict.fromkeys(("key", "value"), declare(g, "key"))),
     B + r"the weight 'gguf/blk\.1\.ple_key\.weight' \(q2_0 \[10240, 2560\]\) is not a weight or constant \[n_out, "
         r"n_in\] of ggml type \['bf16'\]"),
    ("value of 64 columns", _entry(block, "value", (None, 0), shape=(N, 64)),
     B + r"the value 'p' is bf16 \[2560, 64\], not \[2560, 2560\]"),
    *[(f"{r} 32 values wider", _bad(block, lambda g, r=r: {r: declare(g, r, shape=(T, SPECS[r][2][1] + 32))}),
       B + rf"'ngram\.{r}' is f32 \[2, {w + 32}\], not f32 \[2, {w}\] \({why}\)")
      for r, w, why in (("rows", N, "the window's gathered rows"), ("hidden", D, "the window's hidden rows"),
                        ("result", D, "the block's output rows"), ("normalized", D, "the conv norm's rows"))],
    ("norm F16", _entry(block, "norm_query", (None, 0), dtype=DType.F16),
     B + r"'p' is f16 \[10240\], not f32 \[10240\] \(a grouped norm's weights\)"),
    ("taps F32", _entry(block, "taps", (None, 0), dtype=F32),
     B + r"'p' is f32 \[10240, 4\], not f16 \[10240, 4\] \(the convolution's four taps a channel, k \+ 4c\)"),
    ("history an input", _bad(block, lambda g: {"hist": declare(g, "hist", method="input"), "updates": ["ngram.snap"]}),
     B + r"'ngram\.hist' \(the history\) is not a state"),
    ("history of 10 rows", _bad(block, lambda g: {"hist": declare(g, "hist", shape=(D, HIST + 1))}),
     B + r"'ngram\.hist' is f32 \[10240, 10\], not f32 \[10240, 9\] \(the history\)"),
    ("snapshots of 7 slots", _bad(block, lambda g: {"snap": declare(g, "snap", shape=(T_MAX - 1, D, HIST))}),
     B + r"'ngram\.snap' is f32 \[7, 10240, 9\], not f32 \[8, 10240, 9\] \(the snapshots\)"),
    ("history not in updates", _bad(block, lambda g: {"updates": ["ngram.snap"]}),
     B + r"'ngram\.hist' \(the history\) is not in updates; the window writes it in place"),
    ("snapshots not in updates", _bad(block, lambda g: {"updates": ["ngram.hist"]}),
     B + r"'ngram\.snap' \(the snapshots\) is not in updates; the window writes it in place"),
    ("snapshots the history", _bad(block, lambda g: dict.fromkeys(("hist", "snap"), declare(g, "hist"))),
     B + r"'ngram\.hist' is both the history and its snapshots"),
    ("rows off 8 bytes", _entry(block, "rows", (8, 4), method="const", shape=(2, N)),
     B + r"'p' binds at byte 4 of pack\.0; slot 0 needs 8-byte alignment"),
    ("hidden off 4 bytes", _entry(block, "hidden", (4, 2), method="const", shape=(2, D)),
     B + r"'p' binds at byte 2 of pack\.0; slot 0 needs 4-byte alignment"),
    ("norm off 4 bytes", _entry(block, "norm_conv", (4, 2)),
     B + r"'p' binds at byte 2 of pack\.0; slot 1 needs 4-byte alignment"),
    ("taps off 2 bytes", _entry(block, "taps", (2, 1)),
     B + r"'p' binds at byte 1 of pack\.0; slot 2 needs 2-byte alignment"),
    ("result and normalized in one buffer", _bad(block, _views),
     B + r"'ngram\.normalized' and 'ngram\.result' bind one buffer \(ngram\.out\)"),
    ("rows and hidden in one buffer", _one_pack, B + r"'q' and 'p' bind one buffer \(pack\.0\)"),
    ("commit dynamic-T", _dynamic(commit), C + DYNAMIC),
    ("commit history an input", _bad(commit, lambda g: {"hist": declare(g, "hist", method="input"), "updates": []}),
     C + r"'ngram\.hist' \(the history\) is not a state"),
    ("commit history of 10 rows", _bad(commit, lambda g: {"hist": declare(g, "hist", shape=(D, HIST + 1))}),
     C + r"'ngram\.hist' is f32 \[10240, 10\], not f32 \[10240, 9\] \(the history\)"),
    ("commit history not in updates", _bad(commit, lambda g: {"updates": []}),
     C + r"'ngram\.hist' \(the history\) is not in updates; the commit restores it in place"),
    ("commit snapshots an input", _bad(commit, lambda g: {"snap": declare(g, "snap", method="input")}),
     C + r"'ngram\.snap' \(the snapshots\) is not a state"),
    ("commit snapshots of 7 slots", _bad(commit, lambda g: {"snap": declare(g, "snap", shape=(T_MAX - 1, D, HIST))}),
     C + r"'ngram\.snap' is f32 \[7, 10240, 9\], not f32 \[8, 10240, 9\] \(the snapshots\)"),
    ("commit snapshots in updates", _bad(commit, lambda g: {"updates": ["ngram.hist", "ngram.snap"]}),
     C + r"'ngram\.snap' \(the snapshots\) is in updates; it only reads them"),
    ("commit snapshots the history", _bad(commit, lambda g: dict.fromkeys(("hist", "snap"), declare(g, "hist"))),
     C + r"'ngram\.hist' is both the history and its snapshots"),
    ("commit record F32", _bad(commit, lambda g: {"commit": declare(g, "commit", dtype=F32)}),
     C + r"'commit' \(f32 \[10\]\) is not the commit record, an i32 graph input of at least 2 words"),
    ("commit record of 1 word", _bad(commit, lambda g: {"commit": declare(g, "commit", shape=(1,))}),
     C + r"'commit' \(i32 \[1\]\) is not the commit record, an i32 graph input of at least 2 words"),
    ("commit record a constant", _entry(commit, "commit", (None, 0), method="const"),
     C + r"'p' \(i32 \[10\]\) is not the commit record, an i32 graph input of at least 2 words"),
    ("commit record a StepState field",
     _bad(commit, lambda g: {"commit": declare(g, "commit", name="tokens", shape=(T_MAX,))}),
     C + r"'tokens' names a StepState field; the commit record is a host-written input buffer of its own"),
]


@pytest.mark.parametrize("build, why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(src, tmp_path, build, why):
    """X-REF: each case emits as a valid op, then with its one defect is refused before anything is encoded, with its
    reason (``why``)."""
    for bad in (False, True):
        g = Graph("step")
        kw = build(g, src, bad, tmp_path / f"pack{int(bad)}")
        packs = kw.pop("packs", [])
        if not bad:
            emit(g, src, 2, packs, **kw)
            continue
        with pytest.raises(ValueError, match=why):
            emit(g, src, 2, packs, **kw)


# ---- the key's binding (X-BIND) ----------------------------------------------------------------------------------

def key_slot(prog):
    """``(file, absolute offset)`` that row 0's key dispatch reads its weight at."""
    (o,) = [o for o in prog.ops if o.name == "basalt_ple_block:0:key"]
    _, b, off = o.bindings[0]
    return prog.buffers[b].file, prog.buffers[b].file_offset + off


def test_key_binds_gguf_type42(src):
    """X-BIND-1 (native_dense_ple_key_test.cpp:145-158: with a shape-only key row the key stays the source tensor,
    served natively): the key's shape-only dense row is refused, its bytes being in gguf, and the block's key binds
    ``gguf/blk.1.ple_key.weight``, type 42 ``[10240, 2560]``, at the tensor's bytes. On the synthetic file, and on the
    pack's own dense.index (tests/contract/fixtures/basalt_pack.safetensors; the key's row, 46 of the table: kind 0, no
    bytes, code_bits 8) with the key's gguf record (basalt_ple.safetensors), at its offset in the pack."""
    real, fixture = real_source()
    rec = next(e for e in committed()[1]["entries"] if e["key"] == KEY)
    f = real.file
    f.meta()["tensors"][rec["tensor"]] = {"type_id": rec["type_id"], "shape": rec["ne"], "bytes": rec["bytes"],
                                          "offset": rec["offset"] - f.section("gguf").offset}    # into the cached meta
    real = BasaltSource(fixture["file"], f)
    rows = [r for r in f.dense_index().rows if r.name == rec["tensor"]]
    assert [(r.kind, r.src_bytes, r.dst_bytes, r.ne0, r.ne1, r.code_bits, r.in_gguf) for r in rows] == [
        (0, 0, 0, N, D, 8, True)]
    for s, at in ((src, src.file.tensor("blk.1.ple_key.weight").offset), (real, TENSORS["blk.1.ple_key.weight"].gguf)):
        assert s.refused[DENSE_KEY] == f"its bytes are in gguf: bind {KEY}"
        e = s.bind(KEY, "q2_0", (D, N))
        assert (e.type_id, e.shape, e.offset, e.nbytes) == (42, (D, N), at, TENSORS["blk.1.ple_key.weight"].nbytes)
        assert key_slot(window(s, 1)[1]) == (s.path, at)
        g = Graph("step")
        block(g, key=g.const(DENSE_KEY, (D, N), DType.BF16))
        with pytest.raises(ValueError, match=re.escape(f"{DENSE_KEY!r} cannot be bound: its bytes are in gguf: bind "
                                                       f"{KEY}")):
            emit(g, s, 1)


def test_bf16_key_row_is_refused(tmp_path):
    """X-BIND-2 (native_dense_ple_key_test.cpp:116-143: with a BF16 key row basalt keeps the key in its arena as that
    row and attaches no native key): with the key's dense row BF16 (kind 4, code_bits 0), basalt serves the key from
    the row, an arm this port does not build, so the block refuses the gguf key, and refuses the row itself as the key
    (BF16 where the key's arm takes type 42)."""
    s = BasaltSource(write_file(tmp_path / "bf16.basalt", zeros(), key_row="bf16"))
    assert (s.entries[DENSE_KEY].type_id, s.entries[DENSE_KEY].shape) == (30, (D, N)) and KEY in s.entries
    for over, why in (({}, r"the dense\.index holds 'blk\.1\.ple_key\.weight' unquantized \(code_bits 0\), so basalt "
                           r"serves the key from that row, an arm this port does not build"),
                      ({"key": (DENSE_KEY, DType.BF16)}, r"the weight 'dense/blk\.1\.ple_key\.weight' \(bf16 \[10240, "
                                                         r"2560\]\) is not a weight or constant \[n_out, n_in\] of "
                                                         r"ggml type \['q2_0'\]")):
        g = Graph("step")
        block(g, **{r: g.const(name, (D, N), dt) for r, (name, dt) in over.items()})
        with pytest.raises(ValueError, match=B + why):
            emit(g, s, 1)


def test_missing_key_is_refused(src, tmp_path):
    """X-BIND-3 (native_dense_ple_key_test.cpp:160-171, index_code_bits's edge cases): a file with no key in either
    name space fails the binding naming the key, declared either way; a key from a file with no dense.index is
    refused, as basalt's load fails when it cannot open the index (:166-167; src/core/weights.cpp:129-132), here with
    the other five bound from their gguf twins, which emit from the file that has one; and a dense.index without the
    key's row is no refusal: the row's code bits are -1 (:164-165), so the key stays the gguf tensor."""
    none = BasaltSource(write_file(tmp_path / "none.basalt", {n: b for n, b in zeros().items()
                                                              if n != "blk.1.ple_key.weight"}, key_row=None))
    for name, method, dt in ((KEY, "weight", "q2_0"), (DENSE_KEY, "const", DType.BF16)):
        g = Graph("step")
        block(g, key=declare(g, "key", method=method, name=name, dtype=dt))
        with pytest.raises(KeyError, match=re.escape(f"the pack has no entry for {name!r}")):
            emit(g, none, 1)

    def twins(s):                                                    # the five from the gguf section
        g = Graph("step")
        block(g, **{r: declare(g, r, name=SPECS[r][1].replace("dense/", "gguf/")) for r in BLOCK_IN[3:8]})
        return emit(g, s, 1)

    assert key_slot(twins(src)) == (src.path, src.file.tensor("blk.1.ple_key.weight").offset)
    no_index = BasaltSource(write_file(tmp_path / "no_index.basalt", zeros(), index=False))
    with pytest.raises(ValueError, match=B + re.escape(f"{no_index.path} holds {KEY!r} but no dense.index")):
        twins(no_index)
    no_row = BasaltSource(write_file(tmp_path / "no_row.basalt", zeros(), key_row=None))
    assert key_slot(window(no_row, 1)[1]) == (no_row.path, no_row.file.tensor("blk.1.ple_key.weight").offset)
