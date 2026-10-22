"""basalt's prompt-path op kinds (monolith/ops/basalt_prompt.py) as compiler/emit_basalt_prompt.py emits them over the
synthetic .basalt file of test_basalt_ops.py: both kinds registered and covered; every dispatch equal to kernels.py's
builders and the handlers' rules (function, macros, source, grid, threadgroup, params, written slot), with StepState at
slot 15 and the row macros of the program's form (prompt_rows); no per-step value in a params record (invariant 4);
the GEMM's static slices tiling every product once; Gemm::native's dequantization into the program's one FP16 scratch
ordered ahead of its products and behind the previous op's; and each refusal, a valid input with exactly one defect
refused by that defect's own check (its message). The GPU side is tests/kernels/test_basalt_prompt_program.py."""

import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import _Ctx, basalt_windows
from monolith.compiler.emit_basalt_prompt import prompt_rows
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.core.shapes import N_INJ, T
from monolith.core.step_state import StepStateLayout
from monolith.formats.gguf import ggml_row_bytes
from monolith.ops import OPS
from monolith.runtime.program import Program
from tests.contract.test_basalt_ops import emit, op, pack, src, weight  # noqa: F401  (src: the .basalt file fixture)
from tests.contract.test_basalt_windows import PROFILE

F32, F16, BF16 = DType.F32, DType.F16, DType.BF16
KINDS = ("basalt_prompt_gemm", "basalt_prompt_convert")
LAYOUT = StepStateLayout(t_max=128)
TILE_K2560 = 64 * 64 * 2560          # one 64 x 64 tile's multiply-adds at K 2560: a budget that slices rows and columns

# The products basalt_prompt_gemm runs on this pack, (N, K) (spec-kernels-gemm.md section 3; prefill.cpp lines):
# :1569 and :1571 the hc down and up (Gemm::bf16 on the copies' BF16 images, hc8's entry), :1572 the inject, :1585 and
# :1535 qkv and the PLE key, :1586 the GDN gate, :1587-1588 alpha and beta, :1592 and :1688 ssm_out and attn_output,
# :1605-1606, :1609 and :1698 at 512 x 2560, :1607 attn_q, :1608 the indexer's k, :1702-1703 the shared expert's gate and
# up, :1705 its down, :1707 its scalar gate, :1538 the PLE value; and N 4 and 72 at K 2560 (column tiles cut short).
PRODUCTS = ((320, 10240), (10240, 320), (4, 10240), (10240, 2560), (6144, 2560), (48, 2560), (2560, 6144), (512, 2560),
            (12288, 2560), (128, 2560), (640, 2560), (2560, 640), (1, 2560), (2560, 2560), (4, 2560), (72, 2560))


def prompt_graph(g, src, rows=T):
    """Both kinds at ``rows`` rows: to_f16 and to_bf16 of x [rows, 2560] F32 (the PLE's key and value operands of one
    row, prefill.cpp:1534, :1536), and Gemm::bf16 of xb [rows, 2560] BF16 by the file's two BF16 matrices (128 outputs:
    two column tiles; 1 output: one partial tile)."""
    x = g.input("x", (rows, 2560), F32)
    op(g, "basalt_prompt_convert", [x], [g.value("x16", (rows, 2560), F16)])
    op(g, "basalt_prompt_convert", [x], [g.value("xbf16", (rows, 2560), BF16)])
    xb = g.input("xb", (rows, 2560), BF16)
    for name in ("mmvf.q", "mmvf.gate"):
        w = weight(g, f"dense/{name}", src)
        op(g, "basalt_prompt_gemm", [xb, w], [g.value(f"y.{name}", (rows, w.shape[0]), F32)], entry="bf16")


def row_macros(rows, static):
    """The macros prompt_rows(st) reads: T_SRC 0 in a dynamic-T program, 2 with the count for a static one."""
    m = {"STEP_STATE": "1", "T_SRC": "2" if static else "0", "ROWS_MAX": f"{rows}u"}
    return {**m, "T_STATIC_ROWS": f"{rows}u"} if static else m


def record(prog, o):
    """The params record a dispatch binds (its one buffer of role params)."""
    (prm,) = [prog.buffers[b] for _, b, _ in o.bindings if prog.buffers[b].role == "params"]
    return prm.init


# ---- registration, and each dispatch against the builders ----------------------------------------------------------

def test_registered_and_covered(src):
    for kind in KINDS:
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
    g = Graph("step")
    prompt_graph(g, src)
    assert {o.kind for o in g.ops} == set(KINDS)
    check_coverage(g, PROFILE)


# (dynamic-T program, the values' rows, the program's t): T rows in a dynamic program read t_this_step; a static count
# in a dynamic program and every row count of a static one (T included: rows_of reports it as source 0 there too, and a
# static program's advance resets t_this_step to 1) compile T_SRC 2.
FORMS = [(True, T, 128), (True, 17, 128), (False, T, 17), (False, 5, 17)]


@pytest.mark.parametrize("dynamic, rows, t", FORMS, ids=["dynamic", "dynamic-static-rows", "static", "static-rows"])
def test_dispatches_bind_step_state(src, monkeypatch, dynamic, rows, t):
    """Every dispatch of both kinds is the builders' (source, params, slices) and the handler's (grid, threadgroup,
    macros) and binds (15, step_state, 0) once, with STEP_STATE 1, ROWS_MAX the compiled rows B and T_SRC 0 (dynamic-T,
    T rows) or T_SRC 2 with T_STATIC_ROWS B. Under a budget of one tile the products split into static slices, one
    dispatch each in basalt_prompt_gemm_slices' order, each with its own record (row0, col0) and the grid of its block;
    each conversion is one dispatch over the bound's elements, its output's dtype choosing CV_BF16."""
    monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", TILE_K2560)
    g = Graph("step")
    prompt_graph(g, src, rows)
    prog = emit(g, src, t, dynamic_t=dynamic, layout=LAYOUT)
    _, bound = basalt_windows(src, g.values.values(), t)
    b = t if rows is T else rows
    macros = row_macros(b, not dynamic or rows is not T)
    source = kernels.basalt_prompt_gemm_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
    # per dispatch: (function, its macro, the value slots, params, grid, label, written slots)
    want = [("basalt_prompt_convert", {"CV_BF16": cv}, [(0, "x", 0), (1, y, 0)],
             kernels.basalt_prompt_convert_params(2560), (-(-b * 2560 // 256), 1, 1), f"basalt_prompt_convert:{name}:x", [1])
            for cv, y, name in (("0", "x16", "to_f16"), ("1", "xbf16", "to_bf16"))]
    for name in ("mmvf.q", "mmvf.gate"):
        n = src.entries[f"dense/{name}"].shape[0]
        values = [(0, "xb", 0), (1, *bound[f"dense/{name}"]), (2, f"y.{name}", 0)]
        want += [("basalt_prompt_gemm", {"GEMM_BF16": "1"}, values, kernels.basalt_prompt_gemm_params(n, 2560, row0, col0),
                  (-(-ncols // 64), nrows // 64, 1), f"basalt_prompt_gemm:bf16:dense/{name}", [2])
                 for row0, nrows, col0, ncols in kernels.basalt_prompt_gemm_slices(b, n, 2560)]
    assert [w[5] for w in want].count("basalt_prompt_gemm:bf16:dense/mmvf.q") == (4 if b > 64 else 2)   # sliced
    assert len(prog.ops) == len(want)
    for o, (fn, macro, values, params, grid, label, writes) in zip(prog.ops, want):
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, {**macros, **macro}, source), label
        assert (o.name, o.grid, o.threadgroup, o.threadgroup_memory, o.meta["writes"]) == (
            label, grid, (256, 1, 1), [], writes)
        slot, prm, off = o.bindings[len(values)]
        assert o.bindings == values + [(slot, prm, off), (15, prog.step_state, 0)], label
        assert (slot, off, prog.buffers[prm].role, prog.buffers[prm].init) == (len(values), 0, "params", params), label


@pytest.mark.parametrize("dynamic, rows, t", FORMS, ids=["dynamic", "dynamic-static-rows", "static", "static-rows"])
def test_native_orders_its_stages(src, monkeypatch, dynamic, rows, t):
    """Gemm::native (gemm.cu:169-185): dequant_f16 of the whole matrix into the scratch, then Gemm::f16's product on
    it. Two native ops in a row, the smaller image first (Q6_K 35 x 2560, then Q4_K 70 x 2560), each product sliced
    under a budget of one tile: every dispatch is the builders' and the handler's and binds (15, step_state, 0); each
    op's dequantization (dequant's builders, DQ_OUT 1) comes first and writes the scratch at slot 1, and its slices
    (GEMM_BF16 0) read the scratch at slot 1; both ops bind the program's one ws.T{t}.prompt_gemm_f16.shared, sized to
    the larger image. The barrier pass (compiler/barriers.py's conflict rule) puts each op's products behind its
    dequantization (read after write) and the second dequantization behind the first op's products (write after read),
    so no product reads another op's image."""
    monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", TILE_K2560)
    names = ("gguf/pair.14.23.0", "gguf/mmvq.12")
    g = Graph("step")
    x = g.input("x16", (rows, 2560), F16)
    for name in names:
        w = weight(g, name, src)
        op(g, "basalt_prompt_gemm", [x, w], [g.value(f"y.{name}", (rows, w.shape[0]), F32)], entry="native")
    prog = emit(g, src, t, dynamic_t=dynamic, layout=LAYOUT)
    _, bound = basalt_windows(src, g.values.values(), t)
    b = t if rows is T else rows
    scratch = f"ws.T{t}.prompt_gemm_f16.shared"
    dq_source, gemm_source = (s.replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
                              for s in (kernels.basalt_dequant16_source(), kernels.basalt_prompt_gemm_source()))
    # per dispatch: (function, macros, source, the value slots, params, grid, threadgroup, label, written slots)
    want, images = [], []
    for name in names:
        tid, (n, k) = src.entries[name].type_id, src.entries[name].shape
        images.append(2 * n * k)
        grid, threads = kernels.basalt_dequant16_grid(n, k)
        dq = kernels.basalt_dequant16_macros(tid, "f16")      # it reads no rows; STEP_STATE is _Ctx.kernel's (dynamic-T)
        want.append(("basalt_dequant_g32", {**dq, "STEP_STATE": "1"} if dynamic else dq,
                     dq_source, [(0, *bound[name]), (1, scratch, 0)],
                     kernels.basalt_dequant16_params(tid, 0, n, k), (grid, 1, 1), (threads, 1, 1),
                     f"basalt_prompt_gemm:dequant_f16:{name}", [1]))
        want += [("basalt_prompt_gemm", {**row_macros(b, not dynamic or rows is not T), "GEMM_BF16": "0"},
                  gemm_source, [(0, "x16", 0), (1, scratch, 0), (2, f"y.{name}", 0)],
                  kernels.basalt_prompt_gemm_params(n, k, row0, col0), (-(-ncols // 64), nrows // 64, 1), (256, 1, 1),
                  f"basalt_prompt_gemm:native:{name}", [2])
                 for row0, nrows, col0, ncols in kernels.basalt_prompt_gemm_slices(b, n, k)]
    labels = [w[7] for w in want]
    assert labels.count(f"basalt_prompt_gemm:native:{names[1]}") == (4 if b > 64 else 2)   # sliced
    assert len(prog.ops) == len(want)
    for o, (fn, macros, source, values, params, grid, threads, label, writes) in zip(prog.ops, want):
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, macros, source), label
        assert (o.name, o.grid, o.threadgroup, o.threadgroup_memory, o.meta["writes"]) == (
            label, grid, threads, [], writes)
        slot, prm, off = o.bindings[len(values)]
        assert o.bindings == values + [(slot, prm, off), (15, prog.step_state, 0)], label
        assert (slot, off, prog.buffers[prm].role, prog.buffers[prm].init) == (len(values), 0, "params", params), label
    assert images[0] < images[1] and [n for n in prog.buffers if "prompt_gemm_f16" in n] == [scratch]
    assert (prog.buffers[scratch].nbytes, prog.buffers[scratch].role) == (images[1], "arena")
    second = labels.index(f"basalt_prompt_gemm:dequant_f16:{names[1]}")
    assert prog.ops[second].barrier_before, (
        "the second dequantization waits for the first op's products (write after read)")
    assert prog.ops[1].barrier_before and prog.ops[second + 1].barrier_before, (
        "each op's products wait for its dequantization (read after write)")


def test_rows_from_reads_its_value(src):
    """prompt_rows' ``rows_from``: an op whose only output is a placeholder takes its rows from the value it names (the
    same rule as an output's), and without it from that output, a static count."""
    g = Graph("step")
    k = g.input("k", (T, 2560), F32)
    placeholder = g.value("ph", (1,), DType.U32)
    o = op(g, "basalt_prompt_convert", [k], [placeholder])
    for dynamic, want in ((True, row_macros(64, False)), (False, row_macros(64, True))):
        ctx = _Ctx(Program(kernels={}, buffers={}, ops=[], layout=LAYOUT), [src], LAYOUT, 64, 1, 256, dynamic_t=dynamic)
        assert prompt_rows(ctx, o, rows_from=k) == (64, want)
        assert prompt_rows(ctx, o) == (1, row_macros(1, True))


def test_per_step_values_stay_out_of_params(src, monkeypatch):
    """Invariant 4: the program compiled at bound B serves every chunk, so its records hold geometry only. Two emissions
    give byte-identical records and macros; every product record is exactly (n, k, row0, col0), its slice's 64-row and
    64-column origin, every conversion record exactly (n,): no row count, position or pointer."""
    monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", TILE_K2560)
    progs = []
    for _ in range(2):
        g = Graph("step")
        prompt_graph(g, src)
        progs.append(emit(g, src, 128, dynamic_t=True, layout=LAYOUT))
    a, b = ([(record(p, o), p.kernels[o.kernel].macros) for o in p.ops] for p in progs)
    assert a == b
    origins = {(n, row0, col0) for n in (128, 1) for row0, _, col0, _ in kernels.basalt_prompt_gemm_slices(128, n, 2560)}
    seen = set()
    for o in progs[0].ops:
        rec = record(progs[0], o)
        if o.name.startswith("basalt_prompt_convert:"):
            assert struct.unpack("<I", rec) == (2560,)
            continue
        n, k, row0, col0 = struct.unpack("<4I", rec)
        assert k == 2560 and row0 % 64 == 0 and col0 % 64 == 0 and (n, row0, col0) in origins, o.name
        seen.add((n, row0, col0))
    assert seen == origins and len(origins) == 6


@pytest.mark.parametrize("budget", ["tile-", "c0", "c64", "cN", "constant"])
@pytest.mark.parametrize("rows", [1, 17, 256, 512])
def test_slices_cover_the_product(monkeypatch, rows, budget):
    """basalt_prompt_gemm_slices tiles [0, 64 * ceil(B / 64)) x [0, N) exactly once for every product of the pack, under
    a budget below one tile, budgets that make the column slice 0 (rows sliced instead), 64 and N columns wide, and the
    module's own constant: every row slice a multiple of 64 tall, every column slice but the one ending at N a multiple
    of 64 wide (the property the kernel's t < R, n < N stores rely on), each within the budget or a single tile."""
    bp = 64 * -(-rows // 64)
    for n, k in PRODUCTS:
        macs = {"tile-": 64 * 64 * k - 1, "c0": bp * 64 * k - 1, "c64": bp * 64 * k, "cN": bp * 64 * -(-n // 64) * k,
                "constant": kernels.PROMPT_GEMM_MACS}[budget]
        monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", macs)
        slices = kernels.basalt_prompt_gemm_slices(rows, n, k)
        cover = np.zeros((bp, n), np.uint8)
        for row0, nrows, col0, ncols in slices:
            assert row0 % 64 == 0 and nrows % 64 == 0 and nrows > 0 and row0 + nrows <= bp, (n, k, row0, nrows)
            assert col0 % 64 == 0 and ncols > 0 and col0 + ncols <= n, (n, k, col0, ncols)
            assert ncols % 64 == 0 or col0 + ncols == n, (n, k, col0, ncols)
            assert nrows * 64 * -(-ncols // 64) * k <= macs or (nrows == 64 and ncols <= 64), (n, k, row0, col0)
            cover[row0:row0 + nrows, col0:col0 + ncols] += 1
        assert (cover == 1).all(), (n, k)
        widths = {ncols for _, _, _, ncols in slices}
        if budget == "tile-":                   # single tiles, each over the budget
            assert len(slices) == bp // 64 * -(-n // 64) and widths <= set(range(1, 65)), (n, k)
        if budget == "c0" and bp > 64:          # rows sliced: Bp - 64 rows, then 64
            assert sorted({nrows for _, nrows, _, _ in slices}) == [64, bp - 64], (n, k)
        if budget == "cN":
            assert widths == {n} and len(slices) == 1, (n, k)


# ---- refusals: each case's valid input emits, and the same input with its one defect is refused by its own check ---

def gemm(g, src, entry="bf16", w="dense/mmvf.q", x_dtype=BF16, y_dtype=F32, y_extra=0, rows=T, x_rows=None):
    """``basalt_prompt_gemm`` of x [x_rows (default rows), K] by ``w`` into y [rows, N + y_extra]."""
    wv = weight(g, w, src)
    x = g.input("x", (rows if x_rows is None else x_rows, wv.shape[1]), x_dtype)
    op(g, "basalt_prompt_gemm", [x, wv], [g.value("y", (rows, wv.shape[0] + y_extra), y_dtype)], entry=entry)


def convert(g, x_dtype=F32, y_dtype=F16, y_extra=0, y_rows=T):
    """``basalt_prompt_convert`` of x [T, 64] into y [y_rows, 64 + y_extra]."""
    x = g.input("x", (T, 64), x_dtype)
    op(g, "basalt_prompt_convert", [x], [g.value("y", (y_rows, 64 + y_extra), y_dtype)])


def _gemm(**defect):
    def build(g, src, bad, tmp):
        gemm(g, src, **(defect if bad else {}))
        return {}
    return build


def _native(**defect):
    """A valid Gemm::native, F16 rows by gguf/mmvq.12 (Q4_K), and the same with ``defect``."""
    def build(g, src, bad, tmp):
        gemm(g, src, **{"entry": "native", "w": "gguf/mmvq.12", "x_dtype": F16, **(defect if bad else {})})
        return {}
    return build


def _convert(**defect):
    def build(g, src, bad, tmp):
        convert(g, **(defect if bad else {}))
        return {}
    return build


def _k16(g, src, bad, tmp):
    """A BF16 matrix of 4 rows over K 40 (valid: 48) from a pack entry at byte 16."""
    k = 40 if bad else 48
    w = g.const("p", (4, k), BF16)
    op(g, "basalt_prompt_gemm", [g.input("x", (T, k), BF16), w], [g.value("y", (T, 4), F32)], entry="bf16")
    return {"packs": [pack(tmp, {"p": (16, 4 * k * 2)})]}


def _f16_blocks(g, src, bad, tmp):
    """Gemm::native of a pack's 4 GGUF rows over K 64 at byte 16: Q8_0 (valid), or F16, a ggml type outside the ten
    (basalt's dequant_f16 exits on it: dequant_bf16.cu:203, :210-213)."""
    w = g.weight("p", (4, 64), "f16" if bad else "q8_0")
    op(g, "basalt_prompt_gemm", [g.input("x", (T, 64), F16), w], [g.value("y", (T, 4), F32)], entry="native")
    return {"packs": [pack(tmp, {"p": (16, 4 * ggml_row_bytes(8, 64))})]}


TEN = r"of ggml type \['iq4_nl', 'iq4_xs', 'q2_0', 'q3_k', 'q4_0', 'q4_k', 'q5_0', 'q5_k', 'q6_k', 'q8_0'\]"
REFUSALS = [
    ("entry outside bf16 and native", _gemm(entry="f16"), r"attr entry must be one of \('bf16', 'native'\), got 'f16'"),
    ("bf16 entry on GGUF blocks", _gemm(w="gguf/mmvq.12"), r"is not a weight or constant .* of ggml type \['bf16'\]"),
    ("native entry on a BF16 matrix", _gemm(entry="native"), r"'dense/mmvf.q' \(bf16 \[128, 2560\]\) is not .* " + TEN),
    ("native on F16 blocks", _f16_blocks, r"'p' \(f16 \[4, 64\]\) is not .* " + TEN),
    ("bf16 entry on F16 rows", _gemm(x_dtype=F16), r"'x' is f16 \[2, 2560\], not bf16 \[2, 2560\]"),
    ("native entry on BF16 rows", _native(x_dtype=BF16), r"'x' is bf16 \[2, 2560\], not f16 \[2, 2560\]"),
    ("product output not F32", _gemm(y_dtype=F16), r"'y' is f16 \[2, 128\], not f32 \[2, 128\]"),
    ("product output wider than the weight's rows", _gemm(y_extra=1), r"'y' is f32 \[2, 129\], not f32 \[2, 128\]"),
    ("K not a multiple of 16", _k16, "k must be a positive multiple of 16, got 40"),
    ("rows from n_inject", _gemm(rows=N_INJ), r"'y' has the rows n_inject \(T_SRC 1\)"),
    ("product rows from two sources", _gemm(rows=2, x_rows=T), r"'x' has the rows T and 'y' 2; a prompt kernel reads"),
    ("convert to F32", _convert(y_dtype=F32), r"'y' is f32; the output is F16 \(to_f16\) or BF16 \(to_bf16\)"),
    ("convert of F16 rows", _convert(x_dtype=F16), r"'x' is f16 \[2, 64\], not f32 \[2, 64\]"),
    ("convert output of another width", _convert(y_extra=1), r"'y' is f16 \[2, 65\], not f16 \[2, 64\]"),
    ("convert rows from two sources", _convert(y_rows=2), r"'x' has the rows T and 'y' 2; a prompt kernel reads"),
]


@pytest.mark.parametrize("build, why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(src, tmp_path, build, why):
    for bad in (False, True):
        g = Graph("step")
        kw = build(g, src, bad, tmp_path / f"pack{int(bad)}")
        packs = kw.pop("packs", [])
        if not bad:
            emit(g, src, 2, packs, **kw)
            continue
        with pytest.raises(ValueError, match=why):
            emit(g, src, 2, packs, **kw)
