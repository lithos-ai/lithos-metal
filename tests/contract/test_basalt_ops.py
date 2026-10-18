"""basalt's op family (monolith/ops/basalt.py) as compiler/emit_basalt.py emits it over a synthetic .basalt file bound
through its windows (compiler/emit.py ``basalt_windows``): each op's dispatch (function, macros, grid, threadgroup,
params and written slots) equal to monolith/kernels.py's builders, and each refusal, a valid input with exactly one
defect refused by that defect's own check (its message). The GPU side is tests/kernels/test_basalt_ops_program.py."""

import json

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler import emit_program
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import basalt_windows
from monolith.core import DType, Graph
from monolith.core.ir import BlockDomain, OpClass
from monolith.core.shapes import T
from monolith.core.step_state import StepStateLayout
from monolith.formats.gguf import GGML_TYPES, ggml_row_bytes
from monolith.ops import OPS
from monolith.packs import PackFile
from monolith.packs.basalt import BasaltSource
from monolith.packs.packer import ALIGN
from tests.contract.test_basalt_reader import write_basalt
from tests.contract.test_basalt_windows import PROFILE

F32, U8 = DType.F32, DType.U8
KINDS = ("basalt_q8_1", "basalt_mmvq", "basalt_mmvf", "basalt_dense_tc", "basalt_dense_tc_pair")
Q8_1_FUNCTIONS = {"native": "native_quantize_q8_1", "swiglu": "native_swiglu_quantize_q8_1", "ieee": "quantize_q8_1"}
QK = kernels.Q8_1_BLOCK_BYTES

# The tensors (name: (ggml type id, n_out, n_in)): every native_mmvq type at basalt's 2560 inputs, Q8_0 at 640 (the
# shared expert's down, fed by the SwiGLU quantizer), every dense_tc type in every summation order at basalt's defaults
# (170 multiprocessors, an aligned weight; "tile" at 2721 rows, 171 tiles, where a row is whole 16-byte units and 170
# tiles would take the one order), the pair's types, and Q4_K at 2048 x 2720, 170 tiles, which 170 multiprocessors run
# in the one order and any fewer (PROFILE's 20, this M3 Max's 40) in the tile order. BF16 rows (name: (n_out, n_in))
# live in the dense section, as basalt's do.
DTC_IN = {"one": 2048, "tile": 512, "s2": 22528, "s1": 23040}
MMVQ = {f"mmvq.{t}": (t, 70, 2560) for t in kernels.MMVQ_TRAITS}
MMVQ["mmvq.down"] = (8, 70, 640)
DTC = {f"dtc.{t}.{o}": (t, 2721 if o == "tile" and DTC_IN[o] // 256 * sb % 16 == 0 else 16, DTC_IN[o])
       for t, sb in kernels.DENSE_TC_SB.items() for o in DTC_IN}
PAIRS = [(14, 23), (12, 11), (42, 13)]                  # (w0, w1) types at 2560 inputs, 35 and 18 rows
PAIR = {f"pair.{a}.{b}.{m}": (ty, (35, 18)[m], 2560) for a, b in PAIRS for m, ty in enumerate((a, b))}
GGUF = {**MMVQ, **DTC, **PAIR, "order.q4_k": (12, 2720, 2048)}
BF16 = {"mmvf.q": (128, 2560), "mmvf.gate": (1, 2560)}


def gguf_bytes(type_id, n_out, n_in):
    return n_out * ggml_row_bytes(type_id, n_in)


def write_file(path, gguf, bf16):
    """A .basalt file with ``gguf`` ({name: (type id, n_out, n_in, bytes)}) in its gguf section, each tensor at the next
    multiple of 4096 (basalt's kTensorAlign), and the BF16 matrices ``bf16`` ({name: (n_out, n_in, bytes)}) as dense
    rows of kind 4 at multiples of 64 (its arena's alignment), so that tensors bind at offsets inside their windows."""
    blob, tensors = bytearray(), {}
    for name, (t, n_out, n_in, data) in gguf.items():
        blob += bytes(-len(blob) % 4096)
        tensors[name] = {"type_id": t, "shape": [n_in, n_out], "offset": len(blob), "bytes": len(data)}
        blob += data
    dense, rows = bytearray(), []
    for name, (n_out, n_in, data) in bf16.items():
        dense += bytes(-len(dense) % 64)
        rows.append(f"{name} 0 4 {len(dense)} {len(data)} {len(dense)} {len(data)} {n_in} {n_out} 0 0 1 0 0 0 0 0 0 0")
        dense += data
    index = f"# dense index v3\n# align 64 pool {len(dense)} tensors {len(rows)}\n" + "\n".join(rows) + "\n"
    meta = {"format": "basalt", "version": 1, "tensors": tensors}
    path.write_bytes(write_basalt({"meta": json.dumps(meta).encode(), "gguf": bytes(blob), "dense": bytes(dense),
                                   "dense.index": index.encode()}))
    return path


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """The tensors above as random bytes (the planner and the emitter never read them)."""
    rng = np.random.default_rng(3)
    noise = lambda n: rng.integers(0, 256, n, dtype=np.uint8).tobytes()     # noqa: E731
    gguf = {n: (t, o, i, noise(gguf_bytes(t, o, i))) for n, (t, o, i) in GGUF.items()}
    bf16 = {n: (o, i, noise(2 * o * i)) for n, (o, i) in BF16.items()}
    return BasaltSource(write_file(tmp_path_factory.mktemp("ops") / "ops.basalt", gguf, bf16))


# ---- graph builders (the GPU test uses them too) -------------------------------------------------------------------

def weight(g, name, src):
    """The value that declares a gguf tensor or a BF16 dense row exactly (``gguf/`` a weight, ``dense/`` a constant)."""
    e = src.entries[name]
    if name.startswith("dense/"):
        return g.const(name, e.shape, DType.BF16)
    return g.weight(name, e.shape, GGML_TYPES[e.type_id][0])


def op(g, kind, ins, outs, **attrs):
    return g.op(kind, ins, outs, domain=BlockDomain("rows", 1), klass=OpClass.MAP, **attrs)


def q8_1(g, name, xs, kind="native", rows=T, width=None):
    """``basalt_q8_1`` of ``xs`` (``[x]``, or ``[gate, up]`` for swiglu) into q8_1 rows named ``name``."""
    n_in = xs[0].shape[1]
    y = g.value(name, (rows, n_in // 32 * QK if width is None else width), U8)
    op(g, "basalt_q8_1", xs, [y], quantizer=kind)
    return y


def gemv(g, kind, name, x, *ws, rows=T, widths=None):
    """``kind`` (basalt_mmvq, basalt_dense_tc, basalt_dense_tc_pair, basalt_mmvf) of ``x`` by ``ws`` into F32 outputs
    named ``name`` (the pair: ``name.0``, ``name.1``), each as wide as its weight's rows unless ``widths`` says."""
    widths = widths or [w.shape[0] for w in ws]
    names = [name] if len(ws) == 1 else [f"{name}.{m}" for m in range(len(ws))]
    ys = [g.value(n, (rows, wd), F32) for n, wd in zip(names, widths)]
    op(g, kind, [x, *ws], ys)
    return ys


def emit(g, src, t=1, packs=(), **kw):
    return emit_program(g, pack=[src, *packs], profile=PROFILE, t=t, tail=None, **kw)


def specs(prog, kind):
    return [o for o in prog.ops if o.name.startswith(kind + ":")]


def spliced(source):
    """A builder's source as the emitter registers it: the StepState struct after the prelude (``_Ctx.kernel``)."""
    return source.replace(kernels.PRELUDE, kernels.PRELUDE + StepStateLayout().to_msl() + "\n", 1)


def all_ops(g, src):
    """Every kind over every tensor: one native q8_1 per input width, the SwiGLU into the Q8_0 down, the IEEE rows on
    their own."""
    widths = sorted({e[2] for e in GGUF.values()})
    xs = {n: g.input(f"x{n}", (T, n), F32) for n in widths}
    xq = {n: q8_1(g, f"xq{n}", [xs[n]]) for n in widths}
    gate, up = g.input("gate", (T, 640), F32), g.input("up", (T, 640), F32)
    hq = q8_1(g, "hq", [gate, up], "swiglu")
    q8_1(g, "xq_ieee", [xs[2560]], "ieee")
    for name, (_, _, n_in) in {**MMVQ, **DTC, "order.q4_k": GGUF["order.q4_k"]}.items():
        kind = "basalt_mmvq" if name.startswith("mmvq.") else "basalt_dense_tc"
        gemv(g, kind, f"y.{name}", hq if name == "mmvq.down" else xq[n_in], weight(g, f"gguf/{name}", src))
    for a, b in PAIRS:
        gemv(g, "basalt_dense_tc_pair", f"y.pair.{a}.{b}", xq[2560],
             *[weight(g, f"gguf/pair.{a}.{b}.{m}", src) for m in (0, 1)])
    for name in BF16:
        gemv(g, "basalt_mmvf", f"y.{name}", xs[2560], weight(g, f"dense/{name}", src))


# ---- registration, and each dispatch against the builders ----------------------------------------------------------

def test_registered_and_covered(src):
    for kind in KINDS:
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
    g = Graph("step")
    all_ops(g, src)
    assert {o.kind for o in g.ops} == set(KINDS)
    check_coverage(g, PROFILE)


def test_grid_builders_are_the_launchers():
    """The two grid builders this PR adds, against the harness's own formulas (tests/kernels/basalt_gpu.py)."""
    for n in (32, 256, 2560, 8 * 23040):
        assert kernels.basalt_q8_1_grid(n) == (-(-n // 256), 256)
    for n_in, n_out, t in ((2560, 128, 1), (2560, 1, 5), (640, 7, 8)):
        m = kernels.basalt_mmvf_macros(n_in, n_out, t)
        assert kernels.basalt_mmvf_grid(m, n_out) == (n_out, kernels.basalt_mmvf_block(n_in))


def expected(gop, src, t):
    """(function, macros, source, params, grid, threadgroup) of ``gop``'s dispatch from the builders."""
    if gop.kind == "basalt_q8_1":
        n = t * gop.inputs[0].shape[1]
        return (Q8_1_FUNCTIONS[gop.attrs["quantizer"]], {}, kernels.basalt_q8_1_source(), kernels.basalt_q8_1_params(n),
                -(-n // 256), 256)
    ws = [src.entries[w.name] for w in gop.inputs[1:]]
    tid, (n_out, n_in) = ws[0].type_id, ws[0].shape
    if gop.kind == "basalt_mmvq":
        m = kernels.basalt_mmvq_macros(tid, n_in, t)
        return ("basalt_mmvq", m, kernels.basalt_mmvq_source(), kernels.basalt_mmvq_params(n_in, n_out),
                *kernels.basalt_mmvq_grid(m, n_out))
    if gop.kind == "basalt_dense_tc":
        m = kernels.basalt_dense_tc_macros(tid, n_in, n_out)
        return ("basalt_dense_tc", m, kernels.basalt_dense_tc_source(), kernels.basalt_dense_tc_params(n_in, n_out, t),
                *kernels.basalt_dense_tc_grid(m, n_out))
    if gop.kind == "basalt_dense_tc_pair":
        t1, n_out1 = ws[1].type_id, ws[1].shape[0]
        m = kernels.basalt_dense_tc_pair_macros(tid, t1, n_in)
        return ("basalt_dense_tc_pair", m, kernels.basalt_dense_tc_source(),
                kernels.basalt_dense_tc_params(n_in, n_out, t, n_out1),
                *kernels.basalt_dense_tc_pair_grid(n_out, n_out1))
    m = kernels.basalt_mmvf_macros(n_in, n_out, t)
    return ("basalt_mmvf", m, kernels.basalt_mmvf_source(), kernels.basalt_mmvf_params(n_in, n_out, t),
            n_out, kernels.basalt_mmvf_block(n_in))


def slot_values(gop):
    """The values in the kernel's buffer slots 0, 1, ...: weights first for the q8_1 GEMVs, as the kernels take them."""
    ins, outs = list(gop.inputs), list(gop.outputs)
    if gop.kind in ("basalt_mmvq", "basalt_dense_tc", "basalt_dense_tc_pair"):
        ins = ins[1:] + ins[:1]
    return ins + outs


@pytest.mark.parametrize("t", range(1, 9))
def test_dispatches_are_the_builders(src, t):
    """Every op of ``all_ops`` at T = t: one dispatch whose kernel, macros, grid, threadgroup and params record are the
    builders' (the q8_1 and BF16 grids as the harness writes them), binding each weight at the window and offset the
    planner gives it and each activation at 0 of its own buffer, writing exactly its outputs' slots, the params at
    the slot after them; no StepState and no threadgroup memory."""
    g = Graph("step")
    all_ops(g, src)
    prog = emit(g, src, t)
    _, bound = basalt_windows(src, g.values.values(), t)
    assert len(prog.ops) == len(g.ops)
    writer = {name: o for o in prog.ops for i, name, _ in o.bindings if i in o.meta["writes"]}
    for gop in g.ops:
        o = writer[gop.outputs[0].name]
        fn, macros, source, params, grid, tg = expected(gop, src, t)
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, macros, spliced(source)), gop
        assert (o.grid, o.threadgroup, o.threadgroup_memory) == ((grid, 1, 1), (tg, 1, 1), []), gop
        vals = slot_values(gop)
        assert o.bindings[:-1] == [(i, *bound.get(v.name, (v.name, 0))) for i, v in enumerate(vals)], gop
        slot, prm, off = o.bindings[-1]
        assert (slot, off, prog.buffers[prm].role, prog.buffers[prm].init) == (len(vals), 0, "params", params), gop
        assert o.meta["writes"] == list(range(len(gop.inputs), len(vals))), gop


def test_dense_tc_runs_basalts_orders(src):
    """Each dtc.<type>.<order> tensor runs in the order it names (basalt's 170 multiprocessors, aligned weights)."""
    g = Graph("step")
    all_ops(g, src)
    prog = emit(g, src, 8)
    seen = set()
    for o in [o for o in specs(prog, "basalt_dense_tc") if "/dtc." in o.name]:
        _, ty, order = o.name.split("/")[1].split(".")
        assert prog.kernels[o.kernel].macros["DTC_ORDER"] == str(kernels.DENSE_TC_ORDERS[order]), o.name
        seen.add((int(ty), order))
    assert seen == {(t, o) for t in kernels.DENSE_TC_SB for o in DTC_IN}


def test_q8_1_rows_reach_their_readers_behind_a_barrier(src):
    """The barrier pass orders each GEMV behind the quantizer whose rows it reads (``writes`` declares them)."""
    g = Graph("step")
    x = g.input("x", (T, 2560), F32)
    xq = q8_1(g, "xq", [x])
    gemv(g, "basalt_mmvq", "y0", xq, weight(g, "gguf/mmvq.8", src))
    gemv(g, "basalt_dense_tc", "y1", xq, weight(g, "gguf/pair.14.23.0", src))
    gemv(g, "basalt_mmvf", "y2", x, weight(g, "dense/mmvf.q", src))
    prog = emit(g, src, 2)
    assert [o.barrier_before for o in prog.ops] == [True, True, False, False]


# ---- refusals: each case's valid input emits, and the same input with its one defect is refused by its own check ---

def pack(directory, entries, slab=False):
    """A pack of zeros whose aux entries ``{name: (offset, nbytes)}`` sit at those offsets: the only route to a
    misaligned binding (a .basalt entry binds at a multiple of 16, ``basalt_windows``; activations at whole rows).
    ``slab``: the entries are block-lane-major bf16 slabs instead, one row-scale word each after the last."""
    directory.mkdir()
    end = max(o + n for o, n in entries.values())
    size = -(-(end + 4 * len(entries)) // ALIGN) * ALIGN
    (directory / "weights.pack").write_bytes(bytes(size))
    rows = [{"name": n, "offset": o, "nbytes": nb, "dtype": "U8", "shape": [nb]} for n, (o, nb) in entries.items()]
    slabs = [dict(r, format="bf16", n=1, row_scales_offset=end + 4 * i) for i, r in enumerate(rows)] if slab else []
    (directory / "manifest.json").write_text(json.dumps({
        "version": 2, "alignment": ALIGN, "pack": "weights.pack", "nbytes": size, "slabs": slabs,
        "aux": [] if slab else rows}))
    return PackFile(directory)


def one_op(kind, g, src):
    """A valid op of ``kind`` at T columns (with the quantizer feeding a q8_1 GEMV)."""
    x = g.input("x", (T, 2560), F32)
    if kind == "basalt_q8_1":
        return q8_1(g, "xq", [x])
    if kind == "basalt_mmvf":
        return gemv(g, kind, "y", x, weight(g, "dense/mmvf.q", src))
    ws = [weight(g, f"gguf/pair.14.23.{m}", src) for m in (0, 1)]
    return gemv(g, kind, "y", q8_1(g, "xq", [x]), *(ws if kind == "basalt_dense_tc_pair" else ws[:1]))


def _dynamic(kind):
    def build(g, src, bad, tmp):
        one_op(kind, g, src)
        return {"dynamic_t": bad}
    return build


def _q8_1_columns(quantizer):
    def build(g, src, bad, tmp):     # 16 columns: a static count, as the default t_max (8) allows no T past 8
        rows = 16 if bad else 8
        xs = [g.input(n, (rows, 64), F32) for n in (("gate", "up") if quantizer == "swiglu" else ("x",))]
        q8_1(g, "xq", xs, quantizer, rows=rows)
        return {}
    return build


def _q8_1_block(g, src, bad, tmp):
    q8_1(g, "xq", [g.input("x", (T, 48 if bad else 64), F32)], width=72)
    return {}


def _q8_1_width(g, src, bad, tmp):
    q8_1(g, "xq", [g.input("x", (T, 64), F32)], width=76 if bad else 72)
    return {}


def _q8_1_rows(g, src, bad, tmp):
    q8_1(g, "xq", [g.input("x", (T, 64), F32)], rows=1 if bad else T)
    return {}


def _q8_1_quantizer(g, src, bad, tmp):
    q8_1(g, "xq", [g.input("x", (T, 64), F32)], "fast" if bad else "native")
    return {}


def _ieee_reader(g, src, bad, tmp):
    x = g.input("x", (T, 2560), F32)
    ieee, native = q8_1(g, "xq_ieee", [x], "ieee"), q8_1(g, "xq", [x])
    gemv(g, "basalt_mmvq", "y", ieee if bad else native, weight(g, "gguf/mmvq.8", src))
    return {}


def _ieee_view(written):
    """A native GEMV over rows a quantizer writes through a view (``written``) or the GEMV reads through one; the
    rows are native (valid) or IEEE, which then reach the GEMV past the readers check."""
    def build(g, src, bad, tmp):
        rows = g.value("rows", (2, 2880), U8)
        view = g.view("rows_v", rows, 0, 2)
        op(g, "basalt_q8_1", [g.input("x", (2, 2560), F32)], [view if written else rows],
           quantizer="ieee" if bad else "native")
        gemv(g, "basalt_mmvq", "y", rows if written else view, weight(g, "gguf/mmvq.8", src), rows=2)
        return {}
    return build


def _weight_type(kind, good, other):
    def build(g, src, bad, tmp):
        x = g.input("x", (T, 2560), F32)
        gemv(g, kind, "y", x if kind == "basalt_mmvf" else q8_1(g, "xq", [x]), weight(g, other if bad else good, src))
        return {}
    return build


def _xq_width(g, src, bad, tmp):
    xq = q8_1(g, "xq", [g.input("x", (T, 640 if bad else 2560), F32)])
    gemv(g, "basalt_mmvq", "y", xq, weight(g, "gguf/mmvq.8", src))
    return {}


def _out_width(kind, extra):
    """``kind`` whose (last) output is ``extra`` values wider than its weight's rows."""
    def build(g, src, bad, tmp):
        x = g.input("x", (T, 2560), F32)
        names = {"basalt_mmvf": ["dense/mmvf.q"], "basalt_dense_tc_pair": ["gguf/pair.14.23.0", "gguf/pair.14.23.1"]}
        ws = [weight(g, n, src) for n in names.get(kind, ["gguf/pair.14.23.0"])]
        widths = [w.shape[0] + (extra if bad and m == len(ws) - 1 else 0) for m, w in enumerate(ws)]
        gemv(g, kind, "y", x if kind == "basalt_mmvf" else q8_1(g, "xq", [x]), *ws, widths=widths)
        return {}
    return build


def _pair_inputs(g, src, bad, tmp):
    xq = q8_1(g, "xq", [g.input("x", (T, 2560), F32)])
    gemv(g, "basalt_dense_tc_pair", "y", xq, weight(g, "gguf/pair.14.23.0", src),
         weight(g, "gguf/dtc.12.one" if bad else "gguf/pair.14.23.1", src))
    return {}


def _mmvf_x_width(g, src, bad, tmp):
    gemv(g, "basalt_mmvf", "y", g.input("x", (T, 2562 if bad else 2560), F32), weight(g, "dense/mmvf.q", src))
    return {}


def _blm_slab(g, src, bad, tmp):
    """The same bytes as a pack's aux entry (rows as stored) or as its bf16 slab (re-laid-out lanes)."""
    w = g.weight("p", (128, 2560), "bf16")
    gemv(g, "basalt_mmvf", "y", g.input("x", (T, 2560), F32), w)
    return {"packs": [pack(tmp, {"p": (0, 128 * 2560 * 2)}, slab=bad)]}


def _aligned(kind, slot, offset):
    """``kind`` with the value of kernel slot ``slot`` read from a pack entry at byte 16 (valid) or ``offset``."""
    def build(g, src, bad, tmp):
        at = offset if bad else 16
        x = g.input("x", (T, 2560), F32)
        if kind == "basalt_q8_1":
            q8_1(g, "xq", [g.const("p", (2, 64), F32)])
            return {"packs": [pack(tmp, {"p": (at, 2 * 64 * 4)})]}
        if kind == "basalt_mmvf":
            v = g.const("p", *(((2, 2560), F32) if slot == 0 else ((128, 2560), DType.BF16)))
            gemv(g, kind, "y", v if slot == 0 else x, v if slot == 1 else weight(g, "dense/mmvf.q", src))
            return {"packs": [pack(tmp, {"p": (at, int(np.prod(v.shape)) * v.dtype.itemsize)})]}
        ws = [weight(g, f"gguf/pair.14.23.{m}", src) for m in (0, 1)][: 2 if kind == "basalt_dense_tc_pair" else 1]
        if slot == len(ws):                                                     # the q8_1 rows
            gemv(g, kind, "y", g.const("p", (2, 2880), U8), *ws)
            return {"packs": [pack(tmp, {"p": (at, 2 * 2880)})]}
        e = src.entries[ws[slot].name]
        ws[slot] = g.weight("p", e.shape, GGML_TYPES[e.type_id][0])
        gemv(g, kind, "y", q8_1(g, "xq", [x]), *ws)
        return {"packs": [pack(tmp, {"p": (at, e.nbytes)})]}
    return build


def _entry_rows(kind, rows):
    """``kind`` whose weight (the pair's second) is a pack entry at byte 16 holding exactly its rows (valid), or
    ``rows`` rows more (-1: a row short, which the kernel would read past). ``emit_program``'s constant check refuses
    a short BF16 constant before the handler runs, so the BF16 GEMV's case is a row long."""
    def build(g, src, bad, tmp):
        x, more = g.input("x", (T, 2560), F32), rows if bad else 0
        if kind == "basalt_mmvf":
            gemv(g, kind, "y", x, g.const("p", (128, 2560), DType.BF16))
            return {"packs": [pack(tmp, {"p": (16, (128 + more) * 2560 * 2)})]}
        ws = [weight(g, f"gguf/pair.14.23.{m}", src) for m in (0, 1)][: 2 if kind == "basalt_dense_tc_pair" else 1]
        e = src.entries[ws[-1].name]
        ws[-1] = g.weight("p", e.shape, GGML_TYPES[e.type_id][0])
        gemv(g, kind, "y", q8_1(g, "xq", [x]), *ws)
        return {"packs": [pack(tmp, {"p": (16, e.nbytes + more * e.nbytes // e.shape[0])})]}
    return build


REFUSALS = [
    *[(f"dynamic-T {k}", _dynamic(k), "a dynamic-T program cannot run them") for k in KINDS],
    ("native q8_1 of 16 columns", _q8_1_columns("native"), "16 columns; basalt quantizes 1 to 8"),
    ("swiglu q8_1 of 16 columns", _q8_1_columns("swiglu"), "16 columns; basalt quantizes 1 to 8"),
    ("q8_1 column not whole blocks", _q8_1_block, "a column of 48 values is not whole 32-value blocks"),
    ("q8_1 rows of another width", _q8_1_width, r"is u8 \[2, 76\], not u8 \[2, 72\] \(q8_1 rows"),
    ("q8_1 rows for other columns", _q8_1_rows, r"'x' is f32 \[2, 64\], not f32 \[1, 64\] \(the quantized columns\)"),
    ("q8_1 quantizer unknown", _q8_1_quantizer, "attr quantizer must be one of"),
    ("IEEE q8_1 rows read by a GEMV", _ieee_reader, r"is read by \['basalt_mmvq'\]; basalt's IEEE rows feed only"),
    ("IEEE q8_1 rows written to a view", _ieee_view(True), r"'rows_v' shares its buffer with \['rows'\]"),
    ("IEEE q8_1 rows read through a view", _ieee_view(False), r"'rows' shares its buffer with \['rows_v'\]"),
    ("mmvq on a BF16 matrix", _weight_type("basalt_mmvq", "gguf/mmvq.8", "dense/mmvf.q"), "of ggml type"),
    ("dense_tc on a Q8_0 matrix", _weight_type("basalt_dense_tc", "gguf/mmvq.12", "gguf/mmvq.8"), "of ggml type"),
    ("mmvf on a Q8_0 matrix", _weight_type("basalt_mmvf", "dense/mmvf.q", "gguf/mmvq.8"), r"of ggml type \['bf16'\]"),
    ("mmvq rows of another input", _xq_width, r"is u8 \[2, 720\], not u8 \[2, 2880\] \(q8_1 rows of the weight's"),
    *[(f"{k} output wider than its rows", _out_width(k, 2), "contiguous columns of the weight's rows")
      for k in ("basalt_mmvq", "basalt_dense_tc", "basalt_dense_tc_pair", "basalt_mmvf")],
    ("mmvf ldy < n_out", _out_width("basalt_mmvf", -1), r"holds 127 values a row .* so ldy >= n_out"),
    ("mmvf on a BLM bf16 slab", _blm_slab, "is a block-lane-major pack slab, not rows as stored"),
    ("mmvf input wider than its weight", _mmvf_x_width, "contiguous columns of the weight's width"),
    ("pair over two input widths", _pair_inputs, "the pair reads one input"),
    ("q8_1 input off 4 bytes", _aligned("basalt_q8_1", 0, 2), "slot 0 needs 4-byte alignment"),
    ("mmvq weight off 4 bytes", _aligned("basalt_mmvq", 0, 2), "slot 0 needs 4-byte alignment"),
    ("mmvq rows off 4 bytes", _aligned("basalt_mmvq", 1, 2), "slot 1 needs 4-byte alignment"),
    ("dense_tc weight off 4 bytes", _aligned("basalt_dense_tc", 0, 2), "slot 0 needs 4-byte alignment"),
    ("dense_tc rows off 4 bytes", _aligned("basalt_dense_tc", 1, 2), "slot 1 needs 4-byte alignment"),
    ("pair w0 off 16 bytes", _aligned("basalt_dense_tc_pair", 0, 8), "slot 0 needs 16-byte alignment"),
    ("pair w1 off 16 bytes", _aligned("basalt_dense_tc_pair", 1, 8), "slot 1 needs 16-byte alignment"),
    ("pair rows off 4 bytes", _aligned("basalt_dense_tc_pair", 2, 2), "slot 2 needs 4-byte alignment"),
    ("mmvf input off 8 bytes", _aligned("basalt_mmvf", 0, 4), "slot 0 needs 8-byte alignment"),
    ("mmvf weight off 4 bytes", _aligned("basalt_mmvf", 1, 2), "slot 1 needs 4-byte alignment"),
    *[(f"{k} weight entry a row short", _entry_rows(k, -1), r"bytes of \w+ \[\d+, 2560\], but its pack entry holds")
      for k in ("basalt_mmvq", "basalt_dense_tc", "basalt_dense_tc_pair")],
    ("mmvf weight entry a row long", _entry_rows("basalt_mmvf", 1),
     r"'p' is 655360 bytes of bf16 \[128, 2560\], but its pack entry holds 660480"),
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


def test_ieee_q8_1_takes_any_column_count(src):
    """basalt's quantize_q8_1_rows checks only that it has values (iq_kernels.cu:2450-2453): 16 columns emit."""
    g = Graph("step")
    q8_1(g, "xq", [g.input("x", (16, 64), F32)], "ieee", rows=16)
    (o,) = emit(g, src, 1).ops
    assert o.grid == (4, 1, 1)


def test_dense_tc_order_is_basalts_not_this_gpus(src, tmp_path):
    """Q4_K at 2048 x 2720 (170 tiles): basalt's 170 multiprocessors run the one order, any fewer cores would give the
    tile order (PROFILE's 20, this M3 Max's 40, every Apple GPU's), and a weight off a 16-byte boundary the KS 1 order.
    The emitted order stays basalt's for the file's weight and for the same shape bound at byte 8 of a pack (4-byte
    aligned, which dense_tc takes)."""
    rule = kernels.basalt_dense_tc_variant
    assert (rule(12, 2048, 2720), rule(12, 2048, 2720, aligned=False)) == ("one", "s1")
    assert {rule(12, 2048, 2720, sm_count=n) for n in range(1, 170)} == {"tile"}
    for at in (None, 8):
        g = Graph("step")
        xq = q8_1(g, "xq", [g.input("x", (T, 2048), F32)])
        w = weight(g, "gguf/order.q4_k", src) if at is None else g.weight("p", (2720, 2048), "q4_k")
        gemv(g, "basalt_dense_tc", "y", xq, w)
        prog = emit(g, src, 4, [] if at is None else [pack(tmp_path / "p", {"p": (at, 2720 * 1152)})])
        (o,) = specs(prog, "basalt_dense_tc")
        assert o.bindings[0][2] % 16 == (0 if at is None else 8)
        assert prog.kernels[o.kernel].macros == {"DTC_TYPE": "12", "DTC_ORDER": "0"}
