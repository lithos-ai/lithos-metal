"""basalt's op family (monolith/ops/basalt.py) in compiled programs on the GPU: one static program per window width,
T = 1 .. 8 (basalt captures one graph per width), each running every kind over a synthetic .basalt file
(tests/contract/test_basalt_ops.py ``all_ops``): a native q8_1 per input width into every native_mmvq type and every
dense_tc type in every summation order, the pairs and the BF16 GEMVs on dense rows, the SwiGLU quantizer into a Q8_0
down, the IEEE quantizer on its own. The file is bound through its windows, and again through exact per-tensor copies
(``BasaltSource(copy=True)``), so that shader validation bounds every read by its own tensor, not by a window that
holds its neighbours.

- Each output bitwise equal to the merged single-kernel harness (tests/kernels/basalt_gpu.py) on the same bytes: the
  q8_1 rows against the harness's quantizer on the same activations, each GEMV against the harness's launch on the
  program's own q8_1 rows (or activations) and the file's weights. The program's outputs start as 0xff bytes, the
  harness's as 0xfe, so a value neither writes cannot match.
- Window invariance at program level (docs/design/flash-next.md invariant 1, ``CONTRIBUTING.md:14`` of basalt): column
  j of the T-wide program bitwise equal to the 1-wide program run on column j alone, every output finite.
- The eight programs share their weight windows: the same buffer names, and one buffer per name across the engines.
"""

import numpy as np
import pytest

from monolith.compiler import emit_program
from monolith.core import Graph
from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.contract.test_basalt_ops import BF16, DTC, GGUF, MMVQ, PAIRS, PROFILE, all_ops, write_file
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_mmvf import bf16_weights
from tests.kernels.test_basalt_mmvq import weights

F32 = np.float32
WIDTHS = range(1, 9)
PIPELINES = {}                                   # one pipeline cache for every engine of the module


def bits(a):
    return np.ascontiguousarray(a, F32).view(np.uint32)


@pytest.fixture(scope="module", params=[False, True], ids=["windows", "exact-copies"])
def runs(request, tmp_path_factory):
    """The file's tensors, the activations (8 columns per input), the eight programs and engines, every output of the
    T-wide program over columns 0 .. T - 1 (``wide[T]``) and of the 1-wide program over column j alone (``ones[j]``)."""
    rng = np.random.default_rng(2026)
    gguf = {n: (t, o, i, weights(t, i, o, rng).tobytes()) for n, (t, o, i) in GGUF.items()}
    bf16 = {n: (o, i, bf16_weights(rng, o, i).tobytes()) for n, (o, i) in BF16.items()}
    src = BasaltSource(write_file(tmp_path_factory.mktemp("ops") / "ops.basalt", gguf, bf16), copy=request.param)
    graphs = {t: Graph("step") for t in WIDTHS}
    for g in graphs.values():
        all_ops(g, src)
    progs = {t: emit_program(g, pack=src, profile=PROFILE, t=t, tail=None) for t, g in graphs.items()}
    g = graphs[1]
    inputs = {v.name: rng.standard_normal((8, v.shape[1])).astype(F32) for v in g.values.values() if v.is_input}
    outputs = {v.name: v.shape[1] * v.dtype.itemsize for o in g.ops for v in o.outputs}   # bytes per column
    dev = gpu.device()
    engines = {8: Engine(progs[8], dev, pipeline_cache=PIPELINES)}
    shared = {n: b for n, b in engines[8].buffers.items() if progs[8].buffers[n].role == "weights"}
    for t in WIDTHS[:-1]:
        engines[t] = Engine(progs[t], dev, buffers=shared, pipeline_cache=PIPELINES)

    def run(t, cols):
        eng = engines[t]
        for name, x in inputs.items():
            eng.buffers[name].write(x[cols].tobytes(), 0)
        for name in outputs:
            eng.buffers[name].fill(0xFF)
        assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
        return {name: eng.read(name, t * wb) for name, wb in outputs.items()}

    wide = {t: run(t, list(range(t))) for t in WIDTHS}
    ones = [run(1, [j]) for j in range(8)]
    return {"src": src, "gguf": gguf, "bf16": bf16, "progs": progs, "engines": engines, "inputs": inputs,
            "outputs": outputs, "wide": wide, "ones": ones}


def test_programs_share_their_weight_windows(runs):
    progs, engines, src = runs["progs"], runs["engines"], runs["src"]
    names = {t: sorted(n for n, b in p.buffers.items() if b.role == "weights") for t, p in progs.items()}
    assert names[8] and all(names[t] == names[8] for t in WIDTHS)
    for t in WIDTHS:
        assert all(engines[t].buffers[n] is engines[8].buffers[n] for n in names[t])
    assert all((progs[8].buffers[n].file is None) == src.copy for n in names[8])


def test_q8_1_rows_are_the_harness(runs):
    x = runs["inputs"]
    for t in WIDTHS:
        got = runs["wide"][t]
        for name in got:
            if name.startswith("xq") and name != "xq_ieee":
                want = gpu.q8_1(x["x" + name[2:]][:t].ravel(), "native")
            elif name == "hq":
                want = gpu.q8_1(x["gate"][:t].ravel(), "swiglu", x["up"][:t].ravel())
            elif name == "xq_ieee":
                want = gpu.q8_1(x["x2560"][:t].ravel(), "ieee")
            else:
                continue
            assert got[name] == want.tobytes(), (name, t)


def _input_rows(runs, t, n_in, name):
    """The q8_1 rows the program's GEMV ``name`` read at width t."""
    return np.frombuffer(runs["wide"][t]["hq" if name == "mmvq.down" else f"xq{n_in}"], np.uint8)


def _w(runs, name):
    return np.frombuffer(runs["gguf"][name][3], np.uint8)


@pytest.mark.parametrize("name", sorted(MMVQ))
def test_mmvq_is_the_harness(runs, name):
    tid, n_out, n_in = GGUF[name]
    for t in WIDTHS:
        y = np.frombuffer(runs["wide"][t][f"y.{name}"], F32).reshape(t, n_out)
        want = gpu.mmvq(tid, _w(runs, name), _input_rows(runs, t, n_in, name), n_in, n_out, t, fill=0xFE)
        assert np.array_equal(bits(y), bits(want)), t


@pytest.mark.parametrize("name", sorted(DTC) + ["order.q4_k"])
def test_dense_tc_is_the_harness(runs, name):
    tid, n_out, n_in = GGUF[name]
    for t in WIDTHS:
        y = np.frombuffer(runs["wide"][t][f"y.{name}"], F32).reshape(t, n_out)
        want = gpu.dense_tc(tid, _w(runs, name), _input_rows(runs, t, n_in, name), n_in, n_out, (t,), fill=0xFE)[t]
        assert np.array_equal(bits(y), bits(want[:t])), t


@pytest.mark.parametrize("pair", PAIRS, ids=[f"{a}+{b}" for a, b in PAIRS])
def test_dense_tc_pair_is_the_harness(runs, pair):
    a, b = pair
    (t0, n0, n_in), (t1, n1, _) = GGUF[f"pair.{a}.{b}.0"], GGUF[f"pair.{a}.{b}.1"]
    for t in WIDTHS:
        want = gpu.dense_tc_pair(t0, _w(runs, f"pair.{a}.{b}.0"), n0, t1, _w(runs, f"pair.{a}.{b}.1"), n1,
                                 _input_rows(runs, t, n_in, ""), n_in, (t,), fill=0xFE)[t]
        for m, n_out in enumerate((n0, n1)):
            y = np.frombuffer(runs["wide"][t][f"y.pair.{a}.{b}.{m}"], F32).reshape(t, n_out)
            assert np.array_equal(bits(y), bits(want[m][:t])), (t, m)


@pytest.mark.parametrize("name", sorted(BF16))
def test_mmvf_is_the_harness(runs, name):
    n_out, n_in, data = runs["bf16"][name]
    for t in WIDTHS:
        y = np.frombuffer(runs["wide"][t][f"y.{name}"], F32)
        want = gpu.mmvf(np.frombuffer(data, np.uint16), runs["inputs"]["x2560"][:t], n_in, n_out, t, fill=0xFE)
        assert np.array_equal(bits(y), bits(want)), t


def test_window_invariance(runs):
    """Column j of every output of the T-wide program, bit for bit the 1-wide program's on column j alone."""
    for t in WIDTHS:
        for name, wb in runs["outputs"].items():
            got = runs["wide"][t][name]
            if name.startswith("y."):
                assert np.all(np.isfinite(np.frombuffer(got, F32))), (name, t)
            for j in range(t):
                assert got[j * wb: (j + 1) * wb] == runs["ones"][j][name], (name, t, j)
