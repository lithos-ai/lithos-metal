import pytest

from monolith.compiler import CoverageError, check_coverage
from monolith.core import CTX, T, BlockDomain, DType, Graph, OpClass, Sym, bind, is_static, numel
from monolith.core.profile import Profile
from monolith.ops import OPS, KernelBinding, OpDef, register_op


def _profile(family: str) -> Profile:
    return Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0,
                                   "engine": {"family": family, "lane_order": "interleaved16"}})


def _graph():
    g = Graph()
    h = g.input("h", (T, 5120), DType.BF16)
    w = g.weight("w_gate_up", (17408 * 2, 5120), "nvfp4")
    act = g.value("act", (T, 17408), DType.BF16)
    g.op("test_gemv_gateup", [h, w], [act], domain=BlockDomain("rows", 2176), klass=OpClass.MAP, T=T)
    w2 = g.weight("w_down", (5120, 17408), "nvfp4")
    out = g.value("h_out", (T, 5120), DType.BF16)
    g.op("test_gemv_down", [act, w2], [out], domain=BlockDomain("rows", 320), klass=OpClass.MAP)
    return g, out


def test_graph_structure_and_check():
    g, out = _graph()
    g.check()
    assert len(g) == 2 and out.producer is g.ops[1] and g.producers(g.ops[1]) == [g.ops[0]]
    assert T in g.symbols and CTX not in g.symbols
    with pytest.raises(ValueError):
        g.value("h", (1,), DType.F32)                      # duplicate name
    dangling = g.value("dangling", (T, 8), DType.F32)
    y = g.value("y", (T, 8), DType.F32)
    g.op("test_use", [dangling], [y], domain=BlockDomain("span", 1), klass=OpClass.MAP)
    with pytest.raises(ValueError):
        g.check()                                         # reads a value nothing produces


def test_shapes():
    assert not is_static((T, 5120)) and is_static((4, 4))
    assert bind((T, 5120), {T: 3}) == (3, 5120) and numel((T, 5120), {T: 2}) == 10240
    with pytest.raises(KeyError):
        bind((Sym("Z"),), {})
    with pytest.raises(ValueError):
        BlockDomain("rows", 0)


def test_coverage_guard_lists_missing_ops():
    g, _ = _graph()
    register_op(OpDef("test_gemv_gateup", OpClass.MAP, "rows").bind("apple10", KernelBinding("gemv_T")))
    try:
        with pytest.raises(CoverageError) as ei:
            check_coverage(g, _profile("Apple10"))      # test_gemv_down is not registered at all
        msg = str(ei.value)
        assert "test_gemv_down" in msg and "test_gemv_gateup" not in msg and "not registered" in msg
        register_op(OpDef("test_gemv_down", OpClass.MAP, "rows").bind("apple10", KernelBinding("gemv_T")))
        check_coverage(g, _profile("Apple10"))           # now fully covered on Apple10
        with pytest.raises(CoverageError) as ei:
            check_coverage(g, _profile("Apple9"))        # …but not on Apple9
        assert "bound only for ['apple10']" in str(ei.value)
    finally:
        OPS.unregister("test_gemv_gateup")
        OPS.unregister("test_gemv_down")
