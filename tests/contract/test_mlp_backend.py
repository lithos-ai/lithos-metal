"""The measured ragged INT4 MLP policy must not spill into other executions."""
import copy
from types import SimpleNamespace

import pytest

from monolith.backends.metal.m5_max_40c import mlp
from monolith.runtime.program import KernelSpec, OpSpec, Program


def program():
    common = dict(R="16", LANE_ORDER="1", OUT_BF16="1", SCALE_F16="0")
    gate = KernelSpec("", "gemm_tile", dict(common, TM="8", EPILOGUE="2", POST_NORM="1", STAT_OUT="0"))
    down = KernelSpec("", "gemv_T", dict(common, T="8", EPILOGUE="1", NORM="0", STAT_OUT="1", PAIRS="0"))
    return Program({"gate": gate, "down": down}, {}, [
        OpSpec("gate", [], (80, 1, 1), (384, 1, 1),
               meta=dict(format="int4_affine", n=7168, k=1024)),
        OpSpec("down", [], (80, 1, 1), (384, 1, 1),
               meta=dict(format="int4_affine", n=1024, k=3584)),
    ])


def apply(p, **changes):
    config = changes.pop("config", SimpleNamespace(key="apple10", gpu_cores=40,
                          threadgroups_per_core=2, sibling_order="alu_first"))
    options = dict(t=8, dynamic_t=False, speculative=False, commute_norm=True, accelerator="on")
    options.update(changes)
    return mlp.apply_mlp_crews(p, config, **options)


def test_only_gate_tile_is_normalized_and_down_keeps_logical_tasks(monkeypatch):
    calls = []
    def normalize(p, regions):
        calls.extend(regions)
        return copy.deepcopy(p), None
    monkeypatch.setattr(mlp, "fuse_regions", normalize)
    p = program()
    result = apply(p)
    assert [(a, b) for a, b, _, _ in calls] == [(0, 1)]
    assert calls[0][3] == dict(workers=80, sgs=8, tn=16, ksplit=1, compact=True, fuse=False)
    assert len(result.ops) == 2
    assert result.ops[1].grid == (120, 1, 1)
    assert result.ops[1].threadgroup == (256, 1, 1)
    assert result.kernels["down"].macros == p.kernels["down"].macros
    assert p.ops[1].grid == (80, 1, 1)


@pytest.mark.parametrize("options", [
    dict(t=1), dict(t=4), dict(dynamic_t=True), dict(speculative=True),
    dict(commute_norm=False), dict(accelerator="off"),
    *[dict(config=SimpleNamespace(key=key, gpu_cores=cores,
                 threadgroups_per_core=crew, sibling_order=order))
      for key, cores, crew, order in [("apple9", 40, 2, "alu_first"),
             ("apple10", 32, 2, "alu_first"), ("apple10", 40, 1, "alu_first"),
             ("apple10", 40, 2, "bus_first")]],
])
def test_unmeasured_execution_is_unchanged(options):
    p = program()
    original = p.to_json()
    assert apply(p, **options).to_json() == original


@pytest.mark.parametrize("which,field,value", [
    ("gate", "TM", "4"), ("gate", "POST_NORM", "0"),
    ("gate", "EPILOGUE", "0"), ("gate", "STAT_OUT", "1"),
    ("down", "T", "1"), ("down", "PAIRS", "1"),
    ("down", "NORM", "1"), ("down", "STAT_OUT", "0"),
    ("down", "EPILOGUE", "0"), ("gate", "SCALE_F16", "1"),
    ("down", "OUT_BF16", "0"), ("gate", "LANE_ORDER", "0"),
    ("down", "R", "32"),
])
def test_unmeasured_shader_is_unchanged(which, field, value):
    p = program()
    p.kernels[which].macros[field] = value
    original = p.to_json()
    assert apply(p).to_json() == original


@pytest.mark.parametrize("index,field,value", [(0, "format", "nvfp4"),
    (0, "n", 8192), (0, "k", 2048), (1, "n", 2048), (1, "k", 4096)])
def test_unmeasured_projection_geometry_is_unchanged(index, field, value):
    p = program()
    p.ops[index].meta[field] = value
    original = p.to_json()
    assert apply(p).to_json() == original


def test_partial_logical_crew_is_unchanged():
    p = program()
    p.ops[1].grid = (1, 1, 1)
    p.ops[1].threadgroup = (32, 1, 1)
    original = p.to_json()
    assert apply(p).to_json() == original


@pytest.mark.parametrize('transform',['apply_mlp_crews','apply_hybrid_down_fusion'])
def test_backend_preserves_requested_barrier_mode(monkeypatch,transform):
    from monolith.backends.metal.m5_max_40c import scheduling
    from monolith.compiler import barriers

    p, transformed = program(), program()
    calls = []
    for name in ('apply_routed_crews','apply_hybrid_crews','apply_mlp_crews','apply_hybrid_down_fusion'):
        monkeypatch.setattr(scheduling,name,lambda p,*a,**kw:p)
    monkeypatch.setattr(scheduling,transform,lambda *a,**kw:transformed)
    monkeypatch.setattr(barriers, "place_barriers", lambda p, mode: calls.append((p, mode)))
    result = scheduling.finalize(p, None, None, None, t=8, dynamic_t=False,
        speculative=False, commute_norm=True, accelerator="on",
        gdn_mixer_fusion=False, barriers="all")
    assert result is transformed and calls == [(transformed, "all")]
