"""Measured expert crews stay inside the validated chip/shape/step envelope."""
import copy
import struct
from types import SimpleNamespace

import pytest

from monolith.backends.metal.m5_max_40c.routed import apply_routed_crews
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


def program(t, role):
    gate = role == "gate_up"
    n, k = (1536, 2048) if gate else (2048, 768)
    macros = dict(K=str(k), R="16", T="1", RG="2", RSPLIT="1u",
                  PAIRS="1", K_TOPK="8u", LANE_ORDER="1", CHUNK="8",
                  OUT_BF16="1", NORM="0",
                  EPILOGUE="2" if gate else "0", PAIRS_X_SLOT="0" if gate else "1",
                  STATIC_GEMV_P_N_SG="480u")
    kernel = KernelSpec("", "gemv_T", macros)
    params = struct.pack("<8I", n, k, 480, t, 0, 0, 0, 0)
    return Program({"k": kernel}, {"p": BufferSpec(32, params, "params")}, [
        OpSpec("k", [(4, "p", 0)], (40, 1, 1), (384, 1, 1),
               meta=dict(kind="moe_gemv", format="nvfp4", n=n, k=k, top_k=8))])


def apply(p, **overrides):
    options = dict(t=struct.unpack_from("<I", p.buffers["p"].init, 12)[0],
                   dynamic_t=False, speculative=False)
    options.update(overrides)
    config = options.pop("config", SimpleNamespace(key="apple10", gpu_cores=40))
    return apply_routed_crews(p, config, **options)


@pytest.mark.parametrize("t,role,workers,sgs,rg,split", [
    (1, "gate_up", 320, 32, 1, 8), (1, "down", 80, 32, 2, 8),
    (8, "gate_up", 320, 12, 4, 1), (8, "down", 160, 32, 1, 2),
])
def test_measured_crews_update_the_private_params(t, role, workers, sgs, rg, split):
    p = program(t, role)
    apply(p)
    op = p.ops[0]
    assert op.grid == (workers, 1, 1) and op.threadgroup == (sgs * 32, 1, 1)
    macros = p.kernels[op.kernel].macros
    assert (macros["RG"], macros["RSPLIT"]) == (str(rg), f"{split}u")
    name = op.bindings[0][1]
    assert struct.unpack_from("<I", p.buffers[name].init, 8)[0] == workers * sgs
    assert p.buffers["p"].init == program(t, role).buffers["p"].init


@pytest.mark.parametrize("options", [
    dict(t=4), dict(dynamic_t=True), dict(speculative=True),
    dict(config=SimpleNamespace(key="apple10", gpu_cores=32)),
    dict(config=SimpleNamespace(key="apple9", gpu_cores=40)),
])
def test_unmeasured_execution_modes_keep_original_program(options):
    p = program(1, "gate_up")
    original = copy.deepcopy(p)
    apply(p, **options)
    assert p.to_json() == original.to_json()


@pytest.mark.parametrize("field,value", [
    ("T", "4"), ("PAIRS", "0"), ("K_TOPK", "4u"), ("LANE_ORDER", "0"),
    ("R", "32"), ("CHUNK", "4"), ("EPILOGUE", "0"), ("PAIRS_X_SLOT", "1"),
    ("STAT_OUT", "1"),
    ("NORM", "1"), ("OUT_BF16", "0"),
])
def test_unmeasured_shader_variants_keep_original_program(field, value):
    p = program(1, "gate_up")
    p.kernels["k"].macros[field] = value
    original = copy.deepcopy(p)
    apply(p, t=1)
    assert p.to_json() == original.to_json()


@pytest.mark.parametrize("field,value", [("format", "bf16"), ("n", 3072), ("k", 4096)])
def test_unmeasured_weight_geometry_keeps_original_program(field, value):
    p = program(8, "gate_up")
    p.ops[0].meta[field] = value
    original = copy.deepcopy(p)
    apply(p)
    assert p.to_json() == original.to_json()


@pytest.mark.parametrize('role,workers,sgs',[('gate_up',357,23),('down',1024,32)])
@pytest.mark.parametrize('speculative',[False,True])
def test_hybrid_eight_row_crews(role,workers,sgs,speculative):
    from monolith.backends.metal.m5_max_40c.routed import apply_hybrid_crews
    p=program(8,role)
    p.ops[0].meta.update(n=1024 if role=='gate_up' else 2048,k=2048 if role=='gate_up' else 512)
    p.kernels['k'].macros['K']=str(p.ops[0].meta['k'])
    if speculative:
        p.kernels['accept']=KernelSpec('','accept_scan',{})
        p.ops.append(OpSpec('accept',[],(1,1,1),(32,1,1)))
    apply_hybrid_crews(p,SimpleNamespace(key='apple10',gpu_cores=40),t=8,
                      dynamic_t=speculative,speculative=speculative)
    assert p.ops[0].grid==(workers,1,1) and p.ops[0].threadgroup==(sgs*32,1,1)


@pytest.mark.parametrize('options',[dict(t=1),dict(t=16),dict(dynamic_t=True),
    dict(speculative=True),dict(config=SimpleNamespace(key='apple10',gpu_cores=32)),
    dict(config=SimpleNamespace(key='apple9',gpu_cores=40))])
def test_hybrid_crews_do_not_change_other_chips_or_prefill(options):
    from monolith.backends.metal.m5_max_40c.routed import apply_hybrid_crews
    p=program(8,'gate_up');p.ops[0].meta.update(n=1024,k=2048)
    original=copy.deepcopy(p)
    args=dict(config=SimpleNamespace(key='apple10',gpu_cores=40),t=8,dynamic_t=False,speculative=False)
    args.update(options);apply_hybrid_crews(p,**args)
    assert p.to_json()==original.to_json()


def test_hybrid_crews_are_used_by_backend_finalize():
    from monolith.backends.metal.m5_max_40c.scheduling import finalize
    p=program(8,'gate_up');p.ops[0].meta.update(n=1024,k=2048)
    p.kernels['accept']=KernelSpec('','accept_scan',{})
    p.ops.append(OpSpec('accept',[],(1,1,1),(32,1,1)))
    result=finalize(p,None,None,SimpleNamespace(key='apple10',gpu_cores=40),
                    t=8,dynamic_t=True,speculative=True,commute_norm=True,
                    accelerator='on',gdn_mixer_fusion=False,barriers='minimal')
    assert result.ops[0].grid==(357,1,1)
    assert result.ops[0].threadgroup==(736,1,1)
