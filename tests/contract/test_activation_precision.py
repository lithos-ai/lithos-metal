"""Precision choices belong to the layer IR and reach every emitted consumer."""
import pytest
from monolith.compiler import compile_program
from monolith.core import Profile
from monolith.formats import PackLayout
from monolith.models.qwen3 import Qwen3Model
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile
from tests.contract.test_qwen3_package import _checkpoint


@pytest.mark.parametrize('accelerator,t', [('off', 1), ('off', 4), ('on', 1), ('on', 4)])
@pytest.mark.parametrize('rounded', [False, True])
def test_precision_reaches_norm_and_projection(tmp_path, accelerator, t, rounded):
    _checkpoint(tmp_path)
    model = Qwen3Model.from_checkpoint(str(tmp_path), max_context=16)
    for block in model.layers():
        block.input_norm.round_before_scale = rounded
        block.post_norm.round_before_scale = rounded
        block.mlp.gate_up.round_silu = rounded
    model.norm.round_before_scale = rounded
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    profile = Profile.from_dict('test', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16', 'accelerator': accelerator}})
    program = compile_program(model, PackFile(tmp_path / 'pack'), profile, t=t)
    norms = [k for k in program.kernels.values() if k.function == 'norm_apply' or k.macros.get('PERM_NORM') == '1' or k.macros.get('DIRECT_NORM') == '1']
    gates = [k for k in program.kernels.values() if k.function in ('gemv_T', 'gemm_tile', 'gemv_bf16_small', 'gemv_bf16_rows') and k.macros.get('EPILOGUE') == '2']
    assert norms and gates
    assert all((k.macros.get('NORM_ROUND') == '1') == rounded for k in norms)
    assert all((k.macros.get('SILU_ROUND') == '1') == rounded for k in gates)
