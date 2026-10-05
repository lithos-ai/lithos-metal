"""Disabling per-head norms must work through every compiler attention route."""
import pytest
from monolith.compiler import compile_program
from monolith.core import Profile
from monolith.formats import PackLayout
from monolith.models.qwen3 import Qwen3Model
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile
from tests.contract.test_qwen3_package import _checkpoint


@pytest.mark.parametrize('attention', ['v1', 'v2', 'v3', 'mma', 'auto'])
def test_no_query_key_norm_weights_or_loads(tmp_path, attention):
    _checkpoint(tmp_path)
    model = Qwen3Model.from_checkpoint(str(tmp_path), max_context=16)
    for layer in model.layers():
        layer.mixer.qk_norm = False
    assert not any('.q_norm.' in name or '.k_norm.' in name for name in model.full_weight_map())
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    pack = PackFile(tmp_path / 'pack')
    profile = Profile.from_dict('test', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    program = compile_program(model, pack, profile, t=4, attention=attention)
    attention_kernels = [k for k in program.kernels.values() if k.function.startswith('gqa_decode')]
    assert attention_kernels and all(k.macros['QK_NORM'] == '0' for k in attention_kernels)


def test_automatic_mma_requires_validated_norm_or_shape():
    from types import SimpleNamespace
    from monolith.compiler.emit import _gqa_kernel

    ctx = SimpleNamespace(t=8, attention='auto', accelerator='on')
    assert _gqa_kernel(ctx, 24, 8, qk_norm=False) == 'mma'
    assert _gqa_kernel(ctx, 8, 2, qk_norm=False) == 'v3'
    assert _gqa_kernel(ctx, 24, 8, qk_norm=True) == 'mma'
    ctx.attention = 'mma'
    assert _gqa_kernel(ctx, 24, 8, qk_norm=False) == 'mma'


@pytest.mark.parametrize('qk_norm', [False, True])
def test_explicit_mma_selects_probability_precision(tmp_path, qk_norm):
    from tests.moe_synth import write_checkpoint
    from monolith.models.qwen3_moe import Qwen3MoeModel

    write_checkpoint(tmp_path, num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=128)
    model = Qwen3MoeModel.from_checkpoint(str(tmp_path), max_context=16)
    for layer in model.layers():
        layer.mixer.qk_norm = qk_norm
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    profile = Profile.from_dict('test', {'gpu_cores': 40, 'nominal_gbps': 614,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    program = compile_program(model, PackFile(tmp_path / 'pack'), profile, t=4, attention='mma')
    cores = [k for k in program.kernels.values() if k.function == 'gqa_decode_mma']
    assert cores and all(k.macros['MMA_PROB_FP16'] == str(int(not qk_norm)) for k in cores)
    if not qk_norm:
        assert all(k.macros.get('ADAPTIVE_CHUNK') != '1' for k in cores)


def test_v2_grows_partial_buffers_for_long_context(tmp_path):
    """The graph's 64-key workspace cannot hold v2's 32-key chunks."""
    _checkpoint(tmp_path)
    model = Qwen3Model.from_checkpoint(str(tmp_path), max_context=33024)
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    pack = PackFile(tmp_path / 'pack')
    profile = Profile.from_dict('test', {'gpu_cores': 40, 'nominal_gbps': 614,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    program = compile_program(model, pack, profile, t=8, attention='v2')
    cores = [op for op in program.ops if program.kernels[op.kernel].function == 'gqa_decode_v2']
    assert len(cores) == 2
    for op in cores:
        bindings = {index: name for index, name, _ in op.bindings}
        assert program.buffers[bindings[7]].nbytes >= 8 * 8 * 32 * (33024 // 32) * 4
        assert program.buffers[bindings[8]].nbytes >= 8 * 8 * 2 * (33024 // 32) * 4
