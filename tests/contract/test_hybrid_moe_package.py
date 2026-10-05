"""Hybrid MoE registry, mixed-format packing and compiler coverage."""
import copy

import numpy as np
import pytest

from tests.hybrid_moe_synth import CFG, write_checkpoint
from monolith.compiler import compile_program
from monolith.compiler.coverage import check_coverage
from monolith.core import Graph
from monolith.core.profile import Profile
from monolith.formats import PackLayout
from monolith.models import resolve_model
from monolith.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeModel
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile

PROF = Profile.from_dict('p', {'gpu_cores': 40, 'nominal_gbps': 614.4,
                              'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})


@pytest.mark.parametrize('quantized', [False, True])
def test_package_roundtrip(tmp_path, quantized):
    held = write_checkpoint(tmp_path, quantized=quantized)
    assert resolve_model('Qwen3_5MoeForConditionalGeneration') is Qwen3_5MoeModel
    m = Qwen3_5MoeModel.from_checkpoint(str(tmp_path), max_context=256)
    assert set(m.full_weight_map()) == set(held)
    assert m.config.rotary_dim == 32 and m.config.is_linear(0) and not m.config.is_linear(1)
    for block in m.blocks:
        assert block.mlp.shared_intermediate == 256 and block.mlp.renorm
    g = Graph('step')
    m.lower(g)
    g.check()
    check_coverage(g, PROF)
    kinds = [op.kind for op in g.ops]
    assert kinds.count('moe_route') == 2 and kinds.count('moe_gemv') == 4
    assert kinds.count('gdn_mixer') == 1 and kinds.count('gqa_decode') == 1
    assert all(op.attrs['has_shared'] for op in g.ops if op.kind == 'moe_combine')
    out = tmp_path / 'pack'
    pack_model(m, str(tmp_path), str(out), PackLayout(rows=16, scale_placement='block'))
    pf = PackFile(out)
    for dynamic, t in [(False, 1), (False, 8), (True, 8)]:
        prog = compile_program(m, pf, PROF, t=t, dynamic_t=dynamic)
        assert sum(o.name == 'moe_combine' for o in prog.ops) == 2
    slab = m.blocks[0].mlp.gate_up.slab.slab_groups()[0]
    if quantized:
        assert slab.format == 'nvfp4'
        assert m.blocks[0].mixer.out_proj.slab_groups()[0].format == 'fp8_e4m3'
    else:
        rows = np.concatenate([held[spec.hf_name] for _, spec in slab.parts])[slab.row_perm]
        np.testing.assert_array_equal(pf.dequantize_slab(slab.name), rows)


@pytest.mark.parametrize('field,value', [('num_experts_per_tok', 17), ('layer_types', ['unknown', 'full_attention']),
                                        ('attention_bias', True), ('shared_expert_intermediate_size', 0)])
def test_unsupported_config_rejected(field, value):
    cfg = copy.deepcopy(CFG)
    cfg['text_config'][field] = value
    with pytest.raises(ValueError):
        Qwen3_5MoeConfig.from_dict(cfg)


def test_real_checkpoint_shape_config():
    cfg = copy.deepcopy(CFG)
    cfg['text_config'].update(hidden_size=2048, num_hidden_layers=40,
        layer_types=['linear_attention'] * 3 + ['full_attention'], num_experts=256,
        num_experts_per_tok=8, moe_intermediate_size=512, shared_expert_intermediate_size=512,
        num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32, vocab_size=248320)
    cfg['text_config']['layer_types'] *= 10
    cfg['text_config']['rope_parameters'].update(rope_theta=10000000., partial_rotary_factor=.25,
                                                mrope_section=[11, 11, 10])
    parsed = Qwen3_5MoeConfig.from_dict(cfg)
    assert sum(parsed.is_linear(i) for i in range(40)) == 30
    assert parsed.num_experts == 256 and parsed.num_experts_per_tok == 8
    assert parsed.rotary_dim == 64
