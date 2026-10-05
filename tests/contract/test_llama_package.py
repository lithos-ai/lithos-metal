"""Llama-family registration, shapes, packing and unsupported-config guards."""
import json
from collections import Counter

import numpy as np
import pytest

from monolith.compiler import compile_program
from monolith.core import Graph, Profile
from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors
from monolith.models import resolve_model
from monolith.models.llama import LlamaConfig, LlamaModel
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile

CFG = dict(architectures=['LlamaForCausalLM'], hidden_size=256, intermediate_size=512,
           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
           rms_norm_eps=1e-5, vocab_size=64, max_position_embeddings=8192, rope_theta=500000.,
           tie_word_embeddings=True, hidden_act='silu')
SCALING = dict(rope_type='llama3', factor=32., low_freq_factor=1., high_freq_factor=4.,
               original_max_position_embeddings=8192)


def checkpoint(path, cfg):
    model = LlamaModel(LlamaConfig.from_dict(cfg), max_context=32)
    rng = np.random.default_rng(7)
    arrays = {}
    for _, _, spec in model.full_weight_map().values():
        value = rng.normal(0, .03, spec.shape).astype(np.float32)
        if spec.aux:
            value += 1
        arrays[spec.hf_name] = ('BF16', f32_to_bf16(value))
    write_safetensors(path / 'model.safetensors', arrays)
    (path / 'config.json').write_text(json.dumps(cfg))
    return {name: bf16_to_f32(value) for name, (_, value) in arrays.items()}


@pytest.mark.parametrize('heads,kv,dim', [(4, 1, 64), (6, 2, 128), (4, 4, 64)])
@pytest.mark.parametrize('tied', [True, False])
def test_package_roundtrip_and_lowering(tmp_path, heads, kv, dim, tied):
    cfg = dict(CFG, num_attention_heads=heads, num_key_value_heads=kv, head_dim=dim,
               tie_word_embeddings=tied, rope_scaling=SCALING)
    held = checkpoint(tmp_path, cfg)
    assert resolve_model('LlamaForCausalLM') is LlamaModel
    model = LlamaModel.from_checkpoint(str(tmp_path), max_context=32, prefix='target.')
    assert {s.hf_name for _, _, s in model.full_weight_map().values()} == set(held)
    assert all(not layer.mixer.gate and not layer.mixer.qk_norm for layer in model.layers())
    assert all(layer.mixer.o_proj.round_residual and layer.mlp.down.round_residual
               and layer.mlp.gate_up.round_silu for layer in model.layers())
    assert (model.lm_head.tied is model.embed_tokens) == tied
    graph = Graph('test')
    model.lower(graph)
    graph.check()
    assert Counter(op.kind for op in graph.ops)['gqa_decode'] == 2
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout(scale_placement='block'))
    pack = PackFile(tmp_path / 'pack')
    expected = np.concatenate([held[f'model.layers.0.self_attn.{part}_proj.weight'] for part in ('q', 'k', 'v')])
    np.testing.assert_array_equal(pack.dequantize_slab('target.layers.0.self_attn.qkv.q_proj+k_proj+v_proj'), expected)
    assert len(model.state_spec().entries) == 4
    profile = Profile.from_dict('test', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16', 'accelerator': 'on'}})
    for tokens in (1, 4):
        program = compile_program(model, pack, profile, t=tokens, attention='auto')
        attention = [k for k in program.kernels.values() if k.function.startswith('gqa_decode')]
        assert attention and all(k.macros['QK_NORM'] == '0' for k in attention)
    assert all(name.startswith('target.') for name in model.tables())


@pytest.mark.parametrize('change', [
    {'attention_bias': True}, {'mlp_bias': True}, {'hidden_act': 'gelu'}, {'sliding_window': 128},
    {'rope_scaling': {'rope_type': 'yarn'}}, {'rope_scaling': {'type': 'dynamic'}},
    {'rope_scaling': dict(SCALING, high_freq_factor=1.)}, {'rope_scaling': {'rope_type': 'llama3'}},
    {'rope_scaling': dict(SCALING, factor=float('nan'))}, {'num_key_value_heads': 3},
    {'num_hidden_layers': 0}, {'head_dim': 48}, {'hidden_size': 576}, {'rms_norm_eps': 0},
])
def test_unsupported_config_fails_early(change):
    with pytest.raises(ValueError):
        LlamaConfig.from_dict(dict(CFG, **change))


def test_config_variants_and_generation_eos(tmp_path):
    cfg = dict(CFG, rope_parameters=dict(SCALING, rope_theta=12345.))
    checkpoint(tmp_path, cfg)
    (tmp_path / 'generation_config.json').write_text(json.dumps({'eos_token_id': [2, 3]}))
    config = LlamaConfig.from_pretrained(tmp_path)
    assert config.rope_theta == 12345. and config.eos_token_id == [2, 3]
    with pytest.raises(ValueError):
        LlamaModel(config, max_context=8193)
    with pytest.raises(ValueError):
        LlamaModel(config, num_layers_override=3)
