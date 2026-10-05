"""Model 2 (plan M8): the dense Qwen3 package from the layer library alone — weight map, lowering, pack round-trip
and full kernel coverage on a synthetic checkpoint (no torch, no GPU)."""

import json
from collections import Counter

import numpy as np

from monolith.compiler import check_coverage, compile_program
from monolith.core import Graph
from monolith.core.profile import Profile
from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors
from monolith.models import resolve_model
from monolith.models.qwen3 import Qwen3Config, Qwen3Model
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile

CFG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 256, "intermediate_size": 256, "num_hidden_layers": 2,
       "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 64,
       "max_position_embeddings": 4096, "rope_theta": 1000000.0, "tie_word_embeddings": False, "hidden_act": "silu"}
P = "model."


def _checkpoint(tmp_path):
    rng = np.random.default_rng(5)
    h, inter, d = 256, 256, 32

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    raw = {f"{P}embed_tokens.weight": w(64, h), "lm_head.weight": w(64, h), f"{P}norm.weight": w(h)}
    for i in range(2):
        L = f"{P}layers.{i}."
        raw.update({L + "input_layernorm.weight": w(h), L + "post_attention_layernorm.weight": w(h),
                    L + "self_attn.q_proj.weight": w(8 * d, h), L + "self_attn.k_proj.weight": w(2 * d, h), L + "self_attn.v_proj.weight": w(2 * d, h),
                    L + "self_attn.o_proj.weight": w(h, 8 * d), L + "self_attn.q_norm.weight": w(d), L + "self_attn.k_norm.weight": w(d),
                    L + "mlp.gate_proj.weight": w(inter, h), L + "mlp.up_proj.weight": w(inter, h), L + "mlp.down_proj.weight": w(h, inter)})
    tensors = {k: ("BF16", f32_to_bf16(v)) for k, v in raw.items()}
    write_safetensors(tmp_path / "model.safetensors", tensors, {"format": "pt"})
    with open(tmp_path / "config.json", "w") as f:
        json.dump(CFG, f)
    return {k: bf16_to_f32(f32_to_bf16(v)) for k, v in raw.items()}


def test_package_registers_lowers_and_packs(tmp_path):
    held = _checkpoint(tmp_path)
    assert resolve_model("Qwen3ForCausalLM") is Qwen3Model
    m = Qwen3Model.from_checkpoint(str(tmp_path), max_context=16)
    assert set(m.full_weight_map()) == set(held)
    assert m.lm_head.tied is None and m.blocks[0].mixer.gate is False and m.blocks[0].mixer.rotary_dim == 32
    assert [e.name for e in m.state_spec().entries] == ["layers.0.self_attn.k_cache", "layers.0.self_attn.v_cache",
                                                        "layers.1.self_attn.k_cache", "layers.1.self_attn.v_cache"]
    g = Graph("step")
    m.lower(g)
    g.check()
    kinds = Counter(op.kind for op in g.ops)
    assert kinds == {"gemv": 8, "gqa_decode": 2, "gqa_merge": 2, "rmsnorm_stat": 5, "embed": 1, "lm_head": 1, "argmax": 1}
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    check_coverage(g, prof)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    pf = PackFile(tmp_path / "pack")
    L0 = f"{P}layers.0."
    # no gate, full RoPE: the qkv slab is [q | k | v] in checkpoint order
    exp = np.concatenate([held[L0 + "self_attn.q_proj.weight"], held[L0 + "self_attn.k_proj.weight"], held[L0 + "self_attn.v_proj.weight"]])
    assert np.array_equal(pf.dequantize_slab("layers.0.self_attn.qkv.q_proj+k_proj+v_proj"), exp)
    assert np.array_equal(pf.aux_array("layers.0.self_attn.q_norm"), held[L0 + "self_attn.q_norm.weight"])           # w, not 1 + w
    assert np.array_equal(pf.aux_array("layers.0.input_norm.weight"), held[L0 + "input_layernorm.weight"])
    assert np.array_equal(pf.dequantize_slab("lm_head.weight"), held["lm_head.weight"])
    cos = bf16_to_f32(pf.aux_array("rope_cos"))
    assert cos.shape == (16, 32) and abs(cos[1, 0] - np.cos(1.0)) < 1e-2 and abs(cos[1, 16] - np.cos(1.0)) < 1e-2   # cat(freqs, freqs)
    prog = compile_program(m, pf, prof, t=1)
    assert sum(o.name.startswith("gqa") for o in prog.ops) == 4 and not any(o.name == "gdn_mixer" for o in prog.ops)
