"""Tiny ModelOpt-layout hybrid MoE; both mixers, routed and shared experts."""
import copy
import json

import numpy as np

from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors

P = "model.language_model."
CFG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text", "hidden_size": 256, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "layer_types": ["linear_attention", "full_attention"], "attn_output_gate": True,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "moe_intermediate_size": 256, "shared_expert_intermediate_size": 256,
        "num_experts": 8, "num_experts_per_tok": 2, "rms_norm_eps": 1e-6,
        "hidden_act": "silu", "vocab_size": 512, "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 0.5,
                            "mrope_section": [5, 5, 6], "mrope_interleaved": True},
    },
}


def write_checkpoint(path, seed=23, *, quantized=False):
    from monolith.formats import FORMATS

    c = copy.deepcopy(CFG)
    t = c["text_config"]
    rng = np.random.default_rng(seed)
    h, mi, v, E = t["hidden_size"], t["moe_intermediate_size"], t["vocab_size"], t["num_experts"]

    def w(*shape, scale=.025):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    raw = {P + "embed_tokens.weight": w(v, h, scale=.5), P + "norm.weight": w(h),
           "lm_head.weight": w(v, h, scale=.3)}
    for i, kind in enumerate(t["layer_types"]):
        p = f"{P}layers.{i}."
        raw[p + "input_layernorm.weight"] = w(h)
        raw[p + "post_attention_layernorm.weight"] = w(h)
        if kind == "linear_attention":
            a = p + "linear_attn."
            raw.update({a + "in_proj_qkv.weight": w(1024, h), a + "in_proj_z.weight": w(512, h),
                        a + "in_proj_a.weight": w(4, h), a + "in_proj_b.weight": w(4, h),
                        a + "out_proj.weight": w(h, 512), a + "conv1d.weight": w(1024, 1, 4, scale=.2),
                        a + "A_log": w(4), a + "dt_bias": w(4), a + "norm.weight": 1 + w(128)})
        else:
            a = p + "self_attn."
            raw.update({a + "q_proj.weight": w(512, h), a + "k_proj.weight": w(128, h),
                        a + "v_proj.weight": w(128, h), a + "o_proj.weight": w(h, 256),
                        a + "q_norm.weight": w(64), a + "k_norm.weight": w(64)})
        raw[p + "mlp.gate.weight"] = w(E, h, scale=.2)
        for expert in [f"experts.{e}" for e in range(E)] + ["shared_expert"]:
            a = p + f"mlp.{expert}."
            raw.update({a + "gate_proj.weight": w(mi, h), a + "up_proj.weight": w(mi, h),
                        a + "down_proj.weight": w(h, mi)})
        raw[p + "mlp.shared_expert_gate.weight"] = w(1, h, scale=.2)
    tensors = {n: ("BF16", f32_to_bf16(a)) for n, a in raw.items()}
    if quantized:
        for n, a in raw.items():
            if a.ndim != 2 or not n.endswith('.weight'):
                continue
            expert = '.mlp.experts.' in n or '.mlp.shared_expert.' in n or n == 'lm_head.weight'
            mixer = any(f'.{x}.' in n for x in ('in_proj_qkv', 'in_proj_z', 'out_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj'))
            if not (expert or mixer):
                continue
            fmt = FORMATS.get('nvfp4' if expert else 'fp8_e4m3')
            spec = fmt.quantize(a)
            # The format plugin's checkpoint representation, including tensor scales.
            tensors.pop(n)
            base = n.removesuffix('.weight')
            tensors[n] = ('U8' if expert else 'F8_E4M3', spec.tensors['weight'])
            if expert:
                tensors[base + '.weight_scale'] = ('F8_E4M3', spec.tensors['weight_scale'])
                tensors[base + '.weight_scale_2'] = ('F32', np.asarray(spec.params['weight_scale_2'], dtype=np.float32))
            else:
                tensors[base + '.weight_scale'] = ('F32', np.asarray(spec.params['weight_scale'], dtype=np.float32))
    # These must never enter the text pack.
    tensors['model.visual.unused.weight'] = ('BF16', f32_to_bf16(w(1)))
    tensors['mtp.unused.weight'] = ('BF16', f32_to_bf16(w(1)))
    write_safetensors(path / 'model.safetensors', tensors)
    (path / 'config.json').write_text(json.dumps(c))
    return {n: bf16_to_f32(f32_to_bf16(a)) for n, a in raw.items()}
