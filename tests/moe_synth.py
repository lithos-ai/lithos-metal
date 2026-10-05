"""A synthetic sparse Qwen3-MoE checkpoint for the tests (tiny: 2 layers, the first sparse with 8 experts top-2 of width
256 — the GEMV kernels need K % 256 == 0 — the second a dense MLP; hidden 256, GQA 4/2 heads of 64) — BF16 safetensors + config.json in the checkpoint's layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors

P = "model."
CFG: Dict[str, Any] = {
    "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 256,
    "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [1],
    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-6,
    "vocab_size": 512, "max_position_embeddings": 4096, "rope_theta": 10000.0, "tie_word_embeddings": False, "hidden_act": "silu",
}


def write_checkpoint(path: Path, seed: int = 7, **over: Any) -> Dict[str, np.ndarray]:
    """Writes ``model.safetensors`` + ``config.json``; returns the BF16-valued parameters as float32 arrays."""
    c = dict(CFG, **over)
    rng = np.random.default_rng(seed)
    h, inter, mi, d, v = c["hidden_size"], c["intermediate_size"], c["moe_intermediate_size"], c["head_dim"], c["vocab_size"]
    heads, kv, E = c["num_attention_heads"], c["num_key_value_heads"], c["num_experts"]

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n):
        return (1.0 + rng.standard_normal(n) * 0.1).astype(np.float32)

    raw = {f"{P}embed_tokens.weight": w(v, h, scale=0.5), f"{P}norm.weight": norm(h), "lm_head.weight": w(v, h, scale=0.3)}
    for i in range(c["num_hidden_layers"]):
        L = f"{P}layers.{i}."
        raw.update({L + "input_layernorm.weight": norm(h), L + "post_attention_layernorm.weight": norm(h),
                    L + "self_attn.q_proj.weight": w(heads * d, h), L + "self_attn.k_proj.weight": w(kv * d, h),
                    L + "self_attn.v_proj.weight": w(kv * d, h), L + "self_attn.o_proj.weight": w(h, heads * d),
                    L + "self_attn.q_norm.weight": norm(d), L + "self_attn.k_norm.weight": norm(d)})
        sparse = i not in c["mlp_only_layers"] and E > 0 and (i + 1) % c["decoder_sparse_step"] == 0
        if sparse:
            raw[L + "mlp.gate.weight"] = w(E, h, scale=0.2)
            for e in range(E):
                raw.update({L + f"mlp.experts.{e}.gate_proj.weight": w(mi, h), L + f"mlp.experts.{e}.up_proj.weight": w(mi, h),
                            L + f"mlp.experts.{e}.down_proj.weight": w(h, mi)})
        else:
            raw.update({L + "mlp.gate_proj.weight": w(inter, h), L + "mlp.up_proj.weight": w(inter, h), L + "mlp.down_proj.weight": w(h, inter)})
    write_safetensors(path / "model.safetensors", {k: ("BF16", f32_to_bf16(a)) for k, a in raw.items()}, {"format": "pt"})
    with open(path / "config.json", "w") as f:
        json.dump(c, f)
    return {k: bf16_to_f32(f32_to_bf16(a)) for k, a in raw.items()}
