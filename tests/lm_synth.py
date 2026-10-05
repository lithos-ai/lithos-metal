"""A synthetic dense Qwen3 checkpoint (the ``qwen3`` package's architecture) for the LM-drafter tests: two layers of
GQA attention (8 heads / 2 kv heads, head dim 32) over hidden 256, a vocabulary the caller picks (the target's), tied
or separate head; every parameter BF16."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors

P = "model."


def config(vocab_size: int = 50, tie: bool = False, layers: int = 2) -> Dict:
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 256, "intermediate_size": 256,
            "num_hidden_layers": layers, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32, "rms_norm_eps": 1e-6,
            "vocab_size": vocab_size, "max_position_embeddings": 4096, "rope_theta": 10000.0, "tie_word_embeddings": tie, "hidden_act": "silu"}


def write_checkpoint(path: Path, seed: int = 7, *, vocab_size: int = 50, tie: bool = False, layers: int = 2, scale: float = 0.05) -> Dict[str, np.ndarray]:
    """Writes ``model.safetensors`` + ``config.json``; returns the BF16-valued parameters as float32 arrays."""
    rng = np.random.default_rng(seed)
    c = config(vocab_size, tie, layers)
    h, inter, d, heads, kv = c["hidden_size"], c["intermediate_size"], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]

    def w(*shape, s=scale):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def norm(n):
        return (1.0 + rng.standard_normal(n) * 0.1).astype(np.float32)

    raw = {f"{P}embed_tokens.weight": w(vocab_size, h, s=0.5), f"{P}norm.weight": norm(h)}
    if not tie:
        raw["lm_head.weight"] = w(vocab_size, h, s=0.3)
    for i in range(layers):
        L = f"{P}layers.{i}."
        raw.update({L + "input_layernorm.weight": norm(h), L + "post_attention_layernorm.weight": norm(h),
                    L + "self_attn.q_proj.weight": w(heads * d, h), L + "self_attn.k_proj.weight": w(kv * d, h),
                    L + "self_attn.v_proj.weight": w(kv * d, h), L + "self_attn.o_proj.weight": w(h, heads * d),
                    L + "self_attn.q_norm.weight": norm(d), L + "self_attn.k_norm.weight": norm(d),
                    L + "mlp.gate_proj.weight": w(inter, h), L + "mlp.up_proj.weight": w(inter, h), L + "mlp.down_proj.weight": w(h, inter)})
    write_safetensors(path / "model.safetensors", {k: ("BF16", f32_to_bf16(a)) for k, a in raw.items()}, {"format": "pt"})
    with open(path / "config.json", "w") as f:
        json.dump(c, f)
    return {k: bf16_to_f32(f32_to_bf16(a)) for k, a in raw.items()}
