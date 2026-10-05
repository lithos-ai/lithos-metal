"""Text config of the ``qwen3_5`` hybrid, parsed straight from ``config.json['text_config']``.

adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/configuration.py @ 5beaed8 (Apache-2.0): the same
field-for-field mirror, without the tensor-parallel assumptions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Qwen3_5Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: List[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = True
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    eos_token_id: Optional[Any] = None
    architecture: str = "Qwen3_5ForConditionalGeneration"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3_5Config":
        text = d.get("text_config", d)
        rope = text.get("rope_parameters", {})
        cfg = cls(
            hidden_size=text["hidden_size"], intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"], num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"], head_dim=text["head_dim"],
            layer_types=list(text["layer_types"]), linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"], linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"], linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            rms_norm_eps=float(text["rms_norm_eps"]), vocab_size=text["vocab_size"],
            max_position_embeddings=text["max_position_embeddings"],
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10000.0))),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))),
            attn_output_gate=bool(text.get("attn_output_gate", True)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", d.get("tie_word_embeddings", False))),
            hidden_act=text.get("hidden_act", "silu"), eos_token_id=text.get("eos_token_id", d.get("eos_token_id")),
            architecture=(d.get("architectures") or ["Qwen3_5ForConditionalGeneration"])[0],
        )
        if cfg.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {cfg.hidden_act!r}")
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope_type {rope.get('rope_type')!r}")
        if len(cfg.layer_types) != cfg.num_hidden_layers:
            raise ValueError("layer_types does not match num_hidden_layers")
        return cfg

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3_5Config":
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = cls.from_dict(json.load(f))
        gen_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                gen = json.load(f)
            if "eos_token_id" in gen:
                cfg.eos_token_id = gen["eos_token_id"]
        return cfg

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def is_linear(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"
