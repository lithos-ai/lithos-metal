"""Config of the dense ``qwen3`` architecture, parsed from ``config.json`` (the fields the module tree needs)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    eos_token_id: Optional[Any] = None
    architecture: str = "Qwen3ForCausalLM"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3Config":
        text = d.get("text_config", d)
        rope = text.get("rope_parameters") or {}
        cfg = cls(
            hidden_size=text["hidden_size"], intermediate_size=text["intermediate_size"], num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"], num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"], rms_norm_eps=float(text["rms_norm_eps"]),
            vocab_size=text["vocab_size"], max_position_embeddings=text.get("max_position_embeddings", 40960),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 1000000.0))),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", d.get("tie_word_embeddings", False))),
            hidden_act=text.get("hidden_act", "silu"), eos_token_id=text.get("eos_token_id", d.get("eos_token_id")),
            architecture=(d.get("architectures") or ["Qwen3ForCausalLM"])[0],
        )
        if cfg.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {cfg.hidden_act!r}")
        rope_type = rope.get("rope_type") or (text.get("rope_scaling") or {}).get("rope_type") or "default"
        if text.get("use_sliding_window") or rope_type != "default":
            raise ValueError("sliding-window attention and scaled RoPE are not supported by this package")
        return cfg

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3Config":
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = cls.from_dict(json.load(f))
        gen_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                gen = json.load(f)
            if "eos_token_id" in gen:
                cfg.eos_token_id = gen["eos_token_id"]
        return cfg
