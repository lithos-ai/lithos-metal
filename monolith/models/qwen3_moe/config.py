"""Config of the sparse ``qwen3_moe`` architecture (the dense ``qwen3`` fields plus the mixture-of-experts ones),
parsed from ``config.json``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Qwen3MoeConfig:
    hidden_size: int
    intermediate_size: int                # the dense layers' MLP (the layers `mlp_only_layers` names, or every layer when num_experts is 0)
    moe_intermediate_size: int            # each expert's
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    eos_token_id: Optional[Any] = None
    architecture: str = "Qwen3MoeForCausalLM"

    def is_sparse(self, layer: int) -> bool:
        """The reference's rule: a layer is a mixture of experts unless listed in ``mlp_only_layers`` or skipped by
        ``decoder_sparse_step``."""
        return layer not in self.mlp_only_layers and self.num_experts > 0 and (layer + 1) % self.decoder_sparse_step == 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3MoeConfig":
        text = d.get("text_config", d)
        rope = text.get("rope_parameters") or {}
        cfg = cls(
            hidden_size=text["hidden_size"], intermediate_size=text["intermediate_size"], moe_intermediate_size=text["moe_intermediate_size"],
            num_experts=int(text["num_experts"]), num_experts_per_tok=int(text["num_experts_per_tok"]), num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"], num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"], rms_norm_eps=float(text["rms_norm_eps"]),
            vocab_size=text["vocab_size"], max_position_embeddings=text.get("max_position_embeddings", 40960),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 1000000.0))),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)), decoder_sparse_step=int(text.get("decoder_sparse_step", 1)),
            mlp_only_layers=list(text.get("mlp_only_layers") or []),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", d.get("tie_word_embeddings", False))),
            hidden_act=text.get("hidden_act", "silu"), eos_token_id=text.get("eos_token_id", d.get("eos_token_id")),
            architecture=(d.get("architectures") or ["Qwen3MoeForCausalLM"])[0],
        )
        if cfg.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {cfg.hidden_act!r}")
        rope_type = rope.get("rope_type") or (text.get("rope_scaling") or {}).get("rope_type") or "default"
        if text.get("use_sliding_window") or rope_type != "default":
            raise ValueError("sliding-window attention and scaled RoPE are not supported by this package")
        return cfg

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3MoeConfig":
        with open(os.path.join(model_path, "config.json")) as f:
            return cls.from_dict(json.load(f))
