"""The drafter's config from its ``config.json`` (DeepSpec's ``build_draft_config`` fields on top of the target's
transformer fields; gittensor's checkpoint keeps the DSpark fields under ``dflash_config``)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DSparkConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    block_size: int
    target_layer_ids: List[int]
    mask_token_id: int
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    target_hidden_size: Optional[int] = None
    rope_type: str = "default"
    rope_parameters: Dict[str, Any] = field(default_factory=dict)
    architecture: str = ""                  # informational: the checkpoint's architectures[0]
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DSparkConfig":
        ds = dict(d.get("dspark_config") or d.get("dflash_config") or {})
        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        get = lambda k, default=None: d.get(k, ds.get(k, default))          # noqa: E731
        cfg = cls(
            hidden_size=d["hidden_size"], intermediate_size=d["intermediate_size"], num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"], num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"], rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            vocab_size=d.get("draft_vocab_size") or d["vocab_size"], rope_theta=float(rope.get("rope_theta", d.get("rope_theta", 1000000.0))),
            block_size=int(get("block_size") or 7),                # TorchSpec checkpoints omit it; DSpark's block is 7
            target_layer_ids=[int(x) for x in get("target_layer_ids")], mask_token_id=int(get("mask_token_id")),
            markov_rank=int(get("markov_rank", 0) or 0), markov_head_type=str(get("markov_head_type", "vanilla")),
            enable_confidence_head=bool(get("enable_confidence_head", False)), confidence_head_with_markov=bool(get("confidence_head_with_markov", False)),
            target_hidden_size=d.get("target_hidden_size"), rope_type=str(rope.get("rope_type", "default")),
            rope_parameters=dict(rope),
            architecture=(d.get("architectures") or [""])[0],
        )
        if cfg.markov_head_type not in ("vanilla",):
            raise ValueError(f"unsupported markov_head_type {cfg.markov_head_type!r} (vanilla only)")
        from ...nn.rope import scaled_inv_freq
        scaled_inv_freq(cfg.rope_theta, cfg.head_dim, cfg.rope_parameters)
        return cfg

    @classmethod
    def from_pretrained(cls, path: str) -> "DSparkConfig":
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_dict(json.load(f))

    @property
    def target_hidden(self) -> int:
        return self.target_hidden_size or self.hidden_size

    @property
    def n_taps(self) -> int:
        return len(self.target_layer_ids)
