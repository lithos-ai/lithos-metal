"""Llama-compatible dense text models, including Llama 3.2 and SmolLM2."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LlamaConfig:
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
    rope_scaling: dict = field(default_factory=dict)
    tie_word_embeddings: bool = False
    eos_token_id: Any = None
    architecture: str = 'LlamaForCausalLM'

    @classmethod
    def from_dict(cls, d):
        if (d.get("architectures") or ["LlamaForCausalLM"])[0] != "LlamaForCausalLM":
            raise ValueError("expected a LlamaForCausalLM checkpoint")
        rope = dict(d.get('rope_parameters') or d.get('rope_scaling') or {})
        rope_type = rope.get('rope_type', rope.get('type', 'default'))
        if rope_type not in ('default', 'llama3'):
            raise ValueError(f'unsupported RoPE scaling: {rope_type}')
        if d.get('hidden_act', 'silu') != 'silu' or d.get('attention_bias') or d.get('mlp_bias'):
            raise ValueError('this package requires bias-free attention/MLP and silu activation')
        if d.get('sliding_window') or d.get('use_sliding_window') or rope.get('partial_rotary_factor', 1.) != 1.:
            raise ValueError('this package requires full-context attention and full-width RoPE')
        h, heads = int(d['hidden_size']), int(d['num_attention_heads'])
        if heads <= 0 or h <= 0:
            raise ValueError('hidden size and attention head count must be positive')
        c = cls(h, int(d['intermediate_size']), int(d['num_hidden_layers']), heads,
                int(d.get('num_key_value_heads', heads)), int(d.get('head_dim') or h // heads),
                float(d.get('rms_norm_eps', 1e-6)), int(d['vocab_size']), int(d['max_position_embeddings']),
                float(rope.get('rope_theta', d.get('rope_theta', 10000.))),
                dict(rope, rope_type=rope_type), bool(d.get('tie_word_embeddings', False)), d.get('eos_token_id'))
        if min(c.num_key_value_heads, c.num_hidden_layers, c.intermediate_size, c.vocab_size, c.max_position_embeddings) <= 0:
            raise ValueError('model dimensions must be positive')
        if heads % c.num_key_value_heads or c.head_dim not in (32, 64, 128, 256):
            raise ValueError('attention requires integral query replication and head_dim 32/64/128/256')
        if h % 256 or c.intermediate_size % 256 or (heads * c.head_dim) % 256:
            raise ValueError('projection input widths must be multiples of 256 for the weight pack')
        if not all(math.isfinite(x) and x > 0 for x in (c.rope_theta, c.rms_norm_eps)):
            raise ValueError('RoPE theta and norm epsilon must be finite and positive')
        if rope_type == 'llama3':
            keys = ('factor', 'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings')
            if any(k not in rope for k in keys):
                raise ValueError('llama3 RoPE requires factor, frequency bounds and original context length')
            factor, low, high, original = (float(rope[k]) for k in keys)
            if not all(math.isfinite(x) for x in (factor, low, high, original)) or not (factor >= 1 and 0 < low < high and original > 0):
                raise ValueError('invalid llama3 RoPE scaling parameters')
            c.rope_scaling.update(zip(keys, (factor, low, high, original)))
        return c

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)
        c = cls.from_dict(json.loads((path / 'config.json').read_text()))
        if (path / 'generation_config.json').exists():
            c.eos_token_id = json.loads((path / 'generation_config.json').read_text()).get('eos_token_id', c.eos_token_id)
        return c
