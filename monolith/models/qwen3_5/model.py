"""The module tree: embedding → ``num_hidden_layers`` pre-norm decoder layers (GDN or gated GQA attention by
``layer_types``, each with a gated MLP) → final norm → ``lm_head`` (tied to the embedding when the config says so)
→ greedy sampler. The step program follows this order (design §5.1: five all-to-all stages per layer).

adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/modeling.py @ 5beaed8 (Apache-2.0): the layer
structure and the ``[q | gate | k | v]`` / ``[qkv | z | a | b]`` stacked projections, without tensor parallelism.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ...core.dtypes import DType
from ...core.ir import Graph, Value
from ...core.shapes import T
from ...formats.fp import f32_to_bf16
from ...nn import (DecoderLayer, Embedding, GatedDeltaNet, GatedMLP, GQAAttention, GreedySampler, LMHead, LowerContext,
                   Model, Module, RMSNorm, StateEntry, StateSpec, state_shape)
from ...nn.rope import rope_tables_permuted
from ..registry import register_model
from .config import Qwen3_5Config
from .weights import TEXT_PREFIX, bind_checkpoint_formats


@register_model("Qwen3_5ForConditionalGeneration")
class Qwen3_5Model(Model):
    def __init__(self, config: Qwen3_5Config, *, max_context: int = 4096, pack_rows: int = 16,
                 num_layers_override: int | None = None) -> None:
        super().__init__(prefix="")
        self.config = config
        self.max_context = max_context
        c = config
        h, eps = c.hidden_size, c.rms_norm_eps
        n_layers = c.num_hidden_layers if num_layers_override is None else num_layers_override
        self.n_layers = n_layers
        p = TEXT_PREFIX
        self.embed_tokens = Embedding(c.vocab_size, h, f"{p}embed_tokens.weight", prefix="embed_tokens.")
        self.blocks: List[DecoderLayer] = []
        for i in range(n_layers):
            lp, hp = f"layers.{i}.", f"{p}layers.{i}."
            if c.is_linear(i):
                mixer: Module = GatedDeltaNet(h, c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim,
                                              c.linear_value_head_dim, c.linear_conv_kernel_dim, eps,
                                              hf_prefix=f"{hp}linear_attn.", prefix=f"{lp}linear_attn.")
            else:
                mixer = GQAAttention(h, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim, c.rope_theta,
                                     eps, hf_prefix=f"{hp}self_attn.", prefix=f"{lp}self_attn.", max_context=max_context,
                                     gate=c.attn_output_gate)
            self.blocks.append(DecoderLayer(
                i, RMSNorm(h, eps, f"{hp}input_layernorm.weight", prefix=f"{lp}input_norm."), mixer,
                RMSNorm(h, eps, f"{hp}post_attention_layernorm.weight", prefix=f"{lp}post_norm."),
                GatedMLP(h, c.intermediate_size, hf_prefix=f"{hp}mlp.", prefix=f"{lp}mlp.", chunk=pack_rows // 2),
                prefix=lp))
        self.norm = RMSNorm(h, eps, f"{p}norm.weight", prefix="norm.")
        if c.tie_word_embeddings:
            self.lm_head = LMHead(h, c.vocab_size, tied=self.embed_tokens, prefix="lm_head.")
        else:
            self.lm_head = LMHead(h, c.vocab_size, hf_name="lm_head.weight", prefix="lm_head.")
        self.sampler = GreedySampler(prefix="sampler.")
        self.tap_values: Dict[int, Value] = {}

    # ---- construction ---------------------------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, **options: Any) -> "Qwen3_5Model":
        model = cls(Qwen3_5Config.from_pretrained(path), **options)
        bind_checkpoint_formats(model, path)
        return model

    # ---- the Model contract ---------------------------------------------------------------------------------
    def layers(self) -> Sequence[Module]:
        return self.blocks

    def state_spec(self) -> StateSpec:
        entries: List[StateEntry] = []
        for blk in self.blocks:
            entries += blk.mixer.state_entries()
        return StateSpec(tuple(entries))

    def feature_taps(self) -> List[int]:
        return list(range(self.n_layers))

    def tables(self) -> Dict[str, Tuple[str, Any]]:
        if not any(not self.config.is_linear(i) for i in range(self.n_layers)):
            return {}
        cos, sin = rope_tables_permuted(self.config.rope_theta, self.config.head_dim, self.config.rotary_dim, self.max_context)
        return {"rope_cos": ("BF16", f32_to_bf16(cos)), "rope_sin": ("BF16", f32_to_bf16(sin))}

    # ---- oracle ---------------------------------------------------------------------------------------------
    def forward(self, ids: Any, state: Dict[str, Any], pos: int) -> Tuple[Any, List[Any], Dict[str, Any]]:
        """``ids [T]`` → ``(logits [T, V] BF16, residual streams [embeddings, after layer 0, …, after the last
        layer], state)``; the caches/states in ``state`` are advanced in place for the ``T`` tokens at ``pos``."""
        h = self.embed_tokens.forward(ids)
        hiddens = [h]
        for blk in self.blocks:
            h = blk.forward(h, state, pos)
            hiddens.append(h)
        logits = self.lm_head.forward(self.norm.forward(h))
        return logits, hiddens, state

    # ---- IR -------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, *xs: Value) -> Value:
        """Emit the whole step for ``T`` tokens: returns the sampled ``token [T]`` value."""
        tokens = xs[0] if xs else g.input("tokens", (T,), DType.I32)
        ctx = LowerContext(t=tokens.shape[0])
        for e in self.state_spec().entries:
            ctx.states[e.name] = g.state(e.name, state_shape(e), e.dtype)
        for name, (dtype, arr) in self.tables().items():
            ctx.consts[name] = g.const(name, tuple(int(x) for x in np.asarray(arr).shape), DType.parse(dtype.lower()))
        h = self.embed_tokens.lower(g, tokens, ctx)
        self.tap_values = {-1: h}
        for blk in self.blocks:
            h = blk.lower(g, h, ctx)
            self.tap_values[blk.index] = h
        logits = self.lm_head.lower(g, h, self.norm.lower(g, h), ctx)
        return self.sampler.lower(g, logits, ctx)
