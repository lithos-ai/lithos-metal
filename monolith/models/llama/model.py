"""Llama-compatible text decoder composed from the shared layer library."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ...core.dtypes import DType
from ...core.ir import Graph, Value
from ...core.shapes import T
from ...formats.fp import f32_to_bf16
from ...nn import DecoderLayer, Embedding, GatedMLP, GQAAttention, GreedySampler, LMHead, LowerContext, Model, Module, RMSNorm, StateEntry, StateSpec, state_shape
from .rope import tables
from ..registry import register_model
from .config import LlamaConfig
from .weights import PREFIX, bind_checkpoint_formats


class LlamaAttention(GQAAttention):
    def rotary_tables(self, positions):
        import torch
        cos, sin = tables(self.config, positions.cpu().numpy())
        return torch.from_numpy(cos), torch.from_numpy(sin)


@register_model("LlamaForCausalLM")
class LlamaModel(Model):
    def __init__(self, config: LlamaConfig, *, max_context: int = 4096, pack_rows: int = 16,
                 num_layers_override: int | None = None, prefix: str = "") -> None:
        """``prefix``: in front of every slab, aux entry, table, state and activation name — a tree that lives beside
        another in one program (the LM drafter of design §5.8 builds its model with ``prefix="draft."``)."""
        super().__init__(prefix=prefix)
        self.config = config
        if not 0 < max_context <= config.max_position_embeddings:
            raise ValueError("max_context must fit the checkpoint positional capacity")
        if num_layers_override is not None and not 0 < num_layers_override <= config.num_hidden_layers:
            raise ValueError("num_layers_override must select existing layers")
        self.max_context = max_context
        c = config
        h, eps, d = c.hidden_size, c.rms_norm_eps, c.head_dim
        self.n_layers = c.num_hidden_layers if num_layers_override is None else num_layers_override
        p, ap = PREFIX, prefix
        self.embed_tokens = Embedding(c.vocab_size, h, f"{p}embed_tokens.weight", prefix=f"{ap}embed_tokens.")
        self.blocks: List[DecoderLayer] = []
        for i in range(self.n_layers):
            lp, hp = f"{ap}layers.{i}.", f"{p}layers.{i}."
            mixer = LlamaAttention(h, c.num_attention_heads, c.num_key_value_heads, d, d, c.rope_theta, eps,
                                         hf_prefix=f"{hp}self_attn.", prefix=f"{lp}self_attn.", max_context=max_context,
                                         gate=False, norm_one_plus=False, qk_norm=False)
            mixer.config = config
            # Preserve the BF16 boundaries of the reference's separate operations.
            mixer.o_proj.round_residual = True
            self.blocks.append(DecoderLayer(
                i, RMSNorm(h, eps, f"{hp}input_layernorm.weight", prefix=f"{lp}input_norm.", one_plus=False, round_before_scale=True), mixer,
                RMSNorm(h, eps, f"{hp}post_attention_layernorm.weight", prefix=f"{lp}post_norm.", one_plus=False, round_before_scale=True),
                GatedMLP(h, c.intermediate_size, hf_prefix=f"{hp}mlp.", prefix=f"{lp}mlp.", chunk=pack_rows // 2), prefix=lp))
            self.blocks[-1].mlp.down.round_residual = True
            self.blocks[-1].mlp.gate_up.round_silu = True
        self.norm = RMSNorm(h, eps, f"{p}norm.weight", prefix=f"{ap}norm.", one_plus=False, round_before_scale=True)
        if c.tie_word_embeddings:
            self.lm_head = LMHead(h, c.vocab_size, tied=self.embed_tokens, prefix=f"{ap}lm_head.")
        else:
            self.lm_head = LMHead(h, c.vocab_size, hf_name="lm_head.weight", prefix=f"{ap}lm_head.")
        self.sampler = GreedySampler(prefix=f"{ap}sampler.")
        self.tap_values: Dict[int, Value] = {}

    @classmethod
    def from_checkpoint(cls, path: str, **options: Any) -> "LlamaModel":
        model = cls(LlamaConfig.from_pretrained(path), **options)
        bind_checkpoint_formats(model, path)
        # Keep the established fused convention for affine-quantized models;
        # unquantized BF16 checkpoints need eager-operation rounding boundaries.
        rounded = all(spec.format in ("bf16", "f32") for _, _, spec in model.full_weight_map().values())
        for block in model.blocks:
            block.input_norm.round_before_scale = block.post_norm.round_before_scale = rounded
            block.mixer.o_proj.round_residual = block.mlp.down.round_residual = rounded
            block.mlp.gate_up.round_silu = rounded
        model.norm.round_before_scale = rounded
        return model

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
        cos, sin = tables(self.config, np.arange(self.max_context))
        return {f"{self.prefix}rope_cos": ("BF16", f32_to_bf16(cos)), f"{self.prefix}rope_sin": ("BF16", f32_to_bf16(sin))}

    def forward(self, ids: Any, state: Dict[str, Any], pos: int) -> Tuple[Any, List[Any], Dict[str, Any]]:
        h = self.embed_tokens.forward(ids)
        hiddens = [h]
        for blk in self.blocks:
            h = blk.forward(h, state, pos)
            hiddens.append(h)
        return self.lm_head.forward(self.norm.forward(h)), hiddens, state

    def lower(self, g: Graph, *xs: Value) -> Value:
        tokens = xs[0] if xs else g.input("tokens", (T,), DType.I32)
        ctx = LowerContext(t=tokens.shape[0])
        for e in self.state_spec().entries:
            ctx.states[e.name] = g.state(e.name, state_shape(e), e.dtype)
        for name, (dtype, arr) in self.tables().items():                                  # the layers read the tables by their bare names
            ctx.consts[name[len(self.prefix):]] = g.const(name, tuple(int(x) for x in np.asarray(arr).shape), DType.parse(dtype.lower()))
        h = self.embed_tokens.lower(g, tokens, ctx)
        self.tap_values = {-1: h}
        for blk in self.blocks:
            h = blk.lower(g, h, ctx)
            self.tap_values[blk.index] = h
        logits = self.lm_head.lower(g, h, self.norm.lower(g, h), ctx)
        return self.sampler.lower(g, logits, ctx)
