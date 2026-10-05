"""The DSpark drafter as a :class:`Drafter` module (design §5.8, §5.14), built from the layer library plus the three
pieces only a DSpark drafter has: the feature projection, the block attention over the injected context, and the
Markov + confidence heads. Semantics follow DeepSpec's ``modeling/dspark/qwen3/modeling.py`` and
``eval/dspark/draft_ops.py`` (MIT; ``third_party/NOTICE``):

* **Context features**: the target's residual streams after the tapped layers, concatenated ``[n, taps · H_t]``,
  → ``fc`` → standard RMSNorm ``hidden_norm`` → ``[n, H]``. The drafter's per-layer KV caches hold the k/v of these
  features for every committed position (KV injection); each round appends the ``n_new`` positions the last verify
  committed.
* **Draft block**: the embeddings of ``[anchor, mask × (γ − 1)]`` at positions ``start … start + γ − 1`` go through
  the draft layers; every layer's attention has keys = the context cache (positions < start) ∪ the new context
  positions ∪ the block itself, with **no mask** (bidirectional inside the block), q/k RMSNorm and full RoPE as the
  target's; the block's own k/v are not kept.
* **Heads**: base logits use the checkpoint's frozen ``lm_head`` when present, otherwise the target's head,
  over the normed block hidden; the vanilla Markov head adds
  ``W₂ · W₁[prev_k]`` (``prev_0`` = the anchor, ``prev_{k+1}`` = the sampled draft) before the argmax; the confidence
  head is ``σ(w · [h_k ; W₁[prev_k]] + b)``. The reference verifies the confident prefix (confidence ≥ threshold);
  ``lower_select`` can use the profile's cost table instead (design §5.8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.dtypes import DType
from ...core.ir import BlockDomain, Graph, OpClass, Value
from ...core.shapes import N_INJ
from ...core.profile import Profile
from ...nn import DecoderLayer, Embedding, GatedMLP, Linear, LMHead, LowerContext, Module, Part, RMSNorm, StateEntry, state_shape
from ...nn.rope import rope_tables_permuted
from ...packs.transforms import rope_head_perm
from ..drafter import DraftBlock, DraftContext, Drafter
from ..registry import register_drafter
from .config import DSparkConfig


class DraftAttention(Module):
    """The drafter's attention: the target's GQA attention (standard q/k norm, full RoPE, no gate) whose keys come
    from three sources — its own context KV cache, the ``n_new`` context features and the block — and whose
    queries are the block, unmasked."""

    def __init__(self, cfg: DSparkConfig, *, hf_prefix: str, prefix: str, max_context: int, attention: Optional[str] = None) -> None:
        super().__init__(prefix=prefix)
        self.cfg, self.hf_prefix, self.max_context = cfg, hf_prefix, max_context
        self.attention = attention
        h, d, heads, kv = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        self.heads, self.kv_heads, self.head_dim = heads, kv, d
        self.qkv = Linear(h, [Part("q_proj", f"{hf_prefix}q_proj.weight", heads * d), Part("k_proj", f"{hf_prefix}k_proj.weight", kv * d),
                              Part("v_proj", f"{hf_prefix}v_proj.weight", kv * d)], prefix=f"{prefix}qkv.")
        self.kv_ctx = Linear(h, [Part("k_proj", f"{hf_prefix}k_proj.weight", kv * d), Part("v_proj", f"{hf_prefix}v_proj.weight", kv * d)],
                             prefix=f"{prefix}kv_ctx.")                   # the same k/v matrices applied to the context features
        self.o_proj = Linear(heads * d, [Part("o_proj", f"{hf_prefix}o_proj.weight", h)], prefix=f"{prefix}o_proj.", epilogue="residual")

    def weight_map(self):
        from ...nn.module import WeightSpec

        d = self.head_dim
        return {"q_norm": WeightSpec(f"{self.hf_prefix}q_norm.weight", (d,), "f32", transform="bf16_f32", aux=True, perm=rope_head_perm(d, d)),
                "k_norm": WeightSpec(f"{self.hf_prefix}k_norm.weight", (d,), "f32", transform="bf16_f32", aux=True, perm=rope_head_perm(d, d))}

    def full_weight_map(self):
        # k_proj / v_proj are claimed by both stacked projections (the same checkpoint tensors); the second slab
        # streams them again for the context features — Module.full_weight_map refuses duplicate claims, so the
        # drafter lists kv_ctx's parts under distinct keys and the loader routes the tensor to both
        out = {}
        for name, mod in self.named_modules():
            for local, spec in mod.weight_map().items():
                key = spec.hf_name if spec.hf_name not in out else f"{spec.hf_name}#{name}"
                out[key] = (mod, local, spec)
        return out

    def state_entries(self, checkpoints: int = 1) -> List[StateEntry]:
        shape = (self.max_context, self.kv_heads, self.head_dim)
        return [StateEntry(f"{self.prefix}k_ctx", shape, DType.BF16), StateEntry(f"{self.prefix}v_ctx", shape, DType.BF16)]

    # ---- oracle -------------------------------------------------------------------------------------------------
    def forward(self, x: Any, residual: Any, state: Dict[str, Any], pos: int, *, ctx_feats: Any, ctx_len: int) -> Any:
        """``x`` [γ, H] = the normed block; ``ctx_feats`` [n_new, H] = the new context features (positions
        ``ctx_len … ctx_len + n_new``); the block sits at positions ``pos … pos + γ`` with ``pos = ctx_len + n_new``.
        Appends the context k/v to the caches (they now cover ``[0, pos)``) and returns the residual + attention."""
        import torch

        from ...nn import oracle

        cfg, d, heads, kv = self.cfg, self.head_dim, self.heads, self.kv_heads
        g = x.shape[0]
        n_new = ctx_feats.shape[0]
        proj = self.qkv.forward(x)
        q = oracle.rms_norm(proj[:, : heads * d].reshape(g, heads, d), self.param("q_norm"), cfg.rms_norm_eps, one_plus=False)
        k_blk = proj[:, heads * d: heads * d + kv * d].reshape(g, kv, d)
        v_blk = proj[:, heads * d + kv * d:].reshape(g, kv, d)
        kvp = self.kv_ctx.forward(ctx_feats)
        k_new = kvp[:, : kv * d].reshape(n_new, kv, d)
        v_new = kvp[:, kv * d:].reshape(n_new, kv, d)
        k_all = oracle.rms_norm(torch.cat([k_new, k_blk]), self.param("k_norm"), cfg.rms_norm_eps, one_plus=False)
        from ...nn.rope import rope_tables_hf
        cos, sin = rope_tables_hf(cfg.rope_theta, d, ctx_len + n_new + g, parameters=cfg.rope_parameters)
        cos, sin = [torch.from_numpy(v[ctx_len:]).to(device=x.device, dtype=x.dtype) for v in (cos, sin)]
        k_all = oracle.apply_partial_rope(k_all, cos, sin)
        q = oracle.apply_partial_rope(q, cos[n_new:], sin[n_new:])
        kc, vc = state[f"{self.prefix}k_ctx"], state[f"{self.prefix}v_ctx"]
        kc[ctx_len: ctx_len + n_new] = k_all[:n_new]
        vc[ctx_len: ctx_len + n_new] = v_new
        keys = torch.cat([kc[: ctx_len + n_new], k_all[n_new:]]).to(torch.float32)         # context, then the block
        vals = torch.cat([vc[: ctx_len + n_new], v_blk]).to(torch.float32)
        rep = heads // kv
        ks = keys.repeat_interleave(rep, dim=1)
        vs = vals.repeat_interleave(rep, dim=1)
        scores = torch.einsum("ghd,nhd->hgn", q.to(torch.float32), ks).to(x.dtype) * (d ** -0.5)   # no mask: bidirectional
        p = torch.softmax(scores.to(torch.float32), dim=-1).to(x.dtype)
        o = torch.einsum("hgn,nhd->ghd", p.to(torch.float32), vs).to(x.dtype).reshape(g, heads * d)
        return self.o_proj.forward(o, residual)

    # ---- IR -----------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext, *, feats: Optional[Value] = None) -> Value:
        proj = self.qkv.lower(g, h, norm=norm).value
        kvp = self.kv_ctx.lower(g, ctx.consts["draft_ctx_feats"] if feats is None else feats).value
        kc, vc = ctx.states[f"{self.prefix}k_ctx"], ctx.states[f"{self.prefix}v_ctx"]
        cos, sin = ctx.consts["draft_rope_cos"], ctx.consts["draft_rope_sin"]
        qn = self.const_value(g, f"{self.prefix}q_norm", (self.head_dim,), DType.F32)
        kn = self.const_value(g, f"{self.prefix}k_norm", (self.head_dim,), DType.F32)
        o = g.value(f"{self.prefix}attn", (h.shape[0], self.heads * self.head_dim), DType.BF16)
        g.op("draft_attn", [proj, kvp, kc, vc, cos, sin, qn, kn], [o], domain=BlockDomain("heads", self.heads), klass=OpClass.MAP,
             updates=[kc.name, vc.name], heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim, eps=self.cfg.rms_norm_eps,
             scaling=self.head_dim ** -0.5, gamma=self.cfg.block_size, attention=self.attention)
        return self.o_proj.lower(g, o, residual=h, name=f"{self.prefix}h").value


@register_drafter("dspark")
class DSparkDrafter(Drafter):
    """Use frozen checkpoint embeddings/head when present, otherwise share the target's."""

    def __init__(self, cfg: DSparkConfig, *, target_lm_head: Optional[LMHead], max_context: int = 4096, pack_rows: int = 16,
                 confidence_threshold: float = 0.0, sts: Optional[Sequence[float]] = None,
                 shared_embedding: bool = False, kernel_config: Optional[Dict[str, Any]] = None,
                 attention: Optional[str] = None, checkpoint_lm_head: bool = False,
                 block_size: Optional[int] = None) -> None:
        """``target_lm_head``: fallback head when the draft checkpoint does not supply one (None when only packing).
        ``confidence_threshold``: the confident-prefix rule's threshold (≤ 0 verifies the whole block, the
        reference's default) used when the caller of ``lower_select`` has no cost table. ``sts``: per-position
        temperatures dividing the confidence logits (the calibration of the confidence chain, design §5.8;
        ``tools/bench/sts_calibrate.py`` fits them against measured acceptance)."""
        super().__init__(prefix="draft.")
        if block_size is not None:
            from dataclasses import replace
            if isinstance(block_size, bool) or not isinstance(block_size, int) or not 1 <= block_size <= cfg.block_size:
                raise ValueError(f"DSpark: block_size must be in 1..{cfg.block_size}")
            cfg = replace(cfg, block_size=block_size)
        self.cfg, self.gamma, self.max_context = cfg, cfg.block_size, max_context
        self.kernel_config = kernel_config
        if attention not in (None, 'v1', 'mma', 'auto'):
            raise ValueError('draft attention must be v1, mma or auto')
        self.confidence_threshold = float(confidence_threshold)
        self.sts = None if sts is None else [float(x) for x in sts]
        if self.sts is not None and (len(self.sts) != self.gamma or any(x <= 0 for x in self.sts)):
            raise ValueError(f"DSparkDrafter: sts needs {self.gamma} positive temperatures")
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.embed_tokens = None if shared_embedding else Embedding(cfg.vocab_size, h, "embed_tokens.weight", prefix="draft.embed_tokens.")
        self._target_embedding = None
        self.fc = Linear(cfg.n_taps * cfg.target_hidden, [Part("fc", "fc.weight", h)], prefix="draft.fc.")
        self.hidden_norm = RMSNorm(h, eps, "hidden_norm.weight", prefix="draft.hidden_norm.", one_plus=False)
        self.blocks: List[DecoderLayer] = []
        for i in range(cfg.num_hidden_layers):
            lp, hp = f"draft.layers.{i}.", f"layers.{i}."
            attn = DraftAttention(cfg, hf_prefix=f"{hp}self_attn.", prefix=f"{lp}self_attn.", max_context=max_context, attention=attention)
            self.blocks.append(DecoderLayer(i, RMSNorm(h, eps, f"{hp}input_layernorm.weight", prefix=f"{lp}input_norm.", one_plus=False), attn,
                                            RMSNorm(h, eps, f"{hp}post_attention_layernorm.weight", prefix=f"{lp}post_norm.", one_plus=False),
                                            GatedMLP(h, cfg.intermediate_size, hf_prefix=f"{hp}mlp.", prefix=f"{lp}mlp.", chunk=pack_rows // 2), prefix=lp))
        self.norm = RMSNorm(h, eps, "norm.weight", prefix="draft.norm.", one_plus=False)
        # A checkpoint may freeze a different target quantization's head. Keep
        # those proposal weights instead of silently substituting the verifier's.
        self.lm_head = LMHead(h, cfg.vocab_size, hf_name="lm_head.weight", prefix="draft.lm_head.") if checkpoint_lm_head else None
        self._lm_head = self.lm_head if self.lm_head is not None else target_lm_head
        self.markov_w1 = Embedding(cfg.vocab_size, cfg.markov_rank, "markov_head.markov_w1.weight", prefix="draft.markov_w1.")
        # the Markov bias is a separate BF16 linear added to the block's logits in BF16: the residual epilogue with
        # the product rounded first reproduces both roundings
        self.markov_w2 = Linear(cfg.markov_rank, [Part("w2", "markov_head.markov_w2.weight", cfg.vocab_size)], prefix="draft.markov_w2.",
                                epilogue="residual", round_residual=True)

    def optimize_program(self, program, *, prefill=False):
        if self.kernel_config is None or prefill:
            return program
        from .optimization import optimize
        return optimize(program, self.kernel_config)[1]

    @classmethod
    def from_checkpoint(cls, path: str, *, target_lm_head: Optional[LMHead], max_context: int = 4096, **options: Any) -> "DSparkDrafter":
        from .weights import bind_checkpoint_formats
        from ...formats.safetensors_reader import SafetensorsDir

        ckpt = SafetensorsDir(path)
        try:
            options.setdefault("shared_embedding", "embed_tokens.weight" not in ckpt.names())
            options.setdefault("checkpoint_lm_head", "lm_head.weight" in ckpt.names())
        finally:
            ckpt.close()
        drafter = cls(DSparkConfig.from_pretrained(path), target_lm_head=target_lm_head, max_context=max_context, **options)
        bind_checkpoint_formats(drafter, path)
        return drafter

    def bind_target(self, model: Any) -> None:
        if self.embed_tokens is None:
            embedding = getattr(model, "embed_tokens", None)
            if not isinstance(embedding, Embedding):
                raise ValueError("DSpark: this checkpoint needs the target's embedding module")
            spec = embedding.weight_map()["weight"]
            if spec.shape != (self.cfg.vocab_size, self.cfg.hidden_size):
                raise ValueError("DSpark: shared target embedding shape does not match the draft")
            self._target_embedding = embedding

    def embedding(self) -> Embedding:
        emb = self.embed_tokens if self.embed_tokens is not None else self._target_embedding
        if emb is None:
            raise ValueError("DSpark: call bind_target before using a checkpoint with shared embeddings")
        return emb

    def tap_layers(self) -> List[int]:
        return list(self.cfg.target_layer_ids)

    def weight_map(self):
        from ...nn.module import WeightSpec

        cfg = self.cfg
        if not cfg.enable_confidence_head:
            return {}
        n_in = cfg.hidden_size + (cfg.markov_rank if cfg.confidence_head_with_markov else 0)
        return {"conf_w": WeightSpec("confidence_head.proj.weight", (1, n_in), "f32", transform="f32", aux=True),
                "conf_b": WeightSpec("confidence_head.proj.bias", (1,), "f32", transform="f32", aux=True)}

    def full_weight_map(self):
        out = {}
        for name, mod in self.named_modules():
            for local, spec in mod.weight_map().items():
                key = spec.hf_name if spec.hf_name not in out else f"{spec.hf_name}#{name}"
                out[key] = (mod, local, spec)
        return out

    def load_weights(self, weights, *, strict: bool = True):
        """Route each checkpoint tensor to every module that claims it (k/v projections are claimed twice)."""
        table: Dict[str, list] = {}
        for key, hit in self.full_weight_map().items():
            table.setdefault(key.split("#")[0], []).append(hit)
        consumed = set()
        for name, tensor in weights:
            hits = table.get(name)
            if hits is None:
                if strict:
                    raise KeyError(f"DSparkDrafter.load_weights: unexpected checkpoint key {name!r}")
                continue
            for mod, local, _ in hits:
                mod.set_param(local, tensor)
            consumed.add(name)
        missing = sorted(set(table) - consumed)
        if missing:
            raise ValueError(f"DSparkDrafter.load_weights: weights never loaded: {missing[:8]}")
        self.process_weights()
        return consumed

    def state_entries(self) -> List[StateEntry]:
        out: List[StateEntry] = []
        for blk in self.blocks:
            out += blk.mixer.state_entries()
        return out

    def tables(self) -> Dict[str, Tuple[str, Any]]:
        from ...formats.fp import f32_to_bf16

        d = self.cfg.head_dim
        cos, sin = rope_tables_permuted(self.cfg.rope_theta, d, d, self.max_context, parameters=self.cfg.rope_parameters)
        return {"draft_rope_cos": ("BF16", f32_to_bf16(cos)), "draft_rope_sin": ("BF16", f32_to_bf16(sin))}

    # ---- oracle -------------------------------------------------------------------------------------------------
    def project_features(self, taps: Any) -> Any:
        """``taps`` [n, taps · H_t] (the tapped residual streams, concatenated in tap order) → context features [n, H]."""
        return self.hidden_norm.forward(self.fc.forward(taps))

    def draft_block(self, anchor: int, ctx_feats: Any, state: Dict[str, Any], ctx_len: int) -> Any:
        """One draft pass: returns the normed block hidden ``[γ, H]``; the context caches grow by ``ctx_feats``."""
        import torch

        ids = torch.full((self.gamma,), self.cfg.mask_token_id, dtype=torch.int64)
        ids[0] = anchor
        h = self.embedding().forward(ids)
        pos = ctx_len + ctx_feats.shape[0]
        for blk in self.blocks:
            x = blk.input_norm.forward(h)
            h = blk.mixer.forward(x, h, state, pos, ctx_feats=ctx_feats, ctx_len=ctx_len)
            h = blk.mlp.forward(blk.post_norm.forward(h), h)
        return self.norm.forward(h)

    def markov_bias(self, prev: int) -> Any:
        """``W₂ · W₁[prev]`` in BF16 like the reference (an embedding row through a BF16 linear)."""
        import torch

        w1 = self.markov_w1.param("weight")[prev]
        return (w1.to(torch.float32) @ self.markov_w2.param("w2").to(torch.float32).t()).to(w1.dtype)

    def draft_tokens(self, block_hidden: Any, anchor: int) -> Tuple[List[int], Any]:
        """Greedy: the Markov chain over the block's base logits; returns the γ drafts and the corrected logits."""
        import torch

        base = self._lm_head.forward(block_hidden)                    # [γ, V] BF16
        prev, out, corrected = anchor, [], []
        for k in range(self.gamma):
            lg = base[k] + self.markov_bias(prev)
            corrected.append(lg)
            prev = int(torch.argmax(lg.to(torch.float32)))
            out.append(prev)
        return out, torch.stack(corrected)

    def confidences(self, block_hidden: Any, prev_tokens: Sequence[int]) -> Any:
        """``σ(w · [h_k ; W₁[prev_k]] + b)`` per block position (FP32 logits like the reference's ``.float()``)."""
        import torch

        if not self.cfg.enable_confidence_head:
            return None
        feats = block_hidden
        if self.cfg.confidence_head_with_markov:
            emb = self.markov_w1.param("weight")[torch.tensor(list(prev_tokens))].to(block_hidden.dtype)
            feats = torch.cat([block_hidden, emb], dim=-1)
        w, b = self.param("conf_w").to(block_hidden.dtype), self.param("conf_b").to(block_hidden.dtype)
        logits = (feats.to(torch.float32) @ w.to(torch.float32).t()).to(block_hidden.dtype) + b   # BF16 linear, then float
        return torch.sigmoid(logits.to(torch.float32).squeeze(-1))

    @staticmethod
    def confident_prefix(conf: Any, threshold: float, gamma: int) -> int:
        if threshold <= 0.0:
            return gamma
        below = (conf < threshold).nonzero()
        return gamma if below.numel() == 0 else int(below[0])

    # ---- the Drafter contract (IR) ----------------------------------------------------------------------------
    def _lower_context(self, g: Graph) -> LowerContext:
        """The drafter's states (its context caches) and constants (its RoPE tables) in ``g``, created on first use."""
        import numpy as np

        lc = LowerContext(t=self.gamma)
        for e in self.state_entries():
            lc.states[e.name] = g.values[e.name] if e.name in g.values else g.state(e.name, state_shape(e), e.dtype)
        for name, (dtype, arr) in self.tables().items():
            shape = tuple(int(x) for x in np.asarray(arr).shape)
            lc.consts[name] = g.values[name] if name in g.values else g.const(name, shape, DType.parse(dtype.lower()))
        return lc

    @staticmethod
    def _normalized(g: Graph, norm: RMSNorm, h: Value, name: str) -> Value:
        """The norm as a value of its own (``rmsnorm_stat`` + ``norm_apply``): the normalized activation several
        consumers read (the context features feed every layer's k/v projection; the block hidden feeds the target's
        head and the confidence head)."""
        ni = norm.lower(g, h)
        x = g.value(name, h.shape, DType.BF16)
        g.op("norm_apply", [h, ni.stat, ni.weight], [x], domain=BlockDomain("span", norm.dim), klass=OpClass.MAP, eps=ni.eps)
        return x

    def lower_draft(self, g: Graph, ctx: DraftContext, anchor: Optional[Value] = None) -> DraftBlock:
        """The draft pass (design §5.8): the committed positions' tapped features (``N_INJ`` rows) → the feature
        projection → the context features; the block ``[anchor, mask × (γ − 1)]`` through the drafter's layers, whose
        attention injects the context features and attends over the whole context; the target's head on the normed
        block; the Markov chain (per position: the previous token's embedding, the bias added to the base logits,
        argmax — chained through row views); the confidence head."""
        cfg, gamma, h_t = self.cfg, self.gamma, self.cfg.target_hidden
        if self._lm_head is None:
            raise ValueError("DSparkDrafter: lowering needs the target's lm_head (built with target_lm_head=None)")
        taps = list(ctx.taps)
        if len(taps) != cfg.n_taps:
            raise ValueError(f"DSparkDrafter: {cfg.n_taps} taps expected, got {len(taps)}")
        if anchor is None:
            anchor = ctx.anchor if ctx.anchor is not None else g.input("anchor", (1,), DType.I32)
        lc = self._lower_context(g)
        # 1. the context features of the injected rows
        x = g.value("draft.taps", (N_INJ, cfg.n_taps * h_t), DType.BF16)
        g.op("tap_concat", taps, [x], domain=BlockDomain("rows", cfg.n_taps), klass=OpClass.MAP, width=h_t)
        fcy = self.fc.lower(g, x, name="draft.fc.y").value
        feats = self._normalized(g, self.hidden_norm, fcy, "draft.feats")
        lc.consts["draft_ctx_feats"] = feats
        # 2. the block through the drafter's layers
        emb = self.embedding()
        w_emb = emb.weight_value(g, emb.slab_name, (cfg.vocab_size, cfg.hidden_size), emb.format_of("weight"))
        h = g.value("draft.h0", (gamma, cfg.hidden_size), DType.BF16)
        g.op("embed", [anchor, w_emb], [h], domain=BlockDomain("rows", gamma), klass=OpClass.MAP, packed=True, ids="block",
             mask_id=cfg.mask_token_id)
        for blk in self.blocks:
            h = blk.lower(g, h, lc)
        hidden = self._normalized(g, self.norm, h, "draft.hidden")
        base = self._lm_head.lower(g, hidden, None, lc, name="draft.base_logits")
        # 3. the Markov chain: d_k = argmax(base_k + W₂·W₁[d_{k−1}]), d_{−1} = the anchor
        drafts = g.value("draft.tokens", (gamma,), DType.I32)
        markov = g.value("draft.markov.emb", (gamma, cfg.markov_rank), DType.BF16)
        w1 = self.markov_w1.weight_value(g, self.markov_w1.slab_name, (cfg.vocab_size, cfg.markov_rank), self.markov_w1.format_of("weight"))
        prev = anchor
        for k in range(gamma):
            e_k = g.view(f"draft.markov.emb.{k}", markov, k, 1)
            g.op("embed", [prev, w1], [e_k], domain=BlockDomain("rows", 1), klass=OpClass.MAP, packed=True)
            lg = self.markov_w2.lower(g, e_k, residual=g.view(f"draft.base_logits.{k}", base, k, 1), name=f"draft.markov.{k}.logits").value
            d_k = g.view(f"draft.tokens.{k}", drafts, k, 1)
            g.op("argmax", [lg], [d_k], domain=BlockDomain("span", cfg.vocab_size), klass=OpClass.REDUCE)
            prev = d_k
        # 4. the confidence head over [h_k ; W₁[prev_k]]
        conf = None
        if cfg.enable_confidence_head:
            rank = cfg.markov_rank if cfg.confidence_head_with_markov else 0
            w = self.const_value(g, "draft.conf_w", (1, cfg.hidden_size + rank), DType.F32)
            b = self.const_value(g, "draft.conf_b", (1,), DType.F32)
            conf = g.value("draft.confidence", (gamma,), DType.F32)
            attrs: Dict[str, Any] = dict(rank=rank)
            if self.sts is not None:
                attrs["sts"] = list(self.sts)
            g.op("confidence", [hidden, markov, w, b], [conf], domain=BlockDomain("rows", gamma), klass=OpClass.MAP, **attrs)
        return DraftBlock(tokens=drafts, confidences=conf, hidden=hidden, gamma=gamma)

    def lower_select(self, g: Graph, block: DraftBlock, profile: Profile, *, cost: Optional[Sequence[float]] = None,
                     threshold: Optional[float] = None, fixed: Optional[int] = None) -> Value:
        """With ``fixed`` always that many drafts; with ``cost`` (the profile's relative cost of a (1 + l)-token target
        pass, l = 0 … γ) the cost-aware rule of design §5.8; otherwise the confident-prefix rule with ``threshold``
        (the drafter's default when None; the whole block without a confidence head or with a threshold ≤ 0)."""
        sel = g.value("draft.verify_len", (1,), DType.U32)
        ins = [block.tokens] + ([block.confidences] if block.confidences is not None else [])
        thr = self.confidence_threshold if threshold is None else float(threshold)
        attrs: Dict[str, Any] = dict(gamma=block.gamma, threshold=thr if block.confidences is not None else 0.0)
        if fixed is not None:
            attrs["fixed"] = int(fixed)
        elif cost is not None and block.confidences is not None:
            attrs["cost"] = [float(c) for c in cost]
        g.op("verify_select", ins, [sel], domain=BlockDomain("span", 1), klass=OpClass.SERIAL, **attrs)
        return sel

    def lower_context_update(self, g: Graph, taps: List[Value], accepted: Value) -> None:
        """A no-op: the DSpark drafter injects the committed positions' features in its next draft pass (the block
        attention appends them to its context caches), so nothing runs between the accept scan and that pass."""
        return None
