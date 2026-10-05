"""An LM drafter as a :class:`Drafter` module (design §5.8, §5.14).

The draft model is a registered model package (``monolith.models``) built with ``prefix="draft."`` so its slabs, aux
entries, tables, states and activations live beside the target's in one program. Per round the drafter runs

* a **first chain step** over the committed positions it has not processed (``StepState.n_inject`` rows, the last
  ones of the step's ``pending_tokens``: the whole chunk in prefill, in decode one row — the last draft — when every
  draft was accepted, none otherwise) followed by the anchor (``n_chain`` rows: 1, or 0 in a prefill chunk that is
  not the last), ``n_inject + n_chain ≤ T_max + 1`` rows from ``position − n_inject``; the argmax of the last row is
  the first draft;
* then ``gamma − 1`` single-row **chain steps**: each draft at the next position, through the layers and the head to
  the argmax, the k/v appended as it goes (0 rows in a prefill chunk that is not the last: the dispatches return
  at once).

The drafter's caches then hold positions ``< position + gamma``; the rows past the accepted prefix are stale and are
overwritten by the next chain, whose rows only attend to keys at or before their own position. ``drafter_ctx_len``
records that length (``verify_select``), and the accept scan's ``n_inject`` is the rows between it and the new
position. No confidences: the whole chain is verified (or ``fixed`` drafts).

The package's tree must expose ``embed_tokens``, ``layers()``, ``norm``, ``lm_head`` and ``tables()``, take
``prefix`` in ``from_checkpoint`` and use attention mixers only (the GDN kernels have no ingest/chain modes).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.dtypes import DType
from ...core.ir import BlockDomain, Graph, OpClass, Value
from ...core.profile import Profile
from ...core.shapes import N_CHAIN, N_FIRST
from ...models import resolve_model
from ...nn import LMHead, LowerContext, Model, StateEntry, state_shape
from ..drafter import DraftBlock, DraftContext, Drafter
from ..registry import register_drafter

PREFIX = "draft."


@register_drafter("lm")
class LMDrafter(Drafter):
    lm_drafter = True           # the round's serial ops keep the LM bookkeeping (emit.lower_round, kernels/common/spec_ops.metal)

    def __init__(self, model: Model, *, gamma: int = 5, target_lm_head: Optional[LMHead] = None) -> None:
        """``model``: the draft model's tree, built with ``prefix=PREFIX``; ``gamma``: the drafts per round;
        ``target_lm_head``: the target's head (only its vocabulary is checked — the drafter has its own head)."""
        super().__init__(prefix=PREFIX)
        if getattr(model, "prefix", "") != PREFIX:
            raise ValueError(f"LMDrafter: the draft model must be built with prefix={PREFIX!r} (got {getattr(model, 'prefix', '')!r})")
        for attr in ("embed_tokens", "norm", "lm_head"):
            if not hasattr(model, attr):
                raise ValueError(f"LMDrafter: the model package exposes no {attr!r}")
        if not 1 <= int(gamma) <= 15:
            raise ValueError(f"LMDrafter: gamma must be 1..15 (got {gamma})")
        self.model = model
        self.cfg = model.config
        self.gamma, self.max_context = int(gamma), model.max_context
        if target_lm_head is not None and self.cfg.vocab_size > target_lm_head.vocab:
            raise ValueError(f"LMDrafter: the draft model's vocabulary ({self.cfg.vocab_size}) exceeds the target's ({target_lm_head.vocab}); "
                             "the two must share a tokenizer")

    @classmethod
    def from_checkpoint(cls, path: str, *, target_lm_head: Optional[LMHead] = None, max_context: int = 4096, **options: Any) -> "LMDrafter":
        """``options``: ``gamma`` (the drafts per round, default 5) and the model package's own knobs."""
        with open(os.path.join(path, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        mcls = resolve_model(arch)
        if mcls is None:
            raise RuntimeError(f"LMDrafter: no model package registered for {arch!r}")
        gamma = int(options.pop("gamma", 5))
        model = mcls.from_checkpoint(path, max_context=max_context, prefix=PREFIX, **options)
        return cls(model, gamma=gamma, target_lm_head=target_lm_head)

    def tap_layers(self) -> List[int]:
        return []                                    # the drafter reads tokens, not the target's residual streams

    def state_entries(self) -> List[StateEntry]:
        return list(self.model.state_spec().entries)

    def tables(self) -> Dict[str, Tuple[str, Any]]:
        return self.model.tables()

    # ---- oracle -------------------------------------------------------------------------------------------------
    def forward(self, ids: Any, state: Dict[str, Any], pos: int) -> Any:
        """The draft model's logits for ``ids`` at ``pos …`` (its caches advanced): what one chain step computes."""
        return self.model.forward(ids, state, pos)[0]

    def draft_tokens(self, anchor: int, state: Dict[str, Any], pos: int) -> List[int]:
        """The chain's greedy drafts from ``anchor`` at ``pos`` (the caches then hold ``pos + gamma`` positions)."""
        import torch

        out: List[int] = []
        tok = anchor
        for k in range(self.gamma):
            logits = self.forward(torch.tensor([tok], dtype=torch.long), state, pos + k)
            tok = int(torch.argmax(logits[-1].float()))
            out.append(tok)
        return out

    # ---- IR ---------------------------------------------------------------------------------------------------
    def _lower_context(self, g: Graph) -> LowerContext:
        """The draft model's states and tables in ``g`` (created on first use); the layers read the tables by their
        bare names (``rope_cos``), the pack stores them under the model's prefix."""
        lc = LowerContext()
        for e in self.state_entries():
            lc.states[e.name] = g.values[e.name] if e.name in g.values else g.state(e.name, state_shape(e), e.dtype)
        for name, (dtype, arr) in self.tables().items():
            shape = tuple(int(x) for x in np.asarray(arr).shape)
            lc.consts[name[len(PREFIX):]] = g.values[name] if name in g.values else g.const(name, shape, DType.parse(dtype.lower()))
        return lc

    def lower_draft(self, g: Graph, ctx: DraftContext, anchor: Optional[Value] = None) -> DraftBlock:
        m, hidden, vocab = self.model, self.cfg.hidden_size, self.cfg.vocab_size
        if ctx.tokens is None:
            raise ValueError("LMDrafter: the round must hand the step's token rows over (DraftContext.tokens)")
        if anchor is None:
            anchor = ctx.anchor if ctx.anchor is not None else g.input("anchor", (1,), DType.I32)
        lc = self._lower_context(g)
        emb = m.embed_tokens
        w_emb = emb.weight_value(g, emb.slab_name, (vocab, hidden), emb.format_of("weight"))
        drafts = g.value("draft.tokens", (self.gamma,), DType.I32)
        prev = anchor
        for k in range(self.gamma):
            d_k = g.view(f"draft.tokens.{k}", drafts, k, 1)
            with g.scope(f"draft.chain{k}."):
                if k == 0:
                    # the committed rows the drafter has not seen, then the anchor: one pass from position − n_inject
                    lc_k = replace(lc, t=N_FIRST, mixer_attrs={"lm_mode": 3})
                    h = g.value("h0", (N_FIRST, hidden), DType.BF16)
                    g.op("embed", [ctx.tokens, w_emb], [h], domain=BlockDomain("rows", N_FIRST), klass=OpClass.MAP, packed=True, ids="ingest_anchor")
                else:
                    lc_k = replace(lc, t=N_CHAIN, mixer_attrs={"lm_mode": 2, "chain_i": k})
                    h = g.value("h0", (N_CHAIN, hidden), DType.BF16)
                    g.op("embed", [prev, w_emb], [h], domain=BlockDomain("rows", N_CHAIN), klass=OpClass.MAP, packed=True)
                for blk in m.layers():
                    h = blk.lower(g, h, lc_k)
                logits = m.lm_head.lower(g, h, m.norm.lower(g, h), lc_k, name="logits")
                g.op("argmax", [logits], [d_k], domain=BlockDomain("span", vocab), klass=OpClass.REDUCE, **({"last": True} if k == 0 else {}))
            prev = d_k
        return DraftBlock(tokens=drafts, confidences=None, hidden=None, gamma=self.gamma)

    def lower_select(self, g: Graph, block: DraftBlock, profile: Profile, *, cost: Optional[Sequence[float]] = None,
                     threshold: Optional[float] = None, fixed: Optional[int] = None) -> Value:
        """The whole chain is verified (no confidences to choose by), or ``fixed`` drafts; the select also records the
        drafter's context length (``lm``)."""
        sel = g.value("draft.verify_len", (1,), DType.U32)
        attrs: Dict[str, Any] = dict(gamma=block.gamma, threshold=0.0, lm=True)
        if fixed is not None:
            attrs["fixed"] = int(fixed)
        g.op("verify_select", [block.tokens], [sel], domain=BlockDomain("span", 1), klass=OpClass.SERIAL, **attrs)
        return sel

    def lower_context_update(self, g: Graph, taps: List[Value], accepted: Value) -> None:
        return None                                  # the next round's ingest pass consumes the committed rows
