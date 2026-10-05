"""The DSpark round lowered and emitted without torch or a GPU (design §5.8, #24): the draft pass, the Markov chain
through row views, the confidence head and the verify select on a synthetic drafter, compiled into a dynamic-T
program whose kernels read their row counts from the right StepState fields."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_synth import CFG, build, write_checkpoint  # noqa: E402

from monolith.compiler import CoverageError, check_coverage, emit_program  # noqa: E402
from monolith.compiler.passes import fuse_norm_stat  # noqa: E402
from monolith.core import BlockDomain, DType, Graph, N_INJ, OpClass, Profile, T  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime.program import Program  # noqa: E402
from monolith.spec import DraftContext  # noqa: E402

PROF = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})


def _lower(drafter, cfg, g=None):
    g = g or Graph("draft")
    taps = [g.input(f"tap.{i}", (T, cfg.target_hidden), DType.BF16) for i in range(cfg.n_taps)]
    anchor = g.input("anchor", (1,), DType.I32)
    block = drafter.lower_draft(g, DraftContext(taps, anchor))
    sel = drafter.lower_select(g, block, PROF)
    g.check()
    return g, block, sel


def test_views_and_row_coverage():
    g = Graph("v")
    base = g.value("b", (3, 4), DType.BF16)
    a = g.input("a", (1, 4), DType.BF16)
    v0, v1 = g.view("b.0", base, 0), g.view("b.1", base, 1)
    assert v0.is_view and v0.shape == (1, 4) and v1.view_of == ("b", 1) and g.views["b"] == [v0, v1]
    g.op("norm_apply", [a], [v0], domain=BlockDomain("span", 4), klass=OpClass.MAP)
    g.op("norm_apply", [v0], [v1], domain=BlockDomain("span", 4), klass=OpClass.MAP)        # reads a row written before it
    g.check()
    g.op("norm_apply", [base], [g.value("c", (3, 4), DType.BF16)], domain=BlockDomain("span", 4), klass=OpClass.MAP)
    with pytest.raises(ValueError, match="rows \\[2\\]"):
        g.check()                                                                            # row 2 never written
    with pytest.raises(ValueError):
        g.view("bad", base, 2, 2)                                                            # outside the base
    with pytest.raises(ValueError):
        g.view("bad2", v0, 0, 1)                                                             # a view of a view
    with pytest.raises(ValueError):
        g.view("bad3", g.value("sym", (T, 4), DType.BF16), 0, 1)                             # symbolic rows


def test_round_lowers_to_the_registered_kinds(tmp_path):
    write_checkpoint(tmp_path)
    drafter, head, cfg, _ = build(tmp_path)
    g, block, sel = _lower(drafter, cfg)
    from collections import Counter

    kinds = Counter(op.kind for op in g.ops)
    gamma, layers = cfg.block_size, cfg.num_hidden_layers
    # fc + per layer (qkv, kv_ctx, o_proj, gate_up, down) + the Markov bias per position
    assert kinds == {"tap_concat": 1, "gemv": 1 + 5 * layers + gamma, "norm_apply": 2, "rmsnorm_stat": 1 + 2 * layers + 1,
                     "embed": 1 + gamma, "draft_attn": layers, "lm_head": 1, "argmax": gamma, "confidence": 1, "verify_select": 1}
    check_coverage(g, PROF)
    # row symbols: the injected rows through fc / kv_ctx, the block static, the Markov chain single rows through views
    assert g.values["draft.taps"].shape[0] is N_INJ and g.values["draft.feats"].shape[0] is N_INJ
    kv_ctx = [op for op in g.ops if op.kind == "gemv" and op.inputs[1].name.endswith("kv_ctx.k_proj+v_proj")]
    assert len(kv_ctx) == layers and all(op.outputs[0].shape[0] is N_INJ for op in kv_ctx)
    assert g.values["draft.h0"].shape == (gamma, cfg.hidden_size) and block.hidden.shape == (gamma, cfg.hidden_size)
    assert block.tokens.shape == (gamma,) and block.confidences.shape == (gamma,) and block.gamma == gamma
    chain = [op for op in g.ops if op.kind == "argmax"]
    assert all(op.outputs[0].view_of == ("draft.tokens", k) for k, op in enumerate(chain))
    markov = [op for op in g.ops if op.kind == "gemv" and op.attrs.get("round_residual")]
    assert len(markov) == gamma and [op.inputs[2].view_of for op in markov] == [("draft.base_logits", k) for k in range(gamma)]
    embeds = [op for op in g.ops if op.kind == "embed"]
    assert embeds[0].attrs == {"packed": True, "ids": "block", "mask_id": cfg.mask_token_id} and embeds[0].inputs[0].name == "anchor"
    assert embeds[1].inputs[0].name == "anchor" and embeds[2].inputs[0].view_of == ("draft.tokens", 0)
    attn = [op for op in g.ops if op.kind == "draft_attn"]
    assert attn[0].attrs["updates"] == ["draft.layers.0.self_attn.k_ctx", "draft.layers.0.self_attn.v_ctx"] and attn[0].inputs[1].shape[0] is N_INJ
    assert sel.producer.attrs == {"gamma": gamma, "threshold": 0.0}
    assert drafter.lower_context_update(g, [], sel) is None
    # the fuse pass hoists every statistic fed by a residual GEMV (the drafter's post-attention and post-MLP norms and the final norm)
    assert fuse_norm_stat(g) == 2 * layers                    # all but layer 0's input norm (the embedding) and hidden_norm (fc has no residual)
    with pytest.raises(ValueError):
        drafter.lower_draft(Graph("bad"), DraftContext([], None))            # wrong tap count


def test_emit_dynamic_program(tmp_path):
    write_checkpoint(tmp_path)
    drafter, head, cfg, pair = build(tmp_path)
    pack_model(pair, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    pf = PackFile(tmp_path / "pack")
    g, block, sel = _lower(drafter, cfg)
    fuse_norm_stat(g)
    prog = emit_program(g, pack=pf, profile=PROF, dynamic_t=True, tail=None)
    gamma = cfg.block_size
    names = [o.name.split(":")[0] for o in prog.ops]
    assert names[0] == "tap_concat" and names[-1] == "verify_select" and names.count("gqa_decode") == 0
    assert names.count("draft_attn") == 2 and names.count("gqa_merge") == 2 and names.count("confidence") == 1
    assert names.count("argmax") == gamma and names.count("embed") == 1 + gamma
    lay = prog.layout
    kern = {o.name: prog.kernels[o.kernel] for o in prog.ops}
    ops = {o.name: o for o in prog.ops}
    # row sources: the feature projection follows n_inject, the block is static γ, the Markov GEMVs are single rows with two roundings
    fc = kern["gemv:draft.fc.fc"]
    assert fc.macros["T_SRC"] == "1" and fc.macros["T"] == str(lay.t_max) and fc.macros["STEP_STATE"] == "1"
    assert kern["tap_concat"].macros["T_SRC"] == "1" and kern["tap_concat"].macros["N_SRC"] == "2"
    qkv = kern["gemv:draft.layers.0.self_attn.qkv.q_proj+k_proj+v_proj"]
    assert qkv.macros["T_SRC"] == "2" and qkv.macros["T"] == str(gamma)
    kvc = kern["gemv:draft.layers.0.self_attn.kv_ctx.k_proj+v_proj"]
    assert kvc.macros["T_SRC"] == "1" and kvc.macros["T"] == str(lay.t_max)
    mk = kern["gemv:draft.markov_w2.w2"]
    assert mk.macros["T"] == "1" and mk.macros["T_SRC"] == "2" and mk.macros["EPILOGUE"] == "1" and mk.macros["EPILOGUE_ROUND"] == "1"
    assert kern["draft_attn"].macros["DRAFT"] == "1" and kern["draft_attn"].macros["STEP_STATE"] == "1"
    embeds = [o for o in prog.ops if o.name == "embed"]
    blk, mk_emb = prog.kernels[embeds[0].kernel], prog.kernels[embeds[1].kernel]
    assert blk.macros["EMBED_IDS"] == "1" and blk.macros["T_STATIC_ROWS"] == f"{gamma}u" and blk.macros["EMBED_PACKED"] == "1"
    assert "EMBED_IDS" not in mk_emb.macros and mk_emb.macros["T_STATIC_ROWS"] == "1u"
    assert kern["lm_head:lm_head.weight"].macros["T"] == str(gamma) and kern["lm_head:lm_head.weight"].macros["NORM"] == "0"
    # bindings: the anchor from StepState, the chain through row offsets, the injected k/v at slot 11
    assert embeds[0].bindings[0] == (0, "step_state", lay.offset("anchor")) and embeds[1].bindings[0] == (0, "step_state", lay.offset("anchor"))
    assert embeds[2].bindings[0] == (0, "draft.tokens", 0) and embeds[3].bindings[0] == (0, "draft.tokens", 4)
    assert [b for b in embeds[2].bindings if b[0] == 2][0] == (2, "draft.markov.emb", 1 * cfg.markov_rank * 2)
    markov = [o for o in prog.ops if o.name == "gemv:draft.markov_w2.w2"]
    assert [b for b in markov[2].bindings if b[0] == 7][0] == (7, "draft.base_logits", 2 * cfg.vocab_size * 2)
    finals = [o for o in prog.ops if o.name == "argmax_final"]
    assert [b for b in finals[1].bindings if b[0] == 2][0] == (2, "draft.tokens", 4)
    attn = [o for o in prog.ops if o.name == "draft_attn"][0]
    assert any(b[0] == 11 and b[1] == "draft.layers.0.self_attn.kv_ctx.k_proj+v_proj.y" for b in attn.bindings)
    assert any(b[0] == 15 and b[1] == "step_state" for b in attn.bindings)
    vs = ops["verify_select"]
    assert vs.bindings[:3] == [(0, "draft.tokens", 0), (1, "draft.confidence", 0), (2, "step_state", 0)]
    assert not any(b[0] == 15 for b in vs.bindings)
    # buffers: taps are host-written arena inputs at t_max rows, the block values static, the caches states, views none
    assert prog.buffers["tap.0"].nbytes == lay.t_max * cfg.target_hidden * 2 and prog.buffers["tap.0"].role == "arena"
    assert prog.buffers["draft.taps"].nbytes == lay.t_max * 2 * cfg.target_hidden * 2
    assert prog.buffers["draft.h0"].nbytes == gamma * cfg.hidden_size * 2 and prog.buffers["draft.tokens"].nbytes == max(gamma * 4, 16)
    assert prog.buffers["draft.layers.1.self_attn.k_ctx"].role == "state" and "draft.tokens.0" not in prog.buffers
    assert "anchor" not in prog.buffers
    win = prog.buffers["pack.0"]
    assert win.file_offset % 16384 == 0 and win.nbytes % 16384 == 0 and win.file_offset + win.nbytes <= pf.manifest["nbytes"]
    again = Program.from_json(prog.to_json())
    assert [o.bindings for o in again.ops] == [o.bindings for o in prog.ops]
    # a static program of the same graph compiles too (T from params, no StepState macros on the row-count kernels)
    static = emit_program(g, pack=pf, profile=PROF, t=4, tail=None)
    assert "T_SRC" not in static.kernels[[o for o in static.ops if o.name == "gemv:draft.fc.fc"][0].kernel].macros
    # a block larger than the layout's gamma_max is refused
    from monolith.core import StepStateLayout

    with pytest.raises(ValueError):
        emit_program(g, pack=pf, profile=PROF, dynamic_t=True, tail=None, layout=StepStateLayout(t_max=3, gamma_max=2))


def test_unbound_kind_still_fails_coverage(tmp_path):
    from monolith.ops import OPS, OpDef, register_op

    write_checkpoint(tmp_path)
    drafter, head, cfg, _ = build(tmp_path)
    g, block, sel = _lower(drafter, cfg)
    register_op(OpDef("test_draft_unbound", OpClass.SERIAL, "span"))
    try:
        g.op("test_draft_unbound", [sel], [g.value("x", (1,), DType.U32)], domain=BlockDomain("span", 1), klass=OpClass.SERIAL)
        with pytest.raises(CoverageError):
            check_coverage(g, PROF)
    finally:
        OPS.unregister("test_draft_unbound")
