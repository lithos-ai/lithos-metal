"""The LM drafter (design §5.8): a registered model package built with a prefix, run inside the round as the draft
model — the plugin's contract, the graph's activation scopes it relies on, and the round program it lowers into
(no GPU): the ingest pass on the injected rows, the chain of single-row steps, the serial ops' LM flags."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lm_synth import write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402
from test_spec_program import PROF, PROF_COST  # noqa: E402

from monolith.compiler import compile_program  # noqa: E402
from monolith.core import Profile  # noqa: E402
from monolith.core.dtypes import DType  # noqa: E402
from monolith.core.ir import Graph  # noqa: E402
from monolith.core.shapes import N_CHAIN, N_FIRST, N_INJ, T, step_bindings  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3 import Qwen3Model  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.spec import DRAFTERS  # noqa: E402
from monolith.spec.lm import LMDrafter  # noqa: E402


def test_graph_scopes_activations_only():
    g = Graph("s")
    w = g.weight("w", (4, 256), "bf16")
    with g.scope("a."):
        h = g.value("h", (T, 4), DType.BF16)
        s = g.state("cache", (8,), DType.BF16)
        c = g.const("tab", (8,), DType.F32)
        with g.scope("b."):
            h2 = g.value("h", (T, 4), DType.BF16)
    h3 = g.value("h", (T, 4), DType.BF16)
    assert (h.name, h2.name, h3.name, s.name, c.name, w.name) == ("a.h", "a.b.h", "h", "cache", "tab", "w")
    with pytest.raises(ValueError):
        with g.scope("a."):
            g.value("h", (T, 4), DType.BF16)
    assert step_bindings(8) == {T: 8, N_INJ: 8, N_CHAIN: 1, N_FIRST: 9}


@pytest.fixture
def pair(tmp_path):
    """The synthetic hybrid target (vocab 50) and a synthetic dense Qwen3 LM drafter of the same vocabulary."""
    tdir, ddir = tmp_path / "target", tmp_path / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=16)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, vocab_size=50)
    drafter = LMDrafter.from_checkpoint(str(ddir), target_lm_head=model.lm_head, max_context=16, gamma=3)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    return model, drafter, PackFile(tdir / "pack"), PackFile(ddir / "pack")


def test_plugin_contract(pair, tmp_path):
    model, drafter, tp, dp = pair
    assert DRAFTERS.resolve("lm") is LMDrafter and drafter.lm_drafter and drafter.gamma == 3 and drafter.tap_layers() == []
    assert drafter.model.prefix == "draft." and isinstance(drafter.model, Qwen3Model)
    assert [e.name for e in drafter.state_entries()] == ["draft.layers.0.self_attn.k_cache", "draft.layers.0.self_attn.v_cache",
                                                         "draft.layers.1.self_attn.k_cache", "draft.layers.1.self_attn.v_cache"]
    assert set(drafter.tables()) == {"draft.rope_cos", "draft.rope_sin"}
    slabs = {s["name"] for s in dp.manifest["slabs"]}
    assert "draft.embed_tokens.weight" in slabs and "draft.lm_head.weight" in slabs and "draft.layers.0.mlp.gate_up.gate_proj+up_proj" in slabs
    assert {a["name"] for a in dp.manifest["aux"]} >= {"draft.norm.weight", "draft.layers.1.self_attn.q_norm"}
    assert set(drafter.full_weight_map()) == {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"} | {
        f"model.layers.{i}.{n}" for i in range(2) for n in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
                                                            "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                                            "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight",
                                                            "mlp.up_proj.weight", "mlp.down_proj.weight")}
    # the drafter's vocabulary may not exceed the target's; gamma is bounded; a model without a prefix is refused
    big = tmp_path / "big"
    big.mkdir()
    write_checkpoint(big, vocab_size=64)
    with pytest.raises(ValueError):
        LMDrafter.from_checkpoint(str(big), target_lm_head=model.lm_head, max_context=16)
    with pytest.raises(ValueError):
        LMDrafter.from_checkpoint(str(big), target_lm_head=None, max_context=16, gamma=0)
    with pytest.raises(ValueError):
        LMDrafter(Qwen3Model.from_checkpoint(str(big), max_context=16), gamma=2)
    # a hybrid package cannot draft: the GDN kernels have no ingest / chain modes
    with pytest.raises((NotImplementedError, TypeError)):
        LMDrafter.from_checkpoint(str(tmp_path / "target"), target_lm_head=None, max_context=16)


def test_round_program_with_an_lm_drafter(pair):
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF_COST, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="cost")
    names = [o.name for o in prog.ops]
    kinds = [o.name.split(":")[0] for o in prog.ops]
    assert kinds[-1] == "verify_select" and kinds.count("accept_scan") == 1 and kinds.count("gdn_commit") == 1 and "tap_concat" not in kinds
    # the serial ops carry the LM flag: AcceptParams.lm, SelectParams.lm (pad2); no confidences → the whole chain
    acc = [o for o in prog.ops if o.name == "accept_scan"][0]
    a_prm = prog.buffers[[b for b in acc.bindings if b[0] == 3][0][1]].init
    assert len(a_prm) == 32 and struct.unpack_from("<I", a_prm, 16)[0] == 1
    vs = [o for o in prog.ops if o.name == "verify_select"][0]
    s_prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    gamma, thr, t_max, mode = struct.unpack_from("<IfII", s_prm)
    assert (gamma, thr, t_max, mode) == (3, 0.0, 8, 0) and struct.unpack_from("<I", s_prm, 16 + 64 + 8)[0] == 1
    # the first chain step carries the ingest rows and the anchor (EMBED_IDS 3, n_inject + n_chain rows: T_SRC 4, LM mode 3,
    # the per-T variants of a prefill chunk plus one, the argmax of the last row); the later steps one row each (n_chain: T_SRC 3, LM mode 2 at step i)
    K = lambda o: prog.kernels[o.kernel].macros
    emb = [o for o in prog.ops if o.name == "embed"]
    assert [(K(o).get("EMBED_IDS"), K(o).get("T_SRC")) for o in emb] == [(None, "0"), ("3", "4"), (None, "3"), (None, "3")]
    modes = [(None, None), ("3", None), ("3", None)] + [("2", f"{i}u") for i in (1, 2) for _ in range(2)]   # the target's, the first step's two layers, the rest
    assert [(K(o).get("LM_MODE"), K(o).get("CHAIN_I")) for o in prog.ops if o.name == "gqa_decode"] == modes
    assert [(K(o).get("LM_MODE"), K(o).get("CHAIN_I")) for o in prog.ops if o.name == "gqa_merge"] == modes
    gu = [o for o in prog.ops if o.name == "gemv:draft.layers.0.mlp.gate_up.gate_proj+up_proj"]
    assert [(o.meta["t_variant"], K(o)["T_SRC"], K(o).get("T_HI")) for o in gu] == [(1, "4", "1"), (2, "4", "2"), (4, "4", "4"), (8, "4", "8"), (9, "4", "9")] + [(1, "3", None)] * 2
    heads = [(o.name, o.meta["t_variant"], K(o)["T_SRC"]) for o in prog.ops if o.name.startswith("lm_head:")]
    assert heads == [("lm_head:embed_tokens.weight", v, "0") for v in (2, 4, 8)] + [("lm_head:draft.lm_head.weight", v, "4") for v in (1, 2, 4, 8, 9)] \
        + [("lm_head:draft.lm_head.weight", 1, "3")] * 2   # the target's (T = 1 pruned), the first step's, the rest
    am = [(K(o).get("T_SRC"), K(o).get("ARGMAX_LAST")) for o in prog.ops if o.name in ("argmax", "argmax_final")]
    assert am == [("0", None), ("0", None), ("4", "1"), ("4", "1")] + [("3", None), ("3", None)] * 2
    ops = {o.name: o for o in prog.ops}
    # the GDN commit pass reads the committed rows through checkpoint_index (n_inject is the drafter's row count)
    assert "checkpoint_index" in prog.kernels[ops["gdn_commit"].kernel].source.split("kernel void gdn_mixer")[1].split("#if SLOTS")[0]
    # the states of both trees are allocated, distinct
    assert {"layers.1.self_attn.k_cache", "draft.layers.0.self_attn.k_cache", "draft.layers.1.self_attn.v_cache"} <= set(prog.buffers)
    assert prog.layout.field("n_chain") is not None


def test_fixed_length_and_the_step_state_symbol(pair):
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=2)
    vs = [o for o in prog.ops if o.name == "verify_select"][0]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    assert struct.unpack_from("<IfII", prm)[1:] == (2.0, 8, 2)
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=4)
    msl = prog.layout.to_msl()
    assert "uint n_chain;" in msl and "uint stop_at;" in msl


def test_row_split_choices_keep_the_norm_stat_partial_count_consistent(pair):
    """A tuned RSPLIT on the T = 1 shader variant of a GEMV that also has a tile (T > 1 rows: the first chain step in a
    prefill chunk) must not change the statistic partial count its consumer reads: the tile writes one partial per
    block. Without the tile the variants agree on the smallest split they all admit."""
    from monolith.compiler.autotune import Choice

    model, drafter, tp, dp = pair

    class SplitTuner:
        def tune_gemv(self, info, t, epilogue, norm_fed, **kw):
            return Choice({"RG": "2", "RSPLIT": "8u"} if t == 1 else {"RG": "8", "RSPLIT": "2u"}, "crew", False, 0.0, 0.0)

        def tune_gemm(self, info, tm, epilogue, **kw):
            return Choice({}, "crew", False, 0.0, 0.0)

        def tune_gdn(self, *a, **kw):
            return None

        def save(self, *a):
            pass

    prof_on = Profile.from_dict("pa", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16", "accelerator": "on",
                                                                                          "cost_T": {"int4_affine": {"1": 1.0, "2": 1.1, "4": 1.3, "8": 2.0}, "bf16": {"1": 1.0, "2": 1.1, "4": 1.3, "8": 2.0}}}})
    for prof in (prof_on, PROF_COST):
        prog = compile_program(model, tp, prof, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="cost", tuner=SplitTuner())
        writers = {}
        for o in prog.ops:
            so = [b for b in o.bindings if b[0] == 8 and o.name.startswith("gemv:")]
            if so:
                writers.setdefault(so[0][1], set()).add(o.meta.get("rsplit", 1) if o.name.startswith("gemv:") else 0)
        for stat, splits in writers.items():
            assert len(splits) == 1, (stat, splits)                   # every shader variant of an op writes the same number of partials
        first = [o for o in prog.ops if o.name == "gemv:draft.layers.0.self_attn.o_proj.o_proj"]
        with_tile = [o for o in first if o.meta.get("t_range") is not None]                      # the first chain step's shader variant(s) beside a tile
        chain = [o for o in first if o.meta.get("t_range") is None]                              # the pure T = 1 chain steps
        if prof is prof_on:
            assert with_tile and all(o.meta.get("rsplit", 1) == 1 for o in with_tile)
        assert chain and all(o.meta.get("rsplit", 1) == 8 for o in chain)
