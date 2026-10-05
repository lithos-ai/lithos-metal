"""The LM drafter on the GPU (design §5.8): speculative greedy decode with a synthetic dense Qwen3 draft model inside
the round equals plain decode on the synthetic hybrid target (the GDN commit reading the committed rows, the ingest
across prefill chunks, the chain's single-row steps), and a drafter that *is* the target accepts every draft — the
chain's rows compute what the target's verify rows compute. No torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
from lm_synth import write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.formats import PackLayout  # noqa: E402
from monolith.generate import Session  # noqa: E402
from monolith.models.qwen3 import Qwen3Model  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.spec.lm import LMDrafter  # noqa: E402

MAX_CONTEXT = 64


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    root = tmp_path_factory.mktemp("lm")
    tdir, ddir, qdir = root / "target", root / "drafter", root / "qwen3"
    for d in (tdir, ddir, qdir):
        d.mkdir()
    _checkpoint(tdir)                                                                  # the hybrid target, vocab 50
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=MAX_CONTEXT)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, seed=5, vocab_size=50)                                      # a random dense drafter of the same vocabulary
    write_checkpoint(qdir, seed=9, vocab_size=50, tie=True)                            # a dense target = its own drafter
    q = Qwen3Model.from_checkpoint(str(qdir), max_context=MAX_CONTEXT)
    pack_model(q, str(qdir), str(qdir / "pack"), PackLayout(rows=16))
    return tdir, ddir, qdir


def _drafter(ddir, head, gamma):
    drafter = LMDrafter.from_checkpoint(str(ddir), target_lm_head=head, max_context=MAX_CONTEXT, gamma=gamma)
    if not (ddir / "pack").exists():
        pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    return drafter


@pytest.mark.parametrize("verify,gamma", [("cost", 3), ("fixed", 3), ("cost", 5)])
def test_speculative_equals_plain_greedy_with_an_lm_drafter(packs, verify, gamma):
    tdir, ddir, _ = packs
    plain = Session(Qwen3_5Model.from_checkpoint(str(tdir), max_context=MAX_CONTEXT), str(tdir / "pack"), eos=-1, autotune=False)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=MAX_CONTEXT)
    drafter = _drafter(ddir, model.lm_head, gamma)
    opts = {"verify": "fixed", "verify_length": gamma} if verify == "fixed" else {"verify": "cost"}
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=drafter, drafter_pack=str(ddir / "pack"), **opts)
    rng = np.random.default_rng(3)
    for n_prompt, n_new in ((5, 24), (11, 20), (21, 16), (1, 12)):                    # 11 and 21: the prompt in chunks of t_max = 8
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        ref = plain.generate(ids, n_new)
        got = spec.generate(ids, n_new)
        assert got.tokens == ref.tokens, (n_prompt, got.tokens, ref.tokens)
        assert len(got.tokens) == n_new and got.decode_tokens == n_new - 1
        assert got.accepted is not None and len(got.accepted) == got.steps and sum(got.committed) >= got.decode_tokens
        assert all(1 <= c <= gamma + 1 for c in got.committed) and all(a == c - 1 for a, c in zip(got.accepted, got.committed))
        print(f"\n{verify} γ={gamma} prompt {n_prompt}: {got.steps} steps for {got.decode_tokens} tokens, accepted {got.accepted}")
    st = spec.engine(0).state()
    assert st["error"] == 0 and st["done"] == 1 and st["ring_head"] == st["ring_tail"] == 12
    assert st["position"] == 1 + 12 - 1 and st["n_chain"] == 1


def test_a_drafter_that_is_the_target_accepts_every_draft(packs):
    """The dense synthetic model drafts for itself: the chain's row at a position computes exactly what the target's
    verify row at that position computes (the same shader kernels row by row, the same caches), so every draft is
    accepted and each step commits γ + 1 tokens — with the accelerator off; on the tile the T > 1 rows round
    differently and a near tie may flip, so that run only asks for most drafts."""
    _, _, qdir = packs
    gamma = 4
    # one attention kernel for both sessions as well: the drafter's chain steps (one row) take v3 under the profile's
    # auto and the target's verify pass v1, and their folds differ by BF16 rounding — a near-tie argmax would be a
    # rejected draft here, while the real round drafts with a different model anyway
    plain = Session(Qwen3Model.from_checkpoint(str(qdir), max_context=MAX_CONTEXT), str(qdir / "pack"), eos=-1, autotune=False, accelerator="off",
                    attention="v1")
    model = Qwen3Model.from_checkpoint(str(qdir), max_context=MAX_CONTEXT)
    drafter = _drafter(qdir, model.lm_head, gamma)                                     # the same checkpoint, the same pack
    ddir = qdir / "drafter_pack"
    pack_model(drafter, str(qdir), str(ddir), PackLayout(rows=16))
    spec = Session(model, str(qdir / "pack"), eos=-1, autotune=False, drafter=drafter, drafter_pack=str(ddir), verify="cost", accelerator="off",
                   attention="v1")
    rng = np.random.default_rng(11)
    for n_prompt, n_new in ((6, 31), (13, 26), (19, 22), (8, 20), (16, 18)):       # 8 and 16: a last chunk of t_max rows (the first chain step at t_max + 1)
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        ref = plain.generate(ids, n_new)
        got = spec.generate(ids, n_new)
        assert got.tokens == ref.tokens
        assert all(a == gamma for a in got.accepted[:-1]), got.accepted                 # every draft accepted (the last step may stop early)
        assert got.steps <= -(-(n_new - 1) // (gamma + 1)) + 1
        print(f"\nself-draft prompt {n_prompt}: {got.steps} steps, accepted {got.accepted}")
        # the drafter's caches hold what the target's hold over the committed positions: the ingest read the right
        # tokens at the right positions (the prompt in chunks, the last draft after a full acceptance), the chain too
        eng = spec.engine(0)
        st = eng.state()
        pos = min(int(st["position"]), int(st["drafter_ctx_len"]))                     # the last step's draft pass is skipped behind `done`
        assert pos >= int(st["position"]) - 1
        for i in range(2):
            for c in ("k_cache", "v_cache"):
                a = np.frombuffer(eng.read(f"layers.{i}.self_attn.{c}"), dtype=np.uint16).reshape(MAX_CONTEXT, -1)[:pos]
                b = np.frombuffer(eng.read(f"draft.layers.{i}.self_attn.{c}"), dtype=np.uint16).reshape(MAX_CONTEXT, -1)[:pos]
                bad = np.nonzero((a != b).any(axis=1))[0]
                assert bad.size == 0, (n_prompt, i, c, bad[:8], pos)
    spec.engines.clear(); plain.engines.clear()
    # the profile's own path (the tile for T > 1 when the chip has it): most drafts still accepted
    spec2 = Session(Qwen3Model.from_checkpoint(str(qdir), max_context=MAX_CONTEXT), str(qdir / "pack"), eos=-1, autotune=False,
                    drafter=_drafter(qdir, None, gamma), drafter_pack=str(ddir), verify="cost")
    ids = [int(x) for x in rng.integers(0, 50, 9)]
    got = spec2.generate(ids, 30)
    assert sum(got.accepted) >= 0.8 * gamma * len(got.accepted), got.accepted
