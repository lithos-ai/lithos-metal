"""Speculative greedy decode equals plain greedy decode (design §5.8, #38/#39): the synthetic hybrid target of the
contract tier (a GDN layer whose recurrent state the commit pass must roll back on every rejected draft, and an
attention layer) with a random synthetic DSpark drafter (almost every draft is rejected) — the whole round replayed
from one encode, tokens drained from the ring — against the plain step program on the same weights. No torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
from dspark_synth import build, write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.formats import PackLayout  # noqa: E402
from monolith.generate import Session  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec")
    tdir, ddir = root / "target", root / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, seed=5, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    return tdir, ddir


def _model(tdir):
    return Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)


@pytest.mark.parametrize("verify", ["threshold", "cost"])
def test_speculative_equals_plain_greedy(packs, verify):
    tdir, ddir = packs
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    model = _model(tdir)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=drafter, drafter_pack=str(ddir / "pack"), verify=verify)
    rng = np.random.default_rng(3)
    for n_prompt, n_new in ((5, 24), (11, 20), (1, 12)):
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        ref = plain.generate(ids, n_new)
        got = spec.generate(ids, n_new)
        assert got.tokens == ref.tokens, (n_prompt, got.tokens, ref.tokens)
        assert len(got.tokens) == n_new and got.decode_tokens == n_new - 1
        assert got.accepted is not None and len(got.accepted) == got.steps and sum(got.committed) >= got.decode_tokens
        assert all(1 <= c <= 4 for c in got.committed) and all(a == c - 1 for a, c in zip(got.accepted, got.committed))
        assert all(a <= cfg.block_size for a in got.accepted)
        print(f"\nprompt {n_prompt}: {got.steps} steps for {got.decode_tokens} tokens, accepted {got.accepted}")
    # the ring was drained in order and the program stopped itself at the request (stop_at → done): the position
    # counts every committed token; the final step's closing bookkeeping — the drafter's injection of that step's
    # tokens — is skipped behind `done`, as after an EOS, so the drafter's context is one step short
    st = spec.engine(0).state()
    assert st["position"] == 1 + 12 - 1 and st["ring_head"] == st["ring_tail"] == 12 and st["done"] == 1
    assert st["drafter_ctx_len"] == st["position"] - 1
    assert st["error"] == 0


def test_speculative_equals_plain_greedy_with_a_requantized_drafter(packs, tmp_path):
    """The drafter's BF16 matrices quantized at pack time (#103; INT4 affine here — its K % 256 fits the synthetic
    widths, nvfp4 needs K % 512 and the drafter's hidden must match the target head's): the session binds a fresh
    tree to the pack's formats, the round runs the quantized kernels, and the greedy tokens still equal plain decode's."""
    from monolith.formats.safetensors_reader import SafetensorsDir
    from monolith.nn.pack_plan import bind_formats

    tdir, _ = packs
    ddir = tmp_path / "drafter_q"
    ddir.mkdir()
    write_checkpoint(ddir, seed=5, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    model = _model(tdir)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)
    ckpt = SafetensorsDir(str(ddir))
    try:
        bound = bind_formats(drafter, ckpt, requantize="int4_affine", keep=("embed_tokens", "markov"))
    finally:
        ckpt.close()
    assert bound["layers.0.mlp.down_proj.weight"] == "int4_affine" and bound["embed_tokens.weight"] == "bf16"
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    fresh, _, _, _ = build(ddir, target_lm_head=model.lm_head)                       # bound from the BF16 checkpoint
    assert fresh.blocks[0].mlp.down.format_of("down_proj") == "bf16"
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=fresh, drafter_pack=str(ddir / "pack"), verify="cost")
    assert fresh.blocks[0].mlp.down.format_of("down_proj") == "int4_affine"          # the session rebound it from the pack
    rng = np.random.default_rng(7)
    for n_prompt, n_new in ((5, 24), (9, 16)):
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        ref = plain.generate(ids, n_new)
        got = spec.generate(ids, n_new)
        assert got.tokens == ref.tokens, (n_prompt, got.tokens, ref.tokens)
        assert len(got.tokens) == n_new and got.decode_tokens == n_new - 1
        assert all(a <= cfg.block_size for a in got.accepted)
    assert spec.engine(0).state()["error"] == 0


def test_context_capacity_guard(packs):
    """The program knows how many positions a sequence may occupy (the target's KV rows; with a drafter the smaller of
    that and its context cache less the block): a request that cannot fit is refused before it starts, one that just
    fits completes (the pump's over-run past it stops at the guard, error 2, instead of writing past a cache)."""
    tdir, ddir = packs
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    assert plain.engine(0).program.context_capacity == 64                          # the target's max_context
    ids = [3, 1, 4, 1, 5]
    with pytest.raises(ValueError, match="context capacity"):
        plain.generate(ids, 61)                                                     # 5 + 61 - 1 = 65 positions
    ref = plain.generate(ids, 60)                                                   # exactly 64 positions
    assert len(ref.tokens) == 60 and plain.engine(0).state()["error"] in (0, 2)
    model = _model(tdir)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)                # context caches of 32: 32 - 3 + 1 = 30 positions
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=drafter, drafter_pack=str(ddir / "pack"), verify="threshold")
    assert spec.engine(0).program.context_capacity == 30
    with pytest.raises(ValueError, match="context capacity"):
        spec.generate(ids, 27)
    got = spec.generate(ids, 26)
    assert got.tokens == ref.tokens[:26]
    st = spec.engine(0).state()
    assert st["position"] <= 30 and st["drafter_ctx_len"] <= 30 and st["error"] in (0, 2)
    # a roomier drafter takes the target's capacity
    wide, _, _, _ = build(ddir, target_lm_head=model.lm_head, max_context=256)
    pack_model(wide, str(ddir), str(ddir / "wide-pack"), PackLayout(rows=16))
    wide_s = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=wide, drafter_pack=str(ddir / "wide-pack"), verify="threshold")
    assert wide_s.engine(0).program.context_capacity == 64
    assert wide_s.generate(ids, 60).tokens == ref.tokens


def _counts(tokens_by_seed, position):
    c = {}
    for toks in tokens_by_seed:
        t = toks[position]
        c[t] = c.get(t, 0) + 1
    return c


def test_speculative_sampling_preserves_the_target_distribution(packs):
    """Temperature > 0 with a drafter: every committed token is a sample of the target's conditional (design §5.8)
    — the empirical distributions of the 2nd and 3rd generated tokens over many seeds agree with plain sampling's,
    and at a near-zero temperature the speculative sequence equals the plain one token for token."""
    tdir, ddir = packs
    model = _model(tdir)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    ids = [7, 23, 41, 3]
    n_seeds, n_new = 1000, 3
    plain_seqs, spec_seqs, same_seed = [], [], []
    # one attention kernel for both: plain decode runs the static T = 1 program (v3 under the profile's auto) and the
    # speculative session the dynamic program (v1 at rep · t_max rows), whose folds differ by BF16 rounding — the
    # same-seed identity below needs the same logits to the bit; the distribution checks hold under any kernel
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False, temperature=0.9, top_k=12, attention="v2")
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, temperature=0.9, top_k=12, drafter=drafter, drafter_pack=str(ddir / "pack"),
                   verify="threshold", verify_threshold=0.0, attention="v2")
    for seed in range(n_seeds):
        plain.seed, spec.seed = seed, seed + 100_000                        # independent streams: two samples of one distribution
        plain_seqs.append(plain.generate(ids, n_new).tokens)
        spec_seqs.append(spec.generate(ids, n_new).tokens)
        if seed < 50:                                                      # the same seed: the draws coincide while drafts are rejected
            spec.seed = seed
            same_seed.append(spec.generate(ids, 2).tokens)
    assert all(p[:2] == s for p, s in zip(plain_seqs, same_seed))
    # two independent samples of one distribution: per-token counts within 4.5 σ (two independent plain runs give a
    # total variation of 0.10–0.14 here because the first token varies by seed and mixes ~45 contexts)
    for pos in (1, 2):
        cp, cs = _counts(plain_seqs, pos), _counts(spec_seqs, pos)
        tv = 0.5 * sum(abs(cp.get(t, 0) - cs.get(t, 0)) for t in set(cp) | set(cs)) / n_seeds
        worst = max(abs(cp.get(t, 0) - cs.get(t, 0)) / (2.0 * max(cp.get(t, 0), 1)) ** 0.5 for t in cp if cp[t] >= 20)
        print(f"\nposition {pos}: plain {dict(sorted(cp.items(), key=lambda kv: -kv[1])[:5])} spec {dict(sorted(cs.items(), key=lambda kv: -kv[1])[:5])} "
              f"TV {tv:.3f} worst z {worst:.2f}")
        assert tv < 0.2 and worst < 4.5, (pos, tv, worst)
    # the same context: the second token's distribution among the sequences that share the commonest first token
    first = max(_counts(plain_seqs, 0).items(), key=lambda kv: kv[1])[0]
    cp = _counts([s for s in plain_seqs if s[0] == first], 1)
    cs = _counts([s for s in spec_seqs if s[0] == first], 1)
    n_p, n_s = sum(cp.values()), sum(cs.values())
    assert n_p >= 100 and n_s >= 100
    worst = max(abs(cp.get(t, 0) / n_p - cs.get(t, 0) / n_s) / (cp[t] / n_p * (1 - cp[t] / n_p) * (1 / n_p + 1 / n_s)) ** 0.5 for t in cp if cp[t] >= 15)
    print(f"same first token {first}: n = {n_p} / {n_s}, {len(set(cp) | set(cs))} tokens, worst z {worst:.2f}")
    assert worst < 4.5
    assert len(set(tuple(s) for s in spec_seqs)) > 10                        # it does sample
    # a near-greedy temperature: identical sequences (the sampler's draws are the argmax)
    cold_p = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False, temperature=1e-3, seed=5)
    cold_s = Session(model, str(tdir / "pack"), eos=-1, autotune=False, temperature=1e-3, seed=5, drafter=drafter, drafter_pack=str(ddir / "pack"))
    greedy = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    a, b, c = cold_p.generate(ids, 16).tokens, cold_s.generate(ids, 16).tokens, greedy.generate(ids, 16).tokens
    assert a == b == c
