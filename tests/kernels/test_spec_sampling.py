"""Speculative sampling with sampled drafts (``draft_sampling="sample"``): drafts d_k ~ q_k = softmax(logits_k / T) are
accepted with min(1, p / q) and the first rejected one is replaced by a draw from norm(max(p - q, 0)), so the committed
tokens are distributed as plain sampling at the same temperature, top-k and top-p. Checked by a two-sample chi-square
of the first three generated tokens (and the joint of the first two) over independent seed sets, on the synthetic
hybrid target and the random synthetic DSpark drafter of test_spec_rollback at the serving block (seven drafts, eight
verify rows); greedy output is unchanged by the flag."""

import math
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
    root = tmp_path_factory.mktemp("specq")
    tdir, ddir = root / "target", root / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    pack_model(_model(tdir), str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, seed=5, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=7)
    pack_model(_drafter(ddir, _model(tdir)), str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    return tdir, ddir


def _model(tdir):
    return Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)


def _drafter(ddir, model):
    return build(ddir, target_lm_head=model.lm_head, max_context=64)[0]


def _speculative(packs, **options):
    tdir, ddir = packs
    model = _model(tdir)
    return Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=_drafter(ddir, model), drafter_pack=str(ddir / "pack"),
                   draft_sampling="sample", **options)


def _chi2_z(a, b, min_expected=5.0):
    """Two-sample chi-square of two count tables (categories pooled until each expected count reaches
    ``min_expected``) as a z-score (Wilson-Hilferty)."""
    na, nb = sum(a.values()), sum(b.values())
    cells, ca, cb = [], 0, 0
    for k in sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) + b.get(k, 0))):
        ca += a.get(k, 0); cb += b.get(k, 0)
        if (ca + cb) * min(na, nb) / (na + nb) >= min_expected:
            cells.append((ca, cb)); ca = cb = 0
    if ca + cb and cells:
        x, y = cells.pop(); ca += x; cb += y
    if ca + cb:
        cells.append((ca, cb))
    stat = sum((x - (x + y) * na / (na + nb)) ** 2 / ((x + y) * na / (na + nb)) + (y - (x + y) * nb / (na + nb)) ** 2 / ((x + y) * nb / (na + nb))
               for x, y in cells)
    df = max(len(cells) - 1, 1)
    return ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))


def _counts(seqs, key):
    c = {}
    for s in seqs:
        c[key(s)] = c.get(key(s), 0) + 1
    return c


@pytest.mark.parametrize("verify,sampling", [
    ("threshold", dict(temperature=0.9, top_k=12, top_p=0.9)), ("threshold", dict(temperature=1.3, top_p=0.8)),
    ("threshold", dict(temperature=0.6, top_k=8)), ("threshold", dict(temperature=1.0, top_k=1)),
    ("cost", dict(temperature=1.0, top_k=20, top_p=0.95))],                 # the cost rule may verify only part of the block
    ids=["top-k+top-p", "top-p", "top-k", "top-k-1", "cost-rule"])
def test_sampled_drafts_preserve_the_target_distribution(packs, verify, sampling):
    tdir, _ = packs
    # one attention kernel for both programs, as in test_spec_rollback; "q" also for the plain session: the same top-k /
    # top-p order
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False, attention="v2", draft_sampling="sample", **sampling)
    spec = _speculative(packs, attention="v2", verify=verify, **({"verify_threshold": 0.0} if verify == "threshold" else {}), **sampling)
    ids, n_seeds, n_new = [7, 23, 41, 3], 1500, 3
    plain_seqs, spec_seqs, accepted, verified = [], [], 0, []
    for seed in range(n_seeds):
        plain.seed, spec.seed = seed, seed + 100_000                        # independent streams: two samples of one distribution
        plain_seqs.append(tuple(plain.generate(ids, n_new).tokens))
        g = spec.generate(ids, n_new)
        spec_seqs.append(tuple(g.tokens))
        accepted += sum(g.accepted)
        verified += g.verify_len
    if sampling.get("top_k") == 1:                                          # a point mass: the target's argmax whatever was drafted
        assert spec_seqs == plain_seqs
        return
    zs = {name: _chi2_z(_counts(plain_seqs, key), _counts(spec_seqs, key))
          for name, key in (("t0", lambda s: s[0]), ("t1", lambda s: s[1]), ("t2", lambda s: s[2]), ("t0t1", lambda s: s[:2]))}
    print(f"\n{verify} {sampling}: z {zs}; {accepted} drafts accepted in {len(verified)} rounds, verify lengths {sorted(set(verified))}")
    assert all(z < 4.0 for z in zs.values()), zs                            # z ~ N(0, 1) under the null
    if verify == "threshold":
        assert 0 < accepted < sum(verified)                                 # both the accept and the correction paths ran


def test_greedy_is_unchanged(packs):
    tdir, _ = packs
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    spec = _speculative(packs, verify="threshold")
    assert spec.drafter.sampling is None
    rng = np.random.default_rng(3)
    for n_prompt, n_new in ((5, 24), (11, 20), (1, 12)):
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        assert spec.generate(ids, n_new).tokens == plain.generate(ids, n_new).tokens
