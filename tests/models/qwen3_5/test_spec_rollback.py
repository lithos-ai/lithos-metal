"""The speculative round on the real hybrid 0.8B with a random synthetic drafter (#38): every step verifies a block
of garbage drafts, rejects (almost) all of them and rolls the GDN states back through the commit pass — the greedy
tokens must still equal the HF golden. Needs the Metal module and the checkpoint under ~/models; skips otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dspark_synth import build, write_checkpoint  # noqa: E402
from oracle_conftest import require_checkpoint  # noqa: E402

from monolith.formats import PackLayout
from monolith.runtime import is_available

GOLDEN = Path(__file__).parent / "goldens" / "qwen3_5-0.8b"
CKPT = "Qwen3.5-0.8B"

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


def test_speculative_greedy_matches_the_golden(tmp_path):
    ckpt = require_checkpoint(CKPT)
    from monolith.generate import Session
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.nn.pack_plan import pack_model

    with open(str(GOLDEN) + ".json") as f:
        golden = json.load(f)
    model = Qwen3_5Model.from_checkpoint(str(ckpt), max_context=512)
    pack_model(model, str(ckpt), str(tmp_path / "pack"), PackLayout())
    c = model.config
    ddir = tmp_path / "drafter"
    ddir.mkdir()
    # the drafter's hidden width is the target's (its block goes through the target's head) and its context caches
    # hold the whole generation (the program refuses a request past the smaller capacity)
    write_checkpoint(ddir, seed=9, with_head=False, vocab_size=c.vocab_size, hidden_size=c.hidden_size, target_hidden_size=c.hidden_size,
                     target_layer_ids=[-1, 5, 11], num_hidden_layers=1, block_size=3)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head, max_context=model.max_context)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout())
    sess = Session(model, str(tmp_path / "pack"), eos=-1, drafter=drafter, drafter_pack=str(ddir / "pack"), verify="threshold")
    ids = golden["prompt_ids"]
    gen = sess.generate(ids, len(golden["gen_ids"]))
    assert gen.tokens == golden["gen_ids"], (gen.tokens[:8], golden["gen_ids"][:8])
    assert gen.accepted is not None and gen.steps >= 1 and sum(gen.committed) == gen.decode_tokens
    rejected = sum(1 for a in gen.accepted if a < cfg.block_size)
    assert rejected > 0                                            # the rollback path ran
    print(f"\n{len(gen.tokens)} greedy tokens equal the golden through {gen.steps} speculative steps (mean accepted {gen.mean_accepted:.2f}, "
          f"{rejected} steps rolled back); {gen.ms_per_token:.2f} ms/token GPU")
