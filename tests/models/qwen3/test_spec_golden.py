"""Model 2 with its public DSpark drafter (#38/#39): the NVFP4 Qwen3-8B target and the BF16 drafter packed from
their checkpoints, the speculative round replayed from one encode — the greedy tokens must equal the HF golden (the
plain decode's tokens), and the acceptance and the time per token are reported. Needs the Metal module and both
checkpoints under ~/models; skips otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle_conftest import require_checkpoint  # noqa: E402

from monolith.formats import PackLayout
from monolith.runtime import is_available

GOLDEN = Path(__file__).parent / "goldens" / "qwen3-8b-nvfp4"
CKPT = "nvidia-Qwen3-8B-NVFP4"
DRAFTER = "Dogacel-Qwen3-8B-DSpark"

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


@pytest.mark.parametrize("verify", ["cost", "threshold"])
def test_speculative_greedy_matches_the_golden(tmp_path_factory, verify):
    ckpt, ddir = require_checkpoint(CKPT), require_checkpoint(DRAFTER)
    from monolith.generate import Session
    from monolith.models.qwen3 import Qwen3Model
    from monolith.nn.pack_plan import pack_model
    from monolith.spec.dspark import DSparkDrafter

    with open(str(GOLDEN) + ".json") as f:
        golden = json.load(f)
    out = tmp_path_factory.mktemp("pack")
    model = Qwen3Model.from_checkpoint(str(ckpt), max_context=256)
    pack_model(model, str(ckpt), str(out / "target"), PackLayout())
    drafter = DSparkDrafter.from_checkpoint(str(ddir), target_lm_head=model.lm_head, max_context=256)
    pack_model(drafter, str(ddir), str(out / "drafter"), PackLayout())
    sess = Session(model, str(out / "target"), eos=-1, drafter=drafter, drafter_pack=str(out / "drafter"), verify=verify)
    ids = golden["prompt_ids"]
    gen = sess.generate(ids, len(golden["gen_ids"]))
    assert gen.tokens == golden["gen_ids"], (gen.tokens[:8], golden["gen_ids"][:8])
    assert gen.accepted is not None and sum(gen.committed) >= gen.decode_tokens        # the pump may run a few steps past the request
    hist = {}
    for a in gen.accepted:
        hist[a] = hist.get(a, 0) + 1
    print(f"\n[{verify}] {len(gen.tokens)} greedy tokens equal the golden through {gen.steps} speculative steps: "
          f"{gen.decode_tokens / max(1, gen.steps):.2f} tokens/step, mean accepted {gen.mean_accepted:.2f} of {drafter.gamma}, "
          f"histogram {dict(sorted(hist.items()))}; {gen.ms_per_token:.2f} ms/token GPU ({1000 / gen.ms_per_token:.1f} tok/s), "
          f"{gen.decode_ms / max(1, gen.steps):.1f} ms/step")
