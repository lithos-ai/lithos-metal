"""Model 2 (plan M8, #45) on the GPU: the NVFP4 Qwen3-8B packed from its model package, compiled and replayed,
against the HF golden computed on the dequantized weights — the greedy tokens and the prefill residual streams of
every layer (read straight from the program's buffers, no torch model needed). Needs the Metal module and the
checkpoint under ~/models; skips otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle_conftest import require_checkpoint  # noqa: E402

from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.runtime import is_available

GOLDEN = Path(__file__).parent / "goldens" / "qwen3-8b-nvfp4"
CKPT = "nvidia-Qwen3-8B-NVFP4"

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


@pytest.fixture(scope="module", params=["inline", "payload"])
def session(tmp_path_factory, request):
    ckpt = require_checkpoint(CKPT)
    from monolith.generate import Session
    from monolith.models.qwen3 import Qwen3Model
    from monolith.nn.pack_plan import pack_model

    out = tmp_path_factory.mktemp("pack")
    model = Qwen3Model.from_checkpoint(str(ckpt), max_context=256)
    layout = PackLayout() if request.param == "inline" else PackLayout(scale_placement="block", scale_order="payload")
    pack_model(model, str(ckpt), str(out), layout)
    return Session(model, str(out), eos=-1)


def _bars(got, ref):
    got, ref = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    return float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30)), float(np.abs(got - ref).max()), float(np.abs(ref).max())


def test_greedy_tokens_and_prefill_layers_match_the_golden(session):
    with open(str(GOLDEN) + ".json") as f:
        golden = json.load(f)
    st = SafetensorsDir(str(GOLDEN) + ".safetensors")
    hs = bf16_to_f32(st.get("hidden_states"))                      # [L+1, P, H]: embeddings, layer outputs, the last normed
    st.close()
    ids = golden["prompt_ids"]
    session.generate(ids, 1)                                          # prefill only: the arena still holds every prompt row
    p, h = len(ids), session.model.config.hidden_size
    rows, ok_all = [], True
    for i in range(session.model.n_layers - 1):                       # the golden's last entry is after the final norm
        got = bf16_to_f32(np.frombuffer(session.read(f"layers.{i}.mlp.h"), dtype=np.uint16)).reshape(-1, h)[:p]
        cos, max_abs, scale = _bars(got, hs[i + 1])
        ok = cos > 0.999
        ok_all &= ok
        rows.append(f"  layer {i:2d}: cos={cos:.6f} max_abs={max_abs:.4f} scale={scale:.2f} {'ok' if ok else 'FAIL'}")
    print("\n" + "\n".join(rows[:4] + ["  …"] + rows[-3:]))
    assert ok_all
    gen = session.generate(ids, len(golden["gen_ids"]))
    assert gen.tokens == golden["gen_ids"], (gen.tokens[:8], golden["gen_ids"][:8])
    print(f"{len(gen.tokens)} greedy tokens equal the golden; decode {gen.ms_per_token:.2f} ms/token GPU ({1000 / gen.ms_per_token:.1f} tok/s)")
