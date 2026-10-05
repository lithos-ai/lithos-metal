"""The model oracle against the HF golden (plan M4 exit gate (a), oracle path): per-layer hidden states of the
prompt prefill and the greedy continuation. Needs torch and the small checkpoint; skips otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle_conftest import require_checkpoint, require_torch  # noqa: E402

from monolith.formats.fp import bf16_to_f32
from monolith.formats.safetensors_reader import SafetensorsDir

GOLDEN = Path(__file__).parent / "goldens" / "qwen3_5-0.8b"
CKPT = "Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def golden():
    with open(str(GOLDEN) + ".json") as f:
        manifest = json.load(f)
    st = SafetensorsDir(str(GOLDEN) + ".safetensors")
    data = {"hidden_states": bf16_to_f32(st.get("hidden_states")), "logits_last": bf16_to_f32(st.get("logits_last")),
            "tf_top_idx": np.array(st.get("tf_top_idx")), "tf_top_val": np.array(st.get("tf_top_val"))}
    st.close()
    return manifest, data


@pytest.fixture(scope="module")
def model():
    torch = require_torch()
    ckpt = require_checkpoint(CKPT)
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.models.qwen3_5.weights import load_oracle

    torch.manual_seed(0)
    m = Qwen3_5Model.from_checkpoint(str(ckpt), max_context=256)
    load_oracle(m, str(ckpt))
    return m


def _bars(got, ref, tol):
    got, ref = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    scale = float(np.abs(ref).max())
    max_abs = float(np.abs(got - ref).max())
    return cos, max_abs, scale, bool(cos > 0.999 and max_abs <= tol * scale)


def test_prefill_hidden_states_and_logits(model, golden):
    torch = require_torch()
    manifest, data = golden
    ids = torch.tensor(manifest["prompt_ids"], dtype=torch.int64)
    with torch.no_grad():
        logits, hiddens, _ = model.forward(ids, model.init_state(), 0)
        final = model.norm.forward(hiddens[-1])
    hs = data["hidden_states"]
    assert hs.shape[0] == len(hiddens)
    rows = []
    ok_all = True
    # cos > 0.999 is the gate (design §5.9); the max-abs bar is 1/16 of the largest reference value because the
    # reference's prefill runs the chunked delta rule and BF16 matmul chains while the oracle runs the fused-kernel
    # semantics (FP32 inside an op, one rounding at its output), and 24 layers of that reach ~3 % of max
    for i in range(hs.shape[0]):
        ours = (final if i == hs.shape[0] - 1 else hiddens[i]).float().numpy()
        cos, max_abs, scale, ok = _bars(ours, hs[i], tol=1 / 16)
        rows.append(f"  h[{i:2d}] cos={cos:.6f} max_abs={max_abs:.4f} scale={scale:.2f} {'ok' if ok else 'FAIL'}")
        ok_all &= ok
    print("\n".join(rows))
    ref_logits = data["logits_last"]
    ours_logits = logits[-1].float().numpy()
    cos, max_abs, scale, ok = _bars(ours_logits, ref_logits, tol=1 / 16)
    print(f"  logits[-1] cos={cos:.6f} max_abs={max_abs:.4f} scale={scale:.2f} argmax ours={int(ours_logits.argmax())} ref={int(ref_logits.argmax())}")
    assert ok_all, "a hidden state missed cos > 0.999 / max-abs bars (see output)"
    assert ok and int(ours_logits.argmax()) == int(ref_logits.argmax())


def test_greedy_continuation(model, golden):
    torch = require_torch()
    manifest, data = golden
    from monolith.nn.oracle import greedy_decode

    ids = manifest["prompt_ids"]
    ref = manifest["gen_ids"]
    with torch.no_grad():
        gen, _, _ = greedy_decode(model, ids, len(ref))
    n_match = next((i for i, (a, b) in enumerate(zip(gen, ref)) if a != b), len(ref))
    print(f"free-running greedy: {n_match}/{len(ref)} tokens match the golden; ours={gen[:12]}… ref={ref[:12]}…")
    if n_match == len(ref):
        return
    # Diverged: judge the first mismatch against the golden's teacher-forced logit margin at that position. A flip
    # inside the BF16 rounding noise of near-tied logits is not an error (design §5.9: "except at exact ties").
    p = len(ids)
    top_val = data["tf_top_val"][p - 1 + n_match]
    top_idx = data["tf_top_idx"][p - 1 + n_match]
    margin = float(top_val[0] - top_val[1])
    print(f"first mismatch at +{n_match}: ours={gen[n_match]} ref={ref[n_match]} golden top2={top_idx[:2].tolist()} margin={margin:.4f}")
    assert gen[n_match] in top_idx[:2].tolist() and margin < 0.25, "greedy token differs from the golden outside a near-tie"
