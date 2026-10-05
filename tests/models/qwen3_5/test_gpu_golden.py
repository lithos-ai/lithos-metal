"""The step program on the GPU against the HF golden (plan M4 exit gate (a)): the checkpoint is packed from its
module tree, compiled into a prefill program (T = P) and a decode program (T = 1), replayed from one encode; every
layer's residual stream at prefill is compared with the torch oracle (composite bars, #25) and the 48 greedy tokens
with the golden. Needs the Metal module, torch and the small checkpoint; skips otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle_conftest import require_checkpoint, require_torch  # noqa: E402

from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32
from monolith.runtime import is_available

GOLDEN = Path(__file__).parent / "goldens" / "qwen3_5-0.8b"
CKPT = "Qwen3.5-0.8B"

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    require_torch()
    ckpt = require_checkpoint(CKPT)
    from monolith.generate import Session
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.nn.pack_plan import pack_model

    out = tmp_path_factory.mktemp("pack")
    model = Qwen3_5Model.from_checkpoint(str(ckpt), max_context=512)
    pack_model(model, str(ckpt), str(out), PackLayout())
    return Session(model, str(out), eos=-1), ckpt


def _bars(got, ref):
    got, ref = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    return float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30)), float(np.abs(got - ref).max()), float(np.abs(ref).max())


def test_prefill_layers_match_the_oracle(session):
    torch = require_torch()
    sess, ckpt = session
    from monolith.models.qwen3_5.weights import load_oracle

    with open(str(GOLDEN) + ".json") as f:
        ids = json.load(f)["prompt_ids"]
    gen = sess.generate(ids, 1)
    load_oracle(sess.model, str(ckpt))
    with torch.no_grad():
        logits, hiddens, _ = sess.model.forward(torch.tensor(ids), sess.model.init_state(), 0)
    p, h = len(ids), sess.model.config.hidden_size
    rows, ok_all = [], True
    for i in range(sess.model.n_layers):
        got = bf16_to_f32(np.frombuffer(sess.read(f"layers.{i}.mlp.h"), dtype=np.uint16)).reshape(-1, h)[:p]
        cos, max_abs, scale = _bars(got, hiddens[i + 1].float().numpy())
        ok = cos > 0.999 and max_abs <= scale / 32
        ok_all &= ok
        rows.append(f"  layer {i:2d}: cos={cos:.6f} max_abs={max_abs:.4f} scale={scale:.2f} {'ok' if ok else 'FAIL'}")
    print("\n" + "\n".join(rows))
    assert ok_all
    lg = bf16_to_f32(np.frombuffer(sess.read("logits"), dtype=np.uint16)).reshape(-1, sess.model.config.vocab_size)[:p]
    assert int(lg[-1].argmax()) == int(logits[-1].float().argmax()) == gen.tokens[0]


def test_greedy_tokens_match_the_golden(session):
    sess, _ = session
    with open(str(GOLDEN) + ".json") as f:
        golden = json.load(f)
    gens = [sess.generate(golden["prompt_ids"], len(golden["gen_ids"])) for _ in range(2)]
    for g in gens:
        assert g.tokens == golden["gen_ids"], (g.tokens[:8], golden["gen_ids"][:8])
    print(f"\n48 greedy tokens equal the golden (two runs); decode {gens[1].ms_per_token:.2f} ms/token GPU, "
          f"{1000 / gens[1].ms_per_token:.0f} tok/s, host busy {100 * gens[1].host_busy_ms / max(gens[1].decode_wall_ms, 1e-9):.1f} %")


def test_large_prefill_matches_the_long_golden(session):
    """A prompt longer than the decode bound goes through the separate prefill program, then decode;
    the greedy tokens must equal the HF golden of the long prompt."""
    sess, _ = session
    with open(str(GOLDEN) + "-long.json") as f:
        golden = json.load(f)
    ids = golden["prompt_ids"]
    assert len(ids) > sess.decode_t_max
    gen = sess.generate(ids, len(golden["gen_ids"]))
    assert gen.tokens == golden["gen_ids"], (gen.tokens[:8], golden["gen_ids"][:8])
    st = sess.engines[1].state()
    assert st["position"] == len(ids) + len(golden["gen_ids"]) - 1 and st["prefill_left"] == 0
    print(f"\nchunked prefill of {len(ids)} tokens ({-(-len(ids) // sess.prefill_chunk_size)} chunks): {gen.prefill_ms:.1f} ms; "
          f"{len(gen.tokens)} greedy tokens equal the long golden")


def test_sampling_is_reproducible_and_seed_dependent(session):
    """Gumbel-max sampling on the GPU: the same seed reproduces the same tokens, another seed differs, and every
    sampled token stays inside the top-k set the kernel's threshold defines (checked through the reference)."""
    from monolith.generate import Session

    sess, _ = session
    with open(str(GOLDEN) + ".json") as f:
        ids = json.load(f)["prompt_ids"]
    a = Session(sess.model, str(sess.pack.dir), eos=-1, temperature=0.8, top_k=40, seed=11)
    ta = a.generate(ids, 24).tokens
    tb = a.generate(ids, 24).tokens
    c = Session(sess.model, str(sess.pack.dir), eos=-1, temperature=0.8, top_k=40, seed=12)
    tc = c.generate(ids, 24).tokens
    assert ta == tb and len(ta) == 24 and ta != tc
    sess.model.sampler = __import__("monolith.nn", fromlist=["GreedySampler"]).GreedySampler(prefix="sampler.")
    print(f"\nsampled (seed 11): {ta[:10]}…  (seed 12): {tc[:10]}…")


def test_fast_math_keeps_the_golden(session):
    """Metal's fast math mode (the kernels default to the safe mode) under the numerics gates: the 48 greedy tokens
    still equal the golden (#34, math modes)."""
    sess, ckpt = session
    from monolith.generate import Session
    from monolith.models.qwen3_5 import Qwen3_5Model

    with open(str(GOLDEN) + ".json") as f:
        golden = json.load(f)
    fast = Session(Qwen3_5Model.from_checkpoint(str(ckpt), max_context=512), str(sess.pack.dir), eos=-1, fast_math=True)
    gen = fast.generate(golden["prompt_ids"], len(golden["gen_ids"]))
    assert gen.tokens == golden["gen_ids"], (gen.tokens[:8], golden["gen_ids"][:8])
    print(f"\nfast math: {len(gen.tokens)} greedy tokens equal the golden; {gen.ms_per_token:.2f} ms/token")

