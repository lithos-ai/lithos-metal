"""Format 2 on a model (#47): the 0.8B converted by ``mlx_lm.convert`` to affine 4-bit groups (64, BF16 for the rest,
MLX's tensor naming), packed through the package's name map, decoded by the engine — the embedding gathered from the
quantized slab, the tied head and every projection through the affine GEMV — against the torch oracle run on the same
dequantized weights: 32 greedy tokens equal. Needs mlx-lm and the checkpoint under ~/models (the conversion runs into
a temporary directory, ~1 minute); skips otherwise."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle_conftest import require_checkpoint, require_torch  # noqa: E402

from monolith.formats import PackLayout
from monolith.runtime import is_available

CKPT = "Qwen3.5-0.8B"

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


@pytest.fixture(scope="module")
def mlx_dir(tmp_path_factory):
    pytest.importorskip("mlx_lm")
    src = require_checkpoint(CKPT)
    out = tmp_path_factory.mktemp("mlx4")
    r = subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(src), "--mlx-path", str(out / "m"), "-q", "--q-bits", "4",
                        "--q-group-size", "64", "--dtype", "bfloat16"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"mlx_lm convert failed: {r.stderr[-400:]}")
    return out / "m"


def test_mlx_int4_model_matches_its_oracle(mlx_dir, tmp_path):
    torch = require_torch()
    from monolith.generate import Session
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.models.qwen3_5.weights import load_oracle
    from monolith.nn.pack_plan import pack_model

    model = Qwen3_5Model.from_checkpoint(str(mlx_dir), max_context=128)
    assert model.checkpoint_rename is not None and model.checkpoint_adapt is not None
    assert model.embed_tokens.format_of("weight") == "int4_affine"
    assert model.blocks[0].mixer.in_proj.format_of("in_proj_qkv") == "int4_affine" and model.lm_head.tied is model.embed_tokens
    pack_model(model, str(mlx_dir), str(tmp_path / "pack"), PackLayout())
    sess = Session(model, str(tmp_path / "pack"), eos=-1)
    ids = [760, 6511, 314, 9338, 369]                                       # "The capital of France is" (the 248k-vocab tokenizer)
    n = 32
    gen = sess.generate(ids, n)
    # the oracle on the dequantized weights, greedy, token by token through the state
    load_oracle(model, str(mlx_dir))
    state = model.init_state()
    with torch.no_grad():
        logits, _, state = model.forward(torch.tensor(ids), state, 0)
        seq = [int(torch.argmax(logits[-1].float()))]
        pos = len(ids)
        while len(seq) < n:
            logits, _, state = model.forward(torch.tensor(seq[-1:]), state, pos)
            seq.append(int(torch.argmax(logits[-1].float())))
            pos += 1
    assert gen.tokens == seq, (gen.tokens[:8], seq[:8])
    assert gen.tokens[0] == 11751                                            # " Paris" — the norm convention is right
    print(f"\nint4_affine 0.8B: {n} greedy tokens equal the oracle's; {gen.ms_per_token:.2f} ms/token GPU")
