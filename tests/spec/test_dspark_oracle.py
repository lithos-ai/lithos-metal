"""The DSpark drafter oracle against DeepSpec's reference implementation (MIT) on the public Qwen3-8B drafter
(`Dogacel/Qwen3-8B-DSpark`): the feature projection, one draft block (context injection + bidirectional block
attention), the Markov bias and the confidence head. Needs torch, the drafter and target checkpoints under ~/models
and DeepSpec on the path (`MONOLITH_DEEPSPEC` or ~/lithos/DeepSpec); skips otherwise."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_conftest import require_checkpoint, require_torch  # noqa: E402

DRAFTER = "Dogacel-Qwen3-8B-DSpark"
TARGET = "nvidia-Qwen3-8B-NVFP4"


def _bars(got, ref):
    got, ref = got.float().numpy().astype(np.float64).ravel(), ref.float().numpy().astype(np.float64).ravel()
    return float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30)), float(np.abs(got - ref).max()), float(np.abs(ref).max())


@pytest.fixture(scope="module")
def models():
    torch = require_torch()
    ddir, tdir = require_checkpoint(DRAFTER), require_checkpoint(TARGET)
    deepspec = os.environ.get("MONOLITH_DEEPSPEC", os.path.expanduser("~/lithos/DeepSpec"))
    if not os.path.isdir(deepspec):
        pytest.skip("DeepSpec reference not found")
    sys.path.insert(0, deepspec)
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402

    from monolith.formats.fp import bf16_to_f32
    from monolith.formats.safetensors_reader import SafetensorsDir
    from monolith.nn import LMHead
    from monolith.spec.dspark import DSparkConfig, DSparkDrafter
    from monolith.spec.dspark.weights import bind_checkpoint_formats, load_oracle

    cfg = AutoConfig.from_pretrained(str(ddir))
    raw = json.load(open(ddir / "config.json"))
    for k in ("target_layer_ids", "mask_token_id", "markov_rank", "markov_head_type", "enable_confidence_head", "confidence_head_with_markov", "num_target_layers"):
        cfg.__dict__[k] = raw[k]
    cfg.__dict__["block_size"] = raw.get("block_size", 7)          # the TorchSpec checkpoint omits it (block 7 per its README)
    cfg.__dict__["num_anchors"] = 512
    cfg._attn_implementation = "eager"
    ref = Qwen3DSparkModel(cfg)
    ref.load_state_dict(load_file(str(ddir / "model.safetensors")), strict=False)
    ref = ref.to(torch.bfloat16).eval()
    # the target's lm_head (BF16 in the NVFP4 checkpoint) serves both the reference and ours
    st = SafetensorsDir(str(tdir))
    lm = torch.from_numpy(bf16_to_f32(st.get("lm_head.weight"))).to(torch.bfloat16)
    st.close()
    with torch.no_grad():
        ref.lm_head.weight.copy_(lm)
    dcfg = DSparkConfig.from_pretrained(str(ddir))
    head = LMHead(dcfg.hidden_size, dcfg.vocab_size, hf_name="lm_head.weight", prefix="lm_head.")
    head.set_param("weight", lm)
    ours = DSparkDrafter(dcfg, target_lm_head=head, max_context=64)
    bind_checkpoint_formats(ours, str(ddir))
    load_oracle(ours, str(ddir))
    return torch, ref, ours, dcfg


def test_features_block_markov_confidence(models):
    torch, ref, ours, cfg = models
    from transformers import DynamicCache

    torch.manual_seed(0)
    ctx_new, block = 6, cfg.block_size
    taps = (torch.randn(1, ctx_new, cfg.n_taps * cfg.target_hidden) * 3).to(torch.bfloat16)
    anchor = 1234
    ids = torch.full((1, block), cfg.mask_token_id, dtype=torch.long)
    ids[0, 0] = anchor
    with torch.no_grad():
        cache = DynamicCache()
        r_hidden = ref._forward_backbone(target_hidden_states=taps, noise_embedding=ref.embed_tokens(ids),
                                         position_ids=torch.arange(0, ctx_new + block)[None], attention_mask=None,
                                         past_key_values=cache, use_cache=True, is_causal=False)[0]
        r_feats = ref.hidden_norm(ref.fc(taps))[0]
        prev = torch.tensor([[anchor, 5, 6, 7, 8, 9, 10]])
        r_conf = torch.sigmoid(ref.predict_confidence_step(r_hidden[None], prev_token_ids=prev))[0]
        r_bias = ref.markov_head.compute_step_bias(torch.tensor([anchor]), None)[0]
        r_logits = ref.lm_head(r_hidden)
        r_tokens, _ = ref.sample_draft_tokens(r_logits[None], first_prev_token_ids=torch.tensor([anchor]), temperature=0.0)
        # ours
        state = {e.name: torch.zeros(e.shape, dtype=torch.bfloat16) for e in ours.state_entries()}
        o_feats = ours.project_features(taps[0])
        o_hidden = ours.draft_block(anchor, o_feats, state, ctx_len=0)
        o_tokens, _ = ours.draft_tokens(o_hidden, anchor)
        o_conf = ours.confidences(o_hidden, prev[0].tolist())
        o_bias = ours.markov_bias(anchor)
    for name, got, want, tol in (("features", o_feats, r_feats, 1 / 64), ("block hidden", o_hidden, r_hidden, 1 / 32), ("markov bias", o_bias, r_bias, 1 / 64)):
        cos, max_abs, scale = _bars(got, want)
        print(f"{name}: cos={cos:.6f} max_abs={max_abs:.4f} scale={scale:.3f}")
        assert cos > 0.999 and max_abs <= tol * scale, name
    print("confidence ours", [round(float(x), 4) for x in o_conf], "ref", [round(float(x), 4) for x in r_conf])
    assert torch.allclose(o_conf, r_conf.float(), atol=2e-2)
    print("drafts ours", o_tokens, "ref", r_tokens[0].tolist())
    assert o_tokens == r_tokens[0].tolist()
    # the context caches hold the injected positions: layer 0's keys equal the reference cache's first ctx_new rows
    k0 = (cache.layers[0].keys if hasattr(cache, "layers") else cache.key_cache[0])[0].transpose(0, 1)[:ctx_new]   # [ctx_new, kv, D]
    cos, max_abs, scale = _bars(state["draft.layers.0.self_attn.k_ctx"][:ctx_new], k0)
    print(f"context keys: cos={cos:.6f} max_abs={max_abs:.4f} scale={scale:.3f}")
    assert cos > 0.999
