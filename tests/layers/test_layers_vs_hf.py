"""Composite layer oracles on real weights against the HF modules (plan M3 bars: cos > 0.999, bounded max-abs):
one Gated-DeltaNet layer, one attention layer and one MLP of the small checkpoint, fresh-state prefill (T = 4)
and a T = 1 continuation. The HF side runs its own code paths (chunked delta rule + conv for prefill, the recurrent
rule + conv update for the continuation); ours runs the recurrent oracle for both."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_conftest import require_checkpoint, require_torch  # noqa: E402

CKPT = "Qwen3.5-0.8B"
TOL = 1 / 64


def _bars(got, ref):
    got, ref = got.float().numpy().astype(np.float64).ravel(), ref.float().numpy().astype(np.float64).ravel()
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    scale = float(np.abs(ref).max())
    max_abs = float(np.abs(got - ref).max())
    ok = cos > 0.999 and max_abs <= TOL * scale
    return f"cos={cos:.6f} max_abs={max_abs:.5f} scale={scale:.3f} {'ok' if ok else 'FAIL'}", ok


@pytest.fixture(scope="module")
def setup():
    torch = require_torch()
    transformers = pytest.importorskip("transformers")
    ckpt = require_checkpoint(CKPT)
    from transformers import AutoConfig

    from monolith.formats.safetensors_reader import SafetensorsDir
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.models.qwen3_5.weights import TEXT_PREFIX, load_oracle

    cfg = AutoConfig.from_pretrained(str(ckpt)).text_config
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3_5Model.from_checkpoint(str(ckpt), max_context=64, num_layers_override=4)
    load_oracle(model, str(ckpt))
    st = SafetensorsDir(str(ckpt))

    def hf_load(module, hf_prefix):
        sd = {}
        for name in st.names():
            if name.startswith(hf_prefix):
                arr = st.get(name)
                t = torch.from_numpy(arr.view(np.int16).copy()).view(torch.bfloat16) if st.info(name).dtype == "BF16" else torch.from_numpy(np.array(arr))
                sd[name[len(hf_prefix):]] = t
        module.to(torch.bfloat16)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        return module.eval()

    return torch, transformers, cfg, model, hf_load, TEXT_PREFIX


def _x(torch, t, h):
    return (torch.randn(t, h) * 1.0).to(torch.bfloat16)


def test_mlp_and_norm(setup):
    torch, tr, cfg, model, hf_load, P = setup
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

    blk = model.blocks[0]
    hf_mlp = hf_load(Qwen3_5MLP(cfg, cfg.intermediate_size), f"{P}layers.0.mlp.")
    hf_norm = hf_load(Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), f"{P}layers.0.post_attention_layernorm.")
    h = _x(torch, 4, cfg.hidden_size)
    with torch.no_grad():
        ref = h + hf_mlp(hf_norm(h))
        got = blk.mlp.forward(blk.post_norm.forward(h), h)
        n_ref, n_got = hf_norm(h), blk.post_norm.forward(h)
    s1, ok1 = _bars(n_got, n_ref)
    s2, ok2 = _bars(got, ref)
    print(f"norm: {s1}\nmlp+residual: {s2}")
    assert ok1 and ok2


def test_gdn_layer(setup):
    torch, tr, cfg, model, hf_load, P = setup
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    blk = model.blocks[0]
    hf = hf_load(Qwen3_5GatedDeltaNet(cfg, layer_idx=0), f"{P}layers.0.linear_attn.")
    x = _x(torch, 4, cfg.hidden_size)
    x1 = _x(torch, 1, cfg.hidden_size)
    state = model.init_state()
    with torch.no_grad():
        ref = hf(x[None])[0]                                             # fresh state, chunked path
        got = blk.mixer.forward(x, torch.zeros_like(x), state, 0)        # residual 0 → the mixer output alone
        cache = DynamicCache(config=cfg)
        hf(x[None], cache_params=cache)
        ref1 = hf(x1[None], cache_params=cache)[0]                       # continuation: recurrent path
        got1 = blk.mixer.forward(x1, torch.zeros_like(x1), state, 4)
        hf_rec = cache.layers[0].recurrent_states[0][0] if hasattr(cache.layers[0], "recurrent_states") else None
    s, ok = _bars(got, ref)
    s1, ok1 = _bars(got1, ref1)
    print(f"gdn prefill T=4: {s}\ngdn continuation T=1: {s1}")
    if hf_rec is not None:
        ours_rec = state["layers.0.linear_attn.rec_state"]
        d = (ours_rec - hf_rec.to(ours_rec.dtype)).abs().max().item()
        print(f"recurrent state max-abs diff vs HF: {d:.3e} (scale {hf_rec.abs().max().item():.3e})")
    assert ok and ok1


def test_attention_layer(setup):
    torch, tr, cfg, model, hf_load, P = setup
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

    idx = next(i for i in range(len(model.blocks)) if not model.config.is_linear(i))
    blk = model.blocks[idx]
    hf = hf_load(Qwen3_5Attention(cfg, layer_idx=idx), f"{P}layers.{idx}.self_attn.")
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    x = _x(torch, 4, cfg.hidden_size)
    x1 = _x(torch, 1, cfg.hidden_size)
    state = model.init_state()

    def emb(pos0, t):
        pid = (torch.arange(pos0, pos0 + t)[None, None, :]).expand(3, 1, t)
        return rot(x[None], pid)

    with torch.no_grad():
        mask = torch.full((4, 4), float("-inf")).triu(1)[None, None]
        cache = DynamicCache(config=cfg)
        ref, _ = hf(x[None], position_embeddings=emb(0, 4), attention_mask=mask, past_key_values=cache)
        ref1, _ = hf(x1[None], position_embeddings=emb(4, 1), attention_mask=None, past_key_values=cache)
        got = blk.mixer.forward(x, torch.zeros_like(x), state, 0)
        got1 = blk.mixer.forward(x1, torch.zeros_like(x1), state, 4)
    s, ok = _bars(got, ref[0])
    s1, ok1 = _bars(got1, ref1[0])
    print(f"attention prefill T=4: {s}\nattention continuation T=1: {s1}")
    assert ok and ok1
