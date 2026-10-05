"""The DSpark drafter module (plan M6, #37) without torch or a checkpoint: config parsing (DeepSpec and gittensor
layouts), the weight map against the drafter's tensor inventory, state entries and tables."""

import json

import numpy as np
import pytest

from monolith.nn import LMHead
from monolith.spec import DRAFTERS
from monolith.spec.dspark import DSparkConfig, DSparkDrafter

CFG = {"architectures": ["Qwen3DSparkModel"], "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
       "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 50, "rope_theta": 1e6, "block_size": 7,
       "target_layer_ids": [1, 3, 5, 7, 9], "mask_token_id": 49, "markov_rank": 8, "markov_head_type": "vanilla",
       "enable_confidence_head": True, "confidence_head_with_markov": True, "num_target_layers": 12}


def _drafter(cfg=None):
    c = DSparkConfig.from_dict(cfg or CFG)
    head = LMHead(64, 50, hf_name="lm_head.weight", prefix="lm_head.")
    return DSparkDrafter(c, target_lm_head=head, max_context=32), c


def test_config_layouts():
    c = DSparkConfig.from_dict(CFG)
    assert c.gamma if False else c.block_size == 7 and c.n_taps == 5 and c.target_hidden == 64 and c.vocab_size == 50
    git = dict(CFG)
    for k in ("block_size", "target_layer_ids", "mask_token_id", "markov_rank", "markov_head_type", "enable_confidence_head", "confidence_head_with_markov"):
        git.pop(k)
    git["dflash_config"] = {"target_layer_ids": [4, 16, 28], "mask_token_id": 7, "markov_rank": 256, "markov_head_type": "vanilla",
                            "enable_confidence_head": True, "confidence_head_with_markov": True}
    git["block_size"] = 7
    c2 = DSparkConfig.from_dict(git)
    assert c2.target_layer_ids == [4, 16, 28] and c2.mask_token_id == 7 and c2.markov_rank == 256
    with pytest.raises(ValueError):
        DSparkConfig.from_dict(dict(CFG, rope_parameters={"rope_type": "yarn", "rope_theta": 1e7}))


def test_weight_map_matches_the_drafter_inventory():
    d, c = _drafter()
    names = {k.split("#")[0] for k in d.full_weight_map()}
    expected = {"embed_tokens.weight", "fc.weight", "hidden_norm.weight", "norm.weight", "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight", "confidence_head.proj.weight", "confidence_head.proj.bias"}
    for i in range(2):
        L = f"layers.{i}."
        expected |= {L + "input_layernorm.weight", L + "post_attention_layernorm.weight", L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight",
                     L + "self_attn.v_proj.weight", L + "self_attn.o_proj.weight", L + "self_attn.q_norm.weight", L + "self_attn.k_norm.weight",
                     L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight", L + "mlp.down_proj.weight"}
    assert names == expected                                           # no lm_head: the target's is used
    dup = [k for k in d.full_weight_map() if "#" in k]
    assert len(dup) == 4 and all(k.split("#")[0].endswith(("k_proj.weight", "v_proj.weight")) for k in dup)   # k/v claimed twice (block + context)
    assert d.gamma == 7 and DRAFTERS.get("dspark") is DSparkDrafter
    assert [e.name for e in d.state_entries()] == ["draft.layers.0.self_attn.k_ctx", "draft.layers.0.self_attn.v_ctx",
                                                   "draft.layers.1.self_attn.k_ctx", "draft.layers.1.self_attn.v_ctx"]
    assert d.state_entries()[0].shape == (32, 2, 16) and set(d.tables()) == {"draft_rope_cos", "draft_rope_sin"}
    assert d.fc.k == 5 * 64 and d.markov_w2.n == 50 and d.weight_map()["conf_w"].shape == (1, 64 + 8)
