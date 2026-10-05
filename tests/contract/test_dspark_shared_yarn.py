"""Shared target embeddings and checkpoint-defined YaRN, including amplitude."""
from types import SimpleNamespace

import numpy as np
import pytest

from monolith.core import DType, Graph, T
from monolith.nn import Embedding, LMHead
from monolith.nn.rope import scaled_inv_freq, rope_tables_hf, rope_tables_permuted
from monolith.spec import DraftContext
from monolith.spec.dspark import DSparkConfig, DSparkDrafter
CFG = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=50,
           rope_theta=1e6, block_size=7, target_layer_ids=[1, 3, 5, 7, 9], mask_token_id=49,
           markov_rank=8, markov_head_type="vanilla", enable_confidence_head=True,
           confidence_head_with_markov=True)


@pytest.mark.parametrize("overrides", [{}, {"truncate": False}, {"attention_factor": 1.2},
                                     {"mscale": 0.8, "mscale_all_dim": 0.5}])
def test_yarn_matches_hf(overrides):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3Config
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    params = dict(rope_type="yarn", rope_theta=1e7, factor=32.0,
                  original_max_position_embeddings=8192, beta_fast=32.0, beta_slow=1.0, **overrides)
    cfg = Qwen3Config(hidden_size=5120, head_dim=128, num_attention_heads=32,
                      max_position_embeddings=262144, rope_parameters=params)
    ref, amplitude = _compute_yarn_parameters(cfg, torch.device("cpu"))
    actual, scale = scaled_inv_freq(1e7, 128, params)
    np.testing.assert_allclose(actual, ref.numpy(), rtol=2e-7)
    assert scale == pytest.approx(amplitude)
    cos, sin = rope_tables_hf(1e7, 128, 32769, parameters=params)
    np.testing.assert_allclose(cos[0], scale)
    phase = np.outer(np.array([128, 4096, 8192, 16384, 32768], np.float32), ref.numpy())
    np.testing.assert_allclose(cos[[128, 4096, 8192, 16384, 32768], :64], np.cos(phase) * amplitude, atol=3e-4)
    packed = rope_tables_permuted(1e7, 128, 128, 32769, parameters=params)
    np.testing.assert_array_equal(packed[0], cos)
    np.testing.assert_array_equal(packed[1], sin)


def test_shared_embedding_is_external_and_reused_in_ir():
    head = LMHead(64, 50, hf_name="lm_head.weight", prefix="lm_head.")
    draft = DSparkDrafter(DSparkConfig.from_dict(CFG), target_lm_head=head, shared_embedding=True, max_context=32)
    with pytest.raises(ValueError, match="bind_target"):
        draft.embedding()
    embedding = Embedding(50, 64, "target.embed_tokens.weight", prefix="embed_tokens.")
    draft.bind_target(SimpleNamespace(embed_tokens=embedding))
    assert draft.embedding() is embedding
    assert all("embed_tokens" not in k for k in draft.full_weight_map())
    graph = Graph("shared")
    taps = [graph.input(f"tap.{i}", (T, 64), DType.BF16) for i in range(5)]
    draft.lower_draft(graph, DraftContext(taps))
    graph.check()
    assert "embed_tokens.weight" in graph.values
    assert "draft.embed_tokens.weight" not in graph.values
    with pytest.raises(ValueError, match="shape"):
        draft.bind_target(SimpleNamespace(embed_tokens=Embedding(51, 64, "x", prefix="other.")))
