"""The package's frequency rule against Hugging Face, including long positions."""
import numpy as np
import pytest
from monolith.models.llama import LlamaConfig
from monolith.models.llama.rope import frequencies, tables
from tests.contract.test_llama_package import CFG, SCALING


@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('scaled', [False, True])
def test_rope_matches_transformers(dim, scaled):
    torch = pytest.importorskip('torch')
    from transformers import LlamaConfig as HFConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    options = dict(CFG, head_dim=dim)
    if scaled:
        options.update(rope_scaling=SCALING, max_position_embeddings=131072)
    config = LlamaConfig.from_dict(options)
    ref = LlamaRotaryEmbedding(HFConfig(**options))
    np.testing.assert_allclose(frequencies(config), ref.inv_freq.numpy(), rtol=2e-7)
    pos = np.array([0, 1, 127, 4095, 8191, 16384, 65535], np.int64)
    got = tables(config, pos)
    expected = ref(torch.zeros(1, dtype=torch.float32), torch.from_numpy(pos)[None])
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b[0].numpy(), atol=.001)
