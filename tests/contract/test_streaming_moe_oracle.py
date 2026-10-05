"""Streaming changes parameter storage without changing HF eager results."""
import pytest
from tests.moe_synth import write_checkpoint


@pytest.mark.parametrize('tied', [False, True])
def test_streamed_experts_equal_materialized_hf(tmp_path, tied):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from tools.goldens.moe_streaming_hf import load_streamed
    held = write_checkpoint(tmp_path, model_type='qwen3_moe', tie_word_embeddings=tied)
    streamed, checkpoint = load_streamed(str(tmp_path))
    dense, dense_checkpoint = load_streamed(str(tmp_path))
    try:
        assert not any(p.is_meta for p in streamed.parameters())
        if tied:
            assert streamed.lm_head.weight is streamed.model.embed_tokens.weight
        expert = dense.model.layers[0].mlp.experts
        del expert.gate_up_proj, expert.down_proj

        def weight(name):
            return torch.from_numpy(held[name]).to(torch.bfloat16)

        gate_up = [torch.cat([
            weight(f'model.layers.0.mlp.experts.{e}.{projection}.weight')
            for projection in ('gate_proj', 'up_proj')
        ]) for e in range(8)]
        down = [weight(f'model.layers.0.mlp.experts.{e}.down_proj.weight') for e in range(8)]
        expert.gate_up_proj = torch.nn.Parameter(torch.stack(gate_up), requires_grad=False)
        expert.down_proj = torch.nn.Parameter(torch.stack(down), requires_grad=False)
        streamed_cache = dense_cache = None
        with torch.inference_mode():
            for ids in ([1, 7, 21, 100], [11], [44]):
                a = streamed(torch.tensor([ids]), past_key_values=streamed_cache, use_cache=True)
                b = dense(torch.tensor([ids]), past_key_values=dense_cache, use_cache=True)
                assert torch.isfinite(a.logits).all()
                assert torch.equal(a.logits, b.logits)
                streamed_cache, dense_cache = a.past_key_values, b.past_key_values
    finally:
        checkpoint.close()
        dense_checkpoint.close()


def test_streamed_hybrid_equal_materialized_hf(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from tests.hybrid_moe_synth import P, write_checkpoint
    from tools.goldens.moe_streaming_hf import load_streamed

    held = write_checkpoint(tmp_path)
    streamed, checkpoint = load_streamed(str(tmp_path))
    dense, dense_checkpoint = load_streamed(str(tmp_path))
    try:
        for i, layer in enumerate(dense.model.layers):
            expert = layer.mlp.experts
            del expert.gate_up_proj, expert.down_proj
            def weight(e, proj):
                return torch.from_numpy(held[f'{P}layers.{i}.mlp.experts.{e}.{proj}.weight']).to(torch.bfloat16)
            expert.gate_up_proj = torch.nn.Parameter(torch.stack([
                torch.cat([weight(e, 'gate_proj'), weight(e, 'up_proj')]) for e in range(8)]), requires_grad=False)
            expert.down_proj = torch.nn.Parameter(torch.stack([weight(e, 'down_proj') for e in range(8)]), requires_grad=False)
        streamed_cache = dense_cache = None
        with torch.inference_mode():
            for ids in ([1, 7, 21, 100], [11], [44]):
                a = streamed(torch.tensor([ids]), past_key_values=streamed_cache, use_cache=True)
                b = dense(torch.tensor([ids]), past_key_values=dense_cache, use_cache=True)
                assert torch.isfinite(a.logits).all() and torch.equal(a.logits, b.logits)
                streamed_cache, dense_cache = a.past_key_values, b.past_key_values
    finally:
        checkpoint.close()
        dense_checkpoint.close()
