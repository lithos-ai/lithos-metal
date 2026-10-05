"""Hybrid MoE composition on GPU vs independent HF eager code, including shared experts."""
import numpy as np
import pytest

from tests.hybrid_moe_synth import write_checkpoint
from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32
from monolith.models.qwen3_5_moe import Qwen3_5MoeModel
from monolith.nn.pack_plan import pack_model
from monolith.runtime import is_available

pytestmark = pytest.mark.skipif(not is_available(), reason='Metal runtime required')


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize('quantized', [False, True])
def test_hybrid_moe_prefill_and_decode(tmp_path, quantized):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from monolith.generate import Session
    from tools.goldens.moe_streaming_hf import load_streamed

    write_checkpoint(tmp_path, quantized=quantized)
    hf, checkpoint = load_streamed(str(tmp_path))
    m = Qwen3_5MoeModel.from_checkpoint(str(tmp_path), max_context=256)
    pack = tmp_path / 'pack'
    pack_model(m, str(tmp_path), str(pack), PackLayout(scale_placement='block'))
    sess = Session(m, str(pack), eos=-1, autotune=False)
    ids = [1, 7, 21, 100, 5, 8, 13, 2]
    layer_outputs = []
    handles = [layer.register_forward_hook(lambda _m, _i, out: layer_outputs.append(out.detach().clone()))
               for layer in hf.model.layers]
    try:
        with torch.inference_mode():
            ref = hf(torch.tensor([ids]), use_cache=True, output_hidden_states=True)
            for handle in handles:
                handle.remove()
            tokens = [int(ref.logits[0, -1].argmax())]
            cache = ref.past_key_values
            for _ in range(7):
                nxt = hf(torch.tensor([[tokens[-1]]]), past_key_values=cache, use_cache=True)
                cache = nxt.past_key_values
                tokens.append(int(nxt.logits[0, -1].argmax()))
        first = sess.generate(ids, 1)
        assert first.tokens == tokens[:1]
        for i in range(m.n_layers):
            got = bf16_to_f32(np.frombuffer(sess.read(f'layers.{i}.mlp.h'), dtype=np.uint16)).reshape(-1, m.config.hidden_size)[:len(ids)]
            cos = cosine(got, layer_outputs[i][0].float().numpy())
            assert cos > .999, (i, cos)
        got = bf16_to_f32(np.frombuffer(sess.read('logits'), dtype=np.uint16)).reshape(-1, m.config.vocab_size)[:len(ids)]
        assert cosine(got, ref.logits[0].float().numpy()) > .999
        runs = [sess.generate(ids, 8).tokens for _ in range(2)]
        assert runs[0] == runs[1] == tokens
    finally:
        for handle in handles:
            handle.remove()
        checkpoint.close()
