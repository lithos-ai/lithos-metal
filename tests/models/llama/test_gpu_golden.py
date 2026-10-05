"""Real Llama 3.2 and SmolLM2 checkpoints: packed Metal execution vs HF goldens.

Run serially on a bare-metal Mac. Checkpoints are looked up under MONOLITH_MODELS
or ~/models; CPU-only CI skips this tier. Goldens use BF16-dequantized weights.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pytest

from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.runtime import is_available
from tests.oracle_conftest import require_checkpoint

CASES = [
    ('mlx-community-Llama-3.2-1B-Instruct-4bit', 'llama-3.2-1b-int4'),
    ('mlx-community-Llama-3.2-3B-Instruct-4bit', 'llama-3.2-3b-int4'),
    ('HuggingFaceTB-SmolLM2-1.7B-Instruct', 'smollm2-1.7b-bf16'),
]
pytestmark = pytest.mark.skipif(not is_available(), reason='Metal runtime is not built')


@pytest.fixture(scope='module', params=CASES, ids=[case[1] for case in CASES])
def session(request, tmp_path_factory):
    from monolith.generate import Session
    from monolith.models.llama import LlamaModel
    from monolith.nn.pack_plan import pack_model
    name, tag = request.param
    checkpoint = require_checkpoint(name)
    pack = tmp_path_factory.mktemp(tag)
    model = LlamaModel.from_checkpoint(str(checkpoint), max_context=4096)
    pack_model(model, str(checkpoint), str(pack), PackLayout(scale_placement='block'))
    sess = Session(model, str(pack), eos=-1)
    yield sess, tag
    del sess, model
    gc.collect()


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_prefill_layers_greedy_and_replay(session):
    sess, tag = session
    root = Path(__file__).parent / 'goldens' / tag
    golden = json.loads(Path(str(root) + '.json').read_text())
    tensors = SafetensorsDir(str(root) + '.safetensors')
    hidden = bf16_to_f32(tensors.get('hidden_states')).copy()
    logits = bf16_to_f32(tensors.get('logits_last')).copy()
    top_ids = tensors.get('tf_top_idx').copy()
    top_values = tensors.get('tf_top_val').copy()
    tensors.close()
    ids = golden['prompt_ids']
    sess.generate(ids, 1)
    p, h = len(ids), sess.model.config.hidden_size
    minimum = 1.
    for i in range(sess.model.n_layers):
        got = bf16_to_f32(np.frombuffer(sess.read(f'layers.{i}.mlp.h'), np.uint16)).reshape(-1, h)[:p]
        if i == sess.model.n_layers - 1:
            # HF's final hidden-state entry is after the final RMSNorm.
            normed = got * np.reciprocal(np.sqrt(np.mean(got * got, axis=-1, keepdims=True) + sess.model.config.rms_norm_eps))
            got = bf16_to_f32(f32_to_bf16(bf16_to_f32(f32_to_bf16(normed)) * sess.pack.aux_array('norm.weight')))
        ref = hidden[i + 1]
        similarity = cosine(got, ref)
        minimum = min(minimum, similarity)
        assert similarity > .999, (tag, i, similarity)
        assert np.max(np.abs(got - ref)) <= np.max(np.abs(ref)) / 32, (tag, i)
    actual_logits = bf16_to_f32(np.frombuffer(sess.read('logits'), np.uint16)).reshape(-1, sess.model.config.vocab_size)[p - 1]
    assert cosine(actual_logits, logits) > .999
    assert actual_logits.argmax() == logits.argmax() == golden['gen_ids'][0]
    first = sess.generate(ids, len(golden['gen_ids']))
    second = sess.generate(ids, len(golden['gen_ids']))
    assert first.tokens == second.tokens
    exact = first.tokens == golden['gen_ids']
    if not exact:
        # Design §5.9 permits exact logit ties. Check every reference continuation
        # position, not just the first divergence of the free-running sequence.
        mismatch = next(i for i, pair in enumerate(zip(first.tokens, golden['gen_ids'])) if pair[0] != pair[1])
        row = p - 1 + mismatch
        ties = top_ids[row][top_values[row] == top_values[row, 0]]
        assert first.tokens[mismatch] in ties and golden['gen_ids'][mismatch] in ties, (tag, mismatch)
        for i, reference in enumerate(golden['gen_ids']):
            got = sess.generate(ids + golden['gen_ids'][:i], 1).tokens[0]
            row = p - 1 + i
            ties = top_ids[row][top_values[row] == top_values[row, 0]]
            assert got == reference or (got in ties and reference in ties), (tag, i, got, reference)
        print(f'{tag}: first free-running divergence at +{mismatch}, exact HF logit tie; all continuation positions checked', flush=True)
    buffers = {id(b): b for engine in sess.engines.values() for b in engine.buffers.values()}
    allocated = sum(b.nbytes for b in buffers.values())
    assert allocated < sess.dev.info().recommended_working_set
    print(json.dumps(dict(model=tag, context_capacity=4096, pack_bytes=(sess.pack.dir / 'weights.pack').stat().st_size,
                         buffer_bytes=allocated, recommended_working_set=sess.dev.info().recommended_working_set,
                         prompt_tokens=p, generated_tokens=len(first.tokens), exact_free_running=exact, minimum_layer_cosine=minimum)), flush=True)
