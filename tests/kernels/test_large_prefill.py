"""Separate large-prefill/small-decode programs preserve sequence state across chunk boundaries."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contract'))
from test_nn_lowering import _checkpoint
from dspark_synth import write_checkpoint as write_dspark, build
from lm_synth import write_checkpoint as write_lm
from monolith.generate import Session
from monolith.formats import PackLayout
from monolith.models.qwen3_5 import Qwen3_5Model
from monolith.nn.pack_plan import pack_model
from monolith.spec.lm import LMDrafter


@pytest.mark.parametrize('draft_kind', [None, 'lm', 'dspark'])
@pytest.mark.parametrize('accelerator', ['off', 'on'])
@pytest.mark.parametrize('prefix_cache', [False, True])
@pytest.mark.parametrize('resident', [False, True])
def test_large_prefill_shares_state_with_small_decode(tmp_path, draft_kind, accelerator, prefix_cache, resident):
    if resident and draft_kind != 'dspark':
        pytest.skip('Resident verification prefill requires a speculative decoder recipe')
    target, draft = tmp_path / 'target', tmp_path / 'draft'
    target.mkdir(); draft.mkdir()
    _checkpoint(target)
    def model():
        return Qwen3_5Model.from_checkpoint(str(target), max_context=512)
    pack_model(model(), str(target), str(target / 'pack'), PackLayout(rows=16))
    reference = Session(model(), str(target / 'pack'), prefill_chunk_size=8, eos=-1, autotune=False,
                        accelerator='off', attention='v1')
    m = model()
    options = {}
    if draft_kind == 'lm':
        write_lm(draft, vocab_size=50)
        d = LMDrafter.from_checkpoint(str(draft), target_lm_head=m.lm_head, max_context=512, gamma=3)
    elif draft_kind == 'dspark':
        write_dspark(draft, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1])
        d, _, _, _ = build(draft, target_lm_head=m.lm_head, max_context=512)
    if draft_kind:
        pack_model(d, str(draft), str(draft / 'pack'), PackLayout(rows=16))
        options = dict(drafter=d, drafter_pack=str(draft / 'pack'), verify='fixed', verify_length=3)
    session = Session(m, str(target / 'pack'), eos=-1, autotune=False, attention='v1', accelerator=accelerator,
                      prefix_cache=prefix_cache, decoder_kernel_config={} if resident else None, **options)
    # Small-to-large allocation growth, exact boundary, partial chunk, several chunks, then reuse the small graph.
    for count in (5, 128, 129, 259, 5):
        ids = np.random.default_rng(count).integers(0, 50, count).tolist()
        expected = reference.generate(ids, 12).tokens
        result = session.generate(ids, 12)
        assert result.tokens == expected
        dec = session.engine(0 if draft_kind else 1)
        if not resident:
            pre = session.prefill_engine(count)
            assert pre is not dec
            assert pre.buffers['step_state'] is dec.buffers['step_state']
            for name, spec in dec.program.buffers.items():
                if spec.role in ('state', 'weights', 'ring'):
                    assert pre.buffers[name] is dec.buffers[name]
        assert dec.state()['position'] == count + 11
        assert dec.state()['error'] == 0
        vocab = m.config.vocab_size
        assert dec.program.buffers['logits'].nbytes == (8 if draft_kind else 1) * vocab * 2
        assert session.decode_t_max == 8
        assert session.prefill_chunk_size == 128
        if prefix_cache:
            # Reuse a checkpoint while replaying only the final input token.
            repeated = session.generate(ids, 12)
            assert repeated.tokens == result.tokens
            assert repeated.cached_prompt_tokens == count - 1
            if resident:
                assert session.engine(0) is dec
            changed = ids[:-1] + [(ids[-1] + 1) % 50]
            assert session.generate(changed, 12).tokens == reference.generate(changed, 12).tokens
    if resident:
        # A one-token answer ends before decode is allocated. The next short
        # request must not inherit either its state or original weight windows.
        session.generate(list(range(50)) * 3, 1)
        ids = [7, 3, 1, 8, 4]
        assert session.generate(ids, 12).tokens == reference.generate(ids, 12).tokens
        if prefix_cache:
            ids = list(range(50)) * 3
            session.prefix_cache.items.clear()
            session.generate(ids, 12, cache_prefix_tokens=120)
            # The prefill/decode boundary is not another cache checkpoint: it
            # would evict a large shared prefix under the byte budget.
            assert [item.tokens for item in session.prefix_cache.items] == [tuple(ids[:120])]
            changed = ids[:120] + [3, 7, 4, 2]
            result = session.generate(changed, 12, cache_prefix_tokens=120)
            assert result.cached_prompt_tokens == 120
            assert result.tokens == reference.generate(changed, 12).tokens
            # Preserve the earlier system/tool checkpoint when project context
            # inside the last message changes, while preferring the deep hit
            # for a new question with otherwise identical context.
            session.prefix_cache.items.clear()
            session.generate(ids, 12, cache_prefix_tokens=[100, 120])
            assert [len(item.tokens) for item in session.prefix_cache.items] == [100, 120]
            question = ids[:120] + [9, 8, 7]
            reused = session.generate(question, 12, cache_prefix_tokens=[100, 120])
            assert reused.cached_prompt_tokens == 120
            assert reused.tokens == reference.generate(question, 12).tokens
            project_changed = ids.copy()
            project_changed[110] = (project_changed[110] + 1) % 50
            reused = session.generate(project_changed, 12, cache_prefix_tokens=[100, 120])
            assert reused.cached_prompt_tokens == 100
            assert reused.tokens == reference.generate(project_changed, 12).tokens
