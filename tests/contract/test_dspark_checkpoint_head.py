"""A frozen draft head must not be replaced with differently quantized target weights."""
import numpy as np
import pytest

from tests.dspark_synth import write_checkpoint
from monolith.core import DType, Graph, T
from monolith.formats import PackLayout
from monolith.nn import LMHead
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile
from monolith.spec import DraftContext
from monolith.spec.dspark import DSparkDrafter
from monolith.serving.cache import ensure_pack


@pytest.mark.parametrize('with_head', [False, True])
def test_checkpoint_head_or_target_fallback(tmp_path, with_head):
    held = write_checkpoint(tmp_path, with_head=with_head, block_size=8)
    target = LMHead(256, 64, hf_name='target.weight', prefix='target.head.')
    draft = DSparkDrafter.from_checkpoint(str(tmp_path), target_lm_head=target, max_context=32)
    assert draft.gamma == 8
    assert ('lm_head.weight' in draft.full_weight_map()) == with_head
    assert (draft._lm_head is target) == (not with_head)
    graph = Graph('head')
    taps = [graph.input(f'tap.{i}', (T, 256), DType.BF16) for i in range(2)]
    draft.lower_draft(graph, DraftContext(taps))
    graph.check()
    head_op = next(op for op in graph.ops if op.kind == 'lm_head')
    assert head_op.inputs[1].name == ('draft.lm_head.weight' if with_head else 'target.head.weight')
    if with_head:
        pack_model(draft, str(tmp_path), str(tmp_path/'pack'), PackLayout())
        np.testing.assert_array_equal(PackFile(tmp_path/'pack').dequantize_slab('draft.lm_head.weight'), held['lm_head.weight'])
    short = DSparkDrafter.from_checkpoint(str(tmp_path), target_lm_head=target, block_size=7)
    assert short.gamma == 7 and short.cfg.block_size == 7 and draft.cfg.block_size == 8
    for bad in (0, 9, True, 1.5):
        with pytest.raises(ValueError, match='block_size'):
            DSparkDrafter.from_checkpoint(str(tmp_path), target_lm_head=target, block_size=bad)


def test_cache_rebuilds_when_checkpoint_head_was_previously_omitted(tmp_path):
    source = tmp_path / 'model'
    source.mkdir()
    write_checkpoint(source, with_head=True, block_size=8)
    target = LMHead(256, 64, hf_name='target.weight', prefix='target.head.')
    old = DSparkDrafter.from_checkpoint(str(source), target_lm_head=target, checkpoint_lm_head=False, max_context=32)
    current = DSparkDrafter.from_checkpoint(str(source), target_lm_head=target, max_context=32)
    options = dict(capacity=32, layout=PackLayout(), backend='m5_max_40c', role='draft')
    previous = ensure_pack(source, old, tmp_path / 'cache', **options)
    rebuilt = ensure_pack(source, current, tmp_path / 'cache', **options)
    assert rebuilt != previous
    assert ensure_pack(source, current, tmp_path / 'cache', **options) == rebuilt
    assert 'draft.lm_head.weight' in PackFile(rebuilt).slabs
