"""Constant propagation must inspect the scale table, including bit-level edge cases."""
import json

import numpy as np
import pytest

from monolith.packs import PackFile


@pytest.mark.parametrize('values,expected', [
    ([1., 1.], 0x3F800000), ([.375, .375], 0x3EC00000),
    ([-0., -0.], 0x80000000), ([0., -0.], None),
    ([1., 2.], None), ([float('inf')], None), ([float('nan')], None), ([], None),
])
def test_uniform_scale_reads_actual_finite_bits(tmp_path, values, expected):
    data = np.asarray(values, np.float32).tobytes()
    (tmp_path / 'weights.pack').write_bytes(data or bytes(16))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'version': 2, 'pack': 'weights.pack', 'aux': [],
        'slabs': [{'name': 'w', 'n': len(values), 'row_scales_offset': 0}],
    }))
    assert PackFile(tmp_path).uniform_row_scale_bits('w') == expected


def test_compiler_keeps_loads_for_nonuniform_scale_table(tmp_path):
    from tests.contract.test_nn_lowering import _checkpoint
    from monolith.compiler import compile_program
    from monolith.core import Profile
    from monolith.formats import PackLayout
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.nn.pack_plan import pack_model

    _checkpoint(tmp_path)
    model = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=32)
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    pack = PackFile(tmp_path / 'pack')
    profile = Profile.from_dict('constant_scales', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16', 'accelerator': 'on'}})
    before = compile_program(model, pack, profile, t=8)
    op = next(op for op in before.ops if before.kernels[op.kernel].function == 'gemm_tile'
              and op.meta.get('format') == 'bf16')
    assert before.kernels[op.kernel].macros['ROW_SCALE_BITS'] == '1065353216u'
    slab = pack.slabs[op.name.split(':', 1)[1]]
    assert slab['n'] > 1
    with (pack.dir / pack.manifest['pack']).open('r+b') as f:
        f.seek(slab['row_scales_offset'] + 4)
        f.write(np.float32(2).tobytes())
    after = compile_program(model, pack, profile, t=8)
    changed = next(item for item in after.ops if item.name == op.name)
    assert 'ROW_SCALE_BITS' not in after.kernels[changed.kernel].macros


def test_nvfp4_fixed_and_dynamic_statistic_layouts(tmp_path):
    from dataclasses import asdict
    from monolith.bench import pack_spec, random_spec
    from monolith.compiler import emit_program
    from monolith.core import BlockDomain, DType, Graph, OpClass, Profile, StepStateLayout, T
    from monolith.formats import PackLayout

    data, info, _ = pack_spec(random_spec('nvfp4', 96, 4096, np.random.default_rng(7)),
                              PackLayout(rows=16, scale_placement='block'))
    scales = np.ones(96, np.float32)
    packed = data + scales.tobytes()
    packed += bytes(-len(packed) % 16384)
    (tmp_path / 'weights.pack').write_bytes(packed)
    entry = dict(asdict(info), name='w', offset=0, nbytes=len(data), row_scales_offset=len(data))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'version': 2, 'pack': 'weights.pack', 'nbytes': len(packed), 'slabs': [entry], 'aux': []}))
    g = Graph('row_groups')
    x = g.input('x', (T, 4096), DType.BF16)
    w = g.weight('w', (96, 4096), 'nvfp4')
    y = g.value('y', (T, 96), DType.BF16)
    g.value('stat', (T,), DType.F32)
    g.op('gemv', [x, w], [y], domain=BlockDomain('rows', 6), klass=OpClass.MAP, stat_value='stat')
    profile = Profile.from_dict('row_groups', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16', 'accelerator': 'on'}})
    for dynamic, tokens in ((False, 1), (True, 1), (False, 4)):
        p = emit_program(g, pack=PackFile(tmp_path), profile=profile, t=tokens,
                         dynamic_t=dynamic, layout=StepStateLayout(t_max=tokens, gamma_max=tokens - 1), tail=None)
        op = next(op for op in p.ops if op.name == 'gemv:w')
        macros = p.kernels[op.kernel].macros
        assert ('STATIC_GEMM_P_T_ACTIVE' in macros) != dynamic
        if tokens == 1:
            assert op.threadgroup == (256, 1, 1)
            assert op.grid == ((6 if dynamic else 12), 1, 1)
            assert p.buffers['stat'].nbytes == (24 if dynamic else 48)
            assert macros.get('NV_ROWS') == (None if dynamic else '1u')
        else:
            assert macros['COMPACT_PARTIALS'] == '1'
            assert macros['TK'] == '128u'
            assert p.buffers['stat'].nbytes == 4 * 6 * 4
        perm = next(op for op in p.ops if op.name.startswith('x_permute:'))
        groups, simdgroups = (4, 16) if dynamic else (1, 64) if tokens == 1 else (2, 128)
        assert perm.threadgroup == (32 * groups, 1, 1)
        assert perm.grid == (8 * simdgroups // groups, 1, 1)
        pm = p.kernels[perm.kernel].macros
        assert pm['PERM_GROUPS'] == f'{groups}u'
        assert pm['PERM_SG'] == f'{simdgroups}u'
        assert pm['PERM_UNROLL'] == ('4u' if dynamic else '1u')
