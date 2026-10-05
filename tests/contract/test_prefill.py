import copy
import struct

import pytest

from monolith.backends.metal import get_backend
from monolith.compiler.prefill import specialize_prompt
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


def test_prompt_sampling_guards_do_not_change_draft_or_final_sampling():
    source = 'kernel void sample() { if (st->done) return; }'
    p = Program({'sample': KernelSpec(source, 'argmax_final', {'STEP_STATE': '1'}),
                 'commit': KernelSpec('commit', 'gdn_mixer')}, {},
                [OpSpec('sample', [], (1, 1, 1), (32, 1, 1), name='argmax_final'),
                 OpSpec('sample', [], (1, 1, 1), (32, 1, 1), name='accept_scan'),
                 OpSpec('commit', [], (1, 1, 1), (32, 1, 1), name='gdn_commit'),
                 OpSpec('sample', [], (1, 1, 1), (32, 1, 1), name='argmax_final')])
    specialize_prompt(p)
    assert len(p.ops) == 3
    assert 'st->prefill_left > 0u' in p.kernels[p.ops[0].kernel].source
    assert p.kernels[p.ops[-1].kernel].source == source


def test_large_projection_policy_is_chip_and_shape_specific(monkeypatch):
    from monolith.backends.metal.m5_max_40c import prefill
    monkeypatch.setattr(prefill, 'packed_nvfp4_projection', lambda *args: None)
    params = struct.pack('<IIIIfIII', 34816, 1088, 960, 512, 1., 0, 2176, 0)
    k = KernelSpec('', 'gemm_tile', {'TK': '128u', 'TN': '32u', 'TM': '32', 'T_SRC': '0',
                                    'STATIC_GEMM_P_N_TILES': '1088u', 'STATIC_GEMM_P_N_SG': '960u'})
    op = OpSpec('gemm', [(4, 'params', 0)], (80, 16, 1), (384, 1, 1),
                meta={'format': 'nvfp4', 'n': 34816, 'k': 5120, 't_variant': 512})
    original = Program({'gemm': k}, {'params': BufferSpec(32, params, 'params')}, [op])
    for backend in ('common', 'm4_pro', 'm5_pro', 'm5_max_32c'):
        p = copy.deepcopy(original)
        assert get_backend(backend).optimize_prefill(p).to_json() == original.to_json()
    p = get_backend('m5_max_40c').optimize_prefill(copy.deepcopy(original))
    assert p.ops[0].grid == (160, 16, 1)
    assert p.ops[0].threadgroup == (512, 1, 1)
    assert p.kernels[p.ops[0].kernel].macros['TN'] == '16u'
    assert struct.unpack_from('<II', p.buffers['params'].init, 4) == (2176, 2560)
    for field, value in [('k', 4096), ('format', 'bf16'), ('t_variant', 8)]:
        p = copy.deepcopy(original)
        p.ops[0].meta[field] = value
        before = p.to_json()
        assert get_backend('m5_max_40c').optimize_prefill(p).to_json() == before


@pytest.mark.parametrize('n,k', [(10240, 5120), (6144, 5120), (8192, 5120), (5120, 6144)])
@pytest.mark.parametrize('rows', [8, 128, 256, 512, 1024])
def test_prefill_fp8_operands_stay_compact_and_do_not_change_decode(n, k, rows, monkeypatch):
    from monolith.backends.metal.m5_max_40c import prefill
    direct, packed = [], []
    monkeypatch.setattr(prefill, 'direct_bf16_projection', lambda *args: direct.append(args[1].meta['n']))
    monkeypatch.setattr(prefill, 'packed_fp8_projection', lambda *args, **kwargs: packed.append(args[1].meta['n']))
    params = struct.pack('<IIIIfIII', n, n//32, 640, rows, 1., 0, n//16, 0)
    kernel = KernelSpec('static inline float fp8_e4m3(uint q);', 'gemm_tile',
                        {'TK': '128u', 'TN': '32u', 'TM': '32', 'T_SRC': '0'})
    op = OpSpec('gemm', [(0, 'weights', 0), (4, 'params', 0)], (80, 1, 1), (256, 1, 1),
                meta={'format': 'fp8_e4m3', 'n': n, 'k': k, 't_variant': rows})
    p = Program({'gemm': kernel}, {'params': BufferSpec(32, params, 'params'),
                                 'weights': BufferSpec(n*k, role='weights')}, [op])
    get_backend('m5_max_40c').optimize_prefill(p)
    assert p.kernels[p.ops[0].kernel].macros.get('FP8_DECODE', '0') == ('1' if rows == 512 else '0')
    assert 'FP8_DECODE' not in kernel.macros
    assert direct == []
    assert packed == ([n] if rows == 512 else [])
    assert p.buffers['weights'].nbytes == n*k


def test_packed_tails_keep_shared_permutations_and_unrelated_variants():
    from monolith.compiler.prefill import packed_projection_tails

    kernels = {
        'gemm': KernelSpec('', 'gemm_tile', {'STEP_STATE': '1', 'T_LO': '1'}),
        'gemv': KernelSpec('', 'gemv_T', {'STEP_STATE': '1', 'T_LO': '0'}),
        'perm': KernelSpec('', 'x_permute', {'STEP_STATE': '1', 'T_LO': '1'}),
        'norm': KernelSpec('', 'norm_apply'),
    }
    def projection(group, matrix, packed=False):
        meta = dict(variant_group=group, t_range=[1, 512] if matrix else [0, 1])
        if packed:
            meta['prefill_packed_fp8'] = True
        return OpSpec('gemm' if matrix else 'gemv', [(2, 'xp' if matrix else 'normalized', 0)],
                      (1, 1, 1), (32, 1, 1), meta=meta)
    shared = OpSpec('perm', [(0, 'input', 0), (3, 'xp', 0)], (1, 1, 1), (32, 1, 1),
                    meta={'t_range': [1, 512]})
    unrelated = projection('untouched', False)
    unrelated.bindings = [(2, 'input', 0)]
    norm = OpSpec('norm', [(0, 'input', 0), (3, 'normalized', 0)], (1, 1, 1), (32, 1, 1))
    ops = [norm, shared, projection('qk', False), projection('qk', True, True),
           projection('v', False), projection('v', True, True), unrelated]
    buffers = {name: BufferSpec(64, role='arena') for name in ('input', 'xp', 'normalized')}
    p = Program(kernels, buffers, ops)
    packed_projection_tails(p)
    assert p.ops == [shared, ops[3], ops[5], unrelated]
    assert 'normalized' not in p.buffers
    assert p.kernels['perm'].macros['T_LO'] == '1'  # other users of the source remain untouched
    for op in p.ops[:-1]:
        assert op.meta['t_range'] == [0, 512]
        assert p.kernels[op.kernel].macros['T_LO'] == '0'
    assert unrelated.meta['t_range'] == [0, 1]


def test_packed_tails_reject_an_unexpected_shader_variant():
    from monolith.compiler.prefill import packed_projection_tails

    p = Program({'gemm': KernelSpec('', 'gemm_tile', {'STEP_STATE': '1'})}, {},
                [OpSpec('gemm', [(2, 'x', 0)], (1, 1, 1), (32, 1, 1),
                        meta={'prefill_packed_nvfp4': True, 'variant_group': 'w', 't_range': [8, 512]})])
    with pytest.raises(ValueError, match='512-row matrix variant'):
        packed_projection_tails(p)


@pytest.mark.parametrize('fmt', ['fp8_e4m3', 'nvfp4'])
def test_direct_operands_reject_quantized_weight_expansion(fmt):
    from monolith.compiler.prefill import direct_bf16_projection

    op = OpSpec('gemm', [], (1, 1, 1), (32, 1, 1), meta={'format': fmt})
    p = Program({'gemm': KernelSpec('', 'gemm_tile')}, {}, [op])
    with pytest.raises(ValueError, match='quantized weights must stay compact'):
        direct_bf16_projection(p, op)
    assert p.buffers == {}
