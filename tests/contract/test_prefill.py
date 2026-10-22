import copy
import struct

import pytest

from monolith.backends.metal import get_backend
from monolith.compiler.prefill import exact_attention, specialize_prompt
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



def test_exact_attention_marks_chunked_cores_and_their_folds():
    def attention(core_macros, merge_macros=None):
        return Program({'core': KernelSpec('', 'gqa_decode', dict(core_macros, STEP_STATE='1')),
                        'merge': KernelSpec('', 'gqa_merge', dict(merge_macros or {}, STEP_STATE='1'))}, {},
                       [OpSpec('core', [(7, 'part_o', 0), (8, 'part_md', 0)], (60, 1, 1), (384, 1, 1)),
                        OpSpec('merge', [(0, 'part_o', 0), (1, 'part_md', 0)], (64, 1, 1), (32, 1, 1)),
                        OpSpec('core', [(7, 'part_o', 0), (8, 'part_md', 0)], (60, 1, 1), (384, 1, 1)),
                        OpSpec('merge', [(0, 'part_o', 0), (1, 'part_md', 0)], (64, 1, 1), (32, 1, 1))])
    p = exact_attention(attention({}), 128)
    assert [p.kernels[op.kernel].macros.get('EXACT_SUB') for op in p.ops] == ['128u'] * 4
    assert 'EXACT_SUB' not in attention({}).kernels['core'].macros
    for macros in ({'FIXED_CHUNK': '1'}, {'STEAL': '1'}, {'DRAFT': '1'}, {'LM_MODE': '2'}):
        before = attention(macros, macros if 'STEAL' not in macros else None)
        p = exact_attention(copy.deepcopy(before), 128)
        assert p.to_json() == before.to_json()


def test_exact_attention_marks_the_v3_core_without_a_partial_workspace():
    def attention(macros):
        return Program({'core': KernelSpec('', 'gqa_decode_v3', dict(macros, STEP_STATE='1'))}, {},
                       [OpSpec('core', [], (512, 1, 1), (1024, 1, 1))])
    p = exact_attention(attention({}), 128)
    assert p.kernels[p.ops[0].kernel].macros['EXACT_SUB'] == '128u'
    assert 'EXACT_SUB' not in attention({}).kernels['core'].macros
    before = attention({'LM_MODE': '2', 'CHAIN_I': '2u'})
    assert exact_attention(copy.deepcopy(before), 128).to_json() == before.to_json()


def test_large_projection_policy_is_chip_and_shape_specific(monkeypatch):
    from monolith.backends.metal.m5_max_40c import prefill
    monkeypatch.setattr(prefill, 'packed_nvfp4_projection', lambda *args: None)
    params = struct.pack('<IIIIfIII', 34816, 1088, 960, 512, 1., 0, 2176, 0)
    k = KernelSpec('', 'gemm_tile', {'TK': '128u', 'TN': '32u', 'TM': '32', 'T_SRC': '0',
                                    'STATIC_GEMM_P_N_TILES': '1088u', 'STATIC_GEMM_P_N_SG': '960u'})
    op = OpSpec('gemm', [(4, 'params', 0)], (80, 16, 1), (384, 1, 1),
                meta={'format': 'nvfp4', 'n': 34816, 'k': 5120, 't_variant': 512})
    original = Program({'gemm': k}, {'params': BufferSpec(32, params, 'params')}, [op])
    for backend in ('common', 'm4_pro', 'm4_max', 'm5_pro', 'm5_max_32c'):
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


def test_exact_policy_reads_decoder_layouts_and_keeps_bf16_operands(monkeypatch):
    from monolith.backends.metal.m5_max_40c import prefill
    record, calls = {'tn': 32}, []
    monkeypatch.setattr(prefill, 'decoder_layout', lambda program, op, tn, rows: record)
    monkeypatch.setattr(prefill, 'decoder_projection', lambda program, op, layout: calls.append(('decoder', layout)))
    monkeypatch.setattr(prefill, 'packed_nvfp4_projection', lambda *args: calls.append('private'))
    monkeypatch.setattr(prefill, 'direct_bf16_projection', lambda *args: calls.append('direct'))

    def program(fmt, n, k):
        params = struct.pack('<IIIIfIII', n, n//32, 960, 512, 1., 0, n//16, 0)
        kernel = KernelSpec('', 'gemm_tile', {'TK': '128u', 'TN': '32u', 'TM': '32', 'T_SRC': '0'})
        op = OpSpec('gemm', [(0, 'weights', 0), (4, 'params', 0)], (80, 16, 1), (384, 1, 1),
                    meta={'format': fmt, 'n': n, 'k': k, 't_variant': 512})
        return Program({'gemm': kernel}, {'params': BufferSpec(32, params, 'params'),
                                        'weights': BufferSpec(64, role='weights')}, [op])
    p = get_backend('m5_max_40c').optimize_prefill(program('nvfp4', 34816, 5120), exact=True)
    assert calls == [('decoder', record)]
    assert (p.ops[0].grid, p.ops[0].threadgroup, p.kernels[p.ops[0].kernel].macros['TN']) == ((160, 16, 1), (512, 1, 1), '16u')
    assert 'TB2' not in p.kernels[p.ops[0].kernel].macros
    assert p.kernels[p.ops[0].kernel].macros['SPLIT_K'] == '1'      # gate/up: each tile as two 64-column products
    calls.clear()
    p = get_backend('m5_max_40c').optimize_prefill(program('nvfp4', 5120, 17408), exact=True)
    assert calls == [('decoder', record)]                  # down: two 32-token blocks per decoded weight tile
    assert (p.ops[0].grid, p.kernels[p.ops[0].kernel].macros['TB2']) == ((160, 8, 1), '1')
    assert 'SPLIT_K' not in p.kernels[p.ops[0].kernel].macros
    calls.clear()
    p = get_backend('m5_max_40c').optimize_prefill(program('fp8_e4m3', 10240, 5120), exact=True)
    assert calls == [('decoder', record)] and 'SPLIT_K' not in p.kernels[p.ops[0].kernel].macros
    # FP8: one threadgroup per 32-row tile; its four SIMD groups share the decoded tile for four 32-token blocks
    macros = p.kernels[p.ops[0].kernel].macros
    assert (p.ops[0].grid, p.ops[0].threadgroup) == ((320, 4, 1), (128, 1, 1))
    assert (macros['TN'], macros['STAGE_B'], macros['STAGE_SB']) == ('32u', '1', '4u')
    calls.clear()
    widths = []
    narrow = {'tn': 16}                                     # a verification graph whose file tiles hold 16 rows
    monkeypatch.setattr(prefill, 'decoder_layout', lambda program, op, tn, rows: widths.append(tn) or (
        narrow if narrow['tn'] % tn == 0 else None))
    p = get_backend('m5_max_40c').optimize_prefill(program('fp8_e4m3', 10240, 5120), exact=True)
    assert widths == [32, 16] and calls == [('decoder', narrow)]  # not staged: its 32-row tiles cannot read that file
    assert (p.ops[0].grid, p.kernels[p.ops[0].kernel].macros['TN']) == ((80, 16, 1), '16u')
    assert 'STAGE_B' not in p.kernels[p.ops[0].kernel].macros
    monkeypatch.setattr(prefill, 'decoder_layout', lambda program, op, tn, rows: record)
    calls.clear()
    record = None                                          # no verification graph: the private layout, as without exact
    get_backend('m5_max_40c').optimize_prefill(program('nvfp4', 34816, 5120), exact=True)
    get_backend('m5_max_40c').optimize_prefill(program('nvfp4', 34816, 5120))
    original = program('bf16', 96, 5120)
    assert get_backend('m5_max_40c').optimize_prefill(copy.deepcopy(original), exact=True).to_json() == original.to_json()
    assert calls == ['private', 'private']


@pytest.mark.parametrize('exact', [True, False])
def test_affine_4bit_projections_take_two_token_blocks_per_weight_tile(exact):
    def program(n, k, rows=512):
        params = struct.pack('<IIIIfIII', n, n//32, 960, rows, 1., 0, n//32, 0)
        kernel = KernelSpec('', 'gemm_tile', {'TK': '128u', 'TN': '32u', 'TM': '32', 'T_SRC': '0'})
        op = OpSpec('gemm', [(0, 'weights', 0), (4, 'params', 0)], (80, 16, 1), (384, 1, 1),
                    meta={'format': 'int4_affine', 'n': n, 'k': k, 't_variant': rows})
        return Program({'gemm': kernel}, {'params': BufferSpec(32, params, 'params'),
                                        'weights': BufferSpec(64, role='weights')}, [op])
    backend = get_backend('m5_max_40c')
    p = backend.optimize_prefill(program(5120, 17408), exact=exact)
    macros = p.kernels[p.ops[0].kernel].macros
    assert (p.ops[0].grid, p.ops[0].threadgroup) == ((320, 16, 1), (128, 1, 1))      # 16 planes of two 16-row blocks
    assert (macros['TM'], macros['TN'], macros['TB2'], macros['KSPLIT']) == ('16', '32u', '1', '1u')
    assert struct.unpack_from('<II', p.buffers['params'].init, 4) == (160, 1280)    # row tiles, SIMD groups
    p = backend.optimize_prefill(program(34816, 5120), exact=exact)
    assert (p.ops[0].grid, p.kernels[p.ops[0].kernel].macros['TB2']) == ((320, 16, 1), '1')
    p = backend.optimize_prefill(program(6144, 5120), exact=exact)
    assert (p.ops[0].grid, p.ops[0].threadgroup) == ((80, 8, 1), (512, 1, 1))
    for shape in ((8192, 5120), (6144, 5120, 128)):             # a shape not measured; a 128-row graph
        original = program(*shape)
        assert backend.optimize_prefill(copy.deepcopy(original), exact=exact).to_json() == original.to_json()


@pytest.mark.parametrize('fmt', ['nvfp4', 'fp8_e4m3'])
def test_decoder_projection_maps_the_file_a_verification_graph_derived(tmp_path, monkeypatch, fmt):
    import numpy as np
    from monolith.compiler import fp8_tiles, nvfp4_tiles
    from monolith.compiler.prefill import decoder_layout, decoder_projection
    from monolith.formats import PackLayout
    from monolith.formats.blm import pack_blm
    from monolith.kernels import unit_geometry
    for module in (fp8_tiles, nvfp4_tiles):
        monkeypatch.setattr(module.tempfile, 'gettempdir', lambda: str(tmp_path))
    rng = np.random.default_rng(3)
    rows, width, nvfp4 = 64, 1024, fmt == 'nvfp4'
    codes = rng.integers(0, 256, (rows, 32, width // (64 if nvfp4 else 32)), dtype=np.uint8)
    scales = rng.integers(0, 127, (rows, 32, width // 512), dtype=np.uint8) if nvfp4 else None
    raw, info = pack_blm(codes, scales, PackLayout(lane_order='interleaved16'), format=fmt, k=width,
                         **({'scale_group': 16} if nvfp4 else {}))
    (tmp_path / 'pack').write_bytes(raw)
    weights = BufferSpec(len(raw), role='weights', file=str(tmp_path / 'pack'))
    macros = dict(K=str(width), R=str(info.rows), UNIT_WORDS=str(info.words_per_unit), LANE_ORDER='1', TK='128u',
                  TN='16u', KSPLIT='1u', **(unit_geometry(info) if nvfp4 else {}))
    source = ('head\n#pragma clang loop unroll(full)\n      for (uint s = 0; s < NS_B; s++) original fill\n'
              '#if EXP_MODE == 2\n      }\ntail')
    op = OpSpec('gemm', [(0, 'w', 0), (2, 'xp', 0)], (1, 1, 1), (32, 1, 1), meta={'format': fmt})
    prompt = Program({'gemm': KernelSpec(source, 'gemm_tile', macros)}, {'w': weights}, [op])
    decoder = Program({}, {'w': weights}, [])
    # layouts the prompt kernel cannot read in its own order are not offered
    if nvfp4:
        nvfp4_tiles.repack(decoder, ('w', 0), macros, rows, 32, 128, 32)
    else:
        fp8_tiles.repack(decoder, ('w', 0), dict(macros, Q_OUTER='1'), rows, 32, 32, 8)
    assert decoder_layout(prompt, op, 16, rows) is None
    packed = (nvfp4_tiles.repack(decoder, ('w', 0), dict(macros, Q_OUTER='1'), rows, 32, 64, 32) if nvfp4
              else fp8_tiles.repack(decoder, ('w', 0), macros, rows, 32, 32, 8))
    record = decoder_layout(prompt, op, 16, rows)
    assert decoder_layout(prompt, op, 64, rows) is None      # file tiles must hold whole kernel tiles
    assert (record['tn'], record['tk'], record['tile_block'], record['outer']) == (32, 64 if nvfp4 else 32, 32 if nvfp4 else 8, int(nvfp4))
    decoder_projection(prompt, op, record)
    kernel = prompt.kernels['gemm']
    assert op.bindings == [(0, packed[0], 0), (2, 'xp', 0)] and prompt.buffers[packed[0]].file == packed[1].file
    assert 'original fill' not in kernel.source and kernel.source.startswith('head') and kernel.source.endswith('tail')
    assert (kernel.macros['FILE_TN'], kernel.macros['FILE_BLOCK'], kernel.macros['TK']) == ('32u', '32u' if nvfp4 else '8u', '128u')
    assert op.meta == {'format': fmt, 'prefill_packed_nvfp4' if nvfp4 else 'prefill_packed_fp8': True}
    kernel.macros['KSPLIT'] = '2u'
    with pytest.raises(ValueError, match='unsplit 128-column'):
        decoder_projection(prompt, op, record)
    # split products: the same file reads, each 64-column half multiplied as soon as it is decoded (NVFP4 only)
    kernel.macros['KSPLIT'] = '1u'
    kernel.macros['SPLIT_K'] = '1'
    kernel.source = source
    op.bindings = [(0, 'w', 0), (2, 'xp', 0)]
    if nvfp4:
        decoder_projection(prompt, op, record)
        assert kernel.source.count('op.run(sAh, bH, cT);') == 1 and 'original fill' not in kernel.source
    else:
        with pytest.raises(ValueError, match='NVFP4 operands'):
            decoder_projection(prompt, op, record)
    # staged weights: the FP8 tile decoded into threadgroup memory once per threadgroup
    del kernel.macros['SPLIT_K']
    kernel.macros.update(STAGE_B='1', TN='32u')
    kernel.source = source
    op.bindings = [(0, 'w', 0), (2, 'xp', 0)]
    if nvfp4:
        with pytest.raises(ValueError, match='FP8 operands'):
            decoder_projection(prompt, op, record)
    else:
        decoder_projection(prompt, op, record)
        assert 'stage_b + (kt & 1u)' in kernel.source and kernel.source.count('threadgroup_barrier') == 1


def _attention_pair(ch=2048):
    macros = {'STEP_STATE': '1', 'CH': f'{ch}u', 'D': '256u'}
    shared = [(1, 'k', 0), (2, 'v', 0), (9, 'params', 0), (15, 'state', 0)]
    kernels = {'prep': KernelSpec('a prepared_q + t*p.in_stride+p.q_off+h*D+lane*DL b', 'gqa_prepare_mma', dict(macros)),
               'core': KernelSpec('head\n// Matrix-accelerator attention: staged', 'gqa_decode_mma', dict(macros))}
    return Program(kernels, {}, [OpSpec('prep', [(10, 'q', 0)] + shared, (1, 1, 1), (32, 1, 1)),
                                 OpSpec('core', [(0, 'q', 0)] + shared, (1, 1, 1), (32, 1, 1))])


def test_prefill_attention_splits_score_tiles_into_softmax_blocks():
    from monolith.compiler.prefill import device_attention_tiles
    p = device_attention_tiles(_attention_pair(), qm=16, kn=128, ks=32)
    core = p.kernels['core']
    assert {k: core.macros[k] for k in ('QM', 'KN', 'KS')} == {'QM': '16', 'KN': '128', 'KS': '32'}
    assert p.ops[1].meta['prefill_device_tiles'] == [16, 128, 32]
    assert 'staged' not in core.source and '#define KS KN' in core.source       # KS defaults to whole score tiles
    assert device_attention_tiles(_attention_pair(), kn=128).kernels['core'].macros['KS'] == '128'
    for kw in (dict(kn=128, ks=48), dict(kn=128, ks=16), dict(qm=64, kn=128, ks=32)):
        with pytest.raises(ValueError, match='threadgroup scratch'):
            device_attention_tiles(_attention_pair(), **kw)
    # BF16 scores overwritten by the probabilities: a third of the scratch, so 64-query tiles fit
    p = device_attention_tiles(_attention_pair(), qm=64, kn=128, ks=32, score_bf16=True)
    assert p.kernels['core'].macros['SCORE_BF16'] == '1'
    assert 'SCORE_BF16' not in device_attention_tiles(_attention_pair(), qm=16, kn=128, ks=32).kernels['core'].macros


def test_projection_geometry_two_token_blocks_halve_the_token_planes():
    from monolith.compiler.prefill import projection_geometry
    params = struct.pack('<IIIIfIII', 5120, 160, 960, 512, 1., 0, 320, 0)
    kernel = KernelSpec('', 'gemm_tile', {'TK': '128u', 'TN': '32u', 'TM': '32', 'T_SRC': '0'})
    def program():
        op = OpSpec('gemm', [(4, 'params', 0)], (80, 16, 1), (256, 1, 1), meta={'format': 'nvfp4', 'n': 5120})
        return Program({'gemm': copy.deepcopy(kernel)}, {'params': BufferSpec(32, params, 'params')}, [op])
    p = program()
    projection_geometry(p, p.ops[0], tm=32, tn=16, sgs=16, groups=160, token_blocks=2)
    assert (p.ops[0].grid, p.kernels[p.ops[0].kernel].macros['TB2']) == ((160, 8, 1), '1')
    p = program()
    projection_geometry(p, p.ops[0], tm=32, tn=16, sgs=16, groups=160)
    assert p.ops[0].grid == (160, 16, 1) and 'TB2' not in p.kernels[p.ops[0].kernel].macros
    p = program()
    with pytest.raises(ValueError, match='one or two token blocks'):
        projection_geometry(p, p.ops[0], tm=32, tn=16, sgs=16, groups=160, token_blocks=3)


def test_prompt_gdn_recurrence_sums_four_columns_in_one_butterfly():
    from monolith.backends.metal.m5_max_40c.prefill import optimize
    params = struct.pack('<III', 48, 16, 512) + bytes(68)
    def program(sl):
        kernel = KernelSpec('', 'gdn_mixer', {'PREPARED': '1', 'DK': '128u', 'DV': '128u', 'SL': sl})
        op = OpSpec('mix', [(9, 'params', 0)], (1, 1, 1), (32, 1, 1), name='gdn_mixer')
        return Program({'mix': kernel}, {'params': BufferSpec(80, params, 'params')}, [op])
    p = optimize(program('4u'))
    assert p.kernels[p.ops[0].kernel].macros['TREE_REDUCE'] == '1' and p.kernels[p.ops[0].kernel].macros['TP'] == '512u'
    p = optimize(program('8u'))
    assert 'TREE_REDUCE' not in p.kernels[p.ops[0].kernel].macros
