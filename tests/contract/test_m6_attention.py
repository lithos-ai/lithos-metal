"""Measured launch scope and core/merge numerical partition agreement."""
import copy
import struct

import pytest

from monolith.backends.metal import config_for_device
from monolith.backends.metal.m6.attention import finalize, validate
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


def fixture(draft=False, **macros):
    config = copy.deepcopy(config_for_device(12, 11, 'Apple M6'))
    config.raw['engine']['attention_launch'] = dict(workers=24, simdgroups=8)
    m = dict(D='128' if draft else '256', RBMAX='4u', CH='64u', **macros)
    if draft:
        m['DRAFT'] = '1'
    record = struct.pack('<20I', *([32, 8, 7, 0, 144] if draft else [24, 4, 8, 0, 144]), *([0] * 15))
    program = Program(
        {'core': KernelSpec('', 'gqa_decode', dict(m)), 'merge': KernelSpec('', 'gqa_merge', dict(m))},
        {'params': BufferSpec(80, record, 'params')},
        [OpSpec('core', [(9, 'params', 0)], (12, 1, 1), (384, 1, 1), meta={'prefill': True}),
         OpSpec('merge', [(4, 'params', 0)], (192, 1, 1), (32, 1, 1))])
    return config, program


@pytest.mark.parametrize('draft', [False, True])
def test_launch_keeps_core_merge_partition_and_is_idempotent(draft):
    config, program = fixture(draft)
    original = copy.deepcopy(program)
    finalize(program, None, {}, config, t=8, dynamic_t=True)
    assert program.ops[0].grid == (24, 1, 1)
    assert program.ops[0].threadgroup == (256, 1, 1)
    assert program.ops[1].grid == original.ops[1].grid
    assert program.ops[1].threadgroup == original.ops[1].threadgroup
    for op in program.ops:
        macros = program.kernels[op.kernel].macros
        assert macros['GQA_CHUNK_SG'] == '144u'
        assert macros['STATIC_GQA_P_N_SG'] == '192u'
    before, after = original.buffers['params'].init, program.buffers['params'].init
    assert struct.unpack_from('<I', after, 16)[0] == 192
    assert after[:16] == before[:16] and after[20:] == before[20:]
    assert set(program.kernels) == {op.kernel for op in program.ops}
    unchanged = program.to_json()
    finalize(program, None, {}, config, t=8, dynamic_t=True)
    assert program.to_json() == unchanged


@pytest.mark.parametrize('case', ['off', 'static', 'large_t', 'different_shape', 'gamma', 'rows', 'lm', 'mma', 'crew', 'custom_grid'])
def test_unmeasured_routes_remain_unchanged(case):
    config, program = fixture(draft=case == 'gamma', **({'LM_MODE': '1'} if case == 'lm' else {}))
    options = dict(t=8, dynamic_t=True)
    if case == 'off': config.raw['engine'].pop('attention_launch')
    if case == 'static': options['dynamic_t'] = False
    if case == 'large_t': options['t'] = 512
    if case == 'rows': config.attention_rows = 8
    if case == 'crew': config.threadgroups_per_core = 2
    if case == 'custom_grid': program.ops[0].grid = (24, 1, 1)
    if case in ('different_shape', 'gamma'):
        record = bytearray(program.buffers['params'].init)
        struct.pack_into('<I', record, 8 if case == 'gamma' else 0, 15)
        program.buffers['params'].init = bytes(record)
    if case == 'mma':
        program.kernels['core'].function = 'gqa_decode_mma'
        for k in program.kernels.values(): k.macros['FIXED_CHUNK'] = '1'
    expected = program.to_json()
    finalize(program, None, {}, config, **options)
    assert program.to_json() == expected


def test_recipe_is_validated_and_fingerprinted():
    config, _ = fixture()
    validate(config)
    original = config.fingerprint
    config.raw['engine']['attention_launch']['workers'] = 48
    assert config.fingerprint != original
    with pytest.raises(ValueError, match='measured'):
        validate(config)


@pytest.mark.parametrize('dtype', [0,1,2])
@pytest.mark.parametrize('t', [32,64,128])
def test_mma_prefill_reuses_partial_buffers_and_retains_fallback(dtype,t):
    config,program=fixture(K_CACHE_TYPE=str(dtype),V_CACHE_TYPE=str(dtype))
    data=bytearray(program.buffers['params'].init)
    struct.pack_into('<I',data,8,t)
    program.buffers['params'].init=bytes(data)
    original_buffers=set(program.buffers)
    original_size=sum(b.nbytes for b in program.buffers.values())
    finalize(program,None,{},config,t=t,dynamic_t=True)
    names=[program.kernels[op.kernel].function for op in program.ops]
    assert names==['gqa_decode','gqa_decode_mma_m6','gqa_merge']
    assert program.ops[0].bindings==program.ops[1].bindings
    assert program.ops[1].grid==(48,1,1)
    assert set(program.buffers)==original_buffers
    assert sum(b.nbytes for b in program.buffers.values())==original_size
    assert program.kernels[program.ops[1].kernel].macros['GQA_CHUNK_SG']=='144u'
    assert program.kernels[program.ops[1].kernel].macros['M6_MMA_CORE_ONLY']=='1'
    saved=program.to_json()
    finalize(program,None,{},config,t=t,dynamic_t=True)
    assert program.to_json()==saved


@pytest.mark.parametrize('reason', ['q4','mixed','override','no_acceleration','draft','verification','large_verification','off','unknown_shape'])
def test_mma_prefill_scope(reason):
    config,program=fixture(draft=reason=='draft')
    t=8 if reason=='verification' else 128
    data=bytearray(program.buffers['params'].init)
    struct.pack_into('<I',data,8,t)
    if reason=='unknown_shape': struct.pack_into('<I',data,0,32)
    if reason=='large_verification': program.ops[0].meta['prefill'] = False
    program.buffers['params'].init=bytes(data)
    if reason=='off': config.raw['engine']['attention_prefill_mma']=False
    for op in program.ops:
        k=program.kernels[op.kernel]
        if reason=='q4': k.macros.update(K_CACHE_TYPE='3',V_CACHE_TYPE='3')
        if reason=='mixed': k.macros.update(K_CACHE_TYPE='1',V_CACHE_TYPE='2')
        if reason=='override': op.meta['attention_override']='v1'
    finalize(program,None,{},config,t=t,dynamic_t=True,accelerator='off' if reason=='no_acceleration' else 'on')
    assert not any(k.function=='gqa_decode_mma_m6' for k in program.kernels.values())


@pytest.mark.parametrize('attention', [None,'auto','v1'])
@pytest.mark.parametrize('dtype', ['bf16','q8_0','turbo4'])
def test_explicit_auto_uses_m6_policy_not_generic_mma(tmp_path,attention,dtype):
    from monolith.compiler import compile_program
    from monolith.core import StepStateLayout
    from monolith.formats import PackLayout
    from monolith.models.qwen3_moe import Qwen3MoeModel
    from monolith.nn.pack_plan import pack_model
    from monolith.nn.kv_quant import configure_kv_cache
    from monolith.packs import PackFile
    from tests.moe_synth import write_checkpoint
    write_checkpoint(tmp_path,num_hidden_layers=1,num_attention_heads=24,num_key_value_heads=4,head_dim=256)
    model=Qwen3MoeModel.from_checkpoint(str(tmp_path),max_context=2054)
    configure_kv_cache(model,dtype)
    pack_model(model,str(tmp_path),str(tmp_path/'pack'),PackLayout())
    program=compile_program(model,PackFile(tmp_path/'pack'),config_for_device(12,11,'Apple M6'),
        t=128,dynamic_t=True,layout=StepStateLayout(t_max=128),attention=attention,prefill=True)
    functions={k.function for k in program.kernels.values()}
    assert ('gqa_decode_mma_m6' in functions)==(attention!='v1')
    assert 'gqa_decode_mma' not in functions


@pytest.mark.parametrize('backend', ['m6', 'm5_max_40c'])
@pytest.mark.parametrize('attention', [None, 'auto', 'v1'])
@pytest.mark.parametrize('bound', [128, 512])
def test_exact_prefill_reference_keeps_backend_numerics(monkeypatch, backend, attention, bound):
    from types import SimpleNamespace
    import monolith.generate as generation
    session = generation.Session.__new__(generation.Session)
    for name in ('model', 'pack', 'eos', 'ring_capacity', 'layout', 'drafter',
                 'drafter_pack', 'verify', 'verify_threshold', 'verify_length',
                 'barriers', 'accelerator', 'commute_norm', 'gdn_mixer_fusion'):
        setattr(session, name, None)
    session.profile = SimpleNamespace(backend=backend)
    session.prefill_exact = True
    session.prefill_attention = attention
    session.attention = None
    session.decoder_kernel_config = None
    session._tuner = lambda prefill: None

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        expected = 'v1' if backend == 'm6' and attention in (None, 'auto') else attention
        assert kwargs['attention'] == expected
        raise Captured

    monkeypatch.setattr(generation, 'compile_program', capture)
    with pytest.raises(Captured):
        session._compile(bound, dynamic=True, prefill=True)
