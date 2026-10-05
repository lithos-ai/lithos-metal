"""Distinct slab segments can reuse one compiled projection kernel."""
import struct
import pytest
from monolith.runtime.program import Program, KernelSpec, BufferSpec, OpSpec
from tools.bench.gdn_block_static import normalize, merge


def test_wider_tile_updates_each_parameter_record_for_shared_kernel():
    p=Program(kernels={'shared':KernelSpec('', 'gemm_tile', {'TN':'16u','KSPLIT':'1u'})},
              buffers={name:BufferSpec(32,struct.pack('<IIIIfIII',rows,rows//16,320,8,1.,tile0,1024,0),'params')
                       for name,rows,tile0 in [('core',8192,0),('gate',6144,512)]},
              ops=[OpSpec('shared',[(4,name,0)],(40,1,1),(256,1,1)) for name in ('core','gate')])
    got=normalize(p,8,groups=80,tn=32,split=True)
    core,gate=[struct.unpack('<IIIIfIII',got.buffers[n].init) for n in ('core','gate')]
    assert (core[1],core[5])==(256,0)
    assert (gate[1],gate[5])==(192,256)
    assert core[2]==gate[2]==640
    assert struct.unpack('<IIIIfIII',p.buffers['gate'].init)[5]==512


@pytest.mark.parametrize('workers',[0,-1,257])
def test_invalid_worker_stride_is_rejected_before_shader_emission(workers):
    with pytest.raises(ValueError,match='worker count'):
        merge(Program({}, {}, []),workers,16)


def test_zero_projection_stride_is_rejected():
    with pytest.raises(ValueError,match='projection task groups'):
        normalize(Program({}, {}, []),16,groups=0)


def test_long_task_chain_has_bounded_scratch_expression():
    kernel=KernelSpec('kernel void noop(uint gid [[thread_position_in_grid]]) {}','noop')
    p=Program({'noop':kernel},{},[OpSpec('noop',[],(1,1,1),(128,1,1)) for _ in range(28)])
    source=merge(p,8,4).kernels['mega'].source
    assert len(source)<100000
    for index in range(28):assert f'sizeof(s{index}::Scratch)' in source


def test_preprocessing_without_apple_sdk_preserves_task_source(monkeypatch):
    from monolith.compiler import static_fusion
    kernel=KernelSpec('kernel void noop(uint gid [[thread_position_in_grid]]) { uint n=COUNT; }',
                      'noop',{'COUNT':'16u'})
    expected=static_fusion.stage(kernel,0)
    which=static_fusion.shutil.which
    monkeypatch.setattr(static_fusion.shutil,'which',lambda name:None if name=='xcrun' else which(name))
    assert static_fusion.stage(kernel,0)==expected


@pytest.mark.parametrize('kw', [{'attention_groups':0},{'attention_groups':4097},
                               {'attention_qm':32},{'attention_qm':24},{'merge_sgs':16},{'merge_unroll':3},
                               {'attention_compact_partials':1},
                               {'attention_prepare':1},{'attention_chunk_tiles':True},
                               {'attention_chunk_tiles':5},{'attention_chunk_tiles':2048},
                               {'attention_style':'unknown'},{'attention_style':'cooperative'},
                               {'attention_key_tile':64},{'attention_cached_prefix':1},
                               {'attention_prepare':True,'attention_style':'cooperative','attention_key_tile':128},
                               {'attention_prepare':True,'attention_style':'cooperative','attention_task_order':'chunk'}])
def test_invalid_attention_geometry_is_rejected(kw):
    with pytest.raises(ValueError):
        normalize(Program({}, {}, []),8,**kw)


def test_attention_merge_unroll_changes_the_preprocessed_task():
    from monolith import kernels
    from monolith.compiler.static_fusion import stage
    p=Program(kernels={'merge':KernelSpec(kernels.gqa_source(),'gqa_merge',kernels.gqa_macros(256))},
              buffers={},ops=[OpSpec('merge',[],(192,1,1),(32,1,1))])
    got=normalize(p,8,merge_sgs=2,merge_unroll=16)
    source,_=stage(got.kernels['merge'],0)
    assert 'm_c[16u]' in source and 'c0 += 16u' in source
    assert got.ops[0].grid==(96,1,1)
    assert '#define MERGE_UNROLL 4u' in p.kernels['merge'].source


def test_attention_stride_updates_specialization_and_parameter_record():
    from monolith import kernels
    record=bytearray(80)
    struct.pack_into('<I',record,16,160)
    p=Program(kernels={'core':KernelSpec(kernels.gqa_source(mma=True),'gqa_decode_mma',
              dict(kernels.gqa_macros(256),STATIC_GQA_P_N_SG='160u'))},
              buffers={'params':BufferSpec(80,bytes(record),'params')},
              ops=[OpSpec('core',[(9,'params',0)],(160,1,1),(256,1,1))])
    got=normalize(p,4,attention_groups=320,attention_qm=8)
    assert got.ops[0].grid==(320,1,1) and got.ops[0].threadgroup==(128,1,1)
    assert got.kernels['core'].macros['STATIC_GQA_P_N_SG']=='320u'
    assert struct.unpack_from('<I',got.buffers['params'].init,16)[0]==320
    assert '#define QM 8' in got.kernels['core'].source
    assert struct.unpack_from('<I',p.buffers['params'].init,16)[0]==160


@pytest.mark.parametrize('kw',[{'task_grain':'unknown'},{'task_batch':0},{'task_batch':3},
                              {'task_batch':True},{'task_stats':True},{'attention_task_tiles':0},
                              {'task_seed_bound':True},{'task_seed_bound':1}])
def test_invalid_task_schedule_is_rejected(kw):
    with pytest.raises(ValueError):merge(Program({}, {}, []),80,8,**kw)


@pytest.mark.parametrize('kw',[{'gdn_prepare':'unknown'},{'gdn_unroll':3},
                              {'gdn_unroll':True},{'gdn_vector':3},{'gdn_vector':True}])
def test_invalid_recurrence_schedule_is_rejected(kw):
    with pytest.raises(ValueError):merge(Program({}, {}, []),80,8,**kw)


@pytest.mark.parametrize('kw',[{'gdn_tp':3},{'gdn_tp':True},{'perm_sgs':3},
                              {'fp8_decode':'unknown'},{'staged_tk':64},
                              {'mode':'staged','staged_tk':32},{'tk':64},
                              {'tm':8},{'tm':True},{'tm':32,'mode':'staged'}])
def test_invalid_operand_and_recurrence_geometry_is_rejected(kw):
    with pytest.raises(ValueError):normalize(Program({}, {}, []),8,**kw)


@pytest.mark.parametrize('kw',[{'tk':16},{'tk':64,'short_decode':True},
                              {'tk':64,'narrow_scales':True},{'tk':64,'mode':'staged'}])
def test_invalid_reduction_tile_is_rejected(kw):
    with pytest.raises(ValueError):normalize(Program({}, {}, []),8,**kw)


@pytest.mark.parametrize('kw',[{'weight_prefetch':1},
                              {'weight_prefetch':3,'fp8_layout':'tile'},
                              {'weight_prefetch':True,'fp8_layout':'tile'},
                              {'weight_prefetch':2,'fp8_layout':'tile','tk':16,'tn':32}])
def test_invalid_weight_prefetch_is_rejected(kw):
    with pytest.raises(ValueError,match='weight prefetch'):
        normalize(Program({}, {}, []),8,**kw)


@pytest.mark.parametrize('kw',[{'fp8_tile_block':2}, {'fp8_tile_block':True},
                              {'fp8_tile_block':3,'fp8_layout':'tile'}])
def test_invalid_tile_interleaving_is_rejected(kw):
    with pytest.raises(ValueError,match='tile interleaving'):
        normalize(Program({}, {}, []),8,**kw)


@pytest.mark.parametrize('kw',[{'fp8_storage':'half'}, {'fp8_storage':'unknown'},
                              {'fp8_storage':'half','fp8_layout':'tile','weight_prefetch':1},
                              {'fp8_storage':'half','fp8_layout':'tile','fp8_decode':'subtract'}])
def test_invalid_predecoded_storage_is_rejected(kw):
    with pytest.raises(ValueError,match='predecoded FP8 storage'):
        normalize(Program({}, {}, []),8,**kw)


def test_tile_tasks_privatize_shared_projection_strides():
    from monolith.compiler.static_fusion import _tile_tasks
    p=Program(kernels={'shared':KernelSpec('', 'gemm_tile', {'KSPLIT':'8u'})},
              buffers={n:BufferSpec(32,struct.pack('<IIIIfIII',rows,rows//16,640,8,1.,0,1024,0),'params')
                       for n,rows in [('core',8192),('gate',6144)]},
              ops=[OpSpec('shared',[(4,n,0)],(80,1,1),(256,1,1)) for n in ('core','gate')])
    got=_tile_tasks(p,8,80,1)
    assert [o.grid[0] for o in got.ops]==[512,384]
    assert len({o.kernel for o in got.ops})==2
    for o in got.ops:
        n=o.bindings[0][1]
        assert struct.unpack_from('<I',got.buffers[n].init,8)[0]==o.grid[0]*8
        assert got.kernels[o.kernel].macros['STATIC_GEMM_P_N_SG']==f'{o.grid[0]*8}u'
    assert struct.unpack_from('<I',p.buffers['core'].init,8)[0]==640


@pytest.mark.parametrize('kw',[{'poll_sgs':2}, {'poll_sgs':True},
                              {'poll_sgs':3,'barrier':'simd'}, {'poll_sgs':16,'barrier':'simd'}])
def test_invalid_polling_group_count_is_rejected(kw):
    with pytest.raises(ValueError,match='parallel polling groups'):
        merge(Program({}, {}, []),80,8,**kw)


@pytest.mark.parametrize('stride',[0,3,128,True])
def test_invalid_barrier_flag_stride_is_rejected(stride):
    with pytest.raises(ValueError,match='barrier flag stride'):
        merge(Program({}, {}, []),80,8,flag_stride=stride)


def test_invalid_barrier_arrival_is_rejected():
    with pytest.raises(ValueError,match='barrier arrival'):
        merge(Program({}, {}, []),80,8,arrival='unknown')


@pytest.mark.parametrize('hazard',['input_alias','other_writer','not_state'])
def test_cached_attention_prefix_requires_an_immutable_range(hazard):
    from monolith.compiler.attention_fusion import _cache_prefix
    core=OpSpec('core',[(0,'q',0),(1,'k',0),(2,'v',0)],(1,1,1),(128,1,1))
    p=Program({'core':KernelSpec('','gqa_decode_mma')},
              {'q':BufferSpec(64), 'k':BufferSpec(64,role='state'),
               'v':BufferSpec(64,role='state')},[core])
    if hazard=='input_alias':core.bindings[0]=(0,'k',0)
    if hazard=='not_state':p.buffers['k'].role='arena'
    if hazard=='other_writer':
        p.kernels['writer']=KernelSpec('','gemm_tile')
        p.ops.insert(0,OpSpec('writer',[(3,'k',0)],(1,1,1),(128,1,1)))
    with pytest.raises(ValueError,match='state buffers|aliases|modify the KV prefix'):
        _cache_prefix(p,core,p.kernels['core'],False)


@pytest.fixture
def attention_partial_program():
    record=bytearray(80)
    for offset,value in [(4,4),(44,33024),(60,1032),(64,48)]:
        struct.pack_into('<I',record,offset,value)
    kernels={n:KernelSpec('',fn,dict(D='256',CH='1024u',STATIC_GQA_P_N_CHUNKS_MAX='1032u'))
             for n,fn in [('core','gqa_decode_mma'),('fold','gqa_merge')]}
    buffers={'po':BufferSpec(202899456),'pm':BufferSpec(1585152),
             'params':BufferSpec(80,bytes(record),'params')}
    ops=[OpSpec('core',[(7,'po',0),(8,'pm',0),(9,'params',0)],(1,1,1),(256,1,1)),
         OpSpec('fold',[(0,'po',0),(1,'pm',0),(4,'params',0)],(1,1,1),(32,1,1))]
    return Program(kernels,buffers,ops)


def test_partial_compaction_preserves_capacity_and_unrelated_specializations(attention_partial_program):
    from monolith.compiler.attention_fusion import compact_partials
    p=attention_partial_program
    p.ops.append(OpSpec('fold',[],(1,1,1),(32,1,1)))
    compact_partials(p)
    assert struct.unpack_from('<I',p.buffers['params'].init,44)[0]==33024
    assert struct.unpack_from('<I',p.buffers['params'].init,60)[0]==33
    assert (p.buffers['po'].nbytes,p.buffers['pm'].nbytes)==(6488064,50688)
    assert p.kernels[p.ops[0].kernel].macros['STATIC_GQA_P_N_CHUNKS_MAX']=='33u'
    assert p.kernels[p.ops[1].kernel].macros['STATIC_GQA_P_N_CHUNKS_MAX']=='33u'
    assert p.kernels[p.ops[2].kernel].macros['STATIC_GQA_P_N_CHUNKS_MAX']=='1032u'


@pytest.mark.parametrize('hazard',['state','initialized','offset','extra_consumer','different_params'])
def test_partial_compaction_rejects_shared_storage(attention_partial_program,hazard):
    from monolith.compiler.attention_fusion import compact_partials
    p=attention_partial_program
    if hazard=='state':p.buffers['po'].role='state'
    if hazard=='initialized':p.buffers['po'].init=b'\x00'
    if hazard=='offset':p.ops[0].bindings[0]=(7,'po',4)
    if hazard=='extra_consumer':p.ops.append(OpSpec('fold',[(0,'po',0)],(1,1,1),(32,1,1)))
    if hazard=='different_params':p.ops[1].bindings[-1]=(4,'other_params',0)
    with pytest.raises(ValueError):compact_partials(p)


def test_partial_compaction_updates_separate_core_merge_records(attention_partial_program):
    from monolith.compiler.attention_fusion import compact_partials
    p=attention_partial_program
    record=bytearray(p.buffers['params'].init)
    struct.pack_into('<I',record,56,1)  # Merge applies a gate; core does not.
    p.buffers['fold_params']=BufferSpec(80,bytes(record),'params')
    p.ops[1].bindings[-1]=(4,'fold_params',0)
    compact_partials(p)
    for name in ['params','fold_params']:
        assert struct.unpack_from('<I',p.buffers[name].init,60)[0]==33
    assert struct.unpack_from('<I',p.buffers['fold_params'].init,56)[0]==1
    assert struct.unpack_from('<I',p.buffers['params'].init,56)[0]==0


@pytest.mark.parametrize('offset',[0,4,8,44,60,64])
def test_partial_compaction_rejects_incompatible_records(attention_partial_program,offset):
    from monolith.compiler.attention_fusion import compact_partials
    p=attention_partial_program
    record=bytearray(p.buffers['params'].init)
    value=struct.unpack_from('<I',record,offset)[0]
    struct.pack_into('<I',record,offset,value+1)
    p.buffers['fold_params']=BufferSpec(80,bytes(record),'params')
    p.ops[1].bindings[-1]=(4,'fold_params',0)
    with pytest.raises(ValueError,match='workspace geometry'):compact_partials(p)
    assert p.buffers['po'].nbytes==202899456
