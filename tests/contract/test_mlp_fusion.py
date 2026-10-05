"""MLP tensor-layout boundaries and unsupported combinations fail before GPU work."""
import pytest
from monolith.runtime.program import Program,KernelSpec,OpSpec
from monolith.compiler.mlp_fusion import repair_projection_layouts
from monolith.compiler.static_fusion import normalize,merge


def test_independent_projection_layouts_update_only_their_producers():
    p=Program({'producer':KernelSpec('','gemm_tile',dict(TK='128u',PERM_OUT='1',PERM_TK='128u')),
               'consumer':KernelSpec('','gemm_tile',dict(TK='64u'))}, {},
              [OpSpec('producer',[(2,'hidden',0),(3,'middle',0)],(1,1,1),(128,1,1)),
               OpSpec('consumer',[(2,'middle',0)],(1,1,1),(128,1,1))])
    repair_projection_layouts(p)
    assert p.kernels[p.ops[0].kernel].macros['PERM_TK']=='64u'
    assert p.kernels['producer'].macros['PERM_TK']=='128u'


def test_shared_input_rejects_incompatible_consumers():
    p=Program({n:KernelSpec('','gemm_tile',dict(TK=tk)) for n,tk in [('a','64u'),('b','128u')]},{},
        [OpSpec(n,[(2,'shared',0)],(1,1,1),(128,1,1)) for n in ('a','b')])
    with pytest.raises(ValueError,match='shared activation'):repair_projection_layouts(p)


def test_device_tensor_mlp_cannot_be_merged():
    p=Program({'a':KernelSpec('','gemm_tile',dict(STATIC_NATIVE_TENSOR='1'))},{},[OpSpec('a',[],(1,1,1),(128,1,1))])
    with pytest.raises(ValueError,match='dispatch boundaries'):merge(p,4,4)


@pytest.mark.parametrize('config',[
 {'nvfp4_layout':'unknown'}, {'nvfp4_layout':'tile','nvfp4_tile_block':3},
 {'nvfp4_operand':'half4'}, {'nvfp4_layout':'tile','nvfp4_operand':'bad'},
 {'nvfp4_layout':'tile','nvfp4_prefetch':8}, {'nvfp4_layout':'tile','nvfp4_prefetch':True},
 {'nvfp4_layout':'tile','nvfp4_prefetch':1,'mode':'native','staged_tk':128},
 {'nvfp4_vector_loads':True}, {'nvfp4_layout':'tile','nvfp4_vector_loads':1},
 {'nvfp4_layout':'tile','narrow_weights':True}, {'nvfp4_scale_mode':'duplicated'},
 {'post_norm_once':1}, {'post_norm_loads':0}, {'post_norm_loads':True}])
def test_invalid_nvfp4_layout_is_rejected(config):
    with pytest.raises(ValueError):normalize(Program({}, {}, []),4,**config)


@pytest.mark.parametrize('writes_external',[False,True])
def test_external_input_caching_preserves_cross_task_coherence(writes_external):
    from monolith.runtime.program import BufferSpec
    producer='''kernel void producer(device const float* external [[buffer(0)]], device float* intermediate [[buffer(1)]], uint gid [[thread_position_in_grid]]) { intermediate[gid]=external[gid]; }'''
    consumer='''kernel void consumer(device const float* intermediate [[buffer(0)]], device float* result [[buffer(1)]], uint gid [[thread_position_in_grid]]) { result[gid]=intermediate[gid]; }'''
    p=Program({'a':KernelSpec(producer,'producer',{}),'b':KernelSpec(consumer,'consumer',{})},
        {n:BufferSpec(1024) for n in ('external','middle','result')},
        [OpSpec('a',[(0,'external',0),(1,'middle',0)],(4,1,1),(32,1,1)),
         OpSpec('b',[(0,'middle',0),(1,'external' if writes_external else 'result',0)],(4,1,1),(32,1,1),barrier_before=True)])
    got=merge(p,4,1,cache_external_inputs=True).kernels['mega'].source
    assert ('coherent(device) device const float* external' in got)==writes_external
    assert 'coherent(device) device const float* intermediate' in got
    assert 'coherent(device) device float* intermediate' in got


def test_suffix_fixture_never_copies_outputs_or_intermediates():
    from types import SimpleNamespace
    from monolith.runtime.program import BufferSpec
    from tools.bench.modelopt_mlp_suffix_tune import external_inputs,seed_tail
    src='''kernel void projection(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint gid [[thread_position_in_grid]]) { y[gid]=x[gid]; }'''
    p=Program({'k':KernelSpec(src,'projection',{})},
        {n:BufferSpec(16) for n in ('external','middle','result','step_state')},
        [OpSpec('k',[(0,'external',0),(1,'middle',0)],(1,1,1),(32,1,1)),
         OpSpec('k',[(0,'middle',0),(1,'result',0)],(1,1,1),(32,1,1))],step_state='step_state')
    incoming=external_inputs(p)
    assert incoming=={'external','step_state'}
    copied=[]
    target=SimpleNamespace(program=p,buffers={n:SimpleNamespace(write=lambda data,off,n=n:copied.append(n)) for n in p.buffers})
    source=SimpleNamespace(buffers=p.buffers,read=lambda name,count:bytes(count))
    seed_tail(source,target,incoming)
    assert set(copied)==incoming


def test_tensor_api_input_is_readonly_but_aliasing_output_is_not():
    from monolith.compiler.static_fusion import written_buffers
    src='''kernel void gemm_tile(device float* xp [[buffer(2)]], device float* y [[buffer(3)]]) { y[0]=xp[0]; }'''
    p=Program({'k':KernelSpec(src,'gemm_tile',{})},{},
        [OpSpec('k',[(2,'external',0),(3,'middle',0)],(1,1,1),(32,1,1)),
         OpSpec('k',[(2,'middle',0),(3,'result',0)],(1,1,1),(32,1,1))])
    assert written_buffers(p)=={'middle','result'}
    p.ops[1].bindings=[(2,'middle',0),(3,'external',64)]
    assert written_buffers(p)=={'middle','external'}
