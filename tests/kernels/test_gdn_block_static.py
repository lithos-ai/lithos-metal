"""Full projection-inclusive static block: compare with its split control."""
import numpy as np
import pytest

from monolith.formats.fp import f32_to_bf16
from monolith.runtime import Engine, _native as nt
from tools.bench.gdn_block_bench import fixture, program, initialize, checked_run, snapshot
from tools.bench.gdn_block_static import normalize, merge


@pytest.fixture(scope='module')
def block(tmp_path_factory):
    pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    dev = nt.Device()
    root = tmp_path_factory.mktemp('gdn-block')
    fixture(root, hidden=1024, hk=8, hv=16)
    # Reopening an existing fixture must restore the mixed FP8/BF16 slabs.
    module, pack = fixture(root, hidden=1024, hk=8, hv=16)
    return dev, module, program(dev, module, pack)


@pytest.mark.parametrize('sgs,workers,tn,split,extra,sync', [
    (4,4,16,False,{},{}), (16,8,16,False,{},{}),
    (8,8,32,True,{},{}), (16,8,32,True,{},{}),
    (16,8,32,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','task_barrier':False}),
    (16,8,16,True,{'compact':True,'gdn_sl':8},{'barrier':'leader','task_barrier':False}),
    (16,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','task_barrier':False,'schedule':'interleave'}),
    (16,8,32,True,{'compact':True,'scalar_sgs':1},{'barrier':'simd','task_barrier':False}),
    (16,8,16,True,{'short_decode':True},{'barrier':'simd','task_barrier':False}),
    (16,8,16,True,{'compact':True,'narrow_weights':True},{'barrier':'simd','task_barrier':False}),
    (16,8,16,True,{'ksplit':8,'compact':True},{'barrier':'simd','task_barrier':False}),
    (16,8,16,True,{'ksplit':2,'compact':True,'ragged_teams':True},{'barrier':'simd','task_barrier':False}),
    (16,8,16,True,{'ksplit':2,'compact':True,'ragged_teams':True},{'barrier':'simd','schedule':'queue','task_grain':'tile','task_seed':True}),
    (16,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','dual_permute':True}),
    (16,8,16,True,{'compact':True,'gdn_sl':8},{'barrier':'simd','dual_permute':True,'gdn_fused_norm':True,'schedule':'queue','task_grain':'tile','task_seed':True}),
    (16,8,16,True,{'compact':True,'gdn_sl':4,'gdn_tp':4},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'gdn_sl':8,'gdn_tp':2},{'barrier':'simd','gdn_fused_norm':True}),
    (16,8,16,True,{'compact':True,'gdn_sl':8},{'barrier':'simd','gdn_fused_norm':True}),
    (32,4,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','gdn_fused_norm':True}),
    (8,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','gdn_prepare':'global'}),
    (16,8,16,True,{'compact':True,'gdn_sl':8},{'barrier':'simd','gdn_fused_norm':True,'gdn_prepare':'global'}),
    (16,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','gdn_vector':2,'gdn_unroll':2}),
    (16,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','gdn_vector':4,'gdn_unroll':4}),
    (16,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','gdn_unroll':8}),
    (4,8,32,True,{'compact':True,'gdn_sl':32},{'barrier':'simd','gdn_fused_norm':True}),
    (8,8,32,True,{'compact':True,'gdn_sl':16},{'barrier':'simd','gdn_fused_norm':True}),
    (2,8,16,True,{'compact':True,'gdn_sl':8,'scalar_sgs':1},{'barrier':'simd'}),
    (1,8,16,True,{'compact':True,'gdn_sl':8,'scalar_sgs':1},{'barrier':'simd'}),
    (8,8,32,True,{'compact':True,'gdn_sl':8,'tk':16},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'tm':32,'tk':16,'fp8_layout':'tile','q_outer':0},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'tm':32,'tk':32,'fp8_layout':'tile','q_outer':0},{'barrier':'simd'}),
    (16,8,32,True,{'compact':True,'tm':32,'tk':16,'fp8_layout':'tile','q_outer':0},{'barrier':'simd'}),
    (16,8,32,True,{'compact':True,'tm':32,'tk':32,'fp8_layout':'tile','q_outer':0},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'gdn_sl':4,'perm_sgs':4},{'barrier':'simd','direct_norm':True}),
    (16,8,16,True,{'compact':True,'gdn_sl':4,'perm_sgs':8},{'barrier':'simd','direct_norm':True,'dual_permute':True}),
    (16,8,16,True,{'compact':True,'gdn_sl':4,'ksplit':1},{'barrier':'simd','schedule':'queue','task_grain':'tile'}),
    (16,8,16,True,{'compact':True,'fp8_decode':'half'},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'fp8_decode':'bits','narrow_weights':True},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'fp8_decode':'lut'},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'fp8_decode':'subtract'},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'fp8_decode':'vector'},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'fp8_decode':'half_operand'},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True},{'barrier':'simd','noinline':True}),
    (16,8,16,True,{'compact':True,'fp8_layout':'tile','q_outer':0},{'barrier':'simd','restrict_weights':True}),
    (16,8,16,True,{'compact':True,'gdn_tp':1},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'fp8_layout':'tile','fp8_decode':'subtract'}, {'barrier':'simd','poll_sgs':2}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','poll_sgs':4}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','flag_stride':32}),
    (16,8,16,True,{'compact':True}, {'barrier':'leader','flag_stride':8}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','poll_sgs':4,'flag_stride':64}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','poll_sgs':4,'flag_stride':32,'arrival':'store'}),
    (16,8,16,True,{'compact':True}, {'barrier':'leader','arrival':'store'}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','arrival':'register'}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','poll_sgs':4,'flag_stride':32,'arrival':'register'}),
    (16,8,16,True,{'compact':True}, {'barrier':'leader','arrival':'register'}),
    (16,8,16,True,{'compact':True}, {'barrier':'simd','poll_sgs':8,'schedule':'queue','task_grain':'tile'}),
    (12,8,16,True,{'compact':True,'gdn_sl':4,'ksplit':4,'ragged_teams':True,'gdn_global':'global'},{'barrier':'simd'}),
    (24,8,16,True,{'compact':True,'gdn_sl':4,'ksplit':8,'ragged_teams':True,'gdn_global':'shared_qk'},{'barrier':'simd','schedule':'queue','task_grain':'tile','task_seed':True}),
    (12,8,16,True,{'compact':True,'gdn_sl':4,'ksplit':4,'ragged_teams':True,'gdn_global':'shared_qk','gemm_overrides':{'2':{'tn':32,'ksplit':2}}},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'q_outer':0},{'barrier':'simd'}),
    (16,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','gdn_prepare':'shared_qk'}),
    (16,8,16,True,{'compact':True,'gdn_sl':8},{'barrier':'simd','gdn_prepare':'shared_qk','gdn_fused_norm':True}),
    (8,8,16,True,{'compact':True,'gdn_sl':4},{'barrier':'simd','schedule':'queue','task_grain':'tile','task_seed':True})])
def test_full_block_fusion_preserves_control_and_state(block, sgs, workers, tn, split, extra, sync):
    dev, module, p = block
    control = normalize(p, sgs, mode='coop', groups=workers, tn=tn, split=split,**extra)
    fused = merge(control, workers, sgs,**sync)
    assert len(fused.ops) == 1
    engines = [Engine(pr, dev) for pr in (control, fused)]
    for e in engines:
        initialize(e, module)
    rng = np.random.default_rng(100)
    for step in range(4):
        x = f32_to_bf16(rng.normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['hidden'].write(x,0)
            e.buffers[e.program.step_state].write(e.program.layout.pack({'step':step,'t_this_step':8}),0)
            checked_run(e,1)
        assert snapshot(engines[0],1024) == snapshot(engines[1],1024)
    for e in engines:
        checked_run(e,64)
    assert snapshot(engines[0],1024) == snapshot(engines[1],1024)


def test_ragged_multi_team_projection_is_rejected(block):
    _,_,p=block
    with pytest.raises(ValueError,match='ragged multi-team'):
        normalize(p,16,groups=8,tn=16,ksplit=2,compact=True)


def test_mixed_nonpower_projection_splits_preserve_state(tmp_path):
    """Input/output widths can use different divisors in the same worker."""
    pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    dev=nt.Device()
    module,pack=fixture(tmp_path,hidden=2560,hk=16,hv=48)
    p=program(dev,module,pack)
    overrides={str(i):dict(ksplit=5 if i<3 else 3,ragged_teams=True) for i in range(4)}
    control=normalize(p,15,groups=8,tn=32,ksplit=1,compact=True,gdn_sl=4,
                      scalar_sgs=2,gdn_global='shared_qk',fp8_layout='tile',
                      q_outer=0,fp8_decode='subtract',gemm_overrides=overrides)
    engines=[Engine(pr,dev) for pr in (control,merge(control,8,15,barrier='simd'))]
    for e in engines:initialize(e,module)
    for step in range(4):
        for e in engines:
            e.buffers[e.program.step_state].write(e.program.layout.pack({'step':step,'t_this_step':8}),0)
            checked_run(e,1)
        assert snapshot(engines[0],2560)==snapshot(engines[1],2560)
    for e in engines:checked_run(e,64)
    assert snapshot(engines[0],2560)==snapshot(engines[1],2560)


@pytest.mark.parametrize('tk',[64,128,256])
def test_staged_reduction_tiles_preserve_state(block,tk):
    dev,module,p=block
    control=normalize(p,4,mode='staged',groups=8,tn=16,split=True,compact=True,staged_tk=tk)
    engines=[Engine(pr,dev) for pr in (control,merge(control,8,4,barrier='simd'))]
    for e in engines:
        initialize(e,module)
        checked_run(e,64)
    assert snapshot(engines[0],1024)==snapshot(engines[1],1024)


@pytest.mark.parametrize('mode',['half','bits','lut','subtract','vector','half_operand'])
def test_fp8_operand_decoders_preserve_every_finite_code(mode):
    from monolith import kernels
    from monolith.compiler.static_fusion import fp8_tile_decoder
    from monolith.formats import FORMATS
    from monolith.formats.fp import e4m3_to_f32
    dev=nt.Device()
    src=kernels.PRELUDE+FORMATS.get('fp8_e4m3').msl_decode+fp8_tile_decoder(mode)+'''
kernel void values(device ushort* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i]=as_type<ushort>(bfloat(fp8_tile_value(i)));
}
'''
    pso=nt.Pipeline(nt.Library(dev,src),'values')
    out=nt.Buffer(dev,256*2)
    result=nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0,out).grid(8).threadgroup(32)])
    assert not result.error,result.error
    codes=np.arange(256,dtype=np.uint8);finite=(codes&127)!=127
    expected=f32_to_bf16(e4m3_to_f32(codes))
    actual=np.frombuffer(out.read(0,256*2),dtype=np.uint16)
    np.testing.assert_array_equal(actual[finite],expected[finite])


@pytest.mark.parametrize('tn,ks,outer,decoder,mode,tk,prefetch,tile_block',[
    (16,16,1,'standard','coop',32,0,1),(32,4,0,'standard','coop',32,0,1),
    (16,2,0,'half_operand','coop',32,0,1),(32,1,1,'vector','coop',32,0,1),
    (32,4,0,'standard','coop',16,0,1),(16,4,0,'standard','staged',64,0,1),
    (32,4,1,'subtract','staged',128,0,1),(16,4,1,'half_operand','staged',256,0,1),
    (16,16,0,'subtract','coop',32,1,2),(16,16,0,'subtract','coop',32,2,8),
    (32,4,1,'half_operand','coop',32,4,32),
    (16,16,0,'subtract','coop',32,0,512),(32,4,1,'subtract','staged',128,0,8)])
def test_fp8_tile_pack_is_lossless_in_the_complete_mixer(block,tn,ks,outer,decoder,mode,tk,prefetch,tile_block):
    dev,module,p=block
    cfg=dict(groups=8,tn=tn,ksplit=ks,compact=True,ragged_teams=True,q_outer=outer,
             fp8_decode=decoder,mode=mode)
    cfg.update({'tk':tk} if mode=='coop' else {'staged_tk':tk})
    sgs=16 if mode=='coop' else 4
    original=normalize(p,sgs,**cfg)
    packed=normalize(p,sgs,fp8_layout='tile',weight_prefetch=prefetch,fp8_tile_block=tile_block,**cfg)
    fused=merge(packed,8,sgs,barrier='simd')
    engines=[Engine(pr,dev) for pr in (original,packed,fused)]
    for e in engines:initialize(e,module)
    for step in range(4):
        for e in engines:
            e.buffers[e.program.step_state].write(e.program.layout.pack({'step':step,'t_this_step':8}),0)
            checked_run(e,1)
        assert snapshot(engines[0],1024)==snapshot(engines[1],1024)==snapshot(engines[2],1024)
    for e in engines:checked_run(e,64)
    assert snapshot(engines[0],1024)==snapshot(engines[1],1024)==snapshot(engines[2],1024)


@pytest.mark.parametrize('storage',['bf16','half'])
def test_predecoded_fp8_storage_preserves_values(block,storage):
    dev,module,p=block
    cfg=dict(groups=8,tn=16,ksplit=16,compact=True,q_outer=0)
    original=normalize(p,16,**cfg)
    packed=normalize(p,16,fp8_layout='tile',fp8_storage=storage,**cfg)
    engines=[Engine(pr,dev) for pr in (original,packed,merge(packed,8,16,barrier='simd'))]
    for e in engines:
        initialize(e,module)
        checked_run(e,64)
    assert snapshot(engines[0],1024)==snapshot(engines[1],1024)==snapshot(engines[2],1024)
