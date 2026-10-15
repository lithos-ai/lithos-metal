import copy

import pytest

from monolith.backends.metal.m6.backend import finalize
from monolith.runtime.program import BufferSpec
from tests.contract.test_m6_attention import fixture


@pytest.mark.parametrize('dtype',[0,1,2,3])
@pytest.mark.parametrize('t',[32,64,128])
@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('explicit_override',[False,True])
@pytest.mark.parametrize('tiled',[False,True])
def test_probability_partials_reuse_workspace_and_guard_merge(dtype,t,enabled,explicit_override,tiled):
    import struct
    config,p=fixture(K_CACHE_TYPE=f'{dtype}u',V_CACHE_TYPE=f'{dtype}u')
    config.raw['engine']['attention_probability_partials']=enabled
    config.raw['engine']['attention_pv_tiling']=not tiled if explicit_override else tiled
    record=bytearray(p.buffers['params'].init)
    struct.pack_into('<I',record,8,t)
    p.buffers['params'].init=bytes(record)
    p.ops[0].bindings.extend([(7,'partial',0),(8,'md',0),(2,'v',0)])
    p.ops[1].bindings.extend([(0,'partial',0),(1,'md',0)])
    # The real model has a separate gate projection and merge parameter block.
    merge_record=bytearray(record)
    struct.pack_into('<I',merge_record,9*4,6144)
    struct.pack_into('<I',merge_record,14*4,1)
    p.buffers['merge_params']=BufferSpec(80,bytes(merge_record),'params')
    p.ops[1].bindings=[(slot,'merge_params' if slot==4 else name,off) for slot,name,off in p.ops[1].bindings]
    p.kernels['merge'].macros.update(PERM_OUT='1',PERM_K='6144')
    for name in ['partial','md','v']:p.buffers[name]=BufferSpec(4096)
    buffers=copy.deepcopy(p.buffers)
    qualification={'pv_tiling':tiled} if explicit_override else {}
    qualified=enabled and dtype in ((0,1,2) if tiled else (0,2))
    finalize(p,None,{},config,t=t,dynamic_t=True,**qualification)
    names=[p.kernels[o.kernel].function for o in p.ops]
    assert ('gqa_pv_merge_m6' in names)==qualified
    assert p.buffers.keys()==buffers.keys()
    assert {n:b.nbytes for n,b in p.buffers.items()}=={n:b.nbytes for n,b in buffers.items()}
    if qualified:
        assert names==['gqa_decode','gqa_decode_mma_m6','gqa_decode_mma_m6','gqa_merge','gqa_pv_merge_m6'] + (['gqa_pv_merge_m6'] if tiled and dtype==2 and t>=64 else [])
        assert p.kernels[p.ops[3].kernel].macros['M6_PV_MERGE_DISPATCH']=='1'
        assert p.kernels[p.ops[1].kernel].macros['M6_PV_FALLBACK']=='1'
        assert p.kernels[p.ops[2].kernel].macros['M6_PV_MERGE']=='1'
        assert p.ops[4].barrier_before
        assert (5,'v',0) in p.ops[4].bindings
        macros=p.kernels[p.ops[4].kernel].macros
        assert macros['STATIC_GQA_P_HAS_GATE']=='1u'
        assert macros['STATIC_GQA_P_IN_STRIDE']=='6144u'
        assert macros['PERM_OUT']=='1'
        wide=tiled and (dtype==1 or (dtype==2 and t>=64))
        last=p.kernels[p.ops[-1].kernel].macros
        assert ('PV_QM' in last)==wide
        assert p.ops[-1].grid==(4*((t*6+(16 if dtype==0 or wide else 8)-1)//(16 if dtype==0 or wide else 8)),1,1)
        if tiled and dtype==2 and t>=64:
            assert macros['PV_ACTIVE_MAX']=='64'
            assert last['PV_ACTIVE_MIN']=='64'
            assert 'PV_QM' not in macros
            assert p.ops[4].bindings==p.ops[5].bindings
            assert p.ops[4].grid==(4*((t*6+7)//8),1,1)
    else:
        assert names==(['gqa_decode','gqa_decode_mma_m6','gqa_merge'] if dtype!=3 else ['gqa_decode','gqa_merge'])
    saved=p.to_json()
    finalize(p,None,{},config,t=t,dynamic_t=True,**qualification)
    assert p.to_json()==saved
