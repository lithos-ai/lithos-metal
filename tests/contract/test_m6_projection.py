"""Projection changes are limited to measured M6 prompt shapes and bounds."""
import copy
import struct

import pytest

from monolith.backends.metal import config_for_device
from monolith.backends.metal.m6.projection import finalize
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


def fixture(t, down):
    config=copy.deepcopy(config_for_device(12,11,'Apple M6'))
    config.raw['engine']['nvfp4_prefill_geometry']=True
    n,k=(5120,17408) if down else (34816,5120)
    macros=dict(TM='32',TN='32u',TK='128u',STEP_STATE='1',T_SRC='0',T_LO='1',T_HI=str(t),
        LANE_ORDER='1',EPILOGUE='1' if down else '2',SCALE_CACHE='0' if down else '1',
        R='16',Q_OUTER='1',GROUP_SEG='16',OUT_BF16='1',
        UNIT_WORDS='20' if down else '6',PAYLOAD_WORDS='17' if down else '5',
        SCALE_W0='17' if down else '5',SCALE_UOFF='0',SCALE_WORDS='3' if down else '1',
        STATIC_GEMM_P_N_TILES=f'{n//32}u',STATIC_GEMM_P_N_SG='144u',STATIC_GEMM_P_TILE0='0u')
    record=struct.pack('<IIIIfIII',n,n//32,144,t,1.,0,n//16,0)
    op=OpSpec('gemm',[(4,'params',0)],(12,(t+31)//32,1),(384,1,1),
        meta=dict(prefill=True,format='nvfp4',t_variant=t,n=n,k=k))
    return config,Program({'gemm':KernelSpec('', 'gemm_tile', macros)},
        {'params':BufferSpec(len(record),record,'params')},[op])


@pytest.mark.parametrize('t',[32,64,128])
@pytest.mark.parametrize('down',[False,True])
def test_measured_shapes_keep_reduction_and_allocation(t,down):
    config,program=fixture(t,down)
    before=copy.deepcopy(program)
    finalize(program,None,{},config,t=t,dynamic_t=True)
    if t==32 and not down:
        assert program.to_json()==before.to_json()
        return
    op=program.ops[0];kernel=program.kernels[op.kernel]
    assert op.grid==((48 if down else 24),(t+31)//32,1)
    assert op.threadgroup==((256 if down else 128),1,1)
    assert kernel.macros['TK']=='128u'
    assert kernel.macros['KSPLIT']=='1u'
    assert kernel.macros['SCALE_CACHE']=='0'
    assert kernel.source==before.kernels['gemm'].source
    assert op.bindings==before.ops[0].bindings
    assert sum(b.nbytes for b in program.buffers.values())==sum(b.nbytes for b in before.buffers.values())
    once=program.to_json()
    finalize(program,None,{},config,t=t,dynamic_t=True)
    assert program.to_json()==once


@pytest.mark.parametrize('case',['off','other_chip','cores','static','decode','draft','unknown_shape',
    'fp8','bound','tile','custom_grid','custom_crew','custom_partition','custom_epilogue','accelerator','packed',
    'pack_rows','group_seg','scale_layout'])
def test_fallback(case):
    config,program=fixture(128,True)
    op=program.ops[0];macros=program.kernels['gemm'].macros
    opts=dict(t=128,dynamic_t=True)
    if case=='off':config.raw['engine']['nvfp4_prefill_geometry']=False
    if case=='other_chip':config=copy.deepcopy(config_for_device(40,10,'Apple M5 Max'));config.raw['engine']['nvfp4_prefill_geometry']=True
    if case=='cores':config.gpu_cores=20
    if case=='static':opts['dynamic_t']=False
    if case=='decode':op.meta['prefill']=False
    if case=='draft':macros['T_SRC']='1'
    if case=='unknown_shape':op.meta['n']=1024
    if case=='fp8':op.meta['format']='fp8'
    if case=='bound':opts['t']=256
    if case=='tile':macros['TM']='64'
    if case=='custom_grid':op.grid=(24,4,1)
    if case=='custom_crew':op.threadgroup=(256,1,1)
    if case=='custom_partition':macros['T_LO']='8'
    if case=='custom_epilogue':macros['EPILOGUE']='0'
    if case=='accelerator':opts['accelerator']='off'
    if case=='packed':macros['PACKED_NVFP4']='1'
    if case=='pack_rows':macros['R']='32'
    if case=='group_seg':macros['GROUP_SEG']='32'
    if case=='scale_layout':macros['SCALE_W0']='0'
    before=program.to_json()
    finalize(program,None,{},config,**opts)
    assert program.to_json()==before
