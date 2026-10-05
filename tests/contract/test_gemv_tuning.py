"""Explicit GEMV crews preserve shared records and reject unsafe partitions."""
import copy
import struct

import pytest

from monolith.compiler.gemv_tuning import tune_gemv
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


def fixture():
    record = struct.pack('<IIIIf f II',64,4,480,1,1.,1e-5,1,0)
    kernel = KernelSpec('', 'gemv_T',dict(K='256',R='16',T='1',RG='2',RSPLIT='1u',
        STAT_OUT='0',STATIC_GEMV_P_N_SG='480u'))
    return Program({'k':kernel},{'p':BufferSpec(32,record,'params')},[
        OpSpec('k',[(4,'p',0)],(40,1,1),(384,1,1)),
        OpSpec('k',[(4,'p',0)],(40,1,1),(384,1,1))])


def test_private_kernel_and_record_preserve_other_dispatch():
    p=fixture();original=copy.deepcopy(p)
    tune_gemv(p,0,dict(workers=160,sgs=4,rg=4,rsplit=2,x_preconvert=True,x_hoist=True))
    op=p.ops[0];key=op.bindings[0][1]
    assert op.grid==(160,1,1) and op.threadgroup==(128,1,1)
    assert struct.unpack_from('<I',p.buffers[key].init,8)[0]==640
    assert p.kernels[op.kernel].macros['STATIC_GEMV_P_N_SG']=='640u'
    assert p.buffers['p']==original.buffers['p']
    assert p.kernels['k']==original.kernels['k'] and p.ops[1]==original.ops[1]


@pytest.mark.parametrize('config',[dict(workers=0),dict(sgs=33),dict(rg=3),dict(rsplit=32),
    dict(rsplit=4,rg=8),dict(x_preconvert=1),dict(unknown=True)])
def test_invalid_geometry_does_not_mutate_program(config):
    p=fixture();original=copy.deepcopy(p)
    with pytest.raises(ValueError):tune_gemv(p,0,config)
    assert p.kernels==original.kernels and p.buffers==original.buffers and p.ops==original.ops


def test_statistic_extent_cannot_change_without_its_consumer():
    p=fixture();p.kernels['k'].macros['STAT_OUT']='1'
    with pytest.raises(ValueError,match='consumers'):tune_gemv(p,0,dict(rsplit=2))


def test_gate_up_row_pairing_is_preserved():
    p=fixture();p.kernels['k'].macros['EPILOGUE']='2'
    with pytest.raises(ValueError,match='paired'):tune_gemv(p,0,dict(rg=16))


def test_routed_pairs_can_split_rows_but_cannot_hoist_across_experts():
    p=fixture();p.kernels['k'].macros.update(PAIRS='1',X_HOIST='0',EPILOGUE='2')
    tune_gemv(p,0,dict(workers=80,sgs=4,rg=2,rsplit=4))
    assert p.kernels[p.ops[0].kernel].macros['RSPLIT']=='4u'
    with pytest.raises(ValueError,match='hoisting'):
        tune_gemv(p,0,dict(x_hoist=True))
