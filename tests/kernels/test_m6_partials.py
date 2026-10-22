"""Probability partials must reproduce the original MMA bytes, not a looser oracle."""
import numpy as np
import pytest

from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.nn.kv_quant import quantize
from tests.kernels.test_gqa_decode import Cfg, Harness, dev


@pytest.mark.parametrize('dtype', ['bf16', 'q8_0', 'turbo4'])
@pytest.mark.parametrize('position,t,bound', [(0,1,32), (0,32,32), (17,7,32),
    (63,33,64), (1000,32,128), (2047,65,128), (8192,64,64),
    (16384,128,128), (32768,128,128), (65504,32,32)])
@pytest.mark.parametrize('gate', [False, True])
def test_partition_order_and_cache_bytes(dev, dtype, position, t, bound, gate):
    compare(dev,dtype,position,t,bound,gate)


@pytest.mark.parametrize('dtype',['bf16','turbo4'])
@pytest.mark.parametrize('t',[8,15,16,17])
@pytest.mark.parametrize('gate',[False,True])
def test_runtime_tail_selects_exact_original_partials(dev,dtype,t,gate):
    compare(dev,dtype,32768,t,128,gate,dispatch=True)


@pytest.mark.parametrize('dtype',['bf16','q8_0','turbo4'])
@pytest.mark.parametrize('position,t,bound',[(1024,17,32),(32768,128,128),(65504,32,32)])
@pytest.mark.parametrize('gate',[False,True])
def test_probability_partials_permuted_epilogue(dev,dtype,position,t,bound,gate):
    compare(dev,dtype,position,t,bound,gate,perm=True)


def compare(dev,dtype,position,t,bound,gate,dispatch=False,perm=False,tiled=False):
    rng=np.random.default_rng(914)
    cap=position+bound+3
    harnesses=[]
    for merged in (False,True):
        cfg=Cfg(24,4,256,64,cap,bound,gate=gate,mma=True,m6=True,step_state=True)
        cfg.kv_cache_dtype=dtype
        cfg.pv_merge=merged
        cfg.pv_dispatch=merged and dispatch
        cfg.perm_output=perm
        if merged and tiled and (dtype=='q8_0' or (dtype=='turbo4' and bound>=64)):
            cfg.pv_rows=16
            cfg.pv_simdgroups=8
            cfg.pv_adaptive=dtype=='turbo4'
        harnesses.append(Harness(dev,cfg,np.ones(256,np.float32),np.ones(256,np.float32)))
    a,b=harnesses
    for side in ('k','v'):
        values=bf16_to_f32(f32_to_bf16(rng.normal(0,.5,(cap,4,256)).astype(np.float32)))
        data=f32_to_bf16(values) if dtype=='bf16' else quantize(values,dtype)
        for h in harnesses: getattr(h,side+'_cache').write(data.tobytes(),0)
    projection=f32_to_bf16(rng.normal(size=(bound,a.cfg.n1)).astype(np.float32))
    expected,actual=[h.step(projection,position,t_active=t) for h in harnesses]
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual,expected)
    assert actual.tobytes()==expected.tobytes(), 'PV must preserve output bits, including signed zero'
    for side in ('k','v'):
        nb=getattr(a,side+'_cache_bytes')
        assert getattr(a,side+'_cache').read(0,nb)==getattr(b,side+'_cache').read(0,nb)
    # Causal dependence, repeated overwrite, inactive output and done predicate.
    altered=projection.copy();altered[1:]^=np.uint16(0x8000)
    np.testing.assert_array_equal(b.step(altered,position,t_active=t)[0],actual[0])
    np.testing.assert_array_equal(b.step(projection,position,t_active=t),actual)
    done=b.step(projection,position,t_active=t,state=dict(position=position,t_this_step=t,done=1))
    assert not done.any()


@pytest.mark.parametrize('dtype',['q8_0','turbo4'])
@pytest.mark.parametrize('t,bound',[(16,32),(17,128),(31,32),(32,128),(63,128),(64,64),(65,128),(127,128),(128,128)])
@pytest.mark.parametrize('gate',[False,True])
def test_tiled_pv_runtime_crossover(dev,dtype,t,bound,gate):
    compare(dev,dtype,32768,t,bound,gate,dispatch=True,perm=True,tiled=True)
