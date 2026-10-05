"""Check exact-code MLX adapters against independent NumPy dequantization."""
import numpy as np
import pytest
from monolith.formats.fp import e4m3_to_f32, e2m1_to_f32, unpack_nibbles
from tools.bench.modelopt_mlx import scaled_matmul


@pytest.mark.parametrize('mode', ['nvfp4','mxfp8'])
def test_modelopt_scale_and_packing(mode):
    mx = pytest.importorskip('mlx.core')
    rng = np.random.default_rng(141)
    n,k = 48,1024
    scale = np.float32(.000173)
    if mode=='nvfp4':
        codes = rng.integers(0,256,(n,k//2),dtype=np.uint8)
        scales = rng.integers(32,127,(n,k//16),dtype=np.uint8)
        refw=(e2m1_to_f32(unpack_nibbles(codes)).reshape(n,k//16,16)*e4m3_to_f32(scales)[...,None]*scale).reshape(n,k)
    else:
        codes = rng.integers(0,127,(n,k),dtype=np.uint8)
        codes |= rng.integers(0,2,(n,k),dtype=np.uint8)<<7
        scales = np.full((n,k//32),127,dtype=np.uint8)
        refw=e4m3_to_f32(codes)*scale
    x=mx.array(rng.normal(0,.1,(1,8,k)).astype(np.float32),dtype=mx.bfloat16)
    y=scaled_matmul(mx,x,mx.array(codes.view(np.uint32)),mx.array(scales),mx.array(scale),mode)
    got=np.asarray(y.astype(mx.float32)).astype(np.float64)
    ref=np.asarray(x.astype(mx.float32)).astype(np.float64)@refw.astype(np.float64).T
    assert np.isfinite(got).all()
    assert np.linalg.norm(got-ref)/np.linalg.norm(ref)<.007
    assert np.dot(got.ravel(),ref.ravel())/(np.linalg.norm(got)*np.linalg.norm(ref))>.9999
