"""Constant row scales preserve partial ranges, fused outputs and statistics."""
import numpy as np
import pytest

from monolith.formats.fp import f32_to_bf16
from tests.kernels.test_gemm_tile import Gemm, dev


@pytest.mark.parametrize('fmt,function,tokens', [
    ('nvfp4', 'gemv_nvfp4_rows', 1), ('nvfp4', 'gemm_tile', 4),
    ('bf16', 'gemv_bf16_rows', 1), ('bf16', 'gemm_tile', 6),
])
@pytest.mark.parametrize('scale', [1., .375, -1.])
@pytest.mark.parametrize('epilogue', [None, 'residual', 'silu_mul'])
def test_constant_matches_loaded_scale(dev, fmt, function, tokens, scale, epilogue):
    g = Gemm(dev, fmt, 96, 4096 if fmt == 'nvfp4' else 1024, 8,
             function=function, placement='block')
    scales = np.full(96, scale, np.float32)
    g.rsbuf.write(scales.tobytes())
    rng = np.random.default_rng(2026)
    x = f32_to_bf16(rng.normal(0, .1, (8, g.k)).astype(np.float32))
    residual = f32_to_bf16(rng.normal(0, .1, (8, 64)).astype(np.float32))
    kw = dict(t_active=tokens, row_range=(16, 64), epilogue=epilogue,
              residual=residual, out_bf16=True, stat_out=True,
              ksplit=1 if function != 'gemm_tile' else 4)
    loaded = g.run(x, **kw)
    constant = g.run(x, extra_macros={'ROW_SCALE_BITS': f'{int(scales.view(np.uint32)[0])}u'}, **kw)
    fixed_macros = {'ROW_SCALE_BITS': f'{int(scales.view(np.uint32)[0])}u'}
    if function == 'gemm_tile':
        fixed_macros.update(COMPACT_PARTIALS='1', T_HI=str(tokens))
    fixed = g.run(x, fixed_active=True, extra_macros=fixed_macros, **kw)
    for expected, actual in zip(loaded, constant):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    for expected, actual in zip(loaded, fixed):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert not np.any(constant[0][tokens:])
