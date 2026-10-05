"""Every NVFP4 reader must preserve the old layout's results, including fallback paths."""
import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import f32_to_bf16
from monolith.runtime import _native as nt
from tests.kernels.test_gemm_tile import Gemm, dev


@pytest.mark.parametrize('k', [1024, 4096, 12288])
@pytest.mark.parametrize('function,tokens', [('gemv_nvfp4_rows', 1), ('gemm_tile', 4), ('gemm_tile', 8)])
@pytest.mark.parametrize('epilogue', [None, 'residual', 'silu_mul'])
def test_payload_order_row_matrix_fusions(dev, k, function, tokens, epilogue):
    rng = np.random.default_rng(83)
    x = f32_to_bf16(rng.normal(0, .1, (8, k)).astype(np.float32))
    residual = f32_to_bf16(rng.normal(0, .1, (8, 64)).astype(np.float32))
    results = []
    for order in ('lane', 'payload'):
        g = Gemm(dev, 'nvfp4', 96, k, 8, placement='block', scale_order=order, function=function)
        results.append(g.run(x, t_active=tokens, row_range=(16, 64), epilogue=epilogue,
            residual=residual, out_bf16=True, stat_out=True, ksplit=1 if function != 'gemm_tile' else 4))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize('k', [1024, 4096, 12288])
@pytest.mark.parametrize('tokens,rsplit', [(1, 1), (1, 8), (4, 1)])
def test_payload_order_legacy_gemv(dev, k, tokens, rsplit):
    rng = np.random.default_rng(89)
    x = f32_to_bf16(rng.normal(0, .1, (tokens, k)).astype(np.float32))
    outputs = []
    for order in ('lane', 'payload'):
        g = Gemm(dev, 'nvfp4', 100, k, 8, placement='block', scale_order=order)
        macros = kernels.gemv_macros(g.info, t=tokens, out_bf16=True, rg=1, rsplit=rsplit)
        pso = nt.Pipeline(nt.Library(dev, kernels.gemv_source('nvfp4'), macros), 'gemv_T')
        out = nt.Buffer(dev, tokens * g.n * 2); out.fill(0)
        nsg = 12 * dev.info().gpu_cores
        dispatch = (nt.Dispatch().pipeline(pso).buffer(0, g.wbuf).buffer(1, g.rsbuf)
            .buffer(2, nt.Buffer(dev, x.tobytes())).buffer(3, out)
            .bytes(4, kernels.gemv_params(g.n, g.info.n_blocks, nsg, tokens))
            .grid(dev.info().gpu_cores).threadgroup(384))
        result = nt.Queue(dev).run([dispatch])
        assert not result.error, result.error
        outputs.append(out.read(0, tokens * g.n * 2))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('tm,tn,tk,ksplit', [(8, 64, 64, 1), (16, 32, 128, 2), (32, 16, 256, 4)])
def test_payload_order_other_matrix_tiles(dev, tm, tn, tk, ksplit):
    x = f32_to_bf16(np.random.default_rng(91).normal(0, .1, (tm, 4096)).astype(np.float32))
    results = []
    for order in ('lane', 'payload'):
        g = Gemm(dev, 'nvfp4', 100, 4096, tm, placement='block', scale_order=order)
        results.append(g.run(x, t_active=tm - 1, out_bf16=True, ksplit=ksplit,
            extra_macros={'TN': f'{tn}u', 'TK': f'{tk}u'}))
    np.testing.assert_array_equal(results[0][0], results[1][0])
