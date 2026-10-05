"""Opt-in BF16 boundaries must survive all fused norm and MLP implementations."""
import numpy as np
import pytest
from monolith import kernels
from monolith.bench import check_against_oracle
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt
from tests.kernels.test_gemm_tile import Gemm, _rbf, _norm_inputs, EPS, dev
from tests.kernels.test_gemv_fusions import Gemv


@pytest.mark.parametrize('function,fmt,t', [
    ('gemv_T', 'bf16', 1), ('gemv_T', 'bf16', 4),
    ('gemv_T', 'int4_affine', 1), ('gemv_T', 'int4_affine', 4),
    ('gemm_tile', 'bf16', 4), ('gemm_tile', 'int4_affine', 4),
    ('gemv_bf16_rows', 'bf16', 1), ('gemv_bf16_small', 'bf16', 4),
    ('gemv_nvfp4_rows', 'nvfp4', 1),
])
def test_norm_and_silu_intermediate_rounding(dev, function, fmt, t):
    shader = function == 'gemv_T'
    rows, k, n = 16, 1024, 96
    g = Gemv(dev, fmt, n, rows, t) if shader else Gemm(dev, fmt, n, k, 8, function=function)
    count = t if shader else 8
    x, nw, stat, _ = _norm_inputs(np.random.default_rng(71), count, k)
    normed = _rbf(_rbf(bf16_to_f32(x) / np.sqrt(stat[:, None] / k + EPS)) * nw)
    prod = normed[:t].astype(np.float64) @ g.w.T
    block = _rbf(prod).reshape(t, -1, rows)
    gate, up = block[..., :rows//2], block[..., rows//2:]
    expected = _rbf(_rbf(gate / (1 + np.exp(-gate))) * up).reshape(t, -1)
    extra = dict(NORM_ROUND='1', SILU_ROUND='1')
    if function in ('gemv_bf16_rows', 'gemv_bf16_small'):
        extra.update(DIRECT_NORM='1', SHARED_NORM=str(int(t > 1)), STAT_PARTS='1', EPS=str(EPS))
    actual, _ = g.run(x, norm=(stat, 1, nw), epilogue='silu_mul', t_active=t, extra_macros=extra)
    check = check_against_oracle(actual[:t], expected)
    assert check.ok_rounded(), check
    assert not np.any(actual[t:])


def test_standalone_norm_rounding(dev):
    x, nw, stat, _ = _norm_inputs(np.random.default_rng(71), 4, 1024)
    expected = _rbf(_rbf(bf16_to_f32(x) / np.sqrt(stat[:, None] / 1024 + EPS)) * nw)
    pipe = nt.Pipeline(nt.Library(dev, kernels.norm_apply_source(), {'NORM_ROUND': '1'}), 'norm_apply')
    y = nt.Buffer(dev, x.nbytes)
    dispatch = (nt.Dispatch().pipeline(pipe).buffer(0, nt.Buffer(dev, x.tobytes()))
        .buffer(1, nt.Buffer(dev, stat.tobytes())).buffer(2, nt.Buffer(dev, nw.tobytes()))
        .buffer(3, y).bytes(4, kernels.norm_apply_params(1024, 4, 1, EPS)).grid(4).threadgroup(32))
    result = nt.Queue(dev).run([dispatch])
    assert not result.error
    actual = bf16_to_f32(np.frombuffer(y.read(0, x.nbytes), np.uint16)).reshape(x.shape)
    assert check_against_oracle(actual, expected).ok_rounded()
