"""One-row crews preserve projections and expose correctly sized norm partials."""
import numpy as np
import pytest

from monolith.core import StepStateLayout
from monolith.formats.fp import f32_to_bf16
from tests.kernels.test_gemm_tile import Gemm, dev


@pytest.mark.parametrize('placement', ['inline', 'block'])
@pytest.mark.parametrize('epilogue', [None, 'residual', 'silu_mul'])
@pytest.mark.parametrize('row_range', [None, (16, 64)])
def test_one_row_groups_and_statistic_layout(dev, placement, epilogue, row_range):
    n = 96 if epilogue == 'silu_mul' else 100
    g = Gemm(dev, 'nvfp4', n, 4096, 8, function='gemv_nvfp4_rows', placement=placement)
    rng = np.random.default_rng(51)
    x = f32_to_bf16(rng.normal(0, .1, (8, g.k)).astype(np.float32))
    count = row_range[1] if row_range else n
    residual = f32_to_bf16(rng.normal(0, .1, (8, count)).astype(np.float32))
    kwargs = dict(t_active=1, epilogue=epilogue, row_range=row_range,
                  residual=residual, out_bf16=True, stat_out=True)
    baseline, _ = g.run(x, **kwargs)
    groups = 16 if epilogue == 'silu_mul' else 8
    out, stat = g.run(x, fixed_active=True, extra_macros={
        'NV_ROWS': '1u', 'NV_SG': f'{groups}u', 'NV_UNROLL': '4'}, **kwargs)
    np.testing.assert_array_equal(out, baseline)
    assert not np.any(out[1:]) and not np.any(stat[1:])
    # Each group contributes eight output rows, also for the paired gate/up.
    padded = np.pad(out[0].astype(np.float64), (0, stat.shape[1] * 8 - out.shape[1]))
    np.testing.assert_allclose(stat[0], (padded.reshape(-1, 8) ** 2).sum(-1), rtol=1e-6)


@pytest.mark.parametrize('parts', [1, 65, 517])
@pytest.mark.parametrize('active,done', [(0, 0), (1, 0), (3, 0), (4, 0), (7, 0), (4, 1)])
@pytest.mark.parametrize('k', [4096, 12288])
@pytest.mark.parametrize('groups,simdgroups,unroll', [(4, 16, 4), (1, 64, 1), (2, 128, 1)])
def test_shared_permute_norm_preserves_runtime_rows(dev, parts, active, done, k, groups, simdgroups, unroll):
    g = Gemm(dev, 'nvfp4', 96, k, 8, placement='block')
    rng = np.random.default_rng(61)
    x = f32_to_bf16(rng.normal(0, .1, (8, g.k)).astype(np.float32))
    norm = (rng.uniform(.01, 2, (8, parts)).astype(np.float32), parts,
            rng.uniform(.7, 1.3, g.k).astype(np.float32))
    state = (StepStateLayout(t_max=8, gamma_max=7), {'t_this_step': active, 'done': done})
    kwargs = dict(norm=norm, step_state=state, ksplit=4, out_bf16=True)
    old, _ = g.run(x, **kwargs)
    shared, _ = g.run(x, perm_groups=groups, perm_simdgroups=simdgroups, perm_unroll=unroll, **kwargs)
    np.testing.assert_array_equal(shared, old)
    assert not np.any(shared[0 if done else active:])
