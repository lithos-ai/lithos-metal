"""Small-token SIMD projection: state predicates, range tails, fusion and packed input order."""
import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle
from monolith.core import StepStateLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from tests.kernels.test_gemm_tile import Gemm, _rbf, _norm_inputs, EPS, dev


@pytest.mark.parametrize("parts", [0, 65, 517])
@pytest.mark.parametrize("rows", [8, 16])
@pytest.mark.parametrize("tokens", [0, 1, 3, 4])
@pytest.mark.parametrize("epilogue", [None, "residual", "silu_mul"])
def test_small_bf16_fusions(dev, rows, tokens, epilogue, parts):
    g = Gemm(dev, "bf16", 96, 1024, 8, rows=rows, function="gemv_bf16_small")
    rng = np.random.default_rng(97)
    x = f32_to_bf16(rng.normal(0, .2, (8, 1024)).astype(np.float32))
    norm_kwargs = {}
    xref = bf16_to_f32(x)
    if parts:
        x, nw, statistic, _ = _norm_inputs(rng, 8, 1024)
        pieces = np.repeat((statistic / parts)[:, None], parts, axis=1)
        rstd = 1 / np.sqrt(pieces.astype(np.float64).sum(-1) / 1024 + EPS)
        xref = _rbf(bf16_to_f32(x) * rstd[:, None] * nw[None, :])
        norm_kwargs = dict(norm=(pieces, parts, nw), extra_macros=dict(DIRECT_NORM="1", SHARED_NORM="1", STAT_PARTS=str(parts), EPS=str(EPS)))
    # A nonzero row-range start and a partial final pack block exercise clamped reads.
    start, count = (16, 77) if epilogue == "residual" else (0, 96)
    residual = f32_to_bf16(rng.normal(0, .2, (8, count)).astype(np.float32))
    layout = StepStateLayout(t_max=8, gamma_max=7)
    out, stat = g.run(x, epilogue=epilogue, residual=residual, out_bf16=True,
                      stat_out=True, round_residual=epilogue == "residual", row_range=(start, count),
                      step_state=(layout, {"t_this_step": tokens}), t_range=(0, 4), **norm_kwargs)
    assert np.all(out[tokens:] == 0) and np.all(stat[tokens:] == 0)
    if tokens == 0:
        return
    prod = xref[:tokens].astype(np.float64) @ g.w[start:start + count].T
    if epilogue == "residual":
        ref = _rbf(_rbf(prod) + bf16_to_f32(residual[:tokens]))
    elif epilogue == "silu_mul":
        block = prod.reshape(tokens, -1, rows)
        gate, up = block[..., :rows // 2], block[..., rows // 2:]
        ref = _rbf((gate / (1 + np.exp(-gate)) * up).reshape(tokens, -1))
    else:
        ref = _rbf(prod)
    assert check_against_oracle(out[:tokens], ref).ok_rounded()
    block_width = rows // 2 if epilogue == "silu_mul" else rows
    padded = np.pad(out[:tokens], ((0, 0), (0, (-out.shape[1]) % block_width)))
    expected = (padded.astype(np.float64).reshape(tokens, -1, block_width) ** 2).sum(-1)
    assert np.allclose(stat[:tokens], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shared", [False, True])
def test_small_bf16_permutation_and_state_sources(dev, shared):
    g = Gemm(dev, "bf16", 2048, 1024, 8, function="gemv_bf16_small")
    x = f32_to_bf16(np.random.default_rng(17).normal(0, .1, (8, 1024)).astype(np.float32))
    layout = StepStateLayout(t_max=8, gamma_max=7)
    kwargs = dict(epilogue="silu_mul", step_state=(layout, {"n_inject": 3, "t_this_step": 8}), t_range=(0, 4))
    extra = {}
    if shared:
        x, nw, statistic, _ = _norm_inputs(np.random.default_rng(47), 8, 1024)
        kwargs["norm"] = (statistic[:, None], 1, nw)
        extra = dict(DIRECT_NORM="1", SHARED_NORM="1", STAT_PARTS="1", EPS=str(EPS))
    natural, _ = g.run(x, extra_macros=dict(T_SRC="1", **extra), **kwargs)
    permuted, _ = g.run(x, extra_macros=dict(T_SRC="1", **extra, **kernels.perm_out_macros(1024, 32, 256)), **kwargs)
    assert np.array_equal(permuted, natural[:, kernels.x_permute_columns(1024, 32, 256)])
    assert np.any(natural[:3]) and not np.any(natural[3:])
    for values in ({"t_this_step": 8}, {"t_this_step": 4, "done": 1}):
        out, _ = g.run(x, out_bf16=True, step_state=(layout, values), t_range=(0, 4), extra_macros=extra, norm=kwargs.get("norm"))
        assert not np.any(out)
