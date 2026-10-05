"""Two-row BF16 projection, including normalization at the input load."""
import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle
from monolith.core import StepStateLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from tests.kernels.test_gemm_tile import Gemm, _rbf, _norm_inputs, EPS, dev


@pytest.mark.parametrize("rows_per_sg", [1, 2])
@pytest.mark.parametrize("k,rows,parts", [(1024, 8, 1), (2048, 16, 65), (3584, 16, 448)])
@pytest.mark.parametrize("epilogue", [None, "residual", "silu_mul"])
@pytest.mark.parametrize("out_bf16", [False, True])
def test_bf16_rows_norm_and_fusions(dev, k, rows, parts, epilogue, out_bf16, rows_per_sg):
    g = Gemm(dev, "bf16", 96, k, 8, rows=rows, function="gemv_bf16_rows")
    rng = np.random.default_rng(97)
    x, nw, stat, xnorm = _norm_inputs(rng, 8, k)
    # Include non-power-of-two and many-part reductions from preceding projections.
    pieces = np.repeat((stat / parts)[:, None], parts, axis=1)
    rstd = 1 / np.sqrt(pieces.astype(np.float64).sum(-1) / k + EPS)
    xnorm = _rbf(bf16_to_f32(x) * rstd[:, None] * nw[None, :])
    start, count = (16, 77) if epilogue != "silu_mul" else (16, 80)
    residual = f32_to_bf16(rng.normal(0, .2, (8, count)).astype(np.float32))
    out, stat_out = g.run(x, t_active=1, epilogue=epilogue, residual=residual, out_bf16=out_bf16,
                          stat_out=True, round_residual=epilogue == "residual", row_range=(start, count),
                          norm=(pieces, parts, nw),
                          extra_macros=dict(DIRECT_NORM="1", STAT_PARTS=str(parts), EPS=str(EPS), BF_ROWS=str(rows_per_sg)))
    assert not np.any(out[1:]) and not np.any(stat_out[1:])
    prod = xnorm[:1].astype(np.float64) @ g.w[start:start + count].T
    if epilogue == "residual":
        ref = _rbf(prod) + bf16_to_f32(residual[:1])
    elif epilogue == "silu_mul":
        block = prod.reshape(1, -1, rows)
        gate, up = block[..., :rows // 2], block[..., rows // 2:]
        ref = (gate / (1 + np.exp(-gate)) * up).reshape(1, -1)
    else:
        ref = prod
    if out_bf16:
        ref = _rbf(ref)
    chk = check_against_oracle(out[:1], ref)
    assert chk.ok_rounded() if out_bf16 else chk.ok(), chk
    width = rows // 2 if epilogue == "silu_mul" else rows
    padded = np.pad(out[:1], ((0, 0), (0, (-out.shape[1]) % width)))
    expected = (padded.astype(np.float64).reshape(1, -1, width) ** 2).sum(-1)
    assert np.allclose(stat_out[:1], expected, rtol=1e-5, atol=1e-6)


def test_bf16_rows_permutation_and_state_sources(dev):
    g = Gemm(dev, "bf16", 2048, 1024, 8, function="gemv_bf16_rows")
    x = f32_to_bf16(np.random.default_rng(17).normal(0, .1, (8, 1024)).astype(np.float32))
    layout = StepStateLayout(t_max=8, gamma_max=7)
    natural, _ = g.run(x, t_active=1, epilogue="silu_mul")
    refprod = bf16_to_f32(x[:1]).astype(np.float64) @ g.w.T
    blocks = refprod.reshape(1, -1, 16)
    ref = _rbf((blocks[..., :8] / (1 + np.exp(-blocks[..., :8])) * blocks[..., 8:]).reshape(1, -1))
    assert check_against_oracle(natural[:1], ref).ok_rounded()
    for src, values, extra in [(0, {"t_this_step": 1}, {}), (1, {"n_inject": 1}, {}),
                               (2, {}, {"T_STATIC_ROWS": "1"}), (3, {"n_chain": 1}, {}),
                               (4, {"n_inject": 1, "n_chain": 0}, {})]:
        out, _ = g.run(x, epilogue="silu_mul", step_state=(layout, values), t_range=(0, 1),
                       extra_macros=dict(T_SRC=str(src), **extra, **kernels.perm_out_macros(1024, 8, 256)))
        assert np.array_equal(out, natural[:, kernels.x_permute_columns(1024, 8, 256)])
    for values, trange in [({"t_this_step": 0}, (0, 1)), ({"t_this_step": 4}, (0, 1)),
                           ({"t_this_step": 1, "done": 1}, (0, 1)), ({"t_this_step": 1}, (1, 4))]:
        out, stat = g.run(x, out_bf16=True, stat_out=True, step_state=(layout, values), t_range=trange)
        assert not np.any(out) and not np.any(stat)


@pytest.mark.parametrize("rows,width,step", [(8, 2, 0), (16, 4, 0), (16, 4, 1)])
@pytest.mark.parametrize("active,done", [(1, 0), (0, 0), (1, 1)])
def test_bf16_rows_projection_convolution_state(dev, rows, width, step, active, done):
    """Projection writes raw BF16 rows to state, convolved rows to its private output."""
    from monolith.runtime import _native as nt
    g = Gemm(dev, "bf16", 96, 1024, 8, rows=rows, function="gemv_bf16_rows")
    rng = np.random.default_rng(41)
    x = f32_to_bf16(rng.normal(0, .1, (8, 1024)).astype(np.float32))
    layout = StepStateLayout(t_max=8, gamma_max=7)
    # A nonzero projection range and convolution interval leave two kinds of tails.
    start, dim, count = 8, 64, 77
    state = f32_to_bf16(rng.normal(0, .1, (2, dim, width - 1)).astype(np.float32))
    weight = f32_to_bf16(rng.normal(0, .1, (dim, width)).astype(np.float32))
    sb = nt.Buffer(dev, state.tobytes())
    kwargs = dict(out_bf16=True, row_range=(16, count), step_state=(layout, dict(step=step, t_this_step=active, done=done)), t_range=(0, 1))
    # Use an interval fully inside the tail range: channels [8,72), not slab rows.
    raw, _ = g.run(x, **kwargs)
    out, _ = g.run(x, conv=(sb, weight), extra_macros=dict(PROJ_CONV="1", CONV_START=str(start),
                   CONV_DIM=str(dim), CONV_WIDTH=str(width)), **kwargs)
    got_state = np.frombuffer(sb.read(0, state.nbytes), np.uint16).reshape(state.shape)
    expected = state.copy()
    if not active or done:
        assert not np.any(out)
        assert np.array_equal(got_state, expected)
        return
    rd, wr = step & 1, (step + 1) & 1
    win = np.concatenate([bf16_to_f32(state[rd]), raw[0, start:start + dim, None]], axis=1)
    conv = np.zeros(dim, np.float32)
    wf = bf16_to_f32(weight)
    for j in range(width):
        conv = (win[:, j].astype(np.float64) * wf[:, j] + conv.astype(np.float64)).astype(np.float32)
    conv = _rbf(conv)
    ref = raw.copy()
    ref[0, start:start + dim] = _rbf(conv / (1 + np.exp(-conv)))
    assert check_against_oracle(out[:1], ref[:1]).ok_rounded()
    assert np.array_equal(out[:, :start], raw[:, :start])
    assert np.array_equal(out[:, start + dim:], raw[:, start + dim:])
    expected[wr] = f32_to_bf16(win[:, 1:])
    assert np.array_equal(got_state, expected)
