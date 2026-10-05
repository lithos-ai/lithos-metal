"""Two-row NVFP4 decode: independent dot-product oracle and fused output contracts."""
import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle
from monolith.core import StepStateLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from tests.kernels.test_gemm_tile import Gemm, _rbf, dev


@pytest.mark.parametrize("k,rows,placement", [(4096, 8, "block"), (4096, 16, "block"), (12288, 16, "inline"), (12288, 16, "block"), (12288, 8, "block")])
@pytest.mark.parametrize("epilogue", [None, "residual", "silu_mul"])
@pytest.mark.parametrize("out_bf16", [False, True])
def test_nvfp4_rows_fusions(dev, k, rows, placement, epilogue, out_bf16):
    g = Gemm(dev, "nvfp4", 96, k, 8, rows=rows, placement=placement, function="gemv_nvfp4_rows")
    rng = np.random.default_rng(97)
    x = f32_to_bf16(rng.normal(0, .2, (8, k)).astype(np.float32))
    start, count = (16, 77) if epilogue != "silu_mul" else (16, 80)
    residual = f32_to_bf16(rng.normal(0, .2, (8, count)).astype(np.float32))
    out, stat = g.run(x, t_active=1, epilogue=epilogue, residual=residual, out_bf16=out_bf16,
                      stat_out=True, round_residual=epilogue == "residual", row_range=(start, count))
    assert not np.any(out[1:]) and not np.any(stat[1:])
    prod = bf16_to_f32(x[:1]).astype(np.float64) @ g.w[start:start + count].T
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
    assert np.allclose(stat[:1], expected, rtol=1e-5, atol=1e-6)


def test_nvfp4_rows_permutation_and_state_sources(dev):
    g = Gemm(dev, "nvfp4", 2048, 4096, 8, function="gemv_nvfp4_rows", placement="block")
    x = f32_to_bf16(np.random.default_rng(17).normal(0, .1, (8, 4096)).astype(np.float32))
    layout = StepStateLayout(t_max=8, gamma_max=7)
    natural, _ = g.run(x, t_active=1, epilogue="silu_mul")
    for src, values, extra in [(0, {"t_this_step": 1}, {}), (1, {"n_inject": 1}, {}),
                               (2, {}, {"T_STATIC_ROWS": "1"}), (3, {"n_chain": 1}, {}),
                               (4, {"n_inject": 1, "n_chain": 0}, {})]:
        out, _ = g.run(x, epilogue="silu_mul", step_state=(layout, values), t_range=(0, 1),
                       extra_macros=dict(T_SRC=str(src), **extra, **kernels.perm_out_macros(1024, 32, 256)))
        assert np.array_equal(out, natural[:, kernels.x_permute_columns(1024, 32, 256)])
    for values, trange in [({"t_this_step": 0}, (0, 1)), ({"t_this_step": 4}, (0, 1)),
                           ({"t_this_step": 1, "done": 1}, (0, 1)), ({"t_this_step": 1}, (1, 4))]:
        out, stat = g.run(x, out_bf16=True, stat_out=True, step_state=(layout, values), t_range=trange)
        assert not np.any(out) and not np.any(stat)
