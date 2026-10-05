"""The MoE kernels against numpy (ops/moe.py, #46): moe_route (softmax, top-k with ties to the lowest index, the
renormalization, BF16-valued weights), gemv_T's pairs mode (expert blocks addressed through the ids; the token-row and
the per-slot-row inputs; the silu·mul epilogue) and moe_combine (the weighted sum, the gated shared expert, the
residual, one rounding)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith import kernels  # noqa: E402
from monolith.bench import check_against_oracle, pack_spec, random_spec  # noqa: E402
from monolith.formats import FORMATS, PackLayout  # noqa: E402
from monolith.formats.fp import bf16_to_f32, f32_to_bf16  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _run(dev, ds):
    r = nt.Queue(dev).run(ds if isinstance(ds, list) else [ds])
    assert not r.error, r.error


def _bf16_round(x):
    return bf16_to_f32(f32_to_bf16(np.asarray(x, np.float32)))


def route_ref(logits_bf16, k, renorm):
    """The reference routing: softmax in FP32, top-k (ties to the lowest index), optional renormalization, weights
    rounded to BF16."""
    p = np.exp(logits_bf16 - logits_bf16.max(-1, keepdims=True)).astype(np.float32)
    p = p / p.sum(-1, keepdims=True)
    ids = np.zeros((p.shape[0], k), np.int32)
    w = np.zeros((p.shape[0], k), np.float32)
    for t in range(p.shape[0]):
        rem = p[t].copy()
        for j in range(k):
            e = int(np.argmax(rem))                                 # argmax returns the lowest index among ties
            ids[t, j], w[t, j] = e, rem[e]
            rem[e] = -1.0
        if renorm:
            w[t] /= w[t].sum()
    return ids, _bf16_round(w)


@pytest.mark.parametrize("n_experts,k,renorm", [(8, 2, True), (60, 4, False), (128, 8, True), (5, 5, False)])
def test_moe_route(dev, n_experts, k, renorm):
    rng = np.random.default_rng(1)
    t = 5
    logits = _bf16_round(rng.standard_normal((t, n_experts)) * 2)
    logits[1, 3] = logits[1, 0]                                     # a tie: the lower index wins
    lib = nt.Library(dev, kernels.moe_route_source(), kernels.moe_route_macros(n_experts, renorm))
    pso = nt.Pipeline(lib, "moe_route")
    lb = nt.Buffer(dev, f32_to_bf16(logits).tobytes())
    ids, w = nt.Buffer(dev, (t + 1) * k * 4), nt.Buffer(dev, (t + 1) * k * 4)
    ids.fill(0xFF); w.fill(0xFF)
    d = nt.Dispatch().pipeline(pso).buffer(0, lb).buffer(1, ids).buffer(2, w).bytes(3, kernels.moe_route_params(n_experts, k, t)).grid(t + 1).threadgroup(32)
    _run(dev, d)                                                    # one SIMD-group beyond t_active: untouched
    got_ids = np.frombuffer(ids.read(0, (t + 1) * k * 4), dtype=np.int32).reshape(t + 1, k)
    got_w = np.frombuffer(w.read(0, (t + 1) * k * 4), dtype=np.float32).reshape(t + 1, k)
    ref_ids, ref_w = route_ref(logits, k, renorm)
    assert np.array_equal(got_ids[:t], ref_ids), (got_ids[:t], ref_ids)
    assert np.allclose(got_w[:t], ref_w, rtol=0, atol=2 ** -8 * ref_w.max()) and np.array_equal(got_w[:t], _bf16_round(got_w[:t]))
    assert np.all(got_ids[t] == -1) and np.all(np.frombuffer(w.read(t * k * 4, k * 4), dtype=np.uint32) == 0xFFFFFFFF)
    if renorm:
        assert np.allclose(got_w[:t].sum(-1), 1.0, atol=1e-2)


@pytest.mark.parametrize("x_slot,silu", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("rsplit", [1, 2, 4, 8])
@pytest.mark.parametrize("fmt", ["bf16", "nvfp4"])
def test_gemv_pairs_mode_streams_the_chosen_experts(dev, x_slot, silu, rsplit, fmt):
    """Items (token, slot, block) over the ids: y[t, slot·n_out + …] = x_row · W_{ids[t, slot]}ᵀ (silu·mul over the
    chunk-interleaved gate|up rows with the epilogue); x_row = x[t] or, per slot, x[t·k + slot]."""
    rng = np.random.default_rng(2)
    E, rows_per_expert, K, R, t, k = 6, 64, 256, 16, 3, 2
    spec = random_spec(fmt, E * rows_per_expert, K, rng)
    data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=R))
    W = FORMATS.get(fmt).dequantize(spec)                                              # [E·rows, K] exact
    n_in_rows = t * k if x_slot else t
    x = _bf16_round(rng.uniform(-1, 1, size=(n_in_rows, K)))
    ids = np.array([[0, 5], [3, 3], [2, 1]], np.int32)
    epilogue = "silu_mul" if silu else None
    n_out = rows_per_expert // 2 if silu else rows_per_expert
    macros = kernels.gemv_macros(pinfo, t=1, rg=1, rsplit=rsplit, epilogue=epilogue, out_bf16=True, pairs=(k, rows_per_expert // R, x_slot))
    assert macros["PAIRS"] == "1" and macros["T"] == "1"
    pso = nt.Pipeline(nt.Library(dev, kernels.gemv_source(fmt), macros), "gemv_T")
    n_sg = 12 * dev.info().gpu_cores
    y = nt.Buffer(dev, t * k * n_out * 2); y.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, data)).buffer(1, nt.Buffer(dev, row_scales.tobytes()))
         .buffer(2, nt.Buffer(dev, f32_to_bf16(x).tobytes())).buffer(3, y)
         .bytes(4, kernels.gemv_params(rows_per_expert, rows_per_expert // R, n_sg, t)).buffer(9, nt.Buffer(dev, ids.tobytes()))
         .grid(-(-(n_sg * 32) // 384)).threadgroup(384))
    _run(dev, d)
    got = bf16_to_f32(np.frombuffer(y.read(0, t * k * n_out * 2), dtype=np.uint16)).reshape(t, k * n_out)
    ref = np.zeros((t, k * n_out), np.float32)
    for tt in range(t):
        for j in range(k):
            e = ids[tt, j]
            xr = x[tt * k + j] if x_slot else x[tt]
            acc = W[e * rows_per_expert: (e + 1) * rows_per_expert].astype(np.float64) @ xr.astype(np.float64)
            if silu:                                                                       # chunk-interleaved: 8 gate rows, then 8 up rows per block
                acc = acc.reshape(-1, 2, R // 2)
                gate, up = acc[:, 0, :].ravel(), acc[:, 1, :].ravel()
                acc = gate / (1 + np.exp(-gate)) * up
            ref[tt, j * n_out: (j + 1) * n_out] = acc
    chk = check_against_oracle(got, ref)
    assert chk.ok_rounded() and chk.max_ulp_elementwise <= 1, (chk, x_slot, silu)   # BF16 outputs: within a ULP of their own magnitude


@pytest.mark.parametrize("shared,residual", [(False, True), (True, True), (True, False)])
def test_moe_combine(dev, shared, residual):
    rng = np.random.default_rng(3)
    t, k, H = 4, 3, 96
    h = _bf16_round(rng.standard_normal((t, k * H)))
    w = _bf16_round(rng.uniform(0, 1, size=(t, k)))
    sh = _bf16_round(rng.standard_normal((t, H)))
    gate = _bf16_round(rng.standard_normal((t, 1)))
    res = _bf16_round(rng.standard_normal((t, H)))
    lib = nt.Library(dev, kernels.moe_combine_source(), kernels.moe_combine_macros(shared, residual))
    pso = nt.Pipeline(lib, "moe_combine")
    out = nt.Buffer(dev, (t + 1) * H * 2); out.fill(0)
    bufs = [nt.Buffer(dev, f32_to_bf16(a).tobytes()) for a in (h, sh, gate, res)]
    d = (nt.Dispatch().pipeline(pso).buffer(0, bufs[0]).buffer(1, nt.Buffer(dev, w.tobytes())).buffer(2, bufs[1]).buffer(3, bufs[2]).buffer(4, bufs[3])
         .buffer(5, out).bytes(6, kernels.moe_combine_params(H, k, t)).grid(t + 1).threadgroup(32))
    _run(dev, d)
    got = bf16_to_f32(np.frombuffer(out.read(0, (t + 1) * H * 2), dtype=np.uint16)).reshape(t + 1, H)
    ref = np.zeros((t, H), np.float64)
    for j in range(k):
        ref += w[:, j: j + 1] * h[:, j * H: (j + 1) * H]
    if shared:
        ref += _bf16_round(1 / (1 + np.exp(-gate)))[:, :1] * sh
    if residual:
        ref += res
    chk = check_against_oracle(got[:t], ref)
    assert chk.ok_rounded() and chk.max_ulp_elementwise <= 1, chk
    assert np.all(got[t] == 0)                                                            # beyond t_active: untouched
