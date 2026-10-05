"""The accelerator GEMM tile (plan M9, #50): the cooperative-tensor register layout the fill assumes, read back from
the tensor-ops API for every thread and element; then gemm_tile against the format oracle on every format, both lane
orders, TM = 8 / 16 / 32 with T_act < TM, BF16 and float outputs, and the x_permute column order. The reference holds
the dequantized weights in BF16 — the accelerator's operand dtype and the HF reference model's parameter dtype (exact
for NVFP4, FP8 and BF16; INT8 and affine INT4 products are rounded once, unlike the FP32-dequant GEMV path)."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle, pack_spec, random_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt

K = 1024


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _lib(dev, fmt, macros):
    return nt.Library(dev, kernels.gemm_source(fmt), macros, language_version=kernels.MSL_TENSOR_OPS)


@pytest.mark.parametrize("tm,tn,tk", [(8, 64, 64), (16, 64, 64), (32, 64, 64), (8, 32, 128), (16, 16, 256), (32, 16, 256), (8, 16, 64), (8, 16, 128)])
def test_cooperative_layout_matches_the_fill_formula(dev, tm, tn, tk):
    """Thread ``lane`` holds runs of 4 consecutive inner coordinates at 4·(bit0 + 2·bit3) + 16·jump for outer rows
    (bits 1, 2, 4) + 8·slot. The right operand (inner = k, outer = n) orders its elements q, slot (8), jump; the
    destination (inner = n, outer = m) orders them q, slot (2), jump, 16-row block (valid iff m < TM)."""
    spec = random_spec("bf16", 64, K, np.random.default_rng(0))
    _, info, _ = pack_spec(spec, PackLayout(rows=16))
    macros = kernels.gemm_macros(info, tm=tm, tn=tn, tk=tk)
    pso = nt.Pipeline(_lib(dev, "bf16", macros), "coop_layout")
    stride = 4 + 3 * 1024
    out = nt.Buffer(dev, 32 * stride * 4)
    out.fill(0)
    r = nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0, out).grid(1).threadgroup(32)])
    assert not r.error, r.error
    a = np.frombuffer(out.read(0, 32 * stride * 4), dtype=np.int32).reshape(32, stride)
    nb_c, ns_b, nj_c = max(tm, 16) // 16, tn // 8, tn // 16
    for lane in range(32):
        cap_b, cap_c = int(a[lane, 0]), int(a[lane, 1])
        assert cap_b == tn * tk // 32 and cap_c == 8 * nj_c * nb_c
        c0b = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1))
        c1b = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1)
        rec = a[lane, 4:4 + 3 * cap_b].reshape(cap_b, 3)
        for i in range(cap_b):
            q, s, jump = i & 3, (i >> 2) % ns_b, (i >> 2) // ns_b
            assert tuple(rec[i]) == (1, c0b + 16 * jump + q, c1b + 8 * s), (lane, i, rec[i])
        rec = a[lane, 4 + 3 * cap_b:4 + 3 * cap_b + 3 * cap_c].reshape(cap_c, 3)
        for i in range(cap_c):
            q, s2, jump, blk = i & 3, (i >> 2) & 1, ((i >> 2) >> 1) % nj_c, ((i >> 2) >> 1) // nj_c
            n, m = c0b + 16 * jump + q, 16 * blk + c1b + 8 * s2
            assert tuple(rec[i]) == (1 if m < tm else 0, n, m), (lane, i, rec[i])


def test_x_permute_column_order():
    kl = K // 32
    perm = kernels.x_permute_columns(K, 32, tk=64)
    assert sorted(perm.tolist()) == list(range(K))
    # slot (bit0, bit3, jump, q) of a tile <-> pack column (TK/4)·(bit0 + 2·bit3) + 4·jump + q; pack column 32 = lane 1's word 0
    assert perm[:4].tolist() == [0, 1, 2, 3] and perm[4] == 16 and perm[8] == kl and perm[16] == 4 and perm[64] == 2 * kl
    perm = kernels.x_permute_columns(K, 32, tk=256)                     # the default tile: quarters of 64 columns
    assert sorted(perm.tolist()) == list(range(K))
    assert perm[4] == 2 * kl and perm[8] == 4 * kl and perm[16] == 4 and perm[64] == 16 and perm[256] == 8 * kl


def _run(dev, fmt, n, k, tm, t_act, lane_order, out_bf16=False, rows=16, tn=None, tk=None, ksplit=1, placement="inline", specialize=False):
    rng = np.random.default_rng(5)
    spec = random_spec(fmt, n, k, rng)
    data, info, row_scales = pack_spec(spec, PackLayout(rows=rows, lane_order=lane_order, scale_placement=placement))
    f = FORMATS.get(fmt)
    x = rng.uniform(-1, 1, size=(t_act, k)).astype(np.float32)
    xb = f32_to_bf16(x)
    macros = kernels.gemm_macros(info, tm=tm, out_bf16=out_bf16, tn=tn, tk=tk, ksplit=ksplit)
    tn, tk = int(macros["TN"].rstrip("u")), int(macros["TK"].rstrip("u"))
    lib = _lib(dev, fmt, macros)
    pso, ppso = nt.Pipeline(lib, "gemm_tile"), nt.Pipeline(_lib(dev, fmt, dict(macros, **kernels.x_permute_macros(False))), "x_permute")
    padded = max(tm, -(-t_act // tm) * tm)
    xp = nt.Buffer(dev, padded * k * 2)
    y = nt.Buffer(dev, padded * n * (2 if out_bf16 else 4)); y.fill(0)
    n_sg, n_tg, tg = kernels.gemm_geometry(f"ksplit{ksplit}" if ksplit > 1 else "crew", kernels.gemm_tiles(n, tn), dev.info().gpu_cores,
                                           min(384, pso.max_threads_per_threadgroup))
    if specialize:
        source, constants = kernels.specialize_params(kernels.gemm_source(fmt), "gemm",
            kernels.gemm_params(n, kernels.gemm_tiles(n, tn), n_sg, tm))
        pso = nt.Pipeline(nt.Library(dev, source, dict(macros, **constants), kernels.MSL_TENSOR_OPS), "gemm_tile")
    d0 = (nt.Dispatch().pipeline(ppso).buffer(0, nt.Buffer(dev, xb.tobytes())).buffer(3, xp)
          .bytes(4, kernels.x_permute_params(k, t_act, padded, int(f.weights_per_word), tk)).grid(padded * kernels.GEMM_PERM_SG).threadgroup(32).barrier())
    d1 = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, data)).buffer(1, nt.Buffer(dev, row_scales.tobytes())).buffer(2, xp).buffer(3, y)
          .bytes(4, kernels.gemm_params(n, kernels.gemm_tiles(n, tn), n_sg, t_act)).grid(n_tg, padded // tm, 1).threadgroup(tg))
    r = nt.Queue(dev).run([d0, d1])
    assert not r.error, r.error
    if out_bf16:
        out = bf16_to_f32(np.frombuffer(y.read(0, padded * n * 2), dtype=np.uint16).reshape(padded, n))
    else:
        out = np.frombuffer(y.read(0, padded * n * 4), dtype=np.float32).reshape(padded, n)
    xperm = bf16_to_f32(np.frombuffer(xp.read(0, padded * k * 2), dtype=np.uint16).reshape(padded, k))
    assert np.array_equal(xperm[:t_act], bf16_to_f32(xb)[:, kernels.x_permute_columns(k, int(f.weights_per_word), tk)]) and np.all(xperm[t_act:] == 0)
    rs = row_scales.astype(np.float64)[:, None]                            # the per-tensor scale, applied in FP32 at the epilogue
    w = bf16_to_f32(f32_to_bf16((f.dequantize(spec) / rs).astype(np.float32))) * rs   # the BF16 operand the accelerator multiplies
    ref = (bf16_to_f32(xb).astype(np.float64) @ w.T).astype(np.float32)
    return out, ref


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("lane_order", ["contiguous", "interleaved16"])
@pytest.mark.parametrize("tm,t_act", [(8, 8), (8, 3), (16, 16), (32, 21)])
def test_gemm_tile_matches_oracle(dev, fmt, lane_order, tm, t_act):
    out, ref = _run(dev, fmt, 200, K, tm, t_act, lane_order)           # 200 rows: a partial last tile
    chk = check_against_oracle(out[:t_act], ref)
    assert chk.ok(), chk
    assert np.all(out[t_act:] == 0)


def test_gemm_tile_bf16_output_and_k2048(dev):
    out, ref = _run(dev, "nvfp4", 128, 2048, 16, 16, "interleaved16", out_bf16=True)
    chk = check_against_oracle(out, ref)
    assert chk.ok_rounded(), chk                                         # one BF16 rounding of the FP32 result
    out, ref = _run(dev, "int4_affine", 128, 2048, 8, 8, "interleaved16")
    assert check_against_oracle(out[:8], ref).ok()


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("tn,tk", [(64, 64), (32, 128)])
def test_gemm_tile_other_tile_shapes(dev, fmt, tn, tk):
    """The non-default tiles (the default cases above run 16 × 256 up to 16 tokens and 32 × 128 at 32)."""
    out, ref = _run(dev, fmt, 200, 2048, 16, 11, "interleaved16", tn=tn, tk=tk)
    chk = check_against_oracle(out[:11], ref)
    assert chk.ok(), chk


# ---- the fusions the tile takes over from gemv_T (#51) --------------------------------------------------------------

EPS = 1e-6


def _rbf(a):
    return bf16_to_f32(f32_to_bf16(np.asarray(a, np.float32)))


class Gemm:
    """One packed matrix and the dispatch plumbing for the tile's variants (mirrors the fused-GEMV harness)."""

    def __init__(self, dev, fmt, n, k, tm, rows=16, seed=3, lane_order="interleaved16", function="gemm_tile", placement="inline", scale_order="lane"):
        self.dev, self.n, self.k, self.tm, self.fmt = dev, n, k, tm, fmt
        self.function = function
        rng = np.random.default_rng(seed)
        self.spec = random_spec(fmt, n, k, rng)
        self.data, self.info, self.row_scales = pack_spec(self.spec, PackLayout(rows=rows, lane_order=lane_order,
            scale_placement=placement, scale_order=scale_order))
        f = FORMATS.get(fmt)
        rs = self.row_scales.astype(np.float64)[:, None]
        self.w = _rbf((f.dequantize(self.spec) / rs).astype(np.float32)).astype(np.float64) * rs      # the BF16 operand × the tensor scale
        self.wpw = int(f.weights_per_word)
        self.wbuf, self.rsbuf = nt.Buffer(dev, self.data), nt.Buffer(dev, self.row_scales.tobytes())

    def run(self, x_bf16, *, t_active=None, norm=None, epilogue=None, residual=None, stat_out=False, out_bf16=None,
            round_residual=False, row_range=None, step_state=None, t_range=None, extra_macros=None, ksplit=1, conv=None,
            fixed_active=False, perm_groups=1, perm_simdgroups=kernels.GEMM_PERM_SG, perm_unroll=4):
        """``norm`` = (stat [T, parts] float32, parts, norm_w [K]); ``row_range`` = (start, count) in slab rows;
        ``step_state`` = (layout, values) with ``t_range`` = (lo, hi) for a predicated variant."""
        t_act = self.tm if t_active is None else t_active
        padded = max(self.tm, -(-t_act // self.tm) * self.tm)
        if out_bf16 is None:
            out_bf16 = epilogue is not None
        macros = kernels.gemm_macros(self.info, tm=self.tm, out_bf16=out_bf16, epilogue=epilogue, stat_out=stat_out, round_before_residual=round_residual, ksplit=ksplit)
        tk = int(macros["TK"].rstrip("u"))
        pmacros = dict(kernels.x_permute_macros(norm is not None, groups=perm_groups,
                                               simdgroups=perm_simdgroups, unroll=perm_unroll))
        src = kernels.gemm_source(self.fmt)
        if step_state is not None:
            layout, values = step_state
            src = src.replace(kernels.PRELUDE, kernels.PRELUDE + layout.to_msl() + "\n", 1)
            macros["STEP_STATE"] = pmacros["STEP_STATE"] = "1"
            if t_range is not None:
                macros["T_LO"] = pmacros["T_LO"] = str(t_range[0])
                macros["T_HI"] = pmacros["T_HI"] = str(t_range[1])
            st = nt.Buffer(self.dev, layout.pack(values))
        macros.update(extra_macros or {}); pmacros.update(extra_macros or {})
        lib = nt.Library(self.dev, src, macros, language_version=kernels.MSL_TENSOR_OPS)
        plib = nt.Library(self.dev, src, {**macros, **pmacros}, language_version=kernels.MSL_TENSOR_OPS)   # one source, both kernels
        pso, ppso = nt.Pipeline(lib, self.function), nt.Pipeline(plib, "x_permute")
        n = self.n
        if row_range is None:
            tile0, n_tiles, n_rows = 0, kernels.gemm_tiles(n, int(macros["TN"].rstrip("u"))), n
        else:
            tn = int(macros["TN"].rstrip("u"))
            tile0, n_tiles, n_rows = row_range[0] // tn, -(-row_range[1] // tn), row_range[1]
        n_sg, n_tg, tg = kernels.gemm_geometry(f"ksplit{ksplit}" if ksplit > 1 else "crew", n_tiles, self.dev.info().gpu_cores,
                                               min(384, pso.max_threads_per_threadgroup))
        n_blocks = -(-n_rows // self.info.rows)
        n_parts = n_blocks
        if self.function in ("gemv_bf16_small", "gemv_nvfp4_rows", "gemv_bf16_rows"):
            rm = int(str(macros.get("BF_ROWS", "2")).rstrip("u")) if self.function == "gemv_bf16_rows" else 2
            n_sg, n_tg, tg = n_blocks * self.info.rows // rm, n_blocks, 32 * self.info.rows // rm
            if self.function == "gemv_nvfp4_rows":
                rm = int(str(macros.get("NV_ROWS", "2")).rstrip("u"))
                groups = int(str(macros.get("NV_SG", self.info.rows // rm)).rstrip("u"))
                n_sg, n_tg, tg = n_blocks * self.info.rows // rm, n_blocks * self.info.rows // (rm * groups), 32 * groups
                n_parts = n_tg
        if fixed_active:
            assert step_state is None
            specialized, constants = kernels.specialize_params(src, "gemm",
                kernels.gemm_params(n_rows, n_tiles, n_sg, t_act, tile0=tile0, n_blocks=n_blocks), fixed_active=True)
            pso = nt.Pipeline(nt.Library(self.dev, specialized, dict(macros, **constants),
                language_version=kernels.MSL_TENSOR_OPS), self.function)
        n_out = n_rows // 2 if epilogue == "silu_mul" else n_rows
        xp = nt.Buffer(self.dev, padded * self.k * 2)
        y = nt.Buffer(self.dev, padded * n_out * 4); y.fill(0)
        parts = norm[1] if norm else 1
        d0 = (nt.Dispatch().pipeline(ppso).buffer(0, nt.Buffer(self.dev, x_bf16.tobytes())).buffer(3, xp)
              .bytes(4, kernels.x_permute_params(self.k, t_act, padded, self.wpw, tk, parts, EPS))
              .grid(padded * perm_simdgroups // perm_groups).threadgroup(32 * perm_groups).barrier())
        if norm:
            d0.buffer(1, nt.Buffer(self.dev, np.asarray(norm[0], np.float32).tobytes())).buffer(2, nt.Buffer(self.dev, np.asarray(norm[2], np.float32).tobytes()))
        d1 = (nt.Dispatch().pipeline(pso).buffer(0, self.wbuf).buffer(1, self.rsbuf).buffer(2, xp).buffer(3, y)
              .bytes(4, kernels.gemm_params(n_rows, n_tiles, n_sg, t_act, tile0=tile0, n_blocks=n_blocks)).grid(n_tg, padded // self.tm, 1).threadgroup(tg))
        direct_norm = (extra_macros or {}).get("DIRECT_NORM") == "1"
        if direct_norm:
            d1.buffer(2, nt.Buffer(self.dev, x_bf16.tobytes()))
            d1.buffer(5, nt.Buffer(self.dev, np.asarray(norm[0], np.float32).tobytes()))
            d1.buffer(6, nt.Buffer(self.dev, np.asarray(norm[2], np.float32).tobytes()))
        if conv is not None:
            d1.buffer(10, conv[0]).buffer(11, nt.Buffer(self.dev, conv[1].tobytes()))
        if epilogue == "residual":
            d1.buffer(7, nt.Buffer(self.dev, residual.tobytes()))
        so = None
        if stat_out:
            so = nt.Buffer(self.dev, padded * n_parts * 4); so.fill(0)
            d1.buffer(8, so)
        if step_state is not None:
            d0.buffer(15, st); d1.buffer(15, st)
        r = nt.Queue(self.dev).run([d1] if direct_norm else [d0, d1])
        assert not r.error, r.error
        raw = y.read(0, padded * n_out * (2 if out_bf16 else 4))
        out = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(padded, n_out) if out_bf16 else np.frombuffer(raw, dtype=np.float32).reshape(padded, n_out)
        so_arr = np.frombuffer(so.read(0, padded * n_parts * 4), dtype=np.float32).reshape(padded, n_parts) if stat_out else None
        return out, so_arr


def _norm_inputs(rng, t, k, scale=1.0):
    h = f32_to_bf16(rng.standard_normal((t, k)).astype(np.float32) * scale)
    hf = bf16_to_f32(h)
    nw = (1.0 + rng.standard_normal(k).astype(np.float32) * 0.1).astype(np.float32)
    stat = (hf.astype(np.float64) ** 2).sum(-1).astype(np.float32)
    r = 1.0 / np.sqrt(stat.astype(np.float64) / k + EPS)
    return h, nw, stat, _rbf(hf * r[:, None] * nw[None, :])


@pytest.mark.parametrize("fmt,k", [("nvfp4", 1024), ("fp8_e4m3", 1024), ("bf16", 1024), ("int4_affine", 1024), ("bf16", 256), ("nvfp4", 2048)])
def test_gemm_norm_on_the_way_in(dev, fmt, k):
    """x_permute with PERM_NORM: the tile multiplies the reference norm's BF16 output; K = 256 is a slice shorter
    than the permute's unroll (a synthetic drafter's hidden size)."""
    t = 8
    g = Gemm(dev, fmt, 96, k, t)
    h, nw, stat, x_ref = _norm_inputs(np.random.default_rng(11), t, k)
    y, _ = g.run(h, norm=(stat, 1, nw), out_bf16=False)
    ref = (x_ref.astype(np.float64) @ g.w.T).astype(np.float32)
    chk = check_against_oracle(y, ref)
    assert chk.ok(), chk


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16"])
@pytest.mark.parametrize("rounded", [False, True])
def test_gemm_residual_epilogue(dev, fmt, rounded):
    t = 8
    g = Gemm(dev, fmt, 96, 1024, t)
    rng = np.random.default_rng(7)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(t, 1024)).astype(np.float32))
    res = f32_to_bf16(rng.standard_normal((t, 96)).astype(np.float32))
    y, _ = g.run(x, epilogue="residual", residual=res, round_residual=rounded)
    prod = bf16_to_f32(x).astype(np.float64) @ g.w.T
    ref = _rbf((_rbf(prod) if rounded else prod) + bf16_to_f32(res))
    chk = check_against_oracle(y, ref)
    assert chk.ok_rounded() and chk.max_ulp_at_scale <= 1.0, chk     # one BF16 rounding of the sum (the product ~1e-6 off)


@pytest.mark.parametrize("fmt,rows", [("nvfp4", 16), ("fp8_e4m3", 16), ("bf16", 8), ("int4_affine", 16)])
@pytest.mark.parametrize("t_act", [8, 5])
def test_gemm_silu_mul_epilogue(dev, fmt, rows, t_act):
    """Chunk-interleaved gate|up rows (chunk = R/2): the partner row lives in the lane differing in bit 3 (R = 16) or
    bit 0 (R = 8); N/2 outputs, rows beyond t_active untouched."""
    n = 192
    g = Gemm(dev, fmt, n, 1024, 8, rows=rows)
    rng = np.random.default_rng(9)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(8, 1024)).astype(np.float32))
    y, _ = g.run(x, epilogue="silu_mul", t_active=t_act)
    full = bf16_to_f32(x).astype(np.float64) @ g.w.T                      # [T, N] in slab-row order
    chunk = rows // 2
    blocks = full.reshape(8, n // rows, rows)
    gate, up = blocks[:, :, :chunk], blocks[:, :, chunk:]
    ref = _rbf((gate / (1 + np.exp(-gate)) * up).reshape(8, n // 2))
    chk = check_against_oracle(y[:t_act], ref[:t_act])
    assert chk.ok_rounded(), chk
    assert np.all(y[t_act:] == 0)


@pytest.mark.parametrize("fmt,rows", [("fp8_e4m3", 16), ("bf16", 8)])
def test_gemm_stat_out_and_row_range(dev, fmt, rows):
    """STAT_OUT: per pack block, the sum of squares of the BF16-rounded outputs per token; a row range of the slab
    (whole tiles) gives range-relative outputs and partials."""
    t = 8
    n = 320
    g = Gemm(dev, fmt, n, 1024, t, rows=rows)
    rng = np.random.default_rng(5)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(t, 1024)).astype(np.float32))
    res = f32_to_bf16(rng.standard_normal((t, n)).astype(np.float32))
    y, so = g.run(x, epilogue="residual", residual=res, stat_out=True, t_active=6)
    ref = _rbf(bf16_to_f32(x).astype(np.float64) @ g.w.T + bf16_to_f32(res))
    assert check_against_oracle(y[:6], ref[:6]).ok_rounded()
    assert so.shape == (t, n // rows) and np.all(so[6:] == 0)
    blocks = (y[:6].astype(np.float64) ** 2).reshape(6, n // rows, rows).sum(-1)
    assert np.allclose(so[:6], blocks, rtol=1e-5)
    # the last 128 rows as a range (tiles of 16): the same values, relative indices
    start = 192
    y2, so2 = g.run(x, epilogue="residual", residual=np.ascontiguousarray(res[:, start:]), stat_out=True, row_range=(start, n - start), t_active=6)
    assert np.array_equal(y2, np.concatenate([y[:6, start:], np.zeros((2, n - start), np.float32)]))
    assert np.allclose(so2[:6], (y2[:6].astype(np.float64) ** 2).reshape(6, (n - start) // rows, rows).sum(-1), rtol=1e-5) and np.all(so2[6:] == 0)


def test_gemm_step_state_predicate(dev):
    """A per-T variant: the tile and its permute read T from StepState and run only for T_LO < T <= T_HI."""
    from monolith.core import StepStateLayout

    layout = StepStateLayout(t_max=8, gamma_max=7)
    g = Gemm(dev, "nvfp4", 96, 1024, 8)
    rng = np.random.default_rng(3)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(8, 1024)).astype(np.float32))
    ref = (bf16_to_f32(x).astype(np.float64) @ g.w.T).astype(np.float32)
    y, _ = g.run(x, out_bf16=False, step_state=(layout, {"t_this_step": 6}), t_range=(1, 8))
    assert check_against_oracle(y[:6], ref[:6]).ok() and np.all(y[6:] == 0)
    y, _ = g.run(x, out_bf16=False, step_state=(layout, {"t_this_step": 1}), t_range=(1, 8))
    assert np.all(y == 0)                                                    # T = 1 belongs to the shader variant
    y, _ = g.run(x, out_bf16=False, step_state=(layout, {"t_this_step": 6, "done": 1}), t_range=(1, 8))
    assert np.all(y == 0)
    y, _ = g.run(x, out_bf16=False, step_state=(layout, {"n_inject": 3, "t_this_step": 8}), t_range=(1, 8))
    assert not np.all(y == 0)                                                # the t_this_step source by default


def test_gemm_static_rows_source(dev):
    """A static row count in a dynamic-T program (the drafter's block pass): T_SRC = 2 takes T_STATIC_ROWS, no predicate."""
    from monolith.core import StepStateLayout

    layout = StepStateLayout(t_max=8, gamma_max=7)
    g = Gemm(dev, "bf16", 96, 1024, 8)
    rng = np.random.default_rng(4)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(8, 1024)).astype(np.float32))
    ref = (bf16_to_f32(x).astype(np.float64) @ g.w.T).astype(np.float32)
    macros_extra = {"T_SRC": "2", "T_STATIC_ROWS": "3u"}
    y, _ = g.run(x, out_bf16=False, step_state=(layout, {"t_this_step": 1, "n_inject": 0}), extra_macros=macros_extra)
    assert check_against_oracle(y[:3], ref[:3]).ok() and np.all(y[3:] == 0)


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "int4_affine", "bf16"])
@pytest.mark.parametrize("ksplit,tm,t_act", [(2, 8, 8), (4, 8, 5), (8, 8, 8), (2, 16, 16), (2, 32, 30)])
def test_gemm_tile_ksplit_matches_oracle(dev, fmt, ksplit, tm, t_act):
    """The K-split (one tile per threadgroup of ``ksplit`` SIMD-groups, the partials reduced through threadgroup
    memory): 272 rows (a partial last tile), K = 2048 (8 K tiles at TK = 256 — two lane groups per slice at 2, one
    at 4, one K tile per slice at 8), every format, the three TM."""
    if ksplit == 8 and fmt != "fp8_e4m3":
        pytest.skip("8 slices of 8 K tiles: one K tile each, half a lane group — the scale cache needs whole lane groups")
    out, ref = _run(dev, fmt, 272, 2048, tm, t_act, "interleaved16", ksplit=ksplit)
    chk = check_against_oracle(out[:t_act], ref)
    assert chk.ok() and chk.max_rel_err < 2e-6, chk
    assert np.all(out[t_act:] == 0)


def test_gemm_tile_ksplit_rejects_odd_splits():
    rng = np.random.default_rng(0)
    _, info, _ = pack_spec(random_spec("nvfp4", 256, 1024, rng), PackLayout(rows=16))     # 4 K tiles of 256, one word per lane
    kernels.gemm_macros(info, tm=8, ksplit=4)                                                # one K tile (one lane group) per slice
    with pytest.raises(ValueError):
        kernels.gemm_macros(info, tm=8, ksplit=3)
    _, info, _ = pack_spec(random_spec("nvfp4", 256, 512, rng), PackLayout(rows=16))      # 2 K tiles: 4 does not divide them
    with pytest.raises(ValueError):
        kernels.gemm_macros(info, tm=8, ksplit=4)
    _, info, _ = pack_spec(random_spec("nvfp4", 256, 4096, rng), PackLayout(rows=16))     # 16 K tiles, 4 words per lane: whole lane groups per slice
    assert kernels.gemm_macros(info, tm=8, ksplit=4)["KSPLIT"] == "4u"
    with pytest.raises(ValueError, match="threadgroup memory"):                            # TM 32 × 16 slices: 60 KiB of partial tiles
        kernels.gemm_macros(info, tm=32, ksplit=16, scale_cache=False)


@pytest.mark.parametrize("fmt,ksplit", [("nvfp4", 2), ("fp8_e4m3", 4), ("int4_affine", 2)])
def test_gemm_ksplit_epilogues_and_predication(dev, fmt, ksplit):
    """The K-split with the fusions the step uses on it: the residual epilogue with the statistic output, a row range
    starting mid-slab, and the per-T predicate of a dynamic-T program (a variant outside its range writes nothing —
    the barriers are behind the uniform early return)."""
    from monolith.core import StepStateLayout

    rng = np.random.default_rng(11)
    t = Gemm(dev, fmt, 288, 2048, 8)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(8, 2048)).astype(np.float32))
    res = f32_to_bf16(rng.standard_normal((8, 288)).astype(np.float32))
    out, so = t.run(x, epilogue="residual", residual=res, stat_out=True, ksplit=ksplit)
    prod = bf16_to_f32(x).astype(np.float64) @ t.w.T
    ref = _rbf(prod + bf16_to_f32(res))
    assert check_against_oracle(out, ref.astype(np.float32)).max_ulp_elementwise <= 1
    blocks = ref.reshape(8, -1, t.info.rows)
    assert np.allclose(so, (blocks.astype(np.float64) ** 2).sum(-1), rtol=1e-4, atol=1e-3)
    out2, _ = t.run(x, epilogue="residual", residual=np.ascontiguousarray(res[:, 32:288]), row_range=(32, 256), ksplit=ksplit)   # range-relative residual
    assert check_against_oracle(out2, ref[:, 32:288].astype(np.float32)).max_ulp_elementwise <= 1
    layout = StepStateLayout(t_max=8, gamma_max=7)
    out3, _ = t.run(x, epilogue="residual", residual=res, step_state=(layout, {"t_this_step": 6}), t_range=(1, 8), ksplit=ksplit)
    assert check_against_oracle(out3[:6], ref[:6].astype(np.float32)).max_ulp_elementwise <= 1 and np.all(out3[6:] == 0)
    out4, _ = t.run(x, epilogue="residual", residual=res, step_state=(layout, {"t_this_step": 1}), t_range=(1, 8), ksplit=ksplit)
    assert np.all(out4 == 0)


@pytest.mark.parametrize("fmt", ["bf16", "int4_affine", "nvfp4"])
@pytest.mark.parametrize("tokens", [2, 4, 6, 8])
@pytest.mark.parametrize("persistent", [False, True])
def test_ksplit_live_token_partials(dev, monkeypatch, fmt, tokens, persistent):
    """Compact scratch preserves residual/stat results and the reuse barrier.

    A single persistent threadgroup deliberately reuses scratch across row tiles;
    the production geometry normally launches one threadgroup per tile.
    """
    from monolith.core import StepStateLayout
    if persistent:
        geometry = kernels.gemm_geometry
        monkeypatch.setattr(kernels, "gemm_geometry", lambda mode, n, *a:
                            (4, 1, 128) if mode == "ksplit4" else geometry(mode, n, *a))
    rng = np.random.default_rng(29)
    g = Gemm(dev, fmt, 288, 2048, 8)
    x = f32_to_bf16(rng.normal(0, .2, (8, 2048)).astype(np.float32))
    residual = f32_to_bf16(rng.normal(0, .2, (8, 288)).astype(np.float32))
    layout = StepStateLayout(t_max=8, gamma_max=7)
    out, stat = g.run(x, epilogue="residual", residual=residual, stat_out=True, ksplit=4,
                      step_state=(layout, {"t_this_step": tokens}), t_range=(1, tokens),
                      extra_macros={"COMPACT_PARTIALS": "1"})
    ref = _rbf(bf16_to_f32(x[:tokens]).astype(np.float64) @ g.w.T + bf16_to_f32(residual[:tokens]))
    assert check_against_oracle(out[:tokens], ref).max_ulp_elementwise <= 1
    assert np.all(out[tokens:] == 0) and np.all(stat[tokens:] == 0)
    expected = (ref.reshape(tokens, -1, g.info.rows).astype(np.float64) ** 2).sum(-1)
    assert np.allclose(stat[:tokens], expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("fmt,k", [("nvfp4", 4096), ("nvfp4", 5120), ("int8", 4096), ("int4_affine", 4096), ("int4_affine", 1024), ("int4_affine", 3072), ("nvfp4", 2048)])
@pytest.mark.parametrize("ksplit", [1, 2])
def test_gemm_tile_block_scale_placement(dev, fmt, k, ksplit):
    """The tile's cooperative fill reading the block's scale region (#101), with and without the scale cache and the K-split."""
    out, ref = _run(dev, fmt, 272, k, 8, 8, "interleaved16", ksplit=ksplit, placement="block")
    chk = check_against_oracle(out, ref)
    assert chk.ok() and chk.max_rel_err < 2e-6, chk


@pytest.mark.parametrize("ksplit", [1, 2, 8])
@pytest.mark.parametrize("rows", [8, 16])
@pytest.mark.parametrize("n", [269, 272])
def test_nvfp4_24_byte_scale_region_matches_inline(dev, ksplit, rows, n):
    """Moving scales preserves every output bit, including a partial row block.
    The longer reduction also meets the independent kernel ULP contract.
    """
    outputs = []
    for placement in ("inline", "block"):
        out, ref = _run(dev, "nvfp4", n, 12288, 8, 8, "interleaved16",
                        rows=rows, ksplit=ksplit, placement=placement)
        chk = check_against_oracle(out, ref)
        assert chk.ok(), chk
        outputs.append(out)
    assert np.array_equal(*outputs)


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3"])
def test_gemm_silu_mul_perm_out(dev, fmt):
    """The silu·mul epilogue writing its output in the consumer tile's x' order (PERM_OUT): the same bits as x_permute
    applied to the natural output (the consumer: K = N/2 of this tile, NVFP4 words of 32, TK = 256)."""
    rng = np.random.default_rng(23)
    n, k = 2048, 1024                                                        # the consumer's K = N/2 = 1024: whole NVFP4 words per lane
    g = Gemm(dev, fmt, n, k, 8, rows=16)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(8, k)).astype(np.float32))
    natural, _ = g.run(x, epilogue="silu_mul")
    k_next, wpw_next, tk_next = n // 2, 32, 256
    permuted, _ = g.run(x, epilogue="silu_mul", extra_macros=kernels.perm_out_macros(k_next, wpw_next, tk_next))
    cols = kernels.x_permute_columns(k_next, wpw_next, tk_next)
    assert np.array_equal(permuted, natural[:, cols])


@pytest.mark.parametrize("tk", [64, 128])
@pytest.mark.parametrize("ksplit", [1, 4, 8])
def test_small_affine_tile(dev, tk, ksplit):
    out, ref = _run(dev, "int4_affine", 272, 1024, 8, 5, "interleaved16", out_bf16=True,
                    tn=16, tk=tk, ksplit=ksplit, placement="block")
    assert check_against_oracle(out[:5], bf16_to_f32(f32_to_bf16(ref))).ok()
    assert np.all(out[5:] == 0)


@pytest.mark.parametrize("fmt", ["int4_affine", "nvfp4", "bf16"])
@pytest.mark.parametrize("active,ksplit", [(3, 1), (6, 2)])
def test_static_geometry_preserves_matrix_tails_and_partial_tokens(dev, fmt, active, ksplit):
    outputs = [_run(dev, fmt, 100, 1024, 8, active, "interleaved16", ksplit=ksplit, specialize=flag)[0]
               for flag in (False, True)]
    np.testing.assert_array_equal(*outputs)
    assert not outputs[1][active:].any()


@pytest.mark.parametrize("fmt", ["bf16", "nvfp4", "int4_affine"])
@pytest.mark.parametrize("tokens", [33, 127, 128, 129])
def test_token_axis_tiling_for_large_prefill(dev, fmt, tokens):
    out, ref = _run(dev, fmt, 200, K, 32, tokens, "interleaved16", ksplit=2)
    assert check_against_oracle(out[:tokens], ref).ok()
    assert np.all(out[tokens:] == 0)


@pytest.mark.parametrize("fmt", ["bf16", "nvfp4", "int4_affine"])
def test_token_axis_tiling_preserves_epilogues(dev, fmt):
    from monolith.core import StepStateLayout
    rng = np.random.default_rng(12)
    count, active, n = 129, 127, 192
    g = Gemm(dev, fmt, n, 1024, 32)
    x = f32_to_bf16(rng.normal(0, .1, (count, 1024)).astype(np.float32))
    residual = f32_to_bf16(rng.normal(0, .1, (count, n)).astype(np.float32))
    state = (StepStateLayout(t_max=count), {"t_this_step": active})
    opts = dict(t_active=count, step_state=state, t_range=(1, count), ksplit=2)
    y, stat = g.run(x, epilogue="residual", residual=residual, stat_out=True, **opts)
    product = bf16_to_f32(x).astype(np.float64) @ g.w.T
    ref = _rbf(product + bf16_to_f32(residual))
    assert check_against_oracle(y[:active], ref[:active]).ok_rounded()
    expected = (y[:active].astype(np.float64) ** 2).reshape(active, -1, g.info.rows).sum(-1)
    assert np.allclose(stat[:active], expected, rtol=1e-5)
    assert np.all(y[active:] == 0) and np.all(stat[active:] == 0)
    y, _ = g.run(x, epilogue="silu_mul", **opts)
    blocks = product.reshape(count, -1, g.info.rows)
    half = g.info.rows // 2
    gate, up = blocks[:, :, :half], blocks[:, :, half:]
    ref = _rbf((gate / (1 + np.exp(-gate)) * up).reshape(count, n // 2))
    assert check_against_oracle(y[:active], ref[:active]).ok_rounded()
    assert np.all(y[active:] == 0)
