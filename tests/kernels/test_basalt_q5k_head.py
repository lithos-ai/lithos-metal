# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/q5k_head_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its weights and activations, row counts, maps, planted rows and pass criteria.
"""basalt's Q5_K head on Metal (kernels/common/basalt_q5k_head.metal).

q5k_head_parity, ported (basalt src/kernels/q5k_head_parity.cpp, ctest ``q5k_head_parity 20000 5``: 20,000 rows):
random Q5_K rows (random bytes, then d and dmin = fp16(|N(0, 1)| * 0.01), :33-43) and N(0, 1) activations quantized
by native_quantize_q8_1, n_in 2560. numpy's generator stands for basalt's mt19937 + normal_distribution: the same
distributions, other draws. The timing loops are not ported.
- The head against native_mmvq (basalt_mmvq.metal, Q5_K) bit for bit, 1 to 8 columns (:64-92); each column also bit
  for bit the head's one-column call on it (column invariance, flash-next.md invariant 1).
- The contract's test (docs/design/flash-next.md, kernels without a reference test): mmvq_multi_parity.cpp:461's rule
  at the head's shape, 2560 -> 248,320 rows, 8 columns, each bit for bit the head's one-column call on it and within
  1e-4 * max |ref| of the float64 product on the same bytes; the bitwise check against native_mmvq carried over."""

import struct

import numpy as np
import pytest

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_mmvq import reference

F32 = np.float32
N_IN, ROWS, MT = 2560, 20000, 8      # basalt's n_in, ctest's row count, its 8 columns
QCOL = N_IN // 32 * 36               # q8_1 bytes a column


def weights(rows, rng, n_in=N_IN):
    """q5k_head_parity.cpp:33-43: random bytes, then each block's d and dmin fp16(|N(0, 1)| * 0.01)."""
    w = rng.integers(0, 256, (rows * n_in // 256, 176), dtype=np.uint8)
    dm = (np.abs(rng.standard_normal((w.shape[0], 2))).astype(F32) * F32(0.01)).astype(np.float16)
    w[:, :4] = dm.view(np.uint8).reshape(-1, 4)
    return w.reshape(rows, -1)


def activations(ncols, rng):
    """N(0, 1) columns quantized by native_quantize_q8_1 (:47-59)."""
    return gpu.q8_1(rng.standard_normal(ncols * N_IN).astype(F32), "native")


def column(xq, j):
    return xq[j * QCOL: (j + 1) * QCOL]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)                 # basalt's seed, std::mt19937 rng(5) (:32)
    w = weights(ROWS, rng)
    return w, activations(MT, rng), rng


def bad_bits(a, b):
    return int(np.count_nonzero(np.asarray(a, F32).view(np.uint32) != np.asarray(b, F32).view(np.uint32)))


def test_head_against_native_mmvq(data):
    w, xq, _ = data
    fail = diff = 0
    one = np.stack([gpu.q5k_head(w, column(xq, j), N_IN, ROWS, 1)[0] for j in range(MT)])
    for nc in range(1, MT + 1):
        a = gpu.mmvq(13, w, xq[: nc * QCOL], N_IN, ROWS, nc, fill=0xFF)        # basalt's pre-fills (:65-66)
        b = gpu.q5k_head(w, xq[: nc * QCOL], N_IN, ROWS, nc, fill=0xEE)
        bad = bad_bits(a, b)
        # basalt: ``for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;`` (:74) and
        # ``if (bad) fail = 1;`` (:91)
        fail |= bad != 0
        diff += bad_bits(b, one[:nc])
        assert np.all(np.isfinite(b)), nc
    assert not fail and diff == 0, diff


def reference_rows(w, xc, rows, chunk=5000):
    """reference(13, ...) by chunks of rows: its float64 intermediates take ~20 KB a row."""
    return np.concatenate([reference(13, w[r: r + chunk], xc, N_IN, min(chunk, rows - r))
                           for r in range(0, rows, chunk)])


def test_head_full_shape():
    """flash-next.md's test for q5k_head_mmvq: mmvq_multi_parity.cpp:461's rule at the head's shape, 2560 -> 248,320
    rows (basalt's test default, q5k_head_parity.cpp:28), 8 columns. Per column: ``diff``, the values differing from
    the head's one-column call on it (basalt's one-column call against its two-column one, :429-430, :457), and
    ``worst``, the one-column call's largest error against the float64 product on the same bytes, against ``scale``,
    the largest |ref|. The bitwise check against native_mmvq carries over, one column and eight."""
    rows = 248320
    rng = np.random.default_rng(28)
    w = weights(rows, rng)
    xq = activations(MT, rng)
    y = gpu.q5k_head(w, xq, N_IN, rows, MT, fill=0xEE)
    assert bad_bits(gpu.mmvq(13, w, xq, N_IN, rows, MT, fill=0xFF), y) == 0
    for j in range(MT):
        a = gpu.q5k_head(w, column(xq, j), N_IN, rows, 1)[0]
        if j == 0:
            assert bad_bits(gpu.mmvq(13, w, column(xq, 0), N_IN, rows, 1, fill=0xFF), a) == 0
        ref = reference_rows(w, column(xq, j), rows)
        diff = bad_bits(a, y[j])
        worst, scale = np.max(np.abs(a.astype(np.float64) - ref)), np.max(np.abs(ref))
        # basalt: ``const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);`` (mmvq_multi_parity.cpp:461); its
        # std::max fold drops a NaN, so the non-finite values are counted
        assert np.all(np.isfinite(a)) and diff == 0 and worst <= 1e-4 * (scale + 1e-30), (j, diff, worst, scale)


@pytest.mark.parametrize("n_in", [256, 2560, 16384])
def test_head_shapes_and_grids(n_in):
    """Other row lengths (q5k_head_supported's range: 1 to 64 blocks) and grids, each column against native_mmvq."""
    rng = np.random.default_rng(n_in)
    rows = 333
    w = weights(rows, rng, n_in)
    ncols = 1 if n_in == 16384 else 4                 # basalt's 96 KiB staging rule allows 2 columns at 16384
    xq = gpu.q8_1(rng.standard_normal(ncols * n_in).astype(F32), "native")
    a = gpu.mmvq(13, w, xq, n_in, rows, ncols)
    for sm_count, per_sm in ((170, 2), (40, 1), (1, 1)):
        b = gpu.q5k_head(w, xq, n_in, rows, ncols, sm_count=sm_count, per_sm=per_sm)
        assert bad_bits(a, b) == 0, (sm_count, per_sm)


def test_launch_rules():
    """The builders refuse what basalt's API refuses or its int arithmetic cannot hold: n_out and n_rows are its ints
    (q5k_head.cu:372-374, 412, 516-518), n_rows 0 included; n_cap past the 32-bit field; a grid whose int row loop
    would step past INT_MAX."""
    assert kernels.basalt_q5k_head_grid(248320) == 340 and kernels.basalt_q5k_head_grid(2 ** 31 - 8 * 340) == 340
    for bad in [lambda: kernels.basalt_q5k_head_params(2560, 0), lambda: kernels.basalt_q5k_head_params(2560, 2 ** 31),
                lambda: kernels.basalt_q5k_head_grid(2 ** 31 - 8 * 340 + 1),
                lambda: kernels.basalt_q5k_proxy_params(100, n_rows=0),
                lambda: kernels.basalt_q5k_proxy_params(100, n_rows=101),
                lambda: kernels.basalt_q5k_proxy_params(2 ** 32 - 1, n_rows=2 ** 31),
                lambda: kernels.basalt_q5k_proxy_params(2 ** 32, row0=0, n=1),
                lambda: kernels.basalt_q5k_proxy_params(100, row0=50, n=51)]:
        with pytest.raises(ValueError):
            bad()
    assert kernels.basalt_q5k_proxy_params(2 ** 32 - 1, row0=2 ** 32 - 2, n=1)[8:] == struct.pack("<II", 2 ** 32 - 1, 0)


def test_head_emulation_bit_for_bit(data):
    """The head against the numpy emulation of nvcc's sequence (basalt_emu.mmvq, Q5_K: the head's per-value
    sequence is native_mmvq's, q5k_head.md section 1)."""
    w, xq, _ = data
    rows = 96
    got = gpu.q5k_head(w[:rows], xq[: 3 * QCOL], N_IN, rows, 3)
    assert bad_bits(got, emu.mmvq(13, w[:rows], xq[: 3 * QCOL], N_IN, rows, 3)) == 0


# ---- the fused top-1 (q5k_head_parity.cpp:93-211) ------------------------------------------------------------------

def host_top(a, mapped):
    """The host argmax over the GPU logits (the lowest row on ties: basalt's ``a[i] > a[best]`` scan and the sampler's
    greedy rule, flash-next.md "Greedy pick") and the double softmax sum relative to it (:171-179)."""
    rows = np.flatnonzero(mapped)
    best = int(rows[np.argmax(a[rows])])                # np.argmax: the first maximum
    s = float(np.sum(np.exp(a[rows].astype(np.float64) - np.float64(a[best]))))
    return best, s


def test_top1(data):
    """:93-146 at ctest's row counts ({20000, 106299, 40525, 1000}, the larger skipped above 20,000, :105-107), each of
    8 columns: the logits bit for bit against native_mmvq, the id against the host argmax over the GPU logits (basalt:
    sample_greedy_top_prob on the GPU, sampler.cu:930, not on main), the probability against a host double softmax
    over them (the comparison basalt's holes section makes, :171-185)."""
    w, xq, _ = data
    logit_bad = id_bad = nonfinite = 0
    max_rel = 0.0
    for rr in (ROWS, 106299, 40525, 1000):
        if rr > ROWS:
            continue
        for col in range(MT):
            xc = column(xq, col)
            a = gpu.mmvq(13, w[:rr], xc, N_IN, rr, 1)[0]
            b, b_id, b_p, _ = gpu.q5k_top1(w[:rr], xc, rr)
            best, s = host_top(a, np.ones(rr, bool))
            logit_bad += bad_bits(a, b)
            id_bad += b_id != best
            nonfinite += int(not np.isfinite(b_p)) + int(np.count_nonzero(~np.isfinite(b)))
            max_rel = max(max_rel, abs(float(b_p) - 1.0 / s) * s)
    # basalt: ``if (logit_bad || id_bad || max_rel > 1e-5) fail = 1;`` (:145); its std::max fold (:123) drops a NaN,
    # so the non-finite outputs are counted (flash-next.md)
    assert logit_bad == 0 and id_bad == 0 and nonfinite == 0 and max_rel <= 1e-5, (logit_bad, id_bad, nonfinite,
                                                                                      max_rel)


def holes_map(rr):
    """:157-158: every fifth row (i % 5 == 2) and the last 2048 rows without a token, the others id 3i + 7."""
    i = np.arange(rr)
    return np.where((i % 5 == 2) | (i >= rr - 2048), -1, 3 * i + 7).astype(np.int32)


def test_top1_holes(data):
    """:147-211 at min(rows, 106299) = 20,000 rows, 8 columns: masked rows -inf, the others bit for bit native_mmvq's;
    the id against map[host argmax], the probability against the host double softmax; then the mask kernel after a
    plain head the same way (:186-193)."""
    w, xq, _ = data
    rr = min(ROWS, 106299)
    mp = holes_map(rr)
    mapped = mp >= 0
    bad = 0
    max_rel = 0.0
    for col in range(MT):
        xc = column(xq, col)
        a = gpu.mmvq(13, w[:rr], xc, N_IN, rr, 1)[0]
        b, b_id, b_p, _ = gpu.q5k_top1(w[:rr], xc, rr, mp)
        bad += int(np.count_nonzero(~np.isneginf(b[~mapped]))) + bad_bits(a[mapped], b[mapped])
        best, s = host_top(a, mapped)
        bad += b_id != mp[best]
        max_rel = max(max_rel, abs(float(b_p) - 1.0 / s) * s)
        c = gpu.q5k_head_masked(w[:rr], xc, N_IN, rr, mp)
        bad += int(np.count_nonzero(~np.isneginf(c[~mapped]))) + bad_bits(a[mapped], c[mapped])
        # basalt's std::max fold (:185) drops a NaN: the non-finite outputs are counted (flash-next.md)
        bad += int(not np.isfinite(b_p)) + int(np.count_nonzero(~np.isfinite(b[mapped])))
        bad += int(np.count_nonzero(~np.isfinite(c[mapped])))
    # basalt: ``if (bad || max_rel > 1e-5) fail = 1;`` (:210)
    assert bad == 0 and max_rel <= 1e-5, (bad, max_rel)


@pytest.mark.parametrize("rr,grid", [(20000, {}), (1000, {}), (20000, {"sm_count": 40, "per_sm": 1}),
                                     (5000, {"sm_count": 3, "per_sm": 1})])
def test_top1_emulation_bit_for_bit(data, rr, grid):
    """The partials, the id and the probability bit for bit against the numpy emulation of the take and merge
    (basalt_emu.q5k_top1) on the kernel's logits, with the GPU's expf_ftz; with and without the holes map; at basalt's
    grid and others (the probability's last bits follow the grid, mtp.cpp:1007-1008)."""
    w, xq, _ = data
    blocks = kernels.basalt_q5k_head_grid(rr, **grid)
    for mp in (None, holes_map(rr)):
        y, b_id, b_p, part = gpu.q5k_top1(w[:rr], column(xq, 3), rr, mp, **grid)
        mapped = np.ones(rr, bool) if mp is None else mp >= 0
        e_part, e_row, e_p = emu.q5k_top1(y, mapped, blocks, gpu.expf_ftz)
        assert bad_bits(part[:, 0], e_part[0]) == 0 and bad_bits(part[:, 2], e_part[2]) == 0
        assert np.array_equal(part[:, 1].view(np.int32), e_part[1].astype(np.int32))
        e_row = e_row if e_row < rr else 0             # no row taken (every row masked): row 0 (q5k_head.cu:330)
        assert b_id == (e_row if mp is None else mp[e_row]) and bad_bits(b_p, e_p) == 0, (b_id, e_row, b_p, e_p)


def test_expf_ftz():
    """basalt_fp.metal's expf_ftz, whose GPU values the emulations above take for __expf, bit for bit against its
    definition (q5k_head.md section 2): FMUL.FTZ by fl(log2 e) and the .ftz flush of the result on the host, only
    fast::exp2 (MUFU.EX2's counterpart, with no bit-exact model) on the GPU. The arguments: the take's and the score's
    range, the result's flush (x * log2 e below -126: x below -87.34), its underflow to 0 (below -103.3), its overflow
    (above 88.72), -inf (the take's first row: m = -inf), zeros and subnormals (flushed)."""
    log2e = F32(np.log2(np.e))
    assert log2e.view(np.uint32) == 0x3FB8AA3B                       # ptxas's FMUL.FTZ immediate 1.4426950216293334961
    x = np.concatenate([np.linspace(-30, 0, 4001, dtype=F32), np.linspace(-88, -86.5, 4001, dtype=F32),
                        np.linspace(-104.5, -102.5, 2001, dtype=F32), np.linspace(88, 89.5, 201, dtype=F32),
                        F32([-np.inf, 0.0, -0.0]), np.uint32([1, 0x007FFFFF, 0x80000001]).view(F32)]).astype(F32)
    p = emu.mul_ftz(x, log2e)
    src = kernels.PRELUDE + """
kernel void ex2(device const float* p [[buffer(0)]], device float* e [[buffer(1)]], constant uint& n [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    if (i < n) e[i] = fast::exp2(p[i]);
}
"""
    e = gpu.run(gpu.pipeline(src, "ex2"), [p.tobytes(), 4 * p.size], np.uint32(p.size).tobytes(), 2, -(-p.size // 256),
                256)[1]
    assert bad_bits(gpu.expf_ftz(x), emu.ftz(np.frombuffer(e, F32))) == 0


# ---- the 2-bit proxy (q5k_head_parity.cpp:212-350) -----------------------------------------------------------------

@pytest.fixture(scope="module")
def proxy_rows(data):
    """:215-235: min(rows, 41037) = 20,000 rows whose blocks' d and dmin are scaled by f = 3 / sd(column 0's logits),
    so the logits spread 3: fp16(f32(d) * f)."""
    w, xq, _ = data
    rr = min(ROWS, 41037)
    l0 = gpu.mmvq(13, w[:rr], column(xq, 0), N_IN, rr, 1)[0].astype(np.float64)
    f = F32(3.0 / np.sqrt(np.mean(l0 * l0) - np.mean(l0) ** 2))
    blk = w[:rr].copy().reshape(rr * 10, 176)
    dm = blk[:, :4].copy().view(np.float16).astype(F32) * f
    blk[:, :4] = dm.astype(np.float16).view(np.uint8).reshape(-1, 4)
    return blk.reshape(rr, 1760)


def test_proxy_pack_emulation_bit_for_bit(proxy_rows):
    """q5k_proxy_pack byte for byte against the numpy emulation of nvcc's fit and packing (basalt_emu.q5k_proxy_pack;
    q5k_head.md section 4), on basalt's rescaled rows with two planted blocks: one with zero scales (w = 0, so sh = 0
    and every level comes from a NaN, which F2I takes to 0) and one with d = 65504 (s past fp16's range, sh = inf);
    into a copy with spare rows, n_cap = rows + 100 as basalt's test (:216)."""
    rows = 640
    q5 = proxy_rows[:rows].copy()
    blk = q5.reshape(rows * 10, 176)
    blk[3, 4:16] = 0                                         # row 0, block 3: groups 24..31 have sc = m = 0
    blk[25, 0:2] = np.array([65504], np.float16).view(np.uint8)   # row 2, block 5: d = 65504
    n_cap = rows + 100
    got = gpu.q5k_proxy_pack(q5, n_cap, [(0, rows)])
    assert np.array_equal(got, emu.q5k_proxy_pack(q5, 0, rows, n_cap))
    words = got[: emu.px_tiles(n_cap) * 40 * 32 * 16].view(np.uint32).reshape(-1, 40, 32, 4)
    sc = got[emu.px_tiles(n_cap) * 40 * 32 * 16:].view(np.uint16).reshape(-1, 40, 32, 2)
    assert np.all(words[0, 12:16, 0] == 0) and np.all(sc[0, 12:16, 0] == 0)        # zero groups: levels 0, scale 0
    assert np.any(sc[0, 20:24, 2] == 0x7C00)                                          # an infinite scale


def test_proxy_pack_incremental(proxy_rows):
    """basalt's runtime packs the copy as rows arrive (mtp.cpp:680, 1446), its test in one call: two calls give the
    one call's bytes, with n_cap = rows + 100 (629 tiles, not 625: the scales start where n_cap puts them)."""
    rr = len(proxy_rows)
    one = gpu.q5k_proxy_pack(proxy_rows, rr + 100, [(0, rr)])
    two = gpu.q5k_proxy_pack(proxy_rows, rr + 100, [(0, 7777), (7777, rr - 7777)])
    assert np.array_equal(one, two)


def proxy_map(rr):
    """:250: rows i % 7 == 3 and the last 300 rows without a token, the others id 5i + 1."""
    i = np.arange(rr)
    return np.where((i % 7 == 3) | (i >= rr - 300), -1, 5 * i + 1).astype(np.int32)


@pytest.fixture(scope="module")
def proxy_copy(proxy_rows):
    """The copy of the rescaled rows with basalt's spare capacity (:216, :253)."""
    n_cap = len(proxy_rows) + 100
    return gpu.q5k_proxy_pack(proxy_rows, n_cap, [(0, len(proxy_rows))]), n_cap


def test_proxy_score_emulation_bit_for_bit(proxy_copy, data):
    """px::score_kernel's candidates and tile partials bit for bit against the numpy emulation
    (basalt_emu.q5k_proxy_score, with the GPU's expf_ftz for __expf), at a row count that ends inside a tile (19,990:
    a masked tail) with basalt's map (:250), on the first 40 tiles and the last 8."""
    proxy, n_cap = proxy_copy
    rr = n_cap - 110
    mp = proxy_map(rr)
    xc = column(data[1], 1)
    cand, part = gpu.q5k_proxy_score(proxy, xc, rr, n_cap, mp)
    t = kernels.basalt_q5k_proxy_tiles(rr)
    tiles = np.r_[0:40, t - 8:t]
    e_cand, e_part = emu.q5k_proxy_score(proxy, xc, rr, n_cap, mp, tiles, gpu.expf_ftz)
    assert bad_bits(cand[tiles], e_cand) == 0 and bad_bits(part[tiles], e_part) == 0
    assert np.all(np.isneginf(cand[t - 1, :, 0]))       # the last tile: rows 19,968.. masked (map's last 300 rows)


def dequant(row):
    """A Q5_K row's values as basalt's test computes them on the host (:270-290): d * sc * q - dmin * m in float."""
    b = np.asarray(row, np.uint8).reshape(10, 176)
    d, dmin = (b[:, :4].copy().view(np.float16).astype(F32)[:, k] for k in (0, 1))
    out = np.zeros((10, 8, 32), F32)
    for j in range(8):
        scl = b[:, 4:16].astype(np.int64)
        if j < 4:
            sci, mi = scl[:, j] & 63, scl[:, j + 4] & 63
        else:
            sci = (scl[:, j + 4] & 0xF) | ((scl[:, j - 4] >> 6) << 4)
            mi = (scl[:, j + 4] >> 4) | ((scl[:, j] >> 6) << 4)
        qs = b[:, 48 + 32 * (j // 2): 80 + 32 * (j // 2)].astype(np.int64)
        q = ((qs >> 4) if j & 1 else (qs & 15)) | (((b[:, 16:48].astype(np.int64) >> j) & 1) << 4)
        out[:, j] = (d * sci.astype(F32))[:, None] * q.astype(F32) - (dmin * mi.astype(F32))[:, None]
    return out.reshape(-1)


def trial_x(rng, q5, mp):
    """:263-293: 0.8 N(0, 1) noise plus two planted rows among those with a token, the pick at 20 / |row|^2 and a
    runner-up at (16 + 0..3) / |row|^2, as a head's logits have."""
    xh = (F32(0.8) * rng.standard_normal(N_IN).astype(F32)).astype(F32)
    for pl in range(2):
        r = int(rng.integers(len(mp)))
        while mp[r] < 0:
            r = int(rng.integers(len(mp)))
        dq = dequant(q5[r]).astype(np.float64)
        alpha = (20.0 if pl == 0 else 16.0 + int(rng.integers(4))) / float(np.sum(dq * dq))
        xh = (xh + (alpha * dq).astype(F32)).astype(F32)
    return xh


def test_proxy_top1(proxy_copy, proxy_rows):
    """:254-349: the two-stage argmax against the exact one, 400 trials on the rescaled rows with basalt's map; a miss
    is a pick other than map[host argmax over native_mmvq's logits] (the first maximum, basalt's ``a[i] > a[best]``),
    and for a hit |p * sum - 1| with the host double softmax sum. Each trial's id and probability are also checked bit
    for bit against the numpy emulation of collect and pick (basalt_emu.q5k_proxy_final) on the score's own outputs
    and native_mmvq's logits of the rescored rows."""
    proxy, n_cap = proxy_copy
    rr = len(proxy_rows)
    mp = proxy_map(rr)
    rows = np.flatnonzero(mp >= 0)
    rng = np.random.default_rng(349)
    trials, miss, max_p_rel, emu_bad, nonfinite = 400, 0, 0.0, 0, 0
    for _ in range(trials):
        xq1 = gpu.q8_1(trial_x(rng, proxy_rows, mp), "native")
        a = gpu.mmvq(13, proxy_rows, xq1, N_IN, rr, 1)[0]
        pid, p, cand, part = gpu.q5k_proxy_top1(proxy, proxy_rows, xq1, rr, n_cap, mp)
        best = int(rows[np.argmax(a[rows])])
        if pid == mp[best]:
            s = float(np.sum(np.exp(a[rows].astype(np.float64) - np.float64(a[best]))))
            max_p_rel = max(max_p_rel, abs(float(p) * s - 1.0))
        else:
            miss += 1
        nonfinite += int(not np.isfinite(p))       # basalt's std::max fold (:315) drops a NaN (flash-next.md)
        e_id, e_p, _ = emu.q5k_proxy_final(cand, part, a, mp, gpu.expf_ftz)
        emu_bad += e_id != pid or bad_bits(p, e_p) != 0
    # basalt: ``if (miss > trials / 50 || max_p_rel > 1.0) fail = 1;`` (:349)
    assert miss <= trials // 50 and max_p_rel <= 1.0 and nonfinite == 0, (miss, max_p_rel, nonfinite)
    assert emu_bad == 0, emu_bad


def test_proxy_top1_edges(proxy_copy, proxy_rows):
    """Two cases basalt's test does not reach. Every row masked: collect's early exit, map[0] (0 when negative) with
    probability 1 (head_proxy.cuh:193-196). A zero column: every approximate score is 0, so every candidate is within
    kDelta and the list keeps 64 of them; the port's fixed order decides which, checked bit for bit against the
    emulation."""
    proxy, n_cap = proxy_copy
    rr = len(proxy_rows)
    zero = gpu.q8_1(np.zeros(N_IN, F32), "native")
    pid, p, _, _ = gpu.q5k_proxy_top1(proxy, proxy_rows, zero, rr, n_cap, np.full(rr, -1, np.int32))
    assert pid == 0 and p == 1.0
    mp = proxy_map(rr)
    pid, p, cand, part = gpu.q5k_proxy_top1(proxy, proxy_rows, zero, rr, n_cap, mp)
    e_id, e_p, listed = emu.q5k_proxy_final(cand, part, gpu.mmvq(13, proxy_rows, zero, N_IN, rr, 1)[0], mp,
                                            gpu.expf_ftz)
    assert len(listed) == 64 and pid == e_id == mp[0] and bad_bits(p, e_p) == 0, (pid, e_id, p, e_p)
