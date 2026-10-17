"""basalt's native_mmvq on Metal (kernels/common/basalt_mmvq.metal), one column and several, for the ten GGUF types.

- mmvq_multi_parity, ported (basalt src/kernels/mmvq_multi_parity.cpp): the cases of :69-77 at widths 1, 2, 3, 4, 5,
  6, 8 with weights of random bytes and fp16 scales in +-[2^-10, 2^-5) (:79-83) and N(0, 1) activations quantized
  once by the IEEE quantizer (:143-145); T columns against T one-column calls, criterion quoted:
  ``if (exact && diff != 0) ++bad;`` (:215), any non-finite output fails (:531-535). Its negative control: the
  upstream layout must differ somewhere at T > 4 in every case, else ``if (off.powerless_cases != 0) {`` (:555)
  fails for lack of power. The tensor-core section (:333-348) is skipped as basalt skips it without int8 tensor
  cores. The Q8_0 one-column section (:390-472): the one-column call bitwise column 0 of a two-column call and
  ``const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);`` (:461) against a float64 dot on the same bytes.
- New float64 references under :461's rule for Q4_0, Q5_0, Q4_K, Q5_K and Q6_K, which basalt does not check
  against a reference (pr-b-scope-basalt.md section 3.7).
- Bitwise invariance for every type: each column of a T-column call equals the one-column call, T = 1 to 8, in the
  small-K and the regular layout.
- Bit for bit against the numpy emulation of the arithmetic nvcc compiles (tests/kernels/basalt_emu.py)."""

import numpy as np
import pytest

from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32 = np.float32
NAMES = {2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 20: "IQ4_NL", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 23: "IQ4_XS",
         42: "Q2_0"}
SCALES = {42: [0], 2: [0], 6: [0], 8: [0], 11: [108], 12: [0, 2], 13: [0, 2], 14: [208], 20: [0], 23: [0]}


def sane_halves(rng, n):
    """mmvq_multi_parity.cpp:79-83: a normal fp16 in +-[2^-10, 2^-5): exponent field 5..9, any mantissa and sign."""
    return ((rng.integers(0, 2, n) << 15) | (rng.integers(5, 10, n) << 10) | rng.integers(0, 1024, n)).astype(np.uint16)


def weights(type_id, n_in, n_out, rng):
    div, _, _, bb = emu.TRAITS[type_id]
    nb = n_out * (n_in // div)
    w = rng.integers(0, 256, (nb, bb), dtype=np.uint8)
    for at in SCALES[type_id]:
        w[:, at: at + 2] = sane_halves(rng, nb).view(np.uint8).reshape(nb, 2)
    return w.reshape(n_out, -1)


def activations(n_in, ncols, rng, kind="ieee"):
    x = rng.standard_normal(ncols * n_in).astype(F32)
    return x, gpu.q8_1(x, kind)


def columns(xq, n_in, j):
    qcol = n_in // 32 * 36
    return xq[j * qcol: (j + 1) * qcol]


CASES = [("Q4_0", 2, 2048, 512), ("Q8_0", 8, 2048, 512), ("Q4_K", 12, 2048, 512), ("Q5_K", 13, 2048, 512),
         ("Q6_K", 14, 2048, 512), ("IQ4_XS", 23, 4096, 512), ("Q5_K wide", 13, 4096, 2048)]
WIDTHS = [1, 2, 3, 4, 5, 6, 8]


def multi_vs_single(type_id, n_in, n_out, t, upstream, seed):
    rng = np.random.default_rng(seed)
    w = weights(type_id, n_in, n_out, rng)
    _, xq = activations(n_in, t, rng)
    # outputs pre-filled apart, so a row neither call writes differs; the multi side's 0xff is NaN, so an unwritten
    # row of the control's layout is non-finite and cannot pass for its power
    multi = gpu.mmvq(type_id, w, xq, n_in, n_out, t, upstream=upstream, fill=0xFF)
    single = np.stack([gpu.mmvq(type_id, w, columns(xq, n_in, j), n_in, n_out, 1, fill=0xFE)[0] for j in range(t)])
    return multi, single


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mmvq_multi_parity(case):
    name, type_id, n_in, n_out = case
    bad = nonfinite = 0
    for t in WIDTHS:
        multi, single = multi_vs_single(type_id, n_in, n_out, t, False, 1234 + type_id * 97 + t)
        nonfinite += np.count_nonzero(~np.isfinite(multi)) + np.count_nonzero(~np.isfinite(single))
        bad += np.count_nonzero(multi.view(np.uint32) != single.view(np.uint32)) != 0
    # basalt: "any non-finite output fails" (:531-535) and ``if (exact && diff != 0) ++bad;`` (:215)
    assert nonfinite == 0, name
    assert bad == 0, name


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mmvq_multi_parity_negative_control(case):
    """The upstream layout groups the partial sums differently above 4 columns: the comparison must see it. T = 1 is
    the one-column launch in both sweeps, as in basalt (ncols 1 never reaches launch_multi)."""
    name, type_id, n_in, n_out = case
    wide = nonfinite = 0
    for t in WIDTHS:
        multi, single = multi_vs_single(type_id, n_in, n_out, t, True, 1234 + type_id * 97 + t)
        nonfinite += np.count_nonzero(~np.isfinite(multi)) + np.count_nonzero(~np.isfinite(single))
        diff = np.count_nonzero(multi.view(np.uint32) != single.view(np.uint32))
        if t <= 4:
            assert diff == 0, (name, t)          # the layouts coincide up to 4 columns (4 SIMD-groups, 2 rows)
        else:
            wide += diff
    # basalt: case_wide counts the differing finite outputs at T > 4; ``if (off.powerless_cases != 0) {`` (:555) fails
    assert nonfinite == 0 and wide > 0, name


def half_to_double(h):
    return h.view(np.float16).astype(np.float64)


Q80_GEOMS = [(2560, 12288), (6144, 2560), (2560, 512), (2560, 2048), (512, 640), (1024, 130), (2560, 1)]


@pytest.mark.parametrize("geom", Q80_GEOMS, ids=[f"{a}x{b}" for a, b in Q80_GEOMS])
def test_q8_0_one_column(geom):
    n_in, n_out = geom
    rng = np.random.default_rng(4242 + n_in + n_out)
    w = weights(8, n_in, n_out, rng)
    x = rng.standard_normal(n_in).astype(F32)
    xq = gpu.q8_1(np.concatenate([x, x]), "ieee")
    a = gpu.mmvq(8, w, columns(xq, n_in, 0), n_in, n_out, 1, fill=0xFF)[0]    # basalt's pre-fills (:426-427)
    b = gpu.mmvq(8, w, xq, n_in, n_out, 2, fill=0xFE)[0]
    ref = reference(8, w, columns(xq, n_in, 0), n_in, n_out)
    diff = np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))
    worst, scale = np.max(np.abs(a.astype(np.float64) - ref)), np.max(np.abs(ref))
    # basalt: ``const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);`` (:461)
    assert diff == 0 and worst <= 1e-4 * (scale + 1e-30), (diff, worst, scale)


def reference(type_id, w, xq, n_in, n_out):
    """The float64 product on the same bytes: the weights' exact values (fp16 scales times integer codes) times the
    q8_1 activations' d8 * code; for Q4_0 and Q5_0 the offset term takes the stored fp16 block sum s8, as the kernel's
    definition does (native_mmvq.cu:693-695: ``d * (sumi * d8 - k * s8)``)."""
    div, _, _, bb = emu.TRAITS[type_id]
    blk = np.asarray(w, np.uint8).reshape(n_out, n_in // div, bb)
    xb = np.asarray(xq, np.uint8).reshape(n_in // 32, 36)
    d8 = half_to_double(xb[:, 0:2].copy().view(np.uint16)[:, 0])
    s8 = half_to_double(xb[:, 2:4].copy().view(np.uint16)[:, 0])
    a = xb[:, 4:].view(np.int8).astype(np.float64)
    xv = (d8[:, None] * a).reshape(-1)                                         # exact
    h = lambda off: half_to_double(blk[..., off: off + 2].copy().view(np.uint16)[..., 0])
    if type_id in (8, 2, 6):
        if type_id == 8:
            c = blk[..., 2:34].view(np.int8).astype(np.float64)
        else:
            qs = blk[..., 6:22] if type_id == 6 else blk[..., 2:18]
            c = np.concatenate([qs & 15, qs >> 4], axis=-1).astype(np.int64)
            if type_id == 6:
                qh = blk[..., 2:6].copy().view(np.uint32)[..., 0].astype(np.int64)
                c = c | (((qh[..., None] >> np.arange(32)) & 1) << 4)
            c = c.astype(np.float64)
        y = ((h(0)[..., None] * c).reshape(n_out, -1) * xv).sum(axis=1)
        if type_id != 8:
            y -= (h(0) * (8.0 if type_id == 2 else 16.0) * s8[None, :]).sum(axis=1)
        return y
    if type_id in (12, 13):
        sc = blk[..., 4:16]
        sb = np.arange(8)
        s, m = emu._scale_min_k4(np.broadcast_to(sc[..., None, :], sc.shape[:-1] + (8, 12)), sb[:, None])
        s, m = s[..., 0].astype(np.float64), m[..., 0].astype(np.float64)
        qs = blk[..., 48:176] if type_id == 13 else blk[..., 16:144]
        v = np.arange(256)
        q = (qs[..., 32 * (v // 64) + v % 32].astype(np.int64) >> (4 * ((v // 32) % 2))) & 15
        if type_id == 13:
            q = q | (((blk[..., 16 + v % 32].astype(np.int64) >> (v // 32)) & 1) << 4)
        val = h(0)[..., None] * s[..., v // 32] * q - h(2)[..., None] * m[..., v // 32]
        return (val.reshape(n_out, -1) * xv).sum(axis=1)
    if type_id == 14:
        v = np.arange(256)
        hh, l = v // 128, v % 128
        ql = blk[..., 64 * hh + l % 64].astype(np.int64) >> (4 * (l // 64)) & 15
        qh = blk[..., 128 + 32 * hh + l % 32].astype(np.int64) >> (2 * (l // 32)) & 3
        q = (ql | (qh << 4)) - 32
        sc = blk[..., 192:208].view(np.int8).astype(np.float64)
        val = h(208)[..., None] * sc[..., v // 16] * q
        return (val.reshape(n_out, -1) * xv).sum(axis=1)
    raise ValueError(type_id)


REF_TYPES = [8, 2, 6, 12, 13, 14]
REF_SHAPES = [(2560, 640), (640, 2560), (6144, 512), (2048, 512)]


@pytest.mark.parametrize("type_id", REF_TYPES, ids=[NAMES[t] for t in REF_TYPES])
def test_float64_reference(type_id):
    """mmvq_multi_parity.cpp:461's rule, for the types basalt leaves without a reference (and Q8_0 again at this
    pack's shapes): worst |y - ref| <= 1e-4 * max |ref| on the same q8_1 bytes, one column and every column of 8."""
    for n_in, n_out in REF_SHAPES:
        if n_in % emu.TRAITS[type_id][0]:
            continue
        rng = np.random.default_rng(type_id * 1000 + n_in)
        w = weights(type_id, n_in, n_out, rng)
        _, xq = activations(n_in, 8, rng, "native")
        y8 = gpu.mmvq(type_id, w, xq, n_in, n_out, 8)
        for j in range(8):
            ref = reference(type_id, w, columns(xq, n_in, j), n_in, n_out)
            worst, scale = np.max(np.abs(y8[j].astype(np.float64) - ref)), np.max(np.abs(ref))
            assert np.all(np.isfinite(y8[j])) and worst <= 1e-4 * (scale + 1e-30), (n_in, n_out, j, worst, scale)


# n_in for each type's small-K and regular layout (small K: fewer blocks per row than BPI)
SHAPES = {2: (640, 2560), 6: (640, 2560), 20: (640, 2560), 8: (640, 2560), 11: (1024, 2560), 12: (1024, 2560),
          13: (1024, 6144), 14: (512, 2560), 23: (2560, 4096), 42: (2560, 6144)}


@pytest.mark.parametrize("type_id", sorted(NAMES), ids=[NAMES[t] for t in sorted(NAMES)])
def test_column_invariance(type_id):
    """Every column of a T-column call equals basalt's one-column call (its own launch: 4 rows per threadgroup for
    small K) bit for bit, T = 1 to 8."""
    for n_in in SHAPES[type_id]:
        n_out = 71                                                       # odd: a row tail for ROWS 2 and 4
        rng = np.random.default_rng(type_id + n_in)
        w = weights(type_id, n_in, n_out, rng)
        _, xq = activations(n_in, 8, rng, "native")
        single = np.stack([gpu.mmvq(type_id, w, columns(xq, n_in, j), n_in, n_out, 1, fill=0xFE)[0] for j in range(8)])
        assert np.all(np.isfinite(single))
        for t in range(1, 9):
            multi = gpu.mmvq(type_id, w, xq[: t * (n_in // 32) * 36], n_in, n_out, t, fill=0xFF)
            assert np.array_equal(multi.view(np.uint32), single[:t].view(np.uint32)), (n_in, t)


@pytest.mark.parametrize("type_id", sorted(NAMES), ids=[NAMES[t] for t in sorted(NAMES)])
def test_emulation_bit_for_bit(type_id):
    """The kernel against the numpy emulation of what nvcc compiles, one column and three, both layouts' K."""
    for n_in in SHAPES[type_id]:
        n_out = 37
        rng = np.random.default_rng(99 + type_id + n_in)
        w = weights(type_id, n_in, n_out, rng)
        x = rng.standard_normal(3 * n_in).astype(F32)
        x[: 64] = 0.0                                                    # all-zero q8_1 blocks
        xq = gpu.q8_1(x, "native")
        for t in (1, 3):
            got = gpu.mmvq(type_id, w, xq[: t * (n_in // 32) * 36], n_in, n_out, t)
            want = emu.mmvq(type_id, w, xq[: t * (n_in // 32) * 36], n_in, n_out, t)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (n_in, t)
        up = gpu.mmvq(type_id, w, xq, n_in, n_out, 3, upstream=True)       # the upstream layout's NW = 4 at 3 columns
        assert np.array_equal(up.view(np.uint32), emu.mmvq(type_id, w, xq, n_in, n_out, 3).view(np.uint32))
