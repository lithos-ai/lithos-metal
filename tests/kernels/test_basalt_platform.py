"""The platform facts the exact basalt port rests on, measured on this GPU in MTLMathModeSafe (the repository default):
fp32 add, multiply, fma and division round to nearest even; how they treat subnormals; fp16 conversions; and that
``#pragma STDC FP_CONTRACT OFF`` stops contraction in a runtime-compiled source, told apart the way basalt's
elementwise_parity does (src/kernels/elementwise_parity.cpp:410: ``if (mismatches || guards || fma_diff == 0) ++bad;``
- the kernel must match the unfused result, and the inputs must be ones where fusing would change it)."""

import numpy as np
import pytest

from monolith import kernels
from monolith.runtime import _native as nt
from tests.kernels import basalt_emu as emu

F32 = np.float32
NCOL = 14                                    # one output column per operation of OPS

OPS = """
kernel void ops(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                device const float* c [[buffer(2)]], device float* out [[buffer(3)]],
                constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float x = a[i], y = b[i], z = c[i];
  device float* o = out + NCOL * i;
  o[0] = x + y;
  o[1] = x * y;
  o[2] = fma(x, y, z);
  o[3] = x / y;
  o[4] = x * y + z;
  o[5] = fmax(x, y);
  o[6] = fabs(x);
  o[7] = add_ftz(x, y);
  o[8] = mul_ftz(x, y);
  o[9] = fma_ftz(x, y, z);
  o[10] = max_ftz(x, y);
  o[11] = rcp_ftz(x);
  o[12] = abs_ftz(x);
  o[13] = fmin(x, y);
}
"""


def run_ops(a, b, c, pragma=True):
    dev = nt.Device()
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + OPS
    if not pragma:
        src = src.replace("#pragma STDC FP_CONTRACT OFF", "")
    pso = nt.Pipeline(nt.Library(dev, src, {"NCOL": str(NCOL)}), "ops")
    n = len(a)
    out = nt.Buffer(dev, 4 * NCOL * n)
    bufs = [nt.Buffer(dev, np.ascontiguousarray(v, F32).tobytes()) for v in (a, b, c)]
    d = nt.Dispatch().pipeline(pso)
    for k, buf in enumerate(bufs):
        d = d.buffer(k, buf)
    d = d.buffer(3, out).bytes(4, np.uint32(n).tobytes()).grid((n + 255) // 256).threadgroup(256)
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    return np.frombuffer(out.read(0, 4 * NCOL * n), F32).reshape(n, NCOL)


def normals(rng, n, lo=100, hi=154):
    """Random float32 with biased exponents in [lo, hi): far from overflow and from the subnormal range."""
    bits = (rng.integers(0, 2, n, dtype=np.uint32) << 31) | (rng.integers(lo, hi, n, dtype=np.uint32) << 23) \
        | rng.integers(0, 1 << 23, n, dtype=np.uint32)
    return bits.view(F32)


def ulp(x):
    return (np.nextafter(np.abs(x), F32(np.inf)) - np.abs(x)).astype(F32)


def test_rounding_is_to_nearest_even():
    rng = np.random.default_rng(1)
    n = 1 << 20
    a, b, c = normals(rng, n), normals(rng, n), normals(rng, n)
    # planted ties: a + ulp(a) / 2 (both neighbours' parities occur), and the same as fma(a, 1, ulp / 2)
    t = 1 << 16
    a[:t] = normals(rng, t, 120, 135)
    b[:t] = ulp(a[:t]) * F32(0.5) * np.sign(rng.standard_normal(t)).astype(F32)
    b[t: 2 * t], c[t: 2 * t] = F32(1.0), ulp(a[t: 2 * t]) * F32(0.5)
    # products with 13-bit significands: exact 25- and 26-bit products, a quarter of them ties
    m = rng.integers(1 << 12, 1 << 13, (2, t)).astype(F32) * F32(2.0 ** -12)
    a[2 * t: 3 * t], b[2 * t: 3 * t] = m
    got = run_ops(a, b, c)
    with np.errstate(over="ignore", under="ignore"):
        want = [a + b, a * b, emu.fma32(a, b, c), a / b]
    for k, name in enumerate(["add", "mul", "fma", "div"]):
        assert np.array_equal(got[:, k].view(np.uint32), want[k].view(np.uint32)), name
    exact = a[2 * t: 3 * t].astype(np.float64) * b[2 * t: 3 * t]
    assert np.count_nonzero(exact != (a[2 * t: 3 * t] * b[2 * t: 3 * t]).astype(np.float64)) > t // 8   # ties exercised
    # round toward zero would have differed on about half of the inexact random results
    assert np.count_nonzero((a + b).astype(np.float64) != a.astype(np.float64) + b) > n // 4


def test_subnormals_in_safe_mode():
    """Measured: add, multiply, fma and division flush a subnormal input and a subnormal result to the zero of its
    sign (MSL section 8.1 permits it); each input case has a normal exact result, so an input flush is told apart from
    a result flush. fmax and fmin do not flush: their comparison treats a subnormal as zero, and the chosen operand
    comes back unflushed, the second one on a tie. fabs and moves keep the bits. Hence the *_ftz helpers flush
    explicitly (test_ftz_helpers_match_the_emulation)."""
    s, s2 = F32(1.5e-39), F32(3e-39)         # subnormals, s < s2
    m = F32(2.0 ** -126)                     # the smallest normal
    rows = np.array([(s, m, 0), (-s, 2.0 ** 100, 0), (m, 1, s), (1.5 * m, -m, 0), (-1.5 * m, m, 0), (m, 0.5, 0),
                     (-m, 0.5, 0), (m, 2, 0), (s, 0, 0), (0, s, 0), (s2, s, 0), (s, -1, 0), (0, -0.0, 0)], F32)
    got = run_ops(rows[:, 0], rows[:, 1], rows[:, 2])
    bits = got.view(np.uint32)
    sb, nsb, neg0 = s.view(np.uint32), (-s).view(np.uint32), 0x80000000
    # a subnormal input flushed to the zero of its sign, where the exact result is normal
    assert got[0, 0] == m                                        # s + m (exact m + s)
    assert bits[1, 1] == neg0                                    # -s * 2^100 (exact -1.9e-9)
    assert bits[0, 3] == 0                                       # s / m (exact 0.128)
    assert got[2, 2] == m                                        # fma(m, 1, s) (exact m + s)
    # a subnormal result of normal inputs flushed to the zero of its sign
    assert bits[3, 0] == 0 and bits[4, 0] == neg0                # 1.5m - m, -1.5m + m
    assert bits[5, 1] == 0 and bits[6, 1] == neg0                # m * 0.5, -m * 0.5
    assert bits[5, 2] == 0 and bits[6, 2] == neg0                # fma(m, 0.5, 0), fma(-m, 0.5, 0)
    assert bits[5, 4] == 0 and bits[7, 3] == 0                   # m * 0.5 + 0, m / 2
    assert got[7, 1] == F32(2.0 ** -125)                         # normal results are untouched
    # fmax and fmin: a subnormal compares as zero, the chosen operand comes back unflushed, the second on a tie
    assert bits[8, 5] == 0 and bits[9, 5] == sb                  # fmax(s, 0) = +0, fmax(0, s) = s
    assert bits[10, 5] == sb                                     # fmax(s2, s) = s, the smaller
    assert bits[11, 5] == sb                                     # fmax(s, -1) = s
    assert bits[12, 5] == neg0                                   # fmax(+0, -0) = -0
    assert bits[8, 13] == 0 and bits[9, 13] == sb                # fmin(s, 0) = +0, fmin(0, s) = s
    assert bits[1, 13] == nsb                                    # fmin(-s, 2^100) = -s
    assert bits[0, 6] == sb and bits[1, 6] == sb                 # fabs keeps a subnormal


def test_ftz_helpers_match_the_emulation():
    """Each *_ftz helper bit for bit against basalt_emu, on random values of every exponent below 2^13 and planted
    blocks: subnormal inputs (and pairs of them), sums and fma results that land in the subnormal range, exact
    subnormal products beside a normal addend (fused, so they count), and reciprocals that underflow. On this GPU the
    arithmetic flushes anyway (test_subnormals_in_safe_mode); fmax and fabs do not, so max_ftz and abs_ftz carry it."""
    rng = np.random.default_rng(2)
    n, t = 1 << 18, 1 << 14
    a, b, c = normals(rng, n, 0, 140), normals(rng, n, 0, 140), normals(rng, n, 0, 140)
    a[: 2 * t], b[:t] = normals(rng, 2 * t, 0, 1), normals(rng, t, 0, 1)        # subnormal (or zero) inputs
    a[2 * t: 3 * t], b[2 * t: 3 * t] = normals(rng, t, 1, 3), normals(rng, t, 1, 3)   # sums below 2^-126
    p = slice(3 * t, 4 * t)                  # exact products in [2^-150, 2^-102) beside an addend in [2^-126, 2^-123)
    a[p], b[p], c[p] = normals(rng, t, 52, 76), normals(rng, t, 52, 76), normals(rng, t, 1, 4)
    q = slice(4 * t, 5 * t)                  # fma(a, b, -fl(a * b)): the product's rounding error, often subnormal
    a[q], b[q] = normals(rng, t, 40, 88), normals(rng, t, 40, 88)
    c[q] = -(a[q] * b[q])
    a[5 * t: 6 * t] = normals(rng, t, 251, 255)                                # 1 / a below 2^-126
    got = run_ops(a, b, c)
    fa, fb = emu.ftz(a), emu.ftz(b)
    # the port takes max_ftz of flushed absolute values only, so no -0 reaches it; on +0 against -0 this GPU's fmax
    # returns the second operand, numpy's fmax +0, and CUDA's FMNMX was not checked: those pairs are left out
    mixed_zeros = (fa == 0) & (fb == 0) & (np.signbit(fa) != np.signbit(fb))
    want = {7: emu.add_ftz(a, b), 8: emu.mul_ftz(a, b), 9: emu.fma_ftz(a, b, c), 10: emu.max_ftz(a, b),
            11: emu.rcp_ftz(a), 12: emu.ftz(np.abs(a))}
    for k, w in want.items():
        keep = ~mixed_zeros if k == 10 else slice(None)
        assert np.array_equal(got[keep, k].view(np.uint32), w[keep].view(np.uint32)), k


@pytest.mark.parametrize("pragma", [True, False])
def test_contraction_is_off_with_the_pragma(pragma):
    rng = np.random.default_rng(3)
    n = 1 << 16
    a, b = normals(rng, n, 120, 135), normals(rng, n, 120, 135)
    c = -(a * b) * (F32(1.0) + rng.standard_normal(n).astype(F32) * F32(1e-3))     # cancellation: fusing shows
    got = run_ops(a, b, c, pragma)[:, 4]
    unfused = (a * b) + c
    fused = emu.fma32(a, b, c)
    fma_diff = np.count_nonzero(fused.view(np.uint32) != unfused.view(np.uint32))
    mismatches = np.count_nonzero(got.view(np.uint32) != unfused.view(np.uint32))
    assert fma_diff > n // 2                                      # the inputs tell fused from unfused
    if pragma:
        assert mismatches == 0
    else:                                                         # the default (safe mode) contracts a * b + c
        assert np.array_equal(got.view(np.uint32), fused.view(np.uint32))


def test_half_conversions():
    dev = nt.Device()
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + """
kernel void conv(device const ushort* h [[buffer(0)]], device const float* f [[buffer(1)]],
                 device float* hf [[buffer(2)]], device ushort* fh [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  hf[i] = h2f(h[i]);
  fh[i] = as_type<ushort>(half(f[i]));
}
"""
    pso = nt.Pipeline(nt.Library(dev, src, {}), "conv")
    n = 1 << 16
    h = np.arange(n, dtype=np.uint16)
    rng = np.random.default_rng(4)
    f = normals(rng, n, 96, 145)                                  # fp16 normals, subnormals, overflow and ties
    f[: 1 << 12] = (h[: 1 << 12].view(np.float16).astype(np.float64) * (1 + 2.0 ** -11)).astype(F32)
    hf, fh = nt.Buffer(dev, 4 * n), nt.Buffer(dev, 2 * n)
    r = nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, h.tobytes()))
                           .buffer(1, nt.Buffer(dev, f.tobytes())).buffer(2, hf).buffer(3, fh)
                           .grid(n // 256).threadgroup(256)])
    assert not r.error, r.error
    got = np.frombuffer(hf.read(0, 4 * n), F32)
    finite = np.isfinite(h.view(np.float16))
    assert np.array_equal(got[finite].view(np.uint32), emu.h2f(h)[finite].view(np.uint32))   # exact, fp16 subnormals too
    assert np.array_equal(got[~finite], emu.h2f(h)[~finite], equal_nan=True)   # infinities exact; NaN stays NaN
    assert np.array_equal(np.frombuffer(fh.read(0, 2 * n), np.uint16), emu.half_bits(f))     # round to nearest even
