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

OPS = """
kernel void ops(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                device const float* c [[buffer(2)]], device float* out [[buffer(3)]],
                constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float x = a[i], y = b[i], z = c[i];
  out[8 * i + 0] = x + y;
  out[8 * i + 1] = x * y;
  out[8 * i + 2] = fma(x, y, z);
  out[8 * i + 3] = x / y;
  out[8 * i + 4] = x * y + z;
  out[8 * i + 5] = fmax(x, y);
  out[8 * i + 6] = fabs(x);
  out[8 * i + 7] = add_ftz(x, y) + mul_ftz(x, y) * 0.0f + fma_ftz(x, y, z) * 0.0f;
}
"""


def run_ops(a, b, c, pragma=True):
    dev = nt.Device()
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + OPS
    if not pragma:
        src = src.replace("#pragma STDC FP_CONTRACT OFF", "")
    pso = nt.Pipeline(nt.Library(dev, src, {}), "ops")
    n = len(a)
    out = nt.Buffer(dev, 32 * n)
    bufs = [nt.Buffer(dev, np.ascontiguousarray(v, F32).tobytes()) for v in (a, b, c)]
    d = nt.Dispatch().pipeline(pso)
    for k, buf in enumerate(bufs):
        d = d.buffer(k, buf)
    d = d.buffer(3, out).bytes(4, np.uint32(n).tobytes()).grid((n + 255) // 256).threadgroup(256)
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    return np.frombuffer(out.read(0, 32 * n), F32).reshape(n, 8)


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


def test_subnormals_flush_in_safe_mode():
    """Measured: add, multiply, fma, division and fmax flush a subnormal input and a subnormal result to the zero of
    its sign (MSL section 8.1 permits it); fabs and moves keep the bits. The *_ftz helpers flush either way."""
    s = F32(1.5e-39)                         # subnormal
    m = F32(2.0 ** -126)                     # the smallest normal
    a = np.array([s, -s, s, m, m, s, m], F32)
    b = np.array([F32(1.0), F32(1.0), m, F32(0.5), F32(0.5), F32(0.0), F32(2.0)], F32)
    c = np.array([F32(0.0), F32(0.0), F32(0.0), F32(0.0), F32(0.0), F32(0.0), F32(0.0)], F32)
    got = run_ops(a, b, c)
    bits = got.view(np.uint32)
    assert bits[0, 1] == 0 and bits[1, 1] == 0x80000000          # s * 1, -s * 1: input flushed (signed)
    assert bits[0, 2] == 0 and bits[0, 3] == 0                   # fma(s, 1, 0), s / 1
    assert got[2, 0] == m                                        # s + m = m: input flushed
    assert bits[3, 1] == 0 and bits[6, 3] == 0                   # m * 0.5, m / 2: subnormal result flushed
    assert bits[3, 2] == 0 and bits[3, 4] == 0                   # fma(m, 0.5, 0), m * 0.5 + 0
    assert bits[5, 5] == 0                                       # fmax(s, 0)
    assert bits[0, 6] == s.view(np.uint32) and bits[1, 6] == s.view(np.uint32)   # fabs keeps a subnormal
    assert got[6, 1] == F32(2.0 ** -125)                         # normal results are untouched


def test_ftz_helpers_match_the_emulation():
    rng = np.random.default_rng(2)
    n = 1 << 18
    a, b, c = normals(rng, n, 0, 140), normals(rng, n, 0, 140), normals(rng, n, 0, 140)
    a[: n // 8] = (rng.integers(0, 1 << 23, n // 8, dtype=np.uint32) | (rng.integers(0, 2, n // 8, dtype=np.uint32) << 31)).view(F32)
    got = run_ops(a, b, c)
    with np.errstate(under="ignore"):
        want = emu.add_ftz(a, b) + emu.mul_ftz(a, b) * F32(0.0) + emu.fma_ftz(a, b, c) * F32(0.0)
    assert np.array_equal(got[:, 7].view(np.uint32), emu.ftz(want).view(np.uint32))


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
    assert np.array_equal(np.frombuffer(fh.read(0, 2 * n), np.uint16), emu.half_bits(f))     # round to nearest even
