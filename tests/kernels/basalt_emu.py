"""numpy emulation of basalt's float32 arithmetic as nvcc compiles it (docs/research/basalt-nvcc/matvec.md), for the
bit-for-bit tests of the Metal port. Every float operation rounds once to float32, to nearest even: numpy's float32
add, multiply and divide do; ``fma32`` does by rounding the exact float64 sum to odd before the float32 cast. The
``*_ftz`` forms flush subnormal inputs and results to the zero of their sign, as SASS's ``.FTZ`` instructions do."""

import numpy as np

F32 = np.float32


def ftz(x):
    """A subnormal becomes the zero of its sign; every other value keeps its bits."""
    x = np.asarray(x, F32)
    b = x.view(np.uint32)
    return np.where((b & 0x7F800000) == 0, (b & 0x80000000).view(F32), x).astype(F32)


def fma32(a, b, c):
    """a * b + c rounded once to float32 (to nearest even). a * b is exact in float64; the float64 sum is rounded to
    odd (TwoSum gives its exact error), which a float32 cast then rounds correctly (53 >= 24 + 2 bits)."""
    a, b, c = (np.asarray(v, F32).astype(np.float64) for v in (a, b, c))
    s = a * b
    t = s + c
    bp = t - s
    err = (s - (t - bp)) + (c - bp)
    even = (t.view(np.int64) & 1) == 0
    bump = (err != 0) & even & np.isfinite(t)
    t = np.where(bump, np.nextafter(t, np.where(err > 0, np.inf, -np.inf)), t)
    with np.errstate(over="ignore"):
        return t.astype(F32)


def add_ftz(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) + ftz(b))


def mul_ftz(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) * ftz(b))


def fma_ftz(a, b, c):
    return ftz(fma32(ftz(a), ftz(b), ftz(c)))


def rcp_ftz(x):
    """The correctly rounded reciprocal the port writes where SASS has MUFU.RCP."""
    with np.errstate(divide="ignore", over="ignore"):
        return ftz(F32(1.0) / ftz(x))


def roundf(x):
    """CUDA's roundf: half away from zero (trunc of |x| + 0.5 rounded toward zero, which is exact here)."""
    x = np.asarray(x, F32).astype(np.float64)
    return np.trunc(x + np.copysign(0.5, x)).astype(F32)


def half_bits(x):
    """float32 to fp16 bits, round to nearest even (CUDA's __float2half, Metal's half())."""
    with np.errstate(over="ignore"):
        return np.asarray(x, F32).astype(np.float16).view(np.uint16)


def h2f(bits):
    return np.asarray(bits, np.uint16).view(np.float16).astype(F32)
