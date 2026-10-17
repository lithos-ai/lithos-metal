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


# ---- the q8_1 quantizers (matvec.md sections 1 and 3) -------------------------------------------------------------

RCP127 = np.uint32(0x3C010204).view(F32)      # fl(1/127): ptxas's multiply for the native amax / 127


def butterfly(v, op):
    """basalt's XOR butterfly over the 32 lanes of each row of ``v`` (offsets 16, 8, 4, 2, 1); every lane ends equal."""
    for o in (16, 8, 4, 2, 1):
        v = op(v, v[:, np.arange(32) ^ o])
    return v[:, 0]


def q8_1_finite(v):
    v = np.asarray(v, F32)
    return np.where(np.abs(v) > F32(65504.0), np.copysign(F32(65504.0), v), v).astype(F32)


def q8_1_codes(q, amax):
    with np.errstate(invalid="ignore"):
        c = np.where(np.isnan(q), 0, np.clip(q, -127, 127)).astype(np.int8)
    return np.where((amax == 0)[:, None], np.int8(0), c)


def q8_1_pack(d, s, codes):
    out = np.zeros((len(d), 36), np.uint8)
    out[:, 0:2] = half_bits(q8_1_finite(d)).view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = half_bits(q8_1_finite(s)).view(np.uint8).reshape(-1, 2)
    out[:, 4:] = codes.view(np.uint8)
    return out.reshape(-1)


def q8_1_native(x):
    """native_quantize_q8_1 as nvcc compiles it, with the correctly rounded reciprocal in place of MUFU.RCP."""
    v = np.asarray(x, F32).reshape(-1, 32)
    amax = butterfly(ftz(np.abs(v)), lambda p, q: np.fmax(ftz(p), ftz(q)))
    s = butterfly(v, add_ftz)
    d = q8_1_finite(mul_ftz(amax, RCP127))
    with np.errstate(invalid="ignore"):
        q = roundf(mul_ftz(v, rcp_ftz(d)[:, None]))
    return q8_1_pack(d, s, q8_1_codes(q, amax))


def q8_1_ieee(x):
    """quantize_q8_1_kernel (IEEE: correctly rounded divisions, subnormals kept), the host formula numpy computes."""
    v = np.asarray(x, F32).reshape(-1, 32)
    amax = np.fmax.reduce(np.abs(v), axis=1)                  # fmaxf drops a NaN operand
    s = butterfly(v, np.add)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        d = q8_1_finite(amax / F32(127.0))
        q = roundf(v / d[:, None])
    return q8_1_pack(d, s, q8_1_codes(q, amax))
