# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/pool.cpp:434 @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): the CPU expert pool's SwiGLU and its order of operations.
"""``expf_glibc`` and the SwiGLU of ``runtime/src/basalt_cpu_expf.cpp``, the one piece of arithmetic basalt's CPU
expert pool performs itself.

**What the bits are held against.** The function must return glibc's bits, because basalt runs on Linux
with glibc and a token's output has to be the same number whichever tier produced it. Group A is the acceptance
oracle and it is exhaustive: A1 the six values read straight out of a running glibc, and A2 the 4,226 recorded
(g, u, res) triples of basalt's own line; B4 carries the rest, glibc's ``expf`` at every one of the 4,294,967,296
float32 arguments as 4,096 block digests. No value ``expf_glibc`` returns, and no SwiGLU result at a recorded
argument pair, can be wrong without showing up in one of them (``docs/research/basalt-cpu-expf.md``,
``tests/kernels/fixtures/``).

One rule sits outside that: SSE takes the FIRST source operand that is a NaN, where an AArch64 instruction
prefers a signalling NaN wherever it sits. The two part only when `src1` is a quiet NaN and `src2` a signalling
one, and no recorded row has that pair, so D3 asserts it against the manual rather than against a measurement.

The other cases say what each part of the result is, which a digest cannot: group B holds the transcription to
the 0.502 ulp Arm records for this algorithm, against a float64 reference, over the SwiGLU's reachable range, and
names the two arguments at which glibc's own two x86-64 variants part; group C pins the specials and the three
thresholds against that same reference; group D pins the SwiGLU's order of operations and the NaN each of its
three instructions writes.

The float64 reference is ``np.exp`` in double, which carries 29 bits more than the float32 result and is itself
accurate to well under 2^-29 of a float32 ulp. Its own error is therefore invisible at the 10^-3 ulp the group B
assertions turn on. Where a case asserts the *correctly rounded* value it is written ``np.float32(np.exp(x64))``:
one rounding, from the double to the float.

Cases. **A**, the recorded x86 bits: A1 the six ``expf`` values, A2 the 4,226 SwiGLU vectors. **B**, the error
bound and the exhaustive sweep: B1 a stratified sample of [-40, 40] - the SwiGLU's reachable range, with most of
its mass near zero - B2 the measured worst case and the window around it, B3 the two arguments the rounding
step's fusion moves, and B4 all 4,096 block digests. **C**, the specials: C1 both zeros, C2 both infinities,
C3 NaN (quiet and signalling), C4 the overflow threshold and the largest argument with a finite result, C5 the
two underflow thresholds and the ``may_uflow`` band, C6 the subnormal results. **D**, the SwiGLU: D1 its order
against basalt's expression, D2 that the order is load-bearing, D3 the NaN rule at each instruction.

The suite bites, measured by mutating the kernel and rebuilding: ``poly`` in place of ``poly_scaled`` fails A1,
A2, B1, B2, B3, B4, C4 and C6; the platform's ``expf`` fails A2, B1, B2, B3, B4 and C6 - it exceeds the 0.502 ulp
this algorithm holds to, and differs from the transcription at 1,930,129 of the 2,218,786,818 float32 arguments in
[-40, 40] (measured here against this machine's own libm, so the count is a property of Apple's ``expf`` and moves
with macOS), though not at any of the recorded six, so A1 alone would miss it; the rounding step written as
``z + SHIFT`` and ``z - kd`` instead of the two ``fma()`` calls fails A2, B3 and B4, and B4 at exactly the two
blocks the fixture's trailer records ``__expf_sse2`` digests for, 0x420 and 0xc27, with those digests exactly - the
unfused build is ``__expf_sse2`` over those 2,097,152 arguments; and the SwiGLU as ``g * (1 / (1 + e)) * u``
fails D1.

The three x86 NaN rules are not alike there. Dropping the divide's for the C operator fails A2, D1 and D3;
dropping the multiply's fails A2 and D3, on the four recorded rows where the multiply itself creates the NaN from
``0 * inf`` and the C operator writes AArch64's sign (rows 122, 123, 126 and 127), not on the row with a NaN on
both sides, where this machine's own operand order happens to give x86's answer anyway. Dropping the add's fails
nothing: ``1.f`` is never a NaN, and the only other source is ``expf``'s own return, which is never negative and
quiets the same way on both architectures, so neither the add's source order nor its created NaN is reachable. It
is written as the add anyway, because the transcription is of what basalt executes and not of what a test can
catch.

Two mutations fail nothing, and both are findings rather than gaps. ``round``/``lround`` for the rounding - Arm's
other branch, the one glibc compiles on aarch64 - passes, because no float32 argument makes ``InvLn2N * x`` an
exact tie (swept here over all 2^32), so the two branches are the same function. Dropping the
``WANT_ERRNO_UFLOW`` band passes, because the main path computes 0x00000001 across the whole band anyway (C5
asserts exactly that); the band is transcribed because glibc has it, and ``errno``, which nothing here reads, is
the only thing it changes.

Run with ``-s`` for the measured worst ulp and its argument.
"""

from pathlib import Path

import numpy as np
import pytest

from monolith.runtime import _native as _nt

F32, U32 = np.float32, np.uint32

FIXTURES = Path(__file__).parent / "fixtures"      # the recorded x86 corpus, docs/research/basalt-cpu-expf.md

# Read out of a running glibc 2.39 on a Xeon Platinum 8375C, with the ifunc dispatch intact and again with
# GLIBC_TUNABLES disabling AVX2 and FMA (docs/research/basalt-cpu-expf.md section 1). The two variants agree on
# these six, which is why no variant is named here; B4's digests carry the arguments where they do not.
GLIBC_SIX = (
    (0x3F000000, 0x3FD3094C),      # 0.5
    (0xBF800000, 0x3EBC5AB2),      # -1
    (0x3A83126F, 0x3F8020C9),      # 0.001
    (0xC2AF0000, 0x006CB2BC),      # -87.5, a subnormal result
    (0x42B10000, 0x7F4CDCC4),      # 88.5, near overflow
    (0xC2C80000, 0x0000001B),      # -100, a tiny subnormal
)

# the thresholds expf.c branches on, as float32 bit patterns (runtime/src/basalt_cpu_expf.cpp)
OFLOW = 0x42B17217       # 0x1.62e42ep6f, log(0x1p128); x above it takes the overflow return
UFLOW = 0xC2CFF1B4       # -0x1.9fe368p6f, log(0x1p-150); x below it takes the underflow return
MAY_UFLOW = 0xC2CE8ECF   # -0x1.9d1d9ep6f, log(0x1p-149); x below it returns (0x1.4p-75f)^2


def f32(bits):
    """The float32 array of these bit patterns. Bits in, bits out: nothing converts through a double, so a
    signalling NaN reaches the kernel as one."""
    return np.asarray(bits, dtype=U32).view(F32)


def bits_of(x):
    return np.asarray(x, dtype=F32).view(U32)


def expf(x):
    out = np.empty_like(np.asarray(x, dtype=F32))
    _nt.basalt_expf(np.asarray(x, dtype=F32), out)
    return out


def swiglu(g, u):
    g, u = np.asarray(g, dtype=F32), np.asarray(u, dtype=F32)
    out = np.empty_like(g)
    _nt.basalt_swiglu(g, u, out)
    return out


def _recorded(name, ncols):
    """The hex columns of a fixture, as uint32 arrays. Comment and short lines are skipped, which drops the
    trailing __expf_sse2 block of expf_digests.txt with them."""
    cols = [[] for _ in range(ncols)]
    for line in (FIXTURES / name).read_text().splitlines():
        parts = line.split()
        if line.startswith("#") or len(parts) != ncols:
            continue
        for c, part in zip(cols, parts):
            c.append(int(part, 16))
    assert cols[0], f"{name} holds no {ncols}-column rows"
    return [np.array(c, dtype=np.uint64 if max(c) > 0xFFFFFFFF else U32) for c in cols]


def ulp_error(x):
    """|expf(x) - exp(x)| in units of the float32 ulp at the exact value, which is 2^-149 below 2^-126."""
    ref = np.exp(np.asarray(x, dtype=F32).astype(np.float64))
    exponent = np.maximum(np.floor(np.log2(np.abs(ref))), -126.0)
    ulp = np.ldexp(1.0, (exponent - 23).astype(np.int64))
    return np.abs(expf(x).astype(np.float64) - ref) / ulp


def correctly_rounded(x):
    """exp(x) in double, rounded once to float32 - the value an infinitely precise expf would return. The cast is
    allowed to overflow to +inf and to flush to zero: those are the right answers at the two ends."""
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(np.asarray(x, dtype=F32).astype(np.float64)).astype(F32)


# ---- A: the recorded x86 bits ------------------------------------------------------------------------------------

def test_a1_recorded_glibc_values():
    """A1. Six values read straight out of a running glibc 2.39, bit for bit - the only ones in this suite that
    came from a call rather than from a recorded sweep, and the case that says the sweep's own harness was
    measuring the same function."""
    arg, want = zip(*GLIBC_SIX)
    got = bits_of(expf(f32(arg)))
    for a, w, g in zip(arg, want, got):
        assert g == w, f"expf({a:#010x}) = {g:#010x}, glibc 2.39 gives {w:#010x}"


def test_a2_recorded_swiglu_vectors():
    """A2. basalt's own line, recorded: 4,226 (g, u, res) triples from pool.cpp:425-434 compiled as basalt compiles
    it (x86-64 gcc 14.3, -std=c++20 -O3 -DNDEBUG, no -m flags) and run on a Xeon 8375C and a Xeon 8488C under
    glibc 2.39-0ubuntu8.9, six runs agreeing. 57 special g against two u, 16 special-u pairs, and 4,096 random
    pairs with |g| < 40. This is the case that pins the whole expression - the variant of expf, the order gcc
    emits, and the NaN each instruction writes - rather than any part of it."""
    g_bits, u_bits, want = _recorded("swiglu_vectors.txt", 3)
    assert len(want) == 4226, f"expected 4,226 recorded vectors, read {len(want)}"
    got = bits_of(swiglu(f32(g_bits), f32(u_bits)))
    bad = np.nonzero(got != want)[0]
    assert len(bad) == 0, (f"{len(bad)} of {len(want)} rows differ, first at row {bad[0]}: "
                           f"g={g_bits[bad[0]]:#010x} u={u_bits[bad[0]]:#010x} "
                           f"{got[bad[0]]:#010x} against the recorded {want[bad[0]]:#010x}")


# ---- B: the error bound against a float64 reference --------------------------------------------------------------

def test_b1_ulp_bound_over_the_reachable_range(capsys):
    """B1. Arm records this algorithm's worst case as 0.502 ulp under round to nearest, and the transcription has
    to stay inside that. The sample is the SwiGLU's reachable range: uniform over [-40, 40], a normal sample with
    most of its mass near zero, the exact zero neighbourhood, and the whole underflow region, where the result is
    subnormal and the ulp is 2^-149."""
    rng = np.random.default_rng(20261018)
    sample = np.concatenate([
        rng.uniform(-40.0, 40.0, 1 << 20).astype(F32),                        # the range, flat
        rng.normal(0.0, 2.0, 1 << 20).astype(F32),                            # where the mass is
        rng.normal(0.0, 0.01, 1 << 16).astype(F32),                           # and nearer still
        f32(np.arange(0xC2AE0000, 0xC2D10000, dtype=U32)),                    # -87 .. -104.5, subnormal results
        f32(np.arange(0x42B00000, OFLOW + 1, dtype=U32)),                     # 88 .. 88.72, up to the threshold
    ])
    err = ulp_error(sample)
    worst = int(np.argmax(err))
    with capsys.disabled():
        print(f"\n  B1: {sample.size} arguments, worst {err[worst]:.6f} ulp at "
              f"{bits_of(sample)[worst]:#010x} ({float(sample[worst])!r})")
    assert err.max() <= 0.502, f"worst {err.max():.6f} ulp at {bits_of(sample)[worst]:#010x}"


def test_b2_the_measured_worst_case():
    """B2. The worst argument of an exhaustive sweep of every float32 whose result is finite and nonzero (measured
    here, 4,294,967,296 inputs): 0.501637 ulp, above a half and so not the correctly rounded value, which is what
    "not correctly rounded" means for this algorithm. The window around it is swept here so a transcription that
    drifts has to beat the bound somewhere in the neighbourhood too."""
    x = f32([0xBBE7328F])
    assert bits_of(expf(x))[0] == 0x3F7E333C
    assert bits_of(correctly_rounded(x))[0] == 0x3F7E333B, "the correctly rounded value is one ulp below"
    assert 0.5 < ulp_error(x)[0] <= 0.502
    window = f32(np.arange(0xBBE70000, 0xBBE80000, dtype=U32))
    assert ulp_error(window).max() <= 0.502


def test_b3_the_two_arguments_contraction_would_move():
    """B3. The two arguments where glibc's own variants disagree, and the only place the rounding step's fusion is
    visible: it changes the result at exactly two of the 4,294,967,296 float32 inputs (measured here, both orders
    swept). glibc's x86-64 ifunc picks __expf_fma on any host with FMA and AVX2, so these are basalt's bits, and
    the port writes those two mul-adds as explicit fma() to hold them whatever -ffp-contract says. Measured on a
    Xeon 8375C under glibc 2.39-0ubuntu8.9: __expf_fma gives these, __expf_sse2 gives 0x56FC9F1B and 0x11FA2992,
    which are the correctly rounded ones. basalt runs the fused variant, so the port is wrong when it agrees with
    the correctly rounded answer here."""
    assert list(bits_of(expf(f32([0x4202422F, 0xC27C65D9])))) == [0x56FC9F1C, 0x11FA2993]


def test_b4_every_block_of_the_recorded_sweep():
    """B4. glibc's own output at every one of the 4,294,967,296 float32 arguments, as 4,096 digests over blocks of
    2^20, measured on a Xeon 8375C and a Xeon 8488C under glibc 2.39-0ubuntu8.9 with the ifunc resolving to
    __expf_fma. Block k covers bit patterns k << 20 .. (k << 20) | 0xfffff in order; the digest is the FNV-1a of
    the result bits, h = 0xcbf29ce484222325 then h = (h ^ bits) * 0x100000001b3 mod 2^64 per argument, which
    ``_native.basalt_expf_digest`` computes. This is what closes the gap A1's six values leave: nothing about the
    transcription can be wrong anywhere without showing up in a block.

    The trailer of the fixture carries the two blocks __expf_sse2 digests differently, 0x420 and 0xc27, which hold
    the two arguments of B3; those are the blocks the unfused rounding step fails."""
    block, first, last, want = _recorded("expf_digests.txt", 4)
    assert len(want) == 4096, f"expected 4,096 block digests, read {len(want)}"
    # the fixture's own shape, so a duplicated or reordered row cannot stand in for the block it is missing, and
    # the block-to-bit-pattern mapping this case relies on is the one the fixture recorded
    assert np.array_equal(block, np.arange(4096, dtype=U32))
    starts = block.astype(np.uint64) << np.uint64(20)
    assert np.array_equal(first.astype(np.uint64), starts)
    assert np.array_equal(last.astype(np.uint64), starts + 0xFFFFF)
    bad = [(int(k), int(w)) for k, w in zip(block, want)
           if _nt.basalt_expf_digest(int(k) << 20, 1 << 20) != int(w)]
    assert not bad, (f"{len(bad)} of 4,096 blocks differ, first block {bad[0][0]:#x} "
                     f"({_nt.basalt_expf_digest(bad[0][0] << 20, 1 << 20):#018x} "
                     f"against the recorded {bad[0][1]:#018x})")


# ---- C: the specials ---------------------------------------------------------------------------------------------

def test_c1_zeros():
    """C1. Both zeros take the main path and give exactly 1."""
    assert list(bits_of(expf(f32([0x00000000, 0x80000000])))) == [0x3F800000, 0x3F800000]


def test_c2_infinities():
    """C2. +inf passes through `x + x`; -inf is the one early return expf.c makes by bit pattern, before the NaN
    test, and gives +0 rather than an underflow."""
    assert list(bits_of(expf(f32([0x7F800000, 0xFF800000])))) == [0x7F800000, 0x00000000]


def test_c3_nan():
    """C3. A NaN reaches `x + x`, which quiets a signalling NaN and leaves a quiet one alone, payload included.
    B4's blocks 0x7f8-0x7ff and 0xff8-0xfff are glibc's own output at all 16,777,214 NaN arguments, so the bits
    are pinned there; this case names the rule they follow."""
    got = bits_of(expf(f32([0x7FC00001, 0xFFC00001, 0x7F800001])))
    assert list(got) == [0x7FC00001, 0xFFC00001, 0x7FC00001]


def test_c4_overflow_boundary():
    """C4. The literal `0x1.62e42ep6f` is itself the largest argument with a finite result, and the next float up
    takes the overflow return. Both sides agree with the reference: above the literal the true value is past
    FLT_MAX, which rounds to +inf."""
    assert OFLOW == bits_of(F32(float.fromhex("0x1.62e42ep6")))
    assert bits_of(expf(f32([OFLOW])))[0] == 0x7F7FFF84          # finite, just under FLT_MAX
    assert bits_of(correctly_rounded(f32([OFLOW])))[0] == 0x7F7FFF84
    assert bits_of(expf(f32([OFLOW + 1])))[0] == 0x7F800000      # the overflow return
    assert np.isinf(correctly_rounded(f32([OFLOW + 1])))[0]
    assert bits_of(expf(f32([0x7F7FFFFF])))[0] == 0x7F800000     # the largest finite float


def test_c5_underflow_boundaries_and_the_band():
    """C5. The two underflow thresholds. `x < -0x1.9fe368p6f` returns (0x1p-95f)^2 = +0; the band above it,
    `x < -0x1.9d1d9ep6f`, returns (0x1.4p-75f)^2 = 0x00000001, which is WANT_ERRNO_UFLOW - a glibc choice Arm's own
    default build leaves off. Both agree with the correctly rounded value across the band, which is why the
    constant is allowed to stand in for the computation."""
    assert UFLOW == bits_of(F32(-float.fromhex("0x1.9fe368p6")))
    assert MAY_UFLOW == bits_of(F32(-float.fromhex("0x1.9d1d9ep6")))
    assert bits_of(expf(f32([UFLOW])))[0] == 0x00000001          # the comparison is strict: this one computes
    assert bits_of(expf(f32([UFLOW + 1])))[0] == 0x00000000      # and one float further down underflows
    band = f32(np.arange(MAY_UFLOW, UFLOW + 1, dtype=U32))       # the whole may_uflow band, 90,854 arguments
    assert (bits_of(expf(band)) == 0x00000001).all()
    assert (bits_of(expf(band)) == bits_of(correctly_rounded(band))).all()
    below = f32(np.arange(UFLOW + 1, UFLOW + 0x4000, dtype=U32))
    assert (bits_of(expf(below)) == 0).all() and (bits_of(correctly_rounded(below)) == 0).all()


def test_c6_subnormal_results():
    """C6. Every argument whose result is subnormal: from -87.3365478515625, the first whose result falls below
    2^-126, to the underflow threshold, the last with a result above zero - 2,180,453 of them, swept. The bound
    holds there too, where the ulp is 2^-149 and the double has 20-odd bits rounded away rather than one."""
    x = f32(np.arange(0xC2AE8000, UFLOW + 1, dtype=U32))
    out = bits_of(expf(x))
    assert (out > 0).all(), "nothing underflows to zero above the threshold"
    first = int(np.argmax(out < 0x00800000))
    assert bits_of(x)[first] == 0xC2AEAC50 and out[first] == 0x007FFFE6
    assert (out[first:] < 0x00800000).all(), "the results are monotone: subnormal from there down"
    assert ulp_error(x[first:]).max() <= 0.502


# ---- D: the SwiGLU -----------------------------------------------------------------------------------------------

def _swiglu_sample():
    rng = np.random.default_rng(1434)
    g = np.concatenate([
        rng.uniform(-40.0, 40.0, 1 << 18).astype(F32),
        rng.normal(0.0, 2.0, 1 << 18).astype(F32),
        f32([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, OFLOW, UFLOW, MAY_UFLOW]),
    ])
    u = np.concatenate([
        rng.normal(0.0, 1.0, 1 << 19).astype(F32),
        f32([0x3F800000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00002, 0x00000001, 0x7F7FFFFF, 0x00000000]),
    ])
    return g, u


def test_d1_swiglu_is_basalts_expression():
    """D1. `(g / (1.f + std::exp(-g))) * u` (pool.cpp:434): the add, then the divide, then the multiply, each in
    float32. The order is part of the result, so the check is bitwise against that expression and nothing looser.
    Where the expression creates a NaN the two sides hold different bits by construction, numpy writing the
    AArch64 default NaN and the kernel x86's indefinite; those rows are A2's and D3's, and are excluded here so
    that this case says one thing."""
    g, u = _swiglu_sample()
    with np.errstate(invalid="ignore"):                       # -inf / inf is a NaN on both sides, not an error
        want = (g / (F32(1.0) + expf(-g))) * u
    got = swiglu(g, u)
    created_nan = np.isnan(want) & ~np.isnan(g) & ~np.isnan(u)
    bad = np.flatnonzero((bits_of(got) != bits_of(want)) & ~created_nan)
    assert bad.size == 0, (f"{bad.size} of {g.size} differ, first at g={bits_of(g)[bad[0]]:#010x} "
                           f"u={bits_of(u)[bad[0]]:#010x}: {bits_of(got)[bad[0]]:#010x} vs "
                           f"{bits_of(want)[bad[0]]:#010x}")
    assert (bits_of(got)[created_nan] == 0xFFC00000).all()    # and those rows are x86's indefinite, every one


def test_d2_the_order_is_load_bearing():
    """D2. The witness for D1: the same algebra with the reciprocal taken first, which is what a port reaching for
    a sigmoid would write, differs on this sample. Without this, D1 would pass for a reordered kernel."""
    g, u = _swiglu_sample()
    with np.errstate(invalid="ignore"):
        reciprocal_first = (g * (F32(1.0) / (F32(1.0) + expf(-g)))) * u
    assert (bits_of(swiglu(g, u)) != bits_of(reciprocal_first)).any()


def test_d3_swiglu_specials():
    """D3. The specials basalt's expression itself produces, as bits. An operation that creates a NaN writes x86's
    floating-point indefinite 0xffc00000, where an AArch64 instruction writes 0x7fc00000; a NaN that arrives as an
    operand comes back quieted with its payload, from whichever source of the instruction is the first NaN, which
    is `g` at the divide and the quotient at the multiply. A2's recorded vectors carry every rule this case names,
    though not every row: the last two rows below are not among them, and the last is the one place this suite
    asserts against the manual rather than against a measurement."""
    g = f32([0x7F800000, 0xFF800000, 0x00000000, 0x80000000, 0x7F812345, 0x3F800000, 0x7F812345, 0x00000000,
             0x7FC12345])
    u = f32([0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x7F812345, 0x7FC54321, 0x7F800000,
             0x7FA54321])
    got = bits_of(swiglu(g, u))
    assert got[0] == 0x7F800000                                  # inf / (1 + 0) * 1
    assert got[1] == 0xFFC00000                                  # -inf / inf, an indefinite the divide creates
    assert got[2] == 0x00000000 and got[3] == 0x80000000         # +-0 keep their sign
    assert got[4] == 0x7FC12345                                  # a NaN g, quieted, through both instructions
    assert got[5] == 0x7FC12345                                  # a NaN u, the multiply's SRC1 being finite
    assert got[6] == 0x7FC12345                                  # both NaN: the multiply keeps the quotient's
    assert got[7] == 0xFFC00000                                  # 0 * inf, an indefinite the multiply creates
    # The one assertion with no measurement behind it [S, Intel SDM Vol. 1 4.8.3.7 against Arm ARM A1.5.3]: the
    # quotient is a quiet NaN and `u` a signalling one, so SSE returns the first source, 0x7fc12345, where an
    # AArch64 multiply would prefer the signalling second and give 0x7fe54321. No recorded row holds this pair.
    assert got[8] == 0x7FC12345


# ---- the binding's own checks ------------------------------------------------------------------------------------

def test_binding_refuses_mismatched_lengths_and_dtypes():
    x = np.zeros(8, F32)
    with pytest.raises(ValueError, match="as many floats"):
        _nt.basalt_expf(x, np.zeros(7, F32))
    with pytest.raises(ValueError, match="same number of floats"):
        _nt.basalt_swiglu(x, x, np.zeros(9, F32))
    with pytest.raises(TypeError):
        _nt.basalt_expf(np.zeros(8, np.float64), np.zeros(8, np.float64))
    with pytest.raises(TypeError):
        _nt.basalt_expf(np.zeros((2, 4), F32), np.zeros((2, 4), F32))
    with pytest.raises(TypeError):
        _nt.basalt_expf(np.zeros(16, F32)[::2], np.zeros(8, F32))          # not contiguous
    a = np.arange(9, dtype=F32)
    with pytest.raises(ValueError, match="partly overlap"):
        _nt.basalt_expf(a[:8], a[1:])                                      # a shifted view of the same buffer
    with pytest.raises(ValueError, match="partly overlap"):
        _nt.basalt_swiglu(a[:8], np.zeros(8, F32), a[1:])
    with pytest.raises(ValueError, match="partly overlap"):
        _nt.basalt_swiglu(np.zeros(8, F32), a[:8], a[1:])
    b = f32([0x3F000000] * 4)                                              # out being x itself is well defined
    _nt.basalt_expf(b, b)
    assert list(bits_of(b)) == [0x3FD3094C] * 4
    with pytest.raises(ValueError, match="past 2\\^32"):
        _nt.basalt_expf_digest(0xFFFFFFFF, 2)                              # a range that would wrap
    # the top of the range is accepted and digests 0xFFFFFFFF itself rather than wrapping to 0: that pattern is a
    # quiet NaN, which `x + x` returns unchanged, so the digest is one FNV step over its own bits
    assert _nt.basalt_expf_digest(0xFFFFFFFF, 1) == (0xCBF29CE484222325 ^ 0xFFFFFFFF) * 0x100000001B3 % (1 << 64)
    assert _nt.basalt_expf_digest(0, 0) == 0xCBF29CE484222325              # the empty digest is the seed
