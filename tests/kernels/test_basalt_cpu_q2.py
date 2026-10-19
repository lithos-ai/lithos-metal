# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/expert.cpp:143-148, :336-370, :503-556 and
# src/kernels/cpu/expert_multi_test.cpp:56-72 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE):
# the two kernels' operation order and its per-token independence check.
"""basalt's two CPU Q2_0 kernels on ARM NEON (runtime/src/basalt_cpu.cpp), against the numpy emulation of the
AVX-512 semantics they transcribe (tests/kernels/basalt_cpu_emu.py) and, for the row dot, against an independent
float64 reference.

Comparison policy, the same everywhere: **all 32 bits**, NaN payloads included. Nothing is compared loosely. The
kernel reproduces x86's NaN bits as well as its numbers (runtime/src/basalt_cpu.cpp, THE NaN PATH) and the
reference models x86's rule at every site, so a NaN payload is as much a result as any other. Where a case pins a
value rather than a match - the all-zero chunk's ``+0.0``, the band's -127 codes, the untouched output rows - the
test asserts that value directly, because the kernel and the emulator share one reading of the specification and a
case that drifts into a harmless regime would otherwise still pass.

Group C is the answer to "both sides could be wrong together": golden constants recorded from basalt's own kernels
running on an AVX-512 machine (a Xeon Platinum 8488C, GCC 14.3, basalt's Release flags), which depend on neither
the port nor the numpy reference. They are cited as ``pr21-x86-ground-truth.md:<line>``, the PR #21 reviewer's
measurement at ``.context/basalt-port/pr21-ground-truth/`` beside basalt's checkout. C5's are that measurement's
per-site rows composed through its lane model ``mo::row``, which matched basalt's kernel there bit for bit.

Two values are compiler-dependent, and both are pinned here rather than hidden. ``amax`` when a chunk holds a NaN:
``_mm512_reduce_max_ps`` is header-defined, and GCC's first two levels pass the high half to MAXPS as SRC1 where
Clang's pass the low half - case A6 asserts GCC's at the positions where the two differ and prints Clang's beside
it. And a NaN ``res[t]`` of a row a NaN ENTERS: x86's own answer there moves with the token slot, the token count
and the GCC version, so the port pins GCC 14.3's one-token order for every slot, which case C3 asserts against the
recorded nt = 1 column.

Cases (the plan's section 7). A, ``act_quant_q8_1``: A1 the two caller widths, A2 an all-zero chunk, A3 +-0 and the
maximum planted in each of the 32 positions, A4 the scale band's four regimes, A5 +-inf, A6 the NaN sweep over both
reduction orders, A7 the half-away-from-zero rounding and the ``0.49999997f`` tie, A8 the saturation trap, A9 the
contraction trap, A10 the reciprocal's witness, A11 the non-finite chunks' exact outputs. B,
``q2_0_gguf_rows_multi``: B1 both pack shapes, B2 nt = 1..8 token independence, B3 the row ranges and the
absolute-index write, B4 stride padding, B5 the unread tail, B6 all 65,536 fp16 block scales, B7 adversarial
activations, B8 the independent float64 check, B9 determinism. C, the recorded x86 values: C1 the five end-to-end
fixtures, C2 the quantizer's own outputs, C3 the nine site probes, C4 the per-site NaN tables, C5 those tables' rows
through the kernel - signaling NaNs in scale and hx, the NaNs sites 1 and 2 create, the acc tree's last three
levels.

Fixtures are the repository's: ``random_rows`` (test_basalt_experts, basalt's own Q2_0 row fixture - random code
bytes with a finite fp16 scale per block) for the bitwise cases, and ``quantize_q2_0`` (test_basalt_iq_parity,
tools/gguf_writer.py's quantizer) for B8, which needs rows that mean something.

The suite bites, checked by mutating the kernel, with the same failures at -O0 and at the Release flags: the
saturating FCVTZS in place of the integer indefinite fails A4, A5, A6, A8, A11 and C2's c3, d1, d2 and e64;
Clang's operand order in ``reduce_max_ps`` fails A6, A11 and C2's d1 and d2; ``inv = 127.f / amax`` in place of
basalt's ``1.f / s`` fails A10 alone; a reversed code shift in both of the kernel's decodes (the vector unpack and
``x86_row``'s) fails B1 to B8; and the same reversed shift applied to the emulator as well - a misreading the kernel
and its reference share - fails B8 alone, which is what B8 is for. Group C catches the third kind, where both sides
move together: swapping the add tree's level-3 operands in the *reference* fails C3's P2 probe and C5's acc-tree
case, which no kernel-side mutation could have shown. And ``x86_row``'s NaN rules, mutated one site at a time - a
wrong priority, an SNaN-first rule, the hardware's own NaN, a missing quieting, the default NaN for a created one, a
swapped tree operand - fail a case wherever the mutant can change ``res``: a C case, except where the trees take the
default NaN or the hardware's own, which B6 and B7 catch. Those that survive cannot: dd, P, acc, corr and the
trees' sums are never SNaNs and D is never a NaN, a quieting skipped at one site is applied at the next, a NaN dd
makes ``A`` a NaN, which outranks ``B`` whatever site 2 picked, and the zero sign of a masked ``corr`` lane never
reaches ``res``, because ``A`` is never -0.0.

Run with ``-s`` for the A6 Clang-order table, the A8 pre-clamp margin and the B7 cancellation residue.
"""

import numpy as np
import pytest

from monolith.runtime import _native as _nt
from tests.contract import gguf_reference
from tests.kernels import basalt_cpu_emu as emu
from tests.kernels.basalt_emu import fma32
from tests.kernels.test_basalt_experts import random_rows
from tests.kernels.test_basalt_iq_parity import quantize_q2_0

F32 = np.float32
H, MAXC, QKA, QK, BB, MAXT = emu.H, emu.MAXC, emu.QKA, emu.QK, emu.BB, emu.MAXT
TWO_M128 = F32(2.0) ** -128                  # s <= this gives inv = +inf (docs/research/basalt-cpu-q2.md, 4)
TWO_M149 = F32(2.0) ** -149                  # the smallest positive float32


# ---- the comparison policy and the two kernels through their bindings --------------------------------------------

def recip(s):
    """``1.f / s`` in float32, overflowing to +inf quietly as basalt's step 6 does."""
    with np.errstate(divide="ignore", over="ignore"):
        return (F32(1.0) / F32(s)).astype(F32)


def same_bits(got, want, what):
    """All 32 bits of every element, NaN payloads included - the whole comparison, everywhere."""
    gb = np.ascontiguousarray(got, F32).reshape(-1).view(np.uint32)
    wb = np.ascontiguousarray(want, F32).reshape(-1).view(np.uint32)
    assert gb.shape == wb.shape, f"{what}: {gb.shape} against {wb.shape}"
    bad = np.flatnonzero(gb != wb)
    assert bad.size == 0, (f"{what}: bits differ at {bad[:8]} (got {[hex(v) for v in gb[bad[:8]]]}, "
                           f"want {[hex(v) for v in wb[bad[:8]]]})")


def kernel_act(x):
    """``act_quant_q8_1`` through the binding, trimmed to what it writes: (q[:n], scale, sum, hx, nchunks)."""
    x = np.ascontiguousarray(x, F32)
    q, scale = np.zeros(H, np.int8), np.zeros(MAXC, F32)
    total, hx = np.zeros(MAXC, np.int32), np.zeros(MAXC, F32)
    nchunks = _nt.basalt_act_quant_q8_1(x, q, scale, total, hx)
    return q[: x.size], scale[:nchunks], total[:nchunks], hx[:nchunks], nchunks


def check_act(x, order="gcc"):
    """kernel == emulator on every output. Returns the kernel's (q, scale, sum, hx, nchunks)."""
    got, want = kernel_act(x), emu.act_quant_q8_1(x, order)
    assert got[4] == want[4], f"nchunks {got[4]} != {want[4]}"
    assert np.array_equal(got[0], want[0]), f"codes differ at {np.flatnonzero(got[0] != want[0])[:8]}"
    assert np.array_equal(got[2], want[2]), f"sum differs at {np.flatnonzero(got[2] != want[2])[:8]}"
    same_bits(got[1], want[1], "scale")
    same_bits(got[3], want[3], "hx")
    return got


def act_arrays(xs):
    """``len(xs)`` tokens' quantized activations, in the (nt, H) / (nt, MAXC) arrays the row kernel's binding takes.
    Planting ``scale`` and ``hx`` by hand is what makes the adversarial cases of B7 reachable."""
    q, scale, hx = np.zeros((len(xs), H), np.int8), np.zeros((len(xs), MAXC), F32), np.zeros((len(xs), MAXC), F32)
    for t, x in enumerate(xs):
        qt, st, _, ht, nchunks = kernel_act(x)
        q[t, : qt.size], scale[t, :nchunks], hx[t, :nchunks] = qt, st, ht
    return q, scale, hx


def prefilled(ntok, nrows, fill=0xFF):
    """An output array pre-filled with ``fill`` bytes, as basalt's tests pre-fill theirs (basalt_gpu.py:21-28)."""
    return np.full((ntok, nrows, 4), fill, np.uint8).view(F32).reshape(ntok, nrows)


def kernel_rows(w, row_bytes, nblocks, act, nrows, r0, r1, fill=0xFF):
    """``q2_0_gguf_rows_multi`` through the binding, on a caller-owned pre-filled ``out``."""
    q, scale, hx = act
    out = prefilled(q.shape[0], nrows, fill)
    _nt.basalt_q2_0_gguf_rows_multi(np.ascontiguousarray(w, np.uint8), row_bytes, nblocks,
                                    np.ascontiguousarray(q, np.int8), np.ascontiguousarray(scale, F32),
                                    np.ascontiguousarray(hx, F32), out, r0, r1)
    return out


def check_rows(w, row_bytes, nblocks, act, nrows, r0=0, r1=None, what=""):
    """kernel == emulator over the whole output, the pre-filled rows included. Returns the kernel's ``out``."""
    r1 = nrows if r1 is None else r1
    got = kernel_rows(w, row_bytes, nblocks, act, nrows, r0, r1)
    want = emu.q2_0_gguf_rows_multi(w, row_bytes, nblocks, list(zip(*act)), prefilled(act[0].shape[0], nrows), r0, r1)
    same_bits(got, want, what or f"rows({row_bytes}, {nblocks}, {r0}:{r1})")
    return got


# ---- A. act_quant_q8_1 -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2560, 640])
def test_a1_normal_activations(n):
    """A1. N(0, 1) at the two widths the callers pass: the FFN input (2560) and the SwiGLU vector (640)."""
    x = np.random.default_rng(7).standard_normal(n).astype(F32)
    q, scale, total, hx, nchunks = check_act(x)
    assert nchunks == n // QKA and np.isfinite(scale).all() and np.abs(q).max() == 127


def test_a2_all_zero_chunk():
    """A2. An all-zero chunk: ``amax > 0`` is false, so s = 0, inv = 0, every code 0, sum 0 and hx exactly +0.0
    (the sign bit matters: a -0.0 would be a different float32)."""
    q, scale, total, hx, _ = check_act(np.zeros(QKA, F32))
    assert np.array_equal(q, np.zeros(QKA, np.int8)) and total[0] == 0
    assert scale.view(np.uint32)[0] == 0 and hx.view(np.uint32)[0] == 0, "scale and hx must be +0.0, not -0.0"


def test_a3_signed_zeros_and_every_amax_position():
    """A3. +-0 mixes (the sign clear of step 2 leaves +0.0, so s = 0 and every code 0), then the maximum planted in
    each of the 32 positions in turn: the fixed ``maxps(|x0|, |x1|)`` pairing of expert.cpp:352 feeds element j and
    element 16+j to one lane, and the four-level tree must reach every lane. With no NaN present both are
    order-independent (maximum is associative and commutative there), which is why A6 is the case that pins the
    operand order."""
    z = np.where(np.arange(QKA) % 3 == 0, F32(-0.0), F32(0.0)).astype(F32)
    q, scale, total, hx, _ = check_act(z)
    assert np.array_equal(q, np.zeros(QKA, np.int8)) and total[0] == 0
    assert scale.view(np.uint32)[0] == 0 and hx.view(np.uint32)[0] == 0

    base = (F32(0.001) * np.arange(1, QKA + 1)).astype(F32)          # distinct, so the maximum is unique
    for p in range(QKA):
        for sign in (F32(1.0), F32(-1.0)):
            x = base.copy()
            x[p] = sign * F32(9.5)
            _, scale, _, _, _ = check_act(x)
            assert scale[0] == F32(9.5) / F32(127.0), (p, sign, scale[0])


def test_a4_the_scale_band():
    """A4. The four regimes of ``s = amax / 127`` and ``inv = 1 / s``, at the boundaries the plan measured:

    (a) ``amax = 127 * 2^-128`` puts s exactly on 2^-128, where ``1.f / s`` is already +inf (the specification's
        the band's open form is off by this one value; basalt's plain ``1.f / s`` governs). Then
        ``0 * inf = NaN`` and ``x * inf = +-inf``, every one of which VCVTTPS2DQ turns into the integer indefinite:
        **every code -127, the zeros included**, sum -4064 and hx = s * (float) -4064.
    (b) one ulp of s above that, ``inv`` is the finite 3.402822e38 and the codes are ordinary - and subnormal
        arithmetic is live, so this case also fails loudly if FTZ is ever switched on.
    (c) a subnormal s well inside the finite-``inv`` band (``amax = 127 * 2^-127``).
    (d) ``amax = 2^-149`` makes ``fl(amax / 127) = 0``, a regime the specification does not name: the ``s > 0.f``
        guard gives inv = 0, so every code is 0 with scale +0.0 - while ``amax = 127 * 2^-149`` keeps s nonzero
        (2^-149) and therefore lands back in (a)'s band, every code -127."""
    x = np.zeros(QKA, F32)
    x[5], x[9], x[17] = F32(127.0) * TWO_M128, -F32(127.0) * TWO_M128 / F32(2.0), TWO_M128
    q, scale, total, hx, _ = check_act(x)                            # (a)
    assert scale[0] == TWO_M128 and recip(scale[0]) == F32(np.inf)
    assert np.array_equal(q, np.full(QKA, -127, np.int8)), "every code, zeros included, must be -127"
    assert total[0] == -4064 and hx[0] == (scale[0] * F32(-4064.0)).astype(F32)

    s_b = np.nextafter(TWO_M128, F32(1.0))                           # (b)
    x = np.zeros(QKA, F32)
    x[3], x[21] = F32(s_b * F32(127.0)), -F32(s_b * F32(127.0)) / F32(2.0)
    q, scale, _, _, _ = check_act(x)
    assert scale[0] == s_b and recip(scale[0]) == F32(3.402822e38) and np.isfinite(recip(scale[0]))
    assert q[3] == 127 and q[21] == -64 and q[0] == 0, "an ordinary chunk, zeros staying 0"

    x = np.zeros(QKA, F32)                                           # (c)
    x[11], x[30] = F32(127.0) * (F32(2.0) ** -127), -F32(2.0) ** -127
    q, scale, _, _, _ = check_act(x)
    assert scale[0] == F32(2.0) ** -127 and (scale.view(np.uint32)[0] & 0x7F800000) == 0, "s must be subnormal"
    assert np.isfinite(recip(scale[0])) and q[11] == 127 and q[30] == -1

    x = np.zeros(QKA, F32)                                           # (d)
    x[1] = TWO_M149
    q, scale, total, hx, _ = check_act(x)
    assert scale.view(np.uint32)[0] == 0 and np.array_equal(q, np.zeros(QKA, np.int8))
    assert total[0] == 0 and hx.view(np.uint32)[0] == 0

    x = np.zeros(QKA, F32)                                           # (d), the neighbour that is not this regime
    x[1] = F32(127.0) * TWO_M149
    q, scale, total, _, _ = check_act(x)
    assert scale[0] == TWO_M149 and np.array_equal(q, np.full(QKA, -127, np.int8)) and total[0] == -4064


def test_a5_infinite_elements():
    """A5. An infinite element makes ``amax = inf``, so s = inf and inv = 0: the infinite lanes compute
    ``inf * 0 = NaN`` and clamp to -127, while every finite lane gets t = +-0 and code 0 - including the -0.0
    element, whose ``t = -0.0`` still takes the ordered ``t >= 0`` branch and adds +0.5."""
    x = np.zeros(QKA, F32)
    x[0], x[4], x[6], x[8] = F32(np.inf), F32(-np.inf), F32(-0.0), F32(1.0)
    q, scale, total, _, _ = check_act(x)
    assert scale[0] == F32(np.inf) and q[0] == -127 and q[4] == -127
    assert q[6] == 0 and q[8] == 0 and total[0] == -254


def test_a6_nan_sweep_over_both_reduction_orders():
    """A6. A NaN swept over all 32 positions of a ramp peaking at element 31, checked bit for bit against the
    emulator in **GCC's** operand order, with Clang's value printed beside it.

    The divergence is not only NaN-ness. MAXPS returns SRC2 on an unordered compare, so the two orders swallow
    different lanes: at position 19 GCC's tree ends on the NaN (and ``amax > 0.f`` is then false, so s = 0 and every
    finite element's code is 0) where Clang's keeps 0.32; at 31 it is the other way round; at 23 both are finite and
    different (0.31 against 0.32). A flat chunk with the NaN at 16 and the true maximum at 4 shows the same thing
    without any NaN surviving: GCC's tree swallows the 9.0, Clang's keeps it. A test that planted a NaN "somewhere
    in the upper half" would pass under either order and prove nothing."""
    ramp = (F32(0.01) * np.arange(1, QKA + 1)).astype(F32)           # 0.01 .. 0.32, the maximum at element 31
    table = []
    for p in range(QKA):
        x = ramp.copy()
        x[p] = F32(np.nan)
        check_act(x, "gcc")                                          # the port's order, bit for bit
        g, c = (emu.act_quant_q8_1(x, o)[1][0] for o in ("gcc", "clang"))
        table.append((p, float(g), float(c)))
    print("\nA6 scale by NaN position (GCC's order is the port's):")
    print("   " + ", ".join(f"{p}: {g:.8g}/{c:.8g}" for p, g, c in table if g != c))

    s = {p: (g, c) for p, g, c in table}
    assert s[19] == (0.0, float(ramp[31] / F32(127.0))), "19: GCC ends on the NaN (s = 0), Clang keeps 0.32"
    assert s[31] == (float(ramp[30] / F32(127.0)), 0.0), "31: the other way round"
    assert s[23] == (float(ramp[30] / F32(127.0)), float(ramp[31] / F32(127.0))), "23: two different finite maxima"

    flat = np.full(QKA, F32(0.1), F32)
    flat[4], flat[16] = F32(9.0), F32(np.nan)
    check_act(flat, "gcc")
    g, c = (emu.act_quant_q8_1(flat, o)[1][0] for o in ("gcc", "clang"))
    print(f"   flat chunk, NaN at 16, maximum 9.0 at 4: GCC {float(g):.8g}, Clang {float(c):.8g}")
    assert g == F32(0.1) / F32(127.0) and c == F32(9.0) / F32(127.0), "GCC swallows the maximum, Clang keeps it"


def test_a7_rounding_half_away_from_zero_and_the_tie():
    """A7. ``r = t + copysign(0.5, t)`` truncated is lround's rule, round half AWAY from zero: with amax = 127 the
    scale is exactly 1 and every t is exact, so +-2.5 must give +-3. And the second rounding is part of the
    contract: ``t = 0.49999997f`` makes the exact sum ``1 - 2^-25``, a tie that rounds to the even 1.0f, so the code
    is 1 where ``lround(t)`` would give 0 (docs/research/basalt-cpu-q2.md, section 4)."""
    x = np.zeros(QKA, F32)
    x[0], x[1], x[2], x[3], x[4] = F32(127.0), F32(2.5), F32(-2.5), F32(0.5), F32(-0.5)
    q, scale, _, _, _ = check_act(x)
    assert scale[0] == F32(1.0)
    assert (q[1], q[2], q[3], q[4]) == (3, -3, 1, -1), "half away from zero, both signs"

    amax, xt = F32(np.e), F32(0.01070189569145441)                   # t = 0.49999997f exactly
    x = np.zeros(QKA, F32)
    x[0], x[7], x[9] = amax, xt, -xt
    q, scale, _, _, _ = check_act(x)
    inv = recip(scale[0])
    assert F32(xt * inv) == F32(0.49999997), f"t is {F32(xt * inv):.9g}, not the 0.49999997f this case needs"
    assert F32(F32(xt * inv) + F32(0.5)) == F32(1.0) and np.round(F32(xt * inv)) == 0
    assert (q[7], q[9]) == (1, -1), "the second rounding carries t over the integer; lround would give 0"


def test_a8_the_saturation_trap():
    """A8. The clamp's reachable side. VCVTTPS2DQ returns the integer indefinite 0x80000000 for a NaN, for +-inf and
    for an out-of-range value, which the clamp turns into **-127**; FCVTZS instead saturates (NaN -> 0,
    +inf -> INT32_MAX), which would give 0 and +127 here. Both NaN and +inf are reachable: a chunk in the band of
    A4(a) has inv = +inf, so its zeros compute ``0 * inf = NaN`` and its nonzeros ``+-inf``.

    The clamp's high side is not reachable at all: ``|x| <= amax`` and the rounding is monotone, so r is largest at
    x = amax, and over every positive finite float32 amax with a finite inv ``trunc(r)`` never exceeds 127 (the
    largest is 127.50003051757812, at amax = 3.7322113794423347e-37). This asserts that at the measured worst case
    and over a random sweep, so the +127 arm is shown inert rather than tested. A finite ``r >= 2^31`` - the third
    FCVTZS saturation case - is unreachable through this kernel for the same reason, and is checked on the
    reference's own VCVTTPS2DQ instead."""
    x = np.zeros(QKA, F32)
    x[2] = F32(127.0) * TWO_M128                                     # inv = +inf: zeros -> NaN, x[2] -> +inf
    x[3] = -x[2]                                                     # -> -inf
    q, _, _, _, _ = check_act(x)
    assert q[0] == -127 and q[2] == -127 and q[3] == -127, "NaN, +inf and -inf must all clamp to -127"

    def pre_clamp(chunk):
        """The chunk's steps 5-10 without basalt's clamp, in the reference's own arithmetic."""
        a = np.abs(np.asarray(chunk, F32))
        amax = emu.reduce_max_ps(emu.maxps(a[None, :16], a[None, 16:]))
        s = np.where(amax > 0, (amax / F32(127.0)).astype(F32), F32(0)).astype(F32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            inv = np.where(s > 0, (F32(1.0) / s).astype(F32), F32(0)).astype(F32)
            t = (np.asarray(chunk, F32) * inv).astype(F32)
            r = (t + np.where(t >= 0, F32(0.5), F32(-0.5)).astype(F32)).astype(F32)
        return r, emu.cvttps_epi32(r)

    rng = np.random.default_rng(3)
    worst, worst_amax = -np.inf, None
    sweep = [F32(3.7322113794423347e-37)] + [F32(v) for v in rng.standard_normal(64) * 10 ** rng.uniform(-30, 30, 64)]
    for amax in sweep:
        amax = F32(np.abs(amax))                                       # the sweep's draw is signed; amax is not
        if not (amax > 0) or not np.isfinite(recip(F32(amax / F32(127.0)))):
            continue
        chunk = (amax * np.linspace(-1.0, 1.0, QKA)).astype(F32)
        chunk[0], chunk[QKA - 1] = amax, -amax
        r, v = pre_clamp(chunk)
        assert v.max() <= 127, f"the +127 clamp fired at amax = {float(amax)!r}: trunc(r) = {v.max()}"
        check_act(chunk)
        if r.max() > worst:
            worst, worst_amax = float(r.max()), float(amax)
    print(f"\nA8 largest pre-clamp r over the sweep: {worst!r} at amax = {worst_amax!r} (the clamp's high side is "
          f"inert)")
    assert worst == 127.50003051757812, "the tight-band worst case must be in the sweep"

    big = np.asarray([2.0 ** 31, -(2.0 ** 31) - 2.0 ** 8, np.inf, -np.inf, np.nan], F32)   # not reachable through
    assert np.array_equal(emu.cvttps_epi32(big), np.full(5, emu.INDEF, np.int64))          # the kernel; see above
    assert np.array_equal(np.clip(emu.cvttps_epi32(big), -127, 127), np.full(5, -127, np.int64))


def test_a9_the_contraction_trap():
    """A9. Step 7's ``_mm512_mul_ps`` and step 9's ``_mm512_add_ps`` round twice, which GCC 12.2/13.2/14.3/15.2 and
    Clang 21 all emit as a separate vmulps and vaddps. A contracted ``fma(x, inv, +-0.5)`` skips the rounding of t
    and changes the code, so this input fails the moment the two statements are written as one expression. Measured,
    no contraction setting fuses them while they are two statements, because ``t`` also feeds the ``>= 0`` compare
    and has to be materialised either way; what ``-ffast-math`` does to this file instead is substitute FMAXNM for
    the nine ``maxps`` selects, which case A6 catches. The mutant that this case does catch is the one where the two
    statements are merged into one expression by hand, ``vfmaq_f32(+-0.5, x, inv)``, which no flag can undo."""
    amax, xt = F32(np.e), F32(0.01070189569145441)
    x = np.zeros(QKA, F32)
    x[0], x[6], x[11] = amax, xt, -xt
    q, scale, _, _, _ = check_act(x)
    inv = recip(scale[0])
    two = np.asarray([F32(F32(xt * inv) + F32(0.5)), F32(F32(-xt * inv) + F32(-0.5))], F32)
    fused = np.asarray([fma32(xt, inv, F32(0.5)), fma32(-xt, inv, F32(-0.5))], F32)
    assert not np.array_equal(np.trunc(two.astype(np.float64)), np.trunc(fused.astype(np.float64))), \
        f"this input no longer separates the two roundings from the FMA: {two} against {fused}"
    assert (q[6], q[11]) == (1, -1), f"contracted codes would be {np.trunc(fused.astype(np.float64)).astype(int)}"


def test_a10_the_reciprocal_is_one_over_s_not_127_over_amax():
    """A10. basalt divides twice - ``s = amax / 127.f`` then ``inv = 1.f / s`` (expert.cpp:353-354) - and the
    algebraically equal ``inv = 127.f / amax`` (the oracle quantizer's expression at :90) is a different float32,
    because ``s`` rounds first. Every other case in this suite survives that substitution: it moves ``inv`` by an
    ulp, and a code changes only where ``t = x * inv`` straddles a rounding boundary. This chunk straddles one."""
    amax, x = np.uint32([0x3FB8DD4C, 0x3EE5FD24]).view(F32)
    chunk = np.zeros(QKA, F32)
    chunk[0], chunk[5] = amax, x
    q, scale, _, _, _ = check_act(chunk)
    assert scale[0] == F32(amax / F32(127.0))
    mutant = F32(F32(127.0) / amax)
    assert mutant != recip(scale[0]), "this witness no longer separates the two reciprocals"
    assert int(np.trunc(np.float64(F32(F32(x * mutant) + F32(0.5))))) == 40, "the mutant's code is 40 here"
    assert q[5] == 39, "1.f / s gives 39; 127.f / amax gives 40"


def test_a11_a_non_finite_activation_writes_no_nan():
    """A11. The exact outputs for a chunk holding a single +inf, a single NaN, and +inf with -inf, as basalt's
    expert.cpp:352-367 computes them. A single +inf: amax = s = +inf and ``inv = 1.f / s = +0``, so the infinite
    lane's ``t = inf * 0`` is a NaN, which VCVTTPS2DQ converts to INT_MIN and the clamp to -127, while every finite
    lane's t is +-0 and its code 0: sum -127, scale +inf, ``hx = +inf * -127 = -inf``. A NaN lane gives -127 the same
    way wherever it sits. Where it also reaches amax - position 19 alone, in GCC's tree (A6) - the ordered
    ``amax > 0.f`` fails, so s = +0, every finite code is 0 and ``hx = +0 * -127 = -0.0``; elsewhere the tree
    swallows it and the chunk scales by its finite peak. +inf with -inf: -127 twice, sum -254, hx -inf. No case
    writes a NaN scale or hx. FCVTZS converts a NaN to 0, so a port on it would write q = 0, sum 0 and
    ``hx = +inf * 0 = NaN`` for the first chunk."""
    for p in range(QKA):                                             # a single +inf, at every position
        x = np.full(QKA, F32(0.25), F32)
        x[p] = F32(np.inf)
        q, scale, total, hx, _ = check_act(x)
        assert q[p] == -127 and np.count_nonzero(q) == 1 and total[0] == -127, f"+inf at {p}"
        assert scale.view(np.uint32)[0] == 0x7F800000 and hx.view(np.uint32)[0] == 0xFF800000, f"+inf at {p}"

    for p in range(QKA):                                             # a single NaN, at every position
        x = np.full(QKA, F32(0.1), F32)
        x[p] = F32(np.nan)
        q, scale, total, hx, _ = check_act(x)
        assert q[p] == -127, f"NaN at {p}: code {q[p]}"
        if p == 19:                                                  # amax is the NaN: s = +0
            assert np.count_nonzero(q) == 1 and total[0] == -127
            assert scale.view(np.uint32)[0] == 0x00000000 and hx.view(np.uint32)[0] == 0x80000000
        else:                                                        # swallowed: amax = 0.1, the other 31 codes 127
            assert np.count_nonzero(q == 127) == QKA - 1 and total[0] == 3810, f"NaN at {p}"
            assert scale.view(np.uint32)[0] == 0x3A4E69A0 and hx.view(np.uint32)[0] == 0x40400000, f"NaN at {p}"

    x = np.full(QKA, F32(0.25), F32)                                 # +inf and -inf together
    x[2], x[21] = F32(np.inf), F32(-np.inf)
    q, scale, total, hx, _ = check_act(x)
    assert q[2] == -127 and q[21] == -127 and np.count_nonzero(q) == 2 and total[0] == -254
    assert scale.view(np.uint32)[0] == 0x7F800000 and hx.view(np.uint32)[0] == 0xFF800000


# ---- B. q2_0_gguf_rows_multi -------------------------------------------------------------------------------------

SHAPES = [(2560, 40, 720), (640, 10, 180)]                           # (n, nblocks, row_bytes): gate/up, then down


@pytest.mark.parametrize("n, nblocks, row_bytes", SHAPES)
def test_b1_both_pack_shapes(n, nblocks, row_bytes):
    """B1. The two shapes this pack has (native_expert.cpp:49-53 at 2560/640): 40 blocks in 5 full groups of 8, and
    10 blocks in one full group and a ragged one whose mask keeps only ``corr`` lanes 0-3."""
    rng = np.random.default_rng(101)
    nrows = 17
    w = random_rows(rng, nrows, n)
    act = act_arrays([rng.standard_normal(n).astype(F32)])
    out = check_rows(w, row_bytes, nblocks, act, nrows, what=f"B1 {n}")
    assert np.isfinite(out).all()


def test_b2_every_token_equals_its_single_token_run():
    """B2. The kernel's own promise (pool.hpp:54-55, "Every token's output is bitwise the single-token job's"),
    in the shape of expert_multi_test.cpp:56-72: nt = 1..8 against each token's own nt = 1 run, bit for bit."""
    rng = np.random.default_rng(23)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 13
    w = random_rows(rng, nrows, n)
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32) * (1 + t) for t in range(MAXT)])
    single = [kernel_rows(w, row_bytes, nblocks, (q[t: t + 1], scale[t: t + 1], hx[t: t + 1]), nrows, 0, nrows)[0]
              for t in range(MAXT)]
    for ntok in range(1, MAXT + 1):
        got = check_rows(w, row_bytes, nblocks, (q[:ntok], scale[:ntok], hx[:ntok]), nrows, what=f"B2 nt={ntok}")
        for t in range(ntok):
            assert np.array_equal(got[t].view(np.uint32), single[t].view(np.uint32)), (ntok, t)


def test_b3_row_ranges_and_the_absolute_index():
    """B3. ``out[t]`` is indexed by the ABSOLUTE row r, and rows outside [r0, r1) are not touched - which is what
    lets the pool hand row slices of one buffer to different workers (expert.hpp:123-126, pool.cpp:418-431). The
    output is pre-filled with 0xFF bytes and checked raw, so an untouched row is 0xFFFFFFFF, not merely a NaN."""
    rng = np.random.default_rng(55)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 11
    w = random_rows(rng, nrows, n)
    act = act_arrays([rng.standard_normal(n).astype(F32), rng.standard_normal(n).astype(F32)])
    full = check_rows(w, row_bytes, nblocks, act, nrows, what="B3 full")
    for r0, r1 in ((0, nrows), (3, 7), (nrows - 1, nrows), (5, 5)):
        got = check_rows(w, row_bytes, nblocks, act, nrows, r0, r1, what=f"B3 {r0}:{r1}")
        inside = np.zeros(nrows, bool)
        inside[r0:r1] = True
        assert np.all(got.view(np.uint32)[:, ~inside] == 0xFFFFFFFF), f"{r0}:{r1} wrote outside its range"
        assert np.array_equal(got.view(np.uint32)[:, inside], full.view(np.uint32)[:, inside]), \
            f"{r0}:{r1} did not write out[t][r] at the absolute r"


def test_b4_stride_padding_is_never_read():
    """B4. ``row_bytes`` is a stride, not a length: with 64 bytes of garbage after each row's 10 blocks the result
    must equal the same blocks packed at ``row_bytes = 18 * nblocks``."""
    rng = np.random.default_rng(77)
    n, nblocks, nrows = 640, 10, 9
    tight = random_rows(rng, nrows, n).reshape(nrows, BB * nblocks)
    pad = 64
    padded = np.concatenate([tight, rng.integers(0, 256, (nrows, pad), dtype=np.uint8)], axis=1)
    act = act_arrays([rng.standard_normal(n).astype(F32)])
    a = check_rows(tight.reshape(-1), BB * nblocks, nblocks, act, nrows, what="B4 tight")
    b = check_rows(padded.reshape(-1), BB * nblocks + pad, nblocks, act, nrows, what="B4 padded")
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), "the stride padding changed the result"


def test_b5_the_unread_tail():
    """B5. The read set of section 3: ``q[0, 64 * nblocks)``, ``scale[0, 2 * nblocks)`` and ``hx[0, 2 * nblocks)``,
    and nothing else - never ``sum``, never ``nchunks``, and nothing past the width (``_mm512_maskz_loadu_ps`` does
    not touch its masked-off elements). At nblocks = 10 the tail is poisoned with NaNs in scale and hx and with
    alternating +-127 in q; the result must not move a bit."""
    rng = np.random.default_rng(91)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 7
    w = random_rows(rng, nrows, n)
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32)])
    base = check_rows(w, row_bytes, nblocks, (q, scale, hx), nrows, what="B5 clean")
    q[0, QK * nblocks:] = np.where(np.arange(H - QK * nblocks) % 2, 127, -127)
    scale[0, 2 * nblocks:] = F32(np.nan)
    hx[0, 2 * nblocks:] = F32(np.nan)
    poisoned = check_rows(w, row_bytes, nblocks, (q, scale, hx), nrows, what="B5 poisoned")
    assert np.array_equal(base.view(np.uint32), poisoned.view(np.uint32)), "the kernel read past its width"


def test_b6_every_fp16_block_scale():
    """B6. All 65,536 half bit patterns as a block's ``d`` against one fixed activation: +-0, every subnormal, 65504,
    +-inf and every NaN. ``_mm256_cvtph_ps`` and ``vcvt_f32_f16`` are both exact for every half, subnormals included
    (MXCSR.DAZ is ignored, and FPCR's FZ16 is off), and this case is what proves the NaN halves too: the reference
    converts them by VCVTPH2PS's own rule (payload << 13, quiet bit forced), the kernel by the instruction, and the
    comparison is bitwise, so all 2,046 NaN patterns are checked against x86's conversion rather than assumed."""
    rng = np.random.default_rng(13)
    nrows, nblocks, row_bytes = 1 << 16, 1, BB
    w = np.empty((nrows, row_bytes), np.uint8)
    w[:, :2] = np.arange(nrows, dtype=np.uint16).view(np.uint8).reshape(-1, 2)
    w[:, 2:] = rng.integers(0, 256, (nrows, 16), dtype=np.uint8)
    act = act_arrays([rng.standard_normal(QK).astype(F32)])
    out = check_rows(w.reshape(-1), row_bytes, nblocks, act, nrows, what="B6 halves")
    halves = np.frombuffer(np.arange(nrows, dtype=np.uint16).tobytes(), np.float16)
    assert np.isnan(out[0][np.isnan(halves)]).all(), "a NaN scale must give a NaN row"
    assert np.isfinite(out[0][: 0x7C00]).all(), "a finite scale must give a finite row"
    assert np.isnan(out[0][[0x7C00, 0xFC00]]).all(), "an infinite scale reaches inf * 0 = NaN on a zero code"
    assert out[0][[0, 0x8000]].view(np.uint32).tolist() == [0, 0], "+-0 as a scale leaves res at +0.0"


def test_b7_adversarial_activations():
    """B7. ``scale`` and ``hx`` are planted directly, which the binding allows and the quantizer cannot produce:
    subnormal, zero and huge values; a row whose ``P = RN(d * s)`` overflows to +-inf (and whose lanes then compute
    ``inf * 0 = NaN``); blocks carrying all four codes 0..3; and nblocks = 9, a ragged second group where the mask
    keeps only ``corr`` lanes 0-1.

    The cancelling case is the sharpest of them: with every code byte 0x55 every code is 1, so every weight
    dequantizes to ``(1 - 1) * d = 0`` and the exact result is zero - ``R(acc)`` and ``R(corr)`` are then the same
    sum reached through different lane structures (acc's lane L carries weights 4L..4L+3, corr's carries chunk
    16g+L), and what is left is pure rounding residue, where the tree shape and the FMA order are visible."""
    rng = np.random.default_rng(5)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 12
    w = random_rows(rng, nrows, n).reshape(nrows, row_bytes)
    for b in range(nblocks):                                         # all four codes 0..3 in every block
        w[0, b * BB + 2: (b + 1) * BB] = np.tile(np.asarray([0x1B, 0xE4, 0x4E, 0xB1], np.uint8), 4)
        w[1, b * BB + 2: (b + 1) * BB] = 0x1B
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32)])
    scale[0, 0], hx[0, 0] = TWO_M149, -TWO_M149                      # subnormal
    scale[0, 1], hx[0, 1] = F32(0.0), F32(-0.0)                      # zero, and a -0.0 hx
    scale[0, 2], hx[0, 2] = F32(3.0e38), F32(-3.0e38)                # P = d * s overflows for any sane d
    scale[0, 19], hx[0, 19] = F32(np.inf), F32(1.0)                  # the ragged group's last kept lane
    out = check_rows(w.reshape(-1), row_bytes, nblocks, (q, scale, hx), nrows, what="B7 planted")
    assert np.isnan(out).any() or np.isinf(out).any(), "the overflow regime must be reached"

    for nb in (9, 2, 1):                                             # ragged groups: 1, 2 and 8+1 blocks
        rb = BB * nb
        ww = random_rows(rng, nrows, QK * nb)
        check_rows(ww, rb, nb, act_arrays([rng.standard_normal(QK * nb).astype(F32)]), nrows, what=f"B7 nb={nb}")

    nblocks, row_bytes, n = 40, 720, 2560                            # the cancelling row set
    w = random_rows(rng, nrows, n).reshape(nrows, row_bytes)
    for b in range(nblocks):
        w[:, b * BB + 2: (b + 1) * BB] = 0x55                        # every code 1: (code - 1) * d = 0
    act = act_arrays([rng.standard_normal(n).astype(F32)])
    out = check_rows(w.reshape(-1), row_bytes, nblocks, act, nrows, what="B7 cancellation")
    print(f"\nB7 R(acc) - R(corr) where the exact sum is zero: max |res| = {np.abs(out).max():.3e}")
    assert np.abs(out).max() < 1e-5, "the correction no longer cancels the dot; check the lane mapping"


@pytest.mark.parametrize("n, nblocks, row_bytes", SHAPES)
def test_b8_against_a_float64_reference(n, nblocks, row_bytes):
    """B8. The independent check. The emulation and the kernel share one reading of section 3's unpack and lane
    mapping; this builds ``sum (code - 1) * d * q * s`` in float64 from llama.cpp's own numpy dequantizer
    (tests/contract/gguf_reference.dequantize, written apart from the C reference the format plugins follow) and
    compares against it, which is the only case that catches a misreading of either.

    The bound is on ``sum |terms|``, not on ``|ref|``: a row whose terms cancel has a large relative error for a
    correct kernel. A misread unpack or lane mapping is an O(1) error, so 1e-6 of the row's own magnitude - about
    twenty times the worst per-row observation (5.3e-08) and about two and a half times the aggregate (3.9e-07) -
    still catches it."""
    rng = np.random.default_rng(42)
    nrows = 48
    values = (rng.standard_normal((nrows, n)) * 0.02 * (0.5 + (np.arange(nrows)[:, None] % 7) / 7.0)).astype(F32)
    w = quantize_q2_0(values)                                        # real weights, not random bytes
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32)])
    got = check_rows(w, row_bytes, nblocks, (q, scale, hx), nrows, what=f"B8 {n}")[0].astype(np.float64)

    weights = gguf_reference.dequantize(42, w.reshape(nrows, row_bytes)).astype(np.float64)
    x = (q[0, :n].astype(np.float64).reshape(-1, QKA) * scale[0, : n // QKA, None].astype(np.float64)).reshape(-1)
    ref = weights @ x
    magnitude = np.abs(weights) @ np.abs(x)
    rel = np.abs(got - ref) / magnitude
    aggregate = np.abs(got - ref).sum() / np.abs(ref).sum()
    print(f"\nB8 n={n}: max |res - ref| / sum|terms| = {rel.max():.2e}, aggregate L1 = {aggregate:.2e}")
    assert rel.max() < 1e-6 and aggregate < 1e-6, (rel.max(), aggregate)


def test_b9_determinism():
    """B9. Repeated runs bit-identical, the numerics contract's last gate (CLAUDE.md)."""
    rng = np.random.default_rng(2026)
    n, nblocks, row_bytes, nrows = 2560, 40, 720, 9
    w = random_rows(rng, nrows, n)
    act = act_arrays([rng.standard_normal(n).astype(F32) for _ in range(3)])
    first = kernel_rows(w, row_bytes, nblocks, act, nrows, 0, nrows)
    for _ in range(4):
        again = kernel_rows(w, row_bytes, nblocks, act, nrows, 0, nrows)
        assert np.array_equal(first.view(np.uint32), again.view(np.uint32)), "the row dot is not deterministic"
    x = rng.standard_normal(n).astype(F32)
    one = kernel_act(x)
    for _ in range(4):
        other = kernel_act(x)
        assert np.array_equal(one[0], other[0]) and np.array_equal(one[2], other[2])
        assert np.array_equal(one[1].view(np.uint32), other[1].view(np.uint32))
        assert np.array_equal(one[3].view(np.uint32), other[3].view(np.uint32))


@pytest.mark.parametrize(
    "name, w_len, row_bytes, nblocks, nrows, message",
    [("nblocks wraps the int product", 1024, 603979776, 1 << 25, 1, "nblocks must be 1..40"),
     ("row_bytes * nrows wraps size_t", 18, 1 << 48, 1, 65536, "w is shorter than row_bytes")],
)
def test_b10_a_bound_that_would_overflow_still_refuses(name, w_len, row_bytes, nblocks, nrows, message):
    """The binding's bounds divide rather than multiply. Written as products, both of these pass the check meant to
    fail them - ``QK * nblocks`` reaches INT_MIN at nblocks = 2^25, and ``row_bytes * nrows`` wraps to 0 - and the
    kernel then reads past the ActQ arrays or past ``w``, which segfaults."""
    q = np.zeros((1, H), np.int8)
    scale = np.zeros((1, MAXC), np.float32)
    hx = np.zeros((1, MAXC), np.float32)
    out = np.zeros((1, nrows), np.float32)
    with pytest.raises(ValueError, match=message):
        _nt.basalt_q2_0_gguf_rows_multi(np.zeros(w_len, np.uint8), row_bytes, nblocks, q, scale, hx, out, 0, 1)


# ---- C. the recorded x86 ground truth -----------------------------------------------------------------------------
#
# Every value in C1-C4 was measured by running basalt's own kernels, or the instructions they emit, on an AVX-512
# machine (module docstring); C5's compose those measured rows through the reviewer's hardware-validated lane model.
# None of it comes from this port or from the numpy reference, which is the point of the group. Citations are lines
# of ``pr21-x86-ground-truth.md``.

GT = "pr21-x86-ground-truth.md"
PX, PY = 0x7FC000A1, 0x7FC000B1                  # the two planted payloads of the site probes (GT:307-309)
QA, QB, QC, SNAN = 0x7FC0000A, 0x7FC0000B, 0xFFC0000C, 0x7F80000D        # the per-site tables' operands (GT:120-123)
INDEF, INF, NINF = 0xFFC00000, 0x7F800000, 0xFF800000
ONE, FOUR, PZERO, NZERO = 0x3F800000, 0x40800000, 0x00000000, 0x80000000


def f32(bits):
    """The float32 holding these bits."""
    return np.uint32(bits).view(F32)


def u32(x):
    """The 32 bits of a one-element float32 result."""
    return int(np.ascontiguousarray(x, F32).reshape(-1).view(np.uint32)[0])


def packed_row(blocks):
    """One Q2_0 row from a (fp16 scale bits, code byte) pair per block, the weight shape the fixtures use."""
    w = np.zeros((len(blocks), BB), np.uint8)
    for b, (d, code) in enumerate(blocks):
        w[b, :2] = np.asarray([d], np.uint16).view(np.uint8)
        w[b, 2:] = code
    return w.reshape(-1)


def blank_act():
    """A full-width (q, scale, hx) to plant into, as the fixtures' ``memset`` leaves theirs."""
    return np.zeros(H, np.int8), np.zeros(MAXC, F32), np.zeros(MAXC, F32)


def quantized_act(x):
    """``act_quant_q8_1(x)`` as one token's full-width (q, scale, hx)."""
    q, scale, hx = act_arrays([x])
    return q[0], scale[0], hx[0]


def tiled(act, ntok):
    """One activation handed to ``ntok`` identical tokens, as every fixture and probe hands it."""
    return tuple(np.tile(v, (ntok, 1)) for v in act)


def fixture_a():
    return packed_row([(0x3C00, 0x55)]), 1, quantized_act(np.full(64, f32(0x7D800000), F32)), INDEF


def fixture_b():
    return packed_row([(0x7E01, 0x55), (0x7E02, 0x55)]), 2, quantized_act(np.ones(128, F32)), 0x7FC04000


def fixture_c1():
    q, scale, hx = blank_act()
    q[:128] = 1
    scale[:4] = [f32(0x7FC12345), F32(1.0), F32(np.inf), F32(1.0)]
    return packed_row([(0x3C00, 0x55), (0x3C00, 0x00)]), 2, (q, scale, hx), 0x7FC12345


def fixture_c2():
    q, scale, hx = blank_act()
    scale[:18] = F32(1.0)
    hx[0], hx[16] = f32(0x7FC54321), F32(-np.inf)
    return packed_row([(0x3C00, 0x55)] * 8 + [(0x0000, 0x55)]), 9, (q, scale, hx), 0x7FC54321


def fixture_c3():
    x = np.full(64, F32(0.25), F32)
    x[3] = F32(np.inf)
    return packed_row([(0x3C00, 0x1B)]), 1, quantized_act(x), INDEF


@pytest.mark.parametrize(
    "name, build, nts, cite",
    [("a, inf - inf in the final subtraction", fixture_a, range(1, MAXT + 1), "29"),
     ("b, two NaN block scales at the acc FMA", fixture_b, range(1, MAXT + 1), "40"),
     ("c1, a QNaN accumulator against inf * 0", fixture_c1, range(1, MAXT + 1), "53"),
     ("c2, a QNaN correction against 0 * inf", fixture_c2, range(1, MAXT + 1), "64"),
     ("c3, a quantized chunk holding +inf", fixture_c3, (1, 2), "75")],
)
def test_c1_the_end_to_end_fixtures(name, build, nts, cite):
    """C1. The five end-to-end fixtures, each against the 32 bits x86 wrote for it. (a) and (c3) are the
    generated-NaN case - every NaN in the row was made, not propagated, so x86's indefinite is the whole answer
    and all four GCC versions agree (GT:375-378). (b), (c1) and (c2) are the propagated case, where x86 ranks the
    FMA's operands the other way round from AArch64's FMLA: (b) keeps the second block's scale over the
    accumulator, (c1) and (c2) keep a QNaN addend that FMLA would discard for the default NaN.

    x86 gives the same bits at every (nt, t) for all five, so each is checked at every token count the ground
    truth records."""
    w, nblocks, act, golden = build()
    for ntok in nts:
        out = check_rows(w, BB * nblocks, nblocks, tiled(act, ntok), 1, what=f"C1 ({name}) nt={ntok}")
        got = out.view(np.uint32).reshape(-1)
        assert np.all(got == golden), \
            f"({name}) nt={ntok}: {[hex(int(v)) for v in got]} != {hex(golden)} ({GT}:{cite})"


def _nan_swallows(peak):
    """Fixture (d): a flat chunk whose true maximum at element 4 is swallowed by the NaN at 16."""
    x = np.full(QKA, F32(0.1), F32)
    x[4], x[16] = peak, f32(0x7FC00000)
    return x


def _subnormal_peak(k):
    """Fixture (e): a zero chunk with ``k * 2^-149`` at 0 and ``-2^-149`` at 1."""
    x = np.zeros(QKA, F32)
    x[0], x[1] = f32(k), f32(0x80000001)
    return x


# name, x, [(scale, inv, sum, hx, every code, the exceptions) per chunk], citation
QUANTIZER_GOLDENS = [
    ("a", np.full(64, f32(0x7D800000), F32),
     [(0x7A010204, 0x04FE0000, 4064, 0x7F800000, 127, {})] * 2, "24-25"),
    ("b", np.ones(128, F32), [(0x3C010204, 0x42FE0000, 4064, 0x42000000, 127, {})] * 4, "36"),
    ("c3", np.where(np.arange(64) == 3, F32(np.inf), F32(0.25)).astype(F32),
     [(0x7F800000, 0x00000000, -127, 0xFF800000, 0, {3: -127}),
      (0x3B010204, 0x43FE0000, 4064, 0x41000000, 127, {})], "69-71"),
    ("d1", _nan_swallows(F32(0.11)), [(0x3A4E69A0, 0x449EC000, 3810, 0x40400000, 127, {4: 127, 16: -127})], "82"),
    ("d2", _nan_swallows(F32(1e7)), [(0x3A4E69A0, 0x449EC000, 3556, 0x40333333, 127, {4: -127, 16: -127})], "83"),
    ("e63", _subnormal_peak(63), [(0x00000000, 0x00000000, 0, 0x00000000, 0, {})], "92"),
    ("e64", _subnormal_peak(64), [(0x00000001, 0x7F800000, -4064, 0x80000FE0, -127, {})], "93"),
]


@pytest.mark.parametrize("name, x, chunks, cite", QUANTIZER_GOLDENS)
def test_c2_the_quantizer_goldens(name, x, chunks, cite):
    """C2. ``act_quant_q8_1``'s own outputs, recorded from the x86 run: the fixtures' two scales, (c3)'s infinite
    chunk, (d)'s two NaN-swallowed maxima - the one input that reaches the clamp's +127 side (d1) and the one that
    reaches the integer indefinite from a finite element (d2) - and (e)'s 63 against 64 times 2^-149 threshold.
    ``inv`` is not an output; it is basalt's guarded ``s > 0.f ? 1.f / s : 0.f`` on the recorded scale, checked
    because the band's whole behaviour turns on it."""
    got = check_act(np.asarray(x, F32))
    q, scale, total, hx, nchunks = got
    assert nchunks == len(chunks), f"{name}: {nchunks} chunks, not {len(chunks)} ({GT}:{cite})"
    for k, (sc, inv, sm, hxb, default, exceptions) in enumerate(chunks):
        where = f"{name} chunk {k} ({GT}:{cite})"
        assert scale.view(np.uint32)[k] == sc, f"{where}: scale {hex(int(scale.view(np.uint32)[k]))} != {hex(sc)}"
        guarded = recip(scale[k]) if scale[k] > 0 else F32(0.0)      # basalt's `s > 0.f ? 1.f / s : 0.f` (:354)
        assert u32(guarded) == inv, f"{where}: inv is {hex(u32(guarded))}, not {hex(inv)}"
        assert total[k] == sm, f"{where}: sum {total[k]} != {sm}"
        assert hx.view(np.uint32)[k] == hxb, f"{where}: hx {hex(int(hx.view(np.uint32)[k]))} != {hex(hxb)}"
        want = np.full(QKA, default, np.int8)
        for pos, code in exceptions.items():
            want[pos] = code
        assert np.array_equal(q[k * QKA: (k + 1) * QKA], want), f"{where}: codes"


def probe(pat):
    """The nine site probes of ``probes_main.inc``: q = 1 everywhere, scale and hx zero, every block d = 1.0 with
    code bytes 0x55, and two distinct QNaN payloads planted where exactly one site decides between them."""
    q, scale, hx = np.ones(H, np.int8), np.zeros(MAXC, F32), np.zeros(MAXC, F32)
    nblocks, d0 = (8, 8, 8, 8, 1, 1, 9, 2, 1)[pat], 0x3C00
    plant = {0: ((hx, 0, PX), (hx, 8, PY)), 1: ((hx, 0, PX), (hx, 4, PY)), 2: ((hx, 0, PX), (hx, 2, PY)),
             3: ((hx, 0, PX), (hx, 1, PY)), 4: ((scale, 0, PX), (scale, 1, PY)), 5: ((scale, 0, PX), (hx, 0, PY)),
             6: ((hx, 0, PX), (hx, 16, PY)), 7: ((scale, 0, PX), (scale, 2, PY)),
             8: ((scale, 0, PY), (scale, 1, ONE))}[pat]
    for arr, idx, val in plant:
        arr[idx] = f32(val)
    if pat == 8:
        d0 = 0x7E05                                      # -> 0x7fc0a000 after vcvtph2ps, against a QNaN scale
    return packed_row([(d0, 0x55)] + [(0x3C00, 0x55)] * (nblocks - 1)), nblocks, (q, scale, hx)


@pytest.mark.parametrize(
    "pat, golden, cite, what",
    [(0, PY, "313", "add tree level 1, corr lanes 0 and 8: SRC1 is the high half"),
     (1, PY, "313", "add tree level 2, corr lanes 0 and 4: SRC1 is the high quarter"),
     (2, PY, "313", "add tree level 3, corr lanes 0 and 2: SRC1 is the shuffle, in every slot of every GCC (GT:342)"),
     (3, PX, "313", "add tree level 4, corr lanes 0 and 1: SRC1 is lane 0"),
     (4, PY, "313", "add tree level 1 on acc, lanes 0-7 against 8-15"),
     (5, PX, "351", "site 5, vsubss: the acc sum is SRC1 and outranks the corr sum"),
     (6, PY, "231", "site 2, vfmadd231ps corr, dd, hx: hx outranks the accumulated corr"),
     (7, PY, "251", "site 3, vfmadd231ps acc, P, D: P outranks the accumulator"),
     (8, 0x7FC0A000, "209", "site 1, a QNaN dd against a QNaN scale in chunk 0 only: the one-token tree takes "
                            "chunk 1's lanes, where dd meets 1.0, so C5 isolates site 1")],
)
def test_c3_the_site_probes(pat, golden, cite, what):
    """C3. Each probe plants two payloads where one site decides between them, and the golden is x86's answer for
    a ONE-token job (GT:313 for P0-P4; P5-P8 gave one answer in every slot, at the line each cites). That column is
    what the port pins for every token slot, so the same value is also asserted at nt = 2 - where x86 itself would
    answer differently for P0, P1, P3 and P4 (GT:314), because GCC reassigns the add tree's operands per token and
    basalt's own output there stops being the single-token job's (pool.hpp:54-55). Pinning is the deliberate
    divergence; the fixed value is checked, not assumed."""
    w, nblocks, act = probe(pat)
    for ntok in (1, 2):
        out = check_rows(w, BB * nblocks, nblocks, tiled(act, ntok), 1, what=f"C3 P{pat} nt={ntok}")
        got = out.view(np.uint32).reshape(-1)
        assert np.all(got == golden), \
            f"P{pat} ({what}) nt={ntok}: {[hex(int(v)) for v in got]} != {hex(golden)} ({GT}:{cite})"


SITE1 = [(PZERO, INF, INDEF), (INF, PZERO, INDEF), (NZERO, INF, INDEF), (0x7FC02000, ONE, 0x7FC02000),
         (ONE, QA, QA), (0x7FC02000, QA, 0x7FC02000), (0x7FC02000, QC, 0x7FC02000), (0x7FE02000, ONE, 0x7FE02000),
         (ONE, SNAN, 0x7FC0000D), (0x7FE02000, QA, 0x7FE02000), (0x7FC02000, SNAN, 0x7FC02000),
         (0x7FE02000, SNAN, 0x7FE02000), (0x7FC02000, INF, 0x7FC02000), (PZERO, QA, QA)]
SITE2 = [(PZERO, INF, ONE, INDEF), (INF, PZERO, ONE, INDEF), (ONE, INF, NINF, INDEF),
         (0x7FC02000, ONE, ONE, 0x7FC02000), (ONE, QA, ONE, QA), (ONE, ONE, QB, QB), (QA, QB, ONE, QA),
         (QA, ONE, QB, QA), (ONE, QA, QB, QA), (QA, QB, QC, QA), (PZERO, INF, QB, QB), (INF, PZERO, QB, QB),
         (0x7FE02000, ONE, ONE, 0x7FE02000), (ONE, SNAN, ONE, 0x7FC0000D), (ONE, SNAN, QB, 0x7FC0000D),
         (QA, SNAN, ONE, QA), (0x7FE02000, QA, ONE, 0x7FE02000), (0x7FE02000, ONE, QB, 0x7FE02000),
         (PZERO, SNAN, QB, 0x7FC0000D)]
SITE3 = [(INF, PZERO, ONE, INDEF), (INF, FOUR, NINF, INDEF), (QA, FOUR, ONE, QA), (ONE, FOUR, QB, QB),
         (QA, FOUR, QB, QA), (QC, FOUR, QB, QC), (INF, PZERO, QB, QB), (QA, PZERO, ONE, QA)]
SITE5 = [(INF, INF, INDEF), (QA, ONE, QA), (ONE, QB, QB), (QA, QB, QA), (QC, QB, QC), (QA, INF, QA)]
HALVES = [(0x7E01, 0x7FC02000), (0x7E02, 0x7FC04000), (0xFE03, 0xFFC06000), (0x7D01, 0x7FE02000),
          (0xFD05, 0xFFE0A000), (0x7C01, 0x7FC02000), (0x0001, 0x33800000)]


def test_c4_the_per_site_nan_tables():
    """C4. The reference's own rule at each site, against the rows the reviewer measured on the hardware and on
    basalt's emitted instruction forms: site 1 ``vmulps``, SRC1 = dd (GT:202-217); site 2
    ``vfmadd231ps corr, dd, hx``, priority dd, hx, corr (GT:221-241); site 3 ``vfmadd231ps acc, P, D``, priority
    P, D, acc (GT:245-254); site 5 ``vsubss``, SRC1 = A (GT:346-353); and VCVTPH2PS's NaN conversion (GT:142-146).
    An SNaN never outranks a QNaN in a later slot, and a QNaN addend beats an ``inf * 0`` product - the two rules
    AArch64 has backwards. The kernel meets the same rows end to end through C1 and C3."""
    for src1, src2, want in SITE1:
        got = u32(emu.x86_mul(f32(src1), f32(src2)))
        assert got == want, f"site 1 ({GT}:202-217): mul({hex(src1)}, {hex(src2)}) = {hex(got)} != {hex(want)}"
    for dd, hx, corr, want in SITE2:
        got = u32(emu.x86_fma(f32(dd), f32(hx), f32(corr)))
        assert got == want, \
            f"site 2 ({GT}:221-241): fma({hex(dd)}, {hex(hx)}, {hex(corr)}) = {hex(got)} != {hex(want)}"
    for p, d, acc, want in SITE3:
        got = u32(emu.x86_fma(f32(p), f32(d), f32(acc)))
        assert got == want, f"site 3 ({GT}:245-254): fma({hex(p)}, {hex(d)}, {hex(acc)}) = {hex(got)} != {hex(want)}"
    for a, b, want in SITE5:
        got = u32(emu.x86_sub(f32(a), f32(b)))
        assert got == want, f"site 5 ({GT}:346-353): sub({hex(a)}, {hex(b)}) = {hex(got)} != {hex(want)}"
    for half, want in HALVES:
        got = u32(emu.h2f(np.uint16(half)))
        assert got == want, f"vcvtph2ps ({GT}:142-146): h2f({hex(half)}) = {hex(got)} != {hex(want)}"


SNAN2 = 0xFF80000E                               # a second SNaN, sign set: site 5's B below


def planted(nblocks, plants, d0, zero_byte):
    """The probes' shape: q = 1 everywhere, scale and hx +0.0 but for the plants, block 0's fp16 d = ``d0`` and every
    other block's 1.0, code bytes 0x55 - except block 0's byte ``zero_byte``, whose four codes are 0, which leaves
    acc lane ``zero_byte`` alone with a zero dot."""
    q, scale, hx = np.ones(H, np.int8), np.zeros(MAXC, F32), np.zeros(MAXC, F32)
    for arr, idx, bits in plants:
        {"scale": scale, "hx": hx}[arr][idx] = f32(bits)
    w = packed_row([(d0, 0x55)] + [(0x3C00, 0x55)] * (nblocks - 1))
    if zero_byte is not None:
        w[2 + zero_byte] = 0
    return w, nblocks, (q, scale, hx)


# id, nblocks, block 0's d, plants, block 0's zero code byte, golden, the rows that decide it
MODEL_GOLDENS = [
    ("s1-snan-scale", 1, 0x3C00, (("scale", 0, SNAN), ("scale", 1, SNAN)), None, 0x7FC0000D, "212"),
    ("s1-qnan-dd-over-snan-scale", 1, 0x7E01, (("scale", 0, SNAN), ("scale", 1, SNAN)), None, 0x7FC02000, "142, 214"),
    ("s1-snan-half-over-snan-scale", 1, 0x7D01, (("scale", 0, SNAN), ("scale", 1, SNAN)), None, 0x7FE02000,
     "143, 215"),
    ("s1-zero-times-inf", 1, 0x0000, (("scale", 0, INF), ("scale", 1, INF)), None, INDEF, "204"),
    ("s2-snan-hx", 1, 0x3C00, (("hx", 0, SNAN),), None, 0x7FC0000D, "236"),
    ("s2-snan-hx-over-qnan-corr", 9, 0x3C00, (("hx", 0, PX), ("hx", 16, SNAN)), None, 0x7FC0000D, "237"),
    ("s2-zero-times-inf", 1, 0x0000, (("hx", 0, INF),), None, INDEF, "223"),
    ("s3-quieted-snan-p-over-qnan-acc", 2, 0x3C00, (("scale", 0, PX), ("scale", 2, SNAN)), None, 0x7FC0000D,
     "212, 251"),
    ("s4-level-1-high-half", 8, 0x3C00, (("hx", 0, SNAN), ("hx", 8, PY)), None, PY, "236, 270, 307-313"),
    ("s4-acc-levels-2-to-4", 1, 0x3C00, (("scale", 0, PX), ("scale", 1, INF)), 14, INDEF, "247, 270"),
    ("s5-quieted-snans-a-over-b", 1, 0x3C00, (("scale", 0, SNAN), ("hx", 0, SNAN2)), None, 0x7FC0000D, "212, 351"),
]


@pytest.mark.parametrize("name, nblocks, d0, plants, zero_byte, golden, cite", MODEL_GOLDENS,
                         ids=[case[0] for case in MODEL_GOLDENS])
def test_c5_the_site_rows_through_the_kernel(name, nblocks, d0, plants, zero_byte, golden, cite):
    """C5. C4's rows through the kernel, where no recorded fixture or probe takes them: a signaling NaN in
    ``scale`` or ``hx``, alone (its quieting) and against a QNaN at each site it meets; the indefinite that sites 1
    and 2 create from ``+0 * +inf``; and the acc tree's levels 2-4, which the probes reach on the corr tree only
    (lane 14's ``+inf * 0`` is the one indefinite among X's, so a swapped operand at level 2, 3 or 4 gives X).

    These goldens were not recorded end to end. Each composes measured rows (cited) through ``mo::row``, the
    reviewer's lane model, which matched basalt's kernel bit for bit on the AVX-512 host over 62,374 outputs
    (GT:355-371), with GCC 14.3's one-token tree order - asserted at every nt, because the port pins that order for
    every token slot (C3).

    Each case fails a mutant of ``x86_row`` that picks the wrong NaN at its site, and the mutants the rest of the
    suite let through fail here: an SNaN-first or SRC2-first ``vmulps`` and the hardware's own FMUL (the site 1 dd
    cases), a missing quieting (the lone SNaNs), the default NaN 0x7fc00000 for a created one (the ``+0 * +inf``
    cases), and a swapped operand at the acc tree's levels 2-4."""
    w, nblocks, act = planted(nblocks, plants, d0, zero_byte)
    for ntok in range(1, MAXT + 1):
        out = check_rows(w, BB * nblocks, nblocks, tiled(act, ntok), 1, what=f"C5 {name} nt={ntok}")
        got = out.view(np.uint32).reshape(-1)
        assert np.all(got == golden), \
            f"{name} nt={ntok}: {[hex(int(v)) for v in got]} != {hex(golden)} ({GT}:{cite} through mo::row)"
