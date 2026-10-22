# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/gr_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its write fixture (tests/basalt_hc_ref.py's gr_fixture) and its write checks
# (:620-692), their criteria verbatim.
"""basalt's hyper-connection write, ``native_gr_post``, on Metal (kernels/common/basalt_gr_post.metal). Basalt gates
none of its values; these are the contract's tests (docs/design/flash-next.md, the no-reference table).

- HG-EMU: bit for bit against the host emulation of the sequence nvcc compiles (``gr_post_emulation``,
  docs/research/basalt-nvcc/native_gr_postops.md section 1), its exponential the GPU's own ``expf_ftz``, at the real
  shape (n_embd 2560, hc 4) at T = 1 and 8 and at a ragged one (n_embd 100, T = 3: the last threadgroup passes the
  last row), with injections planted in every flush and saturation of the weight (one column of every token reads
  the weight out) and residual/block-output pairs whose sum is subnormal or cancels; at T = 8 a product the write's
  FFMA.FTZ rounds up to the smallest normal, which this GPU's fma flushes (fma_ftz's corner); each planted weight's
  exact value (1, +0 or 2) checked apart from the exponential; in basalt's in-place call and in the program's
  separate output, which give the same bytes.
- HG-PLAIN, HG-DELTA, HG-STREAM: gr_parity's three write checks (gr_parity.cpp:620-692) on its fixture (n_embd 256, T =
  1) and at the real shape (T = 8), into a separate output as gr_parity writes: a zero injection is the plain residual
  add, the rival reading ``1 * sigmoid`` is visible on the delta ``out - R``, and injections of sd 3 (gr_parity's) and
  2 (ple_win_parity's) give the double write's values, by gr_parity's term-relative rule.
- HG-WIN: the window rule in ple_win_parity's call shape (one token row, in place): each row of a T-row dispatch, T =
  1..8, has the bits of its token's one-row dispatch, and the reversed window gives every token the same bits.

Every test checks the sentinels (the output's rows T..7 keep their 0xFF bytes, the block output's and injections'
buffers their input bytes) and counts the output's non-finite values."""

import numpy as np
import pytest

from tests import basalt_hc_ref as ref
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32, U32 = np.float32, np.uint32
HC = 4
FILL = 0xFFFFFFFF                  # gpu.gr_post's 0xFF bytes: the rows past T, and the separate output


def write(R, bo, inj, T, n_embd, in_place=True):
    """``gpu.gr_post`` with the sentinels checked: the output's rows T..7 keep their 0xFF bytes, and the block
    output's and injections' buffers hold what was written (rows < T the inputs, the rest 0xFF). The 8 output rows."""
    out, bo_after, inj_after = gpu.gr_post(R, bo, inj, T, n_embd, HC, in_place=in_place)
    assert np.all(out[T:].view(U32) == FILL), "a store into the rows past the window"
    for name, a, after in (("block_out", bo, bo_after), ("inject", inj, inj_after)):
        want = np.full(after.shape, FILL, U32)
        want[:T] = np.ascontiguousarray(a, F32)[:T].view(U32)
        assert np.array_equal(after.view(U32), want), f"the kernel changed {name}"
    return out


# HG-EMU's injections with the weight each must give exactly (None: a NaN stream), ordered so that the ragged case's
# 12 slots hold every kind (native_gr_postops.md section 1): x = inj / 4 is +0 (+-0, +-subnormal: w = 1); 1 + e is
# above 2^126, its reciprocal flushed (-350, -352: w = +0); e overflows (-356, -1e30, -inf: w = +0); 1 + e rounds to
# 1 (70, 100: w = 2); e is flushed or 0 (351, 1e30, +inf: w = 2).
PLANTS = ((0.0, 1), (-350.0, 0), (70.0, 2), (np.nan, None), (-0.0, 1), (-356.0, 0), (351.0, 2), (np.inf, 2),
          (1e-40, 1), (-1e30, 0), (100.0, 2), (-np.inf, 0), (-1e-40, 1), (-352.0, 0), (1e30, 2))
# Token 0's stream 0 (injection +0, so w = 1 exactly): (block_out, residual, the stored bits) in columns 0..7.
PAIRS = ((1.5 * 2.0 ** -126, -(2.0 ** -126), 0x00000000),     # sum 2^-127, subnormal: flushed to +0
         (-1.5 * 2.0 ** -126, 2.0 ** -126, 0x80000000),       # -2^-127: flushed to -0
         (0.75, -0.75, 0x00000000),                           # cancels: +0
         (-0.0, -0.0, 0x80000000),                            # -0 + -0
         (0.0, -0.0, 0x00000000),                             # +0 + -0
         (2.0 ** -126, -1e-40, 0x00800000),                   # a subnormal residual is read as -0 (were it kept: +0)
         (-1e-40, 2.0 ** -126, 0x00800000),                   # so is a subnormal block output
         (2.0, None, 0x7FFFFFFF))                             # a NaN residual (0xffc00001): sm_120's NaN
NAN_R = 0xFFC00001
W_COL = len(PAIRS)                 # block output 1 and residual +0 in every stream: the output is the weight itself
CORNER = [W_COL + 1, W_COL + 2]    # block outputs +-bo, residual +0, bo * w in [2^-126 - 2^-150, 2^-126): +-2^-126


def planted(n_embd, T):
    """gr_fixture's residuals (stream scales 1, 4, 16, 64) and block outputs with uniform injections in [-100, 100],
    then PLANTS in the injections' first slots (at T = 1 its first four: +0, -350, 70, NaN), PAIRS in token 0's stream
    0, and column W_COL of every token reading out each stream's weight (``fma(1, w, +0) = w``: a weight of +0 and an
    unflushed reciprocal's 2^-125 store the same bits against a residual of order 1). Then the smallest normal's
    corner in the first stream of uniform injections whose weight w (below 1) has one: a bo with bo * w, exact in
    float64, in [2^-126 - 2^-150, 2^-126), which the write's FFMA.FTZ rounds up to 2^-126 where this GPU's fma
    flushes it to 0, as +-bo against a residual of +0 at columns CORNER. Also the planted slot count and the corner's
    stream (t, c), None when every slot holds a plant (T = 1, the ragged case)."""
    R, bo, _ = ref.gr_fixture(n_embd, HC, T)
    inj = np.random.default_rng(5).uniform(-100.0, 100.0, (T, HC)).astype(F32)
    k = min(len(PLANTS), T * HC)
    inj.reshape(-1)[:k] = [v for v, _ in PLANTS[:k]]
    for d, (b, r, _) in enumerate(PAIRS):
        bo[0, d] = b
        R[0, 0, d] = np.nan if r is None else r
    R.view(U32)[0, 0, len(PAIRS) - 1] = NAN_R
    bo[:, W_COL], R[:, :, W_COL] = 1.0, 0.0
    w = ref.gr_post_emulation(R, bo, inj, HC, gpu.expf_ftz)[:, :, W_COL].astype(np.float64)   # each stream's weight
    for t, c in (divmod(j, HC) for j in range(k, T * HC)):
        if not 0 < w[t, c] < 1:
            continue
        near = int(F32(2.0 ** -126 / w[t, c]).view(U32))
        for b in range(near - 2, near + 3):                       # the float32 patterns around 2^-126 / w
            if 2.0 ** -126 - 2.0 ** -150 <= float(U32(b).view(F32)) * w[t, c] < 2.0 ** -126:   # exact in float64
                bo.view(U32)[t, CORNER] = b, b | 0x80000000
                R[t, c, CORNER] = 0.0
                return R, bo, inj, k, (t, c)
    assert k == T * HC, "no stream of uniform injections reaches the smallest normal's corner"
    return R, bo, inj, k, None


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "separate_output"])
@pytest.mark.parametrize("n_embd, T", [(2560, 1), (2560, 8), (100, 3)])
def test_matches_emulation_bit_for_bit(n_embd, T, in_place):
    """HG-EMU. Every element of rows < T has the emulation's bits; each planted injection's stream has the bits of
    its exact weight (``fma(w, bo, R)``, ``.FTZ``, with no exponential; at column W_COL the weight's own bits); token
    0's stream 0 the PAIRS' bits; the corner's stream +-2^-126 at columns CORNER; the non-finite outputs are exactly
    the planted NaN stream and residual. The separate output equals basalt's in-place call byte for byte, all 8
    rows."""
    R, bo, inj, k, corner = planted(n_embd, T)
    out = write(R, bo, inj, T, n_embd, in_place)
    got = out[:T]
    want = ref.gr_post_emulation(R, bo, inj, HC, gpu.expf_ftz)
    bad = np.argwhere(got.view(U32) != want.view(U32))
    nonfinite = np.count_nonzero(~np.isfinite(got))
    planted_nan = n_embd + 1                                  # the NaN injection's stream, and the NaN residual
    print(f"HG-EMU n_embd {n_embd} T {T} {'in place' if in_place else 'separate output'}: {len(bad)} of {got.size} "
          f"elements differ; {nonfinite} non-finite ({planted_nan} planted)")
    if len(bad):
        t, c, d = bad[0]
        print(f"  first at (t, c, d) = ({t}, {c}, {d}): inject {inj[t, c]!r}, R {R[t, c, d]!r}, bo {bo[t, d]!r}: "
              f"got {got[t, c, d].view(U32):#010x}, want {want[t, c, d].view(U32):#010x}")
    assert len(bad) == 0, f"{len(bad)} elements differ from the emulation"
    for j in range(k):
        t, c = divmod(j, HC)
        w = PLANTS[j][1]
        exact = np.full(n_embd, 0x7FFFFFFF, U32) if w is None else \
            ref.cuda_nan(emu.fma_ftz(F32(w), bo[t], R[t, c])).view(U32)
        assert np.array_equal(got[t, c].view(U32), exact), f"inject {PLANTS[j][0]} does not give w = {w}"
    assert [int(v) for v in got[0, 0, : len(PAIRS)].view(U32)] == [p[2] for p in PAIRS]
    if corner:
        assert [int(v) for v in got[corner][CORNER].view(U32)] == [0x00800000, 0x80800000], \
            "FFMA.FTZ rounds bo * w + 0 in [2^-126 - 2^-150, 2^-126) up to the smallest normal"
    assert nonfinite == planted_nan, "a non-finite output nothing planted"
    if not in_place:
        assert np.array_equal(out.view(U32), write(R, bo, inj, T, n_embd, True).view(U32)), \
            "the separate output's bytes differ from basalt's in-place call"


def n_nonfinite(got):
    """The non-finite count of ``got``, output rows of a gr_fixture run: its inputs are finite, so any is a fault."""
    return np.count_nonzero(~np.isfinite(got))


@pytest.mark.parametrize("n_embd, T", [(256, 1), (2560, 8)], ids=["S1", "S2"])
def test_zero_injection_is_the_plain_residual_add(n_embd, T):
    """HG-PLAIN, gr_parity's zero-injection check (gr_parity.cpp:620-648): ``want = double(R) + double(bo)``, each
    element's error over its terms ``|R| + |bo|``; the worst below 1e-6 (numpy's max keeps a NaN, so a NaN fails)."""
    R, bo, inj = ref.gr_fixture(n_embd, HC, T, inj_sd=0.0)
    got = write(R, bo, inj, T, n_embd, in_place=False)[:T]
    want = R.astype(np.float64) + bo.astype(np.float64)[:, None, :]
    worst_plain = ref.rel_terms(want, got, R, bo[:, None, :]).max()
    print(f"HG-PLAIN n_embd {n_embd} T {T}: worst_plain {worst_plain:.3e}, {n_nonfinite(got)} non-finite")
    assert n_nonfinite(got) == 0, "a non-finite output"
    assert worst_plain < 1e-6, "const bool plain = worst_plain < 1e-6;"


def test_the_rival_gate_reading_is_visible():
    """HG-DELTA, gr_parity's rival reading (gr_parity.cpp:650-670) on HG-PLAIN's S1 run: on the delta ``out - R``,
    ``2 sigmoid(0) = 1`` (``want_delta = bo``) and ``1 sigmoid(0) = 0.5`` (``wrong_delta = bo * 0.5f``) are more than
    5 % apart, and the kernel's delta is within 5 % of the right one."""
    R, bo, inj = ref.gr_fixture(256, HC, 1, inj_sd=0.0)
    got = write(R, bo, inj, 1, 256, in_place=False)[:1]
    got_delta = (got.astype(np.float64) - R.astype(np.float64)).astype(F32)   # (float) ((double) got - (double) R)
    want_delta = np.broadcast_to(bo[:, None, :], R.shape)
    wrong_delta = want_delta * F32(0.5)
    rel, rel_got = ref.rel_diff(want_delta, wrong_delta), ref.rel_diff(want_delta, got_delta)
    print(f"HG-DELTA: readings {rel * 100:.0f}% apart, kernel within {rel_got:.2e}, {n_nonfinite(got)} non-finite")
    assert n_nonfinite(got) == 0, "a non-finite output"
    assert rel > 0.05 and rel_got < 0.05, "const bool visible = rel > 0.05 && rel_got < 0.05;"


@pytest.mark.parametrize("n_embd, T, inj_sd", [(256, 1, 3.0), (2560, 8, 3.0), (2560, 8, 2.0)],
                         ids=["S1-sd3", "S2-sd3", "S2-sd2"])
def test_injection_weights_against_double(n_embd, T, inj_sd):
    """HG-STREAM, gr_parity's nonzero-injection check (gr_parity.cpp:673-692): the double write (``gr_write_double``),
    each element's error over its terms ``|R| + |bo w|``; the worst below 1e-6. The injections stay far inside the
    rule's range, ``|inject| < 80`` (beyond it the float32 rounding of ``x log2 e`` that basalt shares dominates where
    ``|R|`` is small; HG-EMU holds the extremes by bits)."""
    R, bo, inj = ref.gr_fixture(n_embd, HC, T, inj_sd=inj_sd)
    got = write(R, bo, inj, T, n_embd, in_place=False)[:T]
    want, bo_w = ref.gr_write_double(R, bo, inj, HC)
    worst_stream = ref.rel_terms(want, got, R, bo_w).max()
    print(f"HG-STREAM n_embd {n_embd} T {T} inj_sd {inj_sd}: worst_stream {worst_stream:.3e} (max |inject| "
          f"{np.abs(inj).max():.2f}), {n_nonfinite(got)} non-finite")
    assert n_nonfinite(got) == 0, "a non-finite output"
    assert worst_stream < 1e-6, "const bool streams_ok = worst_stream < 1e-6;"


def test_rows_of_a_window_equal_one_row_dispatches():
    """HG-WIN, the window rule (invariant 1) in ple_win_parity's call shape, one token row in place: for T = 1..8, row t
    of the T-row dispatch has the bits of the one-row dispatch of token t alone (its rows copied to row 0 of fresh
    buffers), and the 8-row dispatch of the reversed window (token k at row 7 - k) gives every token the forward one's
    bits. Each dispatch's sentinels checked, and its output finite."""
    n_embd = 2560
    R, bo, inj = ref.gr_fixture(n_embd, HC, 8, inj_sd=3.0)

    def run(R, bo, inj, T):
        got = write(R, bo, inj, T, n_embd)[:T]
        assert n_nonfinite(got) == 0, f"a non-finite output in a {T}-row dispatch"
        return got.view(U32)

    alone = [run(R[t:t + 1], bo[t:t + 1], inj[t:t + 1], 1)[0] for t in range(8)]
    for T in range(1, 9):
        got = run(R, bo, inj, T)
        for t in range(T):
            assert np.array_equal(got[t], alone[t]), f"row {t} of the {T}-row window differs from its one-row dispatch"
    rev = run(R[::-1], bo[::-1], inj[::-1], 8)
    for k in range(8):                                  # got: the last window's, the forward 8-row dispatch
        assert np.array_equal(rev[7 - k], got[k]), f"token {k} at row {7 - k} of the reversed window differs"
