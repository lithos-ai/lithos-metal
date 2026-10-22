# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/route_window_parity.cpp and src/kernels/ple_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): route_window_parity's projection half (its
# shapes, widths, data and criterion) and ple_parity's native PLE value check (its witness, float64 reference, rel_l1,
# le and gt, and criterion).
"""basalt's BF16 GEMVs on fp32 activations on Metal (kernels/common/basalt_mmvf.metal): bf16_gemv_fp32_mmvf (the
one-row kernel), bf16_gemv_fp32_mmvf_multi and bf16_gemv_fp32_mmvf_cols.

- route_window_parity, its projection half (basalt src/kernels/route_window_parity.cpp:25-50): _cols at T = 2, 3, 4
  and 6 against T one-row calls at 2560 to 256 and 2560 to 1, weights bf16(N(0, 1) * 0.05) and N(0, 1) activations
  (:29-35), criterion quoted: ``const bool same = std::memcmp(y1.data(), yn.data(), y1.size() * 4) == 0;`` (:48). The
  contract's shapes for this path, 2560 to 128 and 2560 to 512 (docs/design/flash-next.md, "Per-kernel parity
  tests"), run the same check. basalt draws from std::mt19937(7) through std::normal_distribution<float>, whose stream
  is library-specific; the port draws the same distributions from numpy. The outputs start apart (0xFE one-row, 0xFF
  _cols) where basalt's cudaMalloc leaves them unset, so an output neither call writes fails.
- ple_parity's native PLE value check (src/kernels/ple_parity.cpp:785-818) as a standalone test, for the one-row kernel
  and the multi kernel (T = 2 to 8) at 2560 to 2560: the witness of halfway BF16 values (:789) against the float64
  sum over the weights' bits (:790-799), criterion quoted:
  ``const bool correct = le(rel, 2e-6) && nf == 0 && gt(separation, 1e-5);`` (:815). basalt's weights are the PLE
  fixture's ple_value, and the test runs only with PLE_FIXTURE_DIR; here it runs on two: [synthetic],
  route_window_parity's bf16(N(0, 1) * 0.05), and [real], the pack's dense blk.1.ple_value.weight
  (tests/basalt_ple_ref.py's ``real()``, from the ranges under $BASALT_PLE_FIXTURE: unset, the case skips with a reason
  naming the variable; set, a file missing or of the wrong size fails). The separation's comparand, basalt's legacy
  path (the activations rounded to BF16, then bf16_gemv; ple.cu:315-317), is the float64 sum over the BF16-rounded
  witness. Each multi-kernel row takes the witness shifted by its row index and meets the same criterion.
- Column invariance, bitwise: each output is bit-identical to a one-row launch of its own (native_bf16.cu:75-77):
  the multi kernel at T = 1 to 8 (NT 4 and 8), _cols at T = 9 and 17 (a chunk of one runs the one-row kernel), the
  multi kernel for one row with ldy < n_out (native_bf16.cu:141), and strided ldx and ldy, whose gaps stay unread and
  unwritten, at block sizes 32, 96 and 256.
- Bit for bit against the numpy emulation of docs/research/basalt-nvcc/bf16.md (tests/kernels/basalt_emu.py) at
  block sizes 32, 96, 192 and 256, with pair counts that leave threads idle, and with planted subnormal weights,
  subnormal activations, subnormal products and a subnormal block sum, each of which .FTZ turns into +0, and exact
  FMA results just below 2^-126, which FFMA.FTZ rounds up to 2^-126 before it flushes."""

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_ple_ref import real
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32 = np.float32


def bf16_weights(rng, n_out, n_in):
    """route_window_parity.cpp:34: bf16(nd(rng) * 0.05f), the float32 product rounded to BF16 (nearest even)."""
    return emu.bf16_bits(rng.standard_normal((n_out, n_in)).astype(F32) * F32(0.05))


def weight_values(w):
    return (np.asarray(w, np.uint16).astype(np.uint32) << 16).view(F32)


def bits(a):
    return np.ascontiguousarray(a, F32).view(np.uint32)


def one_row(w, x, n_in, n_out):
    return np.stack([gpu.mmvf(w, row, n_in, n_out, fill=0xFE) for row in x])


@pytest.mark.parametrize("n_out", [256, 1, 128, 512])
def test_route_window_parity(n_out):
    n = 2560
    rng = np.random.default_rng(7 + n_out)
    for t in (2, 3, 4, 6):
        w = bf16_weights(rng, n_out, n)
        x = rng.standard_normal((t, n)).astype(F32)
        y1 = one_row(w, x, n, n_out)
        yn = gpu.mmvf_cols(w, x, n, n_out, t, fill=0xFF)
        assert np.isfinite(y1).all() and np.isfinite(yn).all(), (n_out, t)
        # basalt: ``const bool same = std::memcmp(y1.data(), yn.data(), y1.size() * 4) == 0;`` (:48)
        assert y1.tobytes() == yn.tobytes(), (n_out, t)


def rel_l1(a, b):
    """ple_parity.cpp:64-78: sum |a - b| / sum |a| (floored at 1e-30) in float64 over the finite pairs, NaN if any
    pair has a non-finite entry; and the count of those."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    nf = int(np.count_nonzero(~ok))
    if nf:
        return np.nan, nf
    m = np.abs(a).sum()
    return np.abs(a - b).sum() / (m if m > 1e-30 else 1e-30), 0


def le(x, t):
    return bool(np.isfinite(x) and x <= t)       # ple_parity.cpp:48


def gt(x, t):
    return bool(np.isfinite(x) and x > t)        # ple_parity.cpp:49


@pytest.mark.parametrize("name", ["synthetic", "real"])
def test_ple_value_float64(name):
    nd = 2560
    rng = np.random.default_rng(790)
    w = bf16_weights(rng, nd, nd) if name == "synthetic" else real().value
    wv = weight_values(w).astype(np.float64)
    i = np.arange(nd)
    # :789 ``witness[i] = i % 7 == 0 ? 1.00390625f : -0.501953125f;`` (halfway between two BF16 values)
    witness = np.where(i % 7 == 0, F32(1.00390625), F32(-0.501953125)).astype(F32)

    def check(native, x, row):
        reference = (wv @ x.astype(np.float64)).astype(F32)                    # :790-799, float(sum)
        legacy = (wv @ weight_values(emu.bf16_bits(x)).astype(np.float64)).astype(F32)
        rel, nf = rel_l1(reference, native)
        separation, _ = rel_l1(native, legacy)
        assert le(rel, 2e-6) and nf == 0 and gt(separation, 1e-5), (
            "`const bool correct = le(rel, 2e-6) && nf == 0 && gt(separation, 1e-5);` (ple_parity.cpp:815), "
            f"{name} weights, {row}: rel {rel!r}, non-finite {nf}, separation {separation!r}")
        return rel

    rels = [check(gpu.mmvf(w, witness, nd, nd, fill=0xFE), witness, "the one-row kernel")]
    for t in range(2, 9):
        x = np.stack([np.roll(witness, k) for k in range(t)])
        y = gpu.mmvf(w, x, nd, nd, t, fill=0xFF).reshape(t, nd)
        rels += [check(y[k], x[k], f"the multi kernel at T = {t}, row {k}") for k in range(t)]
    print(f"ple value rel_l1 [{name}]: max {max(rels):.3e}")


@pytest.mark.parametrize("n_in", [2560, 64, 130], ids=["block256", "block32", "block96"])
def test_column_invariance(n_in):
    n_out = 37
    rng = np.random.default_rng(n_in)
    w = bf16_weights(rng, n_out, n_in)
    x = rng.standard_normal((17, n_in)).astype(F32)
    single = one_row(w, x, n_in, n_out)
    # every output bit-identical to a one-row launch of its own (native_bf16.cu:75-77); T = 1 is the one-row kernel
    for t in range(1, 9):
        multi = gpu.mmvf(w, x[:t], n_in, n_out, t, fill=0xFF).reshape(t, n_out)
        assert np.array_equal(bits(multi), bits(single[:t])), t
    for t in (9, 17):                                               # chunks 8 + 1 and 8 + 8 + 1
        cols = gpu.mmvf_cols(w, x[:t], n_in, n_out, t, fill=0xFF)
        assert np.array_equal(bits(cols), bits(single[:t])), t
    # one row with ldy < n_out takes the multi kernel (native_bf16.cu:141)
    assert kernels.basalt_mmvf_macros(n_in, n_out, 1, ldy=n_out - 1)["NT"] == "4"
    one = gpu.mmvf(w, x[0], n_in, n_out, 1, ldy=n_out - 1, fill=0xFF)
    assert np.array_equal(bits(one), bits(single[0]))
    # strided rows: NaN in the activation gaps, the output gaps keep their fill
    for t in (3, 6):
        ldx, ldy = n_in + 6, n_out + 5
        xs = np.full((t, ldx), np.nan, F32)
        xs[:, :n_in] = x[:t]
        out = gpu.mmvf(w, xs.reshape(-1)[: (t - 1) * ldx + n_in], n_in, n_out, t, ldx, ldy, fill=0xFF)
        at = (np.arange(t)[:, None] * ldy + np.arange(n_out)).reshape(-1)
        assert np.array_equal(bits(out[at]), bits(single[:t]).reshape(-1)), t
        gap = np.setdiff1d(np.arange(out.size), at)
        assert (bits(out[gap]) == 0xFFFFFFFF).all(), t


EMU_CASES = [(2, 5, 1), (64, 33, 3), (130, 17, 5), (700, 9, 2), (1000, 7, 8), (2560, 128, 1), (2560, 512, 4),
             (2560, 1, 8), (2560, 2560, 1)]


@pytest.mark.parametrize("case", EMU_CASES, ids=[f"{a}x{b}x{c}" for a, b, c in EMU_CASES])
def test_matches_emulation(case):
    n_in, n_out, t = case
    rng = np.random.default_rng(n_in * 31 + n_out)
    w = bf16_weights(rng, n_out, n_in)
    x = rng.standard_normal((t, n_in)).astype(F32)
    got = gpu.mmvf(w, x, n_in, n_out, t, fill=0xFF).reshape(t, n_out)
    want = emu.mmvf(w, x, kernels.basalt_mmvf_block(n_in))
    assert np.array_equal(bits(got), bits(want))


def p2(e):
    return np.ldexp(F32(1), e)                   # 2^e, exact (a subnormal below 2^-126)


def test_flush_to_zero():
    """Each .FTZ site of bf16.md planted so that the flush gives +0 where IEEE float32 arithmetic (the emulation
    without flushes) gives a nonzero result: a subnormal product (row A times x0, output (0, 0)), a subnormal weight
    (row B times x2, (2, 1)), a subnormal activation (row C times x1, (1, 2)) and a block sum whose last add is
    subnormal (row D times x2, (2, 3): 2^-125 from warp 0 plus -1.75 * 2^-126 from warp 1). This GPU's arithmetic
    flushes these itself (docs/research/basalt-nvcc/matvec.md, Notation), so the kernel would pass them without its
    explicit flushes too. Then the other side of the flush: FFMA.FTZ rounds before it flushes (PTX ISA, fma.f32), so an
    exact value in [2^-126 - 2^-150, 2^-126) gives 2^-126, where this GPU's fma gives +0 and fma_ftz rebuilds it: R1
    (2^-126 - 2^-151) at the first FMA of a pair, from +0; the second FMA from a normal acc (2^-126 - 2^-25 * 2^-126)
    and its negative; and the tie 2^-126 - 2^-150, to 2^-126 (the even neighbour)."""
    n_in, n_out = 2560, 8
    rng = np.random.default_rng(3)
    x = np.stack([np.full(n_in, p2(-30), F32), np.full(n_in, p2(-140), F32),
                  rng.standard_normal(n_in).astype(F32)])
    x[2, [0, 64]] = p2(-65)
    w = bf16_weights(rng, n_out, n_in)
    w[4:, ::97], w[4:, 1::97] = 0x0003, 0x8005                          # subnormal weights among normal ones
    w[0], w[1], w[2], w[3] = 0x0D80, 0x0040, 0x5380, 0                  # A 2^-100, B 2^-127 (subnormal), C 2^40, D
    w[3, 0], w[3, 64] = emu.bf16_bits(p2(-60)), emu.bf16_bits(F32(-1.75) * p2(-61))
    got = gpu.mmvf(w, x, n_in, n_out, 3, fill=0xFF).reshape(3, n_out)
    want = emu.mmvf(w, x, kernels.basalt_mmvf_block(n_in))
    assert np.array_equal(bits(got), bits(want))
    ieee = emu.mmvf(w, x, kernels.basalt_mmvf_block(n_in), flush=False)
    for k, r in [(0, 0), (2, 1), (1, 2), (2, 3)]:
        assert bits(got)[k, r] == 0 and ieee[k, r] != 0, (k, r)
    # one pair (n_in 2): output (k, k) for k = 0, 1, 2 and (1, 3), on the one-row and the multi kernel
    w = emu.bf16_bits(np.array([[31 * 2.0 ** -70, 0], [1, -2.0 ** -25], [2.0 ** -126, 0], [-1, 2.0 ** -25]], F32))
    x = np.array([[1082401 * 2.0 ** -81, 0], [2.0 ** -126, 2.0 ** -126], [1 - 2.0 ** -24, 0]], F32)
    want = emu.mmvf(w, x, 32)
    assert list(bits(want)[[0, 1, 2, 1], [0, 1, 2, 3]]) == [0x00800000, 0x00800000, 0x00800000, 0x80800000]
    assert np.array_equal(bits(one_row(w, x, 2, 4)), bits(want))
    assert np.array_equal(bits(gpu.mmvf(w, x, 2, 4, 3, fill=0xFF).reshape(3, 4)), bits(want))


def test_launch_rules():
    """basalt's mmvf_block_size and launcher routing (native_bf16.cu:124-203), values worked by hand."""
    assert [kernels.basalt_mmvf_block(n) for n in (2, 64, 66, 130, 700, 1000, 2560)] == [32, 32, 64, 96, 192, 256, 256]
    assert kernels.basalt_mmvf_macros(2560, 128) == {"BLOCK": "256", "NT": "1"}
    assert [kernels.basalt_mmvf_macros(2560, 1, t)["NT"] for t in (2, 4, 5, 8)] == ["4", "4", "8", "8"]
    assert kernels.basalt_mmvf_cols(17) == [(0, 8), (8, 8), (16, 1)]
    for bad in [(2559, 1, 1, None), (0, 1, 1, None), (2560, 0, 1, None), (2560, 1, 9, None), (2560, 1, 2, 2561)]:
        with pytest.raises(ValueError):
            kernels.basalt_mmvf_macros(*bad)
