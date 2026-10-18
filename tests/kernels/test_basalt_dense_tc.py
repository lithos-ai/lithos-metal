# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dense_tc_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its cases, pairs, scale offsets and sane_half scales, the T = 1..8 sweep, the 0xff output
# pre-fill and the pass criteria.
"""basalt's dense_tc and dense_tc_pair on Metal (kernels/common/basalt_dense_tc.metal) for this pack's Q2_0, Q3_K,
Q4_K, Q5_K, Q6_K and IQ4_XS weights, in every summation order basalt's variant rule can pick.

- dense_tc_parity, ported (basalt src/kernels/dense_tc_parity.cpp): its cases (:37-63) and pairs (:386-388), random
  weights with normal fp16 scales (:65-90) and N(0, 1) activations quantized once by the native quantizer (:108). Each
  case runs through the order the rule picks at 170 multiprocessors (basalt's measured RTX 5090) and through every
  other order; each order: every column of a T-column call (T = 1..8, the activation buffer holding T columns) bitwise
  equal to the same column of the 8-column call, columns past T never stored, and the 8-column output's RMS error
  against native_mmvq's one-column outputs. Criteria, quoted: ``const bool ok = bits == 0 && nonfinite == 0 && rms <
  1e-3;`` (:137) and for the pairs ``const bool k = bits == 0 && rms < 1e-3;`` (:201). The cases and pairs with IQ3_S
  (z IQ3_S, out IQ3_S; {qkv Q6_K, z IQ3_S}, {qkv IQ4_XS, z IQ3_S}) are left out: this pack has no IQ3_S weights, and
  the port has no IQ3_S terms.
- The pair's outputs bitwise equal to dense_tc's tile order (``a tile's arithmetic is dense_tc_tile_block's``,
  dense_tc.cu:704-705).
- Bit for bit against the numpy emulation of what nvcc compiles (tests/kernels/basalt_emu.py ``dense_tc``,
  docs/research/basalt-nvcc/dense_tc.md), every order, at shapes with empty warp groups (n_in 2560 in the one order),
  KS 2 at n_in 1024 and 2048, and row tails; the orders must differ somewhere, so the comparison can tell them apart.
- A float64 reference per type, every order, under the bound docs/design/flash-next.md gives dense_tc's comparand
  native_mmvq: for Q4_K, Q5_K and Q6_K, which no basalt test checks against a reference, mmvq_multi_parity's Q8_0
  rule on the same q8_1 bytes, ``const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);``
  (mmvq_multi_parity.cpp:461); for Q3_K, IQ4_XS and Q2_0, which iq_parity runs (iq_parity.cpp:29), ``const bool ok =
  dq_err < 1e-6 && mm_err < 2e-2 && multi_bad == 0;`` (iq_parity.cpp:112), mm_err the relative L1 error against
  ref . x in float64 on the float activations."""

import numpy as np
import pytest

from monolith import kernels
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_mmvq import columns, reference, weights

F32 = np.float32
NAMES = {11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 23: "IQ4_XS", 42: "Q2_0"}
VARIANTS = ("one", "tile", "s1", "s2")

# dense_tc_parity.cpp:37-63, in its order (the pairs index it); type 21 is IQ3_S
CASES = [("qkv Q6_K", 14, 2560, 10240), ("qkv Q4_K", 12, 2560, 10240), ("qkv Q5_K", 13, 2560, 10240),
         ("qkv IQ4_XS", 23, 2560, 10240), ("z Q6_K", 14, 2560, 6144), ("z IQ4_XS", 23, 2560, 6144),
         ("z IQ3_S", 21, 2560, 6144), ("out Q6_K", 14, 6144, 2560), ("out IQ4_XS", 23, 6144, 2560),
         ("out Q4_K", 12, 6144, 2560), ("out Q5_K", 13, 6144, 2560), ("out IQ3_S", 21, 6144, 2560),
         ("q Q6_K", 14, 2560, 12288), ("kv Q6_K", 14, 2560, 512), ("tail Q4_K", 12, 1024, 1000),
         ("tail Q6_K", 14, 1536, 999), ("qkv Q3_K", 11, 2560, 10240), ("z Q3_K", 11, 2560, 6144),
         ("out Q3_K", 11, 6144, 2560), ("q Q3_K", 11, 2560, 12288), ("qkv Q2_0", 42, 2560, 10240),
         ("z Q2_0", 42, 2560, 6144), ("q Q2_0", 42, 2560, 12288), ("tail Q3_K", 11, 1536, 999),
         ("tail Q2_0", 42, 1024, 1000)]
PAIRS = [(0, 6), (1, 5), (2, 4), (3, 6), (0, 5), (16, 17), (3, 17), (20, 21), (16, 21), (1, 17), (20, 17)]
PACK_CASES = [c for c in CASES if c[1] != 21]
PACK_PAIRS = [p for p in PAIRS if CASES[p[0]][1] != 21 and CASES[p[1]][1] != 21]


def variants(n_in):
    """Every order the rule can pick; KS 2 only where n_in % 1024 == 0 (dense_tc.cu:1056)."""
    return [v for v in VARIANTS if v != "s2" or n_in % 1024 == 0]


PARITY = [(c, v) for c in PACK_CASES for v in variants(c[2])]


def native_ref(type_id, w, xq, n_in, n_out):
    """native_mmvq's one-column outputs of the eight columns (dense_tc_parity.cpp:109-111)."""
    return np.stack([gpu.mmvq(type_id, w, columns(xq, n_in, j), n_in, n_out, 1)[0] for j in range(8)])


def rms_err(got, ref):
    d = got.astype(np.float64) - ref.astype(np.float64)
    return np.sqrt((d * d).sum() / max((ref.astype(np.float64) ** 2).sum(), 1e-300))


def width_checks(outs, n_out):
    """bits: values of a T-column call that differ from the 8-column call's; nonfinite: non-finite values written;
    stray: values written past column T (the output starts as 0xff bytes)."""
    g8 = outs[8]
    bits = nonfinite = stray = 0
    for t, y in outs.items():
        bits += np.count_nonzero(y[:t].view(np.uint32) != g8[:t].view(np.uint32))
        nonfinite += np.count_nonzero(~np.isfinite(y[:t]))
        stray += np.count_nonzero(y[t:].view(np.uint32) != 0xFFFFFFFF)
    return bits, nonfinite, stray


_FIXTURES = {}


def case_fixture(case):
    """dense_tc_parity's per-case data: weights, the 8 columns' q8_1 bytes and native_mmvq's outputs, made once."""
    if case not in _FIXTURES:
        _, type_id, n_in, n_out = case
        rng = np.random.default_rng(777 + type_id * 31 + n_out)
        w = weights(type_id, n_in, n_out, rng)
        xq = gpu.q8_1(rng.standard_normal(8 * n_in).astype(F32), "native")
        _FIXTURES.clear()
        _FIXTURES[case] = (w, xq, native_ref(type_id, w, xq, n_in, n_out))
    return _FIXTURES[case]


@pytest.mark.parametrize("case,variant", PARITY, ids=[f"{c[0]}-{v}" for c, v in PARITY])
def test_dense_tc_parity(case, variant):
    name, type_id, n_in, n_out = case
    w, xq, ref = case_fixture(case)
    outs = gpu.dense_tc(type_id, w, xq, n_in, n_out, range(1, 9), variant)
    bits, nonfinite, stray = width_checks(outs, n_out)
    rms = rms_err(outs[8], ref)
    picked = kernels.basalt_dense_tc_variant(type_id, n_in, n_out)
    print(f"{name} {variant}{' (the rule at 170)' if variant == picked else ''}: {bits} diff, {nonfinite} nonfinite, "
          f"rms {rms:.3e}")
    # basalt: ``const bool ok = bits == 0 && nonfinite == 0 && rms < 1e-3;`` (dense_tc_parity.cpp:137)
    assert bits == 0 and nonfinite == 0 and rms < 1e-3, (bits, nonfinite, rms)
    assert stray == 0                                   # columns past T are never stored


@pytest.mark.parametrize("pair", PACK_PAIRS, ids=[f"{CASES[a][0]}+{CASES[b][0]}" for a, b in PACK_PAIRS])
def test_dense_tc_pair_parity(pair):
    a, b = CASES[pair[0]], CASES[pair[1]]
    rng = np.random.default_rng(4242 + a[1] * 7 + b[1])
    n_in = a[2]
    wa, wb = weights(a[1], n_in, a[3], rng), weights(b[1], n_in, b[3], rng)
    xq = gpu.q8_1(rng.standard_normal(8 * n_in).astype(F32), "native")
    outs = gpu.dense_tc_pair(a[1], wa, a[3], b[1], wb, b[3], xq, n_in, range(1, 9))
    for m, (c, w) in enumerate(((a, wa), (b, wb))):
        mine = {t: o[m] for t, o in outs.items()}
        bits, nonfinite, stray = width_checks(mine, c[3])
        rms = rms_err(mine[8], native_ref(c[1], w, xq, n_in, c[3]))
        print(f"pair {a[0]} + {b[0]} [{m}]: {bits} diff, rms {rms:.3e}")
        # basalt: ``const bool k = bits == 0 && rms < 1e-3;`` (dense_tc_parity.cpp:201)
        assert bits == 0 and rms < 1e-3, (m, bits, rms)
        assert stray == 0 and nonfinite == 0, m
        # the pair's tile arithmetic is dense_tc_tile_block's (dense_tc.cu:704-705)
        tile = gpu.dense_tc(c[1], w, xq, n_in, c[3], (8,), "tile")[8]
        assert np.array_equal(mine[8].view(np.uint32), tile.view(np.uint32)), m


EMU_SHAPES = [(2560, 37), (6144, 21), (1024, 19), (2048, 33)]


@pytest.mark.parametrize("type_id", sorted(NAMES), ids=[NAMES[t] for t in sorted(NAMES)])
def test_emulation_bit_for_bit(type_id):
    """Every order against the numpy emulation, one, three and eight columns; at n_in 2560 the one order's warps of
    group 5 have no chunk (5 chunks for 6 groups) and add their zeros, as in basalt."""
    for n_in, n_out in EMU_SHAPES:
        rng = np.random.default_rng(99 + type_id + n_in)
        w = weights(type_id, n_in, n_out, rng)
        x = rng.standard_normal(8 * n_in).astype(F32)
        x[: 64] = 0.0                                                    # all-zero q8_1 blocks
        xq = gpu.q8_1(x, "native")
        got = {}
        for v in variants(n_in):
            want = emu.dense_tc(type_id, w, xq, n_in, n_out, 8, v)
            got[v] = gpu.dense_tc(type_id, w, xq, n_in, n_out, (1, 3, 8), v)
            for t, y in got[v].items():
                # the bar is docs/design/flash-next.md invariant 5 (nvcc's sequence, bit for bit); basalt has no
                # counterpart test
                assert np.array_equal(y[:t].view(np.uint32), want[:t].view(np.uint32)), (n_in, v, t)
        if n_in == 6144:                                                 # power: the orders' bits differ
            for i, u in enumerate(VARIANTS):
                for v in VARIANTS[i + 1:]:
                    assert not np.array_equal(got[u][8].view(np.uint32), got[v][8].view(np.uint32)), (u, v)


def test_pair_emulation_bit_for_bit():
    """dense_tc_pair with two formats against the emulation of the pair order, row tails in both matrices."""
    n_in = 2560
    for t0, t1 in ((14, 23), (12, 11), (42, 13)):
        rng = np.random.default_rng(5 + t0 + t1)
        w0, w1 = weights(t0, n_in, 35, rng), weights(t1, n_in, 18, rng)
        xq = gpu.q8_1(rng.standard_normal(8 * n_in).astype(F32), "native")
        outs = gpu.dense_tc_pair(t0, w0, 35, t1, w1, 18, xq, n_in, (2, 8))
        for m, (ty, w, n_out) in enumerate(((t0, w0, 35), (t1, w1, 18))):
            want = emu.dense_tc(ty, w, xq, n_in, n_out, 8, "pair")
            for t, o in outs.items():
                # flash-next.md invariant 5, bit for bit; basalt has no counterpart test
                assert np.array_equal(o[m][:t].view(np.uint32), want[:t].view(np.uint32)), (t0, t1, m, t)


@pytest.mark.parametrize("type_id", sorted(NAMES), ids=[NAMES[t] for t in sorted(NAMES)])
def test_float64_reference(type_id):
    """Every order, eight columns. Q4_K, Q5_K and Q6_K: mmvq_multi_parity.cpp:461's rule, worst |y - ref| <= 1e-4 *
    max |ref| per column, ref the float64 product on the same q8_1 bytes (test_basalt_mmvq.reference); its diff == 0
    is test_dense_tc_parity's bits == 0. Q3_K, IQ4_XS and Q2_0: iq_parity's rule (iq_parity.cpp:103-112), mm_err =
    sum |y - ref . x| / sum |ref . x| < 2e-2, ref the weights dequantized by gguf-py's decoders
    (tests/contract/gguf_reference.py) and x the float activations, ref . x in float64."""
    n_in, n_out = 6144, 40
    rng = np.random.default_rng(31 * type_id)
    w = weights(type_id, n_in, n_out, rng)
    x = rng.standard_normal(8 * n_in).astype(F32)
    xq = gpu.q8_1(x, "native")
    same_bytes = type_id in (12, 13, 14)
    if same_bytes:
        ref = np.stack([reference(type_id, w, columns(xq, n_in, j), n_in, n_out) for j in range(8)])   # (8, n_out)
    else:
        ref = x.reshape(8, n_in).astype(np.float64) @ gguf_reference.dequantize(type_id, w).astype(np.float64).T
    for v in variants(n_in):
        y = gpu.dense_tc(type_id, w, xq, n_in, n_out, (8,), v)[8].astype(np.float64)
        assert np.all(np.isfinite(y)), v
        if same_bytes:
            worst, scale = np.abs(y - ref).max(axis=1), np.abs(ref).max(axis=1)
            print(f"{NAMES[type_id]} {v}: worst / scale up to {(worst / scale).max():.2e}")
            # basalt: ``const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);`` (mmvq_multi_parity.cpp:461)
            assert np.all(worst <= 1e-4 * (scale + 1e-30)), (v, worst / scale)
        else:
            mm_err = np.abs(y - ref).sum() / (np.abs(ref).sum() + 1e-30)
            print(f"{NAMES[type_id]} {v}: mm_err {mm_err:.2e}")
            # basalt: ``const bool ok = dq_err < 1e-6 && mm_err < 2e-2 && multi_bad == 0;`` (iq_parity.cpp:112)
            assert mm_err < 2e-2, (v, mm_err)
