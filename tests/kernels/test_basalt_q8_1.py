# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/iq_multi_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): check_q8_1_finite's planted blocks, its host transcription and its pass criterion.
"""basalt's q8_1 activation quantizers on Metal (kernels/common/basalt_q8_1.metal).

- ``check_q8_1_finite``, ported from basalt's iq_multi_parity.cpp:421-484: both quantizers on 3 rows of 2560 N(0, 1)
  values with four planted blocks, against the host formula (d = amax / 127 in IEEE fp32, the butterfly's sum,
  ``std::round(v[i] / d)``). Basalt's criterion, quoted: every finite block exact, or for ``native_quantize_q8_1``
  ``bool near = path == 1 && gs == os && std::abs((int) gd - (int) od) <= 1;`` (:465) with every code within one
  (:466); overflowed blocks finite with codes in +-127, exactly three of them: ``g_fail += bad + (clamped != 3);``
  (:480).
- Bit for bit against the numpy emulation of what nvcc compiles (tests/kernels/basalt_emu.py, matvec.md sections 1-3),
  on normal data and on planted edge blocks (zeros, subnormals, overflow, NaN, a scale that flushes).
- How often the native sequence's quotient through the correctly rounded reciprocal, fl(xi * fl(1 / d)), differs from
  the correctly rounded quotient fl(xi / d), and how often that changes a code (printed; run with -s)."""

import numpy as np

from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32 = np.float32


def finite_case():
    n, rows = 2560, 3
    x = np.random.default_rng(18).standard_normal(rows * n).astype(F32)
    x[5 * 32: 6 * 32] = 3000.0                                     # sum 96,000: overflowed
    x[9 * 32: 10 * 32] = 0.5
    x[9 * 32 + 3] = -70000.0                                       # sum -69,984.5
    x[13 * 32: 14 * 32] = 2000.0 + np.arange(32, dtype=F32)        # sum 64,496: finite, kept
    x[n + 40] = 2.0e7                                              # amax / 127 overflowed too
    return x


def host_formula(v):
    """The test's host transcription (iq_multi_parity.cpp:437-458): od, os, the codes, and the finite flag."""
    amax = np.abs(v).max()
    lane = v.copy()
    for o in (16, 8, 4, 2, 1):
        lane = lane + lane[np.arange(32) ^ o]
    d, s = amax / F32(127.0), lane[0]
    od, os = emu.half_bits(d)[()], emu.half_bits(s)[()]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.zeros(32, F32) if amax == 0 else emu.roundf(v / d)
    return od, os, q.astype(np.int8), (od & 0x7C00) != 0x7C00 and (os & 0x7C00) != 0x7C00, s


def test_check_q8_1_finite():
    x = finite_case()
    for path, kind in ((0, "ieee"), (1, "native")):
        got = gpu.q8_1(x, kind).reshape(-1, 36)
        same = clamped = bad = 0
        for b in range(len(got)):
            v = x[32 * b: 32 * b + 32]
            od, os, want_q, finite_before, s = host_formula(v)
            g = got[b]
            gd, gs = int(g[0:2].view(np.uint16)[0]), int(g[2:4].view(np.uint16)[0])
            codes = g[4:].view(np.int8).astype(int)
            if finite_before:
                want = np.concatenate([np.array([od, os], np.uint16).view(np.uint8), want_q.view(np.uint8)])
                # basalt: ``bool near = path == 1 && gs == os && std::abs((int) gd - (int) od) <= 1;`` (:465), and
                # every code within one (:466)
                near = path == 1 and gs == os and abs(gd - int(od)) <= 1 and np.all(np.abs(codes - want_q) <= 1)
                if np.array_equal(want, g) or near:
                    same += 1
                else:
                    bad += 1
            else:
                fd, fs = emu.h2f(gd), emu.h2f(gs)
                ok = np.isfinite(fd) and np.isfinite(fs) and fd > 0 and (
                    gs == os if (os & 0x7C00) != 0x7C00 else abs(fs) == 65504.0 and (fs < 0) == (s < 0))
                ok = ok and codes.min() >= -127 and codes.max() <= 127
                clamped += ok
                bad += not ok
        # basalt: ``g_fail += bad + (clamped != 3);`` (:480)
        assert bad == 0 and clamped == 3, (kind, same, clamped, bad)
        assert same == len(got) - 3


def edge_blocks(rng):
    """Planted blocks: zeros, subnormals only, subnormals with tiny normals (an IEEE subnormal d), a normal amax whose
    native d flushes, mixed signs at the subnormal boundary, overflow, NaN and infinities."""
    sub = lambda k: ((rng.integers(1, 1 << 23, k, dtype=np.uint32)) | (rng.integers(0, 2, k, dtype=np.uint32) << 31)).view(F32)
    blocks = [np.zeros(32, F32), -np.zeros(32, F32), sub(32), sub(32) * F32(1.0)]
    b = sub(32); b[7] = F32(3e-37); blocks.append(b)                                   # amax normal, d subnormal
    b = sub(32); b[0] = F32(2.0 ** -119); blocks.append(b)                            # d at the boundary
    b = sub(32); b[0] = F32(127 * 2.0 ** -126); blocks.append(b)
    b = np.full(32, F32(2.0 ** -126)); b[::2] *= -1; blocks.append(b)                 # sums that cancel exactly
    b = sub(32); b[1] = F32(2.0 ** -100); blocks.append(b)                            # the IEEE emulation's threshold
    b = rng.standard_normal(32).astype(F32) * F32(1e-38); blocks.append(b)            # normals and subnormals mixed
    b = rng.standard_normal(32).astype(F32); b[3] = np.nan; blocks.append(b)
    b = rng.standard_normal(32).astype(F32); b[4] = np.inf; blocks.append(b)
    blocks.append(np.full(32, np.nan, F32))
    b = rng.standard_normal(32).astype(F32) * F32(1e5); b[0] = F32(3e38); blocks.append(b)
    return np.concatenate(blocks)


def test_quantizers_match_the_emulation_bit_for_bit():
    rng = np.random.default_rng(5)
    x = np.concatenate([finite_case(), rng.standard_normal(1 << 16).astype(F32) * F32(3.0),
                        (rng.standard_normal(1 << 14) * np.exp(rng.uniform(-80, 80, 1 << 14))).astype(F32),
                        edge_blocks(rng)])
    nan_blocks = np.isnan(x.reshape(-1, 32)).any(axis=1)
    for kind, emulate in (("native", emu.q8_1_native), ("ieee", emu.q8_1_ieee)):
        got, want = gpu.q8_1(x, kind).reshape(-1, 36), emulate(x).reshape(-1, 36)
        # a NaN's fp16 payload is the GPU's; its codes, its other half and every other block are bit for bit
        diff = np.any(got != want, axis=1)
        assert not np.any(diff & ~nan_blocks), (kind, np.flatnonzero(diff & ~nan_blocks)[:8])
        assert np.array_equal(got[nan_blocks, 4:], want[nan_blocks, 4:]), kind
        isnan_h = lambda h: (h & 0x7C00) == 0x7C00 and (h & 0x3FF) != 0
        for g, w in zip(got[nan_blocks], want[nan_blocks]):
            for k in (0, 2):
                gh, wh = int(g[k:k + 2].view(np.uint16)[0]), int(w[k:k + 2].view(np.uint16)[0])
                assert gh == wh or (isnan_h(gh) and isnan_h(wh)), kind


def test_reciprocal_division_against_the_correctly_rounded_quotient():
    """Where nvcc compiles xi / d as MUFU.RCP then FMUL, the port multiplies by the correctly rounded reciprocal.
    Count, on N(0, 1) activations, how often that product differs from the correctly rounded quotient that basalt's
    host formula divides with, and how often a code changes; the port's codes are then the host's within one, as
    check_q8_1_finite's near rule allows."""
    rng = np.random.default_rng(6)
    v = rng.standard_normal((1 << 16, 32)).astype(F32)
    amax = np.abs(v).max(axis=1)
    d = emu.q8_1_finite(emu.mul_ftz(amax, emu.RCP127))[:, None]
    via_rcp = emu.mul_ftz(v, emu.rcp_ftz(d))
    quotient = (v / d).astype(F32)
    q_diff = np.count_nonzero(via_rcp.view(np.uint32) != quotient.view(np.uint32))
    c_diff = np.count_nonzero(emu.roundf(via_rcp) != emu.roundf(quotient))
    d_ieee = (amax / F32(127.0)).astype(F32)
    d_diff = np.count_nonzero(d[:, 0].view(np.uint32) != d_ieee.view(np.uint32))
    print(f"\nq8_1 native division: {q_diff} of {v.size} quotients ({q_diff / v.size:.3%}) differ from fl(xi / d), "
          f"{c_diff} codes ({c_diff / v.size:.4%}); d = fl(amax * fl(1/127)) differs from fl(amax / 127) in "
          f"{d_diff} of {len(d_ieee)} blocks ({d_diff / len(d_ieee):.2%})")
    assert np.all(np.abs(emu.roundf(via_rcp) - emu.roundf(quotient)) <= 1)
    got = gpu.q8_1(v.reshape(-1), "native").reshape(-1, 36)[:, 4:].view(np.int8)
    assert np.array_equal(got, emu.q8_1_codes(emu.roundf(via_rcp), amax))


def test_swiglu_quantizer():
    """native_swiglu_quantize_q8_1: no basalt test covers it (pr-b-scope-basalt.md section 2.5). Its value is
    __fmul_rn(__fdividef(g, __fadd_rn(1, __expf(-g))), up); MUFU.EX2 has no documented Metal counterpart, so the
    exponential is taken from the GPU's fast::exp2 of the same product and every other step is checked bit for bit,
    then the codes against the IEEE SiLU product with check_q8_1_finite's near rule."""
    rng = np.random.default_rng(7)
    n = 2560 * 4
    g = (rng.standard_normal(n) * 4).astype(F32)
    up = rng.standard_normal(n).astype(F32)
    got = gpu.q8_1(g, "swiglu", up).reshape(-1, 36)
    src = gpu.kernels.PRELUDE + gpu.kernels.template("basalt_fp.metal") + """
kernel void ex(device const float* g [[buffer(0)]], device float* e [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint i [[thread_position_in_grid]]) {
  if (i < n) e[i] = ftz(fast::exp2(mul_ftz(g[i], as_type<float>(0xBFB8AA3Bu))));
}
"""
    e = np.frombuffer(gpu.run(gpu.pipeline(src, "ex"), [g.tobytes(), 4 * n], np.uint32(n).tobytes(), 2,
                              n // 256, 256)[1], F32)
    xi = emu.mul_ftz(emu.mul_ftz(g, emu.rcp_ftz(emu.add_ftz(F32(1.0), e))), up)
    assert np.array_equal(got.reshape(-1), emu.q8_1_native(xi))
    exact = np.exp2(emu.mul_ftz(g, np.uint32(0xBFB8AA3B).view(F32)).astype(np.float64)).astype(F32)
    print(f"\nfast::exp2 differs from the correctly rounded exp2 on {np.count_nonzero(exact != e)} of {n} values")
    silu = (g.astype(np.float64) / (1 + np.exp(-g.astype(np.float64))) * up).astype(F32)
    host = emu.q8_1_ieee(silu).reshape(-1, 36)
    assert np.all(np.abs(got[:, 4:].view(np.int8).astype(int) - host[:, 4:].view(np.int8)) <= 1)
