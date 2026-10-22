# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp, src/kernels/ple_win_parity.cpp, src/core/verify.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's metric helpers (le, gt,
# nonfinite and rel_l1, :41-77), the seven stages its block check compares (:751-759), its host grouped norm
# (:899-904) and its host f32 reference block with the history slide (:1046-1102, :1127-1131); ple_win_parity's
# fixture scales (:48-56); ple_parity's capture loader rules (:135-164: a file missing or of the wrong size fails);
# verify.cpp's commit record and its count refusal (:2732-2737).
"""The n-gram (PLE) block's host references, numpy only (no Metal import), imported as ``from tests.basalt_ple_ref
import ...``.

- ``le``, ``gt``, ``nonfinite`` and ``rel_l1``: ple_parity's NaN-safe metric helpers (ple_parity.cpp:41-77).
  ``rel_l1(a, b)`` takes the reference first and returns ``(rel, mag, nf)``: rel = sum |a - b| / sum |a| (the sum
  floored at 1e-30) in float64, NaN when any pair has a non-finite entry; mag = mean |a| (the finite pairs' sum over
  the length); nf = the count of non-finite pairs.
- R-struct (``r_struct``, spec T§6.2): the block in float64, from the graph comment (basalt
  include/basalt/kernels/ple.hpp:3-13), not from basalt's host f32 reference. The Q2_0 key is decoded by
  tests/contract/gguf_reference.py's ``q2_0``; BF16, F16 and F32 are widened, exactly. For every row it returns the
  capture's seven slices (ple_parity.cpp:171-172: the normed key, value, gate, gated, normalized, conv_out and result),
  the history after the row and the gate's argument s. The window is one pad, P = [history | normalized rows], read
  at rows t, t + 3, t + 6 and t + 9 (ple.hpp's ``padded[hist + t - (kern - 1 - k) * dil]``), so the 12 rows run from
  one initial history. ``r_contract`` (R-contract) is R-struct entered after the projections, fed a raw key and value.
  ``STAGES`` names the seven slices.
- R-f32 (``r_f32``): basalt's own host f32 reference block (ple_parity.cpp:1046-1102), transcribed, every row in order
  from ``fx.hist`` with its own slide between rows (:1127-1131). It shares only the weights with R-struct: it works on
  basalt's flat buffers (the history row-fastest, row r of channel c at r + 9 c; conv1d at [k + 4 c]; the taps read
  rows 0, 3 and 6 and the new normalized row), rounds to f32 wherever the C++ does (an x86-64 SSE2 build: no FMA, no
  flush), adds each float64 sum in the C++ loop's order (a product of two f32 is exact) and takes std::exp's float
  overload as exp in float64 rounded once to f32. It returns a ``Ref`` of float32 arrays, ``hist`` the slid history
  after each row in R-struct's (row, channel) order.
- The host norm (``host_norm``): ple_parity.cpp:899-904, the grouped norm C8 holds the GPU's against, which is R-f32's
  ``gnorm`` (:1046-1059) operation for operation: per stream, the f32 squares summed in float64 in column order, the
  mean rounded to f32, scale = 1.0f / sqrtf(mean + 1e-6f), then (x * scale) * w, all in f32.
- H-ORACLE (``h_oracle``, spec T§7.1): the history a commit of ``n_keep`` rows of a window run from ``hist`` leaves,
  the last nine rows of [hist rows 0..8 | the window's normalized rows 0..n_keep - 1], in basalt's layout (row r of
  channel c at c * 9 + r). It only moves words, so it holds bit for bit on the GPU's own rows, NaN payloads included;
  it refuses an ``n_keep`` outside 1..T (form (a)'s no-op leaves the history advanced by all T rows, not as it was).
- ``commit_record(n_keep, t, pos0)``: the commit record, the host-written int32 graph input of 2 + T_MAX words, for a
  window of ``t`` rows at positions pos0, pos0 + 1, ..., as basalt's host writes it (verify.cpp:2735-2737): n_keep,
  n_keep - 1 (the snapshot the n-gram commit restores), then pos0 + r for each row r < n_keep of the T_MAX and -1 for
  the rest. It raises ValueError unless 1 <= n_keep <= t, basalt's refusal (verify.cpp:2732).
- The synthetic fixture (``synthetic``, spec T§6.3), ``Fixture``: the weights in the encodings the GPU reads (the key
  as type-42 rows, the value as BF16 bits, the conv kernel as F16 in ggml's flat [k + 4 i]) and float32 activations;
  the history is logical, row r oldest first (its row-fastest buffer layout, hist[r + 9 c], is the harness's).
  Seeds: ``default_rng(SYNTHETIC_SEED)`` draws, in order, the key's codes (uniform bytes: four uniform 2-bit codes
  each) and its d = f16(0.02 * U(0.5, 1.5)) per 64-weight block, w_nk, w_nq and w_nc = f32(1 + 0.1 * N(0,1)) (in that
  order), conv1d = f16(0.3 * N(0,1)), emb = f32(0.2 * N(0,1)) (12 x 2560), the history f32(N(0,1)) (9 x 10240) and
  hidden's unit draw z (12 x 10240); the value is bf16(0.05 * N(0,1)) from ``default_rng(VALUE_SEED)``, the
  first (2560, 2560) draw, bit for bit test_ple_value_float64's weights (tests/kernels/test_basalt_mmvf.py). hidden =
  f32(sigma_h * z), with sigma_h by T§6.3's rule 3: R-struct with the unit-scale hidden f32(z), then sigma_h = mean
  |gated + conv_out| / mean |f32(z)|. ``SYNTHETIC_SEED`` = 54 is the first seed from 0 for which property 2 holds
  (seeds 0-53 put some |s[c]| below 0.1).
- The real weights (spec T§5): ``TENSORS`` places blk.1's six PLE tensors in the pack (make_basalt_fixture.py's pinned
  file; absolute offsets) and ``RANGES`` the two byte ranges that hold them, one per name space. ``committed()`` reads
  ``FIXTURE``, the committed slices (key row 0, the first 1024 B of the five dense rows and of their GGUF twins, all of
  the dense conv1d) with the writer's record of where each lies; ``full_ranges()`` reads the two full ranges from
  $BASALT_PLE_FIXTURE (tests/contract/fixtures/make_basalt_ple_fixture.py --full DIR fetches them): unset, the caller
  skips with a reason naming the variable; set, a file missing or of the wrong size fails (ple_parity's L1-L4, spec
  T§1.1). ``real_tensors()`` cuts the full ranges at TENSORS' offsets. ``real`` is T§6.3's real fixture: the key as
  the session binds it (gguf, type 42), the value, the three norms and conv1d from their dense rows, and
  ``_with_activations(default_rng(REAL_SEED), 0.0158, ...)``: emb = f32(0.0158 * N(0,1)), the history f32(N(0,1)) and
  hidden by rule 3. ``REAL_SEED`` = 279 is the first seed from 0 for which property 2 holds (seeds 0-278 put some
  |s[c]| below 0.1).
- ``check_fixture``: the four properties every generator asserts on R-struct before it returns (T§6.3):
  1. hidden is nonzero (every row's every stream: with hidden = 0, s = 0 and gate = 0.5, the key is never observed);
  2. min over rows and streams of |s[c]| >= 0.1;
  3. the block's share of result, sum |gated + conv_out| / sum |result|, is at least 0.3 on every row;
  4. T = 12 rows from one initial history: each row's normalized row enters the history the next rows read."""

import functools
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from monolith.formats.safetensors_reader import read_header
from tests.contract.gguf_reference import q2_0
from tests.kernels.basalt_emu import bf16_bits

F32, F64 = np.float32, np.float64
N, H = 2560, 4                  # NG_N_EMBD, NG_HC (basalt include/basalt/kernels/ngram.hpp:30-41)
D = H * N                       # NG_HC_DIM: a [D] vector is stream-major, element (c, d) at c * N + d
KERN, DIL = 4, 3                # PLE_CONV_KERNEL, NGRAM_SIZE
HIST = (KERN - 1) * DIL         # NG_HIST: 9 history rows, oldest first
ROWS = 12                       # T§6.1: every block-level comparison runs 12 rows from one initial history
T_MAX = 8                       # kVerifyMaxT (include/basalt/kernels/verify_kernels.hpp:21): a window's most rows
EPS = float(F32(1e-6))          # NG_RMS_EPS as float64(float32(1e-6)): the norms' eps and the gate's clamp
KEY_BLOCK = 18                  # a type-42 (Q2_0) block: the f16 d, then 16 bytes of 2-bit codes, for 64 weights
SYNTHETIC_SEED = 54             # the first seed from 0 for which property 2 holds: min |s| 0.1251, min row share 0.628
VALUE_SEED = 790                # test_ple_value_float64's
REAL_SEED = 279                 # the first seed from 0 for which property 2 holds: min |s| 0.1219, min row share 0.562


class PackTensor(NamedTuple):
    type_id: int          # ggml type
    ne: tuple             # ggml order, fastest first
    nbytes: int
    gguf: int             # the GGUF tensor's absolute offset in the pack
    kind: int             # its dense.index row's kind: 5 F16, 2 F32, 4 BF16 as stored, 0 shape-only
    dense: int | None     # that row's absolute offset (dense base + src_off); None: shape-only, the bytes are in gguf


TENSORS = {  # spec T§5's table in file order; the shapes are ngram.hpp:30-41's
    "blk.1.ple_conv1d.weight": PackTensor(1, (KERN, D), 81920, 36325965824, 5, 90852224),
    "blk.1.ple_key.weight": PackTensor(42, (N, D), 7372800, 36326047744, 0, None),
    "blk.1.ple_norm_conv.weight": PackTensor(0, (D,), 40960, 36333420544, 2, 90934144),
    "blk.1.ple_norm_key.weight": PackTensor(0, (D,), 40960, 36333461504, 2, 90975104),
    "blk.1.ple_norm_query.weight": PackTensor(0, (D,), 40960, 36333502464, 2, 91016064),
    "blk.1.ple_value.weight": PackTensor(30, (N, N), 13107200, 36333543424, 4, 91057024),
}
RANGES = {  # T§5: each name space's tensors lie in one range, (file under $BASALT_PLE_FIXTURE, absolute offset, bytes)
    "gguf": ("gguf_blk1_ple.bin", 36325965824, 20684800),
    "dense": ("dense_blk1_ple.bin", 90852224, 13312000),
}
FIXTURE = Path(__file__).parent / "contract" / "fixtures" / "basalt_ple.safetensors"


def le(x, t):
    return bool(np.isfinite(x) and x <= t)       # ple_parity.cpp:48


def gt(x, t):
    return bool(np.isfinite(x) and x > t)        # ple_parity.cpp:49


def nonfinite(a):
    """ple_parity.cpp:52-57: how many entries of ``a`` are not finite."""
    return int(np.count_nonzero(~np.isfinite(a)))


def rel_l1(a, b):
    """ple_parity.cpp:64-77, ``a`` the reference: (rel, mag, nf) as the module docstring says."""
    a, b = np.asarray(a, F64).ravel(), np.asarray(b, F64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    nf = int(np.count_nonzero(~ok))
    m = np.abs(a[ok]).sum()
    mag = m / (a.size or 1)
    if nf:
        return np.nan, mag, nf
    return np.abs(a - b).sum() / (m if m > 1e-30 else 1e-30), mag, 0


class Fixture(NamedTuple):
    key: np.ndarray       # uint8 (D, 720): W_key's type-42 rows, row o = W_key[o, :] in 40 blocks of 64
    value: np.ndarray     # uint16 (N, N): W_value's BF16 bits, row-major
    w_nk: np.ndarray      # float32 (D,): the key norm's gamma
    w_nq: np.ndarray      # float32 (D,): the query norm's
    w_nc: np.ndarray      # float32 (D,): the conv input norm's
    conv1d: np.ndarray    # float16 (KERN * D,): ggml's flat [k + 4 i]
    emb: np.ndarray       # float32 (ROWS, N): the gathered n-gram rows
    hist: np.ndarray      # float32 (HIST, D): the initial history, row r oldest first
    hidden: np.ndarray    # float32 (ROWS, D): the residual stack


class Ref(NamedTuple):
    key: np.ndarray         # (T, D): gnorm(W_key @ emb, w_nk)
    value: np.ndarray       # (T, N): W_value @ emb
    gate: np.ndarray        # (T, H)
    gated: np.ndarray       # (T, D)
    normalized: np.ndarray  # (T, D): gnorm(gated, w_nc), the conv input
    conv_out: np.ndarray    # (T, D): SiLU(conv)
    result: np.ndarray      # (T, D): hidden + gated + conv_out
    hist: np.ndarray        # (T, HIST, D): the history after row t, P[t + 1 .. t + 9]
    s: np.ndarray           # (T, H): the gate's argument


STAGES = Ref._fields[:7]    # the capture's seven slices, in its order (ple_parity.cpp:171-172)


@functools.lru_cache(maxsize=2)
def _key_weights(key):
    """W_key (D, N) in float32 as gguf_reference.q2_0 decodes ``key``, the bytes of the type-42 rows. The decode is a
    scalar loop of about 10 s, so it is cached per key, read-only."""
    w = q2_0(np.frombuffer(key, np.uint8).reshape(-1, KEY_BLOCK)).reshape(D, N)
    w.flags.writeable = False
    return w


def bf16_values(bits):
    """BF16 bits widened to float32, exactly."""
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(F32)


def gnorm(x, w):
    """T§6.2's grouped norm over each of the H streams of N, in float64:
    x[c, d] / sqrt(mean_d(x[c, :]^2) + eps) * w[c, d]."""
    x = np.asarray(x, F64).reshape(-1, H, N)
    y = x / np.sqrt(np.mean(x * x, axis=2, keepdims=True) + EPS) * np.asarray(w, F64).reshape(H, N)
    return y.reshape(-1, D)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x / (1 + np.exp(-x))


def taps(conv1d):
    """w_conv[i, k] = conv1d flat [k + 4 i] (ggml's ne = (kern, hc_dim)), float64."""
    return np.asarray(conv1d, F64).reshape(D, KERN)


def conv(P, w):
    """conv[t, i] = sum_{k=0..3} w[i, k] * P[t + 3k, i] over the padded rows P = [history | normalized rows], tap 0 the
    furthest back (ple.hpp's ``padded[hist + t - (kern - 1 - k) * dil]``)."""
    t = len(P) - HIST
    return sum(w[:, k] * P[DIL * k:DIL * k + t] for k in range(KERN))


def r_contract(fx, key_raw, value):
    """R-struct after the projections: ``key_raw`` (T, D) and ``value`` (T, N) are given (the GPU's, or R-struct's own),
    every other input is ``fx``'s, and the T rows run from ``fx.hist``."""
    t = len(key_raw)
    key = gnorm(key_raw, fx.w_nk)
    query = gnorm(fx.hidden, fx.w_nq)
    value = np.asarray(value, F64).reshape(t, N)
    s = (key * query).reshape(t, H, N).sum(axis=2) / np.sqrt(N)
    gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), EPS)))      # sign(0) = 0
    gated = (value[:, None, :] * gate[:, :, None]).reshape(t, D)
    normalized = gnorm(gated, fx.w_nc)
    P = np.concatenate([np.asarray(fx.hist, F64), normalized])
    conv_out = silu(conv(P, taps(fx.conv1d)))
    result = np.asarray(fx.hidden, F64) + gated + conv_out
    hist = np.stack([P[r + 1:r + 1 + HIST] for r in range(t)])
    return Ref(key, value, gate, gated, normalized, conv_out, result, hist, s)


def r_struct(fx):
    """T§6.2's R-struct of ``fx``, every row."""
    emb = np.asarray(fx.emb, F64)
    key_raw = emb @ _key_weights(fx.key.tobytes()).astype(F64).T
    value = emb @ bf16_values(fx.value).astype(F64).T
    return r_contract(fx, key_raw, value)


def host_norm(x, w):
    """ple_parity.cpp:899-904 (C8's host norm) and R-f32's ``gnorm`` (:1046-1059), one arithmetic: per stream of N,
    the f32 squares widened and summed in float64 in column order, the mean rounded to f32, scale = 1.0f /
    sqrtf(mean + 1e-6f), then (x * scale) * w, all in f32. ``x`` (..., D); float32, x's shape."""
    x = np.asarray(x, F32)
    xs = x.reshape(-1, H, N)
    total = np.cumsum((xs * xs).astype(F64), axis=2)[..., -1:]       # `sum += double(v * v)`, in order
    scale = F32(1) / np.sqrt((total / N).astype(F32) + F32(EPS))
    return (xs * scale * np.asarray(w, F32).reshape(H, N)).reshape(x.shape)


def _dot_in_order(w, x):
    """a[t, o] = the C++ loop's ``a += (double) w[o * n + i] * (double) x[i]`` for i = 0..n-1 in order, float64 (rows
    of x, rows of w). A product of two f32 is exact in float64, so the order of the adds is all there is to match."""
    wt = np.ascontiguousarray(np.asarray(w, F32).T)
    x = np.asarray(x, F64)
    a = np.zeros((len(x), wt.shape[1]))
    for i, col in enumerate(wt):
        a += np.multiply.outer(x[:, i], col)
    return a


def _expf(x):
    """std::exp's float overload as exp in float64 rounded once to f32, the correctly rounded value. glibc's expf,
    basalt's on Linux, errs by at most 0.502 ulp (runtime/src/basalt_cpu_expf.cpp:106-107): one ulp apart at most."""
    return np.exp(np.asarray(x, F64)).astype(F32)


def r_f32(fx):
    """R-f32, basalt's host f32 reference block (ple_parity.cpp:1046-1102), every row of ``fx`` in order with its
    slide between rows (:1127-1131), on the module docstring's terms."""
    k_raw = _dot_in_order(_key_weights(fx.key.tobytes()), fx.emb).astype(F32)       # :1066-1070, before the loop:
    v_raw = _dot_in_order(bf16_values(fx.value).reshape(N, N), fx.emb).astype(F32)  # :1073-1077, no history in them
    hist = np.asarray(fx.hist, F32).T.flatten()              # hist_ref = cap.hist: row r of channel c at r + 9 c
    w_conv = np.asarray(fx.conv1d, F32)                      # cap.w_conv[k + 4 c]
    i = np.arange(D)
    rows = []
    for t in range(len(fx.emb)):
        hid = np.asarray(fx.hidden[t], F32)
        key = host_norm(k_raw[t], fx.w_nk)                                         # :1071
        query = host_norm(hid, fx.w_nq)                                            # :1072
        a = np.cumsum((key.astype(F64) * query.astype(F64)).reshape(H, N), axis=1)[:, -1]
        s = a.astype(F32) / np.sqrt(F32(N))                                        # :1082
        mag = np.sqrt(np.fmax(np.abs(s), F32(1e-6)))
        sgn = np.where(s > 0, F32(1), np.where(s < 0, F32(-1), F32(0)))
        gate = F32(1) / (F32(1) + _expf(-(sgn * mag)))
        gated = v_raw[t][i % N] * gate[i // N]                                     # :1087-1088
        normalized = host_norm(gated, fx.w_nc)                                     # :1089
        acc = np.zeros(D, F32)
        for kk in range(KERN):                                                     # :1090-1101
            row = HIST - (KERN - 1 - kk) * DIL
            vv = normalized if row == HIST else hist[row::HIST]                    # hist_ref[row + 9 c]
            acc = acc + w_conv[kk::KERN] * vv                                      # w_conv[kk + 4 c]
        conv_out = acc / (F32(1) + _expf(-acc))
        result = hid + gated + conv_out                                            # :1102
        h = hist.reshape(D, HIST)                                                  # :1127-1131, in place
        h[:, :-1] = h[:, 1:]
        h[:, -1] = normalized
        rows.append((key, v_raw[t], gate, gated, normalized, conv_out, result, h.T.copy(), s))
    return Ref(*map(np.stack, zip(*rows)))


def h_oracle(hist, normalized, n_keep):
    """H-ORACLE (spec T§7.1) on the module docstring's terms: ``hist`` (D, 9) or flat and the returned history in
    basalt's layout, ``normalized`` (T, D), both float32 as the GPU stores them."""
    hist, normalized = np.asarray(hist, F32), np.asarray(normalized, F32)
    if not 1 <= n_keep <= len(normalized):
        raise ValueError(f"H-ORACLE keeps 1 to {len(normalized)} rows, not {n_keep}")
    seq = np.concatenate([hist.reshape(D, HIST), normalized[:n_keep].T], axis=1)
    return np.ascontiguousarray(seq[:, -HIST:]).reshape(hist.shape)


def commit_record(n_keep, t, pos0):
    """The commit record of a window of ``t`` rows at positions pos0, pos0 + 1, ..., on the module docstring's
    terms."""
    if n_keep < 1 or n_keep > t:                                      # verify.cpp:2732
        raise ValueError(f"verify: commit count out of range: n_keep {n_keep} for a window of {t} rows")
    return np.array([n_keep, n_keep - 1] + [pos0 + r if r < n_keep else -1 for r in range(T_MAX)], np.int32)


def check_fixture(fx, ref):
    """T§6.3's four properties of a fixture, asserted on its R-struct ``ref``; returns min |s| and the smallest row
    share, for the log."""
    chained = (fx.emb.shape == (ROWS, N) and fx.hidden.shape == (ROWS, D) and fx.hist.shape == (HIST, D)
               and ref.hist.shape == (ROWS, HIST, D) and (ref.hist[0, :-1] == fx.hist[1:]).all()
               and (ref.hist[:, -1] == ref.normalized).all() and (ref.hist[-1] == ref.normalized[ROWS - HIST:]).all())
    assert chained, ("T§6.3 property 4, \"T = 12 rows from one initial history\": the fixture has 12 rows and one "
                     "9-row history, and each row's normalized row enters the history the next rows read")
    zero = int(np.count_nonzero(~(np.abs(fx.hidden.reshape(ROWS, H, N)).max(axis=2) > 0)))
    assert zero == 0, (f"T§6.3 property 1, \"hidden is nonzero\": {zero} of {ROWS * H} row streams are zero (with "
                       "hidden = 0, query = 0, s = 0 and gate = 0.5 exactly, so the key is never observed)")
    min_s = float(np.abs(ref.s).min())
    assert min_s >= 0.1, f"T§6.3 property 2, \"min over rows and streams of |s[c]| ≥ 0.1\": {min_s!r}"
    share = np.abs(ref.gated + ref.conv_out).sum(axis=1) / np.abs(ref.result).sum(axis=1)
    assert (share >= 0.3).all(), ("T§6.3 property 3, \"The block's share of result is at least 0.3. The share is "
                                  f"Σ|gated + conv_out| / Σ|result| per row\": {np.round(share, 4).tolist()}")
    return min_s, float(share.min())


def _with_activations(rng, emb_scale, **weights):
    """The fixture of ``weights`` with T§6.3's activations from ``rng``: emb = f32(emb_scale * N(0,1)), the history
    f32(N(0,1)), hidden = f32(sigma_h * z) by rule 3; its four properties asserted."""
    emb = (emb_scale * rng.standard_normal((ROWS, N))).astype(F32)
    hist = rng.standard_normal((HIST, D)).astype(F32)
    z = rng.standard_normal((ROWS, D))
    fx = Fixture(emb=emb, hist=hist, hidden=z.astype(F32), **weights)
    ref = r_struct(fx)
    sigma = np.abs(ref.gated + ref.conv_out).mean() / np.abs(fx.hidden.astype(F64)).mean()
    fx = fx._replace(hidden=(sigma * z).astype(F32))
    check_fixture(fx, r_struct(fx))
    return fx


def synthetic(seed=SYNTHETIC_SEED):
    """T§6.3's synthetic fixture, drawn as the module docstring says."""
    rng = np.random.default_rng(seed)
    blocks = N // 64
    codes = rng.integers(0, 256, (D, blocks, 16), dtype=np.uint8)
    d = (0.02 * rng.uniform(0.5, 1.5, (D, blocks))).astype(np.float16)
    key = np.concatenate([d.view(np.uint8).reshape(D, blocks, 2), codes], axis=2).reshape(D, blocks * KEY_BLOCK)
    value = bf16_bits(np.random.default_rng(VALUE_SEED).standard_normal((N, N)).astype(F32) * F32(0.05))
    w_nk, w_nq, w_nc = ((1 + 0.1 * rng.standard_normal(D)).astype(F32) for _ in range(3))
    conv1d = (0.3 * rng.standard_normal(KERN * D)).astype(np.float16)
    return _with_activations(rng, 0.2, key=key, value=value, w_nk=w_nk, w_nq=w_nq, w_nc=w_nc, conv1d=conv1d)


def committed():
    """``FIXTURE``'s committed slices, ``{key: bytes}``, and the writer's metadata (``entries``: each slice's tensor and
    name space, where the pack's tables put it and how many of its bytes are kept)."""
    infos, meta = read_header(FIXTURE)
    data = FIXTURE.read_bytes()
    return {k: data[i.start:i.end] for k, i in infos.items()}, json.loads(meta["fixture"])


def full_ranges():
    """The two full ranges from $BASALT_PLE_FIXTURE, ``{name space: bytes}``. Unset, the caller skips with a reason that
    names the variable; set, a file missing or of the wrong size fails."""
    import pytest           # imported here: make_basalt_ple_fixture.py imports this module and needs no pytest

    root = os.environ.get("BASALT_PLE_FIXTURE")
    if not root:
        pytest.skip("set BASALT_PLE_FIXTURE to the directory `make_basalt_ple_fixture.py --full DIR` filled with the "
                    "n-gram block's two real-weight ranges")
    out = {}
    for space, (name, _, size) in RANGES.items():
        path = Path(root) / name
        got = path.stat().st_size if path.is_file() else None
        if got != size:
            pytest.fail(f"$BASALT_PLE_FIXTURE/{name} is {'missing' if got is None else f'{got} bytes'}, not {size} "
                        "bytes: fetch it again with make_basalt_ple_fixture.py --full DIR")
        out[space] = path.read_bytes()
    return out


def real_tensors():
    """``TENSORS`` cut from the full ranges: ``{"gguf/<name>": bytes}`` for all six, ``{"dense/<name>": bytes}`` for
    the five with a dense row."""
    ranges, out = full_ranges(), {}
    for name, t in TENSORS.items():
        for space, at in (("gguf", t.gguf), ("dense", t.dense)):
            if at is not None:
                at -= RANGES[space][1]
                out[f"{space}/{name}"] = ranges[space][at:at + t.nbytes]
    return out


def real(seed=REAL_SEED):
    """T§6.3's real fixture, as the module docstring says."""
    w = real_tensors()

    def dense(name, dtype):
        return np.frombuffer(w[f"dense/blk.1.ple_{name}.weight"], dtype)

    key = np.frombuffer(w["gguf/blk.1.ple_key.weight"], np.uint8).reshape(D, -1)
    return _with_activations(np.random.default_rng(seed), 0.0158, key=key,
                             value=dense("value", np.uint16).reshape(N, N), w_nk=dense("norm_key", F32),
                             w_nq=dense("norm_query", F32), w_nc=dense("norm_conv", F32),
                             conv1d=dense("conv1d", np.float16))
