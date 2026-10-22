# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp, src/kernels/ple_win_parity.cpp and
# src/kernels/gr_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's history
# regression with its guards (:177-228), key-row check (:617-637), block check with its seven stages and its log line
# (:740-783), native key projection check with its host norm, metric and pass criterion (:892-912) and its guards
# (:930-937), separate-exports and in-place result check with its log line (:949-969), host reference check's
# per-stage pass criterion and log line (:1114-1122) and conv cancellation metric (:1139-1165), ple_win_parity's
# projected-form input scales (:47-55), and gr_parity's scalar activation contract's degenerate norm input and the
# exactness it relies on (:167-236).
"""The n-gram (PLE) block's Metal port on the GPU, kernel by kernel (kernels/common/basalt_gr_norm.metal and
basalt_ple.metal) and as the per-row chain (``basalt_ple_gpu.chain``: main's q8_1, mmvq and mmvf, then those kernels,
row by row, as the window op dispatches them). Each norm call runs at its call sites' shape, 4 rows of 2560 with 4
gamma rows and eps = f32(1e-6) (native_ple_postops.cu:352-353, :358), unless stated.

- T-NORM ``test_grouped_norm_float64`` (spec T§7.2): basalt_gr_norm against R-struct's float64 ``gnorm``
  (tests/basalt_ple_ref.py), criterion quoted:
  ``const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;`` (ple_parity.cpp:910), rel =
  rel_l1(reference, GPU) over one call's 4 x 2560 outputs and the non-finite count over the norm's own output (basalt
  counted it over the block's result). The calls: N(0, σ) rows for σ in {1e-3, 1e-2, 1, 1e2} (at 1e-3 the mean square
  is about eps) and a zero row, whose output is 0, each with weights f32(1 + 0.1 N(0, 1)); and "the GPU's raw key,
  hidden and the GPU's gated rows from the 12-row C3 fixtures, synthetic and real" (``C3_NORMS``), one call per row
  at each of the norm's call sites: the chain's raw key with w_nk, the fixture's hidden with w_nq and the chain's
  gated row with w_nc (``block``).
- T-NORM ``test_grouped_norm_bitwise_emu``: bit for bit against R-emu, ``basalt_emu.gr_norm`` (the rounding sequence of
  docs/research/basalt-nvcc/ple.md section 1) with the GPU's rsqrtf_ftz (MUFU.RSQ and fast::rsqrt have no bit-exact
  host model), on the calls above plus a row whose squares all lie below 2^-126, each of which .FTZ turns into +0; and
  on G1a, gr_parity's degenerate input (gr_parity.cpp:169-201: GrShapes{2, 1, 2}, so one row of 2 columns and the
  256-thread block; R = 1, eps = 0, gamma in {1.0f, 1.00390625f}). There, "At R=1 and eps=0, xn=gamma exactly"
  (:169), which basalt's exact checks rely on (:233-236): the scale is rsqrtf(1) and the output equals gamma bit for
  bit, provided the GPU's fast::rsqrt(1.0f) is exactly 1.0f. A failure there is a departure from basalt, to record,
  not to tune away.
- T-POST ``test_postops_bitwise_emu`` (spec T§7.3): native_ple_postops (native_ple_postops.cu:352-359) one token at a
  time, launch by launch (the key and query norms, basalt_ple_gate, basalt_ple_broadcast, the conv norm and
  basalt_ple_conv), each launch bit for bit against R-emu (``basalt_emu.gr_norm``, ``ple_gate``, ``ple_broadcast`` and
  ``ple_conv``: ple.md sections 1-4, with the GPU's rsqrtf_ftz, sqrtf_ftz and expf_ftz evaluated on the emulation's
  own arguments) fed the GPU's outputs of the launches before it, so that each comparison is its own kernel's: T§7.3
  "compares all six stages of every row of every fixture above with R-emu, byte for byte" (the query norm is compared
  too). The fixtures: T§7.3's three 12-row fixtures, each row in order from the GPU's history before it
  (``postops_rows``): the projected form at W1's scales (``projected``), its history the initial one, then each row's
  GPU normalized row shifted in; C3's fixtures' GPU projections, synthetic and real, each row's raw key and value the
  chain's (``block``) and its history the initial one, then the chain's snapshot of the row before; and T§7.3's
  edge cases. The gate's are streams of a direct basalt_ple_gate call
  (``GATE_EDGES``): s = +0 and s = -0 (sign 0, so the gate is exactly the sigmoid of 0, 0.5), 0 < |s| < 1e-6 (the
  clamp: the same gate as |s| = 1e-6f), |s| = 1e-6f, and |s| = 2e4 (the sigmoid saturates at 1, and at 0 where
  fast::exp2 overflows). The others are the projected form's first row with inputs planted (``EDGE_ROWS``): a SiLU
  whose fast::exp2 overflows (act = acc * rcp(inf) = -0), subnormal key, value, hidden and history inputs (read as
  zero under the flush) beside fp16-subnormal taps (widened exactly), and gated products below 2^-126 (flushed).
- T-POST ``test_postops_float64`` (spec T§7.3): those three 12-row fixtures' GPU stages, row by row, against
  R-contract (``r_contract``) fed the row's own inputs, the raw key, the value, hidden and the GPU's history before
  the row, so that "each row's comparison therefore isolates that row's arithmetic" (propagation across rows is C3's),
  criterion quoted: ``const bool ok = le(rel, 1e-4) && nf == 0;`` (ple_parity.cpp:1119), rel and nf =
  rel_l1(R-contract, GPU) per row and stage, on the six stages key (normed), gate, gated, normalized, conv_out and
  result (the value is an input), logged in :1120-1121's format; and "Each fixture asserts min |s[c]| ≥ 0.1 on
  R-contract".
- H1, T-HIST ``test_history_advance_12_steps`` (spec T§7.4): ple_parity's history regression (ple_parity.cpp:177-214)
  as written, on basalt_ple_advance: the initial history ``expected[c * rows + r] = -float(c * 16 + r + 1)``
  (row-fastest) and 12 tokens of ``normalized[c] = float((token + 1) * 1000000 + c)``, one dispatch built once and run
  once per token (basalt captures its advance once and replays it, :195-207), each run followed by the host shift
  (:203-204) and the criterion, quoted: ``if (std::memcmp(actual.data(), expected.data(), count * sizeof(float)) != 0)
  {`` (:210), then ``++bad;`` (:212). Its refusals (H2) are the emitter's, its guards (H3) X-GUARD's.
- X-H ``test_history_moves_bits`` (spec T§8): H1's run with fp32 subnormals (0x00000001, 0x807fffff), -0.0, +-inf, a
  quiet NaN with a payload (0x7fc00001) and a signalling NaN (0x7f800001) planted in every token's normalized row, as
  words, so that they pass through every history row: "every pattern survives the history's moves bit for bit
  (memcmp, as H1)", the advance, then basalt_ple_snapshot of the GPU's histories, then basalt_ple_restore from the
  GPU's snapshots into a history that differs from each of them at every word.
- T-COPY ``test_snapshot_copy_exact`` and ``test_commit_restore_exact`` (spec T§7.5), byte for byte (the precedent
  ``if (ra != rb) {``, dec_commit_parity.cpp:177), every guard byte intact: the snapshot into each slot equals its
  source and leaves the other slots' 0xA5; the restore by the commit record's word 1 (``commit_record``: every
  (T, n_keep) with n_keep - 1 = word 1 has the same record, so words 0 to 7 cover T = 1..8 and n_keep = 1..T) equals
  that slot, a negative word leaves the history unchanged, and a word above 7 restores slot 7 (the port's bound, plan
  D5). The words are slot-tagged random words (``copy_words``), so that a missed, stray or misplaced store shows.
- C8 ``test_key_projection_into_grouped_norm`` (ple_parity.cpp:892-912, spec T§6.4), on every row of C3's fixtures
  (basalt runs two): the raw key of basalt's native_q2_0_f32, native_quantize_q8_1 then native_mmvq type 42 at one
  column (basalt_gpu.q8_1 and mmvq), is the chain's own bit for bit, and so are its q8_1 image and its value
  (basalt_gpu.mmvf): the chain composes main's kernels. Then the chain's key norm against the host norm of that raw
  key (:899-904, ``host_norm``), criterion quoted:
  ``const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;`` (:910), rel = rel_l1(host norm,
  GPU key norm), the non-finite count over the row's result.
- C3 ``test_block_vs_float64_graph[synthetic]``, ``[real]`` (ple_parity.cpp:740-783, spec T§6.4): the chain over
  T§6.3's fixtures, 12 rows from one initial history (the key's Q8_1 rounding the only contract between them), against
  R-struct, rel = rel_l1(R-struct, GPU) per row and stage (the capture's seven slices), criterion quoted:
  ``if (last && !le(rel, 1e-2)) ++bad;`` (:782), on result. The other six stages are logged in :771-772's format,
  with the row's block share (T§6.3 rule 3) and C16's metric (:1139-1165), never gated: the conv_out error against
  its terms' magnitude, sum |R-struct - GPU| / sum_c sum_k |w v| and sum |R-struct| / sum_c sum_k |w v|, the terms
  w v from R-struct's taps and padded rows. Every stage's non-finite count must be 0 (spec T§8, X-NF rule 1).
- C1 ``test_real_key_row0_decodes_like_ggml`` (ple_parity.cpp:617-637, spec T§6.4), real weights only: the committed
  row 0 of the key (720 B of type-42 blocks) decoded by basalt_dequant_f32 (basalt_gpu.dequant) against ggml's
  dequantize_row_q2_0 of the same bytes (tests/contract/gguf_reference.py's ``q2_0``), criterion quoted:
  ``if (!le(worst, 0.0) || nf) ++bad;`` (:636), worst the NaN-safe largest |GPU - ggml| and nf the non-finite pairs;
  the decoded row finite with 1e-4 < rms < 10 (weights, not padding: tests/contract/test_basalt_pack.py's
  ``test_real_rows_match_the_reference``); and row 0 of the key the real fixture binds (gguf/blk.1.ple_key.weight, cut
  from its full range at T§5's offset) equal to the committed slice, byte for byte.
- X-INPLACE ``test_compact_build_equals_full_exports`` (ple_parity.cpp:949-969, spec T§8), on C3's fixtures, every row:
  the chain as the handler binds it (``block``) against two other bindings of the same 12 rows, criterion quoted:
  ``const bool equal = std::memcmp(compact_norm.data(), dev_norm.data(), hcd * sizeof(float)) == 0 &&
  std::memcmp(compact_result.data(), dev_res.data(), hcd * sizeof(float)) == 0;`` (:963-964), then
  ``if (!equal) ++bad;`` (:966), with :965's log line. (1) Query and normalized as one buffer against two buffers
  (``split_query``): gate, gated, normalized and result equal. (2) Result bound onto hidden, basalt's in-place alias
  (``inplace``), against out of place: normalized and result equal. The conv store is kept (plan D3), so there is no
  compact build to compare.
- X-GUARD ``test_store_bounds_guards`` (H3, ple_parity.cpp:221-228; C10, :930-937; spec T§8): the chain's guard run
  (``basalt_ple_gpu.chain(..., guard=True)``: every binding of every kernel, the rows, hidden, weights, history,
  snapshots, outputs and scratch, between 64 B of 0xA5) of a T-row window, T = 1..8, of the synthetic fixture over
  buffers of T_MAX rows and slots, as the T = 8 program's row I/O and its snapshots, then the commit's restore of
  n_keep = 1 into the window's history (``basalt_ple_gpu.restore``, its three bindings guarded too). Criteria quoted:
  ``if (!guard_ok(history_storage, count) || !guard_ok(norm_storage, channels)) ++bad;`` (:228) and
  ``if (!std::all_of(ends.begin(), ends.end(), [](uint8_t x) { return x == 0xa5; })) ++bad;`` (:936), for every
  buffer; "rows from T to T_MAX − 1 of every multi-row output still hold their sentinel" (T§8), 0xFF in normalized,
  result and the snapshots; and rows below T equal the exact-size run's (``block``) byte for byte, so the guards and
  offsets change nothing else. The exact-size half is every test's validated pass (plan §4.9)."""

import functools

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_ple_ref import (D, DIL, EPS, H, HIST, KERN, KEY_BLOCK, N, ROWS, STAGES, T_MAX, Fixture,
                                  commit_record, committed, gnorm, h_oracle, host_norm, le, nonfinite, r_contract,
                                  r_struct, real, real_tensors, rel_l1, synthetic, taps)
from tests.contract.gguf_reference import q2_0
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels import basalt_ple_gpu as ple_gpu

F32, F64 = np.float32, np.float64
SIGMAS = (1e-3, 1e-2, 1.0, 1e2)
NORM_SEED = 910     # default_rng(NORM_SEED) draws, in order: each σ's row then its weights, the zero row's weights,
                    # then the tiny row and its weights
TINY = "squares below 2^-126"


def norm_calls():
    """The standalone norm calls, ``{id: (x, w)}``, each a D-vector (4 rows of 2560) with its own weights: spec T§7.2's
    N(0, σ) rows and zero row, then the tiny row, f32(N(0, 1) 2^-70): normal values whose squares all lie below
    2^-126 (asserted)."""
    rng = np.random.default_rng(NORM_SEED)

    def weights():
        return (1 + 0.1 * rng.standard_normal(D)).astype(F32)

    calls = {}
    for s in SIGMAS:
        x = (s * rng.standard_normal(D)).astype(F32)
        calls[f"sigma={s:g}"] = x, weights()
    calls["zero"] = np.zeros(D, F32), weights()
    tiny = (rng.standard_normal(D) * 2.0 ** -70).astype(F32)
    normal, small = (np.abs(tiny) >= 2.0 ** -126).all(), (np.square(tiny.astype(F64)) < 2.0 ** -127).all()
    assert normal and small, "the tiny row: normal values, squares below 2^-127"
    calls[TINY] = tiny, weights()
    return calls


CALLS = norm_calls()
# G1a: (x, gamma, n_rows, n_cols, gamma_rows, eps) for gr_parity's n_embd = 2, hc = 1 norm at R = 1, eps = 0
G1A = {f"G1a gamma={g}": (np.ones(2, F32), np.full(2, g, F32), 1, 2, 1, 0.0) for g in (1.0, 1.00390625)}
BITWISE = {**{k: (x, w, H, N, H, EPS) for k, (x, w) in CALLS.items()}, **G1A}
C3_NORMS = {  # call: (C3's fixture, the norm's call site), one call per row of the fixture (spec T§7.2)
    f"{name} {site}": (name, site) for name in ("synthetic", "real") for site in ("raw key", "hidden", "gated")}


def bits(a):
    return np.ascontiguousarray(a, F32).view(np.uint32)


def norm_rows(call):
    """``C3_NORMS[call]``'s 12 norm calls, ``[(x, w)]``: the rows of C3's fixture (``block``) at the call site, the
    chain's raw key with w_nk (the key norm's), the fixture's hidden with w_nq (the query norm's) or the chain's gated
    row with w_nc (the conv norm's)."""
    name, site = C3_NORMS[call]
    fx, _, out = block(name)
    x, w = {"raw key": (out["key"], fx.w_nk), "hidden": (fx.hidden, fx.w_nq), "gated": (out["gated"], fx.w_nc)}[site]
    return [(row, w) for row in x]


@pytest.mark.parametrize("call", [k for k in CALLS if k != TINY] + list(C3_NORMS))
def test_grouped_norm_float64(call):
    calls, bad = [CALLS[call]] if call in CALLS else norm_rows(call), []
    for i, (x, w) in enumerate(calls):
        actual = ple_gpu.gr_norm(x, w, H, N, H, EPS)
        rel = rel_l1(gnorm(x, w), actual)[0]
        nf = nonfinite(actual)
        print(f"grouped norm [{call}]{f' row {i}' if call in C3_NORMS else ''}: rel {rel:.3e}, nf {nf}")
        if not (le(rel, 2e-6) and nf == 0):
            bad.append((i, float(rel), nf))
    assert not bad, (
        "`const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;` (ple_parity.cpp:910), rel = "
        "rel_l1(R-struct's gnorm, the GPU) over the call's 4 x 2560 outputs, nf over the norm's output: (call, rel, "
        f"nf) {bad}")
    if call == "zero":
        assert (actual == 0).all(), f"spec T§7.2, a zero row: \"the output is 0\": {actual[actual != 0][:8]}"


@pytest.mark.parametrize("call", list(BITWISE) + list(C3_NORMS))
def test_grouped_norm_bitwise_emu(call):
    calls = [BITWISE[call]] if call in BITWISE else [(x, w, H, N, H, EPS) for x, w in norm_rows(call)]
    for i, (x, w, *shape) in enumerate(calls):
        got = ple_gpu.gr_norm(x, w, *shape)
        want = emu.gr_norm(x, w, *shape, ple_gpu.rsqrtf_ftz)
        diff = np.flatnonzero(bits(got) != bits(want))
        assert diff.size == 0, (
            f"T-NORM, bit for bit against basalt_emu.gr_norm (ple.md section 1): call {i}: {diff.size} of {got.size} "
            f"outputs differ; first at {diff[:4].tolist()}: GPU {[hex(v) for v in bits(got).ravel()[diff[:4]]]}, "
            f"R-emu {[hex(v) for v in bits(want).ravel()[diff[:4]]]}")
    if call in G1A:
        one = hex(bits(ple_gpu.rsqrtf_ftz(np.ones(1, F32)))[0])
        out, gamma = [hex(v) for v in bits(got).ravel()], [hex(v) for v in bits(w)]
        print(f"{call}: output {out}, gamma {gamma}, the GPU's rsqrtf_ftz(1.0f) {one}")
        assert out == gamma, (
            f"G1a, \"At R=1 and eps=0, xn=gamma exactly\" (gr_parity.cpp:169): output {out} != gamma {gamma} with the "
            f"GPU's rsqrtf_ftz(1.0f) = {one}; a departure from basalt (gr_parity.cpp:233-236), to record, not to tune "
            "away (spec T§7.2)")


# ---- T-POST: native_ple_postops kernel by kernel against R-emu, row by row against R-contract (spec T§7.3) ---------
PROJECTED_SEED = 4      # default_rng(PROJECTED_SEED) draws, in W1's order without its hyper-connection inputs
                        # (ple_win_parity.cpp:50-55): conv1d, hidden's unit draw z, the raw keys, the values, the
                        # history, then w_nk, w_nq and w_nc; the first seed from 0 for which min |s| >= 0.1 holds
                        # (min |s| 0.174)
EDGE_SEED = 1119        # default_rng(EDGE_SEED) draws the planted inputs of the edge row being built
SCALE = np.frombuffer(kernels.basalt_ple_gate_params(), F32)[0]   # the gate's scale as the kernel reads it, 0x3CA1E89B
CLAMP_P = np.uint32(0x38543784).view(F32)                         # fl(SCALE * CLAMP_P) is 1e-6f, the clamp (asserted)
GATE_EDGES = {  # stream: (key[0], query[0]), every other entry 0, so the stream's sum is fl(key[0] * query[0]) exactly
    "s = +0": (0.0, 1.0),
    "s = -0": (-2.0 ** -61, 2.0 ** -61),      # fl(SCALE * -2^-122) is subnormal: .FTZ makes s -0
    "0 < s < 1e-6": (2.0 ** -20, 1.0),
    "-1e-6 < s < 0": (-2.0 ** -20, 1.0),
    "s = 1e-6f": (CLAMP_P, 1.0),
    "s = -1e-6f": (-CLAMP_P, 1.0),
    "s = 2e4": (2.0 ** 10, 2.0 ** 10),        # mag 144: ex2(-208) is +0 and the gate rcp(1) = 1
    "s = -2e4": (-2.0 ** 10, 2.0 ** 10),      # ex2(+208) overflows: the gate is rcp(inf) = +0
}
# The SiLU row's sums, channel c = 0, 1, ...: -acc * log2(e) passes 128 below acc = -88.72 (fast::exp2 overflows)
# and 1 + ex2 passes 2^126 below -87.34 (the reciprocal flushes); either way act = acc * +0 = -0.
SILU_ACC = (-np.inf, -3e38, -1e4, -100.0, -89.0, -88.75, -88.7, -88.5, -88.0, -87.5, -87.0, -80.0, -20.0, 20.0, 80.0,
            88.0, 89.0, 100.0, 3e38, np.inf)
EDGE_ROWS = ("SiLU exp2 overflow", "subnormal inputs", "gated below 2^-126")


@functools.lru_cache(maxsize=1)
def projected():
    """T§7.3's projected form at W1's scales (ple_win_parity.cpp:47-55): per row a raw key f32(0.2 N(0,1)) (D values)
    and a value f32(0.5 N(0,1)) (N), the initial history f32(N(0,1)) (9 x D, row r oldest first), the three norms'
    f32(1 + 0.1 N(0,1)) and conv1d f16(0.3 N(0,1)); hidden = f32(sigma_h z) by T§6.3's rule 3 (R-contract with the
    unit hidden f32(z), then sigma_h = mean |gated + conv_out| / mean |f32(z)|). Asserts T§7.3's min |s| >= 0.1 on
    R-contract. Returns (fx, key, value): a basalt_ple_ref Fixture without key, value and emb (the projections are
    inputs here), the raw keys (12, D) and the values (12, N)."""
    rng = np.random.default_rng(PROJECTED_SEED)
    conv1d = (0.3 * rng.standard_normal(KERN * D)).astype(np.float16)
    z = rng.standard_normal((ROWS, D))
    key = (0.2 * rng.standard_normal((ROWS, D))).astype(F32)
    value = (0.5 * rng.standard_normal((ROWS, N))).astype(F32)
    hist = rng.standard_normal((HIST, D)).astype(F32)
    w_nk, w_nq, w_nc = ((1 + 0.1 * rng.standard_normal(D)).astype(F32) for _ in range(3))
    fx = Fixture(key=None, value=None, w_nk=w_nk, w_nq=w_nq, w_nc=w_nc, conv1d=conv1d, emb=None, hist=hist,
                 hidden=z.astype(F32))
    ref = r_contract(fx, key, value)
    sigma = np.abs(ref.gated + ref.conv_out).mean() / np.abs(fx.hidden.astype(F64)).mean()
    fx = fx._replace(hidden=(sigma * z).astype(F32))
    min_s = float(np.abs(r_contract(fx, key, value).s).min())
    assert min_s >= 0.1, f"T§7.3, \"Each fixture asserts min |s[c]| ≥ 0.1 on R-contract\": {min_s!r}"
    return fx, key, value


def postops_row(key, value, hidden, hist, fx):
    """One token of native_ple_postops (native_ple_postops.cu:352-359) on the GPU, launch by launch, with ``fx``'s
    norms and conv1d, and R-emu on the same inputs, each emulated launch fed the GPU's outputs of the launches before
    it: ``{stage: (GPU, R-emu)}`` for the key norm, the query norm, the gate, the broadcast (gated), the conv norm
    (normalized), and the convolution's conv_out and result. ``hist`` is in basalt's layout (row r of channel c at
    c * 9 + r)."""
    def norm(x, w):
        return ple_gpu.gr_norm(x, w, H, N, H, EPS).ravel(), emu.gr_norm(x, w, H, N, H, EPS, ple_gpu.rsqrtf_ftz).ravel()

    out = {"key": norm(key, fx.w_nk), "query": norm(hidden, fx.w_nq)}
    k, q = out["key"][0], out["query"][0]
    out["gate"] = ple_gpu.gate(k, q), emu.ple_gate(k, q, SCALE, ple_gpu.sqrtf_ftz, gpu.expf_ftz)
    out["gated"] = ple_gpu.broadcast(value, out["gate"][0]), emu.ple_broadcast(value, out["gate"][0])
    gated = out["gated"][0]
    out["normalized"] = norm(gated, fx.w_nc)
    nm = out["normalized"][0]
    got = ple_gpu.conv(hist, nm, fx.conv1d, hidden, gated)
    out["conv_out"], out["result"] = zip(got, emu.ple_conv(hist, nm, fx.conv1d, hidden, gated, gpu.expf_ftz))
    return out


POSTOPS_ROWS = ("projected", "synthetic", "real")      # T§7.3's 12-row fixtures: the projected form, C3's two


@functools.lru_cache(maxsize=None)
def postops_rows(name):
    """T§7.3's 12-row fixture ``name`` on ``postops_row``, its rows in order, each from the GPU's history before it,
    once per process: (fx, rows), row t = (inputs, stages), the row's inputs (raw key, value, hidden, history) and
    postops_row's stages. The projected form's raw keys and values are its own, its history the initial one, then
    each row's GPU normalized row shifted in (basalt's advance moves words, so the host's shift, H-ORACLE's, has its
    bits). C3's fixtures' raw keys and values, synthetic and real, are the chain's projections of the row (``block``),
    their history the initial one, then "the GPU's snapshot of row t − 1", the chain's."""
    if name == "projected":
        (fx, key, value), snap = projected(), None
    else:
        fx, _, out = block(name)
        key, value, snap = out["key"], out["value"], out["snap"]
    hist, rows = np.ascontiguousarray(fx.hist.T).ravel(), []
    for t in range(ROWS):
        inputs = key[t], value[t], fx.hidden[t], hist
        rows.append((inputs, postops_row(*inputs, fx)))
        hist = h_oracle(hist, rows[-1][1]["normalized"][0][None], 1) if snap is None else snap[t]
    return fx, rows


def postops_stages(name):
    """``postops_rows(name)``'s stages, row by row."""
    return [stages for _, stages in postops_rows(name)[1]]


def edge_row(case):
    """The projected form's first row, from the initial history, with ``case``'s inputs planted (``EDGE_ROWS``)."""
    fx, key, value = projected()
    rng = np.random.default_rng(EDGE_SEED)
    k, v, h, taps = key[0].copy(), value[0].copy(), fx.hidden[0].copy(), fx.conv1d.copy()
    hist = np.ascontiguousarray(fx.hist.T).ravel()          # a copy
    if case == "SiLU exp2 overflow":                        # channel c's taps (1, 0, 0, 0): its sum is history row 0
        c = np.arange(len(SILU_ACC))
        taps.reshape(D, KERN)[c] = (1.0, 0.0, 0.0, 0.0)
        hist.reshape(D, HIST)[c, 0] = SILU_ACC
    elif case == "subnormal inputs":                        # every 7th input a float32 subnormal, every 5th tap an
        for a in (k, v, h, hist):                           # fp16 subnormal, of either sign
            n = a[::7].size
            a[::7] = (rng.integers(1, 1 << 23, n, dtype=np.uint32) | rng.integers(0, 2, n, dtype=np.uint32) << 31
                      ).view(F32)
        n = taps[::5].size
        taps[::5] = (rng.integers(1, 1 << 10, n, dtype=np.uint16) | rng.integers(0, 2, n, dtype=np.uint16) << 15
                     ).view(np.float16)
    elif case == "gated below 2^-126":                      # |value| in [2^-126, 2^-123), so |value * gate|
                                                            # straddles 2^-126
        v = (rng.choice((-1.0, 1.0), N) * rng.uniform(1, 8, N) * 2.0 ** -126).astype(F32)
    return [postops_row(k, v, h, hist, fx._replace(conv1d=taps))]


def gate_edges():
    """``GATE_EDGES``' streams through basalt_ple_gate, H streams a call (basalt's launch), and R-emu's ple_gate."""
    keys, queries = np.zeros((len(GATE_EDGES), N), F32), np.zeros((len(GATE_EDGES), N), F32)
    keys[:, 0], queries[:, 0] = np.array(list(GATE_EDGES.values()), F32).T
    return [{"gate": (ple_gpu.gate(keys[i:i + H], queries[i:i + H]),
                      emu.ple_gate(keys[i:i + H], queries[i:i + H], SCALE, ple_gpu.sqrtf_ftz, gpu.expf_ftz))}
            for i in range(0, len(GATE_EDGES), H)]


POSTOPS = {**{name: functools.partial(postops_stages, name) for name in POSTOPS_ROWS}, "gate edges": gate_edges,
           **{case: functools.partial(edge_row, case) for case in EDGE_ROWS}}


@pytest.mark.parametrize("case", list(POSTOPS))
def test_postops_bitwise_emu(case):
    rows = POSTOPS[case]()
    for t, stages in enumerate(rows):
        for stage, (got, want) in stages.items():
            diff = np.flatnonzero(bits(got) != bits(want))
            assert diff.size == 0, (
                "T-POST, \"compares all six stages of every row of every fixture above with R-emu, byte for byte\" "
                f"(spec T§7.3): {case}, row {t}, {stage}: {diff.size} of {got.size} values differ; first at "
                f"{diff[:4].tolist()}: GPU {[hex(v) for v in bits(got)[diff[:4]]]}, "
                f"R-emu {[hex(v) for v in bits(want)[diff[:4]]]}")
    if case == "gate edges":
        assert F32(SCALE * CLAMP_P) == F32(1e-6), "the clamp stream's s = fl(SCALE * CLAMP_P) must be 1e-6f"
        g = dict(zip(GATE_EDGES, (int(b) for b in bits(np.concatenate([r["gate"][0] for r in rows])))))
        print("gate edges:", {name: hex(b) for name, b in g.items()})
        assert g["s = +0"] == g["s = -0"] == 0x3F000000, (
            f"T§7.3, s = +0 and -0: \"sign 0, gate exactly the sigmoid of 0\": {hex(g['s = +0'])}, {hex(g['s = -0'])}")
        assert g["0 < s < 1e-6"] == g["s = 1e-6f"] and g["-1e-6 < s < 0"] == g["s = -1e-6f"], (
            "T§7.3, 0 < |s| < 1e-6: the clamp gives the gate of |s| = 1e-6f, sign for sign")
        assert g["s = 2e4"] == 0x3F800000 and g["s = -2e4"] == 0, (
            "T§7.3, large |s|: the sigmoid saturates at 1, and at +0 where fast::exp2 overflows: "
            f"{hex(g['s = 2e4'])}, {hex(g['s = -2e4'])}")
    elif case == "SiLU exp2 overflow":
        acc = np.array(SILU_ACC)
        act = bits(rows[0]["conv_out"][0][:len(acc)])
        under = np.isfinite(acc) & (acc <= -87.5)
        assert (act[under] == 0x80000000).all(), (
            "T§7.3, \"the SiLU where -acc · log2(e) overflows fast::exp2, so acc / inf\" (and where 1 + ex2 passes "
            f"2^126, so the reciprocal flushes): act = -0 for acc {acc[under].tolist()}: "
            f"{[hex(b) for b in act[under]]}")
    elif case == "gated below 2^-126":
        flushed = int(np.count_nonzero(rows[0]["gated"][0] == 0))
        assert 0 < flushed < D, (
            f"T§7.3, gated products below 2^-126 are flushed and the others kept: {flushed} of {D} are 0")


POST_STAGES = tuple(s for s in STAGES if s != "value")     # T§7.3's six stages: the value is an input


@pytest.mark.parametrize("name", POSTOPS_ROWS)
def test_postops_float64(name):
    fx, rows = postops_rows(name)
    bad, min_s = [], np.inf
    for t, ((key, value, hidden, hist), stages) in enumerate(rows):
        ref = r_contract(fx._replace(hist=hist.reshape(D, HIST).T, hidden=hidden[None]), key[None], value[None])
        min_s = min(min_s, float(np.abs(ref.s).min()))
        for stage in POST_STAGES:
            rel, _, nf = rel_l1(getattr(ref, stage)[0], stages[stage][0])
            ok = le(rel, 1e-4) and nf == 0
            print(f"  postops [{name}] against R-contract, token {t} {stage:<12} rel {rel:.3e} (non-finite {nf})"
                  f"{'' if ok else '   *** FAIL ***'}")
            if not ok:
                bad.append((t, stage, float(rel), nf))
    assert min_s >= 0.1, f"T§7.3, \"Each fixture asserts min |s[c]| ≥ 0.1 on R-contract\": {name}: {min_s!r}"
    assert not bad, (
        "`const bool ok = le(rel, 1e-4) && nf == 0;` (ple_parity.cpp:1119), then `if (!ok) ++bad;` (:1122), rel and "
        "nf = rel_l1(R-contract, the GPU) per row and stage, R-contract fed the row's raw key, value, hidden and the "
        f"GPU's history before the row (spec T§7.3): {name}: (row, stage, rel, non-finite) {bad}")


# ---- T-HIST, X-H and T-COPY: the history's moves (spec T§7.4-7.5, T§8) ---------------------------------------------
HS = HIST * D           # a history's words
TOKENS = 12             # ple_parity.cpp:200
PATTERNS = np.array([0x00000001, 0x807FFFFF, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001], np.uint32)
PLANT = 37              # X-H plants PATTERNS, in turn, at every PLANT-th channel from the token's index
COPY_SEED = 177         # default_rng(COPY_SEED) draws T-COPY's random words


def h1_fixture():
    """H1's fixture as words (ple_parity.cpp:181-182, :200-202): the initial history
    ``expected[c * rows + r] = -float(c * 16 + r + 1)`` and the 12 tokens' normalized rows,
    ``normalized[c] = float((token + 1) * 1000000 + c); // distinct exactly represented integers``."""
    c, r = np.meshgrid(np.arange(D), np.arange(HIST), indexing="ij")
    hist = bits((-(c * 16 + r + 1)).astype(F32)).ravel()
    normalized = bits(((np.arange(TOKENS)[:, None] + 1) * 1000000 + np.arange(D)).astype(F32))
    return hist, normalized


def host_shift(expected, normalized):
    """ple_parity.cpp:203-204 for every channel, on words: rows 1 to 8 move to rows 0 to 7, and ``normalized`` enters
    row 8."""
    e = expected.reshape(D, HIST).copy()
    e[:, :-1] = expected.reshape(D, HIST)[:, 1:]
    e[:, -1] = normalized
    return e.ravel()


def test_history_advance_12_steps():
    hist, normalized = h1_fixture()
    actual = ple_gpu.advance(hist, normalized)[0]
    expected, bad = hist, []
    for token in range(TOKENS):
        expected = host_shift(expected, normalized[token])
        if actual[token].tobytes() != expected.tobytes():
            bad.append(token)
    assert not bad, (
        "H1, `if (std::memcmp(actual.data(), expected.data(), count * sizeof(float)) != 0) {` (ple_parity.cpp:210), "
        f"then `++bad;` (:212): history advance mismatch after token {bad}")


def test_history_moves_bits():
    hist, normalized = h1_fixture()
    for token in range(TOKENS):
        planted = np.arange(token, D, PLANT)
        normalized[token, planted] = PATTERNS[np.arange(planted.size) % PATTERNS.size]
    expected = [hist]
    for token in range(TOKENS):
        expected.append(host_shift(expected[-1], normalized[token]))
    expected = expected[1:]                       # expected[t]: the history after token t
    first = TOKENS - T_MAX                        # slot s takes the history after token first + s
    last = expected[-1].reshape(D, HIST)
    assert all(np.isin(PATTERNS, last[:, r]).all() for r in range(HIST)), "X-H: every pattern in every history row"
    assert all((hist != expected[first + s]).all() for s in range(T_MAX)), (
        "X-H: the restore's target history differs from every slot at every word")
    advanced = ple_gpu.advance(hist, normalized)[0]
    bad = [f"advance after token {t}" for t in range(TOKENS) if advanced[t].tobytes() != expected[t].tobytes()]
    slots = np.empty((T_MAX, HS), np.uint32)
    for s in range(T_MAX):
        slots[s] = ple_gpu.snapshot(advanced[first + s], s)[0][s]
        if slots[s].tobytes() != expected[first + s].tobytes():
            bad.append(f"snapshot into slot {s}")
    for s in range(T_MAX):
        if ple_gpu.restore(hist, slots, commit_record(s + 1, T_MAX, 0))[0].tobytes() != expected[first + s].tobytes():
            bad.append(f"restore of slot {s}")
    assert not bad, (
        "X-H, \"every pattern survives the history's moves bit for bit (memcmp, as H1). Under forms (a)/(b) the moves "
        f"are the advance, the snapshot and the commit restore\" (spec T§8): mismatches at {bad}")


@functools.lru_cache(maxsize=1)
def copy_words():
    """T-COPY's words, (T_MAX + 1, HS) uint32: row s is ``w ^ (s << 24)`` for one draw ``w`` of HS random words
    (default_rng(COPY_SEED)), rows 0 to 7 the snapshots and row 8 the history. Each row differs from every other at
    every word, and no word is a fill of the harness (0xFFFFFFFF, 0xA5A5A5A5; asserted), so that a missed, stray or
    misplaced store shows."""
    w = np.random.default_rng(COPY_SEED).integers(0, 2 ** 32, HS, dtype=np.uint32)
    words = w[None] ^ (np.arange(T_MAX + 1, dtype=np.uint32)[:, None] << np.uint32(24))
    assert not np.isin(words, np.array([0xFFFFFFFF, 0xA5A5A5A5], np.uint32)).any(), "T-COPY: a fill word in the data"
    return words


@pytest.mark.parametrize("slot", range(T_MAX))
def test_snapshot_copy_exact(slot):
    src = copy_words()[T_MAX]
    snaps, guards = ple_gpu.snapshot(src, slot)
    equal = snaps[slot].tobytes() == src.tobytes()
    changed = [s for s in range(T_MAX) if s != slot and (snaps[s] != 0xA5A5A5A5).any()]
    assert equal and not changed and guards, (
        "T-COPY, \"for every row t of a T-row window, snapshot t equals the history after row t, byte for byte; the "
        "snapshot rows from T to T_MAX − 1 still hold their sentinel; the guard bytes around the buffer are "
        "untouched\" (spec T§7.5; `if (ra != rb) {`, dec_commit_parity.cpp:177): "
        f"slot {slot} equals its source: {equal}; other slots changed: {changed}; guards intact: {guards}")


RESTORES = {  # case: (the record's word 1, the slot the history must then equal, None: unchanged)
    **{f"word 1 = {i}": (i, i) for i in range(T_MAX)},
    **{f"word 1 = {i}": (i, None) for i in (-1, -5)},
    **{f"word 1 = {i}": (i, T_MAX - 1) for i in (T_MAX, 100)},
}


@pytest.mark.parametrize("case", list(RESTORES))
def test_commit_restore_exact(case):
    word1, slot = RESTORES[case]
    words = copy_words()
    if 0 <= word1 < T_MAX:
        commit = commit_record(word1 + 1, T_MAX, 0)       # the window's record (verify.cpp:2735-2737)
    else:                                                 # a record the host refuses (verify.cpp:2732); word 0 names
        commit = commit_record(3, T_MAX, 0)               # slot 3, so that a read of word 0 shows
        commit[1] = word1
    got, guards = ple_gpu.restore(words[T_MAX], words[:T_MAX], commit)
    want = words[T_MAX] if slot is None else words[slot]
    equal = got.tobytes() == want.tobytes()
    assert equal and guards, (
        "T-COPY, \"the history after the restore equals snapshot n_keep − 1 byte for byte; with index −1 the history "
        "is unchanged; ... guard bytes are untouched\" (spec T§7.5), and word 1 above 7 restores slot 7 (plan D5): "
        f"record {commit.tolist()}, the history equals {'itself' if slot is None else f'slot {slot}'}: {equal}; "
        f"guards intact: {guards}")


# ---- C8, C3 (with C16) and C1: the block on the per-row chain (spec T§6.4) -----------------------------------------
KEY = "gguf/blk.1.ple_key.weight"
FIXTURES = {"synthetic": synthetic, "real": real}
STAGE_OUT = {  # R-struct's slice: the chain's output for it, and basalt's name for the stage (ple_parity.cpp:752-758)
    "key": ("key_norm", "key        (grouped_norm of ple_key @ emb)"),
    "value": ("value", "value      (ple_value @ emb, BF16)"),
    "gate": ("gate", "gate       (signed sqrt, sigmoid)"),
    "gated": ("gated", "gated      (value broadcast * gate)"),
    "normalized": ("normalized", "normalized (grouped_norm(gated, ple_norm_conv))"),
    "conv_out": ("conv", "conv_out   (dilated conv, then SiLU)"),
    "result": ("result", "result     (hidden + gated + conv)"),
}


@functools.lru_cache(maxsize=None)
def block(name):
    """C3's fixture ``name`` (spec T§6.3: 12 rows from one initial history), its R-struct, and the chain's run of its
    12 rows (``basalt_ple_gpu.chain``), once per process: (fx, ref, out)."""
    fx = FIXTURES[name]()
    out = ple_gpu.chain(fx.emb, fx.hidden, np.ascontiguousarray(fx.hist.T).ravel(), fx)
    return fx, r_struct(fx), out


@pytest.mark.parametrize("name", list(FIXTURES))
def test_key_projection_into_grouped_norm(name):
    fx, _, out = block(name)
    differ, bad = [], []
    for t in range(ROWS):
        xq = gpu.q8_1(fx.emb[t])                                    # native_q2_0_f32 (ple_parity.cpp:896)
        raw = gpu.mmvq(42, fx.key, xq, N, D, 1)[0]
        main = {"q8_1": xq, "key": raw, "value": gpu.mmvf(fx.value, fx.emb[t], N, N)}
        differ += [(t, k) for k, a in main.items() if a.tobytes() != out[k][t].tobytes()]
        rel = rel_l1(host_norm(raw, fx.w_nk), out["key_norm"][t])[0]
        wired = le(rel, 2e-6) and nonfinite(out["result"][t]) == 0
        print(f"  native PLE key projection->existing norm input {t}: {'pass' if wired else 'FAIL'} (rel {rel:.3e})")
        if not wired:
            bad.append((t, rel, nonfinite(out["result"][t])))
    assert not differ, (
        "C8's raw key is basalt's native_q2_0_f32 (native_quantize_q8_1, then native_mmvq type 42; basalt_gpu.q8_1 and "
        "mmvq), and the chain's q8_1 image, raw key and value (basalt_gpu.mmvf) must be main's kernels' own, bit for "
        f"bit: (row, output) {differ} differ")
    assert not bad, (
        "`const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;` (ple_parity.cpp:910), then "
        "`if (!wired) ++bad;` (:912), rel = rel_l1(the host norm of the raw key (:899-904), the chain's key norm), the "
        f"non-finite count over the row's result: (row, rel, non-finite) {bad}")


@pytest.mark.parametrize("name", list(FIXTURES))
def test_block_vs_float64_graph(name):
    fx, ref, out = block(name)
    P = np.concatenate([np.asarray(fx.hist, F64), ref.normalized])     # R-struct's padded rows (spec T§6.2)
    w = taps(fx.conv1d)
    rels, over, nf_all = {stage: [] for stage in STAGES}, [], []
    for t in range(ROWS):
        for stage in STAGES:
            got, label = STAGE_OUT[stage]
            rel, mag, nf = rel_l1(getattr(ref, stage)[t], out[got][t])
            last = stage == "result"
            print(f"  token {t} {label:<46} rel {rel:.3e}   (mean |ref| {mag:.4f}, non-finite {nf})"
                  f"{'   *** over 1e-2 ***' if last and not le(rel, 1e-2) else ''}")
            rels[stage].append(rel)
            if last and not le(rel, 1e-2):
                over.append((t, rel))
            if nf:
                nf_all.append((t, stage, nf))
        share = np.abs(ref.gated[t] + ref.conv_out[t]).sum() / np.abs(ref.result[t]).sum()
        terms = sum(np.abs(w[:, k] * P[t + DIL * k]) for k in range(KERN)).sum()
        num, res = np.abs(ref.conv_out[t] - out["conv"][t]).sum(), np.abs(ref.conv_out[t]).sum()
        print(f"  token {t} block share {share:.4f}; conv_out against the TERM magnitude, not the result: rel "
              f"{num / terms:.3e}   (|result|/|terms| = {res / terms:.4f})")
    print(f"C3 [{name}]: worst rel per stage", {stage: f"{np.max(v):.3e}" for stage, v in rels.items()})
    assert not over and not nf_all, (
        "`if (last && !le(rel, 1e-2)) ++bad;` (ple_parity.cpp:782), rel = rel_l1(R-struct, the chain) on result: "
        f"(row, rel) {over}; and spec T§8, X-NF rule 1, \"Every comparison in sections 6-8 counts non-finite outputs "
        "in every stage, including the six that C3 only prints. With finite fixtures the count must be 0.\": (row, "
        f"stage, non-finite) {nf_all}")


def test_real_key_row0_decodes_like_ggml():
    row0 = np.frombuffer(committed()[0][KEY], np.uint8)
    v = gpu.dequant(42, row0, N)
    want = q2_0(row0.reshape(-1, KEY_BLOCK)).ravel()
    ok = np.isfinite(v) & np.isfinite(want)
    nf = int(np.count_nonzero(~ok))
    diff = np.abs(v[ok].astype(F64) - want[ok].astype(F64))
    worst = float(diff.max()) if diff.size else 0.0              # NaN-safe: the non-finite pairs are nf's
    rms = float(np.sqrt(np.mean(v.astype(F64) ** 2)))
    same = le(worst, 0.0) and nf == 0
    print(f"  {'ple_key row 0: the GPU decode == ggml dequantize_row_q2_0':<46} {'yes' if same else '*** NO ***'} "
          f"(worst |d| {worst:.3e}, non-finite {nf}); rms {rms:.4e}")
    assert same, f"`if (!le(worst, 0.0) || nf) ++bad;` (ple_parity.cpp:636): worst |d| {worst!r}, non-finite {nf}"
    assert np.isfinite(v).all() and 1e-4 < rms < 10, (
        f"spec T§6.4, C1: the row is finite with 1e-4 < rms < 10 (weights, not padding): rms {rms!r}")
    bound = real_tensors()[KEY][:row0.size]
    assert bound == row0.tobytes(), (
        f"spec T§6.4, C1: \"the bound key's row 0 equals the committed slice byte for byte\": {KEY} row 0 differs at "
        f"bytes {np.flatnonzero(np.frombuffer(bound, np.uint8) != row0)[:8].tolist()}")


# ---- X-INPLACE and X-GUARD: the chain's aliases and its stores' bounds (spec T§8) ----------------------------------
def differ(a, b, keys, rows):
    """The (output, row) pairs of ``keys`` over ``rows`` where the chain runs ``a`` and ``b`` differ, byte for byte."""
    return [(k, t) for k in keys for t in rows if a[k][t].tobytes() != b[k][t].tobytes()]


@pytest.mark.parametrize("name", list(FIXTURES))
def test_compact_build_equals_full_exports(name):
    fx, _, shared = block(name)
    args = fx.emb, fx.hidden, np.ascontiguousarray(fx.hist.T).ravel(), fx
    split = differ(shared, ple_gpu.chain(*args, split_query=True), ("gate", "gated", "normalized", "result"),
                   range(ROWS))
    inplace = differ(shared, ple_gpu.chain(*args, inplace=True), ("normalized", "result"), range(ROWS))
    equal = not split and not inplace
    print(f"  PLE separate exports and in-place hidden result: {'byte-identical' if equal else 'FAIL'}")
    assert equal, (
        "C12, `const bool equal = std::memcmp(compact_norm.data(), dev_norm.data(), hcd * sizeof(float)) == 0 && "
        "std::memcmp(compact_result.data(), dev_res.data(), hcd * sizeof(float)) == 0;` (ple_parity.cpp:963-964), "
        "then `if (!equal) ++bad;` (:966), on every row: (1) query and normalized bound as one buffer (the handler's "
        f"binding) against two buffers, (output, row) {split} differ; (2) result bound onto hidden (basalt's in-place "
        f"alias) against out of place, (output, row) {inplace} differ")


@pytest.mark.parametrize("t", range(1, T_MAX + 1))
def test_store_bounds_guards(t):
    fx, _, clean = block("synthetic")
    out = ple_gpu.chain(fx.emb[:T_MAX], fx.hidden[:T_MAX], np.ascontiguousarray(fx.hist.T).ravel(), fx, t,
                        guard=True)
    broken = [k for k, kept in out["guards"].items() if not kept]
    touched = [(k, r) for k in ("normalized", "result", "snap") for r in range(t, T_MAX)
               if (bits(out[k][r]) != 0xFFFFFFFF).any()]
    wrong = differ(out, clean, (*ple_gpu.SCRATCH, "hist", "normalized", "result", "snap"), range(t))
    restored, kept = ple_gpu.restore(out["hist"][-1], out["snap"], commit_record(1, t, 0))
    if not kept:
        broken.append("restore: hist, snap or commit")
    if restored.tobytes() != out["snap"][0].tobytes():
        wrong.append(("restore", 0))
    assert not broken and not touched and not wrong, (
        "X-GUARD (spec T§8), H3's `if (!guard_ok(history_storage, count) || !guard_ok(norm_storage, channels)) "
        "++bad;` (ple_parity.cpp:228) and C10's `if (!std::all_of(ends.begin(), ends.end(), [](uint8_t x) { return x "
        "== 0xa5; })) ++bad;` (:936) on every binding: guards broken on {broken}; \"rows from T to T_MAX − 1 of every "
        f"multi-row output still hold their sentinel\": (output, row) {touched} were written; rows below T equal the "
        f"exact-size run's, and the restore of n_keep = 1 equals slot 0: (output, row) {wrong} differ (T = {t})")
