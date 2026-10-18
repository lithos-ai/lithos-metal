# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/iq_multi_parity.cpp, src/kernels/native_grouped_parity.cpp and
# src/kernels/native_expert_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their
# fixtures, plans, pre-fills and pass criteria.
"""basalt's routed experts for Q2_0 / Q2_0 on Metal (kernels/common/basalt_experts.metal): native_expert_grouped's
gate/up, SwiGLU + IEEE q8_1 and down dispatches.

Fixtures follow basalt's tests: Q2_0 rows of random bytes with a finite fp16 scale per block, sign | exponent 2..8 |
any mantissa (2^-13 .. 2^-6; ``random_rows``, iq_multi_parity.cpp:72-90, native_grouped_parity.cpp:73-98); token
activations N(0, 1) quantized by the IEEE q8_1 quantizer (``quantize_q8_1_rows``). basalt draws them with
std::mt19937 and std:: distributions, whose outputs are implementation-defined; these draw the same distributions
with numpy."""

import numpy as np
import pytest

from monolith import kernels
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_iq_parity import quantize_q2_0

F32 = np.float32


def random_rows(rng, rows, n, neg=0.5):
    """``rows`` Q2_0 rows of ``n`` values: random bytes, then each block's fp16 scale (negative with probability
    ``neg``; basalt's iq_multi_parity uses 1/4, native_grouped_parity 1/2)."""
    blk = rng.integers(0, 256, (rows * (n // 64), 18), dtype=np.uint8)
    h = ((rng.random(len(blk)) < neg).astype(np.uint16) << 15) | (rng.integers(2, 9, len(blk)).astype(np.uint16) << 10) \
        | rng.integers(0, 1024, len(blk)).astype(np.uint16)
    blk[:, :2] = h.view(np.uint8).reshape(-1, 2)
    return blk.reshape(-1)


def random_blob(rng, n_embd, n_ff, neg=0.5):
    """One expert blob: gate rows | up rows (n_embd values each) | down rows (n_ff values each)."""
    blob = np.concatenate([random_rows(rng, 2 * n_ff, n_embd, neg), random_rows(rng, n_embd, n_ff, neg)])
    assert blob.size == kernels.basalt_expert_layout(n_embd, n_ff)["bytes"]
    return blob


@pytest.mark.parametrize("n_embd, n_ff", [(512, 256), (2560, 640)])
def test_every_variant_builds_and_runs(n_embd, n_ff):
    """Each kernel variant compiles and a one-group call writes finite rows and nothing else."""
    rng = np.random.default_rng(1)
    blob = random_blob(rng, n_embd, n_ff)
    xq = gpu.q8_1(rng.standard_normal(3 * n_embd).astype(F32), "ieee")
    for gu2 in ((False, True) if n_embd == 2560 else (False,)):
        for sub16 in ((False, True) if n_ff == 640 else (False,)):
            r = gpu.experts(blob, [0], [0, 3], [2, 0, 1], [1, 3, 0], xq, n_embd, n_ff, 5, gu2=gu2, sub16=sub16)
            out = r["out"]
            assert np.isfinite(out[[0, 1, 3]]).all(), (gu2, sub16)
            assert np.all(out[[2, 4]].view(np.uint32) == 0xFFFFFFFF), (gu2, sub16)


# ---- calls as basalt's tests build them --------------------------------------------------------------------------

class Call:
    """One expert call: ``n_blobs`` blobs in one arena (each at a 256-byte aligned slot, native_grouped_parity.cpp:118),
    groups of ``sizes`` entries from entry ``base`` on, each group its own blob while there are enough (plan_sizes,
    :164-188), random tokens and scattered destinations (a permutation of the cap rows), and ``tokens`` q8_1 rows of
    N(0, 1) activations from the IEEE quantizer."""

    def __init__(self, rng, n_embd, n_ff, sizes, tokens, *, base=0, cap=None, n_blobs=None, neg=0.5, cap_groups=None):
        self.n_embd, self.n_ff = n_embd, n_ff
        self.lay = kernels.basalt_expert_layout(n_embd, n_ff)
        slot = (self.lay["bytes"] + 255) // 256 * 256
        n_blobs = n_blobs or max(len(sizes), 1)
        self.arena = np.zeros(slot * n_blobs, np.uint8)
        for b in range(n_blobs):
            self.arena[slot * b: slot * b + self.lay["bytes"]] = random_blob(rng, n_embd, n_ff, neg)
        order = rng.permutation(n_blobs)
        self.grp_off = [int(slot * order[g % n_blobs]) for g in range(len(sizes))]
        self.grp_start = [base + int(v) for v in np.cumsum([0] + list(sizes))]
        self.n_ent = self.grp_start[-1]
        self.cap = cap or self.n_ent + 3
        self.cap_groups = cap_groups or max(len(sizes), 1)
        assert self.n_ent <= self.cap
        self.ent_tok = rng.integers(0, tokens, self.cap).astype(np.int32)
        self.ent_dst = rng.permutation(self.cap).astype(np.int32)
        self.x = rng.standard_normal(tokens * n_embd).astype(F32)
        self.xq = gpu.q8_1(self.x, "ieee")

    def entries(self):
        return range(self.grp_start[0], self.n_ent)

    def group_of(self, e):
        return next(g for g in range(len(self.grp_off)) if self.grp_start[g] <= e < self.grp_start[g + 1])

    def run(self, **kw):
        return gpu.experts(self.arena, self.grp_off, self.grp_start, self.ent_tok, self.ent_dst, self.xq, self.n_embd,
                           self.n_ff, self.cap, cap_groups=self.cap_groups, cap_entries=self.cap, **kw)

    def per_entry(self, **kw):
        """Each entry alone (one group of one entry): the grouped kernels' arithmetic one entry a call, the comparand
        that stands for basalt's older kernels. Rows no entry writes keep the NaN pattern; each call must leave every
        row but its own as it was, as basalt diffs the whole buffer (native_grouped_parity.cpp:226-234)."""
        out = np.full((self.cap, self.n_embd), 0xFFFFFFFF, np.uint32)
        for e in self.entries():
            g, d = self.group_of(e), int(self.ent_dst[e])
            r = gpu.experts(self.arena, [self.grp_off[g]], [0, 1], [self.ent_tok[e]], [d], self.xq, self.n_embd,
                            self.n_ff, self.cap, **kw)
            got = r["out"].view(np.uint32)
            assert np.all(np.delete(got, d, axis=0) == 0xFFFFFFFF), ("stray store", e)
            out[d] = got[d]
        return out

    def blob(self, e, part):
        """Entry e's gate, up or down rows (GGUF bytes)."""
        o, lay = self.grp_off[self.group_of(e)], self.lay
        start = {"gate": 0, "up": lay["up_off"], "down": lay["down_off"]}[part]
        n = lay["n_ff"] * lay["gu_row"] if part != "down" else lay["n_embd"] * lay["d_row"]
        return self.arena[o + start: o + start + n]


def q8_rows(xq, rows, n):
    """The q8_1 rows ``rows`` (n values each) of the q8_1 bytes ``xq``."""
    b = n // 32 * kernels.Q8_1_BLOCK_BYTES
    return np.concatenate([np.asarray(xq)[r * b: (r + 1) * b] for r in rows])


def gpu_expf_neg(g):
    """The SwiGLU's __expf(-g) as the kernel computes it: basalt_fp.metal's expf_neg, the same device function."""
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + """
kernel void ex(device const float* g [[buffer(0)]], device float* e [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint i [[thread_position_in_grid]]) {
    if (i < n) e[i] = expf_neg(g[i]);
}
"""
    g = np.ascontiguousarray(g, F32).reshape(-1)
    out = gpu.run(gpu.pipeline(src, "ex"), [g.tobytes(), 4 * g.size], np.uint32(g.size).tobytes(), 2,
                  -(-g.size // 256), 256)
    return np.frombuffer(out[1], F32)


def check_against_emulation(c, r):
    """Every stage bit for bit against tests/kernels/basalt_emu.py (experts.md): gate and up rows per entry, the q8_1 of
    the SwiGLU product from the GPU's gate and up (with the GPU's __expf), the down rows from the GPU's q8_1."""
    n_embd, n_ff = c.n_embd, c.n_ff
    ents = list(c.entries())
    if not ents:                                                # an empty call: nothing to emulate
        return
    for e in ents:
        xr = q8_rows(c.xq, [c.ent_tok[e]], n_embd)
        for part in ("gate", "up"):
            want = emu.q2_0_dots(c.blob(e, part), n_embd, xr)[0]
            assert np.array_equal(r[part][e].view(np.uint32), want.view(np.uint32)), (part, e)
    lo, hi = c.grp_start[0] * n_ff, c.n_ent * n_ff
    g, u = r["gate"].reshape(-1)[lo:hi], r["up"].reshape(-1)[lo:hi]
    hq = emu.swiglu_q8_1(g, u, gpu_expf_neg(g))
    blk = kernels.Q8_1_BLOCK_BYTES
    assert np.array_equal(r["hq"][lo // 32 * blk: hi // 32 * blk], hq)
    for e in ents:
        want = emu.q2_0_dots(c.blob(e, "down"), n_ff, q8_rows(r["hq"], [e], n_ff))[0]
        assert np.array_equal(r["out"][c.ent_dst[e]].view(np.uint32), want.view(np.uint32)), ("down", e)


# ---- iq_multi_parity's check_grouped (iq_multi_parity.cpp:239-337) -----------------------------------------------

def grouped_case(seed=18):
    """check_grouped's call: 9 groups of {1, 3, 4, 5, 0, 8, 2, 11, 1} entries (one pass of GRP_NC, partial passes, two
    passes and more), 8 tokens, cap_groups = groups + 2 (the extra threadgroup rows must exit), cap = entries + 3, the
    entries' destinations a permutation of the first rows, random tokens (:254-264, :321); Q2_0 / Q2_0 at 2560 x 640
    (:497-500)."""
    rng = np.random.default_rng(seed)
    sizes = [1, 3, 4, 5, 0, 8, 2, 11, 1]
    c = Call(rng, 2560, 640, sizes, 8, neg=0.25, cap_groups=len(sizes) + 2)
    c.ent_dst = np.concatenate([rng.permutation(c.n_ent), np.arange(c.n_ent, c.cap)]).astype(np.int32)
    return c


def test_check_grouped():
    """The grouped call against the same kernels one entry a call (basalt: old kernels against new), every float
    compared, the NaN rows nothing may write included; basalt's criterion, quoted: ``const bool ok = diff == 0 &&
    finite;`` (iq_multi_parity.cpp:329), finite over the written rows. Also the kernel choices against each other
    (gate/up two rows or one row a SIMD-group, down on 16 or 32 lanes) and every stage against the emulation."""
    c = grouped_case()
    r = c.run(scratch_fill=0x00)                                # out NaN-filled, scratch zeroed (:309-310)
    got = r["out"].view(np.uint32)
    diff = np.count_nonzero(got != c.per_entry())
    finite = np.isfinite(r["out"][: c.n_ent]).all()
    assert diff == 0 and finite, (diff, finite)
    assert np.all(got[c.n_ent:] == 0xFFFFFFFF)
    for gu2, sub16 in ((False, True), (True, False), (False, False)):
        other = c.run(gu2=gu2, sub16=sub16)["out"].view(np.uint32)
        assert np.count_nonzero(other != got) == 0, (gu2, sub16)
    check_against_emulation(c, r)


# ---- native_grouped_parity (native_grouped_parity.cpp:219-248, 513-532) ------------------------------------------

@pytest.mark.parametrize("n_embd, n_ff", [(512, 256), (2560, 640)])
def test_native_grouped_parity(n_embd, n_ff):
    """native_grouped_parity's check on Q2_0 / Q2_0: T = 6 tokens x 4 experts (cap 24 entries), 9 blobs, calls of 0, 1,
    2, 3, 5, 8 and 24 groups of 1..T entries from entry 0 and from entry 7, scattered destinations over the cap rows;
    each call with grid_groups 0, 1, 2, 3, 4 and cap + 3 and stale scratch (0x00 or 0x7F alternately) against the same
    kernels one entry a call (basalt: against v1). 512 x 256 is the registered test's shape (gate/up one row a
    SIMD-group); 2560 x 640 is its ``--gu2`` run (the kernels decode runs), where the per-entry comparand takes the
    one-row gate/up kernel as basalt's v1 does, against gu2 in the grouped call. Criteria, quoted: ``for (size_t i = 0; i <
    ref.size(); ++i) diff += ref[i] != got[i];`` (:234) and ``if (written == 0) { std::printf("  nothing was written:
    the check checked nothing  FAIL\\n"); ++bad; }`` (:246). Every stage of each call's first run is also checked
    against the emulation."""
    rng = np.random.default_rng(316 + n_embd)
    T, K, n_blobs = 6, 4, 9
    cap = T * K
    calls = written = 0
    for base in (0, 7):
        for groups in (0, 1, 2, 3, 5, 8, 24):
            sizes, used = [], base
            for _ in range(groups):                                  # plan(): groups of 1..T while they fit
                m = min(int(rng.integers(1, T + 1)), cap - used)
                if m <= 0:
                    break
                sizes.append(m)
                used += m
            c = Call(rng, n_embd, n_ff, sizes, T, base=base, cap=cap, n_blobs=n_blobs, cap_groups=cap)
            # basalt's --gu2 run compares gu2 against v1's gate/up kernel (the one-row kernel): so does the comparand
            ref = c.per_entry(gu2=False if n_embd == 2560 else None)
            written += np.count_nonzero(ref != 0xFFFFFFFF)
            for gy in (0, 1, 2, 3, 4, cap + 3):
                r = c.run(grid_groups=gy, scratch_fill=0x7F if calls & 1 else 0x00)
                if gy == 0:
                    check_against_emulation(c, r)
                diff = np.count_nonzero(ref != r["out"].view(np.uint32))
                calls += 1
                assert diff == 0, (base, groups, gy, diff)
    # basalt: if (written == 0) { ... "nothing was written: the check checked nothing  FAIL" ... }
    assert written > 0
    print(f"\nQ2_0/Q2_0 {n_embd} x {n_ff}: {calls} calls bitwise equal to per-entry calls, {written} floats written")


# ---- IEEE subnormals in the SwiGLU product ------------------------------------------------------------------------

@pytest.mark.parametrize("n_embd, n_ff", [(512, 256), (2560, 640)])
def test_groups_read_their_blobs_across_buffers(n_embd, n_ff):
    """basalt's grp_ptr is an address a group: the same call with its blobs moved into two separate buffers, at offsets
    other than the arena's, writes the same bits (gate, up, the q8_1 of H and the output)."""
    rng = np.random.default_rng(7)
    c = Call(rng, n_embd, n_ff, [3, 1, 2, 6], 4, n_blobs=4)
    ref = c.run()
    blob = c.lay["bytes"]
    pad = 512                                              # each blob 512 bytes past the previous one's end
    arenas = [np.zeros(pad + 2 * (blob + pad), np.uint8), np.zeros(pad + 2 * (blob + pad), np.uint8)]
    grp_buf, grp_off, used = [], [], [0, 0]
    for g, o in enumerate(c.grp_off):
        b = g % 2
        at = pad + used[b] * (blob + pad)
        arenas[b][at: at + blob] = c.arena[o: o + blob]
        grp_buf.append(b)
        grp_off.append(at)
        used[b] += 1
    got = gpu.experts(arenas, grp_off, c.grp_start, c.ent_tok, c.ent_dst, c.xq, n_embd, n_ff, c.cap,
                      cap_groups=c.cap_groups, cap_entries=c.cap, grp_buf=grp_buf)
    assert sorted(set(grp_buf)) == [0, 1] and grp_off != c.grp_off
    for k in ("gate", "up", "hq", "out"):
        assert np.array_equal(np.asarray(got[k]).view(np.uint8), np.asarray(ref[k]).view(np.uint8)), k


def test_ieee_mul_keeps_subnormals():
    """ieee_mul (basalt_q8_1.metal), CUDA's mul.rn.f32 without .ftz, bit for bit against numpy's IEEE float32 product
    on planted pairs: products across the subnormal range, at the 2^-126 boundary (rounding up into it), at ties of
    the subnormal grid, below half the smallest subnormal, signed zeros, subnormal operands, normal products, and
    infinities and NaN against zeros and subnormals (NaN for 0 * inf and a NaN, the signed infinity for a subnormal
    times inf; a NaN's payload is the GPU's)."""
    rng = np.random.default_rng(9)
    n = 1 << 16
    a = (rng.choice([-1, 1], n) * np.exp2(rng.uniform(-90, -20, n))).astype(F32)
    b = (rng.choice([-1, 1], n) * np.exp2(rng.uniform(-90, -20, n))).astype(F32)
    tie = np.full(64, F32(2.0 ** -75))                                 # 2^-75 * (2k + 1) 2^-75 = (k + 0.5) 2^-149
    inf, nan, sub = F32(np.inf), F32(np.nan), np.uint32(5).view(F32)
    planted = [(F32(2.0 ** -63), F32(2.0 ** -63) * F32(1 - 2.0 ** -24)),      # just below 2^-126: rounds up
               (F32(2.0 ** -75), F32(2.0 ** -75)), (F32(2.0 ** -100), F32(2.0 ** -50)),  # exactly 2^-150: to +0
               (F32(-(2.0 ** -100)), F32(2.0 ** -49) * F32(1.5)),   # 1.5 * 2^-149: tie, to even (2 units)
               (F32(0.0), F32(-3.0)), (F32(-0.0), F32(-0.0)), (F32(2.0 ** -140), F32(3.0)),   # zeros, a subnormal
               (np.uint32(1).view(F32), F32(2.0 ** 100)), (F32(1.5), F32(2.0 ** -128)), (F32(3.0), F32(5.0)),
               (F32(0.0), inf), (F32(-0.0), inf), (F32(0.0), -inf), (-inf, F32(-0.0)),    # 0 * inf: NaN
               (F32(0.0), nan), (F32(-0.0), nan), (nan, F32(0.0)), (nan, F32(-0.0)),      # a NaN: NaN
               (sub, inf), (-inf, sub), (-sub, -inf), (sub, nan), (nan, -sub), (inf, F32(-3.0))]   # inf, NaN, -inf
    pa = np.concatenate([a, tie, np.array([p[0] for p in planted], F32)])
    pb = np.concatenate([b, (np.arange(64) * 2 + 1).astype(F32) * F32(2.0 ** -75), np.array([p[1] for p in planted], F32)])
    src = kernels.basalt_q8_1_source() + """
kernel void mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]],
                constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) c[i] = ieee_mul(a[i], b[i]);
}
"""
    out = gpu.run(gpu.pipeline(src, "mul"), [pa.tobytes(), pb.tobytes(), 4 * pa.size], np.uint32(pa.size).tobytes(), 3,
                  -(-pa.size // 256), 256)
    got = np.frombuffer(out[2], F32)
    with np.errstate(under="ignore", invalid="ignore"):
        want = pa * pb
        units = pa.astype(np.float64) * pb.astype(np.float64) * 2.0 ** 149      # exact products, units of 2^-149
        ties = np.count_nonzero((np.abs(units) < 2.0 ** 23) & (np.abs(units) % 1 == 0.5))
    sub = np.count_nonzero((np.abs(want) < F32(2.0 ** -126)) & (want != 0))
    same = (got.view(np.uint32) == want.view(np.uint32)) | (np.isnan(got) & np.isnan(want))
    assert same.all(), np.flatnonzero(~same)[:8]
    assert sub > 1000 and ties >= 64, (sub, ties)                       # the subnormal path ran, on grid ties too
    print(f"\nieee_mul: {pa.size} products bit for bit, {sub} of them subnormal, {ties} grid ties, "
          f"{np.count_nonzero(np.isnan(want))} NaN")


def test_swiglu_q8_1_edges():
    """The SwiGLU + q8_1 dispatch alone on planted gate and up values, bit for bit against the emulation (the
    exponential taken from the GPU): very negative gates (-88.7 .. -60) with small up values, whose IEEE products are
    subnormal; gates past -88.72, where the exponential overflows and the quotient is -0; large positive gates, zeros,
    multiples of 2^-48 (the GEMV's grid), and N(0, 4) gates; non-finite gates and ups (a weight row with a non-finite
    fp16 scale), whose IEEE products are NaN: five in a finite block (its sum NaN) and a whole block of inf * 0 (its
    scale and sum NaN, each stored as sm_120's fp16 0x7fff, as in test_basalt_q8_1). Entries [1, 5) of a 6-entry
    scratch, so the dispatch must leave entry 0's and entry 5's q8_1 blocks as they were."""
    rng = np.random.default_rng(10)
    n_ff, ents = 640, 4
    g = np.concatenate([rng.standard_normal(n_ff) * 4, rng.uniform(-88.7, -60, n_ff), rng.uniform(-91, -86, n_ff),
                        np.concatenate([rng.uniform(80, 120, 160), np.zeros(160),
                                        rng.integers(-1000, 1000, 320) * 2.0 ** -48])]).astype(F32)
    up = np.concatenate([rng.standard_normal(n_ff), rng.choice([-1, 1], n_ff) * np.exp2(rng.uniform(-40, -1, n_ff)),
                         rng.standard_normal(n_ff), rng.standard_normal(n_ff) * 1e-3]).astype(F32)
    nf = 3 * n_ff + 96                                                  # blocks nf / 32 and nf / 32 + 1
    g[nf: nf + 5] = [np.inf, np.nan, 0.0, -100.0, -np.inf]              # s = inf, NaN, 0, -0, NaN against
    up[nf: nf + 5] = [0.0, -0.0, np.inf, np.nan, 0.0]                   # up = 0, -0, inf, NaN, 0: five NaN products
    g[nf + 32: nf + 64], up[nf + 32: nf + 64] = np.inf, 0.0             # a whole block of inf * 0
    gate = np.concatenate([np.full(n_ff, np.nan, F32), g, np.full(n_ff, np.nan, F32)])
    upp = np.concatenate([np.zeros(n_ff, F32), up, np.zeros(n_ff, F32)])
    nb = 6 * n_ff // 32 * kernels.Q8_1_BLOCK_BYTES
    pso = gpu.pipeline(kernels.basalt_experts_source(), "basalt_expert_swiglu_q8_1",
                       tuple(sorted(kernels.basalt_expert_macros(512, n_ff).items())))
    out = gpu.run(pso, [gate.tobytes(), upp.tobytes(), np.array([1, 3, 5], np.int32).tobytes(),
                        np.int32(2).tobytes(), nb], kernels.basalt_expert_params(512, n_ff), 5,
                  -(-6 * n_ff // kernels.EXPERT_THREADS), kernels.EXPERT_THREADS, fill=0xA5)
    hq = np.frombuffer(out[4], np.uint8)
    per = n_ff // 32 * kernels.Q8_1_BLOCK_BYTES
    assert np.all(hq[:per] == 0xA5) and np.all(hq[5 * per:] == 0xA5)
    got, want = hq[per: 5 * per].reshape(-1, 36), emu.swiglu_q8_1(g, up, gpu_expf_neg(g)).reshape(-1, 36)
    gh, wh = (b[:, :4].copy().view(np.uint16) for b in (got, want))    # each block's fp16 d and s
    nan_h = ((wh & 0x7C00) == 0x7C00) & ((wh & 0x3FF) != 0)
    assert nan_h[nf // 32, 1] and nan_h[nf // 32 + 1].all() and nan_h.sum() == 3        # the planted NaN halves
    assert np.array_equal(got[:, 4:], want[:, 4:])                     # every code bit for bit
    # every half bit for bit, the NaN ones sm_120's 0x7fff (nan.md)
    assert np.array_equal(gh, wh) and np.all(gh[nan_h] == 0x7FFF), np.flatnonzero(np.any(gh != wh, axis=1))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        h = (g / (F32(1) + gpu_expf_neg(g))) * up
    sub = np.count_nonzero((h != 0) & (np.abs(h) < F32(2.0 ** -126)))
    assert sub > 50, sub                                                # ieee_mul's subnormal path ran
    print(f"\nSwiGLU + q8_1 edges: {g.size} values bit for bit, {sub} subnormal products")


# ---- float64 references ------------------------------------------------------------------------------------------

def dequant_q8_1(xq, n):
    """q8_1 rows back to float64: d * code (iq_multi_parity.cpp:99-109)."""
    b = np.asarray(xq, np.uint8).reshape(-1, 36)
    return (b[:, :2].copy().view(np.float16).astype(np.float64) * b[:, 4:].view(np.int8)).reshape(-1, n)


def test_stages_against_float64():
    """Each stage of check_grouped's call against a float64 product over the GPU dequantizer's weights (iq_parity
    checks it against gguf-py; basalt_dequant.metal) and the dequantized q8_1 rows each stage reads: gate and up on the
    tokens' rows, down on the GPU's q8_1 of the SwiGLU product. iq_multi_parity's check_mmvq bound for these Q2_0 rows
    (the nearest basalt has for the expert's call arithmetic), quoted: ``if (diff != 0 || !finite || !(rel < 1e-2))
    {`` (iq_multi_parity.cpp:162), rel = sum |y - ref| / sum |ref| over the stage's outputs (:150-160); diff, the
    bitwise check against the other kernels, is test_check_grouped's."""
    c = grouped_case()
    r = c.run()
    blk = kernels.Q8_1_BLOCK_BYTES
    rel, worst = {}, {}
    for part, n_in, n_out in (("gate", c.n_embd, c.n_ff), ("up", c.n_embd, c.n_ff), ("down", c.n_ff, c.n_embd)):
        num = den = 0.0
        finite = True
        per_entry = []
        for g in range(len(c.grp_off)):
            ents = range(c.grp_start[g], c.grp_start[g + 1])
            if not ents:
                continue
            w = gpu.dequant(42, c.blob(ents[0], part), n_in * n_out).reshape(n_out, n_in).astype(np.float64)
            if part == "down":
                x = dequant_q8_1(q8_rows(r["hq"], list(ents), n_in), n_in)
                y = r["out"][c.ent_dst[list(ents)]]
            else:
                x = dequant_q8_1(q8_rows(c.xq, c.ent_tok[list(ents)], n_in), n_in)
                y = r[part][list(ents)]
            ref = x @ w.T
            finite = finite and np.isfinite(y).all()
            num += np.abs(y - ref).sum()
            den += np.abs(ref).sum()
            per_entry += list(np.abs(y - ref).sum(axis=1) / (np.abs(ref).sum(axis=1) + 1e-30))
        rel[part], worst[part] = num / (den + 1e-30), max(per_entry)
        # basalt: if (diff != 0 || !finite || !(rel < 1e-2)) {  -- over the call, and here per entry as well
        assert finite and rel[part] < 1e-2 and worst[part] < 1e-2, (part, rel[part], worst[part])
    print("\nstages against float64: " + ", ".join(f"{k} rel {v:.2e} (worst entry {worst[k]:.2e})"
                                                    for k, v in rel.items()))
    raw = c.blob(0, "gate")[: 64 * c.lay["gu_row"]].reshape(64, -1)
    assert np.array_equal(gpu.dequant(42, raw, 64 * c.n_embd).reshape(64, -1), gguf_reference.dequantize(42, raw))


def test_whole_expert_against_float64():
    """The contract's reference for native_expert_grouped (docs/design/flash-next.md, the table of kernels without a
    basalt reference): the whole expert (gate/up, SwiGLU, the q8_1 of the product, down) against a float64 expert on
    the fp32 activations, built as native_expert_parity.cpp builds it (:46-78, :387-402): 3 tokens of N(0, 1) values,
    2560 x 640, row r's weights 0.02 (0.5 + (r % 7) / 7) N(0, 1) quantized to Q2_0 (tools/gguf_writer.py's
    quantize_q2_0; basalt's test calls ggml's quantizer, and has no Q2_0 pair) and dequantized on the host by ggml's
    dequantize_row_q2_0 (tests/contract/gguf_reference.py; tools/gguf_writer.py's dequantize_q2_0 mirrors it), as
    basalt's reference takes ggml's to_float (:56-61); h = (float) (g / (1 + exp(-g)) * u) in double; out = D h in
    double. Criterion, quoted: ``const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);``
    (native_expert_parity.cpp:368), eg the GPU's relative L1 error (:38-42); ec, the CPU tier's, belongs to the CPU
    expert's port."""
    n_embd, n_ff, nt = 2560, 640, 3
    rng = np.random.default_rng(11)

    def quant(rows, cols):
        scale = (0.02 * (0.5 + (np.arange(rows) % 7) / 7.0)).astype(F32)[:, None]
        return quantize_q2_0(scale * rng.standard_normal((rows, cols)).astype(F32))

    blob = np.concatenate([quant(n_ff, n_embd), quant(n_ff, n_embd), quant(n_embd, n_ff)])
    lay = kernels.basalt_expert_layout(n_embd, n_ff)
    G, U, D = (gguf_reference.dequantize(42, blob[o: o + rows * row].reshape(rows, row)).astype(np.float64)
               for o, rows, row in ((0, n_ff, lay["gu_row"]), (lay["up_off"], n_ff, lay["gu_row"]),
                                    (lay["down_off"], n_embd, lay["d_row"])))
    x = rng.standard_normal((nt, n_embd)).astype(F32)
    g, u = x.astype(np.float64) @ G.T, x.astype(np.float64) @ U.T
    h = (g / (1.0 + np.exp(-g)) * u).astype(F32)
    ref = (h.astype(np.float64) @ D.T).astype(F32)
    got = gpu.experts(blob, [0], [0, nt], np.arange(nt), np.arange(nt), gpu.q8_1(x.reshape(-1), "ieee"), n_embd, n_ff,
                      nt)["out"]
    eg = np.abs(got.astype(np.float64) - ref).sum() / (np.abs(ref.astype(np.float64)).sum() + 1e-30)
    print(f"\nwhole expert Q2_0/Q2_0 {n_embd} x {n_ff}, {nt} tokens: gpu rel {eg:.2e}")
    # basalt: const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
    assert np.isfinite(eg) and eg < 3e-2 and np.isfinite(got).all(), eg
