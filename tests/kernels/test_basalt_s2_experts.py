# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/s2_expert_grouped_parity.cpp and src/kernels/quantize_act_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures, plans, sentinels and pass
# criteria.
"""basalt's drafter experts on Metal (kernels/common/basalt_s2_experts.metal): moe_group_resident,
quantize_q8_0_scaled and moe_grouped_s2 (gate/up, SwiGLU, the intermediate's quantizer, down), with the IEEE helpers
they share (kernels/common/basalt_q8_1.metal) and the fp16 conversion (basalt_fp.metal's f16_bits).

Fixtures follow basalt's tests; basalt draws them with std::mt19937 and std:: distributions, whose outputs are
implementation-defined, and these draw the same distributions with numpy. basalt's fixtures never make a subnormal (its
scales keep every product above 1e-6), so planted inputs exercise the IEEE paths, each with a negative control: the
same kernels with the plain Metal operation in place of each IEEE helper must differ on them."""

import struct

import numpy as np
import pytest

from monolith import kernels
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_iq_parity import quantize_q2_0

F32 = np.float32
H, FF, K = kernels.S2_H, kernels.S2_FF, 10
NC_GU, NC_D = H // 32, FF // 32
XROW = NC_GU * kernels.Q8_0_BLOCK_BYTES                 # one token's activation row of block_q8_0
O_D_CODES = 2 * FF * (H // 4)
O_GU_SCALES = O_D_CODES + H * (FF // 4)
O_D_SCALES = O_GU_SCALES + 2 * FF * (H // 64) * 2


def gpu_eval(body, *arrays, out=1):
    """A one-off kernel over n threads on basalt_q8_1_source's helpers: ``body`` computes ``r`` from ``a``, ``b``, ``c``
    (float inputs at index i); returns the float32 r."""
    names = "abc"[: len(arrays)]
    args = "".join(f"device const float* {v}_ [[buffer({k})]], " for k, v in enumerate(names))
    loads = "".join(f"const float {v} = {v}_[i]; " for v in names)
    src = kernels.basalt_q8_1_source() + f"""
kernel void ev({args}device float* r_ [[buffer({len(names)})]], constant uint& n [[buffer({len(names) + 1})]],
               uint i [[thread_position_in_grid]]) {{
    if (i >= n) return;
    {loads}float r; {body}
    r_[i] = r;
}}
"""
    arrays = [np.ascontiguousarray(a, F32).reshape(-1) for a in arrays]
    n = arrays[0].size
    res = gpu.run(gpu.pipeline(src, "ev"), [a.tobytes() for a in arrays] + [4 * n], np.uint32(n).tobytes(),
                  len(arrays) + 1, -(-n // 256), 256)
    return np.frombuffer(res[len(arrays)], F32)


def same_bits(got, want):
    return got.view(np.uint32) == np.asarray(want, F32).view(np.uint32)


# ---- the fp16 conversion: quantize_act_parity's fp16 half (quantize_act_parity.cpp:283-336) -------------------------
# The file's Q8_0 half (run_case on quantize_q8_0, :70-130, :241-277) and Q8_K half (:337-401) are not ported: this
# path calls neither quantize_q8_0 nor quantize_q8_K; the drafter quantizes with quantize_q8_0_scaled (mtp.cpp:933;
# docs/design/flash-next.md), whose test below borrows the Q8_0 half's five distributions and byte criterion.

def test_f16_bits_is_f16_from_f32():
    """basalt_fp.metal's f16_bits (Metal's half(), which q8_1_put and the quantizer share) against basalt's vectors,
    quoted: ``if (got != c.want) {`` (quantize_act_parity.cpp:322); h2f(0x0001) == 2^-24, quoted:
    ``if (!sub_ok) ++bad;`` (:335); and against the transcription of f16_from_f32 (tests/kernels/basalt_emu.py) on
    the fp16 subnormal range, the 65504-65520 boundary, both overflow edges and random float32 patterns (a NaN's
    payload is the GPU's)."""
    cases = [(1e30, 0x7C00), (-1e30, 0xFC00), (65536.0, 0x7C00), (3.0e38, 0x7C00), (65504.0, 0x7BFF),
             (65519.996, 0x7BFF), (65520.0, 0x7C00), (65536.0, 0x7C00), (5.9604645e-8, 0x0001), (1.0, 0x3C00),
             (0.0, 0x0000)]
    rng = np.random.default_rng(3)
    sweep = np.concatenate([
        np.array([c for c, _ in cases], F32),
        np.arange(0x33000000, 0x38800000, 97, dtype=np.uint32).view(F32),            # 2^-25 .. 2^-14: fp16 subnormals
        np.arange(0x477FE000, 0x47800100, dtype=np.uint32).view(F32),                # 65504 .. 65536
        -np.arange(0x477FE000, 0x47800100, 7, dtype=np.uint32).view(F32),
        np.uint32([1, 0x007FFFFF, 0x00800000, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF]).view(F32),
        rng.integers(0, 2 ** 32, 1 << 20, dtype=np.uint64).astype(np.uint32).view(F32)])
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + """
kernel void f16(device const float* x [[buffer(0)]], device ushort* h [[buffer(1)]], device float* back [[buffer(2)]],
                constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    h[i] = f16_bits(x[i]);
    back[i] = h2f(ushort(i));
}
"""
    n = sweep.size
    out = gpu.run(gpu.pipeline(src, "f16"), [sweep.tobytes(), 2 * n, 4 * n], np.uint32(n).tobytes(), 3,
                  -(-n // 256), 256)
    got = np.frombuffer(out[1], np.uint16)
    fp16_bad = sum(int(got[k]) != want for k, (_, want) in enumerate(cases))
    # basalt: if (got != c.want) {
    assert fp16_bad == 0, [(c, hex(w), hex(int(got[k]))) for k, (c, w) in enumerate(cases)]
    back = np.frombuffer(out[2], F32)[:65536]
    assert back[1] == F32(5.9604645e-8)                    # basalt: sub_ok = (sub == 5.9604645e-8f)
    with np.errstate(invalid="ignore"):
        exact = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(F32)
    assert np.array_equal(back.view(np.uint32)[~np.isnan(exact)], exact.view(np.uint32)[~np.isnan(exact)])
    want = emu.f16_from_f32(sweep)
    nan = np.isnan(sweep)
    assert np.array_equal(got[~nan], want[~nan]), np.flatnonzero(got[~nan] != want[~nan])[:8]
    assert np.all((got[nan] & 0x7C00) == 0x7C00) and np.all(got[nan] & 0x3FF)
    print(f"\nf16_bits: {len(cases)} basalt vectors, {np.count_nonzero(~nan)} values bit for bit with f16_from_f32")


def test_builders_take_basalt_launches_only():
    """The builders refuse the shapes basalt cannot launch (CUDA's largest gridDim.x and .y: past them basalt's launch
    fails and exits, and here a 32-bit threadgroup position or int(tg.y) would wrap) and take its largest: 65,535
    groups, 858,993,458 entries (2,147,483,645 SwiGLU and quantizer threadgroups), and quantizer block counts past 2^32
    (Q80Params.n_blocks is 64-bit, basalt's long long)."""
    g = kernels.basalt_s2_grids(65535, 858_993_458)
    assert g == {"gu": (40, 65535), "swiglu": 2_147_483_645, "quant": 2_147_483_645, "down": (40, 65535)}
    assert struct.unpack("<QQ", kernels.basalt_q8_0_scaled_params((2 ** 31 - 1) * 256)) == (2 ** 34 - 8, 0)
    for f, args in ((kernels.basalt_s2_grids, (65536, 1)), (kernels.basalt_s2_grids, (0, 1)),
                    (kernels.basalt_s2_grids, (1, 858_993_459)), (kernels.basalt_s2_scratch, (858_993_459,)),
                    (kernels.basalt_s2_params, (0,)), (kernels.basalt_q8_0_scaled_params, ((2 ** 31 - 1) * 256 + 32,))):
        with pytest.raises(ValueError):
            f(*args)


# ---- quantize_q8_0_scaled: the contract's host transcription (flash-next.md) -----------------------------------------

def quantize_distributions():
    """quantize_act_parity's five distributions of 131,072 values (quantize_act_parity.cpp:246-275)."""
    rng = np.random.default_rng(7)
    n = 32 * 4096
    d = ((np.arange(n) % 200) - 100 + 0.5).astype(F32)
    d[0] = 127.0
    e = np.zeros(n, F32)
    e[::32] = 1000.0 + (np.arange(n // 32) % 7)
    return {"random normal": rng.standard_normal(n).astype(F32),
            "small magnitude (1e-6)": (rng.standard_normal(n).astype(F32) * F32(1e-6)).astype(F32),
            "all zero": np.zeros(n, F32), "exact .5 boundaries": d, "one extreme per block": e}


def planted_blocks():
    """Blocks where the GPU's plain operations and IEEE's differ: amax subnormal (s on the subnormal grid, inv from the
    scaled reciprocal), amax at 2^-121 .. 2^-119 (s just normal, inv large), amax below about 2^-127 (inv infinite)
    with zero lanes (t NaN, code 0), subnormal lanes in normal blocks, codes on .5 ties of tiny blocks, and blocks with
    NaN, infinite and signed-zero lanes."""
    rng = np.random.default_rng(11)
    blocks = []
    for lo, hi in ((-149, -127), (-130, -121), (-126, -119), (-140, -100)):
        for _ in range(64):
            b = (rng.choice([-1, 1], 32) * np.exp2(rng.uniform(lo, hi, 32))).astype(F32)
            b[rng.integers(0, 32, 3)] = 0.0
            blocks.append(b)
    for k in range(1, 40):                                                     # amax = k units of 2^-149
        b = np.zeros(32, F32)
        b[: min(k, 32)] = np.uint32(np.arange(1, min(k, 32) + 1)).view(F32)
        b[-1] = np.uint32(k).view(F32)
        blocks.append(b)
    for _ in range(64):                                                        # subnormal lanes in normal blocks
        b = rng.standard_normal(32).astype(F32) * F32(2.0 ** -100)
        b[rng.integers(0, 32, 8)] = rng.integers(1, 1 << 23, 8).astype(np.uint32).view(F32)
        blocks.append(b)
    for _ in range(64):                                                        # ordinary blocks with planted lanes
        b = rng.standard_normal(32).astype(F32)
        b[:5] = [np.nan, np.inf, -0.0, 0.0, -np.inf] if rng.random() < 0.5 else [np.nan, 1.0, -0.0, 0.0, -2.0]
        blocks.append(b)
    blocks.append(np.full(32, np.nan, F32))
    return np.concatenate(blocks).astype(F32)


@pytest.mark.parametrize("dist", ["random normal", "small magnitude (1e-6)", "all zero", "exact .5 boundaries",
                                  "one extreme per block", "planted"])
def test_quantize_q8_0_scaled(dist):
    """basalt_quantize_q8_0_scaled against the host transcription of quantize_q8_0_scaled_warp_kernel
    (tests/kernels/basalt_emu.py's q8_0_scaled): the contract's test for this quantizer (docs/design/flash-next.md, the
    table of kernels without a basalt reference), codes and fp32 scales byte for byte, as quantize_act_parity checks
    quantize_q8_0, quoted: ``if (g_blocks[i] != r_blocks[i]) ++byte_bad;`` (quantize_act_parity.cpp:100). On
    quantize_act_parity's five distributions, and on planted blocks (planted_blocks) that only an IEEE kernel gets
    right."""
    x = planted_blocks() if dist == "planted" else quantize_distributions()[dist]
    blocks, scales = gpu.q8_0_scaled(x)
    want_b, want_s = emu.q8_0_scaled(x)
    byte_bad = np.count_nonzero(blocks != want_b)
    scale_bad = np.count_nonzero(scales.view(np.uint32) != want_s.view(np.uint32))
    assert byte_bad == 0 and scale_bad == 0, (dist, byte_bad, scale_bad)
    if dist == "planted":
        with np.errstate(divide="ignore", over="ignore"):
            sub = np.count_nonzero((want_s > 0) & (want_s < F32(2.0 ** -126)))
            inf_inv = np.count_nonzero((want_s > 0) & np.isinf(F32(1.0) / want_s))
        assert sub > 100 and inf_inv > 10, (sub, inf_inv)
    print(f"\nquantize_q8_0_scaled, {dist}: {x.size // 32} blocks byte for byte")


# ---- s2_expert_grouped_parity (s2_expert_grouped_parity.cpp:83-106, :149-260, :263-460) ------------------------------
# Ported: section 4 (moe_grouped_s2, 8 tokens x 10 entries, a blob outside the arena), section 5 (moe_group_resident +
# moe_grouped_s2 at the drafter's shape), and the properties sections 2 and 3 check (a device count below the
# capacity; each entry reading its own token's row), on the grouped kernels. Left out, as this path does not run them:
# - the per-hit kernels (sections 1-3 and 6-8: moe_hit_grouped_s2, _dev, _multi) and the previous grouped kernels
#   (OLD_GROUPED=1): the drafter calls moe_grouped_s2 only (mtp.cpp:934), and the contract compares each kernel with
#   its own per-entry calls instead of a previous kernel (docs/design/flash-next.md);
# - the fp16-d runs (a null x_scales): they select quantize_q8_0 for the intermediate (s2_expert_grouped.cu:1112),
#   and the drafter always passes its fp32 scales (it allocates hit_xs_, mtp.cpp:438);
# - the misaligned fallbacks (sections 6-8): new_grouped (s2_expert_grouped.cu:504-508) sends a misaligned x or
#   scratch to the previous kernels; here 4-byte aligned x and scratch are a precondition the program binding
#   enforces.

BLOB = kernels.S2_BLOB_BYTES


def fill_blob(rng):
    """fill_blob (:84-88): random code bytes, then fp16 scales of either sign with magnitude U[1e-3, 3e-2]."""
    b = np.empty(BLOB, np.uint8)
    b[:O_GU_SCALES] = rng.integers(0, 256, O_GU_SCALES)
    n = (BLOB - O_GU_SCALES) // 2
    f = np.where(rng.integers(0, 2, n) == 1, F32(-1), F32(1)) * rng.uniform(1e-3, 3e-2, n).astype(F32)
    b[O_GU_SCALES:] = emu.f16_from_f32(f).view(np.uint8)
    return b


def fill_x(rng, n_tok):
    """fill_x (:90-106): per 32-value chunk an fp32 scale U[2e-3, 5e-2] (and its fp16 as the block's d), 32 codes
    uniform in [-127, 127], one chunk in 16 all 127 and one all -127 (saturated on purpose)."""
    xs = rng.uniform(2e-3, 5e-2, n_tok * NC_GU).astype(F32)
    blk = np.zeros((n_tok * NC_GU, 34), np.uint8)
    blk[:, :2] = emu.f16_from_f32(xs).view(np.uint8).reshape(-1, 2)
    mode = rng.integers(0, 16, len(blk))
    v = rng.integers(-127, 128, (len(blk), 32))
    v[mode == 0], v[mode == 1] = 127, -127
    blk[:, 2:] = v.astype(np.int8).view(np.uint8)
    return blk.reshape(-1), xs


class Fixture:
    """make_blobs(12) (:244-252) in one buffer, the last blob also alone in a second buffer (the test's mapped host
    blob, :372-376: on this Mac every buffer is host memory; what carries over is a blob outside the arena), and 8
    tokens' activation rows with their fp32 scales."""

    def __init__(self, seed=20260928, nb=12):
        self.rng = np.random.default_rng(seed)
        self.nb = nb
        self.h = np.stack([fill_blob(self.rng) for _ in range(nb)])
        dev = gpu.device()
        self.bufs = [gpu.nt.Buffer(dev, self.h.tobytes()), gpu.nt.Buffer(dev, self.h[nb - 1].tobytes())]
        self.x, self.xs = fill_x(self.rng, 8)

    def blob(self, ref):
        """The host bytes of a group's blob, ``ref`` = (buffer index, byte offset)."""
        return self.h[self.nb - 1] if ref[0] == 1 else self.h[ref[1] // BLOB]


@pytest.fixture(scope="module")
def fx():
    return Fixture()


def groups_by_expert(slots):
    """Hits to groups (one a distinct expert, first-appearance order, entries in hit order): (experts, grp_start, the
    hits in entry order)."""
    experts, members = [], {}
    for h, s in enumerate(slots):
        if s not in members:
            experts.append(s)
            members[s] = []
        members[s].append(h)
    order = [h for s in experts for h in members[s]]
    return experts, [int(v) for v in np.cumsum([0] + [len(members[s]) for s in experts])], order


def regions(e, cap):
    """Entry e's slices of the scratch bytes: its gate (SwiGLU) row, up row, intermediate blocks and their scales."""
    sc = kernels.basalt_s2_scratch(cap)
    blk = NC_D * kernels.Q8_0_BLOCK_BYTES
    return [slice(4 * e * FF, 4 * (e + 1) * FF), slice(4 * (cap + e) * FF, 4 * (cap + e + 1) * FF),
            slice(sc["hq"] + e * blk, sc["hq"] + (e + 1) * blk),
            slice(sc["hs"] + 4 * e * NC_D, sc["hs"] + 4 * (e + 1) * NC_D)]


def per_entry(fx, grp, grp_start, n_groups, ent_tok, ent_dst, cap_groups, cap, n_rows, x, xs, src=None):
    """Each entry of the plan alone (one group of one entry, the same capacities and entry arrays): the same kernels'
    arithmetic one entry a call, the contract's comparand for basalt's previous kernels. Returns the output and scratch
    the grouped call must write: entry e's row and scratch slices from call e, the rest (every capacity slot past the
    plan, the padding) from the first call, after checking each call left every other output row at its sentinel."""
    out = np.full((n_rows, H), 0xA5A5A5A5, np.uint32)
    scratch = None
    for g in range(n_groups):
        for e in range(grp_start[g], min(grp_start[g + 1], grp_start[g] + kernels.S2_GMAX)):
            r = gpu.s2_experts(fx.bufs, [grp[g]], [e, e + 1], 1, ent_tok, ent_dst, x, xs, cap_groups, cap, n_rows,
                               src=src)
            d = int(ent_dst[e])
            assert np.all(np.delete(r["out"], d, axis=0) == 0xA5A5A5A5), ("stray store", e)
            out[d] = r["out"][d]
            if scratch is None:
                scratch = r["scratch"].copy()
            for sl in regions(e, cap):
                scratch[sl] = r["scratch"][sl]
    return out, scratch


def row_ref(codes, scales, xq, xs):
    """row_ref (:110-128) for rows ``codes`` (R, bytes) / fp16 ``scales`` (R, n / 64) against one activation row:
    (ref, mag), sum_c dw * dx * sum_e (code_e - 1) * x_e in float64 and the sum of |terms|."""
    nk = codes.shape[1] // 8
    w = ((codes[:, np.arange(32 * nk) // 4].astype(np.int64) >> (2 * (np.arange(32 * nk) % 4))) & 3) - 1
    x = np.asarray(xq, np.uint8).reshape(nk, 34)[:, 2:].view(np.int8).astype(np.int64)
    s = (w.reshape(-1, nk, 32) * x[None]).sum(axis=-1).astype(np.float64)
    t = emu.h2f(scales).astype(np.float64)[:, np.arange(nk) // 2] * np.asarray(xs, np.float64)[None] * s
    return t.sum(axis=1), np.abs(t).sum(axis=1)


def blob_parts(b):
    """A blob's gate/up codes (1280 slots) and fp16 scale bits, its down codes and scale bits."""
    return (b[: O_D_CODES].reshape(2 * FF, H // 4), b[O_GU_SCALES: O_D_SCALES].view(np.uint16).reshape(2 * FF, H // 64),
            b[O_D_CODES: O_GU_SCALES].reshape(H, FF // 4), b[O_D_SCALES:].view(np.uint16).reshape(H, FF // 64))


def reference(fx, r, plan, cap, x, xs):
    """reference (:200-235): every 7th up row of every entry against the activation row it read, every 13th down row
    against the run's own intermediate; criterion, quoted: ``return std::isfinite(got) && std::fabs((double) got - ref)
    <= 4e-5 * mag + 1e-30;`` (:131) and ``if (bad || checked == 0) ++g_fail;`` (:234). Gate rows are not referenced:
    the SwiGLU overwrites them in place."""
    grp, grp_start, n_groups, ent_tok, ent_dst = plan
    sc = kernels.basalt_s2_scratch(cap)
    gu = r["scratch"][: 4 * 2 * cap * FF].view(F32)
    hq, hs = r["scratch"][sc["hq"]:], r["scratch"][sc["hs"]: sc["hs"] + 4 * cap * NC_D].view(F32)
    out = r["out"].view(F32)
    checked = bad = 0
    worst = 0.0
    up_rows, d_rows = np.arange(0, FF, 7), np.arange(0, H, 13)
    for g in range(n_groups):
        cgu, sgu, cd, sd = blob_parts(fx.blob(grp[g]))
        for e in range(grp_start[g], min(grp_start[g + 1], grp_start[g] + kernels.S2_GMAX)):
            t = int(ent_tok[e])
            for (ref, mag), got in (
                    (row_ref(cgu[2 * up_rows + 1], sgu[2 * up_rows + 1], x[t * XROW: (t + 1) * XROW],
                             xs[t * NC_GU: (t + 1) * NC_GU]), gu[(cap + e) * FF + up_rows]),
                    (row_ref(cd[d_rows], sd[d_rows], hq[e * NC_D * 34: (e + 1) * NC_D * 34],
                             hs[e * NC_D: (e + 1) * NC_D]), out[int(ent_dst[e]), d_rows])):
                with np.errstate(invalid="ignore", over="ignore"):
                    ok = np.isfinite(got) & (np.abs(got.astype(np.float64) - ref) <= 4e-5 * mag + 1e-30)
                    rel = np.abs(got.astype(np.float64) - ref) / np.where(mag > 0, mag, np.inf)
                checked += ok.size
                bad += np.count_nonzero(~ok)
                worst = max(worst, float(np.nanmax(rel)))
    return checked, bad, worst


def emulation(fx, r, plan, cap, x, xs):
    """Every stage bit for bit against tests/kernels/basalt_emu.py (s2_experts.md): each entry's up rows, its SwiGLU
    row from the emulated gate rows (the exponential taken from the GPU's expf_neg), the intermediate's quantizer over
    every capacity entry, and the down rows from the GPU's intermediate."""
    from tests.kernels.test_basalt_experts import gpu_expf_neg
    grp, grp_start, n_groups, ent_tok, ent_dst = plan
    sc = kernels.basalt_s2_scratch(cap)
    s = r["scratch"]
    gu = s[: 4 * 2 * cap * FF].view(F32)
    for g in range(n_groups):
        cgu, sgu, cd, sd = blob_parts(fx.blob(grp[g]))
        ents = np.arange(grp_start[g], min(grp_start[g + 1], grp_start[g] + kernels.S2_GMAX))
        if not ents.size:
            continue
        toks = np.asarray(ent_tok)[ents]
        xq = np.stack([x[t * XROW: (t + 1) * XROW] for t in toks])
        xr = np.stack([xs[t * NC_GU: (t + 1) * NC_GU] for t in toks])
        gate, up = emu.s2_dots(cgu[0::2], sgu[0::2], xq, xr), emu.s2_dots(cgu[1::2], sgu[1::2], xq, xr)
        got_up = np.stack([gu[(cap + e) * FF: (cap + e + 1) * FF] for e in ents])
        assert np.array_equal(got_up.view(np.uint32), up.view(np.uint32)), ("up", g)
        h = emu.s2_swiglu(gate, up, gpu_expf_neg(gate).reshape(gate.shape))
        got_h = np.stack([gu[e * FF: (e + 1) * FF] for e in ents])
        assert np.array_equal(got_h.view(np.uint32), h.view(np.uint32)), ("swiglu", g)
    hq, hs = emu.q8_0_scaled(gu[: cap * FF])
    assert np.array_equal(s[sc["hq"]: sc["hq"] + hq.size], hq), "intermediate"
    assert np.array_equal(s[sc["hs"]: sc["hs"] + 4 * hs.size], hs.view(np.uint8)), "intermediate scales"
    blk = NC_D * 34
    for g in range(n_groups):
        cgu, sgu, cd, sd = blob_parts(fx.blob(grp[g]))
        ents = list(range(grp_start[g], min(grp_start[g + 1], grp_start[g] + kernels.S2_GMAX)))
        if not ents:
            continue
        want = emu.s2_dots(cd, sd, np.stack([hq[e * blk: (e + 1) * blk] for e in ents]),
                           np.stack([hs[e * NC_D: (e + 1) * NC_D] for e in ents]))
        for k, e in enumerate(ents):
            assert np.array_equal(r["out"][int(ent_dst[e])], want[k].view(np.uint32)), ("down", e)


def check_call(name, fx, grp, grp_start, n_groups, ent_tok, ent_dst, cap_groups, cap, n_rows, resident=None, x=None,
               xs=None):
    """``twice`` (:149-191) with the contract's comparand: the grouped call from sentinel-filled buffers (out 0xA5,
    scratch 0x5A, :153-154) against the per-entry calls, the output and the scratch (gate/up, the quantized
    intermediate, its scales) byte for byte, quoted: ``diff_out += std::memcmp(&r[0].out[i], &r[1].out[i], 4) != 0;``
    (:175) and ``const bool ok = diff_out + diff_gu + diff_q8 + diff_hs == 0;`` (:182); then reference() and the
    emulation. Returns the grouped run and its plan."""
    x, xs = (fx.x, fx.xs) if x is None else (x, xs)
    r = gpu.s2_experts(fx.bufs, grp, grp_start, n_groups, ent_tok, ent_dst, x, xs, cap_groups, cap, n_rows,
                       resident=resident)
    if resident is not None:                                 # the device's plan, as basalt's test reads it back (:446)
        n_groups, grp_start = int(r["n_groups"][0]), [int(v) for v in r["grp_start"]]
        ent_tok, ent_dst = r["ent_tok"], r["ent_dst"]
        grp = [(0, int(a) - fx.bufs[0].gpu_address) for a in r["grp_ptr"][:n_groups]]
    plan = (grp, grp_start, n_groups, ent_tok, ent_dst)
    want_out, want_s = per_entry(fx, *plan, cap_groups, cap, n_rows, x, xs)
    sc = kernels.basalt_s2_scratch(cap)
    diff_out = np.count_nonzero(r["out"] != want_out)
    diff = r["scratch"] != want_s
    diff_gu, diff_q8, diff_hs = (np.count_nonzero(diff[a:b]) for a, b in ((0, sc["hq"]), (sc["hq"], sc["hs"]),
                                                                           (sc["hs"], sc["bytes"])))
    ok = diff_out + diff_gu + diff_q8 + diff_hs == 0
    assert ok, (name, diff_out, diff_gu, diff_q8, diff_hs)
    checked, bad, worst = reference(fx, r, plan, cap, x, xs)
    assert not (bad or checked == 0), (name, checked, bad, worst)    # basalt: if (bad || checked == 0) ++g_fail;
    emulation(fx, r, plan, cap, x, xs)
    print(f"\n{name}: bitwise identical to {grp_start[n_groups] - grp_start[0]} per-entry calls; host reference: "
          f"{checked} rows, {bad} outside tolerance, worst {worst:.2e} of sum|term|; emulation bit for bit")
    return r, plan


def test_grouped_8_tokens(fx):
    """Section 4 (:369-425): moe_grouped_s2 with 8 tokens x 10 entries in groups of 8, 1, 5, 2, 7, 3, 1, 6, 4, ...,
    group 3 on the blob in its own buffer (the mapped host blob), the others blob gi % 11 of the arena, destinations a
    shuffle of the 80 rows, each entry reading its own token's row (dst / 10), cap_groups = groups + 3 with the table
    padded by group 0's address and the starts by the last. Only the fp32-scale run: a null x_scales (the fp16 d run)
    selects quantize_q8_0, which the drafter never takes (it allocates hit_xs_, mtp.cpp:438)."""
    n = 8 * K
    sizes = [8, 1, 5, 2, 7, 3, 1, 6, 4, 8, 1, 2, 3, 4, 5, 6, 7, 1, 1, 5]
    order = np.random.default_rng(4).permutation(n)
    grp, gstart, edst = [], [0], []
    at = gi = 0
    while at < n:
        sz = min(sizes[gi % 20], n - at)
        grp.append((1, 0) if gi == 3 else (0, (gi % (fx.nb - 1)) * BLOB))
        edst += [int(o) for o in order[at: at + sz]]
        at += sz
        gstart.append(at)
        gi += 1
    r, _ = check_call(f"moe_grouped_s2 ({gi} groups, fp32 scales)", fx, grp, gstart, gi, [d // K for d in edst], edst,
                      gi + 3, n, n)
    assert np.count_nonzero(r["out"] == 0xA5A5A5A5) == 0                 # every row written


def test_group_resident_drafter_shape(fx):
    """Section 5 (:427-460): moe_group_resident then moe_grouped_s2 at the drafter layer's shape, 5 tokens x 10 routed
    ids (distinct within a token, shared across tokens, from 12 experts resident at base + id * BLOB), cap_groups =
    cap_entries = 50 (mtp.cpp:931-935); the plan read back from the device as basalt's test does, and checked against a
    host transcription of group_resident_kernel (groups in first-appearance order, entries in routing order)."""
    T, n, rng = 5, 5 * K, np.random.default_rng(5)
    ids = np.concatenate([rng.permutation(fx.nb)[:K] for _ in range(T)]).astype(np.int32)
    r, (grp, gstart, ng, etok, edst) = check_call("moe_group_resident + moe_grouped_s2 (fp32)", fx, None, None, 0,
                                                    None, None, n, n, n, resident=(ids, K, 0, 0))
    experts, want_start, order = groups_by_expert(list(ids))
    assert ng == len(experts) and gstart[: ng + 1] == want_start and r["n_groups"][1] == n
    assert [o for _, o in grp] == [int(e) * BLOB for e in experts]
    assert list(edst[:n]) == order and list(etok[:n]) == [i // K for i in order]


def test_verify_window_rows(fx):
    """Section 3's verify window (:339-367) on the grouped kernels: 4 tokens x 10 routed entries, 29 hits (routing
    order, each on a random blob), grouped by expert, each entry reading its own token's row (dst / 10); 40-entry
    capacity and a device group count below cap_groups. basalt runs this section on the per-hit kernels
    (moe_hit_grouped_s2_multi), which the drafter does not call; the property it checks, rows per token, is
    moe_grouped_s2's ent_tok."""
    n, count, rng = 4 * K, 29, np.random.default_rng(3)
    ent_ids = rng.permutation(n)
    slot = rng.integers(0, fx.nb, n)
    dst = np.sort(ent_ids[:count])
    experts, gstart, order = groups_by_expert([int(s) for s in slot[:count]])
    edst = [int(dst[h]) for h in order]
    check_call("moe_grouped_s2 (4 tokens, 29 entries)", fx, [(0, s * BLOB) for s in experts], gstart, len(experts),
               [d // K for d in edst], edst, n, n, n)


def test_device_count(fx):
    """Section 2's device count (:304-337) on the grouped kernels: capacity 10, 7 entries counted (blob (7i + 1) mod 12,
    destinations a shuffle of the 10 rows), one token; the group count on the device, below cap_groups; quoted: ``if
    (touched) ++g_fail;`` (:336), the rows of the entries past the count keep the sentinel. basalt runs it on the
    per-hit kernels (moe_hit_grouped_s2_dev), which the drafter does not call."""
    cap, count = 10, 7
    slot = [(i * 7 + 1) % fx.nb for i in range(cap)]
    dst = np.random.default_rng(2).permutation(cap)
    experts, gstart, order = groups_by_expert(slot[:count])
    edst = [int(dst[h]) for h in order] + [int(d) for d in dst[count:]]
    r, _ = check_call("moe_grouped_s2 (count 7 of 10)", fx, [(0, s * BLOB) for s in experts], gstart, len(experts),
                      [0] * cap, edst, cap, cap, cap)
    touched = np.count_nonzero(r["out"][dst[count:]] != 0xA5A5A5A5)
    assert touched == 0, touched


def test_whole_expert_against_float64():
    """The contract's double-precision reference for the drafter expert (docs/design/flash-next.md): basalt checks
    swiglu_kernel only by comparing a run of its old path with one of its new, both of which launch it
    (s2_expert_grouped.cu:1108), and its float64 reference covers the up rows and the down rows from the run's own
    intermediate (s2_expert_grouped_parity.cpp:200-235). Here the quantizer on fp32 activations, then
    moe_grouped_s2 (gate/up, the SwiGLU, the intermediate's quantizer, down), against a float64 expert on the fp32
    activations, built as test_basalt_experts.py's test_whole_expert_against_float64 builds the routed expert's
    (native_expert_parity.cpp:46-78, :387-402): 3 tokens of N(0, 1) values, row r's weights
    0.02 (0.5 + (r % 7) / 7) N(0, 1) quantized to Q2_0 (quantize_q2_0) and laid out as the drafter's blob, dequantized
    on the host by ggml's dequantize_row_q2_0 (tests/contract/gguf_reference.py); g and u in double,
    h = (float) (g / (1 + exp(-g)) * u) with numpy's exp (no device function), out = D h in double. Criterion, the
    nearest basalt bound, quoted: ``const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);``
    (native_expert_parity.cpp:368), eg the relative L1 error (:38-42), on out and on the SwiGLU half against h."""
    nt, rng = 3, np.random.default_rng(17)

    def q2(rows, cols):
        scale = (0.02 * (0.5 + (np.arange(rows) % 7) / 7.0)).astype(F32)[:, None]
        return quantize_q2_0(scale * rng.standard_normal((rows, cols)).astype(F32)).reshape(rows, cols // 64, 18)

    G, U, D = q2(FF, H), q2(FF, H), q2(H, FF)
    gu = np.stack([G, U], axis=1).reshape(2 * FF, H // 64, 18)               # slot 2r gate row r, 2r + 1 up row r
    blob = np.concatenate([gu[..., 2:].reshape(-1), D[..., 2:].reshape(-1), gu[..., :2].reshape(-1),
                           D[..., :2].reshape(-1)])
    assert blob.size == BLOB
    Wg, Wu, Wd = (gguf_reference.dequantize(42, m.reshape(len(m), -1)).astype(np.float64) for m in (G, U, D))
    x = rng.standard_normal((nt, H)).astype(F32)
    g, u = x.astype(np.float64) @ Wg.T, x.astype(np.float64) @ Wu.T
    h = (g / (1.0 + np.exp(-g)) * u).astype(F32)
    ref = (h.astype(np.float64) @ Wd.T).astype(F32)
    xq, xs = gpu.q8_0_scaled(x.reshape(-1))
    r = gpu.s2_experts([gpu.nt.Buffer(gpu.device(), blob.tobytes())], [(0, 0)], [0, nt], 1, np.arange(nt),
                       np.arange(nt), xq, xs, 1, nt, nt)
    got_h, got = r["scratch"][: 4 * nt * FF].view(F32).reshape(nt, FF), r["out"].view(F32)
    rel = lambda a, b: np.abs(a.astype(np.float64) - b).sum() / (np.abs(b.astype(np.float64)).sum() + 1e-30)
    eh, eg = rel(got_h, h), rel(got, ref)
    print(f"\ndrafter expert Q2_0 {H} x {FF}, {nt} tokens against float64: SwiGLU rel {eh:.2e}, out rel {eg:.2e}")
    # basalt: const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
    assert np.isfinite(eg) and eg < 3e-2 and np.isfinite(got).all(), eg
    assert np.isfinite(eh) and eh < 3e-2 and np.isfinite(got_h).all(), eh


# ---- the IEEE helpers, bit for bit against numpy's IEEE float32 ------------------------------------------------------

def signed(rng, n, lo, hi):
    """n float32 of random sign with log2 |x| uniform in [lo, hi) (subnormals by their bit patterns below -126)."""
    v = rng.choice([-1.0, 1.0], n) * np.exp2(rng.uniform(lo, hi, n))
    sub = np.abs(v) < 2.0 ** -126
    v = v.astype(F32)
    v[sub] = (np.sign(v[sub]) * rng.integers(1, 1 << 23, np.count_nonzero(sub)).astype(np.uint32).view(F32)).astype(F32)
    return v


def helper_case(name, body, plain, want, *args, min_diff=10):
    """The helper (``body``) bit for bit against numpy (``want``), NaN as NaN; the plain Metal operation (``plain``)
    must differ from numpy on at least ``min_diff`` inputs, or the planted inputs would not test the helper."""
    got, raw = gpu_eval(body, *args), gpu_eval(plain, *args)
    nan = np.isnan(want)
    ok = (same_bits(got, want) & ~nan) | (np.isnan(got) & nan)
    assert ok.all(), (name, np.flatnonzero(~ok)[:8], [a[np.flatnonzero(~ok)[:4]] for a in args])
    diff = np.count_nonzero(~((same_bits(raw, want) & ~nan) | (np.isnan(raw) & nan)))
    assert diff >= min_diff, (name, diff)
    return diff


def test_ieee_helpers():
    """basalt_q8_1.metal's ieee_div127, ieee_rcp, ieee_fma (an integer n) and ieee_div (b >= 1) bit for bit against
    numpy's IEEE float32 (fma: tests/kernels/basalt_emu.py's fma32) on planted inputs: subnormal and tiny operands and
    results, the 2^-126 boundary, cancellations to a subnormal sum, ties broken by a subnormal addend, overflow to
    infinity, signed zeros, infinities and NaN; each with the plain Metal operation differing on them (the negative
    control)."""
    rng = np.random.default_rng(12)
    n = 1 << 16
    with np.errstate(all="ignore"):
        amax = np.abs(np.concatenate([signed(rng, n, -149, -90), np.uint32(np.arange(1, 4096)).view(F32),
                                      F32([0.0, 2.0 ** -126, 2.0 ** -119, 2.0 ** -100, 1.0, 3e38, np.inf])]))
        d127 = helper_case("ieee_div127", "r = ieee_div127(a);", "r = a / 127.0f;", amax / F32(127), amax)
        s = np.abs(np.concatenate([signed(rng, n, -149, -95), F32([0.0, -0.0, 2.0 ** -149, 2.0 ** -128, 2.0 ** -127,
                                                                   2.0 ** -126, 2.0 ** -100, 1.0, 2.0 ** 120,
                                                                   np.inf, np.nan])]))
        s[-10] = -0.0
        rcp = helper_case("ieee_rcp", "r = ieee_rcp(a);", "r = 1.0f / a;", F32(1) / s, s)
        # fma(a, n, c): a and c over every magnitude class, n an integer up to 2^13 (|s - hx| <= 8128)
        a = np.concatenate([signed(rng, n, -149, -90), signed(rng, n, -100, 10), signed(rng, n, -149, -90),
                            signed(rng, n, -140, -95), signed(rng, 4096, 60, 120)])
        c = np.concatenate([signed(rng, n, -149, -90), signed(rng, n, -149, -100), signed(rng, n, -100, 70),
                            signed(rng, n, -149, -90), signed(rng, 4096, -149, -127)])
        k = rng.integers(-8128, 8129, a.size).astype(F32)
        cancel = rng.integers(0, a.size, 8192)                     # c = -a * k + a few units: a subnormal sum
        exact = -(a[cancel].astype(np.float64) * k[cancel])
        c[cancel] = (exact + rng.integers(-40, 40, 8192) * 2.0 ** -149).astype(F32)
        tie_a = F32(1 + 2.0 ** -23) * np.exp2(rng.integers(-90, 60, 512)).astype(F32)       # a * 3: a tie of fp32
        extra = [(tie_a, np.full(512, 3, F32), np.where(rng.random(512) < 0.5, F32(1), F32(-1))
                  * np.uint32(rng.integers(1, 1 << 23, 512)).view(F32)),                  # broken by a subnormal c
                 (F32([0.0, -0.0, 0.0, 2.0 ** -90, np.inf, np.nan, 2.0 ** -110, 2.0 ** -110, 2.0 ** -110]),
                  F32([5, 3, 0, 0, 0, 2, 7, 7, 7]),
                  np.concatenate([np.uint32([3, 0x80000005, 9, 0x80000001]).view(F32),
                                  F32([2.0 ** -140, 1.0, 2.0 ** 70, -(2.0 ** 63), np.inf])]))]
        for ea, ek, ec in extra:
            a, k, c = np.concatenate([a, ea]), np.concatenate([k, ek]), np.concatenate([c, ec])
        want = emu.fma32(a, k, c)
        fma_diff = helper_case("ieee_fma", "r = ieee_fma(a, b, c);", "r = fma(a, b, c);", want, a, k, c)
        sub = np.count_nonzero((want != 0) & (np.abs(want) < F32(2.0 ** -126)))
        # a / b for b >= 1: the SwiGLU's quotient g / (1 + e)
        g = np.concatenate([signed(rng, n, -149, -90), signed(rng, n, -100, 0), signed(rng, 4096, -20, 7),
                            F32([0.0, -0.0, np.inf, -np.inf, np.nan, 2.0 ** -100, -(2.0 ** -126)])])
        b = np.concatenate([np.exp2(rng.uniform(0, 1.5, n)), np.exp2(rng.uniform(0, 60, n)),
                            np.exp2(rng.uniform(0, 128, 4096)), F32([2.0, 1.0, np.inf, 3.0, 2.0, np.inf, np.nan])])
        b = b.astype(F32)
        b[-13] = np.inf
        div_diff = helper_case("ieee_div", "r = ieee_div(a, b);", "r = a / b;", g / b, g, b)
    assert sub > 1000, sub
    print(f"\nIEEE helpers bit for bit: ieee_div127 {amax.size} ({d127} differ from the plain division), ieee_rcp "
          f"{s.size} ({rcp}), ieee_fma {a.size} ({fma_diff}; {sub} subnormal results), ieee_div {g.size} ({div_diff})")


def test_warp_sum_ieee():
    """warp_sum_ieee (basalt's warp_sum with IEEE adds) against the 32-lane butterfly of numpy's float32 adds, on lanes
    holding subnormal, tiny and cancelling values; the plain butterfly (warp_sum) must differ on them."""
    rng = np.random.default_rng(13)
    v = np.concatenate([signed(rng, 32 * 4096, -149, -95), signed(rng, 32 * 2048, -149, -60)])
    pair = rng.integers(0, v.size // 32, 2048) * 32
    v[pair + 1] = -v[pair] + np.uint32(rng.integers(1, 64, 2048)).view(F32)     # cancelling neighbours
    want = np.repeat(emu.butterfly(v.reshape(-1, 32), np.add), 32)
    diff = helper_case("warp_sum_ieee", "r = warp_sum_ieee(a);", "r = warp_sum(a);", want, v)
    print(f"\nwarp_sum_ieee: {v.size // 32} sums bit for bit ({diff} lanes differ with the plain butterfly)")


# ---- planted subnormals through the kernels, with the plain kernels as the negative control --------------------------

def plain_source():
    """The drafter kernels with each IEEE helper replaced by the plain Metal operation (what a port without them would
    compute): the negative control the planted tests must catch."""
    body = kernels.template("basalt_s2_experts.metal")
    for ieee, plain in (("ieee_fma(", "fma("), ("ieee_mul(", "mul_plain("), ("ieee_div(", "div_plain("),
                        ("ieee_div127(", "div127_plain("), ("ieee_rcp(", "rcp_plain("),
                        ("warp_sum_ieee(", "warp_sum(")):
        body = body.replace(ieee, plain)
    helpers = """
static inline float mul_plain(float a, float b) { return a * b; }
static inline float div_plain(float a, float b) { return a / b; }
static inline float div127_plain(float a) { return a / 127.0f; }
static inline float rcp_plain(float a) { return 1.0f / a; }
"""
    return (kernels.PRELUDE + kernels.template("basalt_fp.metal") + kernels.template("basalt_q8_1.metal") + helpers
            + body)


def hw_nan_source():
    """The drafter kernels storing the hardware's NaN (0x7fc00000 on this GPU) instead of cuda_nan's 0x7fffffff: the
    negative control of the NaN tests."""
    return (kernels.PRELUDE + kernels.template("basalt_fp.metal") + kernels.template("basalt_q8_1.metal")
            + kernels.template("basalt_s2_experts.metal").replace("cuda_nan(", "("))


def nan_bits(v):
    """The bit patterns of the NaNs among float32 values ``v`` (given as uint32)."""
    v = np.asarray(v).view(np.uint32).reshape(-1)
    return v[(v & 0x7FFFFFFF) > 0x7F800000]


def test_quantizer_planted_negative_control():
    """The planted blocks of test_quantize_q8_0_scaled through the plain kernel: it must get bytes wrong that the IEEE
    kernel gets right."""
    x = planted_blocks()
    want_b, want_s = emu.q8_0_scaled(x)
    blocks, scales = gpu.q8_0_scaled(x, src=plain_source())
    bad = np.count_nonzero(np.any(blocks.reshape(-1, 34) != want_b.reshape(-1, 34), axis=1)
                           | (scales.view(np.uint32) != want_s.view(np.uint32)))
    assert bad > 50, bad
    print(f"\nplain quantizer: {bad} of {x.size // 32} planted blocks wrong")


def test_planted_tiny_activations(fx):
    """moe_grouped_s2 on activation rows whose fp32 scales are subnormal or tiny (the quantizer's s for rows of tiny
    activations): products dw * dx, partial sums and butterfly sums below 2^-126, gate and up values whose SwiGLU
    quotient and product are subnormal, an intermediate quantized on subnormal scales. Every stage bit for bit against
    the emulation and the output and scratch against the per-entry calls; the plain kernels must differ."""
    rng = np.random.default_rng(14)
    x, _ = fill_x(rng, 4)
    xs = np.concatenate([signed(rng, NC_GU, -149, -100), np.abs(signed(rng, NC_GU, -78, -72)),
                         np.abs(signed(rng, NC_GU, -140, -60)), np.abs(signed(rng, NC_GU, -126, -110))]).astype(F32)
    xs = np.abs(xs)
    grp, gstart = [(0, 2 * BLOB), (1, 0), (0, 5 * BLOB)], [0, 4, 7, 10]
    etok, edst = [0, 1, 2, 3, 3, 1, 0, 2, 0, 1], list(rng.permutation(10))
    r, plan = check_call("moe_grouped_s2 (planted tiny activation scales)", fx, grp, gstart, 3, etok, edst, 3, 10, 10,
                         x=x, xs=xs)
    sc = kernels.basalt_s2_scratch(10)
    gu = r["scratch"][: 4 * 2 * 10 * FF].view(F32)
    hs = r["scratch"][sc["hs"]: sc["hs"] + 4 * 10 * NC_D].view(F32)
    tiny = lambda v: np.count_nonzero((v != 0) & (np.abs(v) < F32(2.0 ** -126)))
    with np.errstate(under="ignore"):
        prods = sum(tiny(emu.h2f(blob_parts(fx.blob(grp[g]))[1])[:, np.arange(NC_GU) // 2][None]
                         * np.stack([xs[t * NC_GU: (t + 1) * NC_GU] for t in etok[gstart[g]: gstart[g + 1]]])[:, None])
                    for g in range(3))
    counts = {"gate/up products": prods, "h": tiny(gu[: 10 * FF]), "h scales": tiny(hs)}
    assert all(v > 20 for v in counts.values()), counts
    p = gpu.s2_experts(fx.bufs, grp, gstart, 3, etok, edst, x, xs, 3, 10, 10, src=plain_source())
    wrong = {"out": np.count_nonzero(p["out"] != r["out"]),
             "scratch": np.count_nonzero(p["scratch"][: sc["hq"]].view(np.uint32)
                                         != r["scratch"][: sc["hq"]].view(np.uint32))}
    assert wrong["out"] > 100 and wrong["scratch"] > 100, wrong
    print(f"\nsubnormals in {counts}; the plain kernels differ in {wrong}")


def test_swiglu_planted():
    """basalt_s2_swiglu alone on planted gate and up values, bit for bit against the emulation (the exponential from
    the GPU): gates tiny and subnormal (g / 2 subnormal), +-0, in [-91, -86] (the exponential overflows past -88.72 and
    the quotient is -0) and up values making a subnormal product, large and non-finite gates; the plain kernel must
    differ."""
    from tests.kernels.test_basalt_experts import gpu_expf_neg
    rng = np.random.default_rng(15)
    cap = 8
    n = cap * FF
    g = np.concatenate([signed(rng, n // 4, -149, -90), rng.uniform(-91, -86, n // 4),
                        rng.standard_normal(n // 4) * 4, signed(rng, n // 4, -126, -60)]).astype(F32)
    u = np.concatenate([rng.standard_normal(n // 4), signed(rng, n // 4, -60, -1), rng.standard_normal(n // 4),
                        signed(rng, n // 4, -90, -40)]).astype(F32)
    g[:8] = [0.0, -0.0, np.inf, -np.inf, np.nan, 100.0, -100.0, 2.0 ** -149]
    gu = np.concatenate([g, u])
    params = kernels.basalt_s2_params(cap)

    def run(src):
        out = gpu.run(gpu.pipeline(src, "basalt_s2_swiglu"), [gu.tobytes()], params, 1,
                      -(-n // kernels.S2_THREADS), kernels.S2_THREADS)
        return np.frombuffer(out[0], F32)

    got, plain, hw = run(kernels.basalt_s2_experts_source()), run(plain_source()), run(hw_nan_source())
    want = emu.s2_swiglu(g, u, gpu_expf_neg(g))
    nan = np.isnan(want)
    assert np.array_equal(got[n:].view(np.uint32), u.view(np.uint32))          # the up half is read only
    assert np.all((same_bits(got[:n], want) & ~nan) | (np.isnan(got[:n]) & nan))
    # a NaN stored as sm_120's 0x7fffffff (g = -inf makes a fresh one: -inf / inf); the hardware's differs
    assert nan_bits(got[:n]).size >= 2 and np.all(nan_bits(got[:n]) == 0x7FFFFFFF), np.unique(nan_bits(got[:n]))
    assert np.any(nan_bits(hw[:n]) != 0x7FFFFFFF)
    diff = np.count_nonzero(~((same_bits(plain[:n], want) & ~nan) | (np.isnan(plain[:n]) & nan)))
    sub = np.count_nonzero((want != 0) & (np.abs(want) < F32(2.0 ** -126)))
    assert diff > 100 and sub > 100, (diff, sub)
    print(f"\nSwiGLU planted: {n} values bit for bit, {sub} subnormal; the plain kernel differs on {diff}")


def test_nan_bit_pattern(fx):
    """Every NaN the drafter kernels store is sm_120's canonical 0x7fffffff (s2_experts.md, NaN bit patterns), where
    this GPU's fma, add and division give 0x7fc00000. Token 0's activation row holds an infinity: its block's s is
    infinite and its codes 0, so each gate/up term fma(dw * inf, 0, acc) is NaN, and the SwiGLU of a NaN gate is NaN.
    Token 1's holds a finite 3e38: gate and up values near 1e37 whose SwiGLU products overflow, an intermediate block
    with an infinite s, and each down term fma(dw * inf, 0, acc) NaN. The same kernels storing the hardware's NaN must
    differ."""
    rng = np.random.default_rng(16)
    x = rng.standard_normal((2, H)).astype(F32)
    x[0, 5], x[1, 5] = np.inf, F32(3e38)
    xq, xs = gpu.q8_0_scaled(x.reshape(-1))

    def stored(src):
        r = gpu.s2_experts(fx.bufs, [(0, 0)], [0, 2], 1, [0, 1], [0, 1], xq, xs, 1, 2, 2, src=src)
        gu = r["scratch"][: 4 * 2 * 2 * FF].view(np.uint32)
        return {"up 0": gu[2 * FF: 3 * FF], "swiglu 0": gu[:FF], "gate/up": gu, "out 1": r["out"][1], "out": r["out"]}

    got, hw = stored(None), stored(hw_nan_source())
    pats = {k: (nan_bits(v).size, [hex(int(b)) for b in np.unique(nan_bits(v))]) for k, v in got.items()}
    assert all(pats[k][0] for k in ("up 0", "swiglu 0", "out 1")), pats
    assert all(np.all(nan_bits(v) == 0x7FFFFFFF) for v in got.values()), pats
    assert any(np.any(nan_bits(hw[k]) != 0x7FFFFFFF) for k in ("gate/up", "out"))
    print(f"\nNaN bit patterns stored: {pats}")
