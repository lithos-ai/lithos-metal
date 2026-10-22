"""The NaN every basalt port kernel stores is sm_120's (docs/research/basalt-nvcc/nan.md): 0x7fffffff in fp32 and
0x7fff in an fp16 half, where this GPU's arithmetic gives 0x7fc00000 and its half() 0x7e00.

Each test plants inputs that make a NaN reach a store the page marks (a NaN or infinite fp16 weight scale, an infinite
scale against a block of zero activations, NaN and infinite activations) and runs the kernel twice: as built, and as
the negative control with ``cuda_nan`` and ``cuda_nan_f16`` taken out of the kernel sources (``basalt_fp.metal``, which
defines them, unchanged). ``check`` then requires that every planted value is NaN and sm_120's, that the control
stores another NaN there, and that the rewrite changes nothing else: the same values are NaN in both runs, and every
other value has the same bits. The drafter experts' stores have their own tests (test_basalt_s2_experts.py). The
n-gram block's (basalt_gr_norm.metal, basalt_ple.metal) plant a quiet NaN in one input and also require that it reaches
exactly the values it must (the non-finite count) and that every other value keeps the bits of the run without it."""

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_ple_ref import D, EPS, H, HIST, N, ROWS, nonfinite
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels import basalt_ple_gpu as ple_gpu
from tests.kernels.test_basalt_experts import Call
from tests.kernels.test_basalt_mmvq import SCALES, weights
from tests.kernels.test_basalt_ple import block
from tests.kernels.test_basalt_q5k_head import weights as q5k_weights

F32 = np.float32
NAN16, INF16 = 0x7E00, 0x7C00        # an fp16 NaN and +inf, planted as weight or activation scales
QNAN = 0x7FC00000                    # the quiet NaN planted in the n-gram block's inputs
SOURCES = ("basalt_q8_1_source", "basalt_mmvq_source", "basalt_experts_source", "basalt_dequant_source",
           "basalt_mmvf_source", "basalt_dense_tc_source", "basalt_q5k_head_source", "basalt_gr_norm_source",
           "basalt_ple_source")


@pytest.fixture
def hw(monkeypatch):
    """``hw(call)``: ``call()`` with every port kernel storing this GPU's NaN, the negative control."""
    fp = kernels.PRELUDE + kernels.template("basalt_fp.metal")

    def run(call):
        with monkeypatch.context() as m:
            for name in SOURCES:
                src = getattr(kernels, name)()
                assert src.startswith(fp)
                body = src[len(fp):].replace("cuda_nan_f16(", "(").replace("cuda_nan(", "(")
                m.setattr(kernels, name, lambda s=fp + body: s)
            return call()
    return run


def check(got, ctl, at, fp16=False):
    """``at``: the values a planted NaN reaches. Returns how many there are."""
    u = np.uint16 if fp16 else np.uint32
    canon, inf = (0x7FFF, 0x7C00) if fp16 else (0x7FFFFFFF, 0x7F800000)
    g, c = (np.ascontiguousarray(a).view(u) for a in (got, ctl))
    gn, cn = (g & canon) > inf, (c & canon) > inf
    assert np.count_nonzero(at) and np.all(gn[at]), "a planted NaN did not reach the store"
    assert np.all(g[gn] == canon), [hex(int(v)) for v in np.unique(g[gn])]
    assert np.all(c[at] != canon), "the control stores sm_120's NaN too: the test has no power"
    assert np.array_equal(gn, cn) and np.array_equal(g[~gn], c[~gn]), "the rewrite changed more than NaN bits"
    return int(np.count_nonzero(at))


def put16(a, at, value):
    """Write the fp16 bits ``value`` into the byte array ``a`` at byte offset ``at``."""
    a.reshape(-1)[at: at + 2] = np.array([value], np.uint16).view(np.uint8)


def halves(q):
    """Each q8_1 block's (d, s) fp16 bits, and its codes."""
    b = np.asarray(q, np.uint8).reshape(-1, 36)
    return b[:, :4].copy().view(np.uint16), b[:, 4:]


# ---- the q8_1 quantizers -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["native", "ieee", "swiglu"])
def test_q8_1_halves(hw, kind):
    """q8_1_put: a NaN lane makes its block's sum NaN, an all-NaN block its scale and sum, +inf and -inf in one block
    its sum (an infinity alone is clamped to 65504). For the SwiGLU quantizer the gates make those values."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8 * 32).astype(F32)
    up = rng.standard_normal(x.size).astype(F32)
    x[32 + 3] = np.nan
    x[64: 96] = np.nan
    if kind == "swiglu":
        x[97], x[98], up[97], up[98] = np.inf, np.inf, 1.0, -1.0       # SiLU(inf) * +-1: +inf and -inf
    else:
        x[97], x[98] = np.inf, -np.inf
    x[160 + 4] = np.inf                                                 # block 5: no NaN
    run = lambda: halves(gpu.q8_1(x, kind, up if kind == "swiglu" else None))
    (got, codes), (ctl, ctl_codes) = run(), hw(run)
    at = np.zeros_like(got, bool)
    at[1, 1] = at[2, 0] = at[2, 1] = at[3, 1] = True
    assert check(got, ctl, at, fp16=True) == 4 and np.array_equal(codes, ctl_codes)


# ---- native_mmvq -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("type_id", sorted(SCALES))
def test_mmvq(hw, type_id):
    """Row 1's first block: its fp16 scales +inf against a first superblock of zero activations (inf * 0, a NaN made);
    row 2's: NaN (a NaN passed). One and three columns, at a small K (4 or 2 rows a threadgroup) and a large one (at
    4096 every type has at least BPI blocks a row: one row a threadgroup)."""
    for n_in in (512, 4096):
        rng = np.random.default_rng(type_id)
        n_out = 16
        w = weights(type_id, n_in, n_out, rng)
        for off in SCALES[type_id]:
            put16(w[1], off, INF16)
            put16(w[2], off, NAN16)
        for ncols in (1, 3):
            x = rng.standard_normal((ncols, n_in)).astype(F32)
            x[:, :256] = 0.0
            xq = gpu.q8_1(x.reshape(-1), "ieee")
            run = lambda: gpu.mmvq(type_id, w, xq, n_in, n_out, ncols, fill=0)
            got, ctl = run(), hw(run)
            at = np.zeros_like(got, bool)
            at[:, 1:3] = True
            check(got, ctl, at)


# ---- the dequantizers --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("type_id", kernels.DEQUANT_TYPES)
def test_dequant(hw, type_id):
    """Superblock 0's first block with a NaN scale; for Q2_0 also block 1 (values 64..127) with +inf, a NaN wherever
    its code is 1 (d * 0)."""
    rng = np.random.default_rng(2)
    n = 4 * 256
    raw = weights(type_id, 256, n // 256, rng)
    bb = emu.TRAITS[type_id][3]
    put16(raw[0], SCALES[type_id][0], NAN16)
    if type_id == 42:
        put16(raw[0], bb + SCALES[42][0], INF16)
    run = lambda: gpu.dequant(type_id, raw, n)
    got, ctl = run(), hw(run)
    at = np.zeros(n, bool)
    at[: emu.TRAITS[type_id][0]] = True                                # the NaN block's values
    if type_id == 42:
        codes = np.unpackbits(raw[0, bb + 2: bb + 18][:, None], axis=1, bitorder="little").reshape(-1, 2)
        at[64: 128] = codes @ [1, 2] == 1
    check(got, ctl, at)


# ---- the routed experts ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_embd, n_ff, gu2, sub16", [(2560, 640, True, True), (2560, 640, False, False),
                                                      (512, 256, False, False)])
def test_experts(hw, n_embd, n_ff, gu2, sub16):
    """Groups of 1, 2, 3 and 7 entries (each batch size its own store, six then one for the seven), every blob with a
    NaN scale in gate rows 5 and 8 (rows B and A of a two-row SIMD-group on the gu2 path) and in down row 7: gate[e, 5]
    and gate[e, 8], the q8_1 sum of the intermediate's first block, and out[ent_dst[e], 7] are NaN for every entry."""
    rng = np.random.default_rng(3)
    c = Call(rng, n_embd, n_ff, [1, 2, 3, 7], 5)
    lay = c.lay
    for o in c.grp_off:
        put16(c.arena, o + 5 * lay["gu_row"], NAN16)
        put16(c.arena, o + 8 * lay["gu_row"], NAN16)
        put16(c.arena, o + lay["down_off"] + 7 * lay["d_row"], NAN16)
    run = lambda: c.run(gu2=gu2, sub16=sub16, out_fill=0x00)
    got, ctl = run(), hw(run)
    ents = list(c.entries())
    at = np.zeros_like(got["gate"], bool)
    at[ents, 5] = at[ents, 8] = True
    check(got["gate"], ctl["gate"], at)
    assert np.array_equal(got["up"].view(np.uint32), ctl["up"].view(np.uint32))
    gh, gcodes = halves(got["hq"])
    ch, ccodes = halves(ctl["hq"])
    at = np.zeros_like(gh, bool)
    at[[e * (n_ff // 32) for e in ents], 1] = True                     # block 0 of each entry: its sum
    check(gh, ch, at, fp16=True)
    assert np.array_equal(gcodes, ccodes)
    at = np.zeros_like(got["out"], bool)
    at[c.ent_dst[ents], 7] = True
    check(got["out"], ctl["out"], at)


# ---- the BF16 GEMVs ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_in, n_tok", [(64, 1), (512, 1), (512, 3), (512, 6)])
def test_mmvf(hw, n_in, n_tok):
    """Row 1's first weight a NaN, row 2's +inf against a zero first activation (inf * 0): the one-row kernel and the
    multi kernel (NT 4, 8), threadgroups of 32 and of 256 threads."""
    rng = np.random.default_rng(4)
    n_out = 8
    w = emu.bf16_bits(rng.standard_normal((n_out, n_in)).astype(F32) * F32(0.05))
    w[1, 0], w[2, 0] = 0x7FA5, 0x7F80
    x = rng.standard_normal((n_tok, n_in)).astype(F32)
    x[:, 0] = 0.0
    run = lambda: gpu.mmvf(w, x, n_in, n_out, n_tok, fill=0).reshape(n_tok, n_out)
    got, ctl = run(), hw(run)
    at = np.zeros_like(got, bool)
    at[:, 1:3] = True
    check(got, ctl, at)


# ---- dense_tc ----------------------------------------------------------------------------------------------------

DTC_TYPES = (11, 12, 13, 14, 23, 42)


def dtc_case(type_id, rng, n_in=1024, n_out=32):
    """Weights with row 1's first block's scales +inf (against zero activations: a NaN made) and row 17's NaN (the
    second tile), and 8 columns whose first 256 values are 0."""
    w = weights(type_id, n_in, n_out, rng)
    for off in SCALES[type_id]:
        put16(w[1], off, INF16)
        put16(w[17], off, NAN16)
    x = rng.standard_normal((8, n_in)).astype(F32)
    x[:, :256] = 0.0
    return w, gpu.q8_1(x.reshape(-1), "native")


@pytest.mark.parametrize("type_id", DTC_TYPES)
def test_dense_tc(hw, type_id):
    """Every order (one, tile, s with KS 1 and 2) at 1, 3 and 8 columns."""
    w, xq = dtc_case(type_id, np.random.default_rng(5))
    for variant in ("one", "tile", "s1", "s2"):
        run = lambda: gpu.dense_tc(type_id, w, xq, 1024, 32, (1, 3, 8), variant, fill=0)
        got, ctl = run(), hw(run)
        for t in (1, 3, 8):
            at = np.zeros_like(got[t], bool)
            at[:t, 1] = at[:t, 17] = True
            check(got[t], ctl[t], at)


def test_dense_tc_pair(hw):
    """dense_tc_pair, both matrices planted (Q4_K then Q2_0), at 2 and 8 columns."""
    rng = np.random.default_rng(6)
    w0, xq = dtc_case(12, rng)
    w1, _ = dtc_case(42, rng)
    run = lambda: gpu.dense_tc_pair(12, w0, 32, 42, w1, 32, xq, 1024, (2, 8), fill=0)
    got, ctl = run(), hw(run)
    for t in (2, 8):
        for k in (0, 1):
            at = np.zeros_like(got[t][k], bool)
            at[:t, 1] = at[:t, 17] = True
            check(got[t][k], ctl[t][k], at)


# ---- the Q5_K head -----------------------------------------------------------------------------------------------

def test_q5k_head(hw):
    """basalt_q5k_head at n_in 512, one and three columns: row 1's first block's d +inf against zero activations, row
    2's NaN."""
    rng = np.random.default_rng(7)
    n_in, n_out = 512, 24
    w = q5k_weights(n_out, rng, n_in)
    put16(w[1], 0, INF16)
    put16(w[2], 0, NAN16)
    for ncols in (1, 3):
        x = rng.standard_normal((ncols, n_in)).astype(F32)
        x[:, :256] = 0.0
        xq = gpu.q8_1(x.reshape(-1), "native")
        run = lambda: gpu.q5k_head(w, xq, n_in, n_out, ncols)
        got, ctl = run(), hw(run)
        at = np.zeros_like(got, bool)
        at[:, 1:3] = True
        check(got, ctl, at)


def test_q5k_top1(hw):
    """q5k_head_top1 with row 0's logit NaN (its first block's d NaN): the logit, threadgroup 0's partial sum (its
    SIMD-group 0 holds only that row, so the threadgroup's merge starts from an empty top with a NaN sum) and the
    merged probability are NaN; the partial maxima and the id are not."""
    rng = np.random.default_rng(8)
    n_out = 640                                                         # 80 threadgroups: one row a SIMD-group
    w = q5k_weights(n_out, rng)
    put16(w[0], 0, NAN16)
    xq = gpu.q8_1(rng.standard_normal(2560).astype(F32), "native")
    run = lambda: gpu.q5k_top1(w, xq, n_out)
    (y, i, p, part), (cy, ci, cp, cpart) = run(), hw(run)
    at = np.zeros(n_out, bool)
    at[0] = True
    check(y, cy, at)
    at = np.zeros_like(part, bool)
    at[0, 2] = True
    check(part, cpart, at)
    check(np.array([p]), np.array([cp]), np.array([True]))
    assert i == ci


def test_q5k_proxy_score(hw):
    """px::score_kernel on a copy of 64 rows (two tiles). An activation block with a NaN scale makes every score NaN:
    each tile's maximum and its first candidate are NaN. Then, on positive activations, row 5 with every level 3 and
    every scale +inf scores +inf, so tile 0's sum takes exp(inf - inf); row 32's NaN scales give tile 1's first
    candidate a NaN."""
    rng = np.random.default_rng(9)
    rows, n_cap = 64, 64
    q5 = q5k_weights(rows, rng)
    proxy = gpu.q5k_proxy_pack(q5, n_cap, [(0, rows)])
    mp = np.arange(rows, dtype=np.int32)
    xq = gpu.q8_1((np.abs(rng.standard_normal(2560)) + 0.1).astype(F32), "native")

    xa = xq.copy()
    put16(xa, 0, NAN16)
    run = lambda: gpu.q5k_proxy_score(proxy, xa, rows, n_cap, mp)
    (cand, part), (ccand, cpart) = run(), hw(run)
    at = np.zeros_like(part, bool)
    at[:, 0] = True
    check(part, cpart, at)
    at = np.zeros_like(cand, bool)
    at[:, 0, 0] = True
    check(cand, ccand, at)

    pb = proxy.copy()                                    # per (tile, 64 weights, row): a 16-byte word, then two scales
    words = pb[: 2 * 40 * 32 * 16].reshape(2, 40, 32, 16)
    sc = pb[2 * 40 * 32 * 16:].view(np.uint16).reshape(2, 40, 32, 2)
    words[0, :, 5] = 0xFF
    sc[0, :, 5] = INF16
    sc[1, :, 0] = NAN16
    run = lambda: gpu.q5k_proxy_score(pb, xq, rows, n_cap, mp)
    (cand, part), (ccand, cpart) = run(), hw(run)
    at = np.zeros_like(part, bool)
    at[0, 1] = True
    check(part, cpart, at)
    at = np.zeros_like(cand, bool)
    at[1, 0, 0] = True
    check(cand, ccand, at)


# ---- the n-gram block ----------------------------------------------------------------------------------------------

def exactly(got, ctl, clean, at):
    """``check``, then the comparison helper's non-finite count over ``got`` (``nonfinite``, ple_parity.cpp:52-57)
    equal to the size of ``at``, and every value outside ``at`` with the bits of the run without the NaN, ``clean``.
    Returns the count."""
    n = check(got, ctl, at)
    assert nonfinite(got) == n, (
        "X-NF (spec T§8), \"The comparison helper's non-finite count over result must equal the size of that set\": "
        f"{nonfinite(got)} non-finite values, {n} reached")
    g, c = (np.ascontiguousarray(a, F32).view(np.uint32) for a in (got, clean))
    assert np.array_equal(g[~at], c[~at]), "a value the NaN does not reach changed its bits"
    return n


@pytest.mark.parametrize("row", [0, 4, 8])
def test_ple_conv(hw, row):
    """The 12-row chain (basalt_ple_gpu.chain) on the synthetic fixture with history element (channel i, row ``row``)
    a quiet NaN: only basalt_ple_conv reads the history, and row t's taps read rows t, t + 3 and t + 6 of the initial
    history though it advances in place, so the NaN reaches exactly conv[t][i] and result[t][i] for t in {row,
    row - 3, row - 6} ∩ [0, 12) (plan §4.8: "each count equals that set's size, and those values are 0x7fffffff")."""
    fx, _, clean = block("synthetic")
    i = 2 * N + 777                                                     # a channel of stream 2
    hist = np.ascontiguousarray(fx.hist.T).ravel()                      # a copy: row r of channel c at c * 9 + r
    hist.view(np.uint32)[i * HIST + row] = QNAN
    run = lambda: ple_gpu.chain(fx.emb, fx.hidden, hist, fx)
    got, ctl = run(), hw(run)
    reached = [t for t in (row, row - 3, row - 6) if 0 <= t < ROWS]
    at = np.zeros((ROWS, D), bool)
    at[reached, i] = True
    for k in ("conv", "result"):
        assert exactly(got[k], ctl[k], clean[k], at) == len(reached), (
            f"plan §4.8, \"each count equals that set's size\": {k}, t in {reached}")


def test_ple_norm(hw):
    """basalt_gr_norm at the block's call (4 groups of 2560, 4 gamma rows, eps f32(1e-6)) with element 1234 of group 2
    a quiet NaN: the group's sum and scale are NaN, so it reaches exactly that group's 2560 outputs, and every other
    group keeps its bits."""
    rng = np.random.default_rng(10)
    x, w = rng.standard_normal((H, N)).astype(F32), (1 + 0.1 * rng.standard_normal(D)).astype(F32)
    clean = ple_gpu.gr_norm(x, w, H, N, H, EPS)
    x.view(np.uint32)[2, 1234] = QNAN
    run = lambda: ple_gpu.gr_norm(x, w, H, N, H, EPS)
    got, ctl = run(), hw(run)
    at = np.zeros((H, N), bool)
    at[2] = True
    assert exactly(got, ctl, clean, at) == N, (
        "plan §4.8, \"It reaches exactly that group's 2560 outputs; every other group keeps its bits.\"")


def test_ple_broadcast(hw):
    """basalt_ple_broadcast with value[1234] a quiet NaN: it reaches exactly gated[s][1234] for the four streams s."""
    rng = np.random.default_rng(11)
    value, gate = rng.standard_normal(N).astype(F32), rng.uniform(0.1, 0.9, H).astype(F32)
    clean = ple_gpu.broadcast(value, gate).reshape(H, N)
    value.view(np.uint32)[1234] = QNAN
    run = lambda: ple_gpu.broadcast(value, gate).reshape(H, N)
    got, ctl = run(), hw(run)
    at = np.zeros((H, N), bool)
    at[:, 1234] = True
    assert exactly(got, ctl, clean, at) == H, (
        "plan §4.8, \"It reaches exactly `gated[s][j]` for the four streams s.\"")
