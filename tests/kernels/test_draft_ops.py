"""The DSpark round's kernels (#24) against numpy models of their contracts and the layer oracle: the DRAFT variant
of gqa_decode (three key sources, no mask, the injected positions appended), the confidence head, verify_select and
accept_scan on StepState, tap_concat, the block-ids embed and the row-source selection of the shared kernels."""

import struct
import sys
from pathlib import Path

import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gqa_decode import EPS, THETA, _bars, norm_rope_np, rbf  # noqa: E402

from monolith import kernels  # noqa: E402
from monolith.bench import pack_spec, random_spec  # noqa: E402
from monolith.core.step_state import StepStateLayout  # noqa: E402
from monolith.formats import FORMATS, PackLayout  # noqa: E402
from monolith.formats.fp import bf16_to_f32, f32_to_bf16  # noqa: E402
from monolith.nn.rope import rope_tables_permuted  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402

LAYOUT = StepStateLayout(t_max=8, gamma_max=7)


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _run(dev, ds):
    r = nt.Queue(dev).run(ds if isinstance(ds, list) else [ds])
    assert not r.error, r.error


def _spec_lib(dev, **macros):
    return nt.Library(dev, kernels.spec_ops_source(LAYOUT.to_msl()), {k: str(v) for k, v in macros.items()})


# ---- draft attention ------------------------------------------------------------------------------------------------

class DCfg:
    def __init__(self, heads, kv, d, ctx_max, gamma, chunk=64, rb=4):
        self.heads, self.kv, self.d, self.ctx_max, self.gamma, self.chunk, self.rb = heads, kv, d, ctx_max, gamma, chunk, rb
        self.rep = heads // kv
        hd, kd = heads * d, kv * d
        self.q_off, self.k_off, self.v_off = 0, hd, hd + kd
        self.n1, self.kvp_stride = hd + 2 * kd, 2 * kd
        self.n_chunks_max = -(-ctx_max // chunk)
        self.scaling = d ** -0.5


class DraftHarness:
    """gqa_decode DRAFT=1 + gqa_merge, the counts from params or from a StepState buffer."""

    def __init__(self, dev, cfg, qn, kn, step_state=False):
        self.dev, self.cfg, self.step_state = dev, cfg, step_state
        macros = dict(kernels.gqa_macros(cfg.d, chunk=cfg.chunk, rb_max=cfg.rb), DRAFT="1")
        src = kernels.gqa_source()
        if step_state:
            macros["STEP_STATE"] = "1"
            src = src.replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
        lib = nt.Library(dev, src, macros)
        self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode"), nt.Pipeline(lib, "gqa_merge")
        cos, sin = rope_tables_permuted(THETA, cfg.d, cfg.d, cfg.ctx_max)
        self.cos, self.sin = bf16_to_f32(f32_to_bf16(cos)), bf16_to_f32(f32_to_bf16(sin))
        self.qn, self.kn = qn.astype(np.float32), kn.astype(np.float32)
        self.cache_bytes = cfg.ctx_max * cfg.kv * cfg.d * 2
        self.k_cache, self.v_cache = nt.Buffer(dev, self.cache_bytes), nt.Buffer(dev, self.cache_bytes)
        self.k_cache.fill(0); self.v_cache.fill(0)
        po, pm = kernels.gqa_workspace(cfg.kv, cfg.n_chunks_max, cfg.rep * cfg.gamma, cfg.d)
        self.part_o, self.part_md = nt.Buffer(dev, po), nt.Buffer(dev, pm)
        self.bufs = {"cos": nt.Buffer(dev, f32_to_bf16(cos).tobytes()), "sin": nt.Buffer(dev, f32_to_bf16(sin).tobytes()),
                     "qn": nt.Buffer(dev, self.qn.tobytes()), "kn": nt.Buffer(dev, self.kn.tobytes())}
        self.n_sg = 12 * dev.info().gpu_cores

    def set_caches(self, k, v):
        self.k_cache.write(f32_to_bf16(k).tobytes(), 0)
        self.v_cache.write(f32_to_bf16(v).tobytes(), 0)

    def caches(self):
        c = self.cfg
        shape = (c.ctx_max, c.kv, c.d)
        return (bf16_to_f32(np.frombuffer(self.k_cache.read(0, self.cache_bytes), dtype=np.uint16).reshape(shape)),
                bf16_to_f32(np.frombuffer(self.v_cache.read(0, self.cache_bytes), dtype=np.uint16).reshape(shape)))

    def step(self, proj_bf16, kvp_bf16, ctx_len, n_new, done=0):
        c = self.cfg
        g = proj_bf16.shape[0]
        params = kernels.draft_attn_params(heads=c.heads, kv_heads=c.kv, gamma=g, ctx_len=ctx_len, n_new=n_new, n_sg=self.n_sg,
                                           q_off=c.q_off, k_off=c.k_off, v_off=c.v_off, in_stride=c.n1, kvp_stride=c.kvp_stride,
                                           out_stride=c.heads * c.d, ctx_max=c.ctx_max, eps=EPS, scaling=c.scaling, n_chunks_max=c.n_chunks_max)
        pb, kb = nt.Buffer(self.dev, proj_bf16.tobytes()), nt.Buffer(self.dev, max(kvp_bf16.nbytes, 16))
        if kvp_bf16.nbytes:
            kb.write(kvp_bf16.tobytes(), 0)
        out = nt.Buffer(self.dev, g * c.heads * c.d * 2); out.fill(0)
        d1 = (nt.Dispatch().pipeline(self.p_dec).buffer(0, pb).buffer(1, self.k_cache).buffer(2, self.v_cache)
              .buffer(3, self.bufs["cos"]).buffer(4, self.bufs["sin"]).buffer(5, self.bufs["qn"]).buffer(6, self.bufs["kn"])
              .buffer(7, self.part_o).buffer(8, self.part_md).bytes(9, params).buffer(11, kb)
              .grid(-(-(self.n_sg * 32) // 384)).threadgroup(384).barrier())
        d2 = (nt.Dispatch().pipeline(self.p_merge).buffer(0, self.part_o).buffer(1, self.part_md).buffer(2, pb).buffer(3, out)
              .bytes(4, params).grid(g * c.heads).threadgroup(32))
        if self.step_state:
            st = nt.Buffer(self.dev, LAYOUT.pack({"drafter_ctx_len": ctx_len, "n_inject": n_new, "done": done, "t_this_step": 1, "position": 999}))
            d1.buffer(15, st); d2.buffer(15, st)
        _run(self.dev, [d1, d2])
        return bf16_to_f32(np.frombuffer(out.read(0, g * c.heads * c.d * 2), dtype=np.uint16).reshape(g, c.heads * c.d))


def ref_draft(h, proj, kvp, ctx_len, n_new, k_cache, v_cache):
    """The kernel's contract: keys = cache[:ctx_len] ∪ the n_new context positions (from kvp, normed + RoPE'd,
    appended) ∪ the block (normed + RoPE'd, not appended); queries = the block at ctx_len + n_new + t; no mask;
    the chunked online softmax with the reference's roundings."""
    c = h.cfg
    g, d = proj.shape[0], c.d
    q = proj[:, c.q_off: c.q_off + c.heads * d].reshape(g, c.heads, d)
    k_blk = proj[:, c.k_off: c.k_off + c.kv * d].reshape(g, c.kv, d)
    v_blk = proj[:, c.v_off: c.v_off + c.kv * d].reshape(g, c.kv, d)
    k_new = kvp[:, : c.kv * d].reshape(n_new, c.kv, d)
    v_new = kvp[:, c.kv * d:].reshape(n_new, c.kv, d)
    qpos0 = ctx_len + n_new
    pq, pn = np.arange(qpos0, qpos0 + g), np.arange(ctx_len, qpos0)
    q = norm_rope_np(q, h.qn, h.cos[pq][:, None, :], h.sin[pq][:, None, :], d)
    k_blk = norm_rope_np(k_blk, h.kn, h.cos[pq][:, None, :], h.sin[pq][:, None, :], d)
    if n_new:
        k_new = norm_rope_np(k_new, h.kn, h.cos[pn][:, None, :], h.sin[pn][:, None, :], d)
        k_cache[ctx_len: qpos0] = k_new
        v_cache[ctx_len: qpos0] = v_new
    keys = np.concatenate([k_cache[:qpos0], k_blk])
    vals = np.concatenate([v_cache[:qpos0], v_blk])
    ctx = qpos0 + g
    out = np.zeros((g, c.heads, d), dtype=np.float32)
    for tt in range(g):
        for hh in range(c.heads):
            j = hh // c.rep
            s = rbf(rbf(keys[:, j].astype(np.float64) @ q[tt, hh].astype(np.float64)) * c.scaling)
            parts = []
            for c0 in range(0, ctx, c.chunk):
                sc = s[c0: c0 + c.chunk]
                m = sc.max()
                p = np.exp((sc - m).astype(np.float32)).astype(np.float32)
                parts.append((m, float(p.sum(dtype=np.float32)), rbf(p).astype(np.float64) @ vals[c0: min(c0 + c.chunk, ctx), j].astype(np.float64)))
            m_g = max(m for m, _, _ in parts)
            d_g = sum(dc * np.exp(m - m_g) for m, dc, _ in parts)
            out[tt, hh] = rbf(sum(oc * np.exp(m - m_g) for m, _, oc in parts) / d_g)
    return out.reshape(g, c.heads * d)


def _ulp(x: float) -> float:
    """One BF16 ulp at magnitude ``x`` (8 bits of mantissa)."""
    return 2.0 ** (math.floor(math.log2(max(x, 1e-30))) - 7)


@pytest.mark.parametrize("step_state", [False, True], ids=["params", "step_state"])
def test_draft_attn_matches_kernel_contract(dev, step_state):
    cfg = DCfg(4, 2, 128, 256, 7)
    rng = np.random.default_rng(21)
    qn, kn = (1.0 + rng.standard_normal(cfg.d) * 0.1).astype(np.float32), (1.0 + rng.standard_normal(cfg.d) * 0.1).astype(np.float32)
    h = DraftHarness(dev, cfg, qn, kn, step_state=step_state)
    k_ref, v_ref = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    ctx_len = 0
    # injections of 6, 3, 0 and 60 positions (the last crosses a chunk boundary), blocks of 7 then 3 rows
    for n_new, g in ((6, 7), (3, 7), (0, 7), (60, 3), (5, 7)):
        proj = f32_to_bf16(rng.standard_normal((g, cfg.n1)).astype(np.float32))
        kvp = f32_to_bf16(rng.standard_normal((n_new, cfg.kvp_stride)).astype(np.float32))
        got = h.step(proj, kvp, ctx_len, n_new)
        ref = ref_draft(h, bf16_to_f32(proj), bf16_to_f32(kvp), ctx_len, n_new, k_ref, v_ref)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.99999 and max_abs <= 2 * _ulp(scale), (ctx_len, n_new, g, cos, max_abs, scale)   # 2 BF16 ulps (the run-time chunk)
        kc, vc = h.caches()
        n = ctx_len + n_new
        assert np.abs(kc[:n] - k_ref[:n]).max() <= 1e-2 * max(np.abs(k_ref[:n]).max(), 1e-6)
        assert np.array_equal(vc[:n], v_ref[:n])
        assert np.all(kc[n:] == 0) and np.all(vc[n:] == 0)                  # the block is never appended
        ctx_len = n
    if step_state:
        proj = f32_to_bf16(rng.standard_normal((7, cfg.n1)).astype(np.float32))
        assert np.all(h.step(proj, np.zeros((0, cfg.kvp_stride), np.uint16), ctx_len, 0, done=1) == 0)   # done: returns at once


def test_draft_attn_matches_layer_oracle(dev):
    """The DraftAttention torch oracle (HF-faithful) vs the kernel on the same projections, caches and positions."""
    torch = pytest.importorskip("torch")
    from monolith.spec.dspark import DSparkConfig
    from monolith.spec.dspark.model import DraftAttention

    heads, kv, d, ctx_max, gamma = 4, 2, 64, 128, 5
    hidden = heads * d                                                       # identity o_proj exposes the attention output
    cfgd = DSparkConfig.from_dict({"hidden_size": hidden, "intermediate_size": 8, "num_hidden_layers": 1, "num_attention_heads": heads,
                                   "num_key_value_heads": kv, "head_dim": d, "rms_norm_eps": EPS, "vocab_size": 8, "rope_theta": THETA,
                                   "block_size": gamma, "target_layer_ids": [0], "mask_token_id": 1})
    attn = DraftAttention(cfgd, hf_prefix="x.", prefix="l.", max_context=ctx_max)
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    wq, wk, wv = (torch.randn(heads * d, hidden) * 0.05).to(torch.bfloat16), (torch.randn(kv * d, hidden) * 0.05).to(torch.bfloat16), (torch.randn(kv * d, hidden) * 0.05).to(torch.bfloat16)
    for lin in (attn.qkv, attn.kv_ctx):
        lin.set_param("k_proj", wk); lin.set_param("v_proj", wv)
    attn.qkv.set_param("q_proj", wq)
    attn.o_proj.set_param("o_proj", torch.eye(hidden, dtype=torch.bfloat16))
    qn_w, kn_w = (torch.randn(d) * 0.1).to(torch.bfloat16), (torch.randn(d) * 0.1).to(torch.bfloat16)
    attn.set_param("q_norm", qn_w); attn.set_param("k_norm", kn_w)
    cfg = DCfg(heads, kv, d, ctx_max, gamma)
    h = DraftHarness(dev, cfg, qn_w.float().numpy(), kn_w.float().numpy())          # standard norm: the stored scale is w itself
    state = {"l.k_ctx": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16), "l.v_ctx": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16)}
    ctx_len = 0
    for n_new in (7, 2, 0, 40):
        x = torch.from_numpy(rng.standard_normal((gamma, hidden)).astype(np.float32)).to(torch.bfloat16)
        feats = torch.from_numpy(rng.standard_normal((n_new, hidden)).astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            proj, kvp = attn.qkv.forward(x), attn.kv_ctx.forward(feats)
            ref = attn.forward(x, torch.zeros(gamma, hidden, dtype=torch.bfloat16), state, ctx_len + n_new, ctx_feats=feats, ctx_len=ctx_len).float().numpy()
        got = h.step(f32_to_bf16(proj.float().numpy()), f32_to_bf16(kvp.float().numpy()), ctx_len, n_new)
        cos, max_abs, scale = _bars(got, ref)
        # P rounded per chunk before normalization vs the reference's normalized P: the composite bar (test_gqa_decode)
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, (ctx_len, n_new, cos, max_abs, scale)
        kc, vc = h.caches()
        n = ctx_len + n_new
        k_hf = state["l.k_ctx"][:n].float().numpy()
        assert np.abs(kc[:n] - k_hf).max() <= 1e-2 * max(np.abs(k_hf).max(), 1e-6)
        assert np.array_equal(vc[:n], state["l.v_ctx"][:n].float().numpy())
        ctx_len = n


# ---- the confidence head -----------------------------------------------------------------------------------------

def test_confidence(dev):
    rng = np.random.default_rng(4)
    gamma, hid, rank = 7, 256, 64
    hidden = f32_to_bf16(rng.standard_normal((gamma, hid)).astype(np.float32))
    emb = f32_to_bf16(rng.standard_normal((gamma, rank)).astype(np.float32))
    w = rbf(rng.standard_normal(hid + rank) * 0.05)
    b = rbf(rng.standard_normal(1) * 0.5)
    pso = nt.Pipeline(_spec_lib(dev), "confidence")
    out = nt.Buffer(dev, gamma * 4); out.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, hidden.tobytes())).buffer(1, nt.Buffer(dev, emb.tobytes()))
         .buffer(2, nt.Buffer(dev, w.astype(np.float32).tobytes())).buffer(3, nt.Buffer(dev, b.astype(np.float32).tobytes())).buffer(4, out)
         .bytes(5, kernels.conf_params(gamma - 1, hid, rank)).grid(gamma).threadgroup(32))
    _run(dev, d)
    got = np.frombuffer(out.read(0, gamma * 4), dtype=np.float32)
    feats = np.concatenate([bf16_to_f32(hidden), bf16_to_f32(emb)], axis=1).astype(np.float64)
    z = rbf(rbf(feats @ w.astype(np.float64)) + b[0])
    ref = 1.0 / (1.0 + np.exp(-z))
    assert np.abs(got[:-1] - ref[:-1]).max() <= 2e-3 and got[-1] == 0        # the last position is beyond gamma
    # without the Markov part
    out.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, hidden.tobytes())).buffer(1, nt.Buffer(dev, emb.tobytes()))
         .buffer(2, nt.Buffer(dev, w.astype(np.float32).tobytes())).buffer(3, nt.Buffer(dev, b.astype(np.float32).tobytes())).buffer(4, out)
         .bytes(5, kernels.conf_params(gamma, hid, 0)).grid(gamma).threadgroup(32))
    _run(dev, d)
    got = np.frombuffer(out.read(0, gamma * 4), dtype=np.float32)
    z = rbf(rbf(bf16_to_f32(hidden).astype(np.float64) @ w[:hid].astype(np.float64)) + b[0])
    assert np.abs(got - 1.0 / (1.0 + np.exp(-z))).max() <= 2e-3
    # STS temperatures divide the logit per position
    sts = [0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 3.0]
    out.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, hidden.tobytes())).buffer(1, nt.Buffer(dev, emb.tobytes()))
         .buffer(2, nt.Buffer(dev, w.astype(np.float32).tobytes())).buffer(3, nt.Buffer(dev, b.astype(np.float32).tobytes())).buffer(4, out)
         .bytes(5, kernels.conf_params(gamma, hid, 0, sts=sts)).grid(gamma).threadgroup(32))
    _run(dev, d)
    got = np.frombuffer(out.read(0, gamma * 4), dtype=np.float32)
    assert np.abs(got - 1.0 / (1.0 + np.exp(-z / np.array(sts)))).max() <= 2e-3


# ---- verify_select and accept_scan on StepState ---------------------------------------------------------------

def _select(dev, state, drafts, conf, gamma, threshold, t_max=8, mode=0, cost=None, log=None, ctx_cap=0):
    pso = nt.Pipeline(_spec_lib(dev), "verify_select")
    st = nt.Buffer(dev, LAYOUT.pack(state))
    lb = log if log is not None else nt.Buffer(dev, 16 * 4)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, np.asarray(drafts, np.int32).tobytes()))
         .buffer(1, nt.Buffer(dev, np.asarray(conf, np.float32).tobytes())).buffer(2, st)
         .bytes(3, kernels.select_params(gamma, threshold, t_max, mode=mode, cost=cost, log_cap=4 if log is not None else 0, ctx_cap=ctx_cap))
         .buffer(4, lb).grid(1).threadgroup(32))
    _run(dev, d)
    return LAYOUT.unpack(st.read(0, LAYOUT.size))


def test_verify_select(dev):
    drafts, conf = [11, 12, 13, 14, 15], [0.9, 0.6, 0.4, 0.8, 0.1]
    base = {"anchor": 7, "drafter_ctx_len": 10, "n_inject": 3, "t_this_step": 1}
    s = _select(dev, base, drafts, conf, 5, 0.0)
    assert s["drafter_ctx_len"] == 13 and s["n_inject"] == 0 and s["gamma"] == 5 and s["verify_len"] == 5 and s["t_this_step"] == 6
    assert s["pending_tokens"] == [7, 11, 12, 13, 14, 15, 0, 0] and s["draft_tokens"][:5] == drafts and s["confidence"][:5] == pytest.approx(conf)
    s = _select(dev, base, drafts, conf, 5, 0.5)
    assert s["verify_len"] == 2 and s["t_this_step"] == 3 and s["pending_tokens"][:4] == [7, 11, 12, 0]
    s = _select(dev, base, drafts, conf, 5, 0.0, t_max=3)                       # clamped to t_max − 1
    assert s["verify_len"] == 2 and s["t_this_step"] == 3
    s = _select(dev, dict(base, prefill_left=2, t_this_step=8, pending_tokens=[1, 2, 3, 4, 5, 6, 7, 8]), drafts, conf, 5, 0.0)
    assert s["drafter_ctx_len"] == 13 and s["n_inject"] == 0 and s["t_this_step"] == 8 and s["pending_tokens"] == [1, 2, 3, 4, 5, 6, 7, 8]
    s = _select(dev, dict(base, done=1), drafts, conf, 5, 0.0)
    assert s["drafter_ctx_len"] == 10 and s["n_inject"] == 3
    # the cost-aware rule: argmax_l (1 + Σ a_i) / cost[l] with a_i = Π c_j
    cost = [1.0, 1.28, 1.535, 1.79, 2.67, 3.55, 4.4, 5.3]
    s = _select(dev, base, drafts, conf, 5, 0.0, mode=1, cost=cost)
    a, best, exp_L, e = 1.0, 1.0, 0, 1.0
    for l in range(1, 6):
        a *= conf[l - 1]; e += a
        if e / cost[l] > best:
            best, exp_L = e / cost[l], l
    assert s["verify_len"] == exp_L == 2 and s["t_this_step"] == 3
    # a fixed L, clamped to the block and to t_max - 1; the confidence log
    log = nt.Buffer(dev, 4 * 16 * 4); log.fill(0)
    s = _select(dev, dict(base, step=6), drafts, conf, 5, 2.0, mode=2, log=log)
    assert s["verify_len"] == 2 and np.frombuffer(log.read(0, 4 * 16 * 4), dtype=np.float32).reshape(4, 16)[6 % 4, :5].tolist() == pytest.approx(conf)
    assert _select(dev, base, drafts, conf, 5, 9.0, mode=2)["verify_len"] == 5 and _select(dev, base, drafts, conf, 5, 9.0, t_max=4, mode=2)["verify_len"] == 3
    # the context capacity clamps L so the verify rows position … position + L stay inside the target's caches
    at = dict(base, position=10)
    s = _select(dev, at, drafts, conf, 5, 0.0, ctx_cap=13)
    assert s["verify_len"] == 2 and s["t_this_step"] == 3 and s["error"] == 0 and s["done"] == 0
    s = _select(dev, at, drafts, conf, 5, 0.0, ctx_cap=11)
    assert s["verify_len"] == 0 and s["t_this_step"] == 1 and s["error"] == 0
    s = _select(dev, at, drafts, conf, 5, 0.0, ctx_cap=10)                       # no room for the anchor's row: stop
    assert s["error"] == 2 and s["done"] == 1 and s["t_this_step"] == 1
    assert _select(dev, at, drafts, conf, 5, 0.0, ctx_cap=64)["verify_len"] == 5   # a roomy capacity changes nothing


def model_accept(state, tokens, ring_cap, eos, ctx_cap=0):
    """The Python model of accept_scan (mirrors kernels/common/spec_ops.metal); returns (state, ring writes)."""
    s = dict(state)
    writes = []
    if s["done"]:
        return s, writes
    t = s["t_this_step"]
    if s["prefill_left"] > 0:
        s["position"] += t; s["step"] += 1; s["n_inject"] = t; s["checkpoint_index"] = t; s["n_chain"] = 0
        if ctx_cap and s["position"] >= ctx_cap:
            s["error"] = 2; s["done"] = 1
        return s, writes
    L = s["verify_len"]
    base = t - 1 - L
    acc = 0
    while acc < L and tokens[base + acc] == s["pending_tokens"][base + acc + 1]:
        acc += 1
    bonus = tokens[base + acc]
    head = s["ring_head"]
    if head + acc + 1 - s["ring_tail"] > ring_cap:
        s["error"] = 1; s["done"] = 1
        return s, writes
    committed, last, stop = 0, bonus, False
    for k in range(acc + 1):
        tok = s["pending_tokens"][base + k + 1] if k < acc else bonus
        writes.append((head % ring_cap, ((head + 1) << 32) | (tok & 0xFFFFFFFF)))
        head += 1; committed += 1; last = tok
        if eos >= 0 and tok == eos:
            stop = True
            break
    s["ring_head"] = head; s["accepted"] = acc; s["anchor"] = last          # pending_tokens keeps the step's rows (an LM drafter's ingest reads them)
    s["position"] += base + committed; s["step"] += 1; s["verify_len"] = 0; s["t_this_step"] = 1
    s["n_inject"] = s["checkpoint_index"] = base + committed; s["n_chain"] = 1
    if stop:
        s["done"] = 1
    if ctx_cap and s["position"] >= ctx_cap:
        s["error"] = 2; s["done"] = 1
    return s, writes


def _accept(dev, state, tokens, ring_cap=16, eos=-1, ctx_cap=0):
    pso = nt.Pipeline(_spec_lib(dev), "accept_scan")
    st = nt.Buffer(dev, LAYOUT.pack(state))
    ring = nt.Buffer(dev, ring_cap * 8); ring.fill(0)
    log = nt.Buffer(dev, 64 * 4); log.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, np.asarray(tokens, np.int32).tobytes())).buffer(1, st).buffer(2, ring)
         .bytes(3, kernels.accept_params(ring_cap, eos, 64, ctx_cap=ctx_cap)).buffer(4, log).grid(1).threadgroup(32))
    _run(dev, d)
    got = LAYOUT.unpack(st.read(0, LAYOUT.size))
    slots = np.frombuffer(ring.read(0, ring_cap * 8), dtype=np.uint64)
    entry = int(np.frombuffer(log.read(0, 64 * 4), dtype=np.uint32)[state["step"] % 64])
    if not state.get("done"):
        committed = got["position"] - state["position"] - (state["t_this_step"] - 1 - state["verify_len"])
        exp_entry = ((state["t_this_step"] << 16) | 0xFFFF) if state.get("prefill_left") else ((committed << 16) | (state["verify_len"] << 8) | got["accepted"])
        assert entry == exp_entry or got["error"], (entry, exp_entry)      # the step's log: (committed << 16) | (L << 8) | accepted
    return got, {i: int(v) for i, v in enumerate(slots) if v}


@pytest.mark.parametrize("case", ["all", "partial", "none", "eos_draft", "eos_bonus", "first_token", "overflow", "wrap", "context_full", "context_room"])
def test_accept_scan(dev, case):
    st = LAYOUT.unpack(LAYOUT.pack({}))
    st.update(position=40, step=5, ring_head=3, ring_tail=1, anchor=9, t_this_step=4, verify_len=3, pending_tokens=[9, 21, 22, 23, 0, 0, 0, 0])
    tokens, eos, cap, ctx_cap = [21, 22, 23, 24], -1, 16, 0
    if case == "context_full":                                   # all accepted: the next step would start at 44 = the capacity
        ctx_cap = 44
    elif case == "context_room":
        ctx_cap = 45
    if case == "partial":
        tokens = [21, 50, 23, 24]
    elif case == "none":
        tokens = [51, 22, 23, 24]
    elif case == "eos_draft":
        tokens, eos = [21, 22, 23, 24], 22
    elif case == "eos_bonus":
        tokens, eos = [21, 50, 23, 24], 50
    elif case == "first_token":                                  # the last prefill chunk: t = 3 prompt rows, nothing to verify
        st.update(t_this_step=3, verify_len=0, pending_tokens=[101, 102, 103, 0, 0, 0, 0, 0], ring_head=0, ring_tail=0)
        tokens = [31, 32, 33]
    elif case == "overflow":
        st.update(ring_head=17, ring_tail=0)
    elif case == "wrap":
        st.update(ring_head=15, ring_tail=13)
    got, ring = _accept(dev, st, tokens, cap, eos, ctx_cap)
    exp, writes = model_accept(st, tokens, cap, eos, ctx_cap)
    assert got == exp, case
    assert ring == {slot: val for slot, val in writes}, case
    if case == "context_full":                                   # the committed tokens are in the ring; the program stops
        assert got["error"] == 2 and got["done"] == 1 and got["position"] == 44 and got["ring_head"] == 7 and len(ring) == 4
    if case == "context_room":
        assert got["error"] == 0 and got["done"] == 0 and got["position"] == 44
    if case == "all":
        assert got["accepted"] == 3 and got["position"] == 44 and got["anchor"] == 24 and got["n_inject"] == 4 and got["ring_head"] == 7
    if case == "partial":
        assert got["accepted"] == 1 and got["position"] == 42 and got["anchor"] == 50 and got["n_inject"] == 2 and [v & 0xFFFFFFFF for v in ring.values()] == [21, 50]
    if case == "eos_draft":
        assert got["done"] == 1 and got["ring_head"] == 5 and got["anchor"] == 22 and got["position"] == 42
    if case == "first_token":
        assert got["position"] == 43 and got["n_inject"] == 3 and got["anchor"] == 33 and got["ring_head"] == 1 and got["step"] == 6
    if case == "overflow":
        assert got["error"] == 1 and got["done"] == 1 and not ring
    # a prefill chunk only advances and marks its rows for injection
    got, ring = _accept(dev, dict(st, prefill_left=2, t_this_step=4), tokens, cap, eos, ctx_cap)
    exp, _ = model_accept(dict(st, prefill_left=2, t_this_step=4), tokens, cap, eos, ctx_cap)
    assert got == exp and got["n_inject"] == 4 and got["position"] == 44 and not ring
    assert got["error"] == (2 if case == "context_full" else 0)


# ---- tap_concat, the block-ids embed, the row sources of the shared kernels ---------------------------------------

def test_tap_concat(dev):
    rng = np.random.default_rng(6)
    t, k, n = 4, 64, 3
    taps = [f32_to_bf16(rng.standard_normal((t, k)).astype(np.float32)) for _ in range(n)]
    for step_state, rows in ((False, 4), (True, 2)):
        macros = dict(N_SRC=n, **({"STEP_STATE": 1, "T_SRC": 1} if step_state else {}))
        pso = nt.Pipeline(_spec_lib(dev, **macros), "tap_concat")
        out = nt.Buffer(dev, t * n * k * 2); out.fill(0)
        bufs = [nt.Buffer(dev, x.tobytes()) for x in taps]
        d = nt.Dispatch().pipeline(pso)
        for i in range(8):
            d.buffer(i, bufs[min(i, n - 1)])
        d.buffer(8, out).bytes(9, kernels.concat_params(k, t)).grid(t * n).threadgroup(32)
        if step_state:
            d.buffer(15, nt.Buffer(dev, LAYOUT.pack({"n_inject": rows, "t_this_step": 4})))
        _run(dev, d)
        got = np.frombuffer(out.read(0, t * n * k * 2), dtype=np.uint16).reshape(t, n * k)
        exp = np.concatenate(taps, axis=1)
        assert np.array_equal(got[:rows], exp[:rows]) and np.all(got[rows:] == 0)


def test_embed_block_ids(dev):
    rng = np.random.default_rng(7)
    vocab, k, gamma = 40, 256, 5
    table = f32_to_bf16(rng.standard_normal((vocab, k)).astype(np.float32))
    spec = FORMATS.get("bf16").unpack({"weight": table}, shape=(vocab, k))
    data, info, _ = pack_spec(spec, PackLayout(rows=16))
    src = kernels.embed_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
    macros = dict(kernels.embed_macros(info, ids="block"), STEP_STATE="1", T_SRC="2", T_STATIC_ROWS=f"{gamma}u")
    pso = nt.Pipeline(nt.Library(dev, src, macros), "embed")
    h = nt.Buffer(dev, (gamma + 1) * k * 2); h.fill(0)
    st = nt.Buffer(dev, LAYOUT.pack({"anchor": 17, "t_this_step": 1}))
    d = (nt.Dispatch().pipeline(pso).buffer(0, st, LAYOUT.offset("anchor")).buffer(1, nt.Buffer(dev, data)).buffer(2, h)
         .bytes(3, kernels.embed_params(k, 1, vocab, mask_id=39)).buffer(15, st).grid(gamma + 1).threadgroup(32))
    _run(dev, d)
    out = np.frombuffer(h.read(0, (gamma + 1) * k * 2), dtype=np.uint16).reshape(gamma + 1, k)
    assert np.array_equal(out[0], table[17]) and all(np.array_equal(out[i], table[39]) for i in range(1, gamma)) and np.all(out[gamma] == 0)


def test_row_sources_of_gemv_stat_norm_argmax(dev):
    """With STEP_STATE, T_SRC=1 takes the row count from n_inject and T_SRC=2 the static count, whatever t_this_step says."""
    rng = np.random.default_rng(8)
    k, n, t = 1024, 64, 4
    spec = random_spec("bf16", n, k, rng)
    data, info, rs = pack_spec(spec, PackLayout(rows=16))
    w = FORMATS.get("bf16").dequantize(spec).astype(np.float64)
    x = f32_to_bf16(rng.standard_normal((t, k)).astype(np.float32))
    ref = bf16_to_f32(x).astype(np.float64) @ w.T
    n_sg = 12 * dev.info().gpu_cores
    msl = LAYOUT.to_msl()
    st = nt.Buffer(dev, LAYOUT.pack({"t_this_step": 1, "n_inject": 2}))
    for t_src, rows in ((1, 2), (2, 4), (0, 1)):
        macros = dict(kernels.gemv_macros(info, t=t, out_bf16=True), STEP_STATE="1", T_SRC=str(t_src))
        src = kernels.gemv_source("bf16").replace(kernels.PRELUDE, kernels.PRELUDE + msl + "\n", 1)
        pso = nt.Pipeline(nt.Library(dev, src, macros), "gemv_T")
        y = nt.Buffer(dev, t * n * 2); y.fill(0)
        d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, data)).buffer(1, nt.Buffer(dev, rs.tobytes())).buffer(2, nt.Buffer(dev, x.tobytes()))
             .buffer(3, y).bytes(4, kernels.gemv_params(n, info.n_blocks, n_sg, t)).buffer(15, st).grid(-(-(n_sg * 32) // 384)).threadgroup(384))
        _run(dev, d)
        got = bf16_to_f32(np.frombuffer(y.read(0, t * n * 2), dtype=np.uint16)).reshape(t, n)
        assert np.abs(got[:rows] - ref[:rows]).max() <= 1e-2 * np.abs(ref).max() and np.all(got[rows:] == 0), t_src
    # rmsnorm_stat / norm_apply / argmax with the static row count 3
    h = f32_to_bf16(rng.standard_normal((t, k)).astype(np.float32))
    ssrc = kernels.rmsnorm_stat_source().replace(kernels.PRELUDE, kernels.PRELUDE + msl + "\n", 1)
    pso = nt.Pipeline(nt.Library(dev, ssrc, {"STEP_STATE": "1", "T_SRC": "2", "T_STATIC_ROWS": "3u"}), "rmsnorm_stat")
    stat = nt.Buffer(dev, t * 4); stat.fill(0)
    _run(dev, nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, h.tobytes())).buffer(1, stat).bytes(2, kernels.stat_params(k, t)).buffer(15, st).grid(t).threadgroup(32))
    s = np.frombuffer(stat.read(0, t * 4), dtype=np.float32)
    assert np.allclose(s[:3], (bf16_to_f32(h[:3]).astype(np.float64) ** 2).sum(-1), rtol=1e-5) and s[3] == 0
    asrc = kernels.argmax_source().replace(kernels.PRELUDE, kernels.PRELUDE + msl + "\n", 1)
    lib = nt.Library(dev, asrc, {"STEP_STATE": "1", "T_SRC": "1"})
    pv, pi = nt.Buffer(dev, t * n_sg * 4), nt.Buffer(dev, t * n_sg * 4)
    tok = nt.Buffer(dev, t * 4); tok.fill(0)
    prm = kernels.argmax_params(k, t, n_sg)
    hb = nt.Buffer(dev, h.tobytes())
    d1 = nt.Dispatch().pipeline(nt.Pipeline(lib, "argmax_partial")).buffer(0, hb).buffer(1, pv).buffer(2, pi).bytes(3, prm).buffer(15, st).grid(-(-(n_sg * 32) // 384)).threadgroup(384).barrier()
    d2 = nt.Dispatch().pipeline(nt.Pipeline(lib, "argmax_final")).buffer(0, pv).buffer(1, pi).buffer(2, tok).bytes(3, prm).buffer(15, st).grid(t).threadgroup(32)
    _run(dev, [d1, d2])
    got = np.frombuffer(tok.read(0, t * 4), dtype=np.int32)
    assert list(got[:2]) == list(bf16_to_f32(h[:2]).argmax(-1)) and np.all(got[2:] == 0)
