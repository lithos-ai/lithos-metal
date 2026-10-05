"""gqa_decode + gqa_merge against (a) a numpy model of the kernel's own contract (chunked online softmax with the
reference's roundings) and (b) the GQAAttention layer oracle (torch, the HF-faithful semantics), on fresh and
filled caches, T = 1 and T > 1 (causal inside the step), several chunks and row groups, both head dims."""

import math

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.nn.rope import rope_tables_permuted
from monolith.packs.transforms import rope_head_perm
from monolith.runtime import _native as nt

EPS = 1e-6
THETA = 10000.0


def rbf(x):
    return bf16_to_f32(f32_to_bf16(np.asarray(x, dtype=np.float32)))


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


class Cfg:
    def __init__(self, heads, kv, d, rot, ctx_max, t_max, chunk=64, rb=4, gate=True, v2=False, n_tg=None, steal=False, v3=False, nsg3=None, mma=False, step_state=False, lm_mode=0, adaptive=False, single_block=False, qk_norm=True, direct=False):
        self.heads, self.kv, self.d, self.rot, self.ctx_max, self.t_max = heads, kv, d, rot, ctx_max, t_max
        self.direct = direct
        self.mma, self.step_state, self.lm_mode = mma, step_state, lm_mode
        self.single_block = single_block
        self.adaptive = adaptive
        self.qk_norm = qk_norm
        self.v2, self.n_tg, self.steal = v2, n_tg, steal                # v2: the partial granularity is 32 keys (the numpy model's chunk)
        self.v3, self.nsg3 = v3, nsg3 or kernels.gqa_v3_simdgroups(d)   # v3: one dispatch, a threadgroup of nsg3 SIMD-groups per query row
        if v2 or (mma and d == 256):
            chunk = 32
        self.chunk, self.rb, self.gate = chunk, rb, gate
        self.rep = heads // kv
        hd, kd = heads * d, kv * d
        self.q_off, self.k_off, self.v_off = 0, hd, hd + kd                    # the pack's order: q | k | v | gate
        self.gate_off = hd + 2 * kd
        self.n1 = hd + 2 * kd + (hd if gate else 0)
        self.rows_max = self.rep * t_max
        self.n_chunks_max = kernels.gqa_chunks_max(ctx_max, kv, chunk, 12 * dev.info().gpu_cores) if False else kernels.gqa_chunks_max(ctx_max, kv, chunk)
        if adaptive:
            self.n_chunks_max = -(-ctx_max // 32)
        self.scaling = d ** -0.5


class Harness:
    """The two dispatches with their caches, tables and workspace for one layer config."""

    def __init__(self, dev, cfg, qn, kn):
        self.dev, self.cfg = dev, cfg
        def library(source, macros, language_version=0):
            return nt.Library(dev, source, dict(macros, QK_NORM=str(int(cfg.qk_norm)),
                MMA_PROB_FP16=str(int(getattr(cfg, 'prob_fp16', False)))), language_version)
        if cfg.mma:
            macros = dict(kernels.gqa_macros(cfg.d, chunk=cfg.chunk, rb_max=16, lm_mode=cfg.lm_mode, chain_i=2 if cfg.lm_mode == 2 else 0), FIXED_CHUNK="1", MMA_SG=str(kernels.gqa_mma_simdgroups(cfg.d, cfg.rep)))
            if cfg.direct:
                macros.update(DIRECT_KV="1", MMA_SG="4")
            if cfg.adaptive:
                macros["ADAPTIVE_CHUNK"] = "1"
            source = kernels.gqa_source(mma=True)
            if cfg.step_state:
                from monolith.core import StepStateLayout
                self.layout = StepStateLayout()
                macros["STEP_STATE"] = "1"
                source = source.replace(kernels.PRELUDE, kernels.PRELUDE + self.layout.to_msl(), 1)
            lib = library(source, macros, kernels.MSL_TENSOR_OPS)
            self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode_mma"), nt.Pipeline(lib, "gqa_merge")
            if cfg.direct:
                self.p_prepare = nt.Pipeline(lib, "gqa_prepare_mma")
        elif cfg.v3:
            lib = library(kernels.gqa_source(v3=True), dict(kernels.gqa_v3_macros(cfg.d, nsg=cfg.nsg3), SINGLE_BLOCK=str(int(cfg.single_block))))
            self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode_v3"), None
            assert self.p_dec.max_threads_per_threadgroup >= cfg.nsg3 * 32
        elif cfg.v2:
            lib = library(kernels.gqa_source(True), kernels.gqa_v2_macros(cfg.d, rmax=cfg.rows_max, rg=cfg.rb))
            self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode_v2"), nt.Pipeline(lib, "gqa_merge_v2")
        else:
            lib = library(kernels.gqa_source(steal=cfg.steal), kernels.gqa_macros(cfg.d, chunk=cfg.chunk, rb_max=cfg.rb, steal=cfg.steal, steal_hits=cfg.steal))
            self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode"), nt.Pipeline(lib, "gqa_merge")
            if cfg.steal:
                self.p_reset = nt.Pipeline(lib, "steal_reset")
        cos, sin = rope_tables_permuted(THETA, cfg.d, cfg.rot, cfg.ctx_max)
        self.cos_b, self.sin_b = f32_to_bf16(cos), f32_to_bf16(sin)
        self.cos, self.sin = bf16_to_f32(self.cos_b), bf16_to_f32(self.sin_b)
        self.qn, self.kn = qn.astype(np.float32), kn.astype(np.float32)
        nbytes = self.cache_bytes = cfg.ctx_max * cfg.kv * cfg.d * 2
        self.k_cache, self.v_cache = nt.Buffer(dev, nbytes), nt.Buffer(dev, nbytes)
        self.k_cache.fill(0); self.v_cache.fill(0)
        po, pm = kernels.gqa_workspace(cfg.kv, cfg.n_chunks_max, cfg.rows_max, cfg.d)
        self.part_o, self.part_md = nt.Buffer(dev, po), nt.Buffer(dev, pm)
        self.bufs = {"cos": nt.Buffer(dev, self.cos_b.tobytes()), "sin": nt.Buffer(dev, self.sin_b.tobytes()),
                     "qn": nt.Buffer(dev, self.qn.tobytes()), "kn": nt.Buffer(dev, self.kn.tobytes())}
        self.n_sg = 12 * dev.info().gpu_cores
        self.n_tg = dev.info().gpu_cores if cfg.n_tg is None else cfg.n_tg      # v2: threadgroups (params.n_sg); the grid stays the crew

    def set_caches(self, k, v):
        self.k_cache.write(f32_to_bf16(k).tobytes(), 0)
        self.v_cache.write(f32_to_bf16(v).tobytes(), 0)

    def caches(self):
        c = self.cfg
        shape = (c.ctx_max, c.kv, c.d)
        return (bf16_to_f32(np.frombuffer(self.k_cache.read(0, self.cache_bytes), dtype=np.uint16).reshape(shape)),
                bf16_to_f32(np.frombuffer(self.v_cache.read(0, self.cache_bytes), dtype=np.uint16).reshape(shape)))

    def step(self, proj_bf16, position, t_active=None, dispatch_sg=None, state=None):
        """``dispatch_sg``: the SIMD-groups actually dispatched (the STEAL variant's crew may be short or surplus)."""
        c = self.cfg
        t = proj_bf16.shape[0]
        t_act = t if t_active is None else t_active
        params = kernels.gqa_params(heads=c.heads, kv_heads=c.kv, t_active=t_act, position=position,
                                    n_sg=(dispatch_sg or self.dev.info().gpu_cores * 4) if c.mma else (t * c.heads) if c.v3 else (self.n_tg if c.v2 else self.n_sg),
                                    q_off=c.q_off, gate_off=c.gate_off, k_off=c.k_off, v_off=c.v_off, in_stride=c.n1,
                                    out_stride=c.heads * c.d, ctx_max=c.ctx_max, eps=EPS, scaling=c.scaling, has_gate=c.gate,
                                    n_chunks_max=c.n_chunks_max, rows_max=c.rows_max, nominal_sg=self.n_sg, gate_stride=c.n1 if c.v3 else 0)
        if getattr(c, "specialize", False) and not getattr(self, "specialized", False):
            source = kernels.gqa_source(v2=c.v2, v3=c.v3, mma=c.mma)
            if c.v3:
                macros = dict(kernels.gqa_v3_macros(c.d, nsg=c.nsg3), SINGLE_BLOCK=str(int(c.single_block)))
            elif c.v2:
                macros = kernels.gqa_v2_macros(c.d, rmax=c.rows_max, rg=c.rb)
            else:
                macros = kernels.gqa_macros(c.d, chunk=c.chunk, rb_max=16 if c.mma else c.rb)
            if c.mma:
                macros.update(FIXED_CHUNK="1", MMA_SG=str(kernels.gqa_mma_simdgroups(c.d, c.rep)))
                if c.direct:
                    macros.update(DIRECT_KV="1", MMA_SG="4")
                if c.adaptive:
                    macros["ADAPTIVE_CHUNK"] = "1"
            if c.step_state:
                macros["STEP_STATE"] = "1"
                source = source.replace(kernels.PRELUDE, kernels.PRELUDE + self.layout.to_msl(), 1)
            source, constants = kernels.specialize_params(source, "gqa", params)
            lib = nt.Library(self.dev, source, dict(macros, **constants, QK_NORM=str(int(c.qk_norm))), kernels.MSL_TENSOR_OPS if c.mma else 0)
            name = "gqa_decode_mma" if c.mma else "gqa_decode_v3" if c.v3 else "gqa_decode_v2" if c.v2 else "gqa_decode"
            self.p_dec = nt.Pipeline(lib, name)
            self.p_merge = None if c.v3 else nt.Pipeline(lib, "gqa_merge_v2" if c.v2 else "gqa_merge")
            if c.direct:
                self.p_prepare = nt.Pipeline(lib, "gqa_prepare_mma")
            self.specialized = True
        pb = nt.Buffer(self.dev, proj_bf16.tobytes())
        out = nt.Buffer(self.dev, t * c.heads * c.d * 2); out.fill(0)
        if c.v3:                                                       # one dispatch: heads · T threadgroups; the gate read from the projection (buffer 10, stride n1)
            d3 = (nt.Dispatch().pipeline(self.p_dec).buffer(0, pb).buffer(1, self.k_cache).buffer(2, self.v_cache)
                  .buffer(3, self.bufs["cos"]).buffer(4, self.bufs["sin"]).buffer(5, self.bufs["qn"]).buffer(6, self.bufs["kn"])
                  .buffer(7, out).bytes(9, params).buffer(10, pb).grid(t * c.heads).threadgroup(c.nsg3 * 32).barrier())
            r = nt.Queue(self.dev).run([d3])
            assert not r.error, r.error
            return bf16_to_f32(np.frombuffer(out.read(0, t * c.heads * c.d * 2), dtype=np.uint16).reshape(t, c.heads * c.d))
        n_disp = self.n_sg if dispatch_sg is None else dispatch_sg
        d1 = (nt.Dispatch().pipeline(self.p_dec).buffer(0, pb).buffer(1, self.k_cache).buffer(2, self.v_cache)
              .buffer(3, self.bufs["cos"]).buffer(4, self.bufs["sin"]).buffer(5, self.bufs["qn"]).buffer(6, self.bufs["kn"])
              .buffer(7, self.part_o).buffer(8, self.part_md).bytes(9, params).grid(-(-(n_disp * 32) // 384)).threadgroup(384).barrier())
        if c.mma:
            d1.grid(dispatch_sg or self.dev.info().gpu_cores * 4).threadgroup(32 * kernels.gqa_mma_simdgroups(c.d, c.rep))
        merge_sg = getattr(c, "merge_sg", 1)
        d2 = (nt.Dispatch().pipeline(self.p_merge).buffer(0, self.part_o).buffer(1, self.part_md).buffer(2, pb).buffer(3, out)
              .bytes(4, params).grid(-(-(t * c.heads) // merge_sg)).threadgroup(32 * merge_sg))
        if c.step_state:
            sb = nt.Buffer(self.dev, self.layout.pack(state or {'position': position, 't_this_step': t_act}))
            d1.buffer(15, sb)
            d2.buffer(15, sb)
        ds = [d1, d2]
        if c.direct:
            d1.threadgroup(128)
            d0 = (nt.Dispatch().pipeline(self.p_prepare).buffer(0, pb).buffer(1, self.k_cache).buffer(2, self.v_cache)
                  .buffer(3, self.bufs["cos"]).buffer(4, self.bufs["sin"]).buffer(5, self.bufs["qn"]).buffer(6, self.bufs["kn"])
                  .bytes(9, params).buffer(15, sb).grid(-(-(t * c.heads) // 4)).threadgroup(128).barrier())
            ds = [d0, d1, d2]
        self.hits = None
        if c.steal:
            n_blocks = c.kv * (-(-(position + t_act) // c.chunk)) * (-(-(c.rep * t_act) // c.rb))
            cursors = nt.Buffer(self.dev, max(4 * self.n_sg, 16)); cursors.fill(0)
            self.hits = nt.Buffer(self.dev, max(4 * n_blocks, 16)); self.hits.fill(0)
            self.n_blocks = n_blocks
            d0 = nt.Dispatch().pipeline(self.p_reset).buffer(0, cursors).bytes(1, kernels.steal_reset_params(self.n_sg)).grid(-(-self.n_sg // 64)).threadgroup(64).barrier()
            d1.buffer(10, cursors).buffer(12, self.hits)
            ds = [d0, d1, d2]
        r = nt.Queue(self.dev).run(ds)
        assert not r.error, r.error
        return bf16_to_f32(np.frombuffer(out.read(0, t * c.heads * c.d * 2), dtype=np.uint16).reshape(t, c.heads * c.d))

    def block_hits(self):
        return np.frombuffer(self.hits.read(0, 4 * self.n_blocks), dtype=np.uint32)


# ---- the kernel's contract in numpy -----------------------------------------------------------------------------

def norm_rope_np(x, nw, cos_row, sin_row, d, qk_norm=True):
    """``x [..., D]`` in the permuted layout: (1 + w) RMSNorm then full-width rotary pairs, the reference's roundings."""
    if qk_norm:
        ss = (x.astype(np.float32) ** 2).sum(-1, keepdims=True)
        rstd = (1.0 / np.sqrt(ss / d + EPS)).astype(np.float32)
        x = rbf(x * rstd * nw)
    half = d // 2
    partner = np.concatenate([x[..., half:], x[..., :half]], axis=-1)
    a, b = rbf(x * cos_row), rbf(partner * sin_row)
    sign = np.concatenate([-np.ones(half), np.ones(half)]).astype(np.float32)
    return rbf(a + sign * b)


def ref_step(h, proj, position, k_cache, v_cache):
    """The chunked online-softmax semantics of gqa_decode + gqa_merge; ``k_cache``/``v_cache`` (numpy, BF16-valued)
    are advanced in place. Returns ``[T, heads·D]``."""
    c = h.cfg
    t = proj.shape[0]
    d = c.d
    q = proj[:, c.q_off: c.q_off + c.heads * d].reshape(t, c.heads, d)
    k = proj[:, c.k_off: c.k_off + c.kv * d].reshape(t, c.kv, d)
    v = proj[:, c.v_off: c.v_off + c.kv * d].reshape(t, c.kv, d)
    pos = np.arange(position, position + t)
    q = norm_rope_np(q, h.qn, h.cos[pos][:, None, :], h.sin[pos][:, None, :], d, c.qk_norm)
    k = norm_rope_np(k, h.kn, h.cos[pos][:, None, :], h.sin[pos][:, None, :], d, c.qk_norm)
    k_cache[position: position + t] = k
    v_cache[position: position + t] = v
    ctx = position + t
    out = np.zeros((t, c.heads, d), dtype=np.float32)
    for tt in range(t):
        for hh in range(c.heads):
            j = hh // c.rep
            keys = np.arange(0, ctx)
            s = rbf(rbf(k_cache[:ctx, j].astype(np.float64) @ q[tt, hh].astype(np.float64)) * c.scaling)
            s[keys > position + tt] = -np.inf
            parts = []
            for c0 in range(0, ctx, c.chunk):
                sc = s[c0: c0 + c.chunk]
                m = sc.max()
                if m == -np.inf:
                    continue
                p = np.exp((sc - m).astype(np.float32)).astype(np.float32)
                p[sc == -np.inf] = 0
                rounded_p = p.astype(np.float16).astype(np.float32) if getattr(c, 'prob_fp16', False) else rbf(p)
                parts.append((m, float(p.sum(dtype=np.float32)), rounded_p.astype(np.float64) @ v_cache[c0: min(c0 + c.chunk, ctx), j].astype(np.float64)))
            m_g = max(m for m, _, _ in parts)
            d_g = sum(dc * np.exp(m - m_g) for m, dc, _ in parts)
            o = sum(oc * np.exp(m - m_g) for m, _, oc in parts) / d_g
            y = rbf(o)
            if c.gate:
                g = proj[tt, c.gate_off + hh * d: c.gate_off + (hh + 1) * d]
                y = rbf(y * rbf(1.0 / (1.0 + np.exp(-g))))
            out[tt, hh] = y
    return out.reshape(t, c.heads * d)


def _ulp(x: float) -> float:
    """One BF16 ulp at magnitude ``x`` (8 bits of mantissa)."""
    return 2.0 ** (math.floor(math.log2(max(x, 1e-30))) - 7)


def _bars(got, ref):
    got, ref = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    return cos, float(np.abs(got - ref).max()), float(np.abs(ref).max())


def _random_proj(rng, cfg, t):
    return f32_to_bf16(rng.standard_normal((t, cfg.n1)).astype(np.float32))


def _norms(rng, d):
    return (1.0 + rng.standard_normal(d) * 0.1).astype(np.float32), (1.0 + rng.standard_normal(d) * 0.1).astype(np.float32)


@pytest.mark.parametrize("cfg", [Cfg(8, 2, 256, 64, 512, 8), Cfg(4, 1, 128, 32, 256, 8, chunk=32, rb=2), Cfg(6, 1, 256, 64, 200, 4, rb=8),
                                 Cfg(4, 2, 128, 128, 128, 4, gate=False)], ids=["d256_rep4", "d128_rep4_ch32_rb2", "rep6_rb8", "nogate_fullrope"])
def test_matches_kernel_contract(dev, cfg):
    rng = np.random.default_rng(cfg.heads * 31 + cfg.d)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    k_ref, v_ref = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    # a filled prefix of the caches (random BF16), then three steps: T = 5 prefill, T = 1, T = 4 (causal inside)
    pre = 70
    k_ref[:pre], v_ref[:pre] = rbf(rng.standard_normal((pre, cfg.kv, cfg.d)) * 0.5), rbf(rng.standard_normal((pre, cfg.kv, cfg.d)))
    h.set_caches(k_ref, v_ref)
    pos = pre
    for t in (5, 1, 4):
        if t > cfg.t_max:
            continue
        proj = _random_proj(rng, cfg, t)
        got = h.step(proj, pos)
        ref = ref_step(h, bf16_to_f32(proj), pos, k_ref, v_ref)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.99999 and max_abs <= 2 * _ulp(scale), (t, pos, cos, max_abs, scale)   # 2 BF16 ulps: the chunked softmax rounds p̃ per chunk, the chunk chosen at run time
        kc, vc = h.caches()
        assert np.abs(kc[: pos + t] - k_ref[: pos + t]).max() <= 1e-2 * np.abs(k_ref[: pos + t]).max()   # ≤ 1 ULP flips at rsqrt boundaries
        assert np.array_equal(vc[: pos + t], v_ref[: pos + t])
        pos += t


def test_repeat_runs_are_bit_identical_and_t_active(dev):
    cfg = Cfg(8, 2, 256, 64, 256, 4)
    rng = np.random.default_rng(5)
    h = Harness(dev, cfg, *_norms(rng, cfg.d))
    proj = _random_proj(rng, cfg, 4)
    a = h.step(proj, 0)
    h.set_caches(np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32))
    b = h.step(proj, 0)
    assert np.array_equal(a, b)
    h.set_caches(np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32))
    c = h.step(proj, 0, t_active=2)
    assert np.array_equal(c[:2], a[:2]) and np.all(c[2:] == 0)
    kc, _ = h.caches()
    assert np.all(kc[2:] == 0)                                             # only t_active positions appended


def test_matches_layer_oracle(dev):
    """The HF-faithful GQAAttention.mix (torch) vs the kernel on the same projection, caches and positions."""
    torch = pytest.importorskip("torch")
    from monolith.nn import GQAAttention

    heads, kv, d, rot, hidden, ctx_max = 8, 2, 256, 64, 64, 256
    cfg = Cfg(heads, kv, d, rot, ctx_max, 8)
    rng = np.random.default_rng(9)
    torch.manual_seed(9)
    mod = GQAAttention(hidden, heads, kv, d, rot, THETA, EPS, hf_prefix="x.", prefix="l.", max_context=ctx_max)
    qn_w, kn_w = (rng.standard_normal(d) * 0.1).astype(np.float32), (rng.standard_normal(d) * 0.1).astype(np.float32)
    mod.set_param("q_norm", torch.from_numpy(qn_w).to(torch.bfloat16))
    mod.set_param("k_norm", torch.from_numpy(kn_w).to(torch.bfloat16))
    perm = rope_head_perm(d, rot)
    qn_p, kn_p = (1.0 + bf16_to_f32(f32_to_bf16(qn_w)))[perm], (1.0 + bf16_to_f32(f32_to_bf16(kn_w)))[perm]
    h = Harness(dev, cfg, qn_p, kn_p)
    state = {"l.k_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16), "l.v_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16)}
    col_perm = mod.qkv.row_perm                                             # kernel column n = checkpoint column perm[n]
    inv = np.argsort(perm)
    pos = 0
    for t in (6, 1, 3):
        proj_hf = torch.from_numpy(rng.standard_normal((t, cfg.n1)).astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            ref = mod.mix(proj_hf, state, pos).float().numpy()
        proj_k = proj_hf.float().numpy()[:, col_perm]
        got = h.step(f32_to_bf16(proj_k), pos)
        cos, max_abs, scale = _bars(got, ref)
        # the kernel rounds P before normalization, per chunk (online softmax); the reference rounds the normalized
        # P — a 1–2 BF16 ULP difference at the output's magnitude, inside the composite bar (cos > 0.999, max-abs
        # bounded), not the leaf bar (that one is the kernel-contract test above)
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, (t, pos, cos, max_abs, scale)
        kc, vc = h.caches()
        k_hf = state["l.k_cache"][: pos + t].float().numpy()
        assert np.abs(kc[: pos + t][..., inv] - k_hf).max() <= 1e-2 * np.abs(k_hf).max()
        assert np.array_equal(vc[: pos + t], state["l.v_cache"][: pos + t].float().numpy())
        pos += t


# ---- v2: the long-context structure (#34) --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [Cfg(8, 2, 256, 64, 2048, 8, rb=4, v2=True), Cfg(8, 2, 256, 64, 2048, 8, rb=4, v2=True, n_tg=1),
                                 Cfg(4, 1, 128, 32, 4096, 8, rb=8, v2=True, n_tg=2), Cfg(4, 2, 128, 128, 1024, 4, gate=False, v2=True)],
                         ids=["v2_d256_ch32", "v2_d256_1tg_ch128", "v2_d128_2tg_ch64", "v2_nogate_fullrope"])
def test_v2_matches_kernel_contract(dev, cfg):
    """v2 against the numpy model of the kernel contract at chunk 32 (one partial per 32-key sub-chunk, folded
    exactly): contexts spanning several batches, T = 5 / 1 / 4 (causal inside the step), the new keys appended by
    their batch, and the chunk rule (n_tg = 1 → 128-key chunks, 2 → 64 at these contexts, the crew → 32)."""
    rng = np.random.default_rng(cfg.heads * 31 + cfg.d + cfg.ctx_max)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    k_ref, v_ref = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    pre = min(1500, cfg.ctx_max - 32)                                   # a filled prefix across several 12-chunk batches
    k_ref[:pre], v_ref[:pre] = rbf(rng.standard_normal((pre, cfg.kv, cfg.d)) * 0.5), rbf(rng.standard_normal((pre, cfg.kv, cfg.d)))
    h.set_caches(k_ref, v_ref)
    pos = pre
    for t in (5, 1, 4):
        if t > cfg.t_max:
            continue
        proj = _random_proj(rng, cfg, t)
        got = h.step(proj, pos)
        ref = ref_step(h, bf16_to_f32(proj), pos, k_ref, v_ref)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.99999 and max_abs <= 2 * _ulp(scale), (t, pos, cos, max_abs, scale)   # 2 BF16 ulps: the chunked softmax rounds p̃ per chunk, the chunk chosen at run time
        kc, vc = h.caches()
        assert np.abs(kc[: pos + t] - k_ref[: pos + t]).max() <= 1e-2 * np.abs(k_ref[: pos + t]).max()
        assert np.array_equal(vc[: pos + t], v_ref[: pos + t])
        pos += t


def test_v2_equals_v1_and_is_bit_stable(dev):
    """The two kernels agree within the contract bar on the same inputs at a long context (v1 at chunk 64 vs v2's
    hierarchical fold), v2 repeats bit-identically, and t_active limits the rows and the append."""
    heads, kv, d, ctx_max = 8, 2, 256, 4096
    rng = np.random.default_rng(77)
    qn, kn = _norms(rng, d)
    h1 = Harness(dev, Cfg(heads, kv, d, 64, ctx_max, 8), qn, kn)
    h2 = Harness(dev, Cfg(heads, kv, d, 64, ctx_max, 8, v2=True), qn, kn)
    pre = 3000
    k0, v0 = rbf(rng.standard_normal((pre, kv, d)) * 0.5), rbf(rng.standard_normal((pre, kv, d)))
    for h in (h1, h2):
        kk, vv = np.zeros((ctx_max, kv, d), np.float32), np.zeros((ctx_max, kv, d), np.float32)
        kk[:pre], vv[:pre] = k0, v0
        h.set_caches(kk, vv)
    proj = _random_proj(rng, Cfg(heads, kv, d, 64, ctx_max, 8), 4)
    a, b = h1.step(proj, pre), h2.step(proj, pre)
    cos, max_abs, scale = _bars(b, a)
    # v1 rounds p̃ per 64-key chunk, v2 per 32-key sub-chunk: the two contracts differ by BF16 ULPs of P — the
    # composite bar, as against the layer oracle (each kernel meets the leaf bar against its own numpy model)
    assert cos > 0.9999 and max_abs <= 1e-2 * scale, (cos, max_abs, scale)
    h2.set_caches(*(lambda kk, vv: (kk, vv))(*[np.concatenate([x, np.zeros((ctx_max - pre, kv, d), np.float32)]) for x in (k0, v0)]))
    b2 = h2.step(proj, pre)
    assert np.array_equal(b, b2)
    h2.set_caches(*[np.concatenate([x, np.zeros((ctx_max - pre, kv, d), np.float32)]) for x in (k0, v0)])
    c = h2.step(proj, pre, t_active=2)
    assert np.array_equal(c[:2], b[:2]) and np.all(c[2:] == 0)
    kc, _ = h2.caches()
    assert np.all(kc[pre + 2:] == 0)


def test_v2_matches_layer_oracle(dev):
    torch = pytest.importorskip("torch")
    from monolith.nn import GQAAttention

    heads, kv, d, rot, hidden, ctx_max = 8, 2, 256, 64, 64, 512
    cfg = Cfg(heads, kv, d, rot, ctx_max, 8, v2=True)
    rng = np.random.default_rng(9)
    torch.manual_seed(9)
    mod = GQAAttention(hidden, heads, kv, d, rot, THETA, EPS, hf_prefix="x.", prefix="l.", max_context=ctx_max)
    qn_w, kn_w = (rng.standard_normal(d) * 0.1).astype(np.float32), (rng.standard_normal(d) * 0.1).astype(np.float32)
    mod.set_param("q_norm", torch.from_numpy(qn_w).to(torch.bfloat16))
    mod.set_param("k_norm", torch.from_numpy(kn_w).to(torch.bfloat16))
    perm = rope_head_perm(d, rot)
    h = Harness(dev, cfg, (1.0 + bf16_to_f32(f32_to_bf16(qn_w)))[perm], (1.0 + bf16_to_f32(f32_to_bf16(kn_w)))[perm])
    state = {"l.k_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16), "l.v_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16)}
    col_perm = mod.qkv.row_perm
    pos = 0
    for t in (6, 1, 3, 8):
        proj_hf = torch.from_numpy(rng.standard_normal((t, cfg.n1)).astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            ref = mod.mix(proj_hf, state, pos).float().numpy()
        got = h.step(f32_to_bf16(proj_hf.float().numpy()[:, col_perm]), pos)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, (t, pos, cos, max_abs, scale)
        pos += t


def test_steal_variant_is_exactly_once_and_identical(dev):
    """The own-slice + steal claim protocol (#44, kernels/common/steal.metal) on the attention core: every block is
    claimed exactly once and the outputs equal the static-slice kernel's bit for bit — with the nominal crew, with a
    third of it missing (the present SIMD-groups steal the rest) and with a surplus (the extra ones own nothing)."""
    cfg = Cfg(heads=8, kv=2, d=64, rot=32, ctx_max=1024, t_max=4)
    scfg = Cfg(heads=8, kv=2, d=64, rot=32, ctx_max=1024, t_max=4, steal=True)
    rng = np.random.default_rng(21)
    qn, kn = _norms(rng, cfg.d)
    plain, steal = Harness(dev, cfg, qn, kn), Harness(dev, scfg, qn, kn)
    k = rng.standard_normal((cfg.ctx_max, cfg.kv, cfg.d)).astype(np.float32) * 0.5
    v = rng.standard_normal((cfg.ctx_max, cfg.kv, cfg.d)).astype(np.float32)
    proj = _random_proj(rng, cfg, 4)
    position = 700                                                        # 11 chunks × 2 kv heads × 4 row groups = 88 blocks
    plain.set_caches(k, v)
    ref = plain.step(proj, position)
    for name, disp in (("nominal", None), ("missing a third", plain.n_sg * 2 // 3), ("surplus", plain.n_sg * 2)):
        steal.set_caches(k, v)
        out = steal.step(proj, position, dispatch_sg=disp)
        hits = steal.block_hits()
        assert hits.shape == (88,) and np.all(hits == 1), (name, hits.min(), hits.max())
        assert np.array_equal(out, ref), name
        kk, vv = steal.caches()
        rk, rv = plain.caches()
        assert np.array_equal(kk, rk) and np.array_equal(vv, rv), name


# ---- v3: core and merge in one dispatch, a threadgroup per query row (#113) -----------------------------------------------

@pytest.mark.parametrize("cfg", [Cfg(16, 8, 128, 128, 1024, 8, v3=True), Cfg(8, 2, 256, 64, 512, 8, v3=True, nsg3=16),
                                 Cfg(4, 1, 128, 32, 4096, 8, v3=True, nsg3=16), Cfg(4, 2, 128, 128, 128, 4, gate=False, v3=True)],
                         ids=["v3_0.6B_geometry", "v3_d256_16sg", "v3_d128_16sg_long", "v3_nogate_fullrope"])
def test_v3_matches_kernel_contract(dev, cfg):
    """v3 against the numpy model of the kernel contract: its fold is per key within a SIMD-group (exact FP32 rescaling)
    and over the SIMD-groups in threadgroup memory, so it meets the same 2-ulp bar as v2's hierarchical fold; contexts
    from empty to thousands of keys, T = 1 / 5 / 4 (causal inside the step), the new keys appended by the first row."""
    rng = np.random.default_rng(cfg.heads * 37 + cfg.d + cfg.ctx_max)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    k_ref, v_ref = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    pre = min(3000, cfg.ctx_max - 32) if cfg.ctx_max > 256 else 0        # a long filled prefix, or an empty cache
    k_ref[:pre], v_ref[:pre] = rbf(rng.standard_normal((pre, cfg.kv, cfg.d)) * 0.5), rbf(rng.standard_normal((pre, cfg.kv, cfg.d)))
    h.set_caches(k_ref, v_ref)
    pos = pre
    for t in (1, 5, 1, 4):
        if t > cfg.t_max:
            continue
        proj = _random_proj(rng, cfg, t)
        got = h.step(proj, pos)
        ref = ref_step(h, bf16_to_f32(proj), pos, k_ref, v_ref)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.99999 and max_abs <= 2 * _ulp(scale), (t, pos, cos, max_abs, scale)
        kc, vc = h.caches()
        assert np.abs(kc[: pos + t] - k_ref[: pos + t]).max() <= 1e-2 * max(np.abs(k_ref[: pos + t]).max(), 1e-6)
        assert np.array_equal(vc[: pos + t], v_ref[: pos + t])
        pos += t


@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("heads", [4, 8])
def test_v3_is_bit_stable_and_t_active(dev, mma, heads):
    """Repeats are bit-identical; t_active limits the rows and the append; the same inputs through v1 agree within the
    composite bar (the two contracts round p̃ against different maxima)."""
    kv, d, ctx_max = 2, 128, 2048
    rng = np.random.default_rng(78)
    qn, kn = _norms(rng, d)
    h1 = Harness(dev, Cfg(heads, kv, d, 64, ctx_max, 8), qn, kn)
    h3 = Harness(dev, Cfg(heads, kv, d, 64, ctx_max, 8, v3=not mma, mma=mma), qn, kn)
    pre = 1500
    k0, v0 = rbf(rng.standard_normal((pre, kv, d)) * 0.5), rbf(rng.standard_normal((pre, kv, d)))
    full = lambda x: np.concatenate([x, np.zeros((ctx_max - pre, kv, d), np.float32)])   # noqa: E731
    for h in (h1, h3):
        h.set_caches(full(k0), full(v0))
    proj = _random_proj(rng, Cfg(heads, kv, d, 64, ctx_max, 8), 4)
    a, b = h1.step(proj, pre), h3.step(proj, pre)
    cos, max_abs, scale = _bars(b, a)
    assert cos > 0.9999 and max_abs <= 1e-2 * scale, (cos, max_abs, scale)
    h3.set_caches(full(k0), full(v0))
    assert np.array_equal(b, h3.step(proj, pre))
    h3.set_caches(full(k0), full(v0))
    c = h3.step(proj, pre, t_active=2)
    assert np.array_equal(c[:2], b[:2]) and np.all(c[2:] == 0)
    kc, _ = h3.caches()
    assert np.all(kc[pre + 2:] == 0)


def test_v3_matches_layer_oracle(dev):
    torch = pytest.importorskip("torch")
    from monolith.nn import GQAAttention

    heads, kv, d, rot, hidden, ctx_max = 8, 2, 128, 64, 64, 512
    cfg = Cfg(heads, kv, d, rot, ctx_max, 8, v3=True)
    rng = np.random.default_rng(10)
    torch.manual_seed(10)
    mod = GQAAttention(hidden, heads, kv, d, rot, THETA, EPS, hf_prefix="x.", prefix="l.", max_context=ctx_max)
    qn_w, kn_w = (rng.standard_normal(d) * 0.1).astype(np.float32), (rng.standard_normal(d) * 0.1).astype(np.float32)
    mod.set_param("q_norm", torch.from_numpy(qn_w).to(torch.bfloat16))
    mod.set_param("k_norm", torch.from_numpy(kn_w).to(torch.bfloat16))
    perm = rope_head_perm(d, rot)
    h = Harness(dev, cfg, (1.0 + bf16_to_f32(f32_to_bf16(qn_w)))[perm], (1.0 + bf16_to_f32(f32_to_bf16(kn_w)))[perm])
    state = {"l.k_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16), "l.v_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16)}
    col_perm = mod.qkv.row_perm
    pos = 0
    for t in (6, 1, 3, 8):
        proj_hf = torch.from_numpy(rng.standard_normal((t, cfg.n1)).astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            ref = mod.mix(proj_hf, state, pos).float().numpy()
        got = h.step(f32_to_bf16(proj_hf.float().numpy()[:, col_perm]), pos)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, (t, pos, cos, max_abs, scale)
        pos += t


@pytest.mark.parametrize('heads,kv,rot,ctx_max,gate', [
    (16, 8, 128, 1024, False), (32, 8, 32, 4096, True), (6, 2, 64, 512, True),
])
def test_mma_matches_kernel_contract(dev, heads, kv, rot, ctx_max, gate):
    cfg = Cfg(heads, kv, 128, rot, ctx_max, 8, gate=gate, mma=True)
    test_v3_matches_kernel_contract(dev, cfg)


@pytest.mark.parametrize("position,large_query", [(0, False), (128, False), (4096, False), (32768, False), (128, True)])
def test_mma_fp16_probabilities_keep_bf16_operands(dev, position, large_query):
    cfg = Cfg(6, 2, 128, 128, position+16, 8, gate=False, mma=True, qk_norm=False)
    cfg.prob_fp16 = True
    rng = np.random.default_rng(915)
    h = Harness(dev, cfg, *_norms(rng, cfg.d))
    caches = [rbf(rng.normal(0, .1, (cfg.ctx_max, cfg.kv, cfg.d))) for _ in range(2)]
    h.set_caches(*caches)
    proj = _random_proj(rng, cfg, 8)
    if large_query:
        # A whole-operand FP16 conversion would overflow this valid BF16 query.
        proj[:, 0] = f32_to_bf16(np.array([131072], np.float32))[0]
    expected = ref_step(h, bf16_to_f32(proj), position, *caches)
    got = h.step(proj, position)
    cos, error, scale = _bars(got, expected)
    assert cos > .99999 and error <= 2 * _ulp(scale), (cos, error, scale)
    np.testing.assert_array_equal(h.caches()[1], caches[1])
    np.testing.assert_array_equal(h.step(proj, position), got)


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("heads", [4, 12])
def test_mma_step_state_and_strided_grid(dev, mode, heads):
    """A short crew must visit every chunk; LM row sources and done/empty steps are predicated on GPU."""
    rng = np.random.default_rng(190)
    cfg = Cfg(heads, 2, 128, 64, 256, 8, mma=True)
    qn, kn = _norms(rng, 128)
    ref = Harness(dev, cfg, qn, kn)
    dyn = Harness(dev, Cfg(heads, 2, 128, 64, 256, 8, mma=True, step_state=True, lm_mode=mode), qn, kn)
    pos, t = 123, 5
    proj = _random_proj(rng, cfg, t)
    state = dict(position=pos, t_this_step=t, n_inject=t, n_chain=t)
    if mode == 1:
        state['position'] += t
    elif mode == 2:
        state['position'] -= 2
    elif mode == 3:
        state.update(position=pos + 2, n_inject=2, n_chain=3)
    expected = ref.step(proj, pos)
    got = dyn.step(proj, pos, dispatch_sg=3, state=state)
    assert np.array_equal(expected, got)
    assert all(np.array_equal(a, b) for a, b in zip(ref.caches(), dyn.caches()))
    before = dyn.caches()
    for stop in [dict(state, done=1), dict(state, t_this_step=0, n_inject=0, n_chain=0)]:
        assert np.all(dyn.step(proj, pos, dispatch_sg=3, state=stop) == 0)
        assert all(np.array_equal(a, b) for a, b in zip(before, dyn.caches()))


@pytest.mark.parametrize("ctx", [256, 4096])
def test_mma_wide_heads(dev, ctx):
    test_v3_matches_kernel_contract(dev, Cfg(8, 2, 256, 64, ctx, 8, gate=True, mma=True))


@pytest.mark.parametrize("position", [0, 128, 1024])
def test_packed_mma_merge_preserves_partial_rows_and_caches(dev, position):
    rng = np.random.default_rng(1907)
    cfg = Cfg(32, 8, 128, 128, 1152, 8, gate=False, mma=True, step_state=True)
    qn, kn = _norms(rng, 128)
    original = Harness(dev, cfg, qn, kn)
    packed_cfg = Cfg(32, 8, 128, 128, 1152, 8, gate=False, mma=True, step_state=True)
    packed_cfg.merge_sg = 4
    packed = Harness(dev, packed_cfg, qn, kn)
    caches = [rbf(rng.normal(0, .1, (cfg.ctx_max, cfg.kv, cfg.d))) for _ in range(2)]
    proj = _random_proj(rng, cfg, 8)
    for active in (8, 3, 0, 1, 7):
        for h in (original, packed):
            h.set_caches(*caches)
        expected = original.step(proj, position, t_active=active)
        actual = packed.step(proj, position, t_active=active)
        np.testing.assert_array_equal(actual, expected)
        assert np.all(actual[active:] == 0)
        for a, b in zip(original.caches(), packed.caches()):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("position", [0, 60, 128, 248, 249, 256, 1024])
@pytest.mark.parametrize("t", [1, 4, 8])
def test_adaptive_mma_matches_fixed_chunk_and_cache(dev, position, t):
    """Cross chunk and branch boundaries with live prefixes, inactive rows, and a crew that reuses scratch."""
    rng = np.random.default_rng(211 + position + t)
    qn, kn = _norms(rng, 128)
    chunk = 32 if position + t <= 256 else 64
    gate = t != 4
    fixed = Harness(dev, Cfg(4, 2, 128, 64, 1152, 8, chunk=chunk, gate=gate, mma=True), qn, kn)
    cfg = Cfg(4, 2, 128, 64, 1152, 8, gate=gate, mma=True, adaptive=True, step_state=True)
    adaptive = Harness(dev, cfg, qn, kn)
    caches = [rbf(rng.standard_normal((cfg.ctx_max, cfg.kv, cfg.d)).astype(np.float32) * .1) for _ in range(2)]
    fixed.set_caches(*caches)
    adaptive.set_caches(*caches)
    proj = _random_proj(rng, cfg, 8)
    expected = fixed.step(proj, position, t_active=t, dispatch_sg=3)
    got = adaptive.step(proj, position, t_active=t, dispatch_sg=3)
    assert np.array_equal(got, expected)
    assert np.all(got[t:] == 0)
    assert all(np.array_equal(a, b) for a, b in zip(fixed.caches(), adaptive.caches()))
    before = adaptive.caches()
    for state in [dict(position=position, t_this_step=t, done=1), dict(position=position, t_this_step=0)]:
        assert np.all(adaptive.step(proj, position, dispatch_sg=3, state=state) == 0)
        assert all(np.array_equal(a, b) for a, b in zip(before, adaptive.caches()))


@pytest.mark.parametrize("kind,d,gate", [("v1", 64, True), ("v2", 128, False), ("v3", 128, True),
                                       ("mma", 256, True), ("adaptive", 128, False)])
def test_static_geometry_keeps_position_and_active_length_dynamic(dev, kind, d, gate):
    rng = np.random.default_rng(159)
    configs = [Cfg(4, 2, d, d, 512, 4, gate=gate, v2=kind == "v2", v3=kind == "v3",
                   mma=kind in ("mma", "adaptive"), adaptive=kind == "adaptive",
                   step_state=kind in ("mma", "adaptive")) for _ in range(2)]
    configs[1].specialize = True
    configs[1].single_block = kind == "v3"
    qn, kn = [rng.uniform(.8, 1.2, d).astype(np.float32) for _ in range(2)]
    hs = [Harness(dev, c, qn, kn) for c in configs]
    prefix = [rbf(rng.normal(0, .2, (512, 2, d))) for _ in range(2)]
    for h in hs:
        h.set_caches(*prefix)
    # Reuse one specialized pipeline across partial/full/empty steps and the
    # adaptive chunk boundary. Neither position nor active length may freeze.
    for position, active in [(127, 1), (252, 4), (253, 4), (300, 0)]:
        proj = f32_to_bf16(rng.normal(0, .3, (4, configs[0].n1)).astype(np.float32))
        out = [h.step(proj, position, active) for h in hs]
        np.testing.assert_array_equal(*out)
        for a, b in zip(hs[0].caches(), hs[1].caches()):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nsg", [16, 32])
@pytest.mark.parametrize("position", [31, 32, 63, 64, 65, 127, 128, 129, 1023])
@pytest.mark.parametrize("single_block", [False, True])
def test_v3_cached_key_pair_boundaries(dev, nsg, position, single_block):
    """Pair batching, the unpaired cached tail and new causal keys meet the
    unchanged independent kernel contract, including exactly one or no pair."""
    cfg = Cfg(4, 2, 128, 64, 2048, 4, v3=True, nsg3=nsg, single_block=single_block)
    rng = np.random.default_rng(79 + position)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    kc = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    vc = np.zeros_like(kc)
    kc[:position] = rbf(rng.normal(0, .5, kc[:position].shape))
    vc[:position] = rbf(rng.normal(0, .5, vc[:position].shape))
    h.set_caches(kc, vc)
    proj = _random_proj(rng, cfg, 3)
    got = h.step(proj, position)
    ref = ref_step(h, bf16_to_f32(proj), position, kc, vc)
    cos, max_abs, scale = _bars(got, ref)
    assert cos > 0.99999 and max_abs <= 2 * _ulp(scale), (cos, max_abs, scale)
    got_k, got_v = h.caches()
    assert np.array_equal(got_v[:position + 3], vc[:position + 3])
    assert np.abs(got_k[:position + 3] - kc[:position + 3]).max() <= 1e-2 * max(np.abs(kc[:position + 3]).max(), 1e-6)


@pytest.mark.parametrize("kind", ["v1", "v2", "v3", "mma", "adaptive"])
@pytest.mark.parametrize("position", [0, 249, 1024])
def test_attention_without_query_key_norm(dev, kind, position):
    rng = np.random.default_rng(1729)
    cfg = Cfg(4, 2, 128, 128, 1152, 4, gate=False, v2=kind == "v2", v3=kind == "v3",
              mma=kind in ("mma", "adaptive"), adaptive=kind == "adaptive", qk_norm=False)
    # Poison disabled norm bindings: the kernel must not load either table.
    harness = Harness(dev, cfg, np.full(128, np.nan), np.full(128, np.nan))
    caches = [rbf(rng.normal(0, .1, (cfg.ctx_max, cfg.kv, cfg.d))) for _ in range(2)]
    harness.set_caches(*caches)
    proj = _random_proj(rng, cfg, 4)
    expected = ref_step(harness, bf16_to_f32(proj), position, *caches)
    actual = harness.step(proj, position)
    np.testing.assert_allclose(actual, expected, atol=.003, rtol=.02)
    for actual_cache, expected_cache in zip(harness.caches(), caches):
        np.testing.assert_array_equal(actual_cache, expected_cache)
    np.testing.assert_array_equal(harness.step(proj, position), actual)


@pytest.mark.parametrize("position", [0, 128, 1024])
def test_packed_merge_buffer_bounds(dev, position):
    """Validate the changed merge alone: instrumenting the unchanged MMA core
    exceeds this device's threadgroup-memory budget in Shader Validation."""
    from monolith.core import StepStateLayout
    rng = np.random.default_rng(71)
    chunks = -(-(position + 8) // 64)
    layout = StepStateLayout(t_max=8, gamma_max=7)
    source = kernels.gqa_source().replace(kernels.PRELUDE, kernels.PRELUDE + layout.to_msl())
    macros = dict(kernels.gqa_macros(128, chunk=64), STEP_STATE="1", FIXED_CHUNK="1",
                  **kernels.perm_out_macros(4096, 32, 128))
    pso = nt.Pipeline(nt.Library(dev, source, macros), "gqa_merge")
    partials = rng.normal(0, .2, (8, chunks, 32, 128)).astype(np.float32)
    md = rng.uniform(.5, 1.5, (8, chunks, 32, 2)).astype(np.float32)
    po, pm = nt.Buffer(dev, partials.tobytes()), nt.Buffer(dev, md.tobytes())
    params = kernels.gqa_params(heads=32, kv_heads=8, t_active=8, position=position, n_sg=80,
        q_off=0, gate_off=0, k_off=0, v_off=0, in_stride=4096, out_stride=4096,
        ctx_max=position + 8, eps=1e-6, scaling=1, has_gate=False, n_chunks_max=chunks, rows_max=32)
    for active in (8, 3, 0, 1, 7):
        st = nt.Buffer(dev, layout.pack(dict(position=position, t_this_step=active)))
        results = []
        for groups in (1, 4):
            out = nt.Buffer(dev, 8 * 4096 * 2); out.fill(0)
            dispatch = (nt.Dispatch().pipeline(pso).buffer(0, po).buffer(1, pm).buffer(2, po)
                        .buffer(3, out).bytes(4, params).buffer(15, st)
                        .grid(256 // groups).threadgroup(32 * groups))
            result = nt.Queue(dev).run([dispatch])
            assert not result.error, result.error
            results.append(np.frombuffer(out.read(0, out.nbytes), np.uint16).reshape(8, 4096))
        np.testing.assert_array_equal(*results)
        assert not results[1][active:].any()


@pytest.mark.parametrize("position,t", [(128, 1), (128, 8), (32768, 8)])
def test_v2_merge_permuted_output(dev, position, t):
    """A fused output permutation must preserve every merge result bit."""
    rng = np.random.default_rng(819)
    heads, kv, d = 8, 2, 128
    rows, chunks = t * heads // kv, -(-(position + t) // 32)
    partials = rng.normal(0, .2, (kv, chunks, rows, d)).astype(np.float32)
    md = rng.uniform(.5, 1.5, (kv, chunks, rows, 2)).astype(np.float32)
    po, pm = nt.Buffer(dev, partials.tobytes()), nt.Buffer(dev, md.tobytes())
    params = kernels.gqa_params(heads=heads, kv_heads=kv, t_active=t, position=position,
        n_sg=80, q_off=0, gate_off=0, k_off=0, v_off=0, in_stride=heads*d,
        out_stride=heads*d, ctx_max=position+t, eps=EPS, scaling=1, has_gate=False,
        n_chunks_max=chunks, rows_max=rows)
    outputs = []
    for permuted in (False, True):
        macros = kernels.gqa_v2_macros(d, rmax=rows, rg=4)
        if permuted:
            macros.update(kernels.perm_out_macros(heads*d, 8, 128))
        pipeline = nt.Pipeline(nt.Library(dev, kernels.gqa_source(True), macros), "gqa_merge_v2")
        out = nt.Buffer(dev, t*heads*d*2)
        dispatch = (nt.Dispatch().pipeline(pipeline).buffer(0, po).buffer(1, pm)
                    .buffer(2, po).buffer(3, out).bytes(4, params)
                    .grid(t*heads).threadgroup(32))
        result = nt.Queue(dev).run([dispatch])
        assert not result.error, result.error
        outputs.append(np.frombuffer(out.read(0, out.nbytes), np.uint16).reshape(t, heads*d))
    columns = kernels.x_permute_columns(heads*d, 8, 128)
    np.testing.assert_array_equal(outputs[1], outputs[0][:, columns])


@pytest.mark.parametrize("position,t", [(0, 8), (255, 8), (4095, 1), (8191, 8), (8696, 8)])
@pytest.mark.parametrize("specialize", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_direct_mma_causal_tail_and_cache(dev, position, t, specialize, qk_norm):
    cfg = Cfg(32, 8, 128, 128, 8704, 8, chunk=256, gate=False,
              mma=True, step_state=True, direct=True, qk_norm=qk_norm)
    cfg.specialize = specialize
    rng = np.random.default_rng(3019)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    k = rbf(rng.standard_normal((cfg.ctx_max, cfg.kv, cfg.d)) * .5)
    v = rbf(rng.standard_normal(k.shape))
    proj = _random_proj(rng, cfg, 8)
    h.set_caches(k, v)
    got = h.step(proj, position, t_active=t, dispatch_sg=17)
    kr, vr = k.copy(), v.copy()
    ref = ref_step(h, bf16_to_f32(proj[:t]), position, kr, vr)
    cos, error, scale = _bars(got[:t], ref)
    assert cos > .99999 and error <= 2 * _ulp(scale), (cos, error, scale)
    assert np.all(got[t:] == 0)
    actual_k, actual_v = h.caches()
    np.testing.assert_array_equal(actual_k[:position], k[:position])
    np.testing.assert_array_equal(actual_k[position+t:], k[position+t:])
    np.testing.assert_array_equal(actual_v, vr)
    assert np.max(np.abs(actual_k - kr)) <= .01 * np.max(np.abs(kr))
    h.set_caches(k, v)
    np.testing.assert_array_equal(h.step(proj, position, t_active=t, dispatch_sg=17), got)
    for state in (dict(position=position, t_this_step=t, done=1), dict(position=position, t_this_step=0)):
        h.set_caches(k, v)
        assert np.all(h.step(proj, position, state=state) == 0)
        assert all(np.array_equal(a, b) for a, b in zip(h.caches(), (k, v)))
