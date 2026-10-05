"""gdn_mixer against the GatedDeltaNet layer oracle (torch): the kernel follows the reference's rounding order, so
the bars are the leaf gates — output ≤ 2 BF16 ULP at its RMS, recurrent state ≤ 8 FP32 ULP of its largest value,
conv state exact — on fresh and filled states, T = 1 / 4 / 8 (several token passes), v_heads = k_heads and 3×,
a|b from the same or a second projection buffer, continuation across steps, bit-identical repeat runs."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt

EPS = 1e-6
CW = 4


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _module(hidden, hk, hv, dk, dv, seed):
    torch = pytest.importorskip("torch")
    from monolith.nn import GatedDeltaNet

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = GatedDeltaNet(hidden, hk, hv, dk, dv, CW, EPS, hf_prefix="x.", prefix="l.")
    conv_dim = 2 * hk * dk + hv * dv
    m.set_param("conv_w", torch.from_numpy((rng.standard_normal((conv_dim, 1, CW)) * 0.3).astype(np.float32)).to(torch.bfloat16))
    m.set_param("a_log", torch.from_numpy(rng.uniform(-2, 1, hv).astype(np.float32)).to(torch.bfloat16))
    m.set_param("dt_bias", torch.from_numpy(rng.standard_normal(hv).astype(np.float32)).to(torch.bfloat16))
    m.set_param("norm_w", torch.from_numpy((1 + rng.standard_normal(dv) * 0.2).astype(np.float32)).to(torch.bfloat16))
    return m, rng


class Harness:
    def __init__(self, dev, m, t_max, *, ab_separate=False, slice_cols=8, slices_per_block=4, tokens_per_pass=None, prepared=False, perm_out=None, fused_norm=False, specialize=False, local_prepare=False, local_groups=None, single_pass=False):
        """The engine's configuration: two state slots by step parity read from StepState (a single slot races when
        several value heads share a key head's conv window — the kernel's note)."""
        from monolith.core import StepStateLayout

        self.dev, self.m, self.ab_separate = dev, m, ab_separate
        self.prepared, self.fused_norm = prepared, fused_norm
        self.local_prepare = local_prepare
        assert not local_prepare or (prepared and slices_per_block == 1)
        self.local_groups = local_groups or m.dv // slice_cols
        self.layout = StepStateLayout(t_max=max(8, t_max), gamma_max=7)
        macros = dict(kernels.gdn_macros(m.dk, m.dv, conv_width=CW, t=t_max, slice_cols=slice_cols, slices_per_block=slices_per_block,
                                         tokens_per_pass=tokens_per_pass, slots=2), STEP_STATE="1")
        if perm_out is not None:
            macros.update(kernels.perm_out_macros(m.value_dim, *perm_out))
        if prepared:
            macros["PREPARED"] = "1"
        if fused_norm:
            assert prepared and slice_cols == 4 and slices_per_block == 1
            macros["FUSED_NORM"] = "1"
        if local_prepare:
            macros["LOCAL_PREPARE"] = "1"
            macros["LOCAL_GROUPS"] = f"{self.local_groups}u"
        if single_pass:
            assert local_prepare and t_max <= int(macros["TP"].rstrip("u"))
            macros["SINGLE_PASS"] = "1"
        self.source = kernels.gdn_source().replace(kernels.PRELUDE, kernels.PRELUDE + self.layout.to_msl() + "\n", 1)
        self.macros, self.specialize = macros, specialize
        lib = nt.Library(dev, self.source, macros)
        self.pso, self.pso_norm = nt.Pipeline(lib, "gdn_mixer"), nt.Pipeline(lib, "gdn_norm")
        if prepared and not local_prepare:
            self.pso_prepare = nt.Pipeline(lib, "gdn_prepare")
            self.prep = nt.Buffer(dev, t_max * m.v_heads * (2 * m.dk + m.dv + 2) * 4)
        self.o_part = nt.Buffer(dev, kernels.gdn_workspace(t_max, m.v_heads, m.dv))
        self.conv_bytes, self.rec_bytes = m.conv_dim * (CW - 1) * 2, m.v_heads * m.dk * m.dv * 4
        self.conv_state = nt.Buffer(dev, 2 * self.conv_bytes); self.conv_state.fill(0)
        self.rec_state = nt.Buffer(dev, 2 * self.rec_bytes); self.rec_state.fill(0)
        self.step_no = 0                                                     # the pass reads slot step & 1 and writes the other
        self.st = nt.Buffer(dev, self.layout.size); self.st.fill(0)
        conv_w = m.param("conv_w").reshape(m.conv_dim, CW).float().numpy()
        self.aux = [nt.Buffer(dev, f32_to_bf16(conv_w).tobytes()),
                    nt.Buffer(dev, (-np.exp(m.param("a_log").float().numpy())).astype(np.float32).tobytes()),
                    nt.Buffer(dev, m.param("dt_bias").float().numpy().astype(np.float32).tobytes()),
                    nt.Buffer(dev, m.param("norm_w").float().numpy().astype(np.float32).tobytes())]
        self.n_sg = m.v_heads * m.dv // slice_cols if local_prepare or fused_norm else 12 * dev.info().gpu_cores

    def set_state(self, state):                                              # into the slot the next pass reads
        slot = self.step_no & 1
        self.conv_state.write(f32_to_bf16(state["l.conv_state"].float().numpy()).tobytes(), slot * self.conv_bytes)
        rec = state["l.rec_state"].float().numpy().astype(np.float32)
        self.rec_state.write(rec.tobytes(), slot * self.rec_bytes)

    def get_state(self):                                                     # from the slot the last pass wrote
        m = self.m
        slot = self.step_no & 1
        conv = bf16_to_f32(np.frombuffer(self.conv_state.read(slot * self.conv_bytes, self.conv_bytes), dtype=np.uint16)).reshape(m.conv_dim, CW - 1)
        rec = np.frombuffer(self.rec_state.read(slot * self.rec_bytes, self.rec_bytes), dtype=np.float32).reshape(m.v_heads, m.dk, m.dv)
        return conv, rec

    def step(self, proj, t_active=None, done=False):
        """``proj`` torch BF16 [T, N1] in checkpoint column order."""
        m = self.m
        t = proj.shape[0]
        pf = f32_to_bf16(proj.float().numpy())
        kd, vd, hv = m.key_dim, m.value_dim, m.v_heads
        # the projection's part order: z | qkv | a | b (the z rows lead so the gate GEMV is a block-aligned range)
        if self.ab_separate:                                   # the 27B layout: z|qkv in one slab, a|b in another
            main, ab = pf[:, : vd + m.conv_dim], np.ascontiguousarray(pf[:, vd + m.conv_dim:])
            a_off, b_off, ab_stride = 0, hv, 2 * hv
        else:
            main, ab = pf, pf
            a_off, b_off, ab_stride = vd + m.conv_dim, vd + m.conv_dim + hv, pf.shape[1]
        params = kernels.gdn_params(hv=hv, hk=m.k_heads, t_active=t if t_active is None else t_active, q_off=vd, k_off=vd + kd, v_off=vd + 2 * kd,
                                    z_off=0, a_off=a_off, b_off=b_off, in_stride=main.shape[1], ab_stride=ab_stride,
                                    ab_separate=self.ab_separate, out_stride=vd, n_sg=self.n_sg, key_dim=kd, eps=EPS)
        if self.specialize:
            source, constants = kernels.specialize_params(self.source, "gdn", params)
            if self.fused_norm:
                source, norm_constants = kernels.specialize_params(source, "gdn", params, "np")
                constants.update(norm_constants)
            lib = nt.Library(self.dev, source, dict(self.macros, **constants))
            self.pso, self.pso_norm = nt.Pipeline(lib, "gdn_mixer"), nt.Pipeline(lib, "gdn_norm")
            if self.prepared and not self.local_prepare:
                self.pso_prepare = nt.Pipeline(lib, "gdn_prepare")
            self.specialize = False
        out = nt.Buffer(self.dev, t * vd * 2); out.fill(0)
        mb = nt.Buffer(self.dev, main.tobytes())
        abb = nt.Buffer(self.dev, ab.tobytes()) if self.ab_separate else mb
        self.st.write(self.layout.pack({"step": self.step_no, "t_this_step": t if t_active is None else t_active, "done": int(done)}), 0)
        d = (nt.Dispatch().pipeline(self.pso).buffer(0, mb).buffer(1, abb).buffer(2, self.conv_state).buffer(3, self.rec_state)
             .buffer(4, self.aux[0]).buffer(5, self.aux[1]).buffer(6, self.aux[2]).buffer(7, self.o_part)
             .bytes(9, params).buffer(15, self.st).grid(-(-(self.n_sg * 32) // 384)).threadgroup(384).barrier())
        d2 = (nt.Dispatch().pipeline(self.pso_norm).buffer(0, self.o_part).buffer(1, mb).buffer(2, self.aux[3]).buffer(3, out)
              .bytes(4, params).buffer(15, self.st).grid(t * hv).threadgroup(32))
        ds = [d, d2]
        if self.local_prepare:
            d.grid(self.n_sg // self.local_groups).threadgroup(32 * self.local_groups)
        if self.fused_norm:
            d.bytes(11, params).buffer(12, self.aux[3]).buffer(13, mb).buffer(14, out).grid(hv).threadgroup(32 * m.dv // 4)
            ds = [d]
        if self.prepared and not self.local_prepare:
            dp = (nt.Dispatch().pipeline(self.pso_prepare).buffer(0, mb).buffer(1, abb).buffer(2, self.conv_state)
                  .buffer(4, self.aux[0]).buffer(5, self.aux[1]).buffer(6, self.aux[2]).buffer(8, self.prep)
                  .bytes(9, params).buffer(15, self.st).grid(3 * t * hv).threadgroup(32).barrier())
            if hasattr(self, 'prefill_prepare_geometry'):
                grid, group = self.prefill_prepare_geometry
                dp.grid(*grid).threadgroup(*group)
            d.buffer(8, self.prep)
            ds.insert(0, dp)
        r = nt.Queue(self.dev).run(ds)
        assert not r.error, r.error
        self.step_no += 1
        return bf16_to_f32(np.frombuffer(out.read(0, t * vd * 2), dtype=np.uint16).reshape(t, vd))


def _proj(rng, torch, t, n1):
    return torch.from_numpy(rng.standard_normal((t, n1)).astype(np.float32)).to(torch.bfloat16)


def _check(got, ref_out, got_rec, ref_rec, got_conv, ref_conv):
    # the output is a BF16 tensor whose every element went through the same rounding chain as the oracle's (norm,
    # weight, gate), so the element-wise ULP is the meaningful gate here: FP32 summation-order noise flips a rounding
    # boundary in ~1 element per thousand by one ULP; the at-RMS metric of the GEMV gate over-weights such an
    # element when it is far above the RMS
    mismatch = got_conv != ref_conv
    assert not mismatch.any(), {
        "indices": np.argwhere(mismatch)[:3].tolist(),
        "got": got_conv[mismatch][:3].tolist(), "expected": ref_conv[mismatch][:3].tolist(),
        "count": int(np.count_nonzero(mismatch)),
    }
    scale = float(np.abs(ref_rec).max())
    assert np.abs(got_rec - ref_rec).max() <= 8 * 2.0 ** -23 * max(scale, 1e-30), (np.abs(got_rec - ref_rec).max(), scale)
    chk = check_against_oracle(got, ref_out)
    assert chk.ok_rounded(), chk


@pytest.mark.parametrize("hk,hv,ab_separate", [(16, 16, False), (16, 48, True), (4, 4, False)], ids=["16x16", "16x48_ab_separate", "4x4"])
@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("prepared", [False, True])
def test_matches_layer_oracle(dev, hk, hv, ab_separate, t, prepared):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, hk, hv, 128, 128, seed=hk * 7 + hv + t)
    h = Harness(dev, m, t, ab_separate=ab_separate, prepared=prepared, slice_cols=4 if prepared else 8, slices_per_block=1 if prepared else 4, tokens_per_pass=8 if prepared else None)
    state = {"l.conv_state": torch.from_numpy((rng.standard_normal((m.conv_dim, CW - 1)) * 0.5).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy((rng.standard_normal((hv, 128, 128)) * 0.1).astype(np.float32))}
    h.set_state(state)
    for step in range(2):                                      # a step on the filled state, then a continuation
        proj = _proj(rng, torch, t, m.in_proj.n)
        with torch.no_grad():
            ref = m.mix(proj, state).float().numpy()
        got = h.step(proj)
        conv, rec = h.get_state()
        _check(got, ref, rec, state["l.rec_state"].numpy(), conv, state["l.conv_state"].float().numpy())


@pytest.mark.parametrize("prepared", [False, True])
def test_fresh_state_passes_and_repeat_runs(dev, prepared):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 16, 16, 128, 128, seed=3)
    state = {"l.conv_state": torch.zeros(m.conv_dim, CW - 1, dtype=torch.bfloat16), "l.rec_state": torch.zeros(16, 128, 128)}
    proj = _proj(rng, torch, 6, m.in_proj.n)
    with torch.no_grad():
        ref = m.mix(proj, state).float().numpy()
    for tp, spb in ((2, 1), (3, 4), (6, 16)):                  # 3, 2 and 1 token passes; 1, 4 and 16 slices per block
        h = Harness(dev, m, 6, tokens_per_pass=tp, slices_per_block=spb, prepared=prepared)
        got = h.step(proj)
        conv, rec = h.get_state()
        _check(got, ref, rec, state["l.rec_state"].numpy(), conv, state["l.conv_state"].float().numpy())
    h = Harness(dev, m, 6, prepared=prepared)
    a = h.step(proj)
    h2 = Harness(dev, m, 6, prepared=prepared)
    assert np.array_equal(a, h2.step(proj)) and np.array_equal(h.get_state()[1], h2.get_state()[1])


@pytest.mark.parametrize("prepared", [False, True])
def test_t_active_and_slice_width(dev, prepared):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 16, 16, 128, 128, seed=5)
    state = {"l.conv_state": torch.zeros(m.conv_dim, CW - 1, dtype=torch.bfloat16), "l.rec_state": torch.zeros(16, 128, 128)}
    proj = _proj(rng, torch, 4, m.in_proj.n)
    with torch.no_grad():
        ref = m.mix(proj[:2], state).float().numpy()
    h = Harness(dev, m, 4, slice_cols=16, slices_per_block=2, prepared=prepared)
    got = h.step(proj, t_active=2)
    assert np.all(got[2:] == 0)
    conv, rec = h.get_state()
    _check(got[:2], ref, rec, state["l.rec_state"].numpy(), conv, state["l.conv_state"].float().numpy())


def test_state_slots_and_commit_pass(dev):
    """SLOTS=2: a step reads slot (step & 1) and writes the other; the COMMIT variant, after the accept scan advanced
    ``step``, recomputes the recurrence for the committed tokens (``checkpoint_index``) from the slot the step read
    and overwrites the slot it wrote — the state of a rejected draft never survives; the next step continues from the
    committed state."""
    torch = pytest.importorskip("torch")
    from monolith.core.step_state import StepStateLayout

    m, rng = _module(64, 16, 16, 128, 128, seed=8)
    lay = StepStateLayout()
    src = kernels.PRELUDE + lay.to_msl() + "\n" + kernels.template("gdn_mixer.metal")
    main = dict(kernels.gdn_macros(128, 128, conv_width=CW, t=8, slots=2), STEP_STATE="1")
    com = dict(kernels.gdn_macros(128, 128, conv_width=CW, t=8, slots=2, commit=True), STEP_STATE="1")
    lib_m, lib_c = nt.Library(dev, src, main), nt.Library(dev, src, com)
    pso_m, pso_n, pso_c = nt.Pipeline(lib_m, "gdn_mixer"), nt.Pipeline(lib_m, "gdn_norm"), nt.Pipeline(lib_c, "gdn_mixer")
    hv, kd, vd = m.v_heads, m.key_dim, m.value_dim
    conv_bytes, rec_bytes = m.conv_dim * (CW - 1) * 2, hv * 128 * 128 * 4
    conv, rec = nt.Buffer(dev, 2 * conv_bytes), nt.Buffer(dev, 2 * rec_bytes)
    conv.fill(0); rec.fill(0)
    o_part = nt.Buffer(dev, kernels.gdn_workspace(8, hv, m.dv))
    sentinel = b"\xa5" * kernels.gdn_workspace(8, hv, m.dv)               # the commit pass binds a placeholder it must never write
    o_commit = nt.Buffer(dev, len(sentinel)); o_commit.write(sentinel, 0)
    conv_w = m.param("conv_w").reshape(m.conv_dim, CW).float().numpy()
    aux = [nt.Buffer(dev, f32_to_bf16(conv_w).tobytes()), nt.Buffer(dev, (-np.exp(m.param("a_log").float().numpy())).astype(np.float32).tobytes()),
           nt.Buffer(dev, m.param("dt_bias").float().numpy().astype(np.float32).tobytes()),
           nt.Buffer(dev, m.param("norm_w").float().numpy().astype(np.float32).tobytes())]
    n_sg = 12 * dev.info().gpu_cores

    def run(proj, state_fields, commit=False):
        t = proj.shape[0]
        pf = f32_to_bf16(proj.float().numpy())
        params = kernels.gdn_params(hv=hv, hk=m.k_heads, t_active=t, q_off=vd, k_off=vd + kd, v_off=vd + 2 * kd, z_off=0, a_off=vd + m.conv_dim,
                                    b_off=vd + m.conv_dim + hv, in_stride=pf.shape[1], ab_stride=pf.shape[1], ab_separate=False, out_stride=vd,
                                    n_sg=n_sg, key_dim=kd, eps=EPS)
        st = nt.Buffer(dev, lay.pack(state_fields))
        mb = nt.Buffer(dev, pf.tobytes())
        d = (nt.Dispatch().pipeline(pso_c if commit else pso_m).buffer(0, mb).buffer(1, mb).buffer(2, conv).buffer(3, rec).buffer(4, aux[0])
             .buffer(5, aux[1]).buffer(6, aux[2]).buffer(7, o_commit if commit else o_part).bytes(9, params).buffer(15, st)
             .grid(-(-(n_sg * 32) // 384)).threadgroup(384).barrier())
        ds = [d]
        out = nt.Buffer(dev, t * vd * 2); out.fill(0)
        if not commit:
            ds.append(nt.Dispatch().pipeline(pso_n).buffer(0, o_part).buffer(1, mb).buffer(2, aux[3]).buffer(3, out).bytes(4, params).buffer(15, st)
                      .grid(t * hv).threadgroup(32))
        r = nt.Queue(dev).run(ds)
        assert not r.error, r.error
        return bf16_to_f32(np.frombuffer(out.read(0, t * vd * 2), dtype=np.uint16).reshape(t, vd))

    def slot(i):
        c = bf16_to_f32(np.frombuffer(conv.read(i * conv_bytes, conv_bytes), dtype=np.uint16)).reshape(m.conv_dim, CW - 1)
        r = np.frombuffer(rec.read(i * rec_bytes, rec_bytes), dtype=np.float32).reshape(hv, 128, 128)
        return c, r

    def fresh():
        return {"l.conv_state": torch.zeros(m.conv_dim, CW - 1, dtype=torch.bfloat16), "l.rec_state": torch.zeros(hv, 128, 128)}

    proj = _proj(rng, torch, 3, m.in_proj.n)
    # step 0: three tokens (two drafts follow the anchor) from slot 0 into slot 1
    s_all = fresh()
    with torch.no_grad():
        ref = m.mix(proj, s_all).float().numpy()
    got = run(proj, {"step": 0, "t_this_step": 3})
    c1, r1 = slot(1)
    _check(got, ref, r1, s_all["l.rec_state"].numpy(), c1, s_all["l.conv_state"].float().numpy())
    assert np.all(slot(0)[1] == 0)                                    # the read slot is untouched
    # the accept scan committed two tokens (step → 1, checkpoint_index = 2): the commit pass rewrites slot 1 from slot 0
    s_two = fresh()
    with torch.no_grad():
        m.mix(proj[:2], s_two)
    run(proj, {"step": 1, "checkpoint_index": 2}, commit=True)
    assert o_commit.read(0, len(sentinel)) == sentinel                  # no read-out: the placeholder is untouched
    c1, r1 = slot(1)
    assert np.array_equal(c1, s_two["l.conv_state"].float().numpy())
    assert np.abs(r1 - s_two["l.rec_state"].numpy()).max() <= 8 * 2.0 ** -23 * float(np.abs(s_two["l.rec_state"].numpy()).max())
    # step 1: one token from slot 1 into slot 0 — the continuation of the committed state
    proj2 = _proj(rng, torch, 1, m.in_proj.n)
    with torch.no_grad():
        ref2 = m.mix(proj2, s_two).float().numpy()
    got2 = run(proj2, {"step": 1, "t_this_step": 1})
    c0, r0 = slot(0)
    _check(got2, ref2, r0, s_two["l.rec_state"].numpy(), c0, s_two["l.conv_state"].float().numpy())


@pytest.mark.parametrize("wpw,tk", [(8, 64), (32, 64), (32, 128)])
def test_norm_writes_consumer_permutation(dev, wpw, tk):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 16, 16, 128, 128, seed=25)
    proj = _proj(rng, torch, 4, m.in_proj.n)
    natural = Harness(dev, m, 4, prepared=True).step(proj, t_active=3)
    permuted = Harness(dev, m, 4, prepared=True, perm_out=(wpw, tk)).step(proj, t_active=3)
    columns = kernels.x_permute_columns(m.value_dim, wpw, tk)
    assert np.array_equal(permuted, natural[:, columns])
    assert np.all(permuted[3:] == 0)


@pytest.mark.parametrize("active", [0, 1, 4, 6, 8])
@pytest.mark.parametrize("tp,ab_separate,perm_out", [(8, False, None), (3, True, (8, 64)), (1, False, None)])
@pytest.mark.parametrize("local_prepare", [False, True])
def test_fused_norm_matches_separate_passes(dev, active, tp, ab_separate, perm_out, local_prepare):
    """Exact fusion equivalence with filled state, continuation, inactive rows,
    separate scalar projection, output permutation and multiple token passes."""
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 8, 16, 128, 128, seed=71)
    state = {"l.conv_state": torch.from_numpy(rng.normal(0, .2, (m.conv_dim, CW - 1)).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy(rng.normal(0, .1, (16, 128, 128)).astype(np.float32))}
    hs = [Harness(dev, m, 8, prepared=True, slice_cols=4, slices_per_block=1, tokens_per_pass=tp,
                  ab_separate=ab_separate, perm_out=perm_out, fused_norm=fused, local_prepare=local_prepare and fused) for fused in (False, True)]
    for h in hs:
        h.set_state(state)
    for _ in range(2):
        proj = _proj(rng, torch, 8, m.in_proj.n)
        out = [h.step(proj, t_active=active) for h in hs]
        np.testing.assert_array_equal(*out)
        for attr in ("conv_state", "rec_state"):
            size = 2 * (hs[0].conv_bytes if attr == "conv_state" else hs[0].rec_bytes)
            assert getattr(hs[0], attr).read(0, size) == getattr(hs[1], attr).read(0, size)
        assert not out[1][active:].any()
    h = hs[1]
    before = (h.conv_state.read(0, 2 * h.conv_bytes), h.rec_state.read(0, 2 * h.rec_bytes))
    assert not h.step(proj, done=True).any()
    assert before == (h.conv_state.read(0, 2 * h.conv_bytes), h.rec_state.read(0, 2 * h.rec_bytes))


@pytest.mark.parametrize("sl,groups,tp", [(4, 16, 4), (2, 32, 6), (4, 32, 8), (2, 16, 3)])
def test_local_column_groups_match_workspace_preparation(dev, sl, groups, tp):
    """Split heads retain exclusive convolution writers and synchronize shared
    preparation between token passes, including partial/empty steps and done."""
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 8, 16, 128, 128, seed=83)
    state = {"l.conv_state": torch.from_numpy(rng.normal(0, .2, (m.conv_dim, CW - 1)).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy(rng.normal(0, .1, (16, 128, 128)).astype(np.float32))}
    hs = [Harness(dev, m, 8, prepared=True, slice_cols=sl, slices_per_block=1, tokens_per_pass=tp,
                  ab_separate=True, perm_out=(8, 64), local_prepare=local, local_groups=groups, specialize=True)
          for local in (False, True)]
    for h in hs:
        h.set_state(state)
    for active, done in [(8, False), (3, False), (1, False), (0, False), (8, True)]:
        proj = _proj(rng, torch, 8, m.in_proj.n)
        np.testing.assert_array_equal(*[h.step(proj, t_active=active, done=done) for h in hs])
        for attr, size in (("conv_state", 2 * hs[0].conv_bytes), ("rec_state", 2 * hs[0].rec_bytes)):
            assert getattr(hs[0], attr).read(0, size) == getattr(hs[1], attr).read(0, size)


@pytest.mark.parametrize("prepared,fused,separate,local", [(False, False, False, False), (True, False, True, False), (True, True, False, False), (True, True, True, True)])
def test_static_geometry_preserves_live_state_and_partial_steps(dev, prepared, fused, separate, local):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 8, 16, 128, 128, seed=91)
    state = {"l.conv_state": torch.from_numpy(rng.normal(0, .2, (m.conv_dim, CW - 1)).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy(rng.normal(0, .1, (16, 128, 128)).astype(np.float32))}
    hs = [Harness(dev, m, 8, prepared=prepared, fused_norm=fused, ab_separate=separate,
                  slice_cols=4, slices_per_block=1, tokens_per_pass=8, specialize=flag, local_prepare=local) for flag in (False, True)]
    for h in hs:
        h.set_state(state)
    for active, done in [(1, False), (8, False), (3, False), (0, False), (8, True)]:
        proj = _proj(rng, torch, 8, m.in_proj.n)
        out = [h.step(proj, t_active=active, done=done) for h in hs]
        np.testing.assert_array_equal(*out)
        for attr in ("conv_state", "rec_state"):
            size = 2 * (hs[0].conv_bytes if attr == "conv_state" else hs[0].rec_bytes)
            assert getattr(hs[0], attr).read(0, size) == getattr(hs[1], attr).read(0, size)


@pytest.mark.parametrize("t,sl,groups", [(1, 4, 32), (4, 4, 16), (6, 2, 32), (8, 4, 32), (8, 2, 32)])
@pytest.mark.parametrize("separate", [False, True])
def test_single_pass_matches_general_recurrence_with_continuation(dev, t, sl, groups, separate):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 8, 16, 128, 128, seed=103)
    state = {"l.conv_state": torch.from_numpy(rng.normal(0, .2, (m.conv_dim, CW - 1)).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy(rng.normal(0, .1, (16, 128, 128)).astype(np.float32))}
    hs = [Harness(dev, m, t, prepared=True, slice_cols=sl, slices_per_block=1, tokens_per_pass=t,
                  ab_separate=separate, perm_out=(8, 64), local_prepare=True, local_groups=groups,
                  specialize=True, fused_norm=t == 1, single_pass=single) for single in (False, True)]
    for h in hs:
        h.set_state(state)
    for active, done in [(t, False), (1, False), (max(0, t - 1), False), (0, False), (t, False), (t, True)]:
        proj = _proj(rng, torch, t, m.in_proj.n)
        np.testing.assert_array_equal(*[h.step(proj, t_active=active, done=done) for h in hs])
        for attr, size in (("conv_state", 2 * hs[0].conv_bytes), ("rec_state", 2 * hs[0].rec_bytes)):
            assert getattr(hs[0], attr).read(0, size) == getattr(hs[1], attr).read(0, size)


@pytest.mark.parametrize('t', [128, 512, 1024])
def test_prefill_register_resident_recurrence(dev, t):
    torch = pytest.importorskip('torch')
    m, rng = _module(64, 8, 16, 128, 128, seed=104)
    hs = [Harness(dev, m, t, prepared=True, slice_cols=4, slices_per_block=1,
                  tokens_per_pass=tp, ab_separate=True, specialize=True) for tp in (8, t)]
    for active, done in [(t, False), (1, False), (t-1, False), (0, False), (t, False), (t, True)]:
        proj = _proj(rng, torch, t, m.in_proj.n)
        np.testing.assert_array_equal(*[h.step(proj, t_active=active, done=done) for h in hs])
        for attr, size in (('conv_state', 2*hs[0].conv_bytes), ('rec_state', 2*hs[0].rec_bytes)):
            assert getattr(hs[0], attr).read(0, size) == getattr(hs[1], attr).read(0, size)


def test_shared_prefill_preparation_preserves_partial_chunks_and_state(dev):
    from monolith.compiler.prefill import shared_gdn_preparation
    from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program
    torch = pytest.importorskip('torch')
    m, rng = _module(64, 2, 6, 128, 128, seed=451)
    hs = [Harness(dev, m, 512, ab_separate=True, prepared=True, slice_cols=4,
                  slices_per_block=1, tokens_per_pass=512) for _ in range(2)]
    h = hs[1]
    import struct
    params = struct.pack('<3I', m.v_heads, m.k_heads, 512)
    op = OpSpec('prep', [(9, 'params', 0)], (1, 1, 1), (32, 1, 1))
    p = Program({'prep': KernelSpec(h.source, 'gdn_prepare', h.macros)},
                {'params': BufferSpec(len(params), params, 'params')}, [op])
    shared_gdn_preparation(p, op)
    k = p.kernels[op.kernel]
    h.pso_prepare = nt.Pipeline(nt.Library(dev, k.source, k.macros), 'gdn_prepare')
    h.prefill_prepare_geometry = (op.grid, op.threadgroup)
    for active in (512, 1, 2, 257, 511):
        proj = _proj(rng, torch, 512, m.value_dim + m.conv_dim + 2*m.v_heads)
        outputs = [h.step(proj, t_active=active) for h in hs]
        np.testing.assert_array_equal(*outputs)
        assert hs[0].prep.read(0, hs[0].prep.nbytes) == hs[1].prep.read(0, hs[1].prep.nbytes)
        for a, b in zip(hs[0].get_state(), hs[1].get_state()):
            np.testing.assert_array_equal(a, b)
