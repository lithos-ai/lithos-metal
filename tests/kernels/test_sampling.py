"""The GPU sampler (sample_hist → sample_select → sample_gumbel → argmax_final) against its numpy reference: exact
token equality (same hash, same thresholds), thresholds equal to the HF warpers' masks on tie-free logits, draws
distributed like the warped softmax over many steps, bit-identical repeat runs, dynamic t_active."""

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.nn import sampling_ref as ref
from monolith.runtime import _native as nt


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


class Sampler:
    def __init__(self, dev, vocab, t_max):
        self.dev, self.vocab, self.t_max = dev, vocab, t_max
        lib = nt.Library(dev, kernels.sample_source(), {})
        self.p = {f: nt.Pipeline(lib, f) for f in ("sample_hist", "sample_select", "sample_gumbel", "argmax_final")}
        self.n_sg = 12 * dev.info().gpu_cores
        hb, tb, pb = kernels.sample_workspace(t_max, self.n_sg)
        self.hist, self.tau = nt.Buffer(dev, hb), nt.Buffer(dev, tb)
        self.hist.fill(0)
        self.pv, self.pi = nt.Buffer(dev, pb), nt.Buffer(dev, pb)

    def run(self, logits_bf16, *, t_active=None, seed=0, step=0, **opts):
        t = logits_bf16.shape[0]
        t_act = t if t_active is None else t_active
        prm = kernels.sample_params(vocab=self.vocab, t_active=t_act, n_sg=self.n_sg, seed=seed, step=step, **opts)
        lb = nt.Buffer(self.dev, logits_bf16.tobytes())
        tok = nt.Buffer(self.dev, t * 4); tok.fill(0xFF)
        grid = (-(-(self.n_sg * 32) // 384), 1, 1)
        ds = [nt.Dispatch().pipeline(self.p["sample_hist"]).buffer(0, lb).buffer(1, self.hist).bytes(3, prm).grid(*grid).threadgroup(384).barrier(),
              nt.Dispatch().pipeline(self.p["sample_select"]).buffer(1, self.hist).buffer(2, self.tau).bytes(3, prm).grid(t).threadgroup(32).barrier(),
              nt.Dispatch().pipeline(self.p["sample_gumbel"]).buffer(0, lb).buffer(2, self.tau).bytes(3, prm).buffer(4, self.pv).buffer(5, self.pi).grid(*grid).threadgroup(384).barrier(),
              nt.Dispatch().pipeline(self.p["argmax_final"]).buffer(0, self.pv).buffer(1, self.pi).buffer(2, tok).bytes(3, prm).grid(t).threadgroup(32)]
        r = nt.Queue(self.dev).run(ds)
        assert not r.error, r.error
        tau = np.frombuffer(self.tau.read(0, t * 4), dtype=np.float32).copy()
        return np.frombuffer(tok.read(0, t * 4), dtype=np.int32).copy(), tau


def _logits(rng, t, vocab, scale=4.0):
    return f32_to_bf16((rng.standard_normal((t, vocab)) * scale).astype(np.float32))


@pytest.mark.parametrize("vocab", [248320, 1000])
@pytest.mark.parametrize("opts", [dict(temperature=1.0), dict(temperature=0.7, top_k=40), dict(temperature=1.0, top_p=0.9),
                                  dict(temperature=1.3, min_p=0.05), dict(temperature=0.8, top_k=50, top_p=0.95, min_p=0.02)],
                         ids=["plain", "topk", "topp", "minp", "all"])
def test_draws_equal_the_reference(dev, vocab, opts):
    rng = np.random.default_rng(vocab + len(opts))
    t = 3
    lb = _logits(rng, t, vocab)
    lf = bf16_to_f32(lb)
    s = Sampler(dev, vocab, t)
    for step in (0, 1, 7):
        tok, tau = s.run(lb, seed=12345, step=step, **opts)
        for i in range(t):
            exp_tau = ref.thresholds(lf[i], **opts)
            assert np.isclose(tau[i], exp_tau, rtol=1e-6, atol=0.0) or (np.isinf(exp_tau) and np.isinf(tau[i])), (i, tau[i], exp_tau)
            assert tok[i] == ref.sample(lf[i], seed=12345, step=step, t=i, **opts), (step, i)


def test_thresholds_match_the_hf_warpers(dev):
    torch = pytest.importorskip("torch")
    from transformers import MinPLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    rng = np.random.default_rng(3)
    vocab, t = 4096, 2
    lb = _logits(rng, t, vocab, scale=3.0)
    lf = bf16_to_f32(lb)
    s = Sampler(dev, vocab, t)
    for opts, warper in [(dict(temperature=1.0, top_k=25), TopKLogitsWarper(25)), (dict(temperature=1.0, top_p=0.8), TopPLogitsWarper(0.8)),
                         (dict(temperature=1.0, min_p=0.1), MinPLogitsWarper(0.1))]:
        _, tau = s.run(lb, seed=1, step=0, **opts)
        for i in range(t):
            hf = warper(None, torch.from_numpy(lf[i:i + 1]).clone())[0].numpy()
            kept_hf = np.isfinite(hf)
            kept_ours = lf[i] >= tau[i]
            assert np.array_equal(kept_hf, kept_ours), (opts, kept_hf.sum(), kept_ours.sum())


def test_distribution_and_determinism(dev):
    rng = np.random.default_rng(11)
    vocab = 512
    lf = np.full(vocab, -60.0, dtype=np.float32)
    lf[:16] = rng.standard_normal(16) * 1.5
    lb = f32_to_bf16(lf[None, :])
    lf = bf16_to_f32(lb)[0]
    s = Sampler(dev, vocab, 1)
    n = 3000
    counts = np.zeros(vocab)
    for step in range(n):
        tok, _ = s.run(lb, seed=99, step=step, temperature=0.9)
        counts[tok[0]] += 1
    p = ref.warped_probabilities(lf, temperature=0.9)
    freq = counts / n
    sigma = np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(freq - p)[:16] <= 4 * sigma[:16] + 1e-9) and counts[16:].sum() == 0, (freq[:16], p[:16])
    a, _ = s.run(lb, seed=99, step=5, temperature=0.9)
    b, _ = s.run(lb, seed=99, step=5, temperature=0.9)
    c, _ = s.run(lb, seed=100, step=5, temperature=0.9)
    assert a[0] == b[0]
    # t_active < T: positions beyond are untouched
    lb4 = _logits(rng, 4, vocab)
    s4 = Sampler(dev, vocab, 4)
    tok, _ = s4.run(lb4, t_active=2, seed=7, step=0, temperature=1.0)
    assert tok[2] == -1 and tok[3] == -1 and all(tok[i] == ref.sample(bf16_to_f32(lb4)[i], seed=7, step=0, t=i) for i in range(2))
