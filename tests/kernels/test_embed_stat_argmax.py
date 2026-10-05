"""embed (raw table and packed slab), rmsnorm_stat and the two-dispatch argmax against numpy."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import pack_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt

K = 1024


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _run(dev, d):
    r = nt.Queue(dev).run(d if isinstance(d, list) else [d])
    assert not r.error, r.error


@pytest.mark.parametrize("packed,lane_order", [(False, None), (True, "interleaved16"), (True, "contiguous")])
def test_embed(dev, packed, lane_order):
    rng = np.random.default_rng(1)
    vocab, t = 100, 5
    table = f32_to_bf16(rng.standard_normal((vocab, K)).astype(np.float32))
    tokens = np.array([0, 99, 17, 42, 42], dtype=np.int32)
    if packed:
        spec = FORMATS.get("bf16").unpack({"weight": table}, shape=(vocab, K))
        data, info, _ = pack_spec(spec, PackLayout(rows=16, lane_order=lane_order))
        tbuf, macros = nt.Buffer(dev, data), kernels.embed_macros(info)
    else:
        tbuf, macros = nt.Buffer(dev, table.tobytes()), kernels.embed_macros()
    pso = nt.Pipeline(nt.Library(dev, kernels.embed_source(), macros), "embed")
    h = nt.Buffer(dev, t * K * 2); h.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, tokens.tobytes())).buffer(1, tbuf).buffer(2, h)
         .bytes(3, kernels.embed_params(K, 4, vocab)).grid(t).threadgroup(32))
    _run(dev, d)
    out = np.frombuffer(h.read(0, t * K * 2), dtype=np.uint16).reshape(t, K)
    assert np.array_equal(out[:4], table[tokens[:4]])
    assert np.all(out[4] == 0)                                             # beyond t_active


def test_rmsnorm_stat(dev):
    rng = np.random.default_rng(2)
    t = 6
    h = f32_to_bf16(rng.standard_normal((t, K)).astype(np.float32) * 3)
    pso = nt.Pipeline(nt.Library(dev, kernels.rmsnorm_stat_source(), {}), "rmsnorm_stat")
    stat = nt.Buffer(dev, t * 4); stat.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, h.tobytes())).buffer(1, stat).bytes(2, kernels.stat_params(K, t - 1))
         .grid(t).threadgroup(32))
    _run(dev, d)
    out = np.frombuffer(stat.read(0, t * 4), dtype=np.float32)
    ref = (bf16_to_f32(h).astype(np.float64) ** 2).sum(-1)
    assert np.allclose(out[:-1], ref[:-1], rtol=1e-5) and out[-1] == 0


def test_norm_apply(dev):
    """The standalone scaling equals the reference norm's BF16 output (r from partial sums, (1 + w) weights)."""
    rng = np.random.default_rng(4)
    t, parts, eps = 3, 5, 1e-6
    h = f32_to_bf16(rng.standard_normal((t, K)).astype(np.float32) * 2)
    hf = bf16_to_f32(h).astype(np.float64)
    nw = (1.0 + rng.standard_normal(K).astype(np.float32) * 0.1).astype(np.float32)
    ssq = (hf ** 2).sum(-1)
    partials = rng.dirichlet(np.ones(parts), size=t) * ssq[:, None]                 # any split of the sum of squares
    r = 1.0 / np.sqrt(partials.sum(-1) / K + eps)
    x_ref = hf * r[:, None] * nw[None, :]
    pso = nt.Pipeline(nt.Library(dev, kernels.norm_apply_source(), {}), "norm_apply")
    x = nt.Buffer(dev, t * K * 2); x.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, h.tobytes())).buffer(1, nt.Buffer(dev, partials.astype(np.float32).tobytes()))
         .buffer(2, nt.Buffer(dev, nw.tobytes())).buffer(3, x).bytes(4, kernels.norm_apply_params(K, t - 1, parts, eps))
         .grid(t).threadgroup(32))
    _run(dev, d)
    out = bf16_to_f32(np.frombuffer(x.read(0, t * K * 2), dtype=np.uint16).reshape(t, K))
    from monolith.bench import bf16_ulp_diff
    assert bf16_ulp_diff(out[:-1], x_ref[:-1]).max() <= 1 and np.all(out[-1] == 0)   # fp32 rsqrt vs numpy: ≤ 1 BF16 ULP at a rounding boundary


def _argmax(dev, logits_f32, t_active):
    t, vocab = logits_f32.shape
    lb = f32_to_bf16(logits_f32)
    lib = nt.Library(dev, kernels.argmax_source(), {})
    p1, p2 = nt.Pipeline(lib, "argmax_partial"), nt.Pipeline(lib, "argmax_final")
    n_sg = 12 * dev.info().gpu_cores
    pv, pi = nt.Buffer(dev, t * n_sg * 4), nt.Buffer(dev, t * n_sg * 4)
    tok = nt.Buffer(dev, t * 4); tok.fill(0xFF)
    params = kernels.argmax_params(vocab, t_active, n_sg)
    d1 = (nt.Dispatch().pipeline(p1).buffer(0, nt.Buffer(dev, lb.tobytes())).buffer(1, pv).buffer(2, pi).bytes(3, params)
          .grid(-(-(n_sg * 32) // 384)).threadgroup(384).barrier())
    d2 = nt.Dispatch().pipeline(p2).buffer(0, pv).buffer(1, pi).buffer(2, tok).bytes(3, params).grid(t).threadgroup(32)
    _run(dev, [d1, d2])
    return np.frombuffer(tok.read(0, t * 4), dtype=np.int32), bf16_to_f32(lb)


@pytest.mark.parametrize("vocab", [248320, 1000, 300])
def test_argmax(dev, vocab):
    rng = np.random.default_rng(vocab)
    t = 4
    logits = rng.standard_normal((t, vocab)).astype(np.float32) * 4
    logits[0, vocab - 1] = 100.0                                          # max at the last index (partial span for 1000/300)
    logits[1, 5] = logits[1, 7] = 50.0                                    # a tie: the lowest index wins
    logits[1, 5] = logits[1, 7]
    logits[2] = -np.abs(logits[2]) - 1                                    # all negative
    tok, lf = _argmax(dev, logits, t_active=3)
    ref = lf.argmax(-1)
    assert tok[:3].tolist() == ref[:3].tolist() and tok[1] == 5 and tok[0] == vocab - 1
    assert tok[3] == -1                                                   # untouched beyond t_active


@pytest.mark.parametrize("fmt,K,placement", [("int4_affine", K, "inline"), ("int8", K, "inline"), ("int4_affine", 3584, "inline"),   # 3584: ragged stripes
                                             ("int4_affine", K, "block"), ("int8", K, "block"), ("nvfp4", 4096, "block"), ("nvfp4", 5120, "block"), ("nvfp4", 12288, "block")])
@pytest.mark.parametrize("scale_order", ["lane", "payload"])
def test_embed_dequantizes_a_quantized_table(dev, fmt, K, placement, scale_order):
    """A gather from a quantized packed slab (an MLX 4-bit embedding tied to the head): every row equals the
    format's dequantization rounded to BF16 — with the scales inline or in the block's region (#101)."""
    from monolith.bench import random_spec
    from monolith.formats.fp import f32_to_bf16 as to_bf16

    rng = np.random.default_rng(3)
    vocab, t = 70, 4
    spec = random_spec(fmt, vocab, K, rng)
    if "weight_scale_2" in spec.params:
        spec.params["weight_scale_2"] = 1.0                                   # a gathered table has no per-tensor scale (MLX's nvfp4 layout)
    data, info, _ = pack_spec(spec, PackLayout(rows=16, scale_placement=placement, scale_order=scale_order))
    ref = to_bf16(FORMATS.get(fmt).dequantize(spec))
    tokens = np.array([0, 69, 17, 42], dtype=np.int32)
    pso = nt.Pipeline(nt.Library(dev, kernels.embed_source(fmt), kernels.embed_macros(info)), "embed")
    h = nt.Buffer(dev, t * K * 2); h.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, tokens.tobytes())).buffer(1, nt.Buffer(dev, data)).buffer(2, h)
         .bytes(3, kernels.embed_params(K, t, vocab)).grid(t).threadgroup(32))
    _run(dev, d)
    out = np.frombuffer(h.read(0, t * K * 2), dtype=np.uint16).reshape(t, K)
    assert np.array_equal(out, ref[tokens])


@pytest.mark.parametrize('placement', ['inline', 'block'])
def test_embed_applies_nonunit_nvfp4_tensor_scales(dev, placement):
    """Pack-time NVFP4 quantization has nonunit scales, including distinct row scales."""
    rng = np.random.default_rng(81)
    vocab, width = 48, 2048
    fmt = FORMATS.get('nvfp4')
    spec = fmt.quantize(rng.normal(0, .08, (vocab, width)).astype(np.float32))
    tensor_scale = spec.params['weight_scale_2']
    assert tensor_scale != 1
    data, info, _ = pack_spec(spec, PackLayout(rows=16, scale_placement=placement))
    scales = np.full(vocab, tensor_scale, dtype=np.float32)
    scales[17] *= 2
    # Recover the unscaled block decode so the expected multiplication order is explicit.
    spec.params['weight_scale_2'] = 1.
    expected = f32_to_bf16(fmt.dequantize(spec) * scales[:, None])
    tokens = np.array([17, 47, 0, -1, vocab], dtype=np.int32)
    macros = kernels.embed_macros(info, row_scales=True)
    pso = nt.Pipeline(nt.Library(dev, kernels.embed_source('nvfp4'), macros), 'embed')
    h = nt.Buffer(dev, (len(tokens)+1)*width*2); h.fill(0xAB)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, tokens.tobytes()))
         .buffer(1, nt.Buffer(dev, data)).buffer(2, h)
         .bytes(3, kernels.embed_params(width, len(tokens), vocab))
         .buffer(4, nt.Buffer(dev, scales.tobytes())).grid(len(tokens)+1).threadgroup(32))
    _run(dev, d)
    out = np.frombuffer(h.read(0, h.nbytes), dtype=np.uint16).reshape(-1,width)
    assert np.array_equal(out[:-1], expected[[17,47,0,0,0]])
    assert np.all(out[-1] == 0xABAB)
