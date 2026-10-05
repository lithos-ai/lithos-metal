"""The relaxed norm path must implement its reordered formula and preserve raw residuals."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import pack_spec, random_spec
from monolith.compiler import emit_program
from monolith.compiler.autotune import Choice
from monolith.core import BlockDomain, DType, Graph, OpClass, Profile, StepStateLayout, T
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.packs import PackFile
from monolith.runtime import Engine


def rb(x):
    return bf16_to_f32(f32_to_bf16(np.asarray(x, np.float32)))


@pytest.mark.parametrize('fmt,k,t', [('bf16', 1024, 8), ('bf16', 1280, 8), ('int4_affine', 1024, 8),
                                   ('nvfp4', 1024, 8), ('nvfp4', 4096, 6), ('nvfp4', 4096, 8)])
@pytest.mark.parametrize('dynamic', [False, True])
def test_commuted_norm_formula_and_partial_rows(tmp_path, fmt, k, t, dynamic):
    rng = np.random.default_rng(7)
    spec = random_spec(fmt, k, k, rng)
    data, info, scales = pack_spec(spec, PackLayout(rows=16, scale_placement='block'))
    gamma = rng.uniform(.5, 1.5, k).astype(np.float32)
    offset = len(data) + scales.nbytes
    packed = data + scales.tobytes() + gamma.tobytes() + (gamma * 2).tobytes()
    packed += bytes(-len(packed) % 16384)
    (tmp_path / 'weights.pack').write_bytes(packed)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(version=2, pack='weights.pack', nbytes=len(packed),
        slabs=[dict(asdict(info), name='w', offset=0, nbytes=len(data), row_scales_offset=len(data))],
        aux=[dict(name=name, offset=offset + i * k * 4, nbytes=k * 4, shape=[k], dtype='f32')
             for i, name in enumerate(('gamma', 'other_gamma'))])))
    g = Graph('commuted_norm')
    x = g.input('x', (T, k), DType.BF16)
    residual = g.input('residual', (T, k), DType.BF16)
    w = g.weight('w', (k, k), fmt)
    y = g.value('y', (T, k), DType.BF16)
    stat = g.value('stat', (T,), DType.F32)
    domain = BlockDomain('rows', info.n_blocks)
    g.op('gemv', [x, w, residual], [y], domain=domain, klass=OpClass.MAP,
         epilogue='residual', stat_value=stat.name)
    for name in ('gamma', 'other_gamma'):
        nw = g.const(name, (k,), DType.F32)
        z = g.value(name + '.z', (T, k // 2), DType.BF16)
        g.op('gemv', [y, w, stat, nw], [z], domain=domain, klass=OpClass.MAP,
             norm=True, eps=1e-6, epilogue='silu_mul')
    profile = Profile.from_dict('test', dict(gpu_cores=20, nominal_gbps=307,
        engine=dict(family='Apple10', lane_order='interleaved16', accelerator='on')))
    kwargs = dict(pack=PackFile(tmp_path), profile=profile, t=t, dynamic_t=dynamic,
                  layout=StepStateLayout(t_max=t, gamma_max=t-1), tail=None, t_min=2)
    if k == 4096:
        kwargs['tuner'] = SimpleNamespace(tune_gemm=lambda *args, **kw: Choice({}, 'ksplit8'),
                                         tune_gemv=lambda *args, **kw: Choice({'RG': '1'}, 'crew'))
    if dynamic:
        fallback = emit_program(g, **dict(kwargs, t_min=1), commute_norm=True)
        assert not any(ks.macros.get('POST_NORM') == '1' for ks in fallback.kernels.values())
    base = emit_program(g, **kwargs, commute_norm=False)
    fused = emit_program(g, **kwargs)
    assert not any(ks.macros.get('POST_NORM') == '1' for ks in base.kernels.values())
    consumers = [op for op in fused.ops if fused.kernels[op.kernel].macros.get('POST_NORM') == '1']
    assert len(consumers) == 1  # a different gamma must not reuse this scratch
    assert len(fused.ops) == len(base.ops) - 1
    assert consumers[0].barrier_before
    producer = next(op for op in fused.ops if fused.kernels[op.kernel].macros.get('NORM_OUT') == '1')
    scratch = next(name for index, name, _ in producer.bindings if index == 14)
    consumer = consumers[0]
    cm = fused.kernels[consumer.kernel].macros
    if k == 4096:
        assert consumer.grid == (2 * profile.gpu_cores, 1, 1)
        assert cm['POST_NORM_ONCE'] == '1'
        # 256 tiles over 40 groups exercises unequal loop lengths and scratch reuse.
    perm = kernels.x_permute_columns(k, FORMATS.get(fmt).weights_per_word, int(cm['TK'].rstrip('u')))
    eng = Engine(fused)
    baseline = Engine(base, eng.dev)
    xdata = f32_to_bf16(rng.normal(0, .01, (t, k)).astype(np.float32))
    rdata = f32_to_bf16(rng.normal(0, .1, (t, k)).astype(np.float32))
    for e in (eng, baseline):
        e.buffers['x'].write(xdata.tobytes(), 0)
        e.buffers['residual'].write(rdata.tobytes(), 0)
    rs = scales.astype(np.float64)[:, None]
    weight = rb(FORMATS.get(fmt).dequantize(spec) / rs).astype(np.float64) * rs
    # Replay decreasing and increasing active counts to catch stale padded rows.
    for active in ([t, 3, 0, 1, t-1] if dynamic else [t]):
        for e in (eng, baseline):
            e.buffers[e.program.step_state].write(e.program.layout.pack(dict(t_this_step=active)), 0)
            e.run(1)
        def read(name, dtype=np.uint16):
            return np.frombuffer(eng.read(name), dtype=dtype)
        assert eng.read('y') == baseline.read('y')
        assert eng.read('stat') == baseline.read('stat')
        raw = read('y').reshape(t, k)[:active]
        scaled = f32_to_bf16(bf16_to_f32(raw) * gamma)
        np.testing.assert_array_equal(read(scratch).reshape(-1, k)[:active], scaled[:, perm])
        if active == 0:
            continue
        parts = read('stat', np.float32).reshape(t, -1)[:active]
        r = 1 / np.sqrt(parts.astype(np.float64).sum(-1) / k + 1e-6)
        dot = bf16_to_f32(scaled).astype(np.float64) @ weight.T * r[:, None]
        # Packed gate/up rows are interleaved in half-block chunks.
        pair = dot.reshape(active, -1, 2, info.rows // 2)
        gate, up = pair[:, :, 0], pair[:, :, 1]
        expected = rb(gate / (1 + np.exp(-gate)) * up).reshape(active, k // 2)
        got = bf16_to_f32(read('gamma.z')).reshape(t, k // 2)[:active]
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, expected, rtol=.015, atol=.002)
