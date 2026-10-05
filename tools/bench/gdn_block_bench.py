"""Full GDN block at N=7: input norm, all projections, core, output residual.

Seeded FP8/BF16 fixture at the 27B shapes; excludes the separate MLP block.
Compares production, a geometry/operand control, and one static megakernel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.bench import profile_for_device
from monolith.compiler import emit_program
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core import Graph, DType, T
from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.nn import GatedDeltaNet, RMSNorm, Module, LowerContext, state_shape
from monolith.nn.pack_plan import slab_requests, aux_requests, bind_formats, bind_pack_formats
from monolith.packs.packer import Packer, PackFile
from monolith.runtime import Engine, _native as nt
from tools.bench.gdn_block_static import normalize, merge


def fixture(root, hidden=5120, hk=16, hv=48):
    """Only the attention half of a decoder: no embedding/head/MLP weights."""
    m = Module()
    m.mixer = GatedDeltaNet(hidden, hk, hv, 128, 128, 4, 1e-6, hf_prefix='gdn.', prefix='gdn.')
    m.norm = RMSNorm(hidden, 1e-6, 'input_norm.weight', prefix='input_norm.')
    pack = root / 'pack'
    shape = dict(hidden=hidden, hk=hk, hv=hv, seed=27)
    if not (pack / 'manifest.json').exists():
        import torch
        from safetensors.torch import save_file

        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(27)
        weights = {}
        for _, mod in m.named_modules():
            for local, spec in mod.weight_map().items():
                if spec.aux:
                    v = rng.normal(0, .2, spec.shape).astype(np.float32)
                    if local == 'norm_w':
                        v += 1
                    weights[spec.hf_name] = torch.from_numpy(v).to(torch.bfloat16)
                elif local in ('in_proj_a', 'in_proj_b'):
                    v = rng.normal(0, .01, spec.shape).astype(np.float32)
                    weights[spec.hf_name] = torch.from_numpy(v).to(torch.bfloat16)
                else:
                    codes = rng.integers(0, 112, spec.shape, dtype=np.uint8)
                    codes |= rng.integers(0, 2, spec.shape, dtype=np.uint8) << 7
                    weights[spec.hf_name] = torch.from_numpy(codes).view(torch.float8_e4m3fn)
                    weights[spec.hf_name[:-7] + '.weight_scale'] = torch.tensor(.001, dtype=torch.float32)
        save_file(weights, str(root / 'model.safetensors'))
        del weights
        ck = SafetensorsDir(root)
        bind_formats(m, ck)
        ck.close()
        pk = Packer(root, pack)
        for req in slab_requests(m, PackLayout()):
            pk.add_slab(req)
        for req in aux_requests(m):
            pk.add_aux(req)
        pk.write({'fixture': shape})
    pf = PackFile(pack)
    if pf.manifest.get('fixture') != shape:
        raise ValueError('fixture shape/seed differs; choose another --fixture directory')
    ck = SafetensorsDir(root)
    bind_formats(m, ck)
    ck.close()
    bind_pack_formats(m, pf)
    return m, pf


def program(dev, module, pack):
    g = Graph('full_gdn_block')
    ctx = LowerContext(t=T)
    for e in module.mixer.state_entries():
        ctx.states[e.name] = g.state(e.name, state_shape(e), e.dtype)
    h = g.input('hidden', (T, module.mixer.hidden), DType.BF16)
    module.mixer.lower(g, h, module.norm.lower(g, h), ctx)
    for ps in DEFAULT_PASSES:
        ps(g)
    info = dev.info()
    return emit_program(g, pack=pack, profile=profile_for_device(info.gpu_cores, info.apple_family, info.name),
                        t=8, tail=None, commute_norm=True, gdn_mixer_fusion=False)


def initialize(engine, module, seed=9, step=0):
    rng = np.random.default_rng(seed)
    mix = module.mixer
    engine.buffers['hidden'].write(f32_to_bf16(rng.normal(0, .1, (8, mix.hidden)).astype(np.float32)).tobytes(), 0)
    for name, shape, bf in [('gdn.conv_state', (mix.conv_dim, 3), True),
                            ('gdn.rec_state', (mix.v_heads, 128, 128), False)]:
        a = rng.normal(0, .1, shape).astype(np.float32)
        data = (f32_to_bf16(a) if bf else a).tobytes()
        engine.buffers[name].fill(0)
        engine.buffers[name].write(data, (step % 2) * len(data))
    engine.buffers[engine.program.step_state].write(engine.program.layout.pack({'step': step, 't_this_step': 8}), 0)


def checked_run(engine, steps):
    r = engine.run(steps, steps_per_cb=1, in_flight=2)
    if 'mega.flags' in engine.buffers:
        if np.frombuffer(engine.read('mega.flags'), np.uint32)[-1]:
            raise RuntimeError('static worker barrier timed out')
    return r.gpu_ms * 1000 / steps


def snapshot(engine, hidden):
    return (engine.read('gdn.h', 8 * hidden * 2), engine.read('gdn.conv_state'), engine.read('gdn.rec_state'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, default=Path('/tmp/gdn-full-block-fixture'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sgs', type=int, default=16)
    ap.add_argument('--operand-mode', choices=('coop', 'staged'), default='coop')
    ap.add_argument('--workers', type=int)
    ap.add_argument('--tn', type=int, choices=(16, 32), default=16)
    ap.add_argument('--split', action='store_true', help='split each projection across the worker SIMD groups')
    ap.add_argument('--reps', type=int, default=40)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=5120)
    ap.add_argument('--hk', type=int, default=16)
    ap.add_argument('--hv', type=int, default=48)
    a = ap.parse_args()
    dev = nt.Device()
    workers = a.workers or dev.info().gpu_cores
    if not 1 <= workers <= 2*dev.info().gpu_cores or min(a.steps, a.reps) < 1:
        ap.error('workers must be within twice the core count; steps/reps must be positive')
    module, pack = fixture(a.fixture, a.hidden, a.hk, a.hv)
    p = program(dev, module, pack)
    control = normalize(p, a.sgs, mode=a.operand_mode, groups=workers, tn=a.tn, split=a.split)
    fused = merge(control, workers, a.sgs)
    # Share only immutable weights; outputs and state are independent.
    base = Engine(p, dev)
    weights = {n: b for n, b in base.buffers.items() if p.buffers[n].role == 'weights'}
    engines = [base, Engine(control, dev, buffers=weights), Engine(fused, dev, buffers=weights)]
    for e in engines:
        initialize(e, module)
        checked_run(e, 1)
    snaps = [snapshot(e, a.hidden) for e in engines]
    if snaps[1] != snaps[2]:
        raise AssertionError('fused output or state differs from the matched control')
    ref, got = [bf16_to_f32(np.frombuffer(s[0], np.uint16)).astype(np.float64) for s in (snaps[0], snaps[2])]
    cosine = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
    relative_l2 = float(np.linalg.norm(ref - got) / np.linalg.norm(ref))
    if not np.isfinite(got).all() or cosine < .9999 or relative_l2 > .005:
        raise AssertionError((cosine, relative_l2))
    for e in engines:
        warm = 0
        while warm < 30000:
            warm += checked_run(e, a.steps) * a.steps
    rng = np.random.default_rng(19)
    samples = [[] for e in engines]
    for _ in range(a.reps):
        for i in rng.permutation(3):
            samples[i].append(checked_run(engines[i], a.steps))
    rows = []
    for name, ss, e in zip(('production', 'matched_control', 'single_kernel'), samples, engines):
        row = dict(name=name, dispatches=len(e.program.ops), min_us=min(ss), median_us=float(np.median(ss)), samples_us=ss)
        rows.append(row)
        print({k: v for k, v in row.items() if k != 'samples_us'}, flush=True)
    result = dict(chip=dev.info().name, cores=dev.info().gpu_cores, n=7, rows=8, hidden=a.hidden,
                  hk=a.hk, hv=a.hv, workers=workers, sgs=a.sgs, tn=a.tn, split=a.split, operand_mode=a.operand_mode, fixture='synthetic FP8 projections/BF16 scalar gates',
                  steps=a.steps, reps=a.reps, cosine=cosine, relative_l2=relative_l2,
                  control_bit_exact=True, results=rows)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print('cosine', cosine, 'relative_l2', relative_l2, flush=True)


if __name__ == '__main__':
    main()
