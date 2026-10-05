"""Paired BF16 DSpark layer task tuning on a compiled real-checkpoint program.

All engines see identical deterministic block/features and context caches.
A normalized multi-dispatch control must match fusion byte-for-byte; production
must meet the numerical layer gate. Fixed replay must preserve outputs/caches.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.compiler.region_fusion import subprogram
from monolith.formats.fp import f32_to_bf16, bf16_to_f32
from monolith.runtime import Engine, Program
from monolith.runtime import _native as nt
from monolith.spec.dspark.optimization import optimize
from tools.bench.modelopt_layer_bench import metrics
from tools.bench.modelopt_mega_tune import source_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--program', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contexts', default='128,32768')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--steps', type=int, default=5)
    a = ap.parse_args()
    p = Program.load(a.program)
    start = next(i for i, o in enumerate(p.ops) if o.name == 'rmsnorm_stat' and
                 any(n == 'draft.h0' for _, n, _ in o.bindings))
    end = next(i + 1 for i, o in enumerate(p.ops) if o.name == 'gemv:draft.layers.0.mlp.down.down_proj')
    p = subprogram(p, p.ops[start:end])
    dev = nt.Device()
    base = Engine(p, dev)
    shared = {n: b for n, b in base.buffers.items() if p.buffers[n].role == 'weights'}
    rng = np.random.default_rng(141)
    fixture = {n: f32_to_bf16(rng.normal(0, 0.5, p.buffers[n].nbytes // 2).astype(np.float32)).tobytes()
               for n in ('draft.h0', 'draft.feats')}
    output = 'draft.layers.0.mlp.h'
    cache_names = [n for n, spec in p.buffers.items() if spec.role == 'state']
    configs = json.loads(a.configs.read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        row = dict(config=cfg)
        engines = {'production': base}
        try:
            control, fused = optimize(p, cfg)
            engines['control'] = Engine(control, dev, buffers=shared)
            engines['fused'] = Engine(fused, dev, buffers=shared)
            for ctx in map(int, a.contexts.split(',')):
                snapshots = {}
                for name, e in engines.items():
                    for n, b in e.program.buffers.items():
                        if b.role in ('state', 'step_state', 'arena', 'ring'):
                            e.buffers[n].fill(0)
                    for n, data in fixture.items(): e.buffers[n].write(data, 0)
                    st = e.state()
                    st.update(t_this_step=8, n_inject=8, drafter_ctx_len=ctx-8, position=ctx,
                              anchor=1879, gamma=7, verify_len=7)
                    e.buffers[e.program.step_state].write(e.program.layout.pack(st), 0)
                    e.run(1, steps_per_cb=1, in_flight=1)
                    snapshots[name] = e.read(output)
                    assert not e.state()['error'], e.state()
                assert snapshots['control'] == snapshots['fused'], 'control/fusion hidden mismatch'
                for n in cache_names:
                    assert engines['control'].read(n) == engines['fused'].read(n), 'control/fusion cache mismatch: '+n
                check = metrics(bf16_to_f32(np.frombuffer(snapshots['fused'], np.uint16)),
                                bf16_to_f32(np.frombuffer(snapshots['production'], np.uint16)))
                assert check['finite'] and check['cosine'] > .9999 and check['relative_l2'] < .005, check
                samples = {n: [] for n in engines}
                for e in engines.values(): e.run(a.steps, steps_per_cb=a.steps, in_flight=1)
                for _ in range(a.reps):
                    for name in rng.permutation(list(engines)):
                        r = engines[name].run(a.steps, steps_per_cb=a.steps, in_flight=1)
                        samples[name].append(dict(gpu_ms=r.gpu_ms/a.steps, wall_ms=r.wall_ms/a.steps))
                for name, e in engines.items():
                    assert snapshots[name] == e.read(output), 'fixed replay changed output: '+name
                row = dict(config=cfg, ctx=ctx, check=check, control_bit_exact=True, replay_bit_exact=True,
                           samples=samples, best_ms={n:min(s['gpu_ms'] for s in ss) for n,ss in samples.items()},
                           dispatches={n:len(e.ops) for n,e in engines.items()},
                           source_sha256={n:source_digest(e.program) for n,e in engines.items()},
                           chip=dev.info().name, reps=a.reps, steps=a.steps)
                with a.out.open('a') as f: f.write(json.dumps(row)+'\n')
                print(ctx, cfg, row['best_ms'], flush=True)
        except (ValueError, RuntimeError, AssertionError) as exc:
            row['error'] = str(exc)
            with a.out.open('a') as f: f.write(json.dumps(row)+'\n')
            print('REJECTED', cfg, str(exc)[:400], flush=True)
        del engines
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
