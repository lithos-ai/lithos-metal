"""Paired attention crew search on real checkpoint layer chains.

Explicit benchmark overrides only: context-specific screen winners are not
automatically installed in a serving session. Validate selected configurations
against HF and complete-model timings before using them.
"""
from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith import kernels


def variant(program, config):
    """Copy a program and update attention launch geometry and its parameter ABI.

    v2 counts threadgroups; v1 counts SIMD groups. Fixed-chunk matrix attention
    also counts threadgroups. The merge must see the same updated parameter.
    """
    p = copy.deepcopy(program)
    sgs = config['sgs']
    workers = config.get('workers', 1)
    if not (isinstance(sgs, int) and 1 <= sgs <= 32 and
            isinstance(workers, int) and workers > 0):
        raise ValueError('require 1..32 SIMD groups and positive workers')
    for op in p.ops:
        original = p.kernels[op.kernel]
        fn = original.function
        if fn not in ('gqa_decode', 'gqa_decode_v2', 'gqa_decode_v3',
                      'gqa_decode_mma', 'gqa_merge', 'gqa_merge_v2'):
            continue
        k = original
        if fn == 'gqa_decode_v3':
            if sgs not in (4, 8, 16, 32):
                raise ValueError('v3 supports 4, 8, 16 or 32 SIMD groups')
            k.macros.update(kernels.gqa_v3_macros(int(k.macros['D']), nsg=sgs),
                            SINGLE_BLOCK=str(int(config['single'])))
            op.threadgroup = (32 * sgs, 1, 1)
            continue
        if fn == 'gqa_decode_mma' and k.macros.get('DIRECT_KV') == '1' and sgs != 4:
            raise ValueError('direct-KV matrix attention requires four SIMD groups')
        nsg = workers if fn in ('gqa_decode_v2', 'gqa_merge_v2') or k.macros.get('FIXED_CHUNK') == '1' else workers * sgs
        slot = 4 if fn.startswith('gqa_merge') else 9
        name, offset = next((n, off) for binding, n, off in op.bindings if binding == slot)
        data = bytearray(p.buffers[name].init)
        struct.pack_into('<I', data, offset + 16, nsg)
        p.buffers[name].init = bytes(data)
        k.macros['STATIC_GQA_P_N_SG'] = f'{nsg}u'
        if fn in ('gqa_decode', 'gqa_decode_v2'):
            rows = config['rows']
            if rows not in (1, 2, 4, 8, 16, 32) or rows > int(k.macros.get('RMAX', '32').rstrip('u')):
                raise ValueError('row group exceeds the compiled query geometry')
            k.macros['RBMAX' if fn == 'gqa_decode' else 'RG'] = f'{rows}u'
        if fn in ('gqa_decode_v2', 'gqa_merge_v2'):
            k.macros['NSG'] = f'{sgs}u'
        if fn == 'gqa_decode_mma':
            k.macros['MMA_SG'] = str(sgs)
        if not fn.startswith('gqa_merge'):
            op.grid, op.threadgroup = (workers, 1, 1), (32 * sgs, 1, 1)
    return p


def configurations(kernel, extended=False):
    fn = kernel.function
    workers = (20, 40, 80, 160, 320, 640, 1280) if extended else (20, 40, 80, 160, 320)
    if fn == 'gqa_decode_v3':
        return [dict(sgs=s, single=b) for s, b in itertools.product((4, 8, 16, 32), (False, True))]
    if fn == 'gqa_decode_mma':
        crews = (4,) if kernel.macros.get('DIRECT_KV') == '1' else ((1, 2, 4, 8, 16, 32) if extended else (1, 2, 4, 8, 16))
        return [dict(workers=w, sgs=s) for w, s in itertools.product(workers, crews)]
    rows = (1, 2, 4, 8, 16, 32) if extended else (1, 2, 4, 8)
    crews = (4, 8, 12, 16, 24, 32) if extended else (4, 8, 12, 16)
    return [dict(workers=w, sgs=s, rows=r) for w, s, r in itertools.product(workers, crews, rows)
            if r <= int(kernel.macros.get('RMAX', '16').rstrip('u'))]


def main():
    from monolith.generate import load_session
    from monolith.runtime import Engine
    from monolith.formats.fp import bf16_to_f32, f32_to_bf16
    from tools.bench.layer_fixed_vs_mlx import our_stack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pack', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--configs', type=Path, help='list of {T, context, config} compiler choices')
    parser.add_argument('--geometries', type=Path, help='explicit list of launch configurations instead of the built-in grid')
    parser.add_argument('--layers', help='comma-separated checkpoint layer indices')
    parser.add_argument('--ts', default='1,4,8')
    parser.add_argument('--contexts', default='128,4096,8192,16384,32768')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    contexts, ts = list(map(int, args.contexts.split(','))), list(map(int, args.ts.split(',')))
    if min(args.reps, args.steps, *ts) < 1 or min(contexts) < 0:
        parser.error('require positive repetitions, steps and T; nonnegative contexts')
    session = load_session(args.model, args.pack, max_context=max(contexts) + 256, eos=-1)
    profile = copy.deepcopy(session.profile)
    tables = session.model.tables()
    session.model.tables = lambda: tables
    indices = list(map(int, args.layers.split(','))) if args.layers else sorted({0, session.model.n_layers // 2, session.model.n_layers - 1})
    choices = json.loads(args.configs.read_text()) if args.configs else []
    explicit_geometry = json.loads(args.geometries.read_text()) if args.geometries else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for t, ctx in itertools.product(ts, contexts):
        cfg = next((r['config'] for r in choices if r['T'] == t and r['context'] == ctx), {})
        session.profile = copy.deepcopy(profile)
        session.attention = cfg.get('attention', 'auto')
        session.accelerator = cfg.get('accelerator', 'on')
        session.commute_norm = cfg.get('commute_norm', True)
        for name in ('threadgroups_per_core', 'sibling_order'):
            if name in cfg:
                setattr(session.profile, name, cfg[name])
        x = bf16_to_f32(f32_to_bf16(np.random.default_rng(17).normal(0, .1, (t, session.model.config.hidden_size)).astype(np.float32)))
        base, outputs = our_stack(session, indices, t, ctx, x, True)
        def timing(e):
            return e.run(args.steps, steps_per_cb=1, in_flight=2).gpu_ms / args.steps
        def read(e):
            return [bf16_to_f32(np.frombuffer(e.read(o, x.size * 2), np.uint16)).astype(np.float64) for o in outputs]
        timing(base)
        expected = read(base)
        core = next(k for k in base.program.kernels.values() if k.function.startswith('gqa_decode'))
        for geometry in explicit_geometry if explicit_geometry is not None else configurations(core, args.extended):
            candidate = None
            row = dict(model=args.model, T=t, context=ctx, layers=indices, compiler_config=cfg, function=core.function, geometry=geometry)
            try:
                candidate = Engine(variant(base.program, geometry), session.dev, buffers=base.buffers)
                timing(candidate)
                got = read(candidate)
                cosine = min(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(got, expected))
                row.update(cosine=cosine, exact=all(np.array_equal(a, b) for a, b in zip(got, expected)))
                if not cosine > .999:
                    raise ValueError('output cosine below .999')
                samples = []
                for rep in range(args.reps):
                    order = (base, candidate) if rep % 2 else (candidate, base)
                    values = {id(e): timing(e) for e in order}
                    samples.append(dict(candidate=values[id(candidate)], control=values[id(base)]))
                row.update(samples=samples, ratio=min(r['candidate'] for r in samples) / min(r['control'] for r in samples))
            except (ValueError, RuntimeError, AssertionError) as exc:
                row['rejected'] = str(exc)
            with args.out.open('a') as f:
                f.write(json.dumps(row) + '\n')
            del candidate
        print('POINT', t, ctx, core.function, flush=True)
        del base
        gc.collect()


if __name__ == '__main__':
    main()
