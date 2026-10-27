#!/usr/bin/env python3
"""Time the prompt pass's 32-row tiles at one geometry, inside serve's prompt program (kernel choices, #34).

The live tuner times decode shapes only, so a configuration's ``TM32`` choices are measured here: every 32-row tile
the prompt program asks for is compiled at ``--mode`` (``crew``, ``crew2``, ``ksplit<S>`` or ``ksplit<S>nc``; a shape
that geometry does not fit keeps the default), and the pass runs at ``--position`` tokens of cached context with
per-dispatch counters. One line per tile shape: the counter ms summed over the pass's ops of that shape, min over
``--repeats``. Run one process per mode, in alternating order, then ``--choose`` picks per shape (a geometry replaces
the crew only past the tuner's noise margin) and prints the table entries.

    python tools/bench/prefill_tile_sweep.py --mode ksplit4 --model M --draft D --pack ROOT --chunk 512 --out X.jsonl
    python tools/bench/prefill_tile_sweep.py --choose X.jsonl
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODES = ['crew', 'crew2', 'ksplit2', 'ksplit4', 'ksplit8', 'ksplit16', 'ksplit2nc', 'ksplit4nc', 'ksplit8nc', 'ksplit16nc']


def sweep(args):
    from monolith import kernels
    from monolith.compiler import emit
    from monolith.compiler.autotune import Choice, KernelChoices, gemm_key
    from monolith.generate import load_session
    from monolith.runtime import _native as nt
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare

    argv = ['--model', args.model, '--pack', args.pack, '--local-files-only', '--max-context', str(args.max_context)]
    argv += ['--draft', args.draft] if args.draft else ['--no-draft']
    assets = prepare(parse_args(argv))
    _, options = assets.options(args.position + args.chunk)

    class Sweep(KernelChoices):
        """The prompt pass's 32-row tiles at one geometry; everything else at the default."""

        def __init__(self, mode):
            super().__init__({})
            self.mode, self.last, self.fits = mode, None, {}

        def tune_gemv(self, *a, **k):
            return None

        def tune_gemm(self, info, tm, epilogue, *, permute=False, norm_fed=False, stat_parts=1, **_):
            if tm != 32 or permute:
                return None
            self.last = key = gemm_key(info, tm, epilogue)
            s = kernels.gemm_ksplit(self.mode)
            try:
                if s > 1:
                    kernels.gemm_macros(info, tm=tm, out_bf16=True, epilogue=epilogue, ksplit=s,
                                             scale_cache=False if self.mode.endswith('nc') else None)
                    if self.mode.endswith('nc') and not info.scale_bytes:
                        raise ValueError('no scales to cache')
                self.fits[key] = True
            except ValueError:
                self.fits[key] = False
                return None
            return Choice({}, self.mode, False, 1.0, 1.0)

    tuner = Sweep(args.mode)
    add = emit._Ctx.add

    def tagged_add(ctx, kernel, bindings, grid, tg, name, **meta):
        if meta.get('tm') == 32 and meta.get('accelerator') and tuner.last is not None:
            meta['tile_key'] = tuner.last
        return add(ctx, kernel, bindings, grid, tg, name, **meta)

    emit._Ctx.add = tagged_add
    session = load_session(str(assets.model_dir), str(assets.pack_dir), **options, prefill_chunk_size=args.chunk,
                           prefill_exact=True, autotune=False, eos=-1)
    session.tuner = tuner
    engine = session.prefill_engine()
    program = engine.program

    def reset():
        session.reset(preserve_kv=True)
        state = engine.state()
        state.update(position=args.position, t_this_step=args.chunk, pending_tokens=[9707] * args.chunk,
                     prefill_left=2, stop_at=0)
        engine.buffers[program.step_state].write(program.layout.pack(state), 0)

    reset()
    engine.run(1, steps_per_cb=1, in_flight=1)                         # warm
    best = defaultdict(lambda: float('inf'))
    for _ in range(args.repeats):
        reset()
        times = nt.Queue(engine.dev).profile(engine.ops)
        if engine.state()['error']:
            raise RuntimeError('the prompt pass reported a GPU error')
        sums = defaultdict(float)
        for op, (start, end) in zip(program.ops, times):
            if op.meta.get('tile_key'):
                sums[op.meta['tile_key']] += end - start
        for key, ms in sums.items():
            best[key] = min(best[key], ms)
    with open(args.out, 'a') as f:
        for key, ms in sorted(best.items()):
            row = dict(mode=args.mode, key=key, ms=ms, fits=tuner.fits.get(key, True), chunk=args.chunk,
                       position=args.position, repeats=args.repeats)
            f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)


def choose(paths, margin):
    times = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r['fits']:
                times[r['key']][r['mode']].append(r['ms'])
    out = {}
    for key, modes in sorted(times.items()):
        best = {m: min(v) for m, v in modes.items()}
        default = best.get('crew')
        if default is None:
            continue
        mode = min(best, key=best.get)
        if best[mode] > default * (1 - margin):
            mode = 'crew'
        out[key] = {'macros': {}, 'grid_mode': mode, 'ms': round(best[mode], 4), 'default_ms': round(default, 4)}
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--model')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--draft')
    g.add_argument('--no-draft', action='store_true')
    p.add_argument('--pack', help='pack-cache root, as serve --pack')
    p.add_argument('--max-context', type=int, default=32768)
    p.add_argument('--chunk', type=int, default=512)
    p.add_argument('--position', type=int, default=2048)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--out')
    p.add_argument('--choose', nargs='+', metavar='JSONL')
    p.add_argument('--margin', type=float, default=0.03)
    args = p.parse_args()
    if args.choose:
        return choose(args.choose, args.margin)
    if not (args.mode and args.model and args.pack and args.out and (args.draft or args.no_draft)):
        p.error('a sweep needs --mode, --model, --draft or --no-draft, --pack and --out')
    sweep(args)


if __name__ == '__main__':
    main()
