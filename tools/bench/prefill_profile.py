#!/usr/bin/env python3
"""Profile a serving prefill chunk; synthetic cached KV isolates kernel cost.

Whole ICB timings are the latency measurement. Per-dispatch timestamp counters
introduce encoder boundaries and serve only to identify expensive kernels.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--positions', type=int, nargs='+', default=[0, 4096, 16384])
    parser.add_argument('--attention', default='auto', choices=['auto', 'v1', 'v2', 'v3', 'mma'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--counters', action='store_true')
    parser.add_argument('--original', action='store_true', help='Disable prefill compiler tuning and scratch reuse')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from monolith.generate import load_session
    from monolith.runtime import _native as nt

    assets = prepare(parse_args(['--model', args.model, '--local-files-only',
                                '--max-context', str(max(32768, max(args.positions)+args.chunk+8))]))
    _, options = assets.options(max(args.positions) + args.chunk)
    options['prefill_attention'] = args.attention
    session = load_session(str(assets.model_dir), str(assets.pack_dir), **options,
                           prefill_chunk_size=args.chunk, autotune=False, eos=-1,
                           prefill_optimizations=not args.original)
    print('Compiling prefill', args.chunk, args.attention, flush=True)
    engine = session.prefill_engine()
    program = engine.program
    memory = defaultdict(int)
    bound = {name for op in program.ops for _, name, _ in op.bindings}
    for name, spec in program.buffers.items():
        memory[spec.role] += spec.nbytes
        if name not in bound:
            memory['unbound'] += spec.nbytes
    print('Memory GiB:', {k: round(v / 1024**3, 3) for k, v in memory.items()}, flush=True)
    result = dict(model=args.model, backend=program.backend_id, context_capacity=program.context_capacity,
                  chunk=args.chunk, attention=args.attention, memory_bytes=dict(memory),
                  largest_buffers=sorted([(n,s.nbytes,s.role) for n,s in program.buffers.items()],
                                         key=lambda x:x[1], reverse=True)[:20], points=[])
    for position in args.positions:
        def reset():
            session.reset(preserve_kv=True)
            state = engine.state()
            state.update(position=position, t_this_step=args.chunk,
                         pending_tokens=[9707] * args.chunk, prefill_left=2, stop_at=0)
            engine.buffers[program.step_state].write(program.layout.pack(state), 0)
        samples=[]
        for rep in range(args.repeats+1):
            reset()
            start=time.perf_counter()
            report=engine.run(1, steps_per_cb=1, in_flight=1)
            if engine.state()['error']:
                raise RuntimeError('Prefill benchmark exceeded its context capacity or reported a GPU error')
            point=dict(gpu_ms=report.gpu_ms, wall_ms=(time.perf_counter()-start)*1000)
            if rep: samples.append(point)
        point=dict(position=position, samples=samples,
                   gpu_ms=statistics.median(x['gpu_ms'] for x in samples))
        print('Whole chunk:',point,flush=True)
        if args.counters:
            reset()
            times=nt.Queue(engine.dev).profile(engine.ops)
            groups=defaultdict(float)
            rows=[]
            for op,(start,end) in zip(program.ops,times):
                function=program.kernels[op.kernel].function
                ms=end-start
                groups[function]+=ms
                rows.append(dict(name=op.name,function=function,ms=ms,meta=op.meta,
                                 bindings=op.bindings,grid=op.grid,threadgroup=op.threadgroup))
            point.update(counter_groups=dict(sorted(groups.items(),key=lambda x:-x[1])), kernels=rows)
            print('Counter groups:',point['counter_groups'],flush=True)
        result['points'].append(point)
        Path(args.out).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
