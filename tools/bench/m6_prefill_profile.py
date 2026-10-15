#!/usr/bin/env python3
"""Profile real prefill checkpoints without changing the production graph.

Repeated ICB passes restore recurrent/control state and overwrite only the new
KV rows. Timestamp counters insert one encoder per op: their spans and gaps
are diagnostic, never a measurement of production scheduling overhead.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def category(op, kernel):
    f = kernel.function
    if f.startswith('gqa_'):
        if f == 'gqa_pv_merge_m6':
            return 'attention_pv_merge'
        if f == 'gqa_decode_mma_m6' and kernel.macros.get('M6_PV_MERGE') == '1':
            return 'attention_score_partials'
        return 'attention_merge' if f == 'gqa_merge' else ('attention_prepare' if 'prepare' in f else 'attention_core')
    if f.startswith('gdn_'):
        return f
    if f in ('gemm_tile', 'gemv_T'):
        return f + ':' + op.meta.get('format', 'unknown')
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', type=Path, required=True)
    p.add_argument('--profile', type=Path, help='Explicit chip profile for controlled phase comparisons')
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--chunk', type=int, choices=[32, 64, 128], default=128)
    p.add_argument('--positions', type=int, nargs='+', default=[8064, 16256, 32384])
    p.add_argument('--kv-cache-dtype', choices=['bf16', 'q8_0', 'turbo4'], default='turbo4')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--counter-repeats', type=int, default=3)
    p.add_argument('--synthetic', action='store_true', help='Zero historical KV; quick layout/cost probe only')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if min(a.repeats, a.counter_repeats) < 1 or any(x < 0 or x % a.chunk or x + a.chunk >= a.context - 8 for x in a.positions):
        p.error('positive repeats and chunk-aligned intermediate positions required')
    if os.environ.get('MTL_SHADER_VALIDATION', '0') != '0':
        p.error('performance measurements must run without shader validation')
    from monolith.generate import load_session
    from monolith.backends.metal.config import load_config
    from monolith.runtime import _native as nt
    from tools.bench.m6_prefill_model import allocation_totals, host_memory
    paths = json.loads(a.paths.read_text())
    session = load_session(paths['model'], paths['pack'], max_context=a.context + 6,
        **({'profile':load_config(a.profile)} if a.profile else {}),
        autotune=False, measured_choices=True, prefill_chunk_size=a.chunk,
        prefix_cache=False, temperature=0, kv_cache_dtype=a.kv_cache_dtype,
        drafter_dir=paths['draft'], drafter_pack=paths['draft-pack'],
        drafter_options={'block_size': 7}, verify='fixed', verify_length=7)
    session.prepare()
    session.load()
    engine = session.prefill_engine()
    program = engine.program
    result = dict(status='loaded', arguments=vars(a) | {'paths': str(a.paths), 'out': str(a.out),
        'profile':str(a.profile) if a.profile else None},
        chip=engine.dev.info().name, memory=allocation_totals(session.engines.values()),
        host_before=host_memory(), points=[],
        method='real prefix' if not a.synthetic else 'synthetic zero historical KV',
        counter_warning='Encoder boundaries perturb durations. Do not subtract counter sums from ICB time.',
        attention_pipeline_resources={key:dict(function=program.kernels[key].function,
            max_threads=pipeline.max_threads_per_threadgroup,
            static_threadgroup_bytes=getattr(pipeline,'static_threadgroup_memory_length',None),
            simd_width=pipeline.thread_execution_width)
            for key,pipeline in engine.pipelines.items() if program.kernels[key].function.startswith('gqa_')},
        ops=[dict(index=i, name=o.name, function=program.kernels[o.kernel].function,
                  category=category(o, program.kernels[o.kernel]), meta=o.meta,
                  macros=program.kernels[o.kernel].macros, grid=o.grid,
                  threadgroup=o.threadgroup, barrier=o.barrier_before, bindings=o.bindings)
             for i, o in enumerate(program.ops)])
    bound={name for o in program.ops if program.kernels[o.kernel].function.startswith('gqa_')
           for _,name,_ in o.bindings}
    result['attention_bound_buffers']={name:dict(bytes=program.buffers[name].nbytes,role=program.buffers[name].role)
                                      for name in sorted(bound)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.out.write_text(json.dumps(result, indent=2) + '\n')
    save()
    original_run = engine.run
    def measure():
        state = engine.state()
        position = state['position']
        # Historical KV is immutable during a pass. Only [position, position+T)
        # is overwritten, so restoring its old bytes is unnecessary. Recurrent
        # double buffers and control/ring state must be restored in full.
        snapshot = {n: b.read(0, b.nbytes) for n, b in engine.buffers.items()
                    if program.buffers[n].role in ('state', 'step_state', 'ring')
                    and n not in session._kv_buffers}
        def restore():
            for n, value in snapshot.items():
                engine.buffers[n].write(value, 0)
        samples, profiles = [], []
        try:
            # Warm after CPU copies/residency changes; compare the two methods
            # in alternating order rather than claiming counter sums are ICB.
            for rep in range(max(a.repeats, a.counter_repeats) + 1):
                for mode in (['icb', 'counter'] if rep % 2 == 0 else ['counter', 'icb']):
                    if rep > (a.repeats if mode == 'icb' else a.counter_repeats):
                        continue
                    restore()
                    if mode == 'icb':
                        report = original_run(1, steps_per_cb=1, in_flight=1)
                        if rep:
                            samples.append(asdict(report))
                    else:
                        times = nt.Queue(engine.dev).profile(engine.ops)
                        if len(times) != len(program.ops):
                            raise RuntimeError('incomplete GPU counter results')
                        if any(s < 0 or e < s for s, e in times):
                            raise RuntimeError('invalid GPU timestamps')
                        if rep:
                            profiles.append(times)
                    if engine.state()['error']:
                        raise RuntimeError('profiled chunk reported a GPU error')
        finally:
            restore()
        groups = defaultdict(list)
        for times in profiles:
            row = Counter()
            for op, (start, end) in zip(result['ops'], times):
                row[op['category']] += end - start
            for k, v in row.items():
                groups[k].append(v)
        point = dict(position=position, step_state=state, state_snapshot_bytes=sum(map(len, snapshot.values())),
                     samples=samples, counter_timestamps=profiles,
                     gpu_ms=statistics.median(x['gpu_ms'] for x in samples),
                     groups_ms={k: statistics.median(v) for k, v in groups.items()},
                     counter_span_ms=[t[-1][1] - t[0][0] for t in profiles],
                     counter_sum_ms=[sum(e-s for s,e in t) for t in profiles], host=host_memory())
        result['points'].append(point)
        save()
        print(json.dumps({k: point[k] for k in ('position', 'gpu_ms', 'groups_ms')}), flush=True)
    if a.synthetic:
        for pos in a.positions:
            session.reset(preserve_kv=True)
            state = engine.state()
            state.update(position=pos, t_this_step=a.chunk, pending_tokens=[9707]*a.chunk,
                         prefill_left=2, stop_at=0)
            engine.buffers[program.step_state].write(program.layout.pack(state), 0)
            measure()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(paths['model'], local_files_only=True)
        body = tok.encode('Speculative decoding proposes tokens with a small draft model. '
                          'The target verifies the proposals and accepts a matching prefix.\n', add_special_tokens=False)
        count = a.context - 128
        ids = (body * ((count + len(body) - 1)//len(body)))[:count]
        result['prompt_sha256'] = hashlib.sha256(bytes(str(ids), 'utf8')).hexdigest()
        wanted = set(a.positions)
        last_progress = time.monotonic()
        def run(*args, **kwargs):
            nonlocal last_progress
            pos = engine.state()['position']
            if pos in wanted:
                measure()
                wanted.remove(pos)
            if time.monotonic() - last_progress > 30:
                print('prefill position', pos, flush=True)
                last_progress = time.monotonic()
            return original_run(*args, **kwargs)
        engine.run = run
        gen = session.generate(ids, 2)
        if wanted:
            raise RuntimeError(f'checkpoint positions were not visited: {wanted}')
        result['generation_gpu_ms_excluding_probes'] = gen.prefill_ms
        result['output_tokens'] = gen.tokens
    result.update(status='complete', host_after=host_memory())
    save()


if __name__ == '__main__':
    main()
