#!/usr/bin/env python3
"""Measure one context capacity per process, using existing compatible packs.

Runs nearly-full prompts at temperature zero after a short untimed warmup.
Prefix-cache copies are disabled for timings and can be checked separately.
Use fresh processes for v1/M6 comparisons; --repeats repeats uncached requests.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def allocation_totals(engines):
    """Count shared native allocations once, including superseded arenas."""
    seen, totals = set(), Counter()
    for engine in engines:
        for name, buf in engine.buffers.items():
            if id(buf) in seen:
                continue
            seen.add(id(buf))
            role = engine.program.buffers[name].role
            category = ('target_kv' if name.endswith(('k_cache', 'v_cache')) else
                        'draft_kv' if name.endswith(('k_ctx', 'v_ctx')) else role)
            totals[category] += buf.nbytes
    return dict(total=sum(totals.values()), by_category=dict(totals))


def host_memory():
    result = dict(rss_bytes=int(subprocess.check_output(
        ['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip()) * 1024,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        swap=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip(),
        vm_stat=subprocess.check_output(['vm_stat'], text=True))
    # macOS rusage_info_v4 includes wired/driver-accounted physical footprint,
    # which RSS alone does not describe. Layout: sys/resource.h in the SDK.
    if sys.platform == 'darwin':
        import ctypes
        import plistlib
        class Usage(ctypes.Structure):
            _fields_ = [('uuid', ctypes.c_uint8 * 16), ('values', ctypes.c_uint64 * 35)]
        usage = Usage()
        lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) == 0:
            result.update(physical_footprint_bytes=usage.values[7],
                          peak_physical_footprint_bytes=usage.values[28])
        for name, command in [('thermal', ['pmset', '-g', 'therm']),
                              ('pressure', ['sysctl', 'kern.memorystatus_vm_pressure_level'])]:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=5)
            result[name] = dict(returncode=proc.returncode, output=proc.stdout.strip())
        # System-wide snapshots, NOT process attribution or peak GPU residency.
        # Public timestamp counters cannot supply DRAM traffic or occupancy.
        try:
            proc = subprocess.run(['/usr/sbin/ioreg', '-a', '-r', '-c', 'IOAccelerator'],
                                  capture_output=True, timeout=5, check=True)
            result['gpu_system_snapshots'] = [entry['PerformanceStatistics']
                for entry in plistlib.loads(proc.stdout) if 'PerformanceStatistics' in entry]
        except (subprocess.SubprocessError, plistlib.InvalidFileException):
            result['gpu_system_snapshots'] = None
    return result


def planned_allocations(programs):
    """Mirror Engine's named buffer reuse without allocating GPU memory."""
    shared, allocations = {}, []
    for program in programs:
        for name, spec in program.buffers.items():
            previous = shared.get(name)
            if previous is not None and spec.role != 'params' and previous['nbytes'] >= spec.nbytes:
                continue
            if previous is not None and spec.role in ('state', 'step_state', 'ring', 'weights'):
                raise ValueError(f'shared buffer {name} cannot grow between engines')
            row = dict(name=name, role=spec.role, nbytes=spec.nbytes)
            shared[name] = row
            allocations.append(row)
    return dict(total=sum(row['nbytes'] for row in allocations), buffers=allocations)


def round_statistics(gen):
    """Separate delivered tokens from speculative work and the terminal round.

    The last round may overproduce the requested token budget or stop at EOS.
    Exclude it from the proposal survival estimate; retain every raw log below.
    """
    result = dict(delivered_tokens_per_round=gen.decode_tokens / gen.steps if gen.steps else 0.0)
    if gen.accepted is None:
        return result
    accepted, verified = gen.accepted[:-1], gen.verify_len[:-1]
    result.update(nonterminal_rounds=len(accepted),
                  accepted_histogram=dict(sorted(Counter(gen.accepted).items())),
                  committed_histogram=dict(sorted(Counter(gen.committed).items())),
                  verified_histogram=dict(sorted(Counter(gen.verify_len).items())),
                  nonterminal_mean_accepted=sum(accepted) / len(accepted) if accepted else None,
                  nonterminal_mean_committed=sum(gen.committed[:-1]) / len(accepted) if accepted else None,
                  positions=[])
    for position in range(1, max(verified, default=0) + 1):
        eligible = [(a, v) for a, v in zip(accepted, verified) if v >= position]
        reached = sum(a >= position - 1 for a, _ in eligible)
        matched = sum(a >= position for a, _ in eligible)
        result['positions'].append(dict(position=position, eligible_rounds=len(eligible),
            reached_rounds=reached, accepted_rounds=matched,
            survival=matched / len(eligible), conditional_acceptance=matched / reached if reached else None))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attention-path', choices=['v1','m6'], required=True)
    p.add_argument('--paths', type=Path, required=True, help='JSON: model, pack, draft, draft-pack')
    p.add_argument('--context', type=int, required=True)
    p.add_argument('--kv-cache-dtype', choices=['bf16','q8_0','turbo4'], default='bf16')
    p.add_argument('--warmup-only', action='store_true')
    p.add_argument('--no-draft', action='store_true')
    p.add_argument('--plan-only', action='store_true', help='Compile buffer specifications without GPU allocation')
    p.add_argument('--skip-prefix-checkpoint', action='store_true', help='Skip the separate host-cache memory pass')
    p.add_argument('--validate-prefix', action='store_true', help='Replay a short real-model prefix checkpoint after timing')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--prefill', type=int, default=128)
    p.add_argument('--profile', type=Path, help='Preserve an earlier chip configuration for controlled comparisons')
    p.add_argument('--attention-geometry', type=Path, help='Experimental attention-only launch override JSON')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.tokens < 2 or a.repeats < 1 or a.prefill < 8 or a.context - a.tokens < 128:
        p.error('positive repeat count and at least 128 prompt / 2 output tokens required')
    if os.environ.get('MTL_SHADER_VALIDATION', '0') != '0':
        p.error('shader validation must be tested separately from performance')
    from transformers import AutoTokenizer
    from monolith.backends.metal import config_for_device
    from monolith.backends.metal.config import load_config
    from monolith.generate import load_session
    from monolith.runtime import _native
    from monolith.runtime.prefix_cache import PrefixCache
    from tools.bench.m6_telemetry import VmCounters
    vm = VmCounters()

    paths = json.loads(a.paths.read_text())
    dev = _native.Device()
    info = dev.info()
    profile = load_config(a.profile) if a.profile else config_for_device(info.gpu_cores, info.apple_family, info.name)
    tokenizer = AutoTokenizer.from_pretrained(paths['model'], local_files_only=True)
    question = 'Explain speculative decoding in detail.'
    short = tokenizer.apply_chat_template([dict(role='user', content=question)],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    head = tokenizer.encode('<|im_start|>user\nReference notes:\n', add_special_tokens=False)
    body = tokenizer.encode('Speculative decoding proposes tokens with a small draft model. '
        'A larger target model verifies the proposals in parallel and accepts a matching prefix. '
        'Latency depends on verification cost, acceptance rate, memory bandwidth and context length.\n',
        add_special_tokens=False)
    tail = tokenizer.encode('\nUsing the notes above, ' + question +
        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
    fill = a.context - a.tokens - len(head) - len(tail)
    if fill < 0:
        p.error('context too small for the prompt template')
    long_ids = head + (body * ((fill + len(body) - 1) // len(body)))[:fill] + tail
    opts = dict(profile=profile, max_context=a.context + (0 if a.no_draft else 6),
                autotune=False, measured_choices=True, prefill_chunk_size=a.prefill,
                prefix_cache=False, temperature=0, device=dev, kv_cache_dtype=a.kv_cache_dtype,
                prefill_attention='v1' if a.attention_path == 'v1' else None)
    if not a.no_draft:
        opts.update(drafter_dir=paths['draft'], drafter_pack=paths['draft-pack'],
                    drafter_options={'block_size': 7}, verify='fixed', verify_length=7)
    result = dict(attention_path=a.attention_path, kv_cache_dtype=a.kv_cache_dtype, context=a.context, draft=not a.no_draft, chip=info.name, prefill_chunk=a.prefill,
        profile=profile.fingerprint, recommended_working_set=info.recommended_working_set,
        paths=paths, rows=[], snapshots=[], status='starting', prompt_kind='repeated synthetic reference notes')
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save(status):
        result['status'] = status
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(status, flush=True)

    def snapshot(stage, session=None):
        row = dict(stage=stage, **host_memory())
        if session is not None:
            row['allocations'] = allocation_totals(session.engines.values())
        result['snapshots'].append(row)
        save(stage)

    session = None
    last_progress = time.monotonic()

    def progress():
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress >= 30:
            engine = session._last_engine
            print(json.dumps(dict(stage=result['status'], position=engine.state()['position']
                                  if engine is not None else None)), flush=True)
            last_progress = now
        return False

    try:
        snapshot('before_load')
        session = load_session(paths['model'], paths['pack'], **opts)
        if a.attention_geometry:
            from tools.bench.attention_geometry_tune import variant
            geometry = json.loads(a.attention_geometry.read_text())
            result['attention_geometry'] = geometry
            compile_original = session._compile
            def compile_geometry(*args, **kwargs):
                return variant(compile_original(*args, **kwargs), geometry)
            session._compile = compile_geometry
        if a.plan_only:
            result['plan'] = planned_allocations([
                session._compile(a.prefill, dynamic=True, prefill=True),
                session._compile(1 if a.no_draft else 8, dynamic=not a.no_draft, prefill=False)])
            save('complete')
            return
        session.prepare()
        result['attention_kernels'] = sorted({k.function for prog in session._programs.values() for k in prog.kernels.values() if 'gqa' in k.function})
        result['projection_dispatches'] = [dict(bound=bound, name=op.name, shape=[op.meta.get('n'),op.meta.get('k')],
            grid=op.grid, threadgroup=op.threadgroup, geometry=op.meta.get('geometry'))
            for bound,prog in session._programs.items() for op in prog.ops
            if prog.kernels[op.kernel].function=='gemm_tile' and op.meta.get('format')=='nvfp4'
            and (op.meta.get('n'),op.meta.get('k')) in ((34816,5120),(5120,17408))]
        session.load()
        snapshot('loaded', session)
        session.generate(short, 2)
        snapshot('warmed', session)
        if a.warmup_only:
            save('complete')
            return
        # Capture matched prompt-end logits before speculative decode mutates
        # the state. This CPU read is outside GPU timings (included in wall).
        import numpy as np
        from monolith.formats.fp import bf16_to_f32
        logit_captures = []
        def capture_runner(engine):
            run = engine.run
            def wrapped(*args, **kwargs):
                state = engine.state()
                capture = (state['position'] + state['t_this_step'] == len(long_ids)
                           and state['prefill_left'] == 0 and not state['done'])
                report = run(*args, **kwargs)
                if capture:
                    vocab = session.model.lm_head.vocab
                    offset = (state['t_this_step'] - 1) * vocab * 2
                    data = engine.buffers['logits'].read(offset, vocab * 2)
                    values = bf16_to_f32(np.frombuffer(data, np.uint16))
                    assert np.isfinite(values).all()
                    path = a.out.with_suffix(f'.prompt-logits-{len(logit_captures)}.npy')
                    np.save(path, values)
                    logit_captures.append(str(path))
                return report
            engine.run = wrapped
        for engine in session.engines.values():
            capture_runner(engine)
        result['prompt_logit_captures'] = logit_captures
        for kind, ids in [('full', long_ids)]:
            for repeat in range(a.repeats):
                save(f'generating_{kind}_{repeat}')
                vm_before = vm.read()
                started = time.perf_counter()
                gen = session.generate(ids, a.tokens, cancelled=progress)
                wall_ms = (time.perf_counter() - started) * 1000
                vm_after = vm.read()
                assert len(gen.tokens) == a.tokens
                assert gen.cached_prompt_tokens == 0 and gen.setup_ms == 0
                row = dict(kind=kind, repeat=repeat, timed_vm=vm.delta(vm_before, vm_after), prompt_tokens=len(ids),
                    output_tokens=len(gen.tokens), wall_ms=wall_ms,
                    prefill_gpu_ms=gen.prefill_ms, prefill_wall_ms=gen.prefill_wall_ms,
                    decode_gpu_ms=gen.decode_ms, decode_wall_ms=gen.decode_wall_ms,
                    decode_tokens=gen.decode_tokens, steps=gen.steps,
                    decode_tps=gen.decode_tokens * 1000 / max(gen.decode_wall_ms, 1e-9),
                    e2e_tps=len(gen.tokens) * 1000 / wall_ms,
                    ms_per_round=gen.decode_ms / max(gen.steps, 1),
                    statistics=round_statistics(gen),
                    accepted=gen.accepted, committed=gen.committed, verify_len=gen.verify_len,
                    tokens=list(gen.tokens),
                    setup_ms=gen.setup_ms, cached_prompt_tokens=gen.cached_prompt_tokens,
                    prefill_timings=gen.prefill_timings,
                    prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    token_sha256=hashlib.sha256(json.dumps(gen.tokens).encode()).hexdigest(),
                    text=tokenizer.decode(gen.tokens))
                if repeat:
                    previous = result['rows'][-1]
                    assert row['tokens'] == previous['tokens'], 'Identical requests changed output tokens'
                    assert row['accepted'] == previous['accepted'], 'Identical requests changed speculative acceptance'
                result['rows'].append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ('text', 'statistics', 'prefill_timings', 'tokens')}), flush=True)
                snapshot(f'after_{kind}_{repeat}', session)
        if a.skip_prefix_checkpoint:
            if a.validate_prefix:
                save('validating_prefix_restore')
                entries=list(session.model.state_spec().entries)
                if session.drafter is not None:
                    entries += list(session.drafter.state_entries())
                # Hybrid recurrent state alone is about 294 MiB for this model.
                session.prefix_cache=PrefixCache(entries,max_bytes=512*1024**2)
                # Cross the MMA context threshold after restoring, including
                # full probability tiles and the small original-MMA tail.
                probe=long_ids[:1280]+short
                first=session.generate(probe,32,cache_prefix_tokens=1024,cancelled=progress)
                assert session.prefix_cache.items, 'Checkpoint does not fit probe cache budget'
                second=session.generate(probe,32,cache_prefix_tokens=1024,cancelled=progress)
                assert first.tokens==second.tokens, 'Prefix restore changed output tokens'
                for field in ('accepted','committed','verify_len'):
                    assert getattr(first,field)==getattr(second,field), f'Prefix restore changed {field}'
                assert second.cached_prompt_tokens>=1024, 'Prefix was not restored'
                result['prefix_restore']=dict(prompt_tokens=len(probe),cached_prompt_tokens=second.cached_prompt_tokens,
                    output_tokens=len(second.tokens),identical_tokens=True,identical_speculation=True,
                    cache_bytes=sum(len(item.state)+sum(map(len,item.buffers.values())) for item in session.prefix_cache.items))
                snapshot('after_prefix_restore',session)
            save('complete')
            return
        # Copy a valid prefix after a separate prefill-only generation, avoiding
        # pairing prompt tokens with state that already contains generated tokens.
        entries = list(session.model.state_spec().entries)
        if session.drafter is not None:
            entries += list(session.drafter.state_entries())
        cache = PrefixCache(entries, max_bytes=min(4 * 1024**3, info.recommended_working_set // 8))
        session.prefix_cache = cache
        save('testing_prefix_checkpoint')
        gen = session.generate(long_ids, 2, cancelled=progress)
        result['prefix_cache'] = dict(budget=cache.max_bytes,
            entries=len(cache.items), bytes=sum(len(item.state) + sum(map(len, item.buffers.values())) for item in cache.items),
            output_tokens=len(gen.tokens), prefill_gpu_ms=gen.prefill_ms)
        snapshot('with_prefix_checkpoint', session)
        save('complete')
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc), stage=result['status'])
        save('failed')
        raise
    finally:
        if session is not None:
            session.release_engines()


if __name__ == '__main__':
    main()
