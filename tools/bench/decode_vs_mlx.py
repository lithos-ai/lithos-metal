#!/usr/bin/env python3
"""Paired plain/speculative decoding on identical checkpoints and native chat prompts.

Records every repetition, exact tokens, decode and whole-request wall time. EOS
is disabled in both engines. Run without Metal shader validation, one job at a time.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROMPTS = {
    'code': 'Write a Python function that checks whether a number is prime, with a docstring.',
    'math': 'A train travels 180 km in 2.5 hours. What is its average speed in km/h and in m/s?',
    'chat': 'Explain what a hash map is to a beginner.',
    'text': 'Write a short story about a robot who learns to paint.',
}


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--drafter', required=True)
    ap.add_argument('--drafter-pack', required=True)
    ap.add_argument('--self-draft-control', action='store_true')
    ap.add_argument('--ns', default='1,3,5')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('-n', type=int, default=128)
    ap.add_argument('--max-context', type=int, default=1024)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('MTL_SHADER_VALIDATION', '0') != '0':
        raise RuntimeError('Disable shader validation for timing')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from monolith.generate import load_session

    model, tokenizer = load(a.model)
    draft, draft_tokenizer = load(a.drafter)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('Target and drafter token-ID mappings differ')
    tokenizer.eos_token_ids = []
    prompts = {name: tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
               tokenize=True, add_generation_prompt=True) for name, text in PROMPTS.items()}
    if max(map(len, prompts.values())) + a.n + max(map(int, a.ns.split(','))) > a.max_context:
        raise ValueError('Context capacity is too small')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    plain = {}
    baseline_mode = {}

    def emit(row):
        with a.out.open('a') as f:
            f.write(json.dumps(row) + '\n')

    def mlx_run(ids, gamma):
        mx.synchronize()
        t0 = time.perf_counter()
        tokens, accepted, first = [], 0, None
        kw = dict(draft_model=draft, num_draft_tokens=gamma) if gamma else {}
        for response in stream_generate(model, tokenizer, prompt=ids, max_tokens=a.n, **kw):
            if first is None:
                first = time.perf_counter()
            tokens.append(int(response.token))
            accepted += int(response.from_draft)
        mx.synchronize()
        end = time.perf_counter()
        return dict(tokens=tokens, decode_wall_ms=(end-first)*1000, decode_tokens=len(tokens)-1,
                    request_wall_ms=(end-t0)*1000, accepted_tokens=accepted,
                    # This counts non-draft tokens, not rounds: the final round can be partial.
                    tokens_per_non_draft_token=len(tokens)/max(1, len(tokens)-accepted),
                    reported_generation_tps=response.generation_tps,
                    mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory())

    for gamma in [0] + list(map(int, a.ns.split(','))):
        kw = {} if gamma == 0 else dict(drafter_dir=a.drafter, drafter_pack=a.drafter_pack,
            drafter_kind='lm', drafter_options={'gamma': gamma}, verify='fixed', verify_length=gamma)
        sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1, **kw)
        if gamma == 0:
            emit(dict(kind='metadata', model=a.model, drafter=a.drafter, self_draft_control=a.self_draft_control,
                chip=sess.dev.info().name, recommended_working_set=sess.dev.info().recommended_working_set,
                revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                versions={p: importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy')},
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'), max_context=a.max_context,
                new_tokens=a.n, repetitions=a.reps, draft_lengths=list(map(int,a.ns.split(','))),
                prompts=PROMPTS, prompt_ids=prompts, profile=sess.profile.to_dict() if hasattr(sess.profile,'to_dict') else sess.profile.name,
                eos_disabled=True, temperature=0, shader_validation=False))
        print(f'Warming gamma={gamma}', flush=True)
        # Exclude compilation, tuning and first-touch costs; warm the full measured length.
        sess.generate(next(iter(prompts.values())), a.n)
        mlx_run(next(iter(prompts.values())), gamma)
        allocated = {id(b): b for e in sess.engines.values() for b in e.buffers.values()}
        monolith_bytes = sum(b.nbytes for b in allocated.values())
        del allocated
        for rep in range(a.reps):
            for name, ids in prompts.items():
                for engine in (('monolith','mlx') if rep % 2 == 0 else ('mlx','monolith')):
                    if engine == 'monolith':
                        mx.synchronize()
                        t0 = time.perf_counter()
                        g = sess.generate(ids, a.n)
                        rec = dict(tokens=g.tokens, request_wall_ms=(time.perf_counter()-t0)*1000,
                            decode_wall_ms=g.decode_wall_ms, decode_tokens=g.decode_tokens,
                            gpu_decode_ms=g.decode_ms, tokens_per_step=g.tokens_per_step,
                            accepted_per_step=g.mean_accepted, buffer_bytes=monolith_bytes)
                    else:
                        rec = mlx_run(ids, gamma)
                    assert len(rec['tokens']) == a.n, (engine, gamma, name, len(rec['tokens']))
                    key = (engine, gamma, name)
                    if key in baseline_mode:
                        assert rec['tokens'] == baseline_mode[key], ('non-deterministic replay', key)
                    else:
                        baseline_mode[key] = rec['tokens']
                    if gamma == 0:
                        plain[(engine,name)] = rec['tokens']
                    prefix = common_prefix(rec['tokens'], plain[(engine,name)])
                    rec.update(kind='measurement', engine=engine, model=Path(a.model).name,
                        drafter=Path(a.drafter).name, self_draft_control=a.self_draft_control,
                        gamma=gamma, rep=rep, prompt=name, prompt_tokens=len(ids),
                        ms_per_decode_token=rec['decode_wall_ms']/rec['decode_tokens'],
                        plain_prefix=prefix, equals_plain=prefix == a.n)
                    emit(rec)
                    print(f"{engine:8s} N={gamma} r={rep} {name:4s}: {rec['ms_per_decode_token']:.3f} ms/token; request {rec['request_wall_ms']:.1f} ms; plain prefix {prefix}/{a.n}", flush=True)
        sess.engines.clear()
        del sess
        gc.collect()
        mx.clear_cache()


if __name__ == '__main__':
    main()
