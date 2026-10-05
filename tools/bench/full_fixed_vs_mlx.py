"""Fixed-position full target forwards, including embeddings and vocabulary head.

Uses canonical random BF16 KV prefixes and zero recurrent input state, so this
measures kernel cost rather than prompt quality or prefill latency. Run engines
in separate processes. Sampling, draft, acceptance and serving are excluded.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from tools.bench.layer_fixed_vs_mlx import kv_prefix


INPUT_IDS = [785, 6722, 315, 9625, 374, 12095, 13, 576]


def native_points(args, contexts, ts):
    from monolith.generate import load_session
    from monolith.packs.transforms import rope_head_perm
    from monolith.runtime import Engine

    session = load_session(args.model, args.pack, max_context=max(contexts) + 256, eos=-1,
                           autotune=not args.no_autotune)
    profile = copy.deepcopy(session.profile)
    overrides = json.loads(args.configs.read_text()) if args.configs else []
    for t in ts:
        for ctx in contexts:
            choice = next((r for r in overrides if r['T'] == t and r['context'] == ctx), {})
            cfg = choice.get('config', {})
            session.profile = copy.deepcopy(profile)
            session.attention = cfg.get('attention', 'auto')
            session.accelerator = cfg.get('accelerator', 'on')
            session.commute_norm = cfg.get('commute_norm', True)
            for key in ('threadgroups_per_core', 'sibling_order'):
                if key in cfg:
                    setattr(session.profile, key, cfg[key])
            program = session._compile(t, dynamic=False, prefill=False)
            end = next(i for i, op in enumerate(program.ops) if op.name == 'argmax')
            program.ops = program.ops[:end]
            used = {op.kernel for op in program.ops}
            program.kernels = {k: v for k, v in program.kernels.items() if k in used}
            if choice.get('attention_geometry'):
                from tools.bench.attention_geometry_tune import variant
                program = variant(program, choice['attention_geometry'])
            engine = Engine(program, session.dev)
            for i, layer in enumerate(session.model.layers()):
                mixer = layer.mixer
                if not hasattr(mixer, 'kv_heads'):
                    continue
                keys, values = kv_prefix(i, ctx, mixer.kv_heads, mixer.head_dim)
                perm = rope_head_perm(mixer.head_dim, mixer.rotary_dim)
                engine.buffers[mixer.prefix + 'k_cache'].write(f32_to_bf16(keys[..., perm]).tobytes(), 0)
                engine.buffers[mixer.prefix + 'v_cache'].write(f32_to_bf16(values).tobytes(), 0)
            state = program.layout.pack(dict(position=ctx, t_this_step=t, pending_tokens=INPUT_IDS[:t]))
            engine.buffers[program.step_state].write(state, 0)

            def run():
                start = time.perf_counter()
                report = engine.run(1, steps_per_cb=1, in_flight=1)
                return dict(wall_ms=(time.perf_counter() - start) * 1000, gpu_ms=report.gpu_ms)

            read = lambda: engine.read('logits', t * session.model.config.vocab_size * 2)
            run()
            logits = read()
            run()
            assert logits == read(), 'fixed replay changed logits'
            assert np.isfinite(bf16_to_f32(np.frombuffer(logits, np.uint16))).all()
            warm(run)
            samples = [run() for _ in range(args.reps)]
            assert logits == read(), 'timed replay changed logits'
            assert engine.buffers[program.step_state].read(0, len(state)) == state
            yield dict(T=t, context=ctx, config=cfg, attention_geometry=choice.get('attention_geometry'),
                       samples=samples, dispatches=len(program.ops),
                       logits_sha256=hashlib.sha256(logits).hexdigest(), chip=session.dev.info().name,
                       gpu_cores=session.dev.info().gpu_cores)
            del engine, program
            gc.collect()


def mlx_points(args, contexts, ts):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import ArraysCache, make_prompt_cache

    if args.modelopt:
        from tools.bench.modelopt_qwen_mlx import load as load_modelopt
        model = load_modelopt(args.model)
    else:
        model = load(args.model)[0]
    layers = list(model.layers)
    for t in ts:
        inp = mx.array(INPUT_IDS[:t])[None]
        mx.eval(inp)
        for ctx in contexts:
            caches = make_prompt_cache(model)
            mx.eval(model(mx.array([[INPUT_IDS[0]]]), cache=caches))
            initial = []
            for i, (layer, cache) in enumerate(zip(layers, caches)):
                if isinstance(cache, ArraysCache):
                    state = [mx.zeros_like(cache[j]) if cache[j] is not None else None for j in range(2)]
                    mx.eval(state)
                    initial.append(state)
                else:
                    attn = layer.self_attn
                    heads = getattr(attn, 'n_kv_heads', None) or attn.num_key_value_heads
                    dim = getattr(attn, 'head_dim', None)
                    if dim is None:
                        dim = attn.q_proj.weight.shape[0] // attn.n_heads
                    keys, values = kv_prefix(i, ctx, heads, dim)
                    # Preserve the stock loader's activation dtype. Mixing a
                    # BF16 cache with an FP16 query promotes attention to FP32.
                    # Prefix values are already rounded to BF16 by kv_prefix.
                    dtype = cache.keys.dtype
                    tail = mx.zeros((1, heads, t + 256, dim), dtype=dtype)
                    cache.keys = mx.concatenate([mx.array(keys.transpose(1, 0, 2)[None], dtype=dtype), tail], axis=2)
                    cache.values = mx.concatenate([mx.array(values.transpose(1, 0, 2)[None], dtype=dtype), tail], axis=2)
                    cache.offset = ctx
                    mx.eval(cache.keys, cache.values)
                    initial.append(None)

            def run():
                for cache, state in zip(caches, initial):
                    if state is None:
                        cache.offset = ctx
                    else:
                        for j, value in enumerate(state):
                            cache[j] = value
                start = time.perf_counter()
                logits = model(inp, cache=caches)
                mx.eval(logits, *[cache[j] for cache, state in zip(caches, initial)
                                  if state is not None for j in range(2) if cache[j] is not None])
                return dict(wall_ms=(time.perf_counter() - start) * 1000), logits

            _, logits = run()
            first = np.array(logits.astype(mx.float32))
            _, logits = run()
            assert np.array_equal(first, np.array(logits.astype(mx.float32)))
            assert np.isfinite(first).all()
            warm(run)
            samples = [run()[0] for _ in range(args.reps)]
            _, logits = run()
            assert np.array_equal(first, np.array(logits.astype(mx.float32)))
            yield dict(T=t, context=ctx, samples=samples,
                       logits_dtype=str(logits.dtype),
                       kv_dtypes=sorted({str(cache.keys.dtype) for cache, state in zip(caches, initial) if state is None}),
                       logits_sha256=hashlib.sha256(first.tobytes()).hexdigest(),
                       versions={k: importlib.metadata.version(k) for k in ('mlx', 'mlx-lm')})
            del caches, initial, logits
            gc.collect()
            mx.clear_cache()


def warm(run):
    start = time.monotonic()
    while time.monotonic() - start < .5:
        run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pack')
    parser.add_argument('--engine', choices=('monolith', 'mlx'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modelopt', action='store_true', help='MLX adapter for unchanged ModelOpt NVFP4 weights')
    parser.add_argument('--configs', type=Path,
                        help='list of {T, context, config, attention_geometry?} overrides')
    parser.add_argument('--contexts', default='128,4096,8192,16384,32768')
    parser.add_argument('--ts', default='1,8')
    parser.add_argument('--reps', type=int, default=15)
    parser.add_argument('--no-autotune', action='store_true',
                        help='use default kernel geometry, as the serving CLI does')
    args = parser.parse_args()
    contexts, ts = [int(x) for x in args.contexts.split(',')], [int(x) for x in args.ts.split(',')]
    if args.reps < 1 or min(contexts) < 0 or not all(1 <= t <= len(INPUT_IDS) for t in ts):
        parser.error('require positive repetitions, nonnegative contexts and 1 <= T <= 8')
    if args.engine == 'monolith' and not args.pack:
        parser.error('Monolith requires --pack')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    points = native_points if args.engine == 'monolith' else mlx_points
    for row in points(args, contexts, ts):
        row.update(engine=args.engine, model=args.model, pack=args.pack,
                   modelopt=args.modelopt, scope='embedding + decoder + norm + vocabulary head; fixed random BF16 KV',
                   autotune=not args.no_autotune if args.engine == 'monolith' else None,
                   min_wall_ms=min(s['wall_ms'] for s in row['samples']))
        with args.out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    main()
