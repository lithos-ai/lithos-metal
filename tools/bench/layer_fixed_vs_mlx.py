"""Fixed-T decoder-layer stacks against MLX, excluding embeddings, head and sampling.

Each sample replays the same T input rows at the same cache position. Distinct
checkpoint layers stream their distinct weights in a dependency chain; time divided
by the layer count is a direct mean, not a difference between whole-model timings.
Input values are rounded to BF16; MLX retains the checkpoint's activation dtype
and unchanged quantized weights, avoiding mixed FP16/BF16 attention promotion.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import itertools
import platform
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def kv_prefix(layer, ctx, heads, dim):
    rng = np.random.default_rng(2026 + layer)
    from monolith.formats.fp import f32_to_bf16, bf16_to_f32
    return tuple(bf16_to_f32(f32_to_bf16(rng.normal(0, scale, (ctx, heads, dim)).astype(np.float32)))
                 for scale in (.5, .1))


def layer_program(sess, indices, t, x):
    from monolith.core.ir import Graph
    from monolith.core.dtypes import DType
    from monolith.core.shapes import T
    from monolith.nn import LowerContext, state_shape
    from monolith.compiler import emit_program
    from monolith.compiler.passes import DEFAULT_PASSES
    g = Graph('fixed_layer_stack')
    lc = LowerContext(t=T)
    for e in sess.model.state_spec().entries:
        lc.states[e.name] = g.state(e.name, state_shape(e), e.dtype)
    for name, (dtype, arr) in sess.model.tables().items():
        lc.consts[name] = g.const(name, tuple(arr.shape), DType.parse(dtype.lower()))
    h = g.input('hidden', (T, x.shape[-1]), DType.BF16)
    outputs = []
    for i in indices:
        h = sess.model.layers()[i].lower(g, h, lc)
        outputs.append(h.name)
    g.check()
    for p in DEFAULT_PASSES:
        p(g)
    prog = emit_program(g, pack=sess.pack, profile=sess.profile, t=t, tuner=sess.tuner, tail=None, attention=sess.attention,
                        commute_norm=sess.commute_norm, accelerator=sess.accelerator, barriers=sess.barriers)
    return prog, outputs


def initialize_stack(eng, sess, indices, t, ctx, x, random_prefix=False):
    """Restore the same input and cache prefix outside the timed replay."""
    from monolith.formats.fp import f32_to_bf16

    if random_prefix:
        from monolith.packs.transforms import rope_head_perm
        for i in indices:
            mixer = sess.model.layers()[i].mixer
            if not hasattr(mixer, 'kv_heads'):
                continue
            keys, vals = kv_prefix(i, ctx, mixer.kv_heads, mixer.head_dim)
            perm = rope_head_perm(mixer.head_dim, mixer.rotary_dim)
            eng.buffers[mixer.prefix + 'k_cache'].write(f32_to_bf16(keys[..., perm]).tobytes(), 0)
            eng.buffers[mixer.prefix + 'v_cache'].write(f32_to_bf16(vals).tobytes(), 0)
    eng.buffers['hidden'].write(f32_to_bf16(x).tobytes(), 0)
    prog = eng.program
    eng.buffers[prog.step_state].write(prog.layout.pack({'position': ctx, 't_this_step': t}), 0)


def our_stack(sess, indices, t, ctx, x, random_prefix=False, *, buffers=None):
    from monolith.runtime import Engine

    prog, outputs = layer_program(sess, indices, t, x)
    eng = Engine(prog, sess.dev, buffers=buffers, fast_math=sess.fast_math)
    initialize_stack(eng, sess, indices, t, ctx, x, random_prefix)
    return eng, outputs


def mlx_stack(model, indices, t, ctx, x, random_prefix=False):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, ArraysCache
    layers = list(model.layers)
    dtype = layers[indices[0]].input_layernorm.weight.dtype
    inp = mx.array(x[None], dtype=dtype)
    caches, initial = [], []
    for i in indices:
        layer = layers[i]
        if getattr(layer, 'is_linear', False):
            cache = ArraysCache(size=2)
            # MPK replays a fixed StepState.step: its ping-pong input slot
            # remains zero. Reset MLX to the same input state each replay.
            # Materialize the initial convolution and recurrent state outside timing.
            mx.eval(layer(mx.zeros_like(inp), cache=cache), cache[0], cache[1])
            state = [mx.zeros_like(cache[j]) if cache[j] is not None else None for j in range(2)]
            mx.eval(state)
            initial.append(state)
        else:
            cache = KVCache()
            attn = layer.self_attn
            nh = getattr(attn, 'n_kv_heads', None) or attn.num_key_value_heads
            d = getattr(attn, 'head_dim', None) or attn.q_proj.weight.shape[0] // attn.n_heads
            zeros = mx.zeros((1, nh, ctx + t + 256, d), dtype=dtype)
            cache.keys = zeros
            cache.values = mx.zeros_like(zeros)
            if random_prefix:
                keys, vals = kv_prefix(i, ctx, nh, d)
                cache.keys = mx.concatenate([mx.array(keys.transpose(1, 0, 2)[None], dtype=dtype), zeros[:, :, ctx:]], axis=2)
                cache.values = mx.concatenate([mx.array(vals.transpose(1, 0, 2)[None], dtype=dtype), zeros[:, :, ctx:]], axis=2)
            cache.offset = ctx
            mx.eval(cache.keys, cache.values)
            initial.append(None)
        caches.append(cache)
    mx.eval(inp)

    def step():
        h = inp
        for i, cache, state in zip(indices, caches, initial):
            if state is None:
                cache.offset = ctx
            else:
                for slot, value in enumerate(state):
                    cache[slot] = value
            h = layers[i](h, mask=None if t == 1 or state is not None else 'causal', cache=cache)
        return h

    # A recurrent layer's next state is an independent graph output. Evaluate
    # it together with h so fixed-state replays cannot elide the state update.
    step.state_outputs = lambda: [cache[j] for cache, state in zip(caches, initial)
                                  if state is not None for j in range(2) if cache[j] is not None]
    step.activation_dtype = str(dtype)
    def check(outputs):
        previous = inp
        cosines = []
        for i, cache, state, got in zip(indices, caches, initial, outputs):
            if state is None:
                cache.offset = ctx
            else:
                for slot, value in enumerate(state):
                    cache[slot] = value
            ref = layers[i](previous, mask=None if t == 1 or state is not None else 'causal', cache=cache)
            if ref.dtype != dtype:
                raise ValueError(f'MLX layer {i} promoted {dtype} inputs to {ref.dtype}')
            ref = np.asarray(ref.astype(mx.float32)).reshape(-1).astype(np.float64)
            flat = got.reshape(-1).astype(np.float64)
            cosines.append(float(flat @ ref / (np.linalg.norm(flat) * np.linalg.norm(ref))))
            previous = mx.array(got[None], dtype=dtype)
        return min(cosines)
    return step, check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--modelopt', action='store_true',
                    help='use native MLX-LM with the unchanged ModelOpt NVFP4 codes and tensor scales')
    ap.add_argument('--ts', default='1,4,6,8')
    ap.add_argument('--ctx', default='128,1024')
    ap.add_argument('--kind', default='all', choices=['all', 'attention', 'gdn'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--layers', help='comma-separated checkpoint layer indices, within the selected kind')
    ap.add_argument('--individual', action='store_true', help='benchmark each selected layer separately; retain stack runs to check streaming latency')
    ap.add_argument('--kv-prefix', choices=['random', 'zero'], default='random')
    ap.add_argument('--attention', choices=['auto','v1','v2','v3','mma','mma-direct'], default='auto')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--continue-on-oracle-failure', action='store_true', help='record failed numerical checks and finish the sweep; still exit nonzero')
    ap.add_argument('--fail-on-regression', action='store_true', help='exit nonzero if any measured MPK/MLX ratio is >= 1')
    ap.add_argument('--norm-ab', action='store_true', help='pair the fused and original paths with MLX in each repetition')
    ap.add_argument('--min-cosine', type=float, default=.999, help='record a relaxed numerical gate explicitly (0: finite outputs only)')
    ap.add_argument('--commute-norm', action='store_true', help='use relaxed-rounding normalization fusion')
    a = ap.parse_args()
    if not 0 <= a.min_cosine <= 1:
        ap.error('--min-cosine must be between 0 and 1')
    a.commute_norm |= a.norm_ab
    import mlx.core as mx
    from mlx_lm import load
    from monolith.generate import Session
    from tools.bench.layer_vs_mlx import our_model
    from monolith.formats.fp import bf16_to_f32, f32_to_bf16

    ts, ctxs = list(map(int, a.ts.split(','))), list(map(int, a.ctx.split(',')))
    sess = Session(our_model(a.model, None, max(ctxs) + max(ts) + 256), a.pack, eos=-1, attention=a.attention, commute_norm=a.commute_norm)
    if a.modelopt:
        from tools.bench.modelopt_qwen_mlx import load as load_modelopt
        model = load_modelopt(a.model, layer_indices=list(map(int, a.layers.split(','))) if a.layers else None)
    else:
        model, _ = load(a.model)
    indices = [i for i, l in enumerate(model.layers) if a.kind == 'all' or
               (a.kind == 'gdn') == bool(getattr(l, 'is_linear', False))]
    if a.layers:
        requested = list(map(int, a.layers.split(',')))
        if len(set(requested)) != len(requested) or any(i not in indices for i in requested):
            raise ValueError('--layers must contain distinct valid indices within --kind')
        indices = requested
    if a.limit:
        indices = indices[:a.limit]
    if not indices:
        raise ValueError('no layers selected')
    failed = False
    oracle_failed = False
    selections = [[i] for i in indices] if a.individual else [indices]
    for indices, t, ctx in itertools.product(selections, ts, ctxs):
        x = bf16_to_f32(f32_to_bf16(np.random.default_rng(17).normal(0, .1, (t, sess.model.config.hidden_size)).astype(np.float32)))
        eng, outputs = our_stack(sess, indices, t, ctx, x, a.kv_prefix == 'random')
        if sess.tuner is not None:
            sess.tuner.save(sess.dev.info().name)
        step, check = mlx_stack(model, indices, t, ctx, x, a.kv_prefix == 'random')
        baseline = None
        if a.norm_ab:
            sess.commute_norm = False
            baseline, _ = our_stack(sess, indices, t, ctx, x, a.kv_prefix == 'random')
            sess.commute_norm = True
        def ours(engine=eng):
            r = engine.run(a.steps, steps_per_cb=1, in_flight=2)
            return r.wall_ms / a.steps, r.gpu_ms / a.steps
        def mlx():
            pending = []
            start = time.perf_counter()
            for _ in range(a.steps):
                y = step()
                mx.async_eval(y, *step.state_outputs())
                pending.append(y)
                if len(pending) > 1:
                    mx.eval(pending.pop(0))
            mx.eval(*pending, *step.state_outputs())
            return (time.perf_counter() - start) * 1e3 / a.steps
        ours()
        if baseline is not None:
            ours(baseline)
        mlx()
        # Untimed output check: the two references differ in intermediate
        # rounding. Keep the default contract unless explicitly relaxed.
        got = [bf16_to_f32(np.frombuffer(eng.read(output, x.size * 2), dtype=np.uint16)).reshape(x.shape) for output in outputs]
        cosine = check(got)
        oracle_pass = bool(np.isfinite(cosine) and cosine >= a.min_cosine)
        oracle_failed |= not oracle_pass
        if not oracle_pass and not a.continue_on_oracle_failure:
            raise AssertionError(f'output cosine {cosine} < {a.min_cosine}')
        samples = []
        for rep in range(a.reps):
            if baseline is not None:
                runners = {'ours': ours, 'baseline': lambda: ours(baseline), 'mlx': mlx}
                order = list(itertools.permutations(runners))[rep % 6]
                result = {name: runners[name]() for name in order}
                w, g = result['ours']; m = result['mlx']
                samples.append({'ours_wall_ms': w, 'ours_gpu_ms': g, 'mlx_wall_ms': m,
                                'baseline_wall_ms': result['baseline'][0]})
                continue
            if rep % 2:
                m = mlx(); w, g = ours()
            else:
                w, g = ours(); m = mlx()
            samples.append({'ours_wall_ms': w, 'ours_gpu_ms': g, 'mlx_wall_ms': m})
        w, m = min(s['ours_wall_ms'] for s in samples), min(s['mlx_wall_ms'] for s in samples)
        row = dict(model=Path(a.model).name, chip=sess.dev.info().name, date=time.strftime('%Y-%m-%d %H:%M'),
                   metric='fixed_individual_layer' if a.individual else 'fixed_layer_stack', kind=a.kind, layer_indices=indices, T=t, ctx=ctx, steps=a.steps,
                   attention=a.attention, commute_norm=a.commute_norm, pack=str(Path(a.pack).resolve()), checkpoint=str(Path(a.model).resolve()),
                   mlx_version=importlib.metadata.version('mlx'), mlx_lm_version=importlib.metadata.version('mlx-lm'),
                   mlx_weight_adapter='ModelOpt NVFP4 original codes + tensor scales' if a.modelopt else 'stock mlx-lm loader',
                   os=platform.platform(), repetitions=a.reps, prefix=f'{a.kv_prefix} KV; zero recurrent input slot',
                   dtype='bfloat16', mlx_activation_dtype=step.activation_dtype,
                   cosine=cosine, oracle_pass=oracle_pass, oracle_threshold=a.min_cosine, ours_us=w * 1000 / len(indices), mlx_us=m * 1000 / len(indices),
                   ratio=w / m, faster_in_every_pair=all(s["ours_wall_ms"] < s["mlx_wall_ms"] for s in samples), samples=samples)
        if baseline is not None:
            b = min(s['baseline_wall_ms'] for s in samples)
            row.update(baseline_us=b * 1000 / len(indices), fusion_ratio=w / b,
                       fusion_faster_in_every_pair=all(s['ours_wall_ms'] < s['baseline_wall_ms'] for s in samples),
                       fused_norms=sum(eng.program.kernels[o.kernel].macros.get('POST_NORM') == '1' for o in eng.program.ops))
        failed |= row["ratio"] >= 1
        print(json.dumps(row), flush=True)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            with a.out.open('a') as f:
                f.write(json.dumps(row) + '\n')
        del eng, baseline, step, check, ours
        if a.norm_ab:
            del runners
        gc.collect()
    return int(oracle_failed or (a.fail_on_regression and failed))


if __name__ == '__main__':
    raise SystemExit(main())
