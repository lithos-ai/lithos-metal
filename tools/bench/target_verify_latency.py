"""Isolate the N=7 target forward (8 rows), excluding draft and sampling.

Monolith replays the unchanged speculative ICB prefix up to the first argmax.
MLX evaluates the native model on the identical prefix and eight input IDs.
Each replay starts at the same cache offset; prefill and cache rewind are untimed.
Run the engines in separate processes so only one checkpoint set is resident.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine', choices=['monolith', 'mlx'], required=True)
p.add_argument('--model', required=True)
for name in ('pack', 'drafter', 'drafter-pack', 'prompts'):
    p.add_argument('--' + name)
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--contexts', default='4095,8191')
p.add_argument('--reps', type=int, default=20)
a = p.parse_args()
if a.reps < 1:
    p.error('--reps must be positive')
if a.engine == 'monolith' and not all((a.pack, a.drafter, a.drafter_pack, a.prompts)):
    p.error('Monolith needs --pack, --drafter, --drafter-pack and --prompts')
a.out.parent.mkdir(parents=True, exist_ok=True)
contexts = list(map(int, a.contexts.split(',')))


def emit(row):
    row.update(engine=a.engine, target_rows=8, draft_proposals=7,
               scope='embedding + target decoder + final norm + vocabulary projection',
               prefill_chunk_size=64, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with a.out.open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}), flush=True)


if a.engine == 'monolith':
    from monolith.generate import load_session
    from monolith.runtime import _native as nt
    from transformers import AutoTokenizer
    s = load_session(a.model, a.pack, max_context=8704, autotune=True,
                     commute_norm=True, prefill_chunk_size=64, attention='auto',
                     drafter_dir=a.drafter, drafter_pack=a.drafter_pack,
                     drafter_kind='lm', drafter_options={'gamma': 7},
                     verify='fixed', verify_length=7)
    e = s.engine(0)
    end = next(i for i, op in enumerate(e.program.ops) if op.name == 'argmax')
    assert any(op.name.startswith('lm_head:') for op in e.program.ops[:end])
    assert not any(op.name == 'accept_scan' for op in e.program.ops[:end])
    assert sum(op.name == 'embed' for op in e.program.ops[:end]) == 1
    ops = e.ops[:end]
    resources = {name: e.buffers[name] for op in e.program.ops[:end] for _, name, _ in op.bindings}
    icb = nt.Icb(e.dev, ops)
    lay = e.program.layout
    runner = nt.Runner(e.dev, icb, ops, list(resources.values()), e.buffers[e.program.step_state],
                       lay.offset('done'), lay.offset('ring_head'), lay.offset('ring_tail'),
                       e.buffers[e.program.ring], e.program.ring_capacity)
    logit_binding = next(b for b in e.program.ops[end].bindings if b[0] == 0)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    prompts = json.loads(Path(a.prompts).read_text())
    inputs = json.loads(a.inputs.read_text()) if a.inputs.exists() else []
    for ctx in contexts:
        prompt = next(r for r in prompts if r['input_tokens'] == ctx and r['rep'] == 1)
        ids = tokenizer.encode(prompt['prompt'], add_special_tokens=False)
        assert len(ids) == ctx
        s.reset()
        pre = s.prefill_engine(ctx)
        chunks = [ids[i:i+64] for i in range(0, ctx, 64)]
        for i, chunk in enumerate(chunks):
            state = pre.state()
            state.update(t_this_step=len(chunk), pending_tokens=chunk,
                         prefill_left=len(chunks)-1-i, stop_at=0)
            pre.buffers[pre.program.step_state].write(lay.pack(state), 0)
            pre.run(1, steps_per_cb=1, in_flight=1)
        state = e.state()
        assert state['position'] == ctx and state['t_this_step'] == 8 and not state['done'], state
        batch = list(state['pending_tokens'][:8])
        record = dict(context=ctx, prompt_id=prompt['id'], prefix_ids=ids, batch_ids=batch)
        prior = next((x for x in inputs if x['context'] == ctx), None)
        if prior is not None:
            assert prior == record
        else:
            inputs.append(record)
            a.inputs.write_text(json.dumps(inputs) + '\n')
        print('PREFILLED', ctx, 'batch', batch, 'dispatches', end, flush=True)
        wall, gpu, hashes = [], [], []
        before = e.buffers[e.program.step_state].read(0, lay.size)
        for rep in range(a.reps + 5):
            start = time.perf_counter()
            report = runner.run(1, 1, 1, False, 0)
            elapsed = (time.perf_counter()-start)*1000
            assert not report.error and not report.done
            assert e.buffers[e.program.step_state].read(0, lay.size) == before
            if rep >= 5:
                wall.append(elapsed)
                gpu.append(report.gpu_ms)
                # Read all eight logit rows after timing to verify deterministic replay.
                hashes.append(hashlib.sha256(e.buffers[logit_binding[1]].read(logit_binding[2], 8*s.model.config.vocab_size*2)).hexdigest())
        assert len(set(hashes)) == 1
        emit(dict(context=ctx, prompt_id=prompt['id'], median_wall_ms=statistics.median(wall),
                  median_gpu_ms=statistics.median(gpu), wall_ms=wall, gpu_ms=gpu,
                  dispatches=end, logits_sha256=hashes[0]))
else:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    model, tokenizer = load(a.model)
    for ctx in contexts:
        record = next(r for r in json.loads(a.inputs.read_text()) if r['context'] == ctx)
        cache = make_prompt_cache(model)
        ids = record['prefix_ids']
        for i in range(0, ctx, 64):
            model(mx.array(ids[i:i+64])[None], cache=cache)
            mx.eval([c.state for c in cache])
        batch = mx.array(record['batch_ids'])[None]
        mx.eval(batch)
        assert all(c.offset == ctx for c in cache)
        wall, hashes = [], []
        for rep in range(a.reps + 5):
            mx.synchronize()
            start = time.perf_counter()
            logits = model(batch, cache=cache)
            mx.eval(logits)
            mx.synchronize()
            elapsed = (time.perf_counter()-start)*1000
            assert logits.shape == (1, 8, model.args.vocab_size)
            assert all(c.offset == ctx+8 for c in cache)
            if rep >= 5:
                wall.append(elapsed)
                import numpy as np
                hashes.append(hashlib.sha256(np.array(logits.astype(mx.float32)).tobytes()).hexdigest())
            trim_prompt_cache(cache, 8)
            assert all(c.offset == ctx for c in cache)
        assert len(set(hashes)) == 1
        emit(dict(context=ctx, prompt_id=record['prompt_id'], median_wall_ms=statistics.median(wall),
                  wall_ms=wall, logits_sha256=hashes[0],
                  versions={k: importlib.metadata.version(k) for k in ('mlx', 'mlx-lm')}))
