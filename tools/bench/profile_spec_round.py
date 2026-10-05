"""Stage timing probe used in the Qwen3 N=7 serving benchmark audit.

Splits existing ICB dispatches; compare GPU time against the unsplit controls.
Cache reuse is intentionally partial, not a fresh autotuning run.
"""
import sys
import json
from pathlib import Path
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser(description='Profile Qwen3 N=7 stages; GPU access required.')
for name in ('model', 'pack', 'drafter', 'drafter-pack', 'prompts', 'out'):
    p.add_argument('--' + name, required=True)
p.add_argument('--cache', help='Reuse saved choices only; missing shapes retain defaults')
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
from monolith.generate import load_session
from transformers import AutoTokenizer
s = load_session(a.model, a.pack, max_context=4608, autotune=False, commute_norm=True, prefill_chunk_size=64, drafter_dir=a.drafter, drafter_pack=a.drafter_pack, drafter_kind='lm', drafter_options={'gamma': 7}, verify='fixed', verify_length=7)
from monolith.compiler.autotune import Autotuner, Choice
import shutil
cache_new = str(out / 'tuning-cache.json')
if a.cache:
    shutil.copy(a.cache, cache_new)

class CachedOnly(dict):

    def __contains__(self, key):
        if not super().__contains__(key):
            raise RuntimeError('MISSING CACHED CHOICE: ' + key)
        return True

class ReuseTuner(Autotuner):

    def tune_gemv(self, *args, **kwargs):
        try:
            return super().tune_gemv(*args, **kwargs)
        except RuntimeError as ex:
            if not str(ex).startswith('MISSING CACHED CHOICE:'):
                raise
            print(str(ex), flush=True)
            return Choice({'RG': '2', 'RSPLIT': '1u'}, 'crew')

    def tune_gemm(self, *args, **kwargs):
        try:
            return super().tune_gemm(*args, **kwargs)
        except RuntimeError as ex:
            if not str(ex).startswith('MISSING CACHED CHOICE:'):
                raise
            print(str(ex), flush=True)
            return Choice({}, 'crew')
if a.cache:
    s.tuner = ReuseTuner(s.dev, s.dev.info().gpu_cores, cache_new)
    s.tuner.choices = CachedOnly(s.tuner.choices)
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
prompts = json.loads(Path(a.prompts).read_text())
ids = tok.encode(next((p['prompt'] for p in prompts if p['id'] == 'p128-r1')), add_special_tokens=False)

def prepare():
    """Prefill with stop_at=0, leaving a valid first speculative round."""
    s.reset()
    pre = s.prefill_engine(len(ids))
    chunks = [ids[i:i + 64] for i in range(0, len(ids), 64)]
    for i, chunk in enumerate(chunks):
        state = pre.state()
        state.update(t_this_step=len(chunk), pending_tokens=chunk, prefill_left=len(chunks) - 1 - i, stop_at=0)
        pre.buffers[pre.program.step_state].write(pre.program.layout.pack(state), 0)
        pre.run(1, steps_per_cb=1, in_flight=1)
    return s.engine(0)
e = prepare()
ops = [dict(i=i, name=o.name, kernel=o.kernel, meta=o.meta, bindings=o.bindings, macros=e.program.kernels[o.kernel].macros) for i, o in enumerate(e.program.ops)]
print('OPS', len(ops), 'STATE', e.state(), flush=True)
for i, o in enumerate(ops):
    if any((x in o['name'] for x in ['embed', 'lm_head', 'accept', 'verify', 'draft.tokens', 'argmax'])):
        print(i, o['name'], o['meta'], flush=True)
for j in range(3):
    prepare()
    r = e.run(4, steps_per_cb=1, in_flight=1)
    print('BASELINE', j, r, flush=True)
from monolith.runtime import _native as nt
import statistics
# Split unchanged dispatches, buffers and bindings only at stage boundaries.
starts = [i for i, o in enumerate(e.program.ops) if o.name == 'embed']
assert len(starts) == 8
accept = next((i for i, o in enumerate(e.program.ops) if o.name == 'accept_scan'))
head = next((i for i, o in enumerate(e.program.ops) if o.name.startswith('lm_head:')))
spans = [('target_layers', 0, head), ('target_head_sample', head, accept), ('accept', accept, starts[1])]
for c, begin in enumerate(starts[1:]):
    end = starts[c + 2] if c < 6 else len(e.ops) - 1
    h = next((i for i in range(begin, end) if e.program.ops[i].name.startswith('lm_head:')))
    # A separate norm_apply, when present, belongs to the output head.
    h = h - 1 if e.program.ops[h - 1].name == 'norm_apply' else h
    spans.extend([(f'draft{c}_layers', begin, h), (f'draft{c}_head_sample', h, end)])
spans.append(('select', len(e.ops) - 1, len(e.ops)))
resources = {name: e.buffers[name] for o in e.program.ops for _, name, _ in o.bindings}
lay = e.program.layout
runners = []
icbs = []
for name, begin, end in spans:
    ds = e.ops[begin:end]
    icb = nt.Icb(e.dev, ds)
    icbs.append(icb)
    runner = nt.Runner(e.dev, icb, ds, list(resources.values()), e.buffers[e.program.step_state], lay.offset('done'), lay.offset('ring_head'), lay.offset('ring_tail'), e.buffers[e.program.ring], e.program.ring_capacity)
    runners.append((name, runner))
records = []
for rep in range(6):
    for mode in ['full', 'split'] if rep % 2 == 0 else ['split', 'full']:
        prepare()
        if mode == 'full':
            r = e.run(4, steps_per_cb=1, in_flight=2)
            rec = dict(rep=rep, mode=mode, gpu_ms=r.gpu_ms / 4, wall_ms=r.wall_ms / 4, tokens=r.tokens)
            records.append(rec)
            print('CONTROL', rec, flush=True)
        else:
            for step in range(4):
                stage = []
                tokens = []
                for name, runner in runners:
                    r = runner.run(1, 1, 1, False, 0)
                    assert not r.error, r.error
                    tokens.extend(runner.drain())
                    stage.append(dict(name=name, gpu_ms=r.gpu_ms, wall_ms=r.wall_ms))
                rec = dict(rep=rep, mode=mode, step=step, stages=stage, tokens=tokens)
                records.append(rec)
            print('SPLIT', rep, {name: round(statistics.mean((x['gpu_ms'] for r in records if r['rep'] == rep and r['mode'] == 'split' for x in r['stages'] if x['name'] == name)), 4) for name, _, _ in spans}, flush=True)
for rep in range(6):
    full = next((r['tokens'] for r in records if r['rep'] == rep and r['mode'] == 'full'))
    split = [t for r in records if r['rep'] == rep and r['mode'] == 'split' for t in r['tokens']]
    assert full == split, (full, split)
(out / 'round-stage-profile.json').write_text(json.dumps(dict(prompt_id='p128-r1', prompt_tokens=len(ids), spans=spans, records=records), indent=2))
print('PROFILE DONE', flush=True)
generation = []
for p in prompts:
    if p['input_tokens'] != 126:
        continue
    ids_run = tok.encode(p['prompt'], add_special_tokens=False)
    g = s.generate(ids_run, 128)
    r = dict(prompt_id=p['id'], warmup=p['rep'] == 0, decode_wall_ms=g.decode_wall_ms, decode_gpu_ms=g.decode_ms, steps=len(g.accepted), accepted=g.accepted, committed=g.committed, verify_len=g.verify_len, tokens=g.tokens)
    generation.append(r)
    print('GENERATION', p['id'], g.decode_wall_ms / len(g.accepted), 'ms/round', g.decode_wall_ms / g.decode_tokens, 'ms/token', flush=True)
(out / 'round-cached-generation.json').write_text(json.dumps(generation, indent=2))
