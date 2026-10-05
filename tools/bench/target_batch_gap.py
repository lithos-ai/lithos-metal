"""Paired target-forward timing and per-dispatch profiles for T=1 and 8.

T=1 is plain decode; N=7 uses T=8. With --drafter and --drafter-pack, T=8
replays the actual fixed-N=7 program prefix; otherwise it is a static target
program. Inputs contain exact prefix_ids and batch_ids.
No drafting, sampling, acceptance or prefill is timed.
Profiles use separate encoder timestamps and are for attribution, not pass totals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.generate import load_session
from monolith.runtime import _native as nt, Engine
import monolith.compiler.emit as emit
from monolith.compiler import compile_program

p = argparse.ArgumentParser(description=__doc__)
for name in ('model', 'pack', 'inputs', 'out'):
    p.add_argument('--' + name, required=True)
p.add_argument('--reps', type=int, default=6)
p.add_argument('--max-context', type=int, default=8704)
p.add_argument('--profile', action='store_true')
p.add_argument('--drafter')
p.add_argument('--drafter-pack')
a = p.parse_args()
if a.reps < 1:
    p.error('--reps must be positive')
if bool(a.drafter) != bool(a.drafter_pack):
    p.error('--drafter and --drafter-pack must be specified together')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if (out/'passes.jsonl').exists():
    p.error('--out already contains passes.jsonl; use a fresh directory')
s = load_session(a.model, a.pack, max_context=a.max_context, autotune=True,
                 commute_norm=True, prefill_chunk_size=64, attention='auto')
plain = s.engine(1)
spec = {}
if a.drafter:
    ds = load_session(a.model, a.pack, max_context=a.max_context,
                      drafter_dir=a.drafter, drafter_pack=a.drafter_pack,
                      drafter_kind='lm', drafter_options={'gamma': 7}, prefill_chunk_size=64)
    spec = dict(dynamic_t=True, drafter=ds.drafter, drafter_pack=ds.drafter_pack,
                verify='fixed', verify_length=7)
def target_program():
    return compile_program(s.model, s.pack, s.profile, t=8, layout=s.layout,
                           tuner=s.tuner, attention='auto', commute_norm=True, **spec)
engines = {'new_t8': Engine(target_program(), s.dev, buffers=s.buffers), 'plain_t1': plain}
# Retain the original attention path with every other compiler choice identical.
select = emit._gqa_mma_direct
emit._gqa_mma_direct = lambda *args: False
try:
    prog = target_program()
finally:
    emit._gqa_mma_direct = select
engines['old_t8'] = Engine(prog, s.dev, buffers=s.buffers)
runners = {}
logit_bindings = {}
for label, e in engines.items():
    end = next(i for i, op in enumerate(e.program.ops) if op.name == 'argmax')
    ops = e.ops[:end]
    resources = {name: e.buffers[name] for op in e.program.ops[:end] for _, name, _ in op.bindings}
    icb = nt.Icb(e.dev, ops)
    lay = e.program.layout
    runner = nt.Runner(e.dev, icb, ops, list(resources.values()), e.buffers[e.program.step_state],
                       lay.offset('done'), lay.offset('ring_head'), lay.offset('ring_tail'),
                       e.buffers[e.program.ring], e.program.ring_capacity)
    runners[label] = (icb, runner, end)
    logit_bindings[label] = next(b for b in e.program.ops[end].bindings if b[0] == 0)
    (out / (label + '-ops.json')).write_text(json.dumps([
        dict(name=o.name, kernel=e.program.kernels[o.kernel].function, meta=o.meta,
             grid=o.grid, threadgroup=o.threadgroup) for o in e.program.ops[:end]], indent=2))
records = []
for inp in json.loads(Path(a.inputs).read_text()):
    ctx = len(inp['prefix_ids'])
    s.generate(inp['prefix_ids'], 1)
    hashes = {}
    def prepare(label):
        e = engines[label]
        t = int(label[-1])
        state = e.state()
        state.update(position=ctx, t_this_step=t, pending_tokens=inp['batch_ids'][:t], done=0, stop_at=0)
        e.buffers[e.program.step_state].write(e.program.layout.pack(state), 0)
    for rep in range(a.reps + 1):
        order = list(engines) if rep % 2 == 0 else list(reversed(engines))
        for label in order:
            prepare(label)
            e = engines[label]
            before = e.buffers[e.program.step_state].read(0, e.program.layout.size)
            report = runners[label][1].run(8, 1, 2, False, 0)
            assert not report.error and not report.done
            assert e.buffers[e.program.step_state].read(0, e.program.layout.size) == before
            _, name, offset = logit_bindings[label]
            digest = hashlib.sha256(e.buffers[name].read(offset, int(label[-1])*s.model.config.vocab_size*2)).hexdigest()
            assert hashes.setdefault(label, digest) == digest, 'non-deterministic target replay'
            row = dict(context=ctx, label=label, rep=rep, warmup=rep == 0,
                       gpu_ms=report.gpu_ms/8, wall_ms=report.wall_ms/8, logits_sha256=digest)
            records.append(row)
            with (out/'passes.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
    print('PASS', ctx, {label: statistics.median(r['gpu_ms'] for r in records
          if r['context'] == ctx and r['label'] == label and not r['warmup']) for label in engines}, flush=True)
    if a.profile:
        samples_by_label = {label: [] for label in engines}
        q = nt.Queue(s.dev)
        for rep in range(12):
            order = list(engines) if rep % 2 == 0 else list(reversed(engines))
            for label in order:
                prepare(label)
                samples_by_label[label].append(q.profile(engines[label].ops[:runners[label][2]]))
        for label, samples in samples_by_label.items():
            (out/f'profile-{ctx}-{label}.json').write_text(json.dumps(samples))
        print('PROFILE', ctx, flush=True)
