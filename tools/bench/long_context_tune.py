"""Qwen3 8B long-context tuning and fresh-generation validation.

The screen compares attention algorithms/geometries using identical real prefills.
Before each replay it restores small mutable buffers and KV tails from prefix-16
onward; earlier KV rows are read-only. Large temporary arenas are overwritten by
the graph. Warmup is excluded and candidate order alternates. Geometry edits use
the existing layer_grid_search helper to keep parameter bytes/macros consistent.
Use --generate --autotune to validate with normal Session.generate and fresh KV.
Only one GPU benchmark process should run at a time.
"""
import sys, json, argparse, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.generate import load_session
from monolith.runtime import Engine
from monolith.compiler import compile_program
import monolith.compiler.emit as emit
from transformers import AutoTokenizer
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['plain', 'n7'], required=True)
p.add_argument('--geometry', action='store_true')
p.add_argument('--generate', action='store_true', help='Run fresh 128-token generations instead of the screen')
p.add_argument('--autotune', action='store_true', help='Enable tuning for --generate; the screen always compares both')
for name in ('model', 'pack', 'prompts', 'outdir'):
    p.add_argument('--' + name, required=True)
for name in ('drafter', 'drafter-pack'):
    p.add_argument('--' + name)
p.add_argument('--max-context', type=int, default=8704)
a = p.parse_args()
if a.mode == 'n7' and (not (a.drafter and a.drafter_pack)):
    p.error('N=7 needs both drafter paths')
root = Path(a.outdir)
root.mkdir(parents=True, exist_ok=True)
spec = a.mode == 'n7'
s = load_session(a.model, a.pack, max_context=a.max_context, autotune=a.autotune if a.generate else True, commute_norm=True, prefill_chunk_size=64, drafter_dir=a.drafter if spec else None, drafter_pack=a.drafter_pack if spec else None, drafter_kind='lm', drafter_options={'gamma': 7}, verify='fixed' if spec else 'cost', verify_length=7 if spec else None, attention='auto')
if a.generate:
    s.engine(0 if spec else 1)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    for prompt in json.loads(Path(a.prompts).read_text()):
        ids = tok.encode(prompt['prompt'], add_special_tokens=False)
        start = time.monotonic()
        g = s.generate(ids, 128)
        assert len(g.tokens) == 128 and g.decode_tokens == 127
        if spec:
            assert len(g.accepted) > 0 and set(g.verify_len) == {7}
        row = dict(mode=a.mode, tuned=a.autotune, attention='auto', prompt_id=prompt['id'], context=len(ids), warmup=prompt['rep'] == 0, elapsed_s=time.monotonic() - start, decode_wall_ms=g.decode_wall_ms, decode_gpu_ms=g.decode_ms, decode_tokens=g.decode_tokens, steps=g.steps, ms_per_step=g.decode_wall_ms / g.steps, ms_per_token=g.decode_wall_ms / g.decode_tokens, accepted=g.accepted, committed=g.committed, verify_len=g.verify_len, tokens=g.tokens)
        with open(root / 'generation.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print('GENERATION', prompt['id'], a.mode, a.autotune, round(row['ms_per_step'], 3), round(row['ms_per_token'], 3), 'elapsed', round(row['elapsed_s'], 1), flush=True)
    raise SystemExit(0)
original = emit._gqa_kernel
active = {}

def select(ctx, heads, kv, lm_mode=0, t_c=None, d=128, qk_norm=True):
    prior = ctx.attention
    ctx.attention = active['draft' if lm_mode else 'target']
    try:
        return original(ctx, heads, kv, lm_mode, t_c, d, qk_norm)
    finally:
        ctx.attention = prior
configs = [('off-auto', False, 'auto', 'auto'), ('tuned-auto', True, 'auto', 'auto'), ('tuned-mma', True, 'mma', 'mma'), ('tuned-v1', True, 'v1', 'v1')]
configs += [('tuned-draft-v1', True, 'auto', 'v1')] if spec else []
engines = {}
for (label, tune, target, draft) in configs:
    active.update(target=target, draft=draft)
    emit._gqa_kernel = select
    try:
        prog = compile_program(s.model, s.pack, s.profile, t=s.decode_t_max if spec else 1, dynamic_t=spec, eos=s.eos, ring_capacity=s.ring_capacity, layout=s.layout, tuner=s.tuner if tune else None, drafter=s.drafter, drafter_pack=s.drafter_pack, verify=s.verify, verify_length=s.verify_length, commute_norm=True)
    finally:
        emit._gqa_kernel = original
    engine = Engine(prog, s.dev, buffers=s.buffers)
    if s.buffers is None:
        s.buffers = dict(engine.buffers)
    else:
        s.buffers.update(engine.buffers)
    s.engines[label] = engine
    engines[label] = engine
    s.tuner.save(s.dev.info().name)
    print('COMPILED', label, len(engine.ops), flush=True)
from tools.bench.layer_grid_search import variant
base = engines['tuned-auto']
if a.geometry:
    keep = {'off-auto': engines['off-auto'], 'tuned-auto': base}
    for (g, sg) in ((20, 4), (40, 4), (80, 4), (160, 4), (40, 8), (160, 8), (80, 16)):
        if not spec:
            break
        label = f'mma-g{g}-sg{sg}'
        prog = variant(base.program, 'gqa_decode', dict(groups=g, sgs=sg))
        eng = Engine(prog, s.dev, buffers=s.buffers)
        keep[label] = eng
        s.engines[label] = eng
    engines = keep
for nsg in (16, 8):
    label = f'v3-sg{nsg}'
    prog = copy.deepcopy(base.program)
    for op in prog.ops:
        k = prog.kernels[op.kernel]
        if k.function == 'gqa_decode_v3':
            k.macros['NSG3'] = str(nsg) + 'u'
            op.threadgroup = (nsg * 32, 1, 1)
    eng = Engine(prog, s.dev, buffers=s.buffers)
    engines[label] = eng
    s.engines[label] = eng
    print('COMPILED', label, flush=True)
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
prompts = json.loads(Path(a.prompts).read_text())
records = []
for prompt in [r for r in prompts if r['rep'] == 1]:
    ids = tok.encode(prompt['prompt'], add_special_tokens=False)
    s.reset()
    pre = s.prefill_engine(len(ids))
    chunks = [ids[i:i + 64] for i in range(0, len(ids), 64)]
    for (i, chunk) in enumerate(chunks):
        state = pre.state()
        state.update(t_this_step=len(chunk), pending_tokens=chunk, prefill_left=len(chunks) - i - 1, stop_at=0)
        pre.buffers[pre.program.step_state].write(pre.program.layout.pack(state), 0)
        pre.run(1, steps_per_cb=1, in_flight=1)
    print('PREFILLED', prompt['id'], pre.state()['position'], flush=True)
    snapshots = []
    seen = set()
    for eng in [pre, *engines.values()]:
        for (name, bs) in eng.program.buffers.items():
            buf = eng.buffers[name]
            if (name, id(buf)) in seen or bs.role in ('weights', 'params'):
                continue
            seen.add((name, id(buf)))
            if name.endswith(('k_cache', 'v_cache')):
                assert buf.nbytes % a.max_context == 0
                offset = (len(ids) - 16) * (buf.nbytes // a.max_context)
            elif bs.role == 'arena' and buf.nbytes > 4 * 1024 * 1024:
                continue
            else:
                offset = 0
            snapshots.append((buf, offset, buf.read(offset, buf.nbytes - offset)))
    print('SNAPSHOT', sum((len(x[2]) for x in snapshots)), flush=True)
    for rep in range(5):
        order = list(engines) if rep % 2 == 0 else list(reversed(engines))
        for label in order:
            for (buf, offset, data) in snapshots:
                buf.write(data, offset)
            eng = engines[label]
            n = 8 if spec else 32
            r = eng.run(n, steps_per_cb=1 if spec else 8, in_flight=2 if spec else 3)
            assert not r.done and len(r.tokens) > 0 and (eng.state()['error'] == 0)
            row = dict(mode=a.mode, label=label, prompt_id=prompt['id'], context=len(ids), rep=rep, warmup=rep == 0, gpu_ms=r.gpu_ms / n, wall_ms=r.wall_ms / n, tokens=r.tokens, steps=r.steps)
            records.append(row)
            print('RESULT', json.dumps(row), flush=True)
    (root / (a.mode + ('-geometry' if a.geometry else '-screen') + '.json')).write_text(json.dumps(records, indent=1))
print('DONE', flush=True)
