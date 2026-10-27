"""Exact prefill at its edges: are large exact prompt chunks bit-identical to 128-row chunks for every prompt length?

Runs the served configuration (serving.setup.prepare + the recipe-selected session) twice per recipe tier — 128-row
chunks (the reference) and ``--chunk``-row exact chunks — on prompts whose prompt-graph part (prompt - 8 tokens) is
0, 1, 2 and 127 past a multiple of 128, plus requests split by a prefix-cache checkpoint (unaligned chunk starts,
a cache hit, one-token chunks made by a boundary). Compares greedy tokens, the bits of the verification graph's
final logits, and the speculative rounds (accepted / committed per round: the drafter's context).

usage (loads the served model; run nothing else on the GPU meanwhile):
  python tools/bench/prefill_exact_edges.py run --out DIR [--groups 128,cache,4096,8192,16384] [--max-new 32]
                                                 [--chunk 512] [--budget-s 1000]
  python tools/bench/prefill_exact_edges.py report --out DIR        # exit 0 iff every case that ran is identical
  python tools/bench/prefill_exact_edges.py plan --out DIR          # CPU only: the cases and their recipe tiers
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('cmd', choices=['run', 'report', 'plan'])
ap.add_argument('--out', required=True)
ap.add_argument('--model', default='nvidia/Qwen3.8-27B-NVFP4')
ap.add_argument('--draft', default=None, help="the drafter (serve's --draft; default: the model's catalog drafter)")
ap.add_argument('--pack', default=None, help="pack-cache root (serve's --pack)")
ap.add_argument('--measured-choices', action='store_true',
                help="compile with the configuration's measured kernel choices, as serve does (default: the default geometry)")
ap.add_argument('--groups', default='128,cache,4096,8192,16384')
ap.add_argument('--chunk', type=int, default=512)
ap.add_argument('--max-new', type=int, default=32)
ap.add_argument('--max-context', type=int, default=32768)
ap.add_argument('--budget-s', type=float, default=1000.0,
                help='do not start another session after this many seconds (finished halves of earlier runs are kept)')
a = ap.parse_args()
if a.chunk <= 128:
    ap.error('--chunk must exceed 128: the reference half runs 128-row chunks, and exact mode needs larger ones')
OUT = Path(a.out).expanduser()

# prompt-graph tokens (prompt length - 8) per recipe tier: 0, 1, 2 and 127 past a multiple of 128; the first tier
# also has 257 (one past 256) and 513 (one past 512: the large chunking isolates that token by itself).
LENGTHS = {'128': [128, 129, 130, 255, 257, 513, 514, 639],
           '4096': [4224, 4225, 4226, 4351],
           '8192': [8320, 8321, 8322, 8447],
           '16384': [16512, 16513, 16514, 16639],
           'long': [28800, 28801, 30758],
           'quick': [128, 130, 639]}
# (name, shared prefix tokens, tail source offset, tail tokens, checkpoint) with a prefix cache, in this order:
CACHE = [('fresh-unaligned', 1000, 1000, 1300, 1000),      # checkpoint at 1000: the next segment starts unaligned
         ('hit-edge', 1000, 6000, 1281, 1000),             # cache hit at 1000, then 10 * 128 + 1 prompt-graph tokens
         ('hit-unaligned', 1000, 9000, 1300, 1000),        # cache hit at 1000, no one-token chunk
         ('boundary-edge', 0, 12000, 1405, 897)]           # no hit; the checkpoint leaves 7 * 128 + 1 tokens before it


def report():
    cases = json.loads((OUT / 'cases.json').read_text())
    if '__run__' not in cases:
        print('NOT_IDENTICAL: the cases carry no run configuration (written by an older version)')
        return 1
    run = cases.pop('__run__')
    bad = unchecked = 0
    for name, c in cases.items():
        ref, new = c.get('reference'), c.get('exact')
        if ref is None or new is None:
            print(f'{name:28s} INCOMPLETE'); bad += 1
            continue
        k = min(len(ref['tokens']), len(new['tokens']))
        first = next((i for i in range(k) if ref['tokens'][i] != new['tokens'][i]), None)
        same_tokens = first is None and len(ref['tokens']) == len(new['tokens'])
        rows = [i for i, (x, y) in enumerate(zip(ref['logits_rows'], new['logits_rows'])) if x != y]
        same_logits = not rows and len(ref['logits_rows']) == len(new['logits_rows'])
        unchecked += not ref['logits_rows']                    # (no logits buffer found: tokens only)
        same_rounds = ref.get('rounds') == new.get('rounds')
        ok = same_tokens and same_logits and same_rounds
        bad += not ok
        print(f"{name:28s} P={c['prompt_tokens']:6d} recipe={str(c['recipe']):>5s} reference passes {str(ref['passes'][-4:]):>22s} "
              f"exact passes {str(new['passes'][-4:]):>22s} ref-path={str(new['reference_path']):5s} tokens "
              f"{'same' if same_tokens else f'DIFFER@{first}'} logits "
              f"{('same' if ref['logits_rows'] else 'n/a') if same_logits else f'DIFFER rows {rows}'} "
              f"rounds {'same' if same_rounds else 'DIFFER'} "
              f"prefill {ref['prefill_wall_ms'] / 1e3:6.2f}s -> {new['prefill_wall_ms'] / 1e3:6.2f}s  {'OK' if ok else 'FAIL'}")
    meta = json.loads((OUT / 'meta.json').read_text()) if (OUT / 'meta.json').exists() else {}
    if any(meta.get(k) != v for k, v in run.items()):
        print('NOT_IDENTICAL: meta.json describes another run (an interrupted rerun?)')
        return 1
    skipped = meta.get('skipped', [])
    print(f"RESULT cases={len(cases)} failed={bad} logits_unchecked={unchecked} skipped_groups={skipped} "
          f"model={run['model']} chunk={run['chunk']} head={run['repo_head']} dirty={run['repo_dirty']} code={run['code']}")
    print('ALL_IDENTICAL' if cases and not bad else 'NOT_IDENTICAL')
    return 0 if cases and not bad else 1


if a.cmd == 'report':
    sys.exit(report())

sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np                                             # noqa: E402
from transformers import AutoTokenizer                         # noqa: E402

from monolith.generate import load_session                     # noqa: E402
from monolith.serving.setup import prepare                     # noqa: E402


class NS:
    pass


ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft, ns.no_draft, ns.draft_revision, ns.draft_block_size = a.draft, False, None, None
ns.max_context, ns.draft_quantization, ns.draft_pack, ns.pack = a.max_context, 'auto', None, a.pack
ns.kernel_config, ns.kernel_config_key = None, None
assets = prepare(ns)
tok = AutoTokenizer.from_pretrained(str(assets.model_dir))
text = '\n\n'.join((ROOT / 'docs/design' / f).read_text() for f in ('design.md', 'apple-gpu.md', 'serving.md',
                                                                       'speculative-decoding.md', 'mixers.md')) * 6
doc = list(tok(text, add_special_tokens=False)['input_ids'])
templated = tok.apply_chat_template([{'role': 'user', 'content': '@@DOC@@'}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
head_text, tail_text = templated.split('@@DOC@@')
HEAD = list(tok(head_text, add_special_tokens=False)['input_ids'])
TAIL = list(tok(tail_text, add_special_tokens=False)['input_ids'])


def chat_ids(body):
    return HEAD + body + TAIL


def sized(p):
    """A chat prompt of exactly p tokens around the document."""
    return chat_ids(doc[:p - len(HEAD) - len(TAIL)])


if a.cmd == 'plan':
    for group in a.groups.split(','):
        if group == 'cache':
            for name, prefix, source, n, checkpoint in CACHE:
                ids = chat_ids(doc[:prefix - len(HEAD)] + doc[source:source + n + 8 - len(TAIL)]) if prefix else \
                    chat_ids(doc[source:source + n + 8 - len(HEAD) - len(TAIL)])
                print(f'cache/{name}: P={len(ids)} recipe={assets.options(len(ids))[0]} checkpoint={checkpoint} '
                      f'shared prefix={prefix} head={len(HEAD)} tail={len(TAIL)}')
        else:
            for n in LENGTHS[group]:
                ids = sized(n + 8)
                print(f'{group}/{n}+8: P={len(ids)} recipe={assets.options(len(ids))[0]} '
                      f'128-row passes end {[128] * (n // 128) and n % 128 or 128} 512-row passes end {n % a.chunk or a.chunk}')
    print('document tokens', len(doc))
    sys.exit(0)


OUT.mkdir(parents=True, exist_ok=True)


ARTIFACTS = ('cases.json', 'meta.json')         # what this harness writes into OUT
OWN = {OUT.resolve() / name for name in ARTIFACTS} | {OUT.resolve() / f'.{name}.tmp' for name in ARTIFACTS}


def save(name, text):
    """Replace OUT/name in one step: an interrupted write never leaves a truncated file behind."""
    tmp = OUT / f'.{name}.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT / name)


def repo_dirty():
    """Uncommitted changes in the checkout, apart from this harness's own files."""
    spec = ['.'] + [f':(exclude){path.relative_to(ROOT)}' for path in sorted(OWN) if path.is_relative_to(ROOT)]
    # every untracked file listed by name: a new output directory is not one entry that hides the files beside it
    return bool(subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=all', '--', *spec],
                               capture_output=True, text=True).stdout.strip())


def code_digest():
    """The engine's files (Python, kernels, recipes, native runtime sources) and this harness, committed or not, and
    the native module this process loaded, wherever it was built."""
    from monolith.runtime import _native
    digest = hashlib.sha256()
    files = [path for part in ('monolith', 'kernels', 'runtime') for path in sorted((ROOT / part).rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts and path.resolve() not in OWN]
    for path in files + [Path(__file__).resolve()]:
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0' + path.read_bytes())
    native = Path(getattr(_native, '__file__', '') or '')
    digest.update(b'native\0' + (native.read_bytes() if native.is_file() else b'-'))
    return digest.hexdigest()[:16]


def artifact_stamp(*dirs):
    """Every file the model, pack and draft directories hold, by path, size and modification time (a file replaced or
    repacked in place changes it), apart from this harness's own files; directories that are not set count as empty."""
    digest = hashlib.sha256()
    for root in dirs:
        if root is None or not Path(root).exists():
            digest.update(b'-')
            continue
        for path in sorted(Path(root).rglob('*')):
            if path.is_file() and path.resolve() not in OWN:
                info = path.stat()
                digest.update(f'{path.relative_to(root)}\0{info.st_size}\0{info.st_mtime_ns}\n'.encode())
    return digest.hexdigest()[:16]


RUN = dict(repo_head=os.popen(f'git -C {ROOT} rev-parse --short HEAD 2>/dev/null').read().strip(),
           repo_dirty=repo_dirty(), code=code_digest(),
           # the token sequences every case is cut from, and the weights they run on
           prompts=hashlib.sha256(json.dumps([HEAD, TAIL, doc]).encode()).hexdigest()[:16],
           model_dir=str(assets.model_dir), pack=str(assets.pack_dir),
           draft=[str(assets.draft_dir), str(assets.draft_pack)],
           artifacts=artifact_stamp(assets.model_dir, assets.pack_dir, assets.draft_dir, assets.draft_pack),
           model=a.model, chunk=a.chunk, max_new=a.max_new, max_context=a.max_context,
           # engine switches (MONOLITH_ACCELERATOR, LITHOS_*) and Metal's (MTL_*, e.g. shader validation); no API keys
           env={k: v for k, v in os.environ.items() if k.startswith(('LITHOS_', 'MONOLITH_', 'MTL_'))
                and not k.endswith('API_KEY')})
cases = json.loads((OUT / 'cases.json').read_text()) if (OUT / 'cases.json').exists() else {}
if cases and cases.get('__run__') != RUN:
    # Cases resume only within one configuration and the same engine files: a reference and an exact half of
    # different runs never pair.
    print('prefill_exact_edges: the output holds another configuration; starting over', flush=True)
    cases = {}
cases['__run__'] = RUN
save('cases.json', json.dumps(cases))     # a reset holds even if this run stops before its first case
meta = dict(RUN, time=time.strftime('%Y-%m-%dT%H:%M:%S'), skipped=[], sessions=[])
started = time.time()
shared = {}


def session_for(prompt_tokens, exact, cache):
    key, options = assets.options(prompt_tokens)
    kw = dict(prefill_chunk_size=a.chunk, prefill_exact=True) if exact else dict(prefill_chunk_size=128)
    if cache:
        kw.update(prefix_cache=True, prefix_cache_min_tokens=128)
    t = time.time()
    session = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, temperature=0.0, autotune=False,
                           measured_choices=a.measured_choices, **shared, **kw)
    if exact and not session.prefill_exact:
        raise SystemExit(f'prefill_exact_edges: --chunk {a.chunk} does not run the exact graph')
    shared.update(device=session.dev, pipeline_cache=session._pipelines)
    meta['sessions'].append(dict(recipe=key, exact=exact, cache=cache, load_s=round(time.time() - t, 2)))
    return key, session


def run_case(session, name, key, config, ids, **kw):
    dec = session.engines.get(0)
    if dec is not None:
        logits = logits_buffer(dec)
        if logits is not None:
            dec.buffers[logits].fill(0)                        # rows a request does not write must not differ
    del dec                                                    # generate may release this engine: hold no reference
    t = time.time()
    g = session.generate(ids, a.max_new, **kw)
    dec = session.engines.get(0)
    logits = logits_buffer(dec) if dec is not None else None
    rows = []
    if logits is not None:
        v = session.model.config.vocab_size
        raw = np.frombuffer(dec.buffers[logits].read(0, session.decode_t_max * v * 2), dtype=np.uint16)
        rows = [hashlib.sha256(raw[i * v:(i + 1) * v].tobytes()).hexdigest()[:16] for i in range(session.decode_t_max)]
    rec = dict(tokens=list(g.tokens[:a.max_new]), logits_rows=rows, passes=[t_['tokens'] for t_ in g.prefill_timings],
               rounds=[g.steps, g.accepted, g.committed],     # the drafter saw the same context: same rounds
               reference_path=bool(getattr(session, 'last_prefill_reference', False)),
               cached_prompt_tokens=g.cached_prompt_tokens, prefill_wall_ms=g.prefill_wall_ms, setup_ms=g.setup_ms,
               wall_s=round(time.time() - t, 2))
    cases.setdefault(name, dict(prompt_tokens=len(ids), recipe=key))[config] = rec
    save('cases.json', json.dumps(cases))
    print(f"{name:28s} {config:9s} P={len(ids):6d} recipe={key} passes={rec['passes'][-4:]} ref-path={rec['reference_path']} "
          f"cached={rec['cached_prompt_tokens']} prefill={g.prefill_wall_ms / 1e3:.2f}s wall={rec['wall_s']}s "
          f"t={time.time() - started:.0f}", flush=True)


def logits_buffer(dec):
    for op in dec.program.ops:
        if op.name.startswith('lm_head') and 'draft' not in str(op.meta.get('fusion_region', '')):
            for slot, name, _ in op.bindings:
                if slot == 3:
                    return name
    return None


def case_names(group):
    return [f'cache/{c[0]}' for c in CACHE] if group == 'cache' else [f'{group}/{n}+8' for n in LENGTHS[group]]


for group in a.groups.split(','):
    for config in ('reference', 'exact'):
        # A half (one group, one config, one session) resumes whole: the cache cases build on each other's checkpoints.
        if all(config in cases.get(name, {}) for name in case_names(group)):
            continue
        if time.time() - started > a.budget_s:
            meta['skipped'].append(f'{group}:{config}')
            continue
        if group == 'cache':
            key, session = session_for(2308, config == 'exact', True)
            for name, prefix, source, n, checkpoint in CACHE:
                ids = chat_ids(doc[:prefix - len(HEAD)] + doc[source:source + n + 8 - len(TAIL)]) if prefix else \
                    chat_ids(doc[source:source + n + 8 - len(HEAD) - len(TAIL)])
                run_case(session, f'cache/{name}', key, config, ids, cache_prefix_tokens=[checkpoint])
        else:
            key, session = session_for(LENGTHS[group][0] + 8, config == 'exact', False)
            for n in LENGTHS[group]:
                ids = sized(n + 8)
                if assets.options(len(ids))[0] != key or key != {'long': '16384', 'quick': '128'}.get(group, group):
                    print(f'NOTE {group}/{n}+8: serving would select recipe {assets.options(len(ids))[0]}, session has {key}', flush=True)
                run_case(session, f'{group}/{n}+8', key, config, ids)
        session.release_engines()
        del session
meta['elapsed_s'] = round(time.time() - started, 1)
save('meta.json', json.dumps(meta, indent=1))
sys.exit(report())
