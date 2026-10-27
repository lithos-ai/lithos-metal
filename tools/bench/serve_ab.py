#!/usr/bin/env python3
"""Serve-equivalent decode and time-to-first-token measurement, one arm per process.

decode: sessions built the way serve builds them (serving.setup.prepare, then
Backend.select_session's load_session options), greedy, EOS off, N new tokens on
spec_bench prompts; GPU decode tok/s = committed tokens / decode GPU time.
ttft: an in-process serve Backend (warmed like serve), chat prompts of exactly
L tokens, a nonce first so no prefix-cache reuse; first_text_ms as serve logs it.

    python tools/bench/serve_ab.py decode --arm after --model M --draft D --pack ROOT --out X.jsonl
    python tools/bench/serve_ab.py ttft --arm after --model M --no-draft --pack ROOT --lengths 1024,4096,9216 --out X.jsonl
Before-arm knobs: --profile <main's config JSON> (load_config), --no-choices
(measured_choices=False), --chunk 128 (serve's prompt chunk before the hook).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def assets_for(a):
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare

    argv = ['--model', a.model, '--pack', a.pack, '--local-files-only', '--max-context', str(a.max_context)]
    argv += ['--draft', a.draft] if a.draft else ['--no-draft']
    if a.chunk:
        argv += ['--prefill-chunk-size', a.chunk]
    ns = parse_args(argv)
    assets = prepare(ns)
    if a.profile:                                          # e.g. origin/main's config: the "before" arm
        from monolith.backends.metal.config import load_config
        assets.profile = load_config(a.profile)
    rows, exact = ns.prefill_chunk_size                    # as serve resolves its prompt chunk
    chunk = rows or assets.prefill_chunk_size or 128
    return ns, assets, chunk, exact


def meta(a, assets, chunk, exact, dev):
    info = dev.info()
    return dict(kind='metadata', arm=a.arm, mode=a.mode, chip=info.name, cores=info.gpu_cores,
                revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                dirty=bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'monolith'], text=True).strip()),
                profile=assets.profile.name, profile_fingerprint=getattr(assets.profile, 'fingerprint', None),
                measured_choices=not a.no_choices, chunk=chunk, exact=exact, gamma=assets.gamma,
                model=str(assets.model_dir), pack=str(assets.pack_dir), draft=str(assets.draft_dir), draft_pack=str(assets.draft_pack),
                max_context=a.max_context, capacity=assets.capacity, timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'))


def decode(a, emit):
    from tokenizers import Tokenizer
    from monolith.generate import load_session
    from spec_bench import PROMPTS

    ns, assets, chunk, exact = assets_for(a)
    tok = Tokenizer.from_file(str(Path(assets.model_dir) / 'tokenizer.json'))
    names = a.prompts.split(',')
    ids = {n: tok.encode(PROMPTS[n.rstrip('0123456789')][int(n[len(n.rstrip('0123456789')):])],
                         add_special_tokens=False).ids for n in names}
    _, options = assets.options(max(map(len, ids.values())))
    # serve's session options, minus the prefix cache (decode only), plus eos=-1
    sess = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, temperature=0.0,
                        autotune=False, measured_choices=not a.no_choices, prefill_chunk_size=chunk, prefill_exact=exact)
    emit(meta(a, assets, chunk, exact, sess.dev))
    sess.generate(ids[names[0]], a.n)                                  # compile + warm the measured length
    for rep in range(a.reps):
        for n in names:
            g = sess.generate(ids[n], a.n)
            committed = sum(g.committed) if g.committed else g.decode_tokens
            emit(dict(kind='measurement', arm=a.arm, rep=rep, prompt=n, prompt_tokens=len(ids[n]),
                      new_tokens=len(g.tokens), steps=g.steps, committed=committed,
                      gpu_tps=1000.0 / g.ms_per_token, wall_tps=g.decode_tokens / (g.decode_wall_ms / 1e3),
                      ms_per_step=g.decode_ms / max(1, g.steps), tokens_per_step=g.tokens_per_step,
                      mean_accepted=g.mean_accepted, host_pct=100 * g.host_busy_ms / max(g.decode_wall_ms, 1e-9),
                      tokens=g.tokens[:a.n]))


def ttft(a, emit):
    from monolith.serve import Backend
    from monolith.serving.protocol import ChatRequest, Message

    ns, assets, chunk, exact = assets_for(a)
    backend = Backend(str(assets.model_dir), str(assets.pack_dir), ns.max_context, chunk,
                      assets=assets, prefill_exact=exact)
    if a.no_choices:
        raise SystemExit('ttft: Backend always passes measured_choices=True; run the before arm from a main checkout')
    backend.warmup('bench')                                                # what serve does at startup
    emit(meta(a, assets, chunk, exact, backend.session.dev))
    tk = backend.tokenizer

    def prompt_of(length, nonce):
        words = ['This paragraph is padding for a latency benchmark.'] * (length // 8)
        head = f'[{nonce}] '
        q = '\nIgnore the padding. List the integers from 1 through 20.'
        def count(text):
            return len(tk.apply_chat_template([dict(role='user', content=text)], tokenize=True, return_dict=False,
                                              add_generation_prompt=True, enable_thinking=False))
        text = head + ' '.join(words) + q
        while count(text) < length:
            words.append('pad'); text = head + ' '.join(words) + q
        while count(text) > length and words:
            words.pop(); text = head + ' '.join(words) + q
        return text, count(text)

    for rep in range(a.reps):
        for length in map(int, a.lengths.split(',')):
            text, n = prompt_of(length, uuid.uuid4().hex[:8])
            req = ChatRequest(model='bench', messages=[Message(role='user', content=text)], max_tokens=a.n)
            start = time.perf_counter()
            out, finish, prompt_tokens, completion = backend.complete(req, on_text=lambda t: None)
            m = dict(backend.last_metrics)
            assert m.get('cached_prompt_tokens', 0) == 0, 'prefix cache hit: TTFT not comparable'
            emit(dict(kind='measurement', arm=a.arm, rep=rep, target_tokens=length, prompt_tokens=prompt_tokens,
                      request_ms=(time.perf_counter() - start) * 1000, first_text_ms=m.get('first_text_ms'),
                      prefill_gpu_ms=m.get('prefill_gpu_ms'), prefill_wall_ms=m.get('prefill_wall_ms'),
                      setup_ms=m.get('setup_ms'), load_ms=m.get('load_ms'), text=out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['decode', 'ttft'])
    ap.add_argument('--arm', required=True)
    ap.add_argument('--model', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--draft')
    g.add_argument('--no-draft', action='store_true')
    ap.add_argument('--pack', required=True, help='pack-cache root, as serve --pack')
    ap.add_argument('--max-context', type=int, default=32768)
    ap.add_argument('--chunk', default=None, help="serve's --prefill-chunk-size (N, N-exact, auto-exact)")
    ap.add_argument('--profile', default=None)
    ap.add_argument('--no-choices', action='store_true')
    ap.add_argument('--prompts', default='code0,math0,chat0,text0')
    ap.add_argument('--lengths', default='1024,4096,9216')
    ap.add_argument('-n', type=int, default=256)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with a.out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ('tokens', 'text')}), flush=True)

    (decode if a.mode == 'decode' else ttft)(a, emit)


if __name__ == '__main__':
    main()
