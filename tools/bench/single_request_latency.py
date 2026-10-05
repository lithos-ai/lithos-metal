"""Compare serial decode latency using each server's engine timing counters.

Use --serve-monolith for a benchmark-only adapter around Session.generate.
Raw prompts avoid server-specific chat templates. Decode excludes prefill and
the first generated token; HTTP wall time is retained as a secondary metric.
All requests are non-streaming. Warmup rows and generated text are retained.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def serve(args):
    from fastapi import FastAPI
    import uvicorn
    from transformers import AutoTokenizer
    from monolith.generate import load_session

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    session = load_session(args.model, args.pack, max_context=args.max_context,
                           temperature=0, autotune=args.autotune, commute_norm=True,
                           attention=args.attention,
                           prefill_chunk_size=args.prefill_chunk_size,
                           drafter_dir=args.drafter, drafter_pack=args.drafter_pack,
                           drafter_kind='lm', drafter_options={'gamma': 7},
                           verify='fixed' if args.drafter else 'cost',
                           verify_length=7 if args.drafter else None)
    app = FastAPI()

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.post('/generate')
    def generate(body: dict):
        ids = tokenizer.encode(body['prompt'], add_special_tokens=False)
        start = time.perf_counter()
        result = session.generate(ids, body['max_tokens'])
        wall = time.perf_counter() - start
        return dict(text=tokenizer.decode(result.tokens),
                    usage=dict(prompt_tokens=len(ids), completion_tokens=len(result.tokens)),
                    engine=dict(autotune=args.autotune, attention=args.attention or session.profile.attention,
                                max_context=args.max_context,
                                prefill_gpu_ms=result.prefill_ms,
                                decode_gpu_ms=result.decode_ms,
                                decode_wall_ms=result.decode_wall_ms,
                                generation_wall_ms=wall * 1000,
                                decode_tokens=result.decode_tokens,
                                accepted=result.accepted, committed=result.committed,
                                verify_len=result.verify_len))

    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1)


def decode_counter(url):
    with urllib.request.urlopen(url + '/metrics', timeout=30) as response:
        lines = response.read().decode().splitlines()
    values = []
    for prefix in ('vllm:request_decode_time_seconds_sum',
                   'vllm:request_decode_time_seconds_count',
                   'vllm:spec_decode_num_drafts_total',
                   'vllm:spec_decode_num_draft_tokens_total',
                   'vllm:spec_decode_num_accepted_tokens_total'):
        values.append(sum(float(line.split()[-1]) for line in lines
                          if line.startswith(prefix + '{') or line.startswith(prefix + ' ')))
    return values


def request(args, prompt, count):
    if args.kind == 'llama':
        path = '/completion'
        body = dict(prompt=prompt, n_predict=count, temperature=0, seed=17,
                    stream=False, cache_prompt=False, repeat_penalty=1.0)
    elif args.kind == 'ollama':
        path = '/api/generate'
        body = dict(model=args.model, prompt=prompt, raw=True, stream=False,
                    keep_alive='30m', options=dict(num_predict=count, temperature=0,
                    seed=17, num_ctx=4608, repeat_penalty=1.0))
    elif args.kind == 'vllm':
        path = '/v1/completions'
        body = dict(model=args.model, prompt=prompt, max_tokens=count,
                    temperature=0, seed=17, stream=False)
    else:
        path = '/generate'
        body = dict(prompt=prompt, max_tokens=count)
    if args.kind == 'ollama':
        # With LLAMA_ARG_CACHE_RAM=0, an unrelated one-token request replaces
        # the active slot. This untimed primer prevents all prefix-cache hits.
        primer = dict(body, prompt='x', options=dict(body['options'], num_predict=1))
        reset = urllib.request.Request(args.url + path, json.dumps(primer).encode(),
                                       {'Content-Type': 'application/json'})
        with urllib.request.urlopen(reset, timeout=600) as response:
            json.load(response)
    before = decode_counter(args.url) if args.kind == 'vllm' else None
    req = urllib.request.Request(args.url + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        data = json.load(response)
    wall = (time.perf_counter() - start) * 1000
    if args.kind == 'llama':
        text = data['content']
        usage = dict(prompt_tokens=data['tokens_evaluated'], completion_tokens=data['tokens_predicted'])
        engine = data.get('timings', {})
    elif args.kind == 'ollama':
        if data.get('prompt_eval_cached_count', 0):
            raise RuntimeError('Ollama reused a prompt: disable LLAMA_ARG_CACHE_RAM')
        text = data['response']
        usage = dict(prompt_tokens=data['prompt_eval_count'], completion_tokens=data['eval_count'])
        engine = {k: v for k, v in data.items() if k.endswith(('_duration', '_count'))}
    elif args.kind == 'vllm':
        text = data['choices'][0]['text']
        usage = data['usage']
        deadline = time.monotonic() + 10
        after = decode_counter(args.url)
        while after[1] == before[1] and time.monotonic() < deadline:
            time.sleep(.01)
            after = decode_counter(args.url)
        if after[1] - before[1] != 1:
            raise RuntimeError(f'Expected one finished request: {before}, {after}')
        engine = dict(decode_wall_ms=(after[0] - before[0]) * 1000,
                      decode_tokens=usage['completion_tokens'] - 1,
                      metric='vllm:request_decode_time_seconds_sum',
                      counter_before=before, counter_after=after)
        if after[2] > before[2]:
            engine.update(draft_rounds=after[2] - before[2],
                          draft_tokens=after[3] - before[3],
                          accepted_draft_tokens=after[4] - before[4])
    else:
        text, usage, engine = data['text'], data['usage'], data['engine']
    if args.kind in ('monolith', 'vllm'):
        decode_ms = engine['decode_wall_ms']
    elif args.kind == 'llama':
        decode_ms = engine['predicted_ms']
    else:
        decode_ms = engine['eval_duration'] / 1e6
    decode_tokens = engine.get('decode_tokens', usage['completion_tokens'] - 1)
    return dict(wall_ms=wall, usage=usage, engine=engine, text=text,
                decode_ms_per_token=decode_ms / decode_tokens if decode_tokens else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve-monolith', action='store_true')
    parser.add_argument('--port', type=int, default=18100)
    parser.add_argument('--prefill-chunk-size', type=int, default=128)
    parser.add_argument('--max-context', type=int, default=4608)
    parser.add_argument('--autotune', action='store_true',
                        help='Enable projection tuning/cache reuse in the Monolith adapter')
    parser.add_argument('--attention', choices=['auto', 'v1', 'v2', 'v3', 'mma', 'mma-direct'])
    parser.add_argument('--pack')
    parser.add_argument('--drafter')
    parser.add_argument('--drafter-pack')
    parser.add_argument('--kind', choices=['monolith', 'llama', 'ollama', 'vllm'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--url')
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--counts', default='128')
    args = parser.parse_args()
    if args.serve_monolith:
        if not args.pack or bool(args.drafter) != bool(args.drafter_pack):
            parser.error('server needs --pack and either both draft paths or neither')
        return serve(args)
    if not all((args.kind, args.url, args.prompts, args.out)):
        parser.error('client needs --kind, --url, --prompts and --out')
    counts = [int(n) for n in args.counts.split(',')]
    if not counts or min(counts) < 1:
        parser.error('--counts must contain positive token counts')
    prompts = json.loads(args.prompts.read_text())
    for length in dict.fromkeys(p['input_tokens'] for p in prompts):
        selected = [p for p in prompts if p['input_tokens'] == length]
        for count in counts:
            for p in selected:
                row = dict(engine_name=args.label or args.kind, model=args.model,
                           prompt_id=p['id'], input_tokens=p['input_tokens'],
                           requested_output_tokens=count, warmup=p['rep'] == 0,
                           timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                row.update(request(args, p['prompt'], count))
                if row['usage']['prompt_tokens'] != p['input_tokens']:
                    raise RuntimeError(f"Prompt token count differs for {p['id']}")
                row['full_length'] = row['usage']['completion_tokens'] == count
                with args.out.open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(row['engine_name'], p['id'], count, round(row['wall_ms'], 2),
                      row['usage'], 'warmup' if row['warmup'] else '', flush=True)


if __name__ == '__main__':
    main()
