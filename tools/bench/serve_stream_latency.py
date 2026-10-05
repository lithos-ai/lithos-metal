"""Measure real SSE time to first text, output cadence, and complete HTTP time.

Examples: --request /tmp/request.json --out /tmp/stream-results.json --reps 3
The first run is retained separately; no empty role event or keepalive counts as
the first token. Server GPU timing is reported independently in its log.
"""
import argparse
import json
import os
from pathlib import Path
import time

import httpx


def measure(client, url, payload, tokenizer=None):
    started = time.perf_counter()
    first = last = None
    text, usage, chunks, gaps, first_content = '', {}, 0, [], ''
    with client.stream('POST', url, json={**payload, 'stream': True,
                       'stream_options': {'include_usage': True}}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if 'error' in event:
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if not content:
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                    first_content = content
                    print(json.dumps({'first_text_ms': (first - started) * 1000}), flush=True)
                elif last is not None:
                    gaps.append((now - last) * 1000)
                last = now
                chunks += 1
                text += content
    elapsed = time.perf_counter() - started
    result = dict(ttft_ms=(first-started)*1000 if first else None, wall_ms=elapsed*1000,
                # Includes protocol EOS in usage; count-1 excludes the prefill
                # token. Treat this as a cadence estimate for text completions.
                usage_tps_estimate=(usage.get('completion_tokens', 0)-1)/(last-first) if first and last > first else None,
                text_events=chunks, max_text_gap_ms=max(gaps, default=0), usage=usage, text=text)
    if tokenizer is not None:
        visible = len(tokenizer.encode(text, add_special_tokens=False))
        first_tokens = len(tokenizer.encode(first_content, add_special_tokens=False))
        result.update(visible_tokens=visible, first_event_tokens=first_tokens,
                      visible_output_tps=(visible-first_tokens)/(last-first) if first and last > first else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--tokenizer', type=Path, help='Local tokenizer for visible-text throughput, excluding EOS')
    args = parser.parse_args()
    payload = json.loads(args.request.read_text())
    if args.max_tokens:
        payload.pop('max_completion_tokens', None)
        payload['max_tokens'] = args.max_tokens
    rows = []
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    key = os.environ.get('LITHOS_METAL_API_KEY', 'lithos-metal-local')
    with httpx.Client(timeout=1800, headers={'Authorization': f'Bearer {key}'}) as client:
        for index in range(args.reps):
            row = dict(run=index, **measure(client, args.url, payload, tokenizer))
            rows.append(row)
            args.out.write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != 'text'}), flush=True)


if __name__ == '__main__':
    main()
