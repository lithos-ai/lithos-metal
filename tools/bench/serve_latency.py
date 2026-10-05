"""Measure HTTP wall time and server-reported complete decode-step GPU time separately.

Requires httpx. Input is an ordinary Chat Completions JSON request. Warmups are
saved but excluded from the summary; terminal rounds can verify fewer tokens
than the configured width reported by the server.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    if args.warmup < 0 or args.reps < 1:
        parser.error('warmup must be nonnegative and reps must be positive')
    payload = json.loads(args.request.read_text())
    rows = []
    api_key = os.environ.get('MONOLITH_API_KEY')
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        for i in range(args.warmup + args.reps):
            start = time.perf_counter()
            response = client.post(args.url, json=payload)
            wall_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            h = response.headers
            row = dict(warmup=i < args.warmup, http_wall_ms=wall_ms,
                       decode_steps=int(h['x-monolith-decode-steps']),
                       decode_gpu_ms=float(h['x-monolith-decode-gpu-ms']),
                       mean_decode_step_ms=float(h['x-monolith-decode-step-ms']),
                       configured_verify_tokens=int(h['x-monolith-verify-tokens']),
                       usage=response.json()['usage'])
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(request=payload, samples=rows), indent=2))
    measured = [r for r in rows if not r['warmup']]
    summary = {key: dict(min=min(r[key] for r in measured),
                        median=statistics.median(r[key] for r in measured),
                        max=max(r[key] for r in measured))
               for key in ('mean_decode_step_ms', 'http_wall_ms')}
    args.out.write_text(json.dumps(dict(request=payload, samples=rows, summary=summary), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
