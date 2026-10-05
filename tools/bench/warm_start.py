#!/usr/bin/env python3
"""Profile a real serving prompt, including exact-prefix reuse and Metal host phases.

Pass a locally captured Anthropic Messages or OpenAI Chat request. Compilation
is warmed separately. Output contains timings and completions, not prompt text.
The message-boundary control flattens text blocks to reproduce the older cache
policy without changing the tokenized prompt. Do not run beside another server
holding the same large model on the GPU.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--format', choices=('messages', 'chat'), default='messages')
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=32)
    ap.add_argument('--message-boundary-control', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from monolith.serve import Backend, parse_args
    from monolith.serving.protocol import ChatRequest, anthropic_request
    from monolith.serving.setup import prepare
    assets = prepare(parse_args(['--model', args.model, '--local-files-only']))
    backend = Backend(str(assets.model_dir), str(assets.pack_dir), max_context=assets.max_context,
                      prefill_chunk_size=args.chunk, assets=assets)
    body = json.loads(args.request.read_text())
    body.update(model=args.model, max_tokens=args.max_tokens)
    body.pop('max_completion_tokens', None)
    request = anthropic_request(body) if args.format == 'messages' else ChatRequest.model_validate(body)
    if args.message_boundary_control:
        for message in request.messages:
            if isinstance(message.content, list):
                message.content = ''.join(part.text for part in message.content)
    messages, tools = request.template_inputs()
    ids = backend.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False, **({'tools': tools} if tools else {}))
    backend.select_session(request, len(ids))
    print(f'Compiling selected recipe for {len(ids)} prompt tokens', flush=True)
    backend.session.prepare()
    # Keep the full generation diagnostics without changing Backend's public
    # completion result or timing its JSON serialization.
    generate = backend.session.generate
    captured = []
    def record(*a, **kw):
        result = generate(*a, **kw)
        captured.append(result)
        return result
    backend.session.generate = record
    rows = []
    for rep in range(args.reps):
        first = None
        start = time.perf_counter()
        def on_text(text):
            nonlocal first
            if text and first is None:
                first = (time.perf_counter() - start) * 1000
        text, finish, prompt, completion = backend.complete(request, on_text=on_text)
        generation = asdict(captured.pop())
        generation.pop('tokens')
        row = dict(run=rep, prompt_tokens=prompt, completion_tokens=completion,
                   first_text_ms=first, text=text, finish_reason=finish,
                   metrics=backend.last_metrics, generation=generation)
        rows.append(row)
        args.out.write_text(json.dumps(dict(model=args.model, chunk=args.chunk,
            message_boundary_control=args.message_boundary_control, runs=rows), indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'generation'}), flush=True)


if __name__ == '__main__':
    main()
