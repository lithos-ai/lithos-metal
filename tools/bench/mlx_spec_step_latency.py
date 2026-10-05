"""Time native MLX-LM plain decode or full speculative rounds.

Uses unmodified generators, excluding prefill from decode timing. For N=7,
non-draft yields mark round ends; suspended locals identify shortened tail
rounds. This instrumentation depends on the pinned generator's local names.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--drafter')
    parser.add_argument('--mode', choices=['plain', 'n7'], default='n7')
    parser.add_argument('--prefill-step-size', type=int,
                        help='Defaults to the native generator value: plain 2048, N=7 512')
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'n7' and not args.drafter:
        parser.error('N=7 needs --drafter')
    if args.prefill_step_size is not None and args.prefill_step_size < 1:
        parser.error('--prefill-step-size must be positive')

    import mlx.core as mx
    from mlx_lm import load

    generation = importlib.import_module('mlx_lm.generate')
    model, tokenizer = load(args.model)
    draft = None
    if args.mode == 'n7':
        draft, draft_tokenizer = load(args.drafter)
        if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
            raise ValueError('Target and drafter token-ID mappings differ')
    source_hash = hashlib.sha256(Path(generation.__file__).read_bytes()).hexdigest()
    versions = {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm')}
    prefill = args.prefill_step_size or (512 if draft is not None else 2048)

    for prompt in json.loads(args.prompts.read_text()):
        ids = tokenizer.encode(prompt['prompt'], add_special_tokens=False)
        assert len(ids) == prompt['input_tokens']
        tokens, timestamps, boundaries, intervals = [], [], [], []
        interval_tokens = []
        mx.synchronize()
        start = time.perf_counter()
        kwargs = dict(max_tokens=128, prefill_step_size=prefill)
        generator = (generation.speculative_generate_step(
            mx.array(ids), model, draft, num_draft_tokens=7, **kwargs)
            if draft is not None else generation.generate_step(mx.array(ids), model, **kwargs))
        with generation.wired_limit(model, [generation.generation_stream]):
            try:
                for item in generator:
                    token = int(item[0])
                    now = time.perf_counter()
                    if token in tokenizer.eos_token_ids:
                        break
                    tokens.append(token)
                    timestamps.append((now-start)*1000)
                    if draft is not None and not item[2]:
                        state = generator.gi_frame.f_locals
                        boundary = dict(elapsed_ms=timestamps[-1],
                                        proposed=int(state['num_draft']),
                                        accepted=int(state['n']),
                                        emitted_tokens=len(tokens))
                        if boundaries and boundaries[-1]['proposed'] == boundary['proposed'] == 7:
                            intervals.append(boundary['elapsed_ms'] - boundaries[-1]['elapsed_ms'])
                            interval_tokens.append(boundary['emitted_tokens'] - boundaries[-1]['emitted_tokens'])
                        boundaries.append(boundary)
            finally:
                generator.close()
        assert len(tokens) == 128, 'Early EOS: not comparable to the 128-token screen'
        assert len(timestamps) == 128 and timestamps[-1] > timestamps[0]
        if draft is not None:
            assert intervals and min(intervals) > 0
        decode_ms = timestamps[-1] - timestamps[0]
        row = dict(engine='mlx-lm-' + args.mode, prompt_id=prompt['id'],
                   input_tokens=len(ids), output_tokens=len(tokens),
                   warmup=prompt['rep'] == 0,
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   model=args.model, drafter=args.drafter if draft is not None else None,
                   versions=versions, generator_source_sha256=source_hash,
                   prefill_step_size=prefill, num_draft_tokens=7 if draft is not None else 0,
                   decode_wall_ms=decode_ms, decode_tokens=127,
                   ms_per_token=decode_ms/127, token_yield_times_ms=timestamps,
                   ms_per_full_step=statistics.mean(intervals) if intervals else None,
                   full_step_intervals_ms=intervals, full_step_emitted_tokens=interval_tokens,
                   full_steps_ms_per_token=sum(intervals)/sum(interval_tokens) if intervals else None,
                   round_end_boundaries=boundaries, tokens=tokens,
                   text=tokenizer.decode(tokens), peak_memory_bytes=mx.get_peak_memory())
        with args.out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(prompt['id'], args.mode, round(row['ms_per_token'], 3), 'ms/token',
              round(row['ms_per_full_step'], 3) if intervals else '-', 'ms/full round',
              'warmup' if row['warmup'] else '', flush=True)


if __name__ == '__main__':
    main()
