#!/usr/bin/env python3
"""Paired end-to-end prefill comparison, excluding compilation and checking greedy output equality."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.generate import load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pack', required=True)
    parser.add_argument('--chunks', type=int, nargs='+', default=[8, 128])
    parser.add_argument('--lengths', type=int, nargs='+', default=[128, 512, 1024])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-context', type=int, default=2048)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(args.model) / 'tokenizer.json'))
    unit = tokenizer.encode('The quick brown fox jumps over the lazy dog. ', add_special_tokens=False).ids
    sessions = {chunk: load_session(args.model, args.pack, max_context=args.max_context,
                                    prefill_chunk_size=chunk, eos=-1, autotune=False) for chunk in args.chunks}
    for length in args.lengths:
        ids = (unit * (-(-length // len(unit))))[:length]
        expected = None
        for session in sessions.values():
            result = session.generate(ids, 16)
            if expected is None:
                expected = result.tokens
            if result.tokens != expected:
                raise RuntimeError(f'Greedy output mismatch at prompt length {length}')
        samples = {chunk: [] for chunk in sessions}
        for repeat in range(args.repeats):
            order = args.chunks if repeat % 2 == 0 else args.chunks[::-1]
            for chunk in order:
                start = time.perf_counter()
                result = sessions[chunk].generate(ids, 1)
                wall = (time.perf_counter() - start) * 1000
                assert result.tokens == expected[:1]
                samples[chunk].append({'gpu_ms': result.prefill_ms, 'wall_ms': wall})
        print(json.dumps({'model': Path(args.model).name, 'chip': next(iter(sessions.values())).dev.info().name,
                          'prompt_tokens': length, 'greedy_16_equal': True, 'samples': samples}), flush=True)


if __name__ == '__main__':
    main()
