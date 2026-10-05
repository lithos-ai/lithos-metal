# Separate prefill and decode programs

`Session` processes prompts in 128-token chunks by default. Set `prefill_chunk_size` in Python or
`--prefill-chunk-size` in `monolith.generate` / `monolith.serve` to change the chunk size.
This is a per-pass limit; longer prompts use multiple chunks, up to the configured context capacity.

Prefill and decode are lowered and compiled independently into separate programs and indirect command buffers.
Short prompts select a smaller power-of-two prefill bucket, capped by the configured chunk size.
Plain decode stays at one token. Speculative decode keeps its original verification bound (normally eight),
while the prefill program ingests the drafter's context and bootstraps the first draft block.
The prefill graph retains one-token variants for a partial final chunk even when verification cannot use them.

Both programs share weights, KV caches, recurrent state, the token ring, and one StepState ABI. Its pending-token
storage accommodates the larger chunk, but an explicit compiler `t` bound controls each graph's activations and
dispatches independently. Increasing prefill capacity therefore does not increase decode's logical row counts.
Prefill skips the decode-oriented leaf autotuner; decode retains its existing tuning behavior.

The M5 GEMM dispatch tiles the token dimension in blocks of at most 32 rows using `grid.y`. The input permute
allocates and pads every token tile, and the GEMM offsets outputs, residuals, and statistics for each tile.
Inactive token tiles return before accessing data. The last weight tile clamps loads to a valid packed block;
its extra output rows remain masked. Larger chunks need more temporary GPU memory.

## Paired measurement [M]

Apple M5 Pro, Qwen3.5-0.8B BF16, 2026-09-28; 2,048-position context, accelerator and attention selected by the
device profile, autotuning disabled. Three alternating A/B repetitions after compilation and warmup, min-of-three
reported below. Timing covers prompt processing and sampling the first token; compilation is excluded.
The 8-token baseline uses the same implementation with `prefill_chunk_size=8`.

| Prompt tokens | 8-token chunks, GPU ms | 128-token chunks, GPU ms | GPU speedup | Wall ms, 8 → 128 |
|---:|---:|---:|---:|---:|
| 128 | 117.38 | 18.29 | 6.42× | 123.92 → 19.46 |
| 512 | 473.94 | 76.64 | 6.18× | 499.35 → 78.93 |
| 1,024 | 1,005.37 | 162.38 | 6.19× | 1,055.49 → 165.92 |

All 16 greedy output tokens matched between chunk sizes at every prompt length. These measurements cover this
model and device; they do not establish a speedup for other models, drafters, or Apple GPU generations.
Raw samples: [`apple-m5-pro-20c_prefill_20260928.jsonl`](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_prefill_20260928.jsonl).

```sh
python tools/bench/prefill_chunks.py --model /path/to/checkpoint --pack /path/to/pack
```

Regression coverage includes 128-, 129-, and 259-token prompts followed by a short request, plain/LM/DSpark
state handoff, partial token and weight tiles, residual/statistic/SiLU epilogues, and independent compiler bounds.
GPU tests run with `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`.
