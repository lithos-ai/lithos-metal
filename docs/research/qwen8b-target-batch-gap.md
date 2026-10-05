# Qwen3 8B target verification versus plain decode

The parity target is **plain T=1**, one target row. N=7 verifies **T=8** rows
(anchor plus seven proposals). Parity is **not achieved**; [#136](https://github.com/jiazhihao/mpk-apple/issues/136)
tracks the remaining gap. Speculative gamma=1/T=2 is not the acceptance baseline.

On the 20-core M5 Pro, the retained change reduces long-context target latency:

| Prefix tokens | Plain T=1 GPU ms | Previous N=7 GPU ms | New N=7 GPU ms | Paired N=7 improvement | Remaining paired gap to plain |
|---|---:|---:|---:|---:|---:|
| 4,095 | 20.23 | 26.33 | 22.63 | 14.1% | 11.8% |
| 8,191 | 23.36 | 33.08 | 26.03 | 21.3% | 11.3% |

Timings replay the **actual fixed-N=7 speculative program prefix**, ending before
argmax. They include embedding, all 36 decoder layers, final normalization and
all eight vocabulary-logit rows. Drafting, sampling, acceptance, prefill and output
reads are excluded. Plain uses the normal one-row decode program, similarly
clipped before argmax. Median wall times are 22.67/26.07 ms for N=7 and
20.27/23.40 ms for plain. Ratios are medians of paired ratios, not ratios of medians.

Ten alternating blocks of eight replays follow one warmup block. Both modes use
the same real prefix and input IDs, checkpoint, unchanged mixed block/inline pack,
cached projection choices, commute_norm=True, attention=auto, safe math, capacity
8,704 and prefill64. State is restored outside timing; StepState remains unchanged
and all-logit hashes are stable within each configuration. One process owns the
GPU. Repeats are not independent prompts. Absolute times drift between sessions;
use paired comparisons. The 126-token control regresses 1.5% (18.98→19.27 ms).

## Change

A preparation dispatch normalizes/rotates queries once and writes new K/V into the
cache. Matrix attention then reads device KV tensors directly, removing repeated
threadgroup staging. Key partitions grow from 64 to 256, reducing partial-output
traffic and merge work. Four SIMD-groups per threadgroup and eight groups per core
won the measured search. The compiler selects this path only for the relaxed
commuted-normalization path, T=8, 32 query/8 KV heads, D=128, no LM-attention mode,
and cache capacity at least 1,024 and divisible by 256. Aligned capacity keeps
full matrix slices within the allocated cache. Other shapes retain their paths.

The larger softmax partition changes rounding. The kernel oracle keeps cosine
>0.99999 and maximum error ≤2 BF16 ULPs for its partitioned contract. Attention and
contract regression checks passed (140 tests); the extended direct-path suite
passed all 20 cases with/without Q/K normalization and parameter specialization.
Causal tails and the last allocated cache tile, untouched KV prefix/tail, inactive/done states and deterministic
replay are checked. Fresh N=7 generation at both contexts produced the same 32
tokens and acceptance sequences as the old path, including rejected drafts.

## Remaining cost by layer operation

These are **sums of per-dispatch medians**, from 12 alternating timestamp traces,
for attribution only; they must not replace the full-pass timer above.

| Operation, all 36 layers | Plain / N=7 ms at 4K | Plain / N=7 ms at 8K |
|---|---:|---:|
| QKV projection and initial permutation | 2.01 / 2.17 | 2.01 / 2.15 |
| Attention, including preparation and merge | 3.17 / 3.62 | 6.06 / 6.54 |
| Output projection | 1.33 / 1.61 | 1.34 / 1.66 |
| Gate/up projection | 7.20 / 7.93 | 7.18 / 8.13 |
| Down projection | 3.83 / 4.35 | 3.81 / 4.33 |

Every individual decoder layer still has a gap: 10.4–16.3% at 4K and 10.0–13.1%
at 8K in these profiles. [All 36 layer measurements](https://github.com/jiazhihao/mpk-apple/blob/c66ea840a62b4d3b9e6d32a12fc0942d00581dd9/tools/bench/results/qwen8b-target-batch-20260930/layers.csv)
are retained. Weight-dominated execution does not establish zero cost for extra
rows: the matrix path also pays operand conversion, accumulation, activation,
output and reduction costs. The measured projection gap remains the main target.

Additional tile/K-split/grid searches, weight prefetching, scale caching, alternate
NVFP4 decoders, wider matrix tiles, direct query tensors, parallel merge, fast math,
and alternate matrix modes did not establish a further reliable full-pass win.
Payload-ordered/all-block repacking helped both modes but left the parity gap.
Computing commuted RMS scalars in separate dispatches also did not improve latency.
These experiments are not shipped. Static T=8 programs run about 0.3–0.4 ms faster
than the actual N=7 prefix here; their lower measurements are not used as the result.

## Reproduction and evidence

[Summary](https://github.com/jiazhihao/mpk-apple/blob/c66ea840a62b4d3b9e6d32a12fc0942d00581dd9/tools/bench/results/qwen8b-target-batch-20260930/summary.json),
[paired samples](https://github.com/jiazhihao/mpk-apple/blob/c66ea840a62b4d3b9e6d32a12fc0942d00581dd9/tools/bench/results/qwen8b-target-batch-20260930/passes.jsonl),
and [compressed raw traces, inputs, cache choices and searches](https://github.com/jiazhihao/mpk-apple/blob/c66ea840a62b4d3b9e6d32a12fc0942d00581dd9/tools/bench/results/qwen8b-target-batch-20260930/raw.json.gz)
include the rejected experiments. Initial unspecialized grid prototypes in the raw
history are superseded by the final paired measurements; they are not acceptance evidence.

```sh
git show ade38fb5f81ebdf852a2b65a616703b03f4ec424:tools/bench/results/qwen8b-target-verify-20260930/inputs.json > /tmp/qwen8b-target-inputs.json
python tools/bench/target_batch_gap.py \
  --model /path/to/mlx-Qwen3-8B-nvfp4 --pack /path/to/target-pack \
  --drafter /path/to/mlx-community-Qwen3-0.6B-4bit --drafter-pack /path/to/draft-pack \
  --inputs /tmp/qwen8b-target-inputs.json \
  --out /tmp/target-batch-check --reps 10 --profile
```

Use a fresh output directory. Omitting the drafter arguments measures fixed-row
T=8 target programs instead of the actual speculative prefix. Both exclude draft
execution. The kernel change builds on `634ca76`; the benchmark disables only its
direct-KV selector when constructing the old-path control.
