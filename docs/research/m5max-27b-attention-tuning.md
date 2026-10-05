# Attention megakernel tuning at N=7 on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

The subsequent [task-based 4K–32K study](m5max-27b-attention-tasks.md) adds
runtime work queues and context-adaptive task sizes. The measurements below
describe the preceding static-schedule experiment.

**[M] Attention fusion remains an opt-in experiment.** Further tuning reduces
the earlier megakernel's median complete-layer latency by 1.1% at a 4096-token
prefix and 2.4% at 8192, but the selected configurations are still about 3%
slower than original Monolith. These small improvements are mostly within the
study's 3% noise margin. Native attention remains the production default; the
previously selected GDN mixer/native-MLP default is unchanged.

This is the attention follow-up to [the GDN/default study](m5max-27b-megakernel-tuning.md)
for [issue #141](https://github.com/jiazhihao/mpk-apple/issues/141), measured
2026-10-01 on the **40-core M5 Max**, using
`nvidia/Qwen3.8-27B-NVFP4` revision
`482ca0f3832238542f8f5295dde86b5f22711d80`.
N=7 means eight fixed input rows in this experiment. No speculative-decoding,
token-generation, drafting, acceptance, embedding or vocabulary-projection
performance was measured.

## Complete-layer comparison

The attention mixer becomes one dispatch; the original two MLP projections
remain native. This reduces a full attention layer from nine to three
dispatches. The fused region includes input normalization/permutation, QKV and
gate projections, Q/K normalization and RoPE, causal attention and KV append,
partial-result merge and gating, output projection, and the residual plus
normalization/permutation boundary feeding the native MLP.

[M] Median across the 16 attention layers of each layer's minimum wall time
over nine randomized paired repetitions, 32 fixed-input steps per repetition:

| Prefix tokens | Original Monolith | Earlier attention hybrid | Selected attention hybrid | Faster MLX-LM path |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 626.73 µs | same geometry | 645.80 µs | 1218.43 µs |
| 4096 | 850.98 µs | 889.57 µs | 879.89 µs | 1446.03 µs |
| 8192 | 1084.97 µs | 1141.63 µs | 1118.15 µs | 1771.24 µs |

The earlier hybrid is rebuilt by the same compiler in each long-context paired
round, using 80 workers, eight SIMD groups and 160 attention task groups. The
selected long-context hybrid uses 160 workers, four SIMD groups and 320 attention
task groups. Both preserve the native MLP and use the same original weight pack.

The median **per-layer** selected/original ratios are **1.0302, 1.0339 and
1.0295**, respectively. Across all 48 cases, the ratio ranges from 0.9949 to
1.0543; only one minimum is below original, by 0.5%, and none beats original
by the 3% margin. The new long-context geometry has a lower minimum than the
earlier hybrid in 29/32 cases; four of those improvements exceed 3%.

At 8192 tokens, the matched nine-dispatch control takes a median **1117.32 µs**,
almost identical to the fused path's **1118.15 µs**, versus original Monolith's
**1084.97 µs**. This suggests that the remaining gap includes the common
geometry/cooperative task implementation itself. Changing barrier settings
alone did not remove it in the search.

Both original Monolith and the selected attention hybrid beat the faster MLX-LM
reference in **every one of the 432 paired samples**. The selected hybrid's
per-case minimum latency is **0.528–0.645× MLX-LM** (1.55–1.89× throughput for
this fixed work). This satisfies the measured layer comparison against MLX-LM,
but does not establish a performance win from attention fusion over original
Monolith.

MLX-LM 0.32.0 / MLX 0.32.3 run the stock layer code. The reference is the faster
of the two FP8 handling paths at each point, with native MLX NVFP4 matrix
operations. See [the initial study](m5max-27b-megakernel.md) for conversion,
precision and timing methodology. Every comparison uses the same checkpoint,
nonzero seeded inputs and KV prefixes, BF16 residuals and fixed positions.

## Search and retained controls

Selection used layer 3; confirmation covered all 16 attention layers
(`3,7,...,63`) at prefixes 128, 4096 and 8192. Timings exclude shader
instrumentation. GPU jobs ran serially.

- **49 geometry trials / 98 points:** four worker counts (40, 80, 120, 160),
  four/eight SIMD groups, eight/sixteen query rows per matrix tile and
  80/160/320 attention task groups, plus the earlier configuration.
- **62 refinement trials / 124 points:** merge subgroup counts, independent
  QKV/gate/output K splits, and per-projection task counts. The requested
  `merge_unroll` values in this screen were overridden by the shader's built-in
  definition, so all ran with unroll four. They are **not evidence for different
  unroll factors**. The compact evidence records requested and effective values
  separately.
- **30 corrected confirmation trials / 60 points:** actual merge unroll
  1/4/8/16/32, two merge subgroup counts, serial/SIMD/leader barriers,
  task barriers, noinline tasks and interleaved sibling scheduling. Seven
  paired repetitions and 24 steps per sample. Large merge unrolls substantially
  regressed the long-context cases.

These are **141 configuration trials**, including repeated/equivalent
configurations, rather than 141 distinct kernels. All 282 timing points passed
the matched-control correctness screen. A matched control retains nine
dispatches while using the candidate's tile geometry; this distinguishes the
effects of geometry from fusion.

Retained opt-in settings:

| Setting | Short prefix (128) | Long prefixes (4096/8192) |
| --- | ---: | ---: |
| Fixed workers | 80 | 160 |
| SIMD groups per worker | 8 | 4 |
| Attention task groups | 160 | 320 |
| Query tile rows | 16 | 16 |
| Active merge SIMD groups | 8 | 4 |
| Merge unroll | 4 | 4 |
| Projection output tile | 16 | 16 |
| Projection K split | 8 | 4 |

Both use compact eight-row partials, SIMD-parallel bounded global barriers and
omit the redundant final task barrier. Configurations are separate benchmark
inputs, **not an automatic context-length policy**; intermediate contexts and
a production switching threshold have not been selected.

The experimental compiler now exposes `attention_groups`, `attention_qm`,
`merge_sgs` and `merge_unroll`. Attention stride specialization and its parameter
record change together. A preprocessing contract test verifies that the merge
unroll actually reaches the task body. The benchmark can pair two mixer-prefix
geometries with `--reference-fusion-scope mixer-prefix`.

A separate larger-chunk cooperative K/V experiment was rejected. MPP rejected
input cooperative tensors spanning multiple SIMD groups; the subsequent
per-SIMD version failed numerical checks or bounded barriers for larger chunks.
That implementation was removed from the retained compiler. Its failed logs and
source snapshot remain in the raw evidence; its instrumented times are not
performance results.

## Correctness and evidence

All **48 full-layer timed cases** pass the 0.999 cosine gate; the minimum checked
cosine is **0.9996271690** across hidden outputs and appended KV comparisons.
Matched-control and fused outputs/state are bit-identical. Each case verifies
the existing KV prefix is unchanged, compares appended K/V with MLX-LM, and
checks fixed-input output replay after timing.

The new attention GPU tests cover both query tiles, both worker SIMD widths,
different task counts and merge settings, cache boundaries, changing positions
0/8/31/39/127/4095/8191, consumption of prior appends, unchanged cache prefixes
and suffixes, and 32 repeated invocations.

Final checks:

- **48/48 real-layer shader audits** passed under
  `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`.
  No shader validation diagnostics were reported. These timings were excluded.
- **270 tests passed, one skipped:** 15 shader-validated attention/GDN-default
  GPU regressions plus 255 contract tests. The first sandboxed contract attempt
  could not initialize Metal for its MLX test; the complete suite passed when
  rerun with GPU access.
- Repository hygiene and `git diff --check` passed.

Compact evidence lives in
[`tools/bench/results/m5max-27b-n7/attention-tuning/`](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/attention-tuning):
`all-layers.csv`, all 432 repetitions in `paired-samples.csv`, the effective
search settings in `tuning.csv`, selected configs, `summary.json` and
`metadata.json` with source/raw hashes. The raw archive includes source
snapshots, full numerical checks, failed experiments and logs.
The local archive is `/tmp/monolith-m5max/attention-tuning-evidence.tar.gz`;
its SHA-256 and size are recorded in `metadata.json`.

## Reproduce

Run GPU commands serially, with shader validation disabled for timing.
`MODEL` and `PACK` denote the checkpoint and its original packed weights;
`EVIDENCE=tools/bench/results/m5max-27b-n7/attention-tuning`.

```sh
python tools/bench/modelopt_mega_tune.py --model MODEL --pack PACK \
  --kind attention-prefix --ctx 128,8192 \
  --configs "$EVIDENCE/merge-confirm-configs.json" --reps 7 --steps 24 \
  --out merge-confirm.jsonl

python tools/bench/modelopt_layer_bench.py --model MODEL --pack PACK \
  --layers 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 --ctx 128 \
  --fusion --fusion-scope mixer-prefix --config "$EVIDENCE/short-config.json" \
  --reps 9 --steps 32 --fp8-mode both --fail-on-regression --out all-short.jsonl

python tools/bench/modelopt_layer_bench.py --model MODEL --pack PACK \
  --layers 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 --ctx 4096,8192 \
  --fusion --fusion-scope mixer-prefix --config "$EVIDENCE/long-config.json" \
  --reference-config "$EVIDENCE/short-config.json" \
  --reference-fusion-scope mixer-prefix --reps 9 --steps 32 \
  --fp8-mode both --fail-on-regression --out all-long.jsonl
```

`--fail-on-regression` checks the MLX comparison; it does not assert a win over
original Monolith. Reproduce shader audits with
`MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`,
`--reps 1 --steps 1`, no `--fail-on-regression`, and separate output files.
The historical refinement screen needs the recorded effective-unroll correction
to reproduce its executed configurations with the fixed compiler.
