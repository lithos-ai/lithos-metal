# Task-based attention megakernels at 4K–32K on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

The [full-attention optimization follow-up](m5max-27b-attention-optimization.md)
extends this historical recipe with local key partitions and further projection,
memory-layout and scheduling searches.

**[M] The task-based attention hybrid improves the median per-layer latency by
4.9% versus original Monolith at 32K.** All 16 attention layers have lower
minimum latency there; 15 improve by more than the study's 3% noise margin.
At 4K, 8K and 16K the median differences are within that margin. Attention
fusion remains opt-in; the existing GDN mixer/native-MLP default is unchanged.

Follow-up to [the static attention study](m5max-27b-attention-tuning.md) and
[issue #141](https://github.com/jiazhihao/mpk-apple/issues/141), measured
2026-10-01 on the **40-core M5 Max / 48 GB**, using
`nvidia/Qwen3.8-27B-NVFP4` revision
`482ca0f3832238542f8f5295dde86b5f22711d80`.
N=7 means eight fixed input rows. These are isolated, complete attention-layer
replays with seeded nonzero inputs and KV prefixes. They exclude speculative
decoding, generation, sampling, embeddings and vocabulary projection.

## Complete-layer results

Confirmation covers all 16 attention layers (`3,7,...,63`) at each tier. Each
point uses nine randomized paired repetitions, with 32 fixed-input steps per
repetition. The original program, prior static megakernel, geometry-matched
multi-dispatch control, new queue hybrid, and both MLX-LM FP8 handling paths
are measured contemporaneously. MLX-LM runs its stock layer with native NVFP4
matrix operations; the reference is the faster of its MXFP8 and materialized
BF16 projection paths. See [the initial study](m5max-27b-megakernel.md) for the
weight conversion and numerical methodology.

[M] Median across layers of each layer's minimum wall time:

| Prefix | Original Monolith | Prior static hybrid | Queue hybrid | Faster MLX-LM path | Median queue/original ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 849.24 µs | 877.46 µs | 862.78 µs | 1442.88 µs | 1.0160 |
| 8192 | 1064.07 µs | 1093.12 µs | 1073.58 µs | 1714.43 µs | 1.0093 |
| 16384 | 1513.40 µs | 1527.72 µs | 1493.38 µs | 2184.68 µs | 0.9875 |
| 32768 | 2462.54 µs | 2455.11 µs | 2377.88 µs | 3302.85 µs | 0.9505 |

The ratio column is the **median of paired per-layer ratios**, not the ratio
of the independently aggregated latency columns. The two differ, particularly
at 32K, because layer latencies vary. Against the previous static hybrid, the
median per-layer reductions are 1.5%, 1.9%, 2.1% and 2.9%, respectively.

At 32K the per-layer queue/original ratios range from **0.9468 to 0.9813**.
The queue also wins 141/144 individual repetitions against original. Three
repetitions regress, so this is not an every-repetition win against original
Monolith. At 16K, 14/16 layer minima improve and none clears 3%; at 4K and 8K,
no layer minimum improves over original. All **576/576 paired samples** at the
four requested tiers beat the faster MLX-LM path, for both original Monolith
and the queue hybrid.

The matched nine-dispatch control has median latencies of 883.32, 1094.70,
1514.58 and 2386.44 µs. The improvement therefore reflects the combined
geometry, scalar subgroup and scheduling choices; it cannot all be attributed
to the atomic queue alone.

The separate 128-token sanity case takes 650.12 µs with the queue versus
629.54 µs original and 645.85 µs with the prior short-context static hybrid.
Its median per-layer regression versus original is 3.3%. It passes all 144
MLX paired comparisons, but supplies no performance reason to replace native
short-context attention.

## What changed

The experimental compiler exposes output-tile teams and bounded groups of
attention tiles as tasks. Persistent threadgroups claim work from an atomic
counter for each readiness phase. QKV projection, attention/gate siblings,
merge and output projection remain ordered by the existing bounded device
barriers. Attention and gate tasks share a queue, interleaved without padding
when their counts differ. Each worker receives an initial task, then claims
more work dynamically. Counters are initialized once at kernel entry and
published before use; an active queue is never reset.

An attention tile still computes the same 32-key chunk and retains its original
reduction order. A task groups contiguous tiles to amortize queue claims. The
group size is selected from the actual context at invocation:

```text
tiles_per_task = min(configured_max, max(1, total_tiles // (2 * workers)))
```

This leaves at least two waves of tasks when sufficient work exists, and
shrinks to one tile at short contexts. It does not partition physical GPU
cores or require a particular threadgroup-to-core mapping. GEMM parameter
records and specializations become private per operation, so the QKV and gate
segments can share source kernels while retaining independent task strides.
All task-claim loops and global barriers are bounded.

The attention mixer is still one dispatch followed by the two original native
MLP projections: **three dispatches per layer, versus nine in original
Monolith**. The fused region includes input normalization/permutation, QKV and
gate projections, Q/K normalization and RoPE, causal attention and KV append,
partial-result merge and gating, output projection, and the residual plus
normalization/permutation boundary feeding the MLP.

## Configurations and search

Selection used layer 3. Each requested tier was selected independently; all four
converged on the same recipe in the measured candidate set. One shared
`attention-config.json` serves 4K, 8K, 16K and 32K. This is an opt-in
benchmark selection, not an automatic production routing policy.

| Setting | 128-token sanity case | 4K / 8K / 16K / 32K |
| --- | ---: | ---: |
| Persistent workers | 80 | 160 |
| SIMD groups per worker | 8 | 4 |
| Maximum attention tiles per task | 8 | 3 |
| Tasks per atomic claim | 1 | 1 |
| Initial task per worker | yes | yes |
| Active scalar/merge SIMD groups | 8 | 2 |
| Projection output tile | 16 | 16 |
| Projection K split | 8 | 4 |
| Query tile rows / merge unroll | 16 / 4 | 16 / 4 |

All use compact eight-row partials and the SIMD-parallel bounded global barrier.
The queue requires a threadgroup barrier between tasks to protect shared
scratch. `attention_groups=320` is retained for the matched multi-dispatch
control; the tile queue uses its runtime task count instead of that static grid.

There were **100 configuration trials / 327 context points**, including repeated
and equivalent configurations. All passed the geometry screen: bit-identical
output/state against the matched control, cosine at least 0.9999 and relative
L2 below 0.005 against original Monolith.

| Search | Trials | Points | Repetitions × steps | Main variables |
| --- | ---: | ---: | ---: | --- |
| Initial queue | 23 | 69 | 5 × 16 | 40–160 workers, 4/8 SIMD groups, static versus queue, claim batches 1/4/16 |
| Adaptive tile groups | 50 | 150 | 5 × 16 | 1/3/8/16/32 tiles per task, scalar/merge groups, worker counts through 200, initial task assignment |
| Confirmation screen | 9 | 36 | 9 × 24 | 128–200 workers, initial assignment, noinline, scalar groups |
| Four-tier selection | 18 | 72 | 7 × 24 | 4K/8K/16K/32K independently, K splits, scalar/merge groups and task settings |

Large claim batches regressed because workers retained too much work. Grouping
a small number of attention tiles inside one task was more effective than
claiming many heterogeneous tasks at once. Partial K splits and noinline did
not displace the selected recipe. GPU jobs ran serially, with shader validation
disabled for performance measurements.

## Worker distribution

Optional `task_stats` counters record completed tasks per worker. The audit
uses shader validation and is excluded from timing results. The heavy QKV,
attention-plus-gate and output phases used every configured worker in all five
selected cases:

| Prefix | Workers | QKV tasks/worker | Attention tasks total | Attention + gate tasks/worker | Output tasks/worker |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 80 | 6–7 | 60 | 5–7 | 4 |
| 4096 | 160 | 3–4 | 516 | 5–7 | 2 |
| 8192 | 160 | 3–4 | 1028 | 8–10 | 2 |
| 16384 | 160 | 3–4 | 2052 | 14–16 | 2 |
| 32768 | 160 | 3–4 | 4100 | 27–29 | 2 |

Gate projection has 384 tasks in each case; QKV has 512 and output projection
320. Small scalar phases have fewer tasks: the selected long-context merge has
96, so it cannot occupy all 160 workers. The initial normalization runs
statically and intentionally has zero queue counters.

These are **worker task counts, not GPU occupancy or per-core timing**. Attention
and projection tasks also have different costs, so equal counts alone would
not prove equal elapsed work. The dynamic queue lets workers claim available
tasks as they finish; the wall-time comparisons establish its measured benefit.

## Correctness and final checks

All **64/64 timed tier cases** and the 16 short sanity cases pass the layer
cosine gate of 0.999. The minimum checked cosine is **0.9996271690**, across
hidden outputs and appended KV comparisons. Matched-control and fused
output/state are bit-identical. Existing KV prefixes remain unchanged, and
fixed-input replays reproduce the checked outputs after timing.

- **64/64 real-layer shader audits** passed at 4K/8K/16K/32K under
  `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`, without
  shader diagnostics. Their instrumented times are excluded.
- **10/10 instrumented worker-distribution audits** passed: both short and
  long configurations at 128/4K/8K/16K/32K.
- **301 tests passed, one skipped:** 30 attention GPU cases, nine GDN-default
  GPU regressions and 262 contract tests, all in the shader-validation process.
  Attention tests cover multiple query tiles, SIMD widths, merge settings,
  claim batches, adaptive tile groups and initial assignment; changing
  positions across chunk boundaries; consumption of prior appends; unchanged
  cache prefixes/suffixes; bounded-barrier status; and 32 repeated invocations.
- Python compilation, repository hygiene and `git diff --check` passed.

The measured production decision is unchanged: use native attention by
default and retain these queue configurations for explicit experiments.
The long-context result is a useful improvement at fixed T=8, but it does not
establish a whole-model or speculative-decoding speedup, nor justify replacing
native attention at every context length.

## Capacity and reproducibility

All four requested tiers use a fixed Monolith KV/RoPE capacity of **33024**. The
original weight pack had tables for 8704 positions. A copy-on-write APFS clone
preserves its tensor bytes and appends larger RoPE tables whose original
prefixes are byte-identical. The new
`tools/bench/modelopt_extend_tables.py` utility reproduced the extended manifest
byte-for-byte. The original pack is unchanged. MLX-LM uses its usual
context-plus-256 cache allocation; every measured invocation consumes the same
prefix and appends eight rows.

The 128-token sanity confirmation predates the fixed-capacity flag and used
capacity 384. Treat it as a separate short-context check. Its control and queue
still share the same allocation and inputs. The geometry screens use capacity
33024 because each includes the 32K point.

`MODEL` is the NVIDIA checkpoint, `PACK` its original Monolith pack, and
`EXTENDED` a new directory. Run GPU commands serially. Use the shared checked-in
attention config for all four tiers:

```sh
python tools/bench/modelopt_extend_tables.py --model "$MODEL" --pack "$PACK" \
  --out "$EXTENDED" --max-context 33024

EVIDENCE=tools/bench/results/m5max-27b-n7/attention-tasks
PREVIOUS=tools/bench/results/m5max-27b-n7/attention-tuning
for CTX in 4096 8192 16384 32768; do
  python tools/bench/modelopt_layer_bench.py --model "$MODEL" --pack "$EXTENDED" \
    --layers 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 \
    --ctx "$CTX" --capacity 33024 --fusion --fusion-scope mixer-prefix \
    --config "$EVIDENCE/attention-config.json" \
    --reference-config "$PREVIOUS/long-config.json" \
    --reference-fusion-scope mixer-prefix --reps 9 --steps 32 \
    --fp8-mode both --fail-on-regression --out "all-$CTX.jsonl"
done

python tools/bench/modelopt_mega_tune.py --model "$MODEL" --pack "$EXTENDED" \
  --kind attention-prefix --ctx 4096,8192,16384,32768 \
  --configs "$EVIDENCE/tier-screen-configs.json" --reps 7 --steps 24 \
  --out tier-screen.jsonl
```

`--fail-on-regression` checks original and fused Monolith against MLX-LM; it does
not assert a win over original Monolith. Shader audits use
`MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`,
`--reps 1 --steps 1`, no performance gate, and separate output files. The
`balance-configs.json` file adds `task_stats=true` for worker-count audits.

Compact evidence is in
[`tools/bench/results/m5max-27b-n7/attention-tasks/`](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/attention-tasks):
per-layer minima, every paired sample, search settings, per-worker counters,
selected configurations, and summaries with source/raw hashes. The local raw
archive is `/tmp/monolith-m5max/attention-tasks-evidence.tar.gz`; it contains
source snapshots, all samples, test logs and pack manifests, excluding checkpoint
weights. Its SHA-256 and byte count are recorded in `metadata.json`.
