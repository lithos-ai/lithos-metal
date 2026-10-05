# Commuted input normalization on M5

Relaxed-rounding input-normalization fusion is enabled by default in generation,
`Session`, and the compiler APIs. Use `--no-commute-norm` or `commute_norm=False`
to restore the non-commuted path; `--commute-norm` remains accepted. Existing packs
work unchanged. Latency improves on some shapes and regresses on others, as recorded below.

For reciprocal RMS `r(h) = rsqrt(sum(h*h)/K + eps)`, a projection can use
`r(h) * ((h * gamma) @ W.T)`. This uses the same algebra as
[MPK's normalization fusion](https://github.com/lithos-ai/mirage/blob/5beaed87bbf3ec341bde00917c400c411e3f6a33/docs/mpk/decode_linear.md).
MPK folds gamma into weights; this implementation instead writes `BF16(h * gamma)`
from the preceding residual projection, in the consumer's packed input order.
The consumer applies `r(h)` to its FP32 accumulator before SiLU/multiplication or
residual addition. The existing raw BF16 residual and its norm statistics remain intact.
No external implementation was copied.

The compiler removes the intervening norm/permutation dispatch for eligible
2–8-token tile consumers. Matching siblings share the new scratch; different
norm weights retain separate paths. The first embedding-fed normalization,
one-token decoding, larger prefill tiles and mixed shader/tile schedules retain
the original path. The producer must also be a tile covering the full token range.

Moving BF16 rounding changes results: this is an explicit exception to the
original leaf-ULP/token-equality gate for eligible fused projections. The explicit
non-commuted path retains its prior numerical contract.
Tests check the reordered formula, unchanged residual/statistic outputs, distinct
norm weights, barriers and shrinking/growing active rows. Layer cosine below is
measured against MLX on each layer's actual Monolith input; it does not establish
whole-generation token equality or task quality.

## Initial measurements

[M] Apple M5 Pro, 20 GPU cores, 24 GB, macOS 26.5.1, MLX 0.32.2 / mlx-lm 0.31.3.
All checkpoint layers execute in a dependency chain with distinct weights, fixed
context 128, random BF16 input/KV prefix, and unchanged checkpoint weights. Times
are wall microseconds per layer (stack latency divided by layer count), not
isolated kernel times or acceptance-dependent generation throughput. `T=N+1`.
The harness specializes each program to T; live speculative sessions may compile
a larger row bound, so these timings do not establish a generation speedup.

Each result uses nine paired repetitions of 48 replays, rotating through the six
orders of original Monolith, fused Monolith and MLX. Compilation/first-touch is
warmed outside timing; shader validation is disabled. Values are minimum [range]
over repetitions. Raw samples, cosine and settings are in
[the result file](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_commute-norm.jsonl).

| Model | N | Original µs [range] | Fused µs [range] | MLX µs [range] | Fusion change | Min layer cosine |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B INT4 | 3 | 71.30 [71.30–76.28] | 69.51 [69.51–76.10] | 71.98 [71.98–76.35] | -2.52% | 0.999970 |
| Qwen3 0.6B INT4 | 5 | 77.36 [77.36–77.46] | 75.07 [75.07–75.16] | 87.53 [87.53–87.77] | -2.95% | 0.999965 |
| Qwen3 0.6B INT4 | 7 | 79.87 [79.87–80.26] | 76.37 [76.37–76.44] | 103.28 [103.28–103.50] | -4.39% | 0.999965 |
| Llama 3.2 3B INT4 | 5 | 306.07 [306.07–317.74] | 291.54 [291.54–302.27] | 456.16 [456.16–458.50] | -4.75% | 0.999992 |
| Llama 3.2 3B INT4 | 7 | 293.44 [293.44–305.04] | 308.95 [308.95–323.35] | 566.66 [566.66–574.73] | +5.29% | 0.999992 |
| Llama 3.2 1B INT4 | 5 | 179.91 [179.91–180.26] | 182.27 [182.27–182.73] | 280.70 [280.70–281.58] | +1.32% | 0.999993 |
| Llama 3.2 1B INT4 | 7 | 183.80 [183.80–184.20] | 189.70 [189.70–189.89] | 343.39 [343.39–345.40] | +3.21% | 0.999992 |
| SmolLM2 1.7B BF16 | 5 | 499.27 [499.27–531.17] | 495.36 [495.36–525.13] | 491.14 [491.14–515.60] | -0.78% | 0.998086 |
| SmolLM2 1.7B BF16 | 7 | 499.29 [499.29–535.91] | 494.67 [494.67–533.73] | 496.77 [496.77–506.70] | -0.92% | 0.999867 |
| Qwen3 8B NVFP4 | 5 | 440.40 [440.40–475.23] | 445.02 [445.02–492.03] | 646.22 [646.22–661.04] | +1.05% | 0.999975 |
| Qwen3 8B NVFP4 | 7 | 440.15 [440.15–514.04] | 447.65 [447.65–532.55] | 825.42 [825.42–860.21] | +1.70% | 0.999976 |

In this initial sweep, Qwen 0.6B at N=5/7 and Llama 3B at N=5 have non-overlapping original/fused ranges. Qwen 0.6B N=3 and SmolLM2 show overlapping ranges, so their lower minima are not conclusive wins. Llama 1B, Llama 3B N=7 and Qwen 8B do not benefit from enabling this option. SmolLM2 N=5 also remains slower than MLX by the measured minimum.

The fusion removes 55 normalization dispatches on the 28-layer models, 31 on Llama 1B, 47 on SmolLM2 and 71 on Qwen 8B. Its added producer stores and repeated per-tile statistic reductions can outweigh the saved dispatches. Moving the reciprocal computation after MMA and using adjacent four-lane reductions reduced the preliminary 8B regression, but did not eliminate it. Further optimization and shape-aware selection are tracked in [#131](https://github.com/jiazhihao/mpk-apple/issues/131).

Reproduce with the same target checkpoint and pack:

```sh
python tools/bench/layer_fixed_vs_mlx.py --model CHECKPOINT --pack PACK \
  --ts 6,8 --ctx 128 --reps 9 --steps 48 --norm-ab --min-cosine 0 --out results.jsonl
```

`--norm-ab` compares original/fused/MLX in one process. `--min-cosine 0` explicitly
requests finite-output sanity while recording the measured cosine; the default
remains 0.999. `--commute-norm` without `--norm-ab` compares only fused Monolith
against MLX. No claim is made that every individual layer beats MLX.

## Why Qwen 8B regresses

[M] Follow-up at T=8/context 128, all 36 layers, using the **same legacy
block-scale/lane-order NVFP4 pack as the original comparison**. This is not a
claim about every pack layout. The fresh paired run reproduces the regression:
462.21 → 467.27 µs/layer (+1.10%).
[Raw samples, profiles and native metadata](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_commute-norm-diagnosis.jsonl).

The dominant cost is repeating the RMS-statistic reduction in every output tile:

- Hidden width 4096 produces 256 partial sums per token. QKV has 384 output tiles;
  gate/up has 1,536. Each tile independently reduces the same token statistics.
- At T=8, a typical interior layer therefore reads **15 MiB of logical statistic
  data**, versus **256 KiB** in the original two normalization/permutation
  dispatches (16 SIMD-groups per token each): **60×**. These are logical loads,
  largely cache-reused, not measured DRAM traffic.
- Qwen 0.6B has 64 partials, 256 QKV tiles and 384 gate/up tiles: only **1.25 MiB**
  of fused statistic reads. The 8B case does **12×** as much of this work while
  still removing just two dispatches.
- The added reduction runs after the K-split join, in the sole writing SIMD-group
  of each eight-SIMD-group threadgroup. It extends the projection's final phase.

Separate-encoder timestamp profiles localize the cost (mean µs/layer, five-run
per-op minima; their sums are not an ICB wall-time measurement):

| Work | Original | Fused | Change |
|---|---:|---:|---:|
| Normalization/permutation | 9.83 | 0.13 | -9.69 |
| QKV | 56.38 | 60.42 | +4.03 |
| Gate/up | 205.93 | 214.94 | +9.01 |
| Output + down producers | 158.29 | 160.13 | +1.84 |

Controlled fixed-replay ablations preserve the final output **bit-for-bit**.
Twelve rotated/reversed repetitions × 48 replays give these minima:

| Fused path control | µs/layer | Saving from fused |
|---|---:|---:|
| Full fusion | 464.78 | — |
| Reuse the already-computed gamma-scaled inputs; omit producer work | 463.07 | 1.71 |
| Reuse exact FP32 sums; replace 256 partials with one sum per token | 455.50 | 9.28 |
| Both controls | 452.05 | 12.74 |

The cached controls are diagnostic only: arbitrary new inputs require new inputs
and sums. In a valid prototype that **recomputes** each sum once on the GPU every
replay, the extra dispatch/barriers are included: original 460.03, fused 464.99,
compact-sum 459.76 µs/layer. Ranges overlap (original 460.03–464.05, compact
459.76–473.66), so this establishes approximate break-even, not a reliable win.

Native archives also argue against spilling as the main explanation. QKV's
experimental register field changes 82 → 84; gate/up's 83 → 85. Both scratch
fields remain zero, and shared memory stays 3,584 bytes. Compact sums retain the
**same** 84/85 register fields while recovering most of the time. Native code
sizes are QKV 4,422 → 5,134 → 4,650 bytes and gate/up 6,668 → 7,388 → 6,904 bytes
(original → fused → compact). No decoded M5 instruction counts or exact occupancy
are inferred from these fields.

The next optimization should amortize statistic reduction across output tiles
and avoid paying a replacement dispatch for each input. Removing gamma stores
alone does not address most of the loss. The arithmetic identity is valid; this
particular placement duplicates too much work on the wider model.

## Persistent projection crews

[M] The fused consumer now caches its reciprocal RMS after its first output tile
and reuses it across the existing grid-stride loop. The measured legacy NVFP4
geometry (K=4096, TK=128, eight-way K split, T=6/8) uses **two threadgroups per
GPU core**, or 40 on this M5 Pro. There is no new dispatch or scratch buffer.
Other geometries retain their previous schedule. This fusion is now enabled by default.

Exactly one group per core was tested first. With eight SIMD-groups/group it
slowed T=8 to **581.14 µs/layer**, versus **493.86** for the old fusion and
**482.28** for 40 groups in the same sweep (six rotations × 24 replays).
Larger 20-group variants, including 16-way K splitting and multiple independent
crews per group, also lost. Merely reducing the launch count to 20 is insufficient:
the projection must retain enough parallel work. The selected 40-group schedule
reduces an interior layer's RMS folds from 384 + 1,536 to 40 + 40: **24× fewer**,
or 640 KiB of logical statistic loads instead of 15 MiB at T=8. This is not a
measurement of physical memory traffic.

Final implementation, all 36 layers, context 128, same pack/input as above;
rotated paired runs, 48 replays per sample, minimum wall µs/layer:

| N | Repetitions | Unfused | Old fusion | Persistent fusion |
|---|---:|---:|---:|---:|
| 5 | 12 | 493.11 | 510.48 | 504.19 |
| 5, confirmation | 6 | 506.69 | 495.57 | 487.43 |
| 7 | 12 | 492.08 | 495.87 | 482.83 |

The new schedule beats the old fusion in 11/12 pairs at both N=5 and N=7;
median paired reductions are 1.32% and 1.53%. N=7's minimum is 2.63% below the
old fusion and 1.88% below unfused. N=5's first unfused minimum is inconsistent
with its other samples, so the warmed confirmation is included in full rather
than replacing that run. These measurements support an improvement over the
old fusion, but not a universal or noise-free win over unfused execution.

The separate MLX-resident comparison gives persistent/MLX minima of
491.16/723.12 at N=5 and 493.06/914.26 at N=7, with substantial outliers in all
arms. Every layer's output is **bit-identical to the prior fused schedule**;
minimum per-layer cosine against MLX remains 0.999975/0.999976. The 12 fusion
kernel cases plus the MLX format contract pass with Metal shader validation,
including unequal tile counts, padded T=6 scratch and changing active rows.

[Raw grid sweeps, all paired samples and profiles](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_commute-norm-grid.jsonl).
The existing `--norm-ab --ts 6,8 --ctx 128` command above reproduces the current
persistent/unfused/MLX comparison; use revision `ff69a7c` for the prior fusion.

## Grid search across the remaining Qwen 8B operations

[M] On revision `f3105e0`, the same legacy pack was used to screen **388 launch
configurations** across nine roles at T=6/8, followed by 130 amortized rechecks
of the single-operation roles and 56 full-stack comparisons (including unchanged
controls). The full 36-layer stack remained the acceptance test. Grid counts
covered 20/40/80/160 groups and one group per output tile; GEMM K splits covered
2/4/8/16 plus unsplit 8/12/16-SIMD-group crews. Attention compute additionally
covered 48 groups and 2/4/8/16 SIMD-groups. Merge/statistic packing and permutation
slices/unrolling were also swept.

| Role | Result |
|---|---|
| Interior QKV and gate/up | Retain 40 groups and K split 8. |
| Initial QKV | Retain the existing tile grid; cached leaf timings did not establish a stack win. |
| Attention output projection | Smaller grids gave about 1% leaf gains, but no robust general stack gain. |
| Down projection | 20 groups/K split 16 and 40 groups/K split 8 gave 1–3% leaf gains; full-stack/context checks rejected a general change. |
| Attention compute, including Q/K norm and RoPE | 16 SIMD-groups helped short context, but the tested attention combination regressed about 2.3% at context 1024. Retain 8. |
| Attention merge | **Pack four independent SIMD-groups per threadgroup at T=8** for the tested 32-head/8-KV-head, D=128 commuted-norm schedule. |
| Initial normalization statistic | Retain one SIMD-group per token. |
| Initial normalization/permutation | Wider slices/unrolling choices helped the isolated dispatch, but it executes only once per stack; no additional robust stack gain. |

Only merge packing is added to the production schedule. It keeps all 256 logical
SIMD-groups at T=8, but launches 64 threadgroups of 128 threads instead of 256
threadgroups of 32. The arithmetic and buffer addressing are unchanged. T=6 and
the explicitly disabled, non-commuted path keep their prior geometry.

Final N=7 comparisons, eight paired repetitions × 48 replays, minimum wall
µs/layer (paired median ratios are also shown):

| Context | Previous | Packed merge | Median new/previous | Unchanged-control ratio |
|---|---:|---:|---:|---:|
| 128 | 438.89 | 437.30 | 0.99637 | 0.99892 |
| 1024 | 480.76 | 479.54 | 0.99745 | 0.99831 |

The candidate wins all eight pairs at each context, but the unchanged control
also shows a timing bias. After that control, the supported improvement is small,
roughly **0.1–0.3%**. The earlier larger combination's N=5 gain did not repeat;
it is not enabled. Bootstrap intervals in the raw data are exploratory and do
not correct for searching many configurations.

All 518 leaf measurements remained finite (minimum cosine 0.999999952); changing
K splits can alter rounding. The retained merge configuration preserves every
checkpoint layer bit-for-bit. Fourteen attention checks pass with Metal API
validation, including new changing-row/cache cases at contexts 0/128/1024.
Three standalone merge tests pass full shader validation, including permuted
output and empty/partial rows. Full shader instrumentation of the unchanged MMA
core exceeds the device's threadgroup-memory budget (45,056 > 32,768 bytes), so
the merge's buffer checks are exercised separately. The search harness also
passes a real-model T=6/8 merge sweep with API validation; 248 contract checks pass.

[All configurations, discarded noisy samples, paired controls and context checks](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_qwen8b_grid_search.jsonl).
Reproduce the screen with `tools/bench/layer_grid_search.py` as described in
[the benchmark README](../../tools/bench/README.md). Leaf timings alone are not
used to claim a decoder-layer or generation speedup.
