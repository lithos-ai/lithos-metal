# Full-attention mixer optimization on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

**[M] The tuned full-attention mixer reduces complete-layer latency by 10.6%, 22.1%, 30.2%, 38.2%, 47.0% at 128 / 4K / 8K / 16K / 32K, respectively.** All 16 attention layers at all five tiers beat the fastest tested MLX-LM reference in every paired repetition (**720/720**).

Follow-up to the [task-based attention study](m5max-27b-attention-tasks.md) and
[issue #141](https://github.com/jiazhihao/mpk-apple/issues/141). Measurements on 2026-10-02 use
the 40-core M5 Max with 48 GB memory, macOS 26.5.1, and
`nvidia/Qwen3.8-27B-NVFP4` revision
`482ca0f3832238542f8f5295dde86b5f22711d80`.

N=7 means eight fixed input rows. Prefix lengths are 128, 4096, 8192, 16384 and
32768, with capacity fixed at 33024. The workload is fixed-position replay with
seeded nonzero inputs and KV prefixes. It does not measure speculative decoding,
generation throughput, prefill, sampling, embeddings or vocabulary projection.
Attention fusion remains an explicit benchmark/compiler option. The existing
GDN mixer default is preserved.

## Complete-layer confirmation

[M] Median across layers of each layer's minimum wall time, in microseconds:

| Prefix | Original | Previous mixer | New mixer | Matched control | Selected control | Fastest MLX-LM | New/original ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 628.09 | 649.98 | 561.80 | 562.61 | 553.83 | 999.45 | 0.8938 |
| 4096 | 852.79 | 863.08 | 663.27 | 654.46 | 645.03 | 1273.76 | 0.7786 |
| 8192 | 1064.44 | 1072.62 | 742.54 | 735.34 | 733.99 | 1597.39 | 0.6982 |
| 16384 | 1514.34 | 1490.04 | 934.22 | 975.20 | 926.15 | 2180.20 | 0.6177 |
| 32768 | 2487.33 | 2369.28 | 1319.01 | 1374.93 | 1333.61 | 3283.45 | 0.5298 |

The ratio column is the median of per-layer ratios, not a ratio of the independently aggregated latency columns. Every row represents 16 individual layers.

| Prefix | Layer minima beating original | Paired wins vs original | Paired wins vs fastest MLX | Per-layer new/original ratio range |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 16/16 | 144/144 | 144/144 | 0.8895–0.8987 |
| 4096 | 16/16 | 144/144 | 144/144 | 0.7628–0.7847 |
| 8192 | 16/16 | 144/144 | 144/144 | 0.6955–0.7148 |
| 16384 | 16/16 | 144/144 | 144/144 | 0.6145–0.6274 |
| 32768 | 16/16 | 144/144 | 144/144 | 0.5233–0.5427 |

The independently selected multi-dispatch control is competitive. Median new/control ratios are 1.0137, 1.0292, 1.0153, 1.0075, 0.9894, in increasing context order. Thus the improvement over original Monolith includes the better attention algorithm and projection layout; it is not all a benefit of combining dispatches. Differences inside 3% are treated as near parity.

## Isolated mixer

These measurements exclude the two native MLP projections and confirm the single-dispatch mixer on layers 3, 31 and 63 (nine repetitions by 32 replays). They are not a 64-layer streaming or generation measurement.

| Prefix | Original mixer | Previous mixer | New mixer | Matched control | Median new/original ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 306.38 | 331.36 | 243.36 | 243.57 | 0.7942 |
| 4096 | 534.32 | 548.16 | 337.56 | 339.46 | 0.6330 |
| 8192 | 750.72 | 756.54 | 437.60 | 425.23 | 0.5839 |
| 16384 | 1202.49 | 1179.40 | 624.22 | 660.37 | 0.5200 |
| 32768 | 2130.80 | 2007.12 | 962.44 | 1026.10 | 0.4516 |

## Selected configurations

Recipe files use content-based names, with identical contents sharing one file across both context and control maps. The five selected context recipes differ. These are explicit measured tiers, not an automatic interpolation or production routing policy.

| Prefix | Workers | SIMD groups | Query rows | Keys per local partition | Prepare Q/K | Compact partials | Tighter queue bound | Recipe |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 128 | 160 | 4 | 8 | 96 | no | yes | yes | [JSON](../../monolith/backends/metal/m5_max_40c/recipes/attention-optimization/attention-e686867e2c.json) |
| 4096 | 120 | 8 | 16 | 512 | no | no | no | [JSON](../../monolith/backends/metal/m5_max_40c/recipes/attention-optimization/attention-ee10038aa6.json) |
| 8192 | 120 | 8 | 16 | 1024 | no | no | no | [JSON](../../monolith/backends/metal/m5_max_40c/recipes/attention-optimization/attention-195c6979d4.json) |
| 16384 | 120 | 8 | 16 | 384 | no | no | no | [JSON](../../monolith/backends/metal/m5_max_40c/recipes/attention-optimization/attention-f0959614b4.json) |
| 32768 | 157 | 8 | 16 | 1024 | yes | no | no | [JSON](../../monolith/backends/metal/m5_max_40c/recipes/attention-optimization/attention-198c66a1ca.json) |

The JSON files carry the full projection overrides, packing, scalar/merge crews and barrier settings. [The context map](../../monolith/backends/metal/m5_max_40c/recipes/attention-optimization/selected-contexts.json) is the single lookup for the five measured tiers.

## Changes

The mixer still executes as **one megakernel dispatch**. Complete-layer timing
adds the two unchanged native MLP projections, for three dispatches versus nine
in original Monolith. The fused region includes input normalization/permutation,
QKV and gate projections, Q/K normalization and RoPE, causal attention and KV
append, partial-result merge and gating, output projection, and the residual
plus normalization/permutation boundary feeding the MLP.

The largest algorithmic change is local reduction of key tiles. Matrix operations
still score 16 or 32 keys at a time. A bounded loop combines several tiles using
FP32 online maximum, denominator and output accumulators before writing a single
global partial. Larger key partitions therefore reduce global partial traffic
and merge work without requiring proportionally larger threadgroup storage.
The loop retains the query tile across its local partitions.

The optional preparation stage normalizes and rotates Q/K and appends K/V once,
using a separate prepared-Q buffer. It preserves the original projection for
its gate consumer and fixed replay. Optional scratch aliasing reuses staged-K
storage for scores and V, with a threadgroup barrier protecting score readers.
The optional ordinary-load path reads only immutable KV-prefix addresses;
newly appended rows retain coherent accesses.

Private partial buffers can be sized to the selected global partition width.
Their compatible core/merge parameter records and every consuming kernel
specialization are updated together; foreign consumers, aliased bindings, and initialized/non-arena
workspaces are rejected. This changes addresses and allocation sizes without
changing the reduction order or context capacity.

Projection tuning uses exact FP8 payloads in a tile-oriented layout, with
independent geometry and traversal choices for the QKV, gate and output
projections. Payload decoding and tensor scales retain the checkpoint's
weight-only semantics. This is W4A16/W8A16, not NVIDIA's W4A4 activation path.

Persistent logical workers claim tasks from one atomic queue per readiness
phase. Initial assignments spread work before dynamic claims begin. Every queue
loop and global barrier is bounded. The tighter optional seeded-queue bound
avoids an unnecessary failed claim when initial assignments cover all tasks.
Mandatory threadgroup barriers still protect scratch between tasks. No
correctness property depends on physical core mapping or on a promised number
of resident threadgroups; a bounded barrier timeout rejects the configuration.

## Measurement and selection

The initial and adaptive screens use layer 3, randomized paired order, and
contemporaneous original Monolith, previous attention recipe, geometry-matched
multi-dispatch control and candidate megakernel measurements. Screening uses
three repetitions of eight replays; late layout refinements use seven by 24.
Finalists repeat nine by 32 on layers 3, 15, 31 and 63. Absolute-latency,
paired-ratio, median and maximum-time leaders enter confirmation, so a slow
baseline sample or lucky minimum cannot alone select a winner. Longer 32K
repeats exposed intermittent long latency with some 160-worker recipes despite
unchanged shader hashes. Selection therefore requires every paired confirmation
sample to beat original Monolith. Among recipes within 3% of the best eligible
minimum-time ratio, it retains those whose worst paired ratio is within 5% of
the best worst ratio, then selects the fastest remaining recipe. Complexity only
breaks exact ties. These consistency requirements were added after observing
the stalls. The initial preference for simpler recipes inside 3% was superseded
when repeated short/4K measurements supported retaining the faster projection
overrides. All raw samples and policy amendments are retained; the following
all-16-layer comparison is a separate confirmation run.

Final complete-layer confirmation covers all 16 attention layers (`3,7,...,63`)
at every context tier, with nine randomized paired repetitions of 32 replays.
Reported layer latencies are minima across repetitions; aggregate latency is
the median across layers. Ratios are computed per layer before aggregation.
Individual paired wins are reported separately from the minima.

Controls distinguish three questions: improvement over original Monolith,
improvement over the previous attention megakernel, and the effect of fusion
relative to the same optimized algorithm/layout. An independently selected
multi-dispatch control is also measured. A faster optimized control must not be
hidden or described as a megakernel win.

MLX runs the installed MLX-LM decoder layer with an exact-code ModelOpt adapter,
since this MLX-LM version does not directly load the NVIDIA checkpoint format.
NVFP4 uses native quantized matrix multiplication and its FP32 tensor scale;
FP8 is evaluated through both MXFP8 with unity block scales and materialized
BF16 weights. This is not a claim that the checkpoint loader is unmodified.

The MLX reference also checks cache evaluation/lifetime choices: stock-style
hidden-output evaluation, explicit state evaluation, and retaining state output
tuples until synchronization. Every timed run synchronizes the final output and
cache state. Both weight paths and all three choices are measured; each paired
comparison uses the fastest of these six MLX variants. The cache audit checks
bit-identical outputs across evaluation/lifetime choices. Cross-engine claims
use wall time, not Monolith's GPU timestamps.

GPU timing jobs run serially. Shader validation and task counters are used only
for correctness/distribution audits, never performance claims. One early overlap
was discovered: the last 20 rows of the initial 32K architecture screen were
discarded and rerun serially. The evidence records that exclusion and uses only
the repaired combined file. No thermal control is claimed.

## Search scope

This is a finite, adaptive search, not a proof over every possible kernel
algorithm or the full Cartesian product of all knobs. The evidence includes
explicit configuration lists, accepted and rejected points, source snapshots,
and generation scripts. The dense worker sweep evaluates every count from
1 through 256 for one selected recipe at each of 128, 8K and 32K.

The screen completed **4,902 configuration/context points** representing
**3,459 distinct explicit configurations**: 4,705 accepted and 197 rejected.
Rejections comprise 189 bounded barrier timeouts and eight invalid K-split
combinations. This count excludes shader smoke tests, final confirmations and
superseded measurements.

The separate four-layer finalist phase attempted 188 points: 184 passed and
four 8K trials with 240 workers hit the bounded barrier limit. That recipe was
excluded from selection. The final selected recipes passed all 80 complete-layer
points and all 15 isolated-mixer points.

| Phase | Measured points | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| Packing and initial projection geometry | 336 | 336 | 0 |
| Key partition and preparation grid | 930 | 895 | 35 |
| Architecture refinement | 542 | 508 | 34 |
| Scratch/key-tile memory variants | 255 | 255 | 0 |
| Projection refinement | 280 | 272 | 8 |
| 24-row query tiles | 90 | 88 | 2 |
| Per-projection tuning | 315 | 315 | 0 |
| Coupled projection combinations | 81 | 81 | 0 |
| Scheduling/synchronization | 969 | 969 | 0 |
| Every worker count 1–256, three tiers | 768 | 650 | 118 |
| Compact workspace / seeded-bound A/B | 40 | 40 | 0 |
| Direct 4K/16K grid | 296 | 296 | 0 |

Explored choices include:

- 1/2/4/8/16/32 SIMD groups, query tiles of 8/16/24/32 rows, physical key tiles
  of 16/32, and local key partitions from 32 through 32768 keys.
- Separate Q/K preparation, immutable-prefix loads, scratch aliasing, compact
  global partials, and an alternative independent-SIMD attention algorithm.
- FP8 layout, decode method, packing block, traversal order, operand storage,
  prefetch, matrix tile dimensions, K splits and per-projection combinations.
- Static/interleaved/queue schedules, task grain, claim batches, attention
  tiles per task, initial assignments, scalar/merge crews and merge unrolling.
- Bounded barrier implementations, polling crews, flag spacing, arrival method,
  inlining and read-only weight qualifiers.
- A direct partition/worker/preparation grid at 4K and 16K, including unchanged
  endpoint winners, followed by held-out layer confirmation.

Large claim batches and the alternative independent-SIMD attention path did not
improve the leading recipes. Wider query tiles and larger worker counts are
retained as measured alternatives or rejected configurations, not assumed to
benefit a 40-core GPU. Global search-optimality and performance at unmeasured
shapes, chips or context lengths are not established.

## Validation and evidence

**680 shared compiler, attention, GDN, default-selection, MLP and MLX-adapter tests passed** under Metal shader validation. The optional HTTP-serving contract module was skipped because FastAPI is absent. All 80 real-layer shader audits also passed. The checks require control/fusion bit identity
for output and state, fixed replay identity, native comparison cosine at least
0.9999 with relative L2 below 0.005, and layer cosine above 0.999 against MLX.
Kernel tests cover cache boundaries through 32767, causal append/prefix
preservation, local softmax stress ranges, task totals, workspace compaction,
shared specialization isolation and repeated execution.



[M] Across the final real-layer checks, the lowest oracle cosine is 0.99962717. Instrumented task audits record logical worker assignments at each tier; they do not establish physical threadgroup-to-core mapping. Unit tests verify exact task totals and seeded assignment semantics.

The preliminary seed-batch assertion was corrected to account for more than one task per initial assignment. A real-model smoke test also exposed distinct core/merge parameter records; compaction now checks compatible workspace geometry and updates both, with added CPU and shader regressions. Superseded failures remain in the archive and are not performance evidence.

[The evidence directory](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/attention-optimization) contains all-layer CSV/JSON summaries, deduplicated recipes, search coverage, numerical validation and source hashes. Its compressed raw archive includes every paired sample, generated-shader hashes, configuration list, rejected trial, exclusion, source snapshot and reproduction script. It excludes model weights and derived weight caches. Reproduction scripts use local checkpoint/pack paths and require serial GPU execution.

## Logical worker audit

[M] The separate shader-instrumented audit records the following attention-core
work. These counters cover queued stages; an initial nonqueued norm/permute
stage can have zero counters while still executing. Gate projection tasks may
share the attention readiness phase, so attention-only active counts need not
reach the full worker count at short contexts.

| Prefix | Logical workers | Attention tasks | Workers handling attention | Attention tasks per worker |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 160 | 48 | 48 | 0–1 |
| 4096 | 120 | 108 | 98 | 0–2 |
| 8192 | 120 | 108 | 102 | 0–3 |
| 16384 | 120 | 516 | 107 | 0–7 |
| 32768 | 157 | 396 | 157 | 1–7 |

At 4K–16K, all 120 workers process QKV projection tasks (512 total, three to
five each). At 32K, the output projection distributes 160 tasks across all 157
workers. Task counts differ because tasks and sibling operations have different
costs. This demonstrates dynamic assignment and coverage, not equal execution
time per worker or a measured physical core mapping.
