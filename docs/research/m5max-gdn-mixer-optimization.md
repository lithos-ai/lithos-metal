# GDN mixer optimization on the 40-core M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

[M] Completed 2026-10-02. The optimized GDN mixer is installed in
[`apple-m5-max-40c.json`](../../monolith/backends/metal/m5_max_40c/config.json). Across all 48
GDN layers, its complete-layer wall time is 7.20% lower than
original Monolith and 6.58% lower than the previous mixer
default on average. Every layer passes the MLX-LM comparison in every paired
round. The independently tuned packed multi-dispatch control is essentially tied:
the default/control mean latency ratio is 1.0014. The matched
packed control is about 0.62% faster. The gain cannot
be attributed entirely to merging dispatches; this study does not establish
a fusion advantage over those controls.

This follows the [earlier tuning](m5max-27b-megakernel-tuning.md) and
[issue #141](https://github.com/jiazhihao/mpk-apple/issues/141). The target is
`nvidia/Qwen3.8-27B-NVFP4`, revision
`482ca0f3832238542f8f5295dde86b5f22711d80`, on the 40-core M5 Max with 48 GB,
macOS 26.5.1, MLX 0.32.3 and MLX-LM 0.32.0. N=7 denotes a fixed eight-row
pass. These are layer and decoder-chain measurements; no drafting, sampling,
generation throughput or speculative-decoding performance is evaluated.

## Selected configuration and scope

The mixer contains input normalization/permutation, QKV/AB/Z projections,
convolution and its state, Q/K normalization, GDN recurrence and its state,
gated normalization, output projection and residual addition. It retains the
producer of the native MLP's normalized input. The two original MLP kernels
remain, reducing a standalone GDN layer from 11 dispatches to three.

The selected recipe has 80 workers × 8 SIMD groups, TN=32, compact partials,
eight-column recurrence slices and eight active scalar SIMD groups. Lossless
FP8 tile packing uses `q_outer=0`, subtraction decoding and tile-block=8.
Input permutations use 64 active SIMD groups, direct normalization and dual
permutation when those operations are present. The projection tasks are:

| Projection | TN | K split | Logical grid |
|---|---:|---:|---:|
| QKV | 32 | 2 | 240 |
| AB (BF16) | 16 | 8 | 240 |
| Z | 32 | 8 | 120 |
| Output | 32 | 4 | 120 |

The logical projection grids are serviced by the 80 persistent workers; they
are not GPU core counts. The stage scheduler won over the tested queue and
interleaving variants. The SIMD barrier remains bounded, with device fences
and explicit timeout propagation. Attention configuration is unchanged.

A preceding native residual can already produce the normalized FP8 input
layout and statistic. The compiler skips direct-normalization/dual-permutation
transforms at those boundaries. Consequently, standalone layer improvements
must not be multiplied by layer count to predict a decoder-chain gain.

Automatic selection remains limited to Apple10, the exact measured shape
`[5120,16,48,128,128,4]`, fixed T=8, commuted normalization, accelerator-enabled
FP8/BF16 projections and the native MLP boundary. Dynamic and speculative
programs retain native dispatches. GDN has fixed recurrent state, so this
recipe does not require context-specific 4K/8K/16K/32K files.

## Complete-layer measurements

Each of the 48 real GDN layers ran nine randomized paired rounds of 32
fixed-input replays after warm-up. The table reports the arithmetic mean of
each layer's minimum wall time in microseconds. Both MLX-LM FP8 paths use the
checkpoint's original codes; the faster of MXFP8 and BF16-decoded FP8 is the
comparison baseline. Timed execution excludes compilation and layout caching.

| Engine | Mean layer wall time (µs) |
|---|---:|
| Original Monolith | 646.25 |
| Previous mixer default | 642.00 |
| Optimized automatic default | 599.72 |
| Explicit optimized fusion | 598.86 |
| Matched packed multi-dispatch | 596.02 |
| Independently tuned packed control | 598.87 |
| Faster MLX-LM path | 1050.12 |

Ratios below one favor the automatic optimized default. Per-layer minima and
per-round wins are reported separately; the strict MLX gate uses the faster
MLX path in each paired round, not only its global minimum.

| Comparator | Mean ratio | Layer ratio range | Layers faster | Paired wins |
|---|---:|---:|---:|---:|
| Original | 0.9280 | 0.9242–0.9324 | 48/48 | 431/432 |
| Previous default | 0.9342 | 0.9094–0.9436 | 48/48 | 432/432 |
| MLX-LM (faster path) | 0.5714 | 0.5447–0.6081 | 48/48 | 432/432 |
| Matched packed control | 1.0062 | 1.0028–1.0111 | 0/48 | 24/432 |
| Independently tuned packed control | 1.0014 | 0.9926–1.0076 | 17/48 | 187/432 |

All automatic outputs/states match explicit fusion bit for bit. The candidate's
matched control also matches fusion bit for bit. Both packed controls preserve
native MLP kernels. Their results show that lossless operand packing and
projection geometry supply much of this improvement; fusion does not establish
a speed advantage over the same packing and projection optimizations.

## Isolated mixer and decoder chain

The isolated mixer excludes the MLP and its complete-layer boundary. These
independent nine-round × 32-replay measurements report wall time in µs:

| Layer | Original | Previous mixer | Optimized mixer | Matched packed control |
|---|---:|---:|---:|---:|
| 0 | 328.95 | 322.97 | 285.38 | 280.68 |
| 1 | 327.05 | 321.39 | 283.76 | 279.47 |
| 30 | 327.28 | 319.55 | 283.16 | 279.36 |
| 62 | 328.85 | 321.36 | 283.63 | 279.65 |

The chain contains consecutive real decoder layers, including intervening
native attention and MLPs. It excludes embedding and language-model head.
The five-layer test uses seeded synthetic inputs/states. The 64-layer test starts
with zero state and consumes four consecutive eight-token chunks from a fixed
32-token prompt, using checkpoint embeddings. Position advances 0/8/16/24; the
last chunk is replayed for timing. Both tests use nine paired rounds of 16
replays after the four changing-input/state steps.
Minimum wall times are in milliseconds. These are actual chained measurements,
not sums of standalone layer timings.

| Layers | Original | Previous default | Optimized | Optimized/original | Minimum hidden/state cosine |
|---|---:|---:|---:|---:|---:|
| 5 | 3.151 | 3.329 | 2.991 | 0.9493 | 0.9999743 |
| 64 | 40.116 | 42.034 | 38.155 | 0.9511 | 0.9999095 |

All nine finalists were also checked on the five-layer chain, with changing
state and nine paired timing rounds. The selected recipe's ratio to the prior
default is 0.8984; the best finalist's ratio is
0.8984 (index 2). This checks the boundary mode that
already receives normalized input instead of assuming isolated-layer rankings
transfer unchanged. All nine pass the chain correctness and replay gates.

## Search coverage and rejected approaches

7,105 trials were completed: 6,358 accepted timed trials and
747 rejections. There are 6,326 unique accepted configuration
JSONs. Generated hashes cover 5,220 accepted trials and
5,072 distinct generated sources; earlier phases predate hash logging.
The rejection set is 723 bounded barrier timeouts, 12 threadgroup-memory
compiler failures and 12 unsupported state-slice proposals. None is counted
as a valid performance sample.

| Domain | Trials | Accepted | Rejected |
|---|---:|---:|---:|
| Geometry and occupancy | 2930 | 2259 | 671 |
| Projection grids and splits | 952 | 952 | 0 |
| Operand layouts, decoding and prefetch | 862 | 862 | 0 |
| Scheduling and synchronization | 506 | 453 | 53 |
| Structural and local refinement | 1487 | 1464 | 23 |
| Final combinations | 368 | 368 | 0 |

The search includes workers up to 256 and every integer near competitive
occupancy levels; SIMD counts 1–32 including non-powers of two; cooperative
and staged tile shapes; per-projection grids, splits, decoding and traversal;
recurrence token passes, state slices, preparation, vector transfers and
unrolling; scalar crews; seven FP8 decoders; tile layouts and interleaving;
prefetch; exact predecoded FP16/BF16 storage; stage/interleave/queue schedules;
task grain, batching and seeding; polling width, flag spacing and arrival
updates; restricted pointers; direct normalization and dual permutation.
Adaptive phases combined the strongest SG=4/8/16 families and projection
subsets, then independently repeated nine finalists on layers 0,1,30,62.

The selected finalist won all 36 pairs against the previous recipe, with
geometric-mean wall ratio 0.9372 (0.9306 against original). Simple packing of
the previous geometry reached 0.9749 against that baseline, so the later
geometry/input-transform tuning adds a repeatable improvement. Very large
worker populations often timed out; non-power crews, staged K=64/128/256,
prefetch and doubled-byte FP16/BF16 weights did not beat the selected recipe.

A separate compile-only probe checked 48 M/N/K descriptors. Seven compiled:
each dimension is 16 or 32, with at least one 32. All seven cooperative shapes
were included in the search; larger K tiles use the separate staged path.
The probe creates no command queue and is excluded from performance counts.

This exhausts the recorded finite adaptive search, including its final
combination pass. It is not a proof over every Cartesian product or every
possible new kernel algorithm. Sub-3% screening differences were not promoted
without independent repeats. The packed-control advantage is reported even
though it is small.

## Correctness and evidence

587 shared GDN/default/attention/MLP and contract tests passed with Metal shader
validation. All 48 real GDN layers passed a separate instrumented audit, and a
five-layer chain passed with changing input/state. The uninstrumented all-layer
run checks hidden output, convolution state and recurrence state against
MLX-LM (cosine ≥0.999); candidate hidden output against original also requires
cosine ≥0.9999 and relative L2 <0.005. The minimum cosine over the complete
all-layer audit is 0.99988335. Fixed-input output and original model-state
snapshots remain bit-identical after timing, including both chain lengths.
Instrumented timings are not performance evidence.

The synthetic 64-layer stress case **failed** the end-to-end cosine >0.999
criterion for both the previous and selected recipes (worst hidden cosine
0.9910 and 0.9920 respectively). It produced no accepted timing result. The
real-prompt chain above passes the unchanged gate, but that does not erase the
synthetic failure or establish token-level equivalence for arbitrary inputs.
A separate per-layer diagnostic feeds the original chain's inputs and states
into the exact optimized chain segments. All 256 local checks pass the stricter
cosine ≥0.9999 / relative L2 <0.005 gate, with minimum cosine
0.99999918. Scratch is captured before each layer
runs, and native TK=256 normalized layouts are translated to TK=32 for the
47 hoisted GDN boundaries before copying them. Results are retained in
`chain64-diagnostic.json`. This distinction between local accuracy and
accumulated full-depth sensitivity is a remaining numerical limitation.

`fp8_tiles.py` copies original FP8 codes into cooperative operand order without
re-quantization. Content-addressed, page-aligned derived files live outside the
repository; the original checkpoint and pack are unchanged. Layout tests cover
120 mappings across lane orders, tiles, reduction sizes and traversal choices.

[Per-layer CSV](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/gdn-optimization/all48.csv),
[summary](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/gdn-optimization/all48.summary.json),
[coverage](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/gdn-optimization/coverage.json) and
[raw evidence](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/gdn-optimization/raw-evidence.tar.gz)
retain exact configurations, paired samples, rejections, validation logs,
source snapshots/hashes and reproduction scripts. The archive contains no
checkpoint, original pack or derived weight binaries. Extract `study/` and
review its `run_final_validation.sh` and `chain_compare.py`; paths target the
local model/pack and should be adjusted for another machine. The profile is
specific to this measured chip and shape.
