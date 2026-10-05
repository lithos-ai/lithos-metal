# Qwen3.8-27B + DSpark on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

Measured 2026-10-03 on Apple M5 Max, 40 GPU cores, 48 GB, macOS 26.5.1.
All latency figures in this report are measurements [M].

This report preserves the initial integration and first tuning sweep. The
[draft-kernel refinement](m5max-27b-dspark-refinement.md) contains the latest
selected BF16 recipes, paired latency improvements and precision experiments.

The RadixArk DSpark checkpoint runs in the GPU-resident Monolith speculative
pipeline. A complete round verifies the anchor plus seven proposals, accepts a
prefix, commits the target's recurrent state, and generates the next seven
proposals. The measurements below include that entire round, including both
vocabulary projections, the sequential Markov corrections, confidence scoring
and verification-length selection.

## Checkpoints and integration

- Target: `nvidia/Qwen3.8-27B-NVFP4`, revision
  `482ca0f3832238542f8f5295dde86b5f22711d80`. The existing weight-only
  NVFP4/FP8 target pack and BF16 activations are unchanged.
- Drafter: [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark),
  revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
  `model.safetensors`: 3,714,723,322 bytes, SHA-256
  `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`.
  Original BF16 draft weights throughout; no draft requantization.
- Five attention layers, hidden 5120, intermediate 17408, 32 query heads,
  eight KV heads, head dimension 128, Markov rank 256. Target taps are layers
  5, 19, 33, 47 and 61; mask ID 248070; seven proposals per round.
- The checkpoint omits embedding and vocabulary-head weights. The drafter now
  binds the target embedding and uses the existing shared target vocabulary head.
- Added checkpoint-defined YaRN frequencies **and amplitude scaling**: factor 32,
  original context 8192, theta 1e7, beta-fast 32 and beta-slow 1. NumPy packing and
  the torch oracle follow the same tables, checked against Hugging Face.
- Matrix draft attention preserves all three KV sources: cached context,
  newly injected target features, and the bidirectional proposal block. Only
  injected context is appended to the persistent draft cache.

## Full-round measurements

| Context | Original complete round | Tuned complete round | Tuned wall median | Speedup |
|---|---:|---:|---:|---:|
| 128 | 84.45 ms | **50.68 ms** | 51.08 ms | 1.67× |
| 4K | 90.78 ms | **52.14 ms** | 53.13 ms | 1.74× |
| 8K | 96.25 ms | **53.75 ms** | 54.95 ms | 1.79× |
| 16K | 108.59 ms | **57.02 ms** | 58.35 ms | 1.90× |
| 32K | 140.58 ms | **64.16 ms** | 65.52 ms | 2.19× |

These are non-terminal rounds: no final-request shortcut skips drafting. Seven
warm measured samples follow three warmups. Each replay restores the same
StepState and both GDN state slots after real prompt prefill. The unchanged KV
prefix stays on the GPU; current-step tail entries are overwritten by the round.
Full ICB replay, including kernel boundaries, is the primary latency measure.
Compilation, prefill, checkpoint copying and host-side fixture preparation are
outside the measurement. Wall latency includes submission and completion.
The table reports the initial sweep. Later 11-sample confirmations measured
52.76 ms at 128 tokens and 70.08 ms at 32K, showing approximately 4–9% variation
between runs. All samples are retained alongside the initial sweep; the table
should not be read as a latency guarantee.

The original-kernel baseline uses the same checkpoint compatibility changes and
weights, with the original profile's BF16 threshold of eight rows, scalar draft
attention and unfused target/draft regions. The tuned configuration uses the
previous target mixer/MLP recipes plus the draft changes described below.
Thus this is a complete-pipeline comparison, distinct from the previous report's
sum of isolated target-layer times.

| Context | Verification | Accept/commit | Draft | Tuned full GPU range |
|---|---:|---:|---:|---:|
| 128 | 38.03 ms | 1.30 ms | 10.66 ms | 48.99–51.79 ms |
| 4K | 39.34 ms | 1.30 ms | 11.08 ms | 51.52–55.71 ms |
| 8K | 41.35 ms | 1.33 ms | 11.58 ms | 53.50–56.42 ms |
| 16K | 43.35 ms | 1.32 ms | 12.43 ms | 56.95–57.09 ms |
| 32K | 48.69 ms | 1.30 ms | 14.25 ms | 64.10–64.41 ms |

Stage columns are separate ICB measurements over the **same dispatches**. They
need not sum exactly to the full-round measurement. Split and unsplit rounds
produce identical committed tokens and next-block proposals in the replay checks.
The timing fixture is the retained deterministic repeated-prose prompt at each
context length. Its first measured round accepts all seven proposals; these
acceptances must not be extrapolated to other workloads.

## Draft kernel graph


The orange boundary is one mixer megakernel dispatch. Each of the five draft
layers keeps its context-K/V projection outside that boundary and its MLP in
two native kernels. The shared vocabulary projection feeds seven sequential
Markov positions, each containing gather, W2 projection and two argmax kernels.
This grouped graph shows the current 128/32K selection with 68 draft dispatches.
At 4K–16K, each mixer uses five native dispatches, for 88 draft dispatches total;
the surrounding data dependencies are unchanged. An [editable SVG](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/docs/research/figures/dspark-current-kernel-graph.svg)
is generated by `tools/bench/plot_dspark_kernel_graph.py`.

## Draft kernel breakdown

A profile of the first-sweep recipes uses the same checkpoint bytes, real prompt
prefill and eight newly committed feature rows. It captures every draft dispatch
with GPU timestamp counters (seven measured passes after three warmups). The
table reports **means across those captures**, not a rescaling of the earlier
10.66/14.25 ms sweep. Each row aggregates the indicated kernels across all five
draft layers or seven Markov steps.

| Draft work | Dispatches or repetitions | 128 context | 32K context |
|---|---:|---:|---:|
| MLP gate + up projections / activation | 5 native kernels | 3.25 ms | 3.44 ms |
| MLP down projections / residual output | 5 native kernels | 1.71 ms | 1.78 ms |
| Attention mixer megakernels: QKV, attention, merge, output | 5 megakernels | 1.26 ms | 6.19 ms |
| Shared target vocabulary projection, plus input permutation | 1 projection | 1.96 ms | 2.06 ms |
| Markov W2 vocabulary corrections | 7 sequential projections | 1.68 ms | 1.75 ms |
| Tapped-feature projection | 1 active projection | 0.61 ms | 0.73 ms |
| Context KV projections | 5 active projections | 0.58 ms | 0.77 ms |
| Norms, embeddings, argmax, confidence, selection and glue | remainder | 0.34 ms | 0.42 ms |
| **Sum of kernel attribution** | **68 encoded dispatches** | **11.39 ms** | **17.14 ms** |
| **Uninstrumented whole draft, mean in this run** | one ICB | **11.43 ms** | **16.99 ms** |

Six encoded projection variants return immediately at this injected-row count;
their measured dispatch costs remain included. Timestamp sampling needs one
encoder per dispatch on this device, so the breakdown is an attribution study,
not a replacement for whole-ICB latency. A second check partitions the original
ICB into contiguous spans while preserving operation order; its summed means
were 11.67/16.28 ms. Every counter pass and split pass produced the same next
proposals as normal replay. Generated kernel hashes match the original sweep.
Normal whole-draft ranges in this follow-up were 10.96–11.85 ms and
15.92–17.73 ms, respectively. The earlier sweep was faster; the cause of the
between-run variation has not been isolated.

Why drafting is substantial:

- This head is a five-layer, approximately 1.86-billion-parameter **BF16** model.
  Its MLPs retain BF16 weights while the target MLPs use NVFP4. Counting layers
  alone therefore understates the draft's work. The five MLPs process 2.674 GB
  of packed matrix weights per block and account for **43.5%** of short-context
  attributed time.
- The shared vocabulary projection still maps seven hidden rows of width 5120
  to a vocabulary of 248320, using a 763 MB NVFP4 matrix. Sharing storage does
  not remove that computation.
- Each rank-256 Markov correction projects to the entire vocabulary. W2 is
  127 MB in BF16 and is applied **seven times sequentially**, because the next
  correction depends on the previous selected token. These are full-vocabulary
  projections, not seven small scalar corrections.
- The main matrix payloads total **5.218 GB per draft block** when repeated
  Markov applications are counted. This is logical operand volume, not a DRAM
  counter measurement; it excludes cache effects, activations and small auxiliary
  tensors. Exact byte counts are in `draft-weight-payloads.json`.
- At 32K, the five draft layers hold approximately 671 MB of K/V values before
  accounting for repeated reads across query tiles. Mixer time rises by 4.94 ms,
  accounting for **85.9%** of the measured short-to-long-context increase.

Per-layer means (mixer and both MLP kernels only):

| Draft layer | Mixer, 128 | Gate/up, 128 | Down, 128 | Mixer, 32K | Gate/up, 32K | Down, 32K |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.270 | 0.625 | 0.322 | 1.215 | 0.734 | 0.363 |
| 1 | 0.237 | 0.624 | 0.322 | 1.324 | 0.692 | 0.346 |
| 2 | 0.258 | 0.732 | 0.402 | 1.234 | 0.670 | 0.343 |
| 3 | 0.246 | 0.631 | 0.336 | 1.197 | 0.647 | 0.324 |
| 4 | 0.244 | 0.635 | 0.328 | 1.224 | 0.700 | 0.401 |

All per-layer values are milliseconds. Full names, geometry, raw per-dispatch
samples, independent ICB spans and output checks are retained in
[`draft-profile-128.json`](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-dspark/draft-profile-128.json)
and [`draft-profile-32768.json`](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-dspark/draft-profile-32768.json).
Add `--profile-draft` to the reproduction command below to repeat the study.
The largest remaining opportunities are reducing BF16 draft weight volume
(which requires acceptance/correctness evaluation), improving the vocabulary
projections, and reducing long-context attention traffic. Launch fusion alone
does not remove these costs.

## Configuration search

The searches covered 36 initial mixer recipes, 54 MLP recipes, 24 long-context
attention refinements, and four combined finalists. They varied workers
(40–320, depending on kernel kind), SIMD groups (4/8/16), projection output tiles
(16/32), reduction splits, native reduction tiles (64/128/256), queue/stage
scheduling, attention partitions and query tiles. The complete raw trials,
including six invalid K-split configurations, are retained in
[the results directory](../../tools/bench/results/m5max-27b-dspark).

The draft mixer megakernel contains the block QKV projection, matrix attention,
partial reduction, output projection and residual/normalization output. Context
KV injection is prepared before that region so its variable accepted-row count
and one-row fallback retain their predicates. Feature projection, shared LM head,
Markov chain and confidence/selection operations remain separate dispatches.

Selected draft settings:

| Context | Workers | SIMD groups | QKV/output TN | K split | Attention partition | Query rows/tile |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 80 | 4 | 32 | 4 | 32 keys | 16 |
| 4K–32K | 160 | 8 | 32 | 8 | 256 keys, accumulated as eight 32-key tiles | 16 |

Both use bounded, seeded task queues. Long contexts additionally cache reads
from the immutable KV prefix and compact the partial workspace. Draft MLPs use
**two native kernels**, with 80 groups, four SIMD groups, TN16, TK128 and K-split
four. Fused MLPs were tested; they did not win the final combined comparison.
The BF16 accelerator threshold is two rows in the measured tuned profile, so
seven-row draft projections use tensor operations.

Fusion alone is not the whole speedup. The matched normalized controls were
competitive with fused mixers, and sometimes faster in the layer screen.
Most draft improvement comes from matrix attention, projection geometry and
MLP tuning. Raw control/fusion timings are retained rather than attributing every
gain to removing dispatch boundaries.

A final full-round comparison alternated 11 pairs with identical tuned tasks,
shared resident weights and only draft mixer fusion changed:

| Context | Unfused tuned tasks, median (range) | Draft megakernels, median (range) |
|---|---:|---:|
| 128 | 52.83 ms (52.11–53.75) | 53.04 ms (51.76–53.70) |
| 32K | 69.45 ms (68.52–70.19) | 69.38 ms (67.75–70.76) |

The overlapping ranges do **not** establish a speedup from fusion alone.
Minima favor fusion slightly, while medians are effectively tied. The selected
megakernel recipe remains available with 313 full-round dispatches versus 328
for the matched unfused draft control. Both produced identical committed tokens
and next proposals in every pair. This control retains the tuned target
megakernels; it isolates draft mixer fusion only.

The target uses GDN mixer megakernels, full-attention mixer megakernels, and the
previously selected two native MLP kernels. Explicit decoder recipes now support
the dynamic verification program, preserving active-row predicates and GDN
commit outputs. Exact per-context target and draft settings are in the single
[previous-selected-contexts.json](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-dspark/refinement-20261003/previous-selected-contexts.json).

## Correctness and memory

Validation completed:

- 132 relevant CPU contract tests passed, including shared embeddings, YaRN
  frequencies/amplitude against Hugging Face, weight-window compaction and the
  generation API.
- Under Metal shader validation, 12 draft oracle cases passed across scalar,
  matrix and fused attention, with and without YaRN and confidence selection.
  Each case exercises injected-row counts 3, 2, 1 and 8 over successive steps,
  checking hidden states, logits, Markov corrections, confidences and KV updates.
- Four dynamic target GDN recipe cases and 14 target attention regression cases
  passed under shader validation. Dynamic recipes preserve active-row behavior
  and recurrent commit state.
- Complete generation with the real checkpoints produced **identical 64-token
  outputs and acceptance histories** to the original kernels for each of four
  code, math, chat and text prompts: 256 tokens over 72 speculative steps.
  The histories include zero through seven accepted proposals and exercise
  rejection/commit paths and repeated requests in the same session. These are
  regression checks, not an acceptance-quality evaluation.
- Full and split ICB timing replays produce identical committed tokens and
  next proposals at every measured context. The layer screen additionally
  requires cosine similarity above 0.9999 and relative L2 error below 0.005
  against the production program, and byte equality against matched normalized
  controls.

The initial final GPU run exposed a variable-shadowing bug in four newly added
test fixtures; renaming the recipe variable fixed it, and all 12 draft cases
passed on rerun. Failed trials and their resolutions are retained in
`rejected-trials.json`; shader-validation logs and generation comparisons are
saved with the measurements.

Two integration failures were diagnosed and corrected before measurement:

1. Matrix-attention partials at 128-token prefill and 33K capacity allocated
   82.1 GB. `prefill_attention='v3'` selects the lower-memory prefill path while
   verification retains matrix attention.
2. Derived packed projections plus original two-GB weight windows required
   48.2 GB of bound buffers. Compaction retains only original entries still used
   after repacking, without changing tensor bytes, reducing the bound program
   to approximately 30.7 GB. Already fused bindings are protected because they
   carry internal offsets. The session releases prefill allocations before
   decode and retains persistent caches, StepState, ring and acceptance logs.
   CPU programs remain cached between requests.

The study does not optimize prefill/TTFT. Allocating the verification program
and changing resident weight layouts occur before the timed rounds. Instrumented
Metal shader-validation timings are excluded from performance results.

## Reproduce

Pack the drafter with the normal packer; it detects the missing shared embedding:

```sh
.venv/bin/python tools/pack_weights.py \
  --model /tmp/monolith-models/Qwen3.8-27B-DSpark \
  --out /tmp/monolith-m5max/dspark/pack-bf16-33k \
  --drafter-kind dspark --max-context 33024 --scale-placement block
```

Measure a complete tuned round (replace context and configuration key together):

```sh
.venv/bin/python tools/bench/dspark_round_latency.py \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 \
  --pack /tmp/monolith-m5max/attention-tasks/pack-33k \
  --drafter /tmp/monolith-models/Qwen3.8-27B-DSpark \
  --drafter-pack /tmp/monolith-m5max/dspark/pack-bf16-33k \
  --profile monolith/backends/metal/m5_max_40c/config.json \
  --inputs tools/bench/results/m5max-27b-dspark/inputs.json \
  --config monolith/backends/metal/m5_max_40c/recipes/dspark/selected-contexts.json \
  --config-key 128 --contexts 128 --mode optimized \
  --out /tmp/dspark-round-128.json
```

For the original kernel baseline, omit `--config` and `--config-key`, and use
`--mode original`. `--check-generation` additionally generates 64 tokens for
code, math, chat and story prompts and records the actual acceptance history.
Raw full-round results record generated-kernel source hashes, the input hash,
all latency samples, proposals and committed tokens.
`--compare-draft-fusion` adds alternating paired full-round measurements with
identical tuned tasks and shared resident weights, changing only draft mixer
fusion. It verifies equal committed tokens and next proposals on every run.
