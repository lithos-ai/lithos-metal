# Qwen3.6-35B-A3B NVFP4

Experimental text-generation support for
[`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
is registered under the checkpoint's `Qwen3_5MoeForConditionalGeneration`
architecture. The model package is `monolith/models/qwen3_5_moe`.

The 40 decoder layers contain 30 Gated-DeltaNet mixers and 10 gated full-attention
mixers. Every layer has 256 routed experts, selects eight per token, and adds a
sigmoid-gated shared expert. Routed and shared intermediate widths are 512;
the residual width is 2048. The package composes existing library modules and
uses existing compiler/kernel paths. It does not apply the dense 27B model's
shape-specific tuning recipes to these different shapes.

The NVIDIA checkpoint stores separate expert matrices, NVFP4 experts/head,
and FP8 mixer projections. Their quantized codes/scales are preserved, using
the engine's weight-only BF16-activation arithmetic. Native HF checkpoints with
fused 3-D expert parameters are not supported by this adapter. Vision and MTP
weights are omitted from the text graph. Ordinary decode and the experimental
Koopah DSpark pairing described below are available.

## CLI

```bash
.venv/bin/python -m monolith.serve \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --max-context 32768 --port 8000
```

The Hugging Face resolver downloads the checkpoint and the ordinary pack cache
creates its text pack automatically. A local checkpoint directory is also
accepted. `--pack` can point to an existing compatible pack or a cache root.
The default served alias for the repository ID is `Qwen3.6-35B-A3B-NVFP4`.

## Validation on October 4, 2026

[M] M5 Max, 40 GPU cores, 48 GB; checkpoint revision
`1355db6a052410cfd62085d94b58866fd0f2c3c5`. The download contains 23,424,338,320
bytes of safetensors; the text pack occupies 20,997,521,408 bytes. All 31,333
claimed text tensors match the model's expected shapes. The pack reserves
33,024 context positions for the measurements below.

* Synthetic two-layer tests cover both mixer types and shared/routed experts,
  BF16 and mixed NVFP4/FP8 storage, static T=1/T=8 and dynamic compilation,
  independent HF eager oracles and GPU execution with Metal shader validation.
* A lazy expert-storage HF reference is bit-identical to materialized HF
  on the synthetic hybrid, including recurrent continuation. It changes
  parameter storage only; HF's routing/attention/expert arithmetic is retained.
* The real model matches all 16 HF greedy continuation tokens for
  `The capital of France is`, on two identical GPU runs.
* Each of the 40 real decoder layers passes cosine >0.999 against HF when
  given the same HF input on that five-token prefill; minimum **0.999798**.
  This is not a long-context or T=8 accuracy qualification.
* The full chain has a stricter outstanding qualification gap: minimum cosine
  across the first 39 decoder outputs is **0.998293**, and last-token logit
  cosine is **0.999134**. Of 18 differing route selections in the isolated-layer
  check, 17 have an exact HF tie at the eighth/ninth expert boundary. The
  remaining difference has a one-BF16-step reference margin. These observations
  indicate routing sensitivity, but do not prove all accumulated error is due
  to ties. Further multi-prompt and long-context qualification is needed.
* Two real HTTP Chat Completions calls return HTTP 200 and
  `The capital of France is Paris.` The warmed call takes 440 ms wall time
  for a 24-token chat prompt and eight generated tokens including EOS.
  This smoke test is not a serving throughput benchmark. The current runtime's
  step counter also counts stopped pump replays, so its response step-time
  header is not interpreted as latency per generated token here.

## Initial target-forward latency

[M] Existing 40-core backend defaults, automatic per-op tuning disabled as in
the server, no manual shape/context overrides. Static one/eight-token full
forwards include embeddings, all decoder layers, final norm and vocabulary head.
They exclude sampling, draft generation, acceptance, prefill and HTTP overhead.
The fixtures have fixed random BF16 KV prefixes and zero input recurrent state;
they measure kernel cost, not a real speculative decoding request.

| Context | T=1 minimum | T=8 minimum | T=8 median | T=8 range |
|---|---:|---:|---:|---:|
| 128 | 13.46 ms | 29.56 ms | 30.22 ms | 29.56–30.76 ms |
| 4K | 14.06 ms | 31.62 ms | 32.22 ms | 31.62–32.42 ms |
| 8K | 16.34 ms | 33.66 ms | 34.34 ms | 33.66–34.53 ms |
| 16K | 17.73 ms | 36.69 ms | 37.23 ms | 36.69–37.94 ms |
| 32K | 20.18 ms | 44.18 ms | 44.74 ms | 44.18–45.13 ms |

Each point has a 0.5-second warmup and 15 measured host wall-time samples.
Logits are finite and repeated fixed-state replays are bit-identical.
The real T=8 graphs also pass Metal shader validation at 128 and 32K; those
instrumented timings are excluded from the table.
T=1/T=8 have 674/744 dispatches respectively. This is an initial baseline,
not an exhaustive model-specific search or an MLX-LM comparison. A preceding
generic-autotuner run completed T=1 but was interrupted during T=8 tuning;
its results are separate and are not mixed into this table.

Raw evidence, scripts and source hashes are retained in the ignored local
directory `tools/bench/results/m5max-qwen-llama/hybrid35/`, with working files
under `/tmp/monolith-model-audit/hybrid35/`. They are not committed result artifacts.

## Experimental DSpark pairing

[M] October 4, 2026: tested the NVIDIA target above with
[`Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2`](https://huggingface.co/Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2),
revision `60f38a99168b6f6552501270e4efa0108fcc2c42`.

```bash
.venv/bin/python -m monolith.serve \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --draft Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2 \
  --max-context 4096 --port 8000
```

Both arguments also accept local paths; packs are cached automatically. The
served model name for these repository IDs is `Qwen3.6-35B-A3B-NVFP4`.
The draft has six dense attention layers, hidden width 2048, intermediate width
6144, 32 query heads, eight KV heads and head dimension 128. It taps target
layers `[1, 6, 11, 16, 22, 27, 32, 37]`. The checkpoint was trained with an
eight-token block. Serving and `load_session` default to **seven proposals plus
one anchor**, verifying **eight rows**. `--draft-block-size 8` restores the full
checkpoint block; smaller checkpoints keep their supported proposal count.

Despite NVFP4 in the repository name, the draft checkpoint's tensors are BF16.
These measurements preserve them in BF16 (3,132,768,256-byte pack). The loader
now preserves a checkpoint-owned frozen vocabulary head as well as embeddings;
it shares the target vocabulary head only when the checkpoint has none. Cache
keys distinguish this corrected draft pack from older packs that omitted the
head. The 40-core M5 Max selects BF16 matrix projections from two draft rows and
matrix attention for the seven-proposal block. These earlier measurements explicitly
retain source precision with `--draft-quantization none`; the later
[NVFP4 conversion](#nvfp4-draft-conversion) is now the matching chip's automatic choice.
The performance tables below record the earlier explicit
eight- and seven-proposal measurements; the later MoE tuning section measures
the new default.

The model card identifies a frozen **Unsloth** quantization as the training
target and recommends `Koopah/Qwen3.6-35B-A3B-NVFP4`. Our measurements instead use
the requested **NVIDIA** checkpoint. The card's acceptance claims are not assumed
to apply to this different target.

### Correctness checks and remaining gap

* The real checkpoint loads strictly into the independent DeepSpec implementation
  at revision `005e03b81cec38b7da6399833d609ee89a2587f2`. On a deterministic
  three-row synthetic target-feature fixture, Metal and CPU DeepSpec produce
  identical eight-token proposals. Feature cosine is 0.999996, block-hidden
  cosine 0.999383, base-logit cosine 0.999588, and every Markov-corrected logit
  row has cosine above 0.99949. Maximum confidence difference is 0.00798.
  Metal shader validation reports no violations. This is a draft arithmetic
  check, not a model-quality evaluation.
* Three real prompts run for 48 generated tokens with both ordinary decode and
  DSpark. The math prompt agrees exactly; hash-map and prime-function prompts
  first differ at zero-based output indices 30 and 34. Repeated runs are stable.
  Both the eight-proposal and shortened seven-proposal versions pass shader
  validation, but neither is qualified as bit-exact ordinary decoding.
* In an identical-state check at these divergence prefixes, changing the active
  verification rows from one to nine does **not** change the first argmax. The
  code prefix ingested through prefill nevertheless chooses the speculative
  token rather than the sequential-decode token. Execution-history numerics
  in the target are therefore implicated; a single batch-width comparison
  does not explain all accumulated differences or prove rollback correctness.
  Resolving this and the target's HF qualification gap remains necessary before
  claiming a lossless speculative speedup.

### Complete-round latency

[M] Same 40-core M5 Max. Real prefill of a repeated hash-map explanation; each
nonterminal replay restores StepState and GDN state. Fifteen GPU samples after
five warmups, no shader instrumentation. Compilation, packing, prefill and HTTP
overhead are excluded. All replay and split/full committed tokens and next
drafts agree exactly on this fixture; every proposal is accepted here.

| Context | Proposals + anchor | Complete round median | Verification | Accept/commit | Next draft |
|---|---:|---:|---:|---:|---:|
| 128 | 8 + 1 | 40.47 ms | 33.46 ms | 0.89 ms | 6.26 ms |
| 4K | 8 + 1 | 43.73 ms | 35.03 ms | 0.91 ms | 7.21 ms |
| 128 | 7 + 1 | 36.89 ms | 30.24 ms | 0.68 ms | 6.13 ms |
| 4K | 7 + 1 | 38.96 ms | 31.27 ms | 0.69 ms | 6.87 ms |

Stages are timed separately and need not sum to the full-round median. The
eight-proposal rows use the backend defaults. The seven-proposal experiment
uses `drafter_options={'block_size': 7}` and BF16 tensor projections from two
rows (`bf16_min_t=2`), rather than the then-default threshold of eight. It is
reproducible with `tools/bench/dspark_round_latency.py`
using `--draft-block-size 7` and a config containing `{"bf16_min_t":2}`.
These are initial measurements, not an exhaustive configuration search.
The subsequent MoE tuning measurements below extend complete-round coverage
through 32K context.

On the three short chat-template prompts above, a separate uninstrumented run
measures the following actual decode rates. Each produces 48 tokens, of which
47 are generated after prefill. Rates divide those 47 returned tokens by GPU
decode time, including any terminal over-generation. They do not use an assumed
acceptance rate. The differing outputs prevent interpreting every row as a
correctness-qualified speedup.

| Prompt | Plain decode | DSpark decode | DSpark rounds | Exact output match |
|---|---:|---:|---:|---|
| Hash-map explanation | 70.0 tok/s | 78.4 tok/s | 15 | No |
| Prime-number function | 69.7 tok/s | 195.5 tok/s | 6 | No |
| 17 × 24 explanation | 69.5 tok/s | 131.9 tok/s | 9 | Yes |

Three actual HTTP requests also complete successfully with the earlier
eight-proposal head: a 24-token prime-function chat prompt and 48 output tokens
use six decode rounds. Warm requests take 335 and 325 ms end to end, with
39.23 and 39.21 ms mean GPU round time. Cold loading/compilation takes the first
request to 26.36 seconds. These are smoke measurements on one prompt, not a
serving benchmark suite. Evidence is under the ignored `hybrid35/dspark/`
subdirectory of the audit results.

The 874 contract tests pass, including checkpoint-head preservation, automatic
cache migration and shorter-block validation. The MLX-dependent format check
was rerun with GPU access after the sandboxed suite could not initialize Metal.

## Eight-row MoE task tuning

[M] The October 4 study uses the NVIDIA target and BF16 Koopah draft above on
40-core M5 Max. **Seven proposals plus the anchor are now the default** for
serving, generation and the complete-round benchmark. Every measured round
verifies eight rows, commits accepted state and produces the next seven
proposals. The baseline and candidate use the same BF16 threshold of two and
matrix draft attention; only the routed-expert implementation changes.

Eleven alternating A/B pairs after three warmups give the complete nonterminal
GPU round medians below. Each context has real prefill of the same repeated
hash-map fixture. Packing, compilation, prefill, HTTP work and correctness
readback are excluded. All seven proposals are accepted on this fixture; this
is not an acceptance estimate for arbitrary requests.

| Context | Original expert path | New automatic path | Speedup |
|---|---:|---:|---:|
| 128 | 34.35 ms | 26.57 ms | 1.29× |
| 4K | 36.74 ms | 29.25 ms | 1.26× |
| 8K | 39.32 ms | 31.55 ms | 1.25× |
| 16K | 44.47 ms | 36.76 ms | 1.21× |
| 32K | 54.99 ms | 47.15 ms | 1.17× |

### Selected kernel boundary

The fastest routed-expert path uses **two kernels**:

1. Expert gate/up projection with SiLU/multiply.
2. A task-based megakernel containing expert down projection, ordered top-k
   reduction, shared-expert addition and residual addition.

The router, normalization and shared-expert projections remain separate.
This is not a single kernel for the whole MoE block. The complete DSpark graph
has 908 dispatches, down from 948. Each down/reduction task owns one token's
128-column output tile and all eight selected experts. Expert outputs retain
BF16 rounding in 2 KiB of threadgroup memory; the original FP32 combine order
is retained. Workers need only local barriers and never wait on another worker.

| Kernel | Row group | Row split | SIMD groups/worker | Workers | Preconvert activation |
|---|---:|---:|---:|---:|---|
| Gate/up + SiLU/multiply | 4 | 2 | 23 | 357 | Yes |
| Down + weighted/shared/residual reduction | 1 | 2 | 7 | 160 | Yes |

The second kernel uses eight 16-row weight blocks per task. The same policy
covers every measured context because expert matrix shapes do not depend on
KV length. It is restricted to NVFP4 top-eight, 2048/512 expert shapes at eight
verification rows on the 40-core chip. Other chips, shapes and prefill keep
their existing choices. Configuration lives in
`monolith/backends/metal/m5_max_40c/routed.py`, without duplicated context files.
Explicit MoE recipes can restore the original dispatches before retuning.

All 40 MoE outputs, committed tokens, acceptance, position and next proposals
are bit-identical to original Monolith in every paired context comparison.
Three 48-token generation fixtures, including partial draft rejection and a
repeated prompt, also preserve the previous seven-proposal output exactly.
The final suite passes **902 contract tests and 259 GPU tests**, the latter
under Metal shader validation. Real generation and the additional fusion
probes also pass shader validation. This preserves existing Monolith numerics;
it does **not** close the independent HF or plain/speculative accuracy gaps
reported above.

Three real Chat Completions requests with no block-size or kernel override
return HTTP 200 and report eight verification rows. The 24-token prime-function
prompt produces 48 output tokens in seven decode rounds. Warm requests take
276.1 and 267.7 ms end to end, with reported mean GPU round times of 25.90 and
25.87 ms. These request averages include the terminal round; the table above
is the nonterminal complete-step measurement. The cold request takes 26.1 s,
including loading and compilation. The smoke-test server is stopped afterward.

### Megakernel search and rejected alternatives

The implementation follows expert-task grouping and task/worker separation in
[Mirage PR #786](https://github.com/mirage-project/mirage/pull/786), inspected at
`c28cac618b98fda1e0f1d590923b5f69b4ef3603`. It is an independent Metal
implementation with no Mirage runtime dependency.

The documented finite sweeps contain **18,132 configuration trials**:

| Search family | Trials |
|---|---:|
| Scalar/grouped projections and worker geometry | 9,384 |
| Global-stage expert fusion, including task granularity | 1,890 |
| Threadgroup-local complete experts | 1,680 |
| Singleton specialization | 246 |
| Per-expert readiness queues, including idle-worker retirement | 3,840 |
| Cooperative/native matrix expert tasks | 456 |
| Down projection plus output reduction | 636 |

These include grouping 1/2/4/8 selected tokens per expert; legal power-of-two
row groups/splits through 16; every SIMD count from 1 through 32 in the extended
geometry searches; worker counts through 1,536 for scalar tasks; input
preconversion, immutable-input caching, inlining and singleton choices;
static/atomic task claims; local activation storage and partitioned down work;
and raw/losslessly repacked matrix operands. The scripts retain the exact
coarse grids and coordinate refinements. This is an extensive finite search,
not a proof of optimality over every possible Metal implementation.

The larger gate/up + SiLU + down megakernel leaves the router, shared expert
and weighted reduction outside. It improves on the original geometry, but the
selected down/reduction boundary is faster. At 128 tokens, the final paired
comparison measures 26.73 ms for the selected path versus 33.21 ms for that
larger expert megakernel. The best separately tuned expert projections take
29.33 ms against their paired 26.56 ms fused-down baseline.

A whole-MoE prototype, including normalization, router, shared projections and
combine, also passes exact-output and shader-validation checks at 4/8/16/32
SIMD groups. Its best isolated result is about 0.90 ms, substantially slower
than the selected block's approximately 0.33 ms. The grouped matrix variants
change reduction order slightly and do not beat the best scalar result; their
456 trials are excluded from the bit-exact selection. Sixty-eight polling-queue
trials hit the bounded timeout and are rejected. Both revised retirement-queue
grids, totaling 1,920 trials, finish exactly without timeouts, but remain slower.

At 128 tokens, the 40 real routers select an average of 45.3 distinct experts
from 64 token/expert pairs (range 33–56); long-context means are around 42.
That limits opportunities to reuse weights across many tokens in one expert.
The successful fusion instead groups all expert slots for an output tile,
avoiding the intermediate global expert-output buffer traffic and its separate
reduction dispatch.

Raw records, paired samples, source hashes and reproduction scripts are kept
locally under the ignored
`tools/bench/results/m5max-qwen-llama/hybrid35/moe-megakernel/` directory.
The contaminated early queue sweep is explicitly excluded from trial totals;
it was rerun with StepState reset after each trial.

## NVFP4 draft conversion

[M] The October 4 conversion keeps the NVIDIA target and its selected MoE
kernels unchanged and requantizes the Koopah draft's BF16 matrices into NVFP4.
The draft pack shrinks from **3,132,768,256 to 993,951,744 bytes**, a **68.3%**
reduction. The 46 converted source matrices cover the token embedding, feature
projection, all six attention/MLP layers, frozen vocabulary projection and
Markov output projection. The Markov embedding stays BF16; normalization and
confidence parameters retain their source precision. Activations retain the
existing BF16/FP32 numerics. The original checkpoint and BF16 pack are preserved.

The conversion exposed a quantized-embedding gather bug: the kernel applied
block scales but omitted the NVFP4 tensor scale. The compiler now binds the
pack's per-row scale table, and the gather multiplies by that scale before BF16
rounding. Regression tests cover nonunit and differing row scales, both scale
layouts, invalid token IDs and inactive-row bounds. The early pre-fix acceptance
run diagnoses this bug and is excluded from the conversion's quality results.

### Proposal quality

`tools/bench/dspark_quantization.py` compares 16 prompts: four code, four math,
four explanation, two writing and two language prompts (Chinese and Spanish).
Each generates 96 tokens greedily with seven proposals plus the anchor. A
separate comparison re-prefills the same BF16-reference continuation at offsets
0, 24, 48 and 72 and measures accepted proposals from one verification round.
All 64 matched prefixes produce the same target anchor for both packs.

| Metric | BF16 draft | NVFP4 draft |
|---|---:|---:|
| Mean accepted proposals per matched block, out of 7 | 3.9219 | 3.8594 |
| Matched-prefix proposal acceptance | 56.03% | 55.13% |
| Free-running rounds for the 16 generations | 409 | 418 |
| Free-running mean accepted proposals per round | 2.8093 | 2.7297 |
| Aggregate GPU decode throughput | 134.7 tok/s | 151.4 tok/s |

NVFP4 retains **98.4%** of BF16's mean matched-prefix acceptance. Fifty-seven
blocks tie, four accept fewer and three accept more proposals. **All 16 complete
96-token output sequences match BF16 exactly.** Decode throughput uses the
1,520 tokens generated after prefill divided by their total GPU decode time;
it includes terminal rounds and actual acceptance. These are finite prompt-set
results, not a general quality guarantee. The independent target/HF and
plain/speculative numerical gaps documented above remain open.

Keeping the large token embedding in BF16 increases the pack to 1.725 GB and
does not improve aggregate matched-prefix acceptance (3.8125/7). Additional
16-prefix screens preserving the feature or vocabulary projections also cost
more storage. The selected pack quantizes those matrices and retains only the
Markov embedding among the weight slabs.

### Runtime and use

The preconverted safetensors checkpoint is published at
[`LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4`](https://huggingface.co/LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4),
revision `5335e125637b290836ffe260e5314a06de6f78bb`. Its 952,165,546 bytes of
safetensors repack to the exact 993,951,744-byte tested Metal pack. Loading and
packing succeed with CPU NVFP4 quantization and dequantization disabled; only
the hardware layout is built. All 13 uploaded files have verified hashes.

Eleven alternating pairs after three warmups compare source-precision and NVFP4
drafts with the same target configuration and independent real prefills. Full
rounds include target verification, acceptance/commit and the next draft block.
Draft spans are measured separately, so their medians need not sum with other
stage medians to the full-round median.

| Context | BF16 draft stage | NVFP4 draft stage | BF16 full round | NVFP4 full round |
|---|---:|---:|---:|---:|
| 128 | 5.62 ms | 3.00 ms | 27.42 ms | 25.17 ms |
| 4K | 6.56 ms | 3.89 ms | 30.07 ms | 27.49 ms |
| 8K | 7.25 ms | 4.71 ms | 32.29 ms | 29.50 ms |
| 16K | 9.32 ms | 6.76 ms | 36.75 ms | 34.43 ms |
| 32K | 13.91 ms | 11.27 ms | 48.76 ms | 46.38 ms |

At 128 tokens this reduces draft latency by **46.6%** and complete-round latency
by **8.2%**. Both packs accept all seven proposals on this repeated hash-map
fixture at every context, with identical committed tokens and next proposals;
each pack's replays and split/full execution agree exactly. Loading, compilation,
prefill, HTTP work and correctness readback are excluded. These paired BF16
measurements are the comparison baseline, rather than the earlier MoE study's
measurements from a different run.

The serving CLI now selects this NVFP4 conversion automatically for the
validated 35B/DSpark shapes on 40-core M5 Max. Seven proposals plus the anchor
remain the default. The same configuration serves every measured context:
BF16 matrix threshold two and matrix draft attention, with the existing target
megakernels. No additional context-specific recipe files are needed.
An explicitly supplied existing draft pack still determines precision;
`--draft-quantization none` selects a source-precision cache.

```bash
.venv/bin/python -m monolith.serve \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --draft LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4 \
  --draft-quantization nvfp4 \
  --max-context 32768 --host 127.0.0.1 --port 8000
```

Missing packs are created automatically. To create the evaluated draft pack
explicitly from a local checkpoint, use:

```bash
.venv/bin/python tools/pack_weights.py \
  --model /path/to/Koopah-Qwen3.6-35B-A3B-NVFP4-DSPARK-v2 \
  --out /path/to/draft-nvfp4 --drafter-kind dspark \
  --max-context 33024 --lane-order interleaved16 --scale-placement block \
  --quantize nvfp4 --quantize-keep markov_w1
```

Validation passes **905 contract tests and 141 relevant GPU tests**, with Metal
shader validation enabled for the latter. Three real 96-token generations also
pass shader validation and match the BF16 reference. The wider draft checks
exposed stale probability-type names in partitioned-attention specialization;
those now preserve the kernel's BF16/FP16 probability type, with causal-cache,
replay and 16/32-key FP16 regression coverage. This repair does not change the
selected serving graph or the paired performance measurements.

A missing automatic draft cache builds in 35.21 seconds; a second `prepare`
reuses it in 1.48 seconds. Its `weights.pack` SHA-256 matches the manually
converted, quality-tested pack exactly. These startup times are separate from
the per-round measurements.

Three real Chat Completions requests using that automatic cache return HTTP 200
and identical 48-token output for the 24-token prime-function prompt. Each
reports eight verification rows and seven decode rounds. Warm requests take
300.6 and 262.5 ms end to end and report 23.22 and 24.02 ms mean GPU round time,
including the terminal round. The cold request takes 22.82 seconds with loading
and compilation. This is an endpoint smoke test on one prompt, separate from
the paired context benchmark and prompt-set quality evaluation. The test server
is stopped afterward.

Prompt fixtures, quality outputs, paired samples, validation logs, source hashes
and reproduction scripts are retained locally under the ignored
`tools/bench/results/m5max-qwen-llama/hybrid35/dspark-nvfp4/` directory. The
pre-fix `quality-all.json` is explicitly excluded; the qualified full-conversion
result is `quality-all-fixed.json`.
