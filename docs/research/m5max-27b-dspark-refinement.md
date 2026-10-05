# DSpark draft-kernel refinement on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

Measured 2026-10-03, Apple M5 Max, 40 GPU cores, 48 GB, macOS 26.5.1.
All timing values below are measured [M], in milliseconds unless stated otherwise.
This follows the [initial integration study](m5max-27b-dspark.md).

The refined BF16 recipe reduces draft latency by 9–10.5% relative to the preceding selected draft
recipe, saving about 1.0–1.5 ms per draft and 1.0–1.4 ms per complete round.
The target verification recipes and checkpoint weights are held fixed in this
comparison. This is an incremental improvement over the already tuned pipeline;
it is not the initial scalar-kernel baseline from the integration study.

## Paired complete-round result

The target is `nvidia/Qwen3.8-27B-NVFP4`; the draft is the original BF16
`RadixArk/Qwen3.8-27B-DSpark` checkpoint. Revisions and checkpoint SHA-256 are
unchanged from the integration study. A nonterminal round verifies eight
positions (anchor plus seven proposals), accepts/commits, then prepares the next
seven proposals. It includes the shared vocabulary head and sequential Markov
chain. Prefill, compilation and packing are excluded.

| Context | Previous draft | Refined draft | Draft reduction | Previous full round | Refined full round |
|---|---:|---:|---:|---:|---:|
| 128 | 10.420 | **9.424** | 9.6% | 48.882 | **47.921** |
| 4K | 11.069 | **9.906** | 10.5% | 51.156 | **49.989** |
| 8K | 11.480 | **10.424** | 9.2% | 53.063 | **51.967** |
| 16K | 12.433 | **11.273** | 9.3% | 57.089 | **55.979** |
| 32K | 14.263 | **12.795** | 10.3% | 64.278 | **62.905** |

Nine alternating paired samples follow five warmups. Each side restores its
StepState and target recurrent states to the same real-prefill checkpoint.
Compiler scratch barriers and task queues are private to each contender.
The table uses paired medians, not a sum of independently timed stages.
The timing fixture accepts all seven proposals in this first measured round;
that is a property of the fixture, not an assumed general acceptance rate.

Observed ranges, including the minimum of nine samples:

| Context | Previous draft range | Refined draft range | Previous full range | Refined full range |
|---|---:|---:|---:|---:|
| 128 | 10.379–10.475 | 9.413–9.460 | 48.782–49.014 | 47.824–47.977 |
| 4K | 11.029–11.100 | 9.890–9.927 | 50.986–51.358 | 49.848–50.230 |
| 8K | 11.457–11.510 | 10.357–10.434 | 52.949–53.157 | 51.776–52.428 |
| 16K | 12.416–12.484 | 11.256–11.483 | 56.834–57.170 | 55.900–56.508 |
| 32K | 14.228–14.393 | 12.782–12.816 | 64.048–64.584 | 62.716–62.974 |

All five paired comparisons preserve committed target tokens and the next draft
proposals. Split and complete replays agree on tokens. Independent subsequent
runs show appreciable clock/cache/system variation, so comparisons across files
must use their paired controls; absolute latency is not a service guarantee.

## Selected BF16 configuration

The exact recipes reside in the single
[selected-contexts.json](../../monolith/backends/metal/m5_max_40c/recipes/dspark/selected-contexts.json).
The previous file is retained in the evidence directory. All contexts share the
same projection/MLP/Markov recipes; only mixer geometry changes. `W` denotes
threadgroups, `SG` SIMD-groups per threadgroup, `TN` output rows per projection
tile, `TK` reduction tile, and `KS` cooperative K split.

| Component | Selected recipe |
|---|---|
| Draft MLP | Two native kernels; W40, SG8, TN16, TK64, KS1 |
| Feature projection | Native matrix kernel; W40, SG8, TN16, TK64, KS1 |
| Injected context K/V | Native matrix kernel; W80, SG8, TN16, TK64, KS4 |
| Shared vocabulary head | Native NVFP4 matrix kernel; W160, SG4, TN32, TK64, KS4, tile-block16, q-outer0 |
| Single-row feature fallback | W80, SG16, row-group2, row-split1, preconvert on |
| Single-row context K/V fallback | W320, SG16, row-group2, row-split8, preconvert on |
| Markov W2 | W320, SG16, row-group4, row-split4, preconvert and activation hoist on |

| Context | Mixer | W / SG | Query tile / key chunk | Projection details |
|---|---|---|---|---|
| 128 | One megakernel per layer | 80 / 8 | 24 / 32 | QKV TN16, KS8; output TN32 |
| 4K, 8K, 16K | Five native dispatches per layer | 80 / 8 | 16 / 256 | TN16, TK64, KS4; separate Q/K preparation |
| 32K | One megakernel per layer | 240 / 4 | 16 / 512 | QKV TN16, KS4; output TN32 |

The megakernels use bounded seeded task queues, compact attention partials,
aliased threadgroup scratch, and ordinary cached reads only for the immutable
KV prefix. Newly injected KV and cross-task intermediates retain coherent
accesses. The 32K queue assigns one attention tile per task and one task per
claim. The middle-context native path uses chunk-first task ordering.
The selected BF16 matrix threshold remains two rows; scalar fallbacks serve
one injected feature row. The Markov chain remains seven sequential groups of
embedding, W2, partial argmax and final argmax: fusing all 28 stages did not win.

The dedicated middle-context sweep covered 146 recipes at each of 4K, 8K and
16K, followed by eight longer-running finalists each and complete paired rounds.
The best fused alternatives did not establish a clear full-round improvement
(the observed ranges overlap), so the validated native selections remain.
There are 68 encoded draft dispatches at 128/32K and 88 at 4K–16K. Predicated
scalar and matrix fallback entries are included in those counts.

## Where the remaining draft time goes

GPU timestamp-counter attribution, summed across all five layers/seven Markov
positions, using the selected endpoint recipes:

| Work | 128 | 32K |
|---|---:|---:|
| Target-feature projection | 0.523 | 0.519 |
| Injected context K/V, five layers | 0.223 | 0.223 |
| Five mixer megakernels | 1.113 | 4.484 |
| Five gate/up projections | 3.083 | 3.091 |
| Five down projections | 1.550 | 1.549 |
| Shared target vocabulary head | 1.224 | 1.220 |
| Seven Markov W2 projections | 1.599 | 1.584 |
| Fourteen Markov argmax kernels | 0.112 | 0.106 |
| Seven Markov embedding gathers | 0.019 | 0.019 |
| Confidence and selection | 0.061 | 0.055 |
| Remaining norms, embedding, layout | 0.076 | 0.074 |

Counter profiling uses separate encoders and is an attribution study; its sum is
not the complete ICB latency. Raw per-dispatch counters and contiguous ICB span
checks are retained in `full-bf16-128.json` and `full-bf16-32768.json`.
The dominant BF16 projection screens process logical packed operand bytes at
roughly 580 GB/s, near the local streaming probe's approximately 599 GB/s.
These are logical byte-rate estimates, not measured DRAM traffic or a proof of
an absolute hardware lower bound.

## Correctness and benchmark repairs

The four fixed code/math/chat/text prompts generate 128 tokens each with EOS
stopping disabled. All 512 target tokens match the preceding recipe. Both
versions take 152 speculative rounds and accept 365 proposals in total
(mean 2.401 accepted proposals per round). Code, math and text acceptance
histories match exactly. Chat has different late acceptance decisions, with
the same 40 rounds and 87 accepted proposals. This is a finite regression set,
not a universal acceptance-quality claim.

The screens use actual checkpoint weights, deterministic features, finite-output
checks, cosine > 0.9999, relative L2 < 0.005 and exact replay-output bytes.
The synthetic draft oracle additionally exercises random cached prefixes,
injections of 3, 2, 1 and 8 rows, YaRN, confidence selection and rejected-block
KV isolation. Real-prefill full rounds and actual generation are the final gate;
hot microbenchmarks are used only to shortlist candidates.

A mixed-worker paired harness originally shared megakernel barrier storage
between the two programs. Its release counter could overlap another program's
worker epoch, causing a bounded timeout. The harness now excludes all
megakernel flags and queues from sharing, with a contract regression test. The
failed runs remain recorded and were not used in the result table.

Other repairs include bounded source growth for the 28-stage scratch maximum,
ceil-divided scale scratch for narrow NVFP4 reduction tiles, the draft attention
score-lane guard, and distinguishing constant parameter records from device
inputs in the optional external-cache rewrite. Unsafe or numerically rejected
recipes are retained as failed trials rather than silently omitted.

## Search coverage

The complete experiment set contains **5,606 recorded trials: 5,406 passed and
200 rejected**. The initial BF16 refinement accounts for 4,378 of those trials.
Counts include shorter screens, longer finalist confirmations, instrumented
bounds checks and rejected candidates. The final
[summary.json](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-dspark/refinement-20261003/summary.json)
records every family, including precision and cache-repair trials. Counts describe
executed trials, not unique configurations or a Cartesian product of every knob.

| Family | Choices exercised |
|---|---|
| Native projection geometry | Workers 40–320, SG4/8, TN16/32, TK64/128/256, KS1/2/4; independent gate/up and down crews |
| Scalar feature/KV fallbacks | Workers 40/80/160/320, SG4/8/12/16, row groups and row splits 1/2/4/8/16 where legal, preconversion on/off |
| Markov W2 | Workers 40–640, SG2/4/8/16, row groups/splits 1–16, activation hoisting/preconversion; 28-stage whole-chain fusion |
| Shared NVFP4 vocabulary head | TN16/32/64/128, TK32/64/128/256, KS1/2/4, native/cooperative paths, tile-block1–512, both weight traversal orders |
| Mixer matrix attention | Workers 80/160/240, SG4/8, query tiles16/24/32, key tiles16/32, key chunks32–1024, preparation on/off, native and fused paths |
| Attention scheduling | Head/chunk ordering, 1/2/4/8 tiles per task, queue batches1/2/4, queue/staged schedules, bounded seeded queues, merge SG1–8 and unroll1–32 |
| Synchronization and caching | SIMD/leader/serial barriers, task barriers, arrival-store/register variants, flag spacing, immutable-prefix reads, scratch aliasing, external-input caching |
| Alternative attention algorithm | Independent SIMD attention tiles with query tiles8/16/32 and key tiles32/64/128 within the shared-memory limit, cached prefix on/off |

The independent-SIMD attention algorithm passed its oracle checks but was slower
at both endpoints. Whole-chain Markov fusion was also a tie or regression.
All 168 key-tile16 real-weight screens failed the stricter numerical gate; those
recipes remain excluded even though the looser synthetic layer oracle passed.
The initial shared-head layout sweep had 26 narrow-tile compilation failures;
the scale-scratch repair allowed all 26 to be rerun successfully. Four optional
cache trials initially failed compilation because of the qualifier rewrite;
the repaired reruns and oracle checks pass. Against the final BF16 recipe,
external caching is slower at 128 and no better at 32K, so it stays disabled.
The last two rejected trials belong to the attempted arithmetic-order variant
described below.

Screens use zero-filled old KV prefixes and deterministic feature tensors; full
rounds use real prompt prefills. During the middle-context search, preparation
was shortened to avoid unnecessary later-layer/vocabulary work. New rows record
`preparation_layers`. Identity-control timings can differ between independent
allocations despite identical source, so microbenchmark rankings alone are not
a promotion gate. The complete paired runs determine the selected recipes.

This is a finite search over the implemented families. It does not prove a
global optimum or rule out future algorithms. The BF16 results retain safe math,
seven proposals, the full vocabulary, and the existing five-layer draft model;
changing model depth, proposal count or vocabulary semantics is outside this
kernel-tuning comparison.

## Precision experiments

All experimental packs preserve the BF16 Markov W1 embedding table and
requantize 37 eligible draft matrices. Confidence weights, norms and shared
target embedding/vocabulary weights retain their existing storage. Keeping W1
also avoids unsupported quantized gather layouts; each step only gathers seven
256-element rows from it, so its BF16 storage adds negligible streamed bytes.
Packs are separate and do not overwrite the original BF16 checkpoint pack.

Initial format-specific geometry searches tune MLP, feature, context K/V and
Markov projections. Each format passes actual-weight shader validation at 128
and 32K context with one and eight injected feature rows. All four formats
produce the same 512 target tokens as the original generation baseline on the
four fixed prompts; acceptance histories differ.

| Draft matrix format | Draft 128 | Draft 32K | Full round 128 | Full round 32K | Generation rounds | Mean accepted | GPU ms / delivered decode token |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 9.414 | 12.993 | 47.846 | 62.708 | 152 | 2.401 | 14.191 |
| FP8 E4M3 | 5.944 | 9.294 | 44.460 | 59.299 | 151 | 2.417 | 13.085 |
| INT8 | 6.719 | 10.206 | 45.164 | 59.981 | 151 | 2.424 | 13.302 |
| Affine INT4 | 4.719 | 7.986 | 43.167 | 57.913 | 153 | 2.386 | 12.876 |
| NVFP4 | 4.351 | 7.651 | 42.784 | 57.670 | 149 | 2.510 | 12.461 |

These are independent-run medians, not paired cross-format speedup estimates.
The generation column uses the actual 508 delivered decode tokens: each prompt's
first token comes from prefill, which is outside the decode interval. Acceptance
counters include the last capped round, so accepted proposals plus round count
need not equal delivered tokens. EOS stopping is disabled for these tests.
Better acceptance on a small prompt set does not establish a general quality
advantage for a quantized drafter.

NVFP4 is the strongest initial candidate on this set. A separate paired test
performs independent real prefills and keeps both draft caches and recurrent
states private, sharing only byte-identical weight bindings. Nine alternating
samples follow five warmups. Both sides accept seven proposals in the timed
fixture and produce identical eight-token committed prefixes and next proposals.

| Context | BF16 draft median (range) | NVFP4 draft median (range) | BF16 full median (range) | NVFP4 full median (range) |
|---|---:|---:|---:|---:|
| 128 | 9.552 (9.485–9.718) | 4.392 (4.350–4.461) | 48.427 (47.853–49.460) | 43.188 (42.848–43.487) |
| 32K | 12.974 (12.780–22.263) | 8.154 (7.670–19.571) | 63.100 (62.530–77.854) | 58.151 (57.671–67.041) |

The latency ranges do not overlap in these paired tests. A final format-specific
sweep then varies NVFP4 mixer geometry/layout, all Markov activation/row splits,
one-row feature/KV fallbacks, and independent MLP projection crews/layouts.
Its selected [NVFP4 endpoint recipes](../../monolith/backends/metal/m5_max_40c/recipes/dspark/selected-nvfp4-endpoints.json)
produce these **paired incremental** results against the initial NVFP4 recipes:

| Context | Initial NVFP4 draft | Refined NVFP4 draft | Initial full round | Refined full round |
|---|---:|---:|---:|---:|
| 128 | 4.352 | **4.186** | 42.794 | **42.611** |
| 32K | 7.672 | **7.467** | 57.640 | **57.472** |

Draft ranges are 4.346–4.362 versus 4.166–4.199 ms at 128, and 7.665–7.697 versus
7.462–7.509 ms at 32K. The complete-round ranges overlap, so the incremental
whole-round gain is not established beyond the observed spread. It is the
isolated draft span that clearly improves in this final refinement.

All 512 target tokens still match. The refined four-prompt run takes 150 rounds
and 12.470 GPU ms per delivered token, versus 149 rounds and 12.461 ms for the
initial NVFP4 recipe: effectively unchanged generation throughput on this set.
The chat prompt needs one extra round; code, math and text keep their previous
round counts. The final recipes minimize measured draft latency, with no claim
of improved acceptance or universal generation throughput.

An additional variant restores the old MLP and attempts to retain the old mixer
projection traversal while keeping the faster geometry. It does **not** preserve
whole-stack hidden values: relative L2 is 1.28%/1.46% for one/eight injected rows,
above that experiment's 0.5% comparison gate. It was rejected before generation,
not promoted or used in a performance claim. No shader bounds violation was
reported in these rejected comparisons.

The final selected NVFP4 recipe passes four further actual-weight shader checks
(128/32K × one/eight injected rows). The original BF16 recipe remains in
`selected-contexts.json`; NVFP4 requires its separate requantized pack and
endpoint configuration file. Only the two endpoints have full NVFP4 measurements;
intermediate-context NVFP4 performance is not inferred from the BF16 table.

## Final validation

- 100 relevant CPU contracts passed.
- 179 draft/attention oracle cases passed with Metal shader validation, including
  BF16 and all four experimental quantization formats, YaRN, multiple injection
  lengths, cache boundaries, native/fused paths, and speculative-KV isolation.
- 185 GDN/MLP regression cases passed with Metal shader validation after the
  shared compiler repair.
- All five BF16 full-round pairs and both NVFP4 precision pairs preserve the
  committed target tokens and next proposals in the timing fixtures.
- Both selected BF16 and refined NVFP4 recipes reproduce all 512 target tokens
  in the four-prompt generation regression. The acceptance differences above
  are retained rather than hidden behind a fixed tokens-per-round assumption.

All GPU work is serial. Shader-validation timings are never used as performance
measurements. The manifest records source and packed-weight SHA-256 hashes;
the repository working tree contains uncommitted implementation changes, so
its base Git commit alone is not the full source identity.

## Reproduce

Use the checkpoint packing commands and target pack from the integration study.
For the paired incremental BF16 comparison, change the configuration key and
context together (128, 4096, 8192, 16384 or 32768):

```sh
.venv/bin/python tools/bench/dspark_round_latency.py \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 \
  --pack /tmp/monolith-m5max/attention-tasks/pack-33k \
  --drafter /tmp/monolith-models/Qwen3.8-27B-DSpark \
  --drafter-pack /tmp/monolith-m5max/dspark/pack-bf16-33k \
  --profile monolith/backends/metal/m5_max_40c/config.json \
  --inputs tools/bench/results/m5max-27b-dspark/inputs.json \
  --config monolith/backends/metal/m5_max_40c/recipes/dspark/selected-contexts.json \
  --compare-config tools/bench/results/m5max-27b-dspark/refinement-20261003/previous-selected-contexts.json \
  --config-key 128 --contexts 128 --reps 9 --warmup 5 \
  --check-generation --generation-tokens 128 --profile-draft \
  --out /tmp/dspark-refined-128.json
```

To reproduce NVFP4, make a separate draft pack, then use it and the endpoint
recipe with the round command above:

```sh
.venv/bin/python tools/pack_weights.py \
  --model /tmp/monolith-models/Qwen3.8-27B-DSpark \
  --out /tmp/monolith-m5max/dspark/pack-nvfp4-keep-w1-33k \
  --drafter-kind dspark --max-context 33024 --scale-placement block \
  --quantize nvfp4 --quantize-keep markov_w1
```

Use `--drafter-pack /tmp/monolith-m5max/dspark/pack-nvfp4-keep-w1-33k` and
`--config monolith/backends/metal/m5_max_40c/recipes/dspark/selected-nvfp4-endpoints.json`.
For the within-NVFP4 incremental comparison, use the retained
`refinement-20261003/selected-nvfp4-contexts.json` as `--compare-config`.
For a cross-precision pair use `dspark_compare_packs.py`, which gives each pack
its own real prefill and private cache (exact invocations are in `repack-driver.py`).

The [evidence directory](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-dspark/refinement-20261003)
contains the configuration grids, every trial row, full-round measurements,
generation histories, test logs and experiment drivers with their exact local
commands. `summarize.py` rebuilds the compact trial/generation summary from those
files. Generated programs and multi-gigabyte weight packs are not copied into
the repository; source digests, manifests and checkpoint identities are retained.
