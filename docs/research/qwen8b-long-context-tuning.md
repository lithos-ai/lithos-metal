# Qwen3 8B: 4K and 8K decode tuning on M5 Pro

Measured 2026-09-29 on Apple M5 Pro, 20 GPU cores, 24 GB, macOS 26.5.1.
Uses the implementation measured in the [serving comparison](qwen8b-serving-decode.md),
with unchanged NVFP4 target and Qwen3 0.6B affine INT4 draft weights. This study
completes missing projection-tuning entries and tests attention choices at long
context. No production kernel or default selection rule changes. The fresh
[MLX-LM comparison](#comparison-with-mlx-lm) below confirms an N=7 advantage,
but not a meaningful plain-decode advantage.

## Selected configuration

- Enable autotuning/cache reuse and commuted normalization; keep safe math.
- Keep `attention=auto`: matrix attention for N=7 target verification and v3 for
  plain decoding and the sequential draft passes.
- Reserve 8,704 context positions in both packs and the session; use 64-token
  prefill chunks for both plain and speculative measurements.
- For speculation, keep gamma=7, fixed verify length=7 and the existing one-round
  command buffers with two in flight. Plain keeps eight steps/CB and three in flight.

The [saved cache](../../tools/bench/results/qwen8b-long-context-20260929/autotune.apple-m5-pro.json)
contains 77 choices: 68 previously measured choices plus nine missing tile choices
measured here. The earlier short-context probe retained defaults on those misses;
this run uses the normal autotuner. Projection keys depend on matrix/pack shape,
not KV length, so long-context attention was screened separately.

## Controlled tuning screen

One real prefill per context, restored mutable state before each replay, one
excluded warmup and four measured repetitions. Candidate order alternates.
Plain replays 32 tokens; N=7 replays eight full rounds. Native wall milliseconds:

| Mode / configuration | 4K (4,095 tokens) | 8K (8,191 tokens) |
|---|---:|---:|
| Plain, tuning off | 18.27 / token | 21.11 / token |
| Plain, tuning on | 18.24 / token | 21.10 / token |
| N=7, tuning off | 69.66 / round | 89.60 / round |
| N=7, tuning on | 51.69 / round | 71.53 / round |

Projection tuning reduces N=7 round latency by **25.8% / 20.2%** in this paired
screen. Plain changes by less than 0.2%; that is not evidence of a plain-decode
speedup. This table is a controlled replay, not the full-generation result below.

Forcing matrix attention everywhere costs 62.58 / 90.04 ms per N=7 round;
forcing v1 costs 117.48 / 187.87. Reducing v3 to 16 or 8 SIMD-groups also loses.
The target matrix-attention grid was screened at 20/40/80/160 groups and
4/8/16 SIMD-groups. Its best alternative, 160 groups × 8 SIMD-groups, improves
whole rounds by only 1.4% / 2.5% against the interleaved control, below the
3% promotion margin in the design. The existing 80 × 8 geometry remains selected.
An explicit plain-v2 attempt failed the existing workspace-capacity check before
execution and produced no timing. It is excluded from the ranking.

Every measured replay repeats its token sequence exactly for its configuration.
The fresh-generation prefix checks below independently verify the restored-state
path. There is no claim that changing attention algorithms or projection reductions
preserves all output tokens.

## Full-generation validation

Each context uses the same task with six request-number variants: one excluded
warmup and five measured requests. Every request gets fresh prefill/KV and
generates 128 tokens through unmodified `Session.generate`; decode timing covers
the following 127 tokens. Median request-average wall milliseconds:

| Metric | 4K | 8K |
|---|---:|---:|
| Plain ms/output token | 18.26 | 21.12 |
| N=7 ms/full round | 51.92 | 71.08 |
| N=7 ms/output token | 21.27 | 30.20 |

All 24 requests completed; all speculative rounds verified seven drafts. The
fresh `r1` output prefix matches the restored-state screen exactly for both
modes and contexts. Plain/speculative full outputs match on 1/10 measured
prompt pairs; this is performance validation, not a reference-accuracy gate.
The measured N=7 round ranges are 51.64–52.21 ms at 4K and 70.998–73.60 ms at 8K.

N=7 still loses to plain decoding per output token on this low-acceptance
workload. The tuning reduces the round cost; it does not establish a useful
speculation speedup here. [Issue #133](https://github.com/jiazhihao/mpk-apple/issues/133)
tracks that remaining policy/acceptance problem. The prior report's untuned plain
run used 128-token prefill chunks and smaller cache capacity, so use this study's
controlled screen when attributing changes to tuning. The following follow-up measures MLX-LM at both contexts; vLLM/llama.cpp
were not rerun at 8K.


## Comparison with MLX-LM

A fresh comparison uses MLX 0.32.1 / mlx-lm 0.32.0, the same checkpoints, exact
input token IDs, greedy sampling, unquantized full KV and 128 outputs. There is
one resident engine and one request at a time. Engine order reverses at 8K.
Both native MLX prefill defaults and 64-token chunks matching Monolith are tested.
Each primary/sensitivity cell has five measured prompts plus an excluded warmup.

### N=7: the advantage holds

Median request-average wall milliseconds per full N=7 round:

| Engine/configuration | 4K | 8K |
|---|---:|---:|
| Monolith, fresh comparison | **58.51** | **72.92** |
| MLX-LM, native prefill 512 | 83.38 | 99.86 |
| MLX-LM, matched prefill 64 | 75.21 | 99.60 |

Monolith's round latency is **22.2% / 26.8% lower than the faster MLX setting**.
Its per-output latency is 24.18 / 30.93 ms, versus MLX's best tested per-output
medians of 30.32 / 36.77 ms (prefill64 at 4K, native512 at 8K). That is a
**20.3% / 15.9%** output-latency reduction on this workload. Acceptance and output
sequences differ, so this is not a claim of output equivalence.

Timer endpoints need care. MLX round intervals exclude the first round and
shortened tail; Monolith's request average includes a terminal round that can
skip the next draft chain. To bound that benefit, divide Monolith's entire decode
wall time by `rounds-1`, charging all terminal/queued work to the preceding full
cycles. The resulting conservative medians are **59.59 / 74.30 ms**, still
**20.8% / 25.4% below** the faster MLX round baseline. Every one of the ten
measured prompt pairs wins under this conservative bound against both MLX
settings, and also wins in per-output latency.

MLX per-output latency is first-to-last native token-yield time divided by 127;
Monolith uses its native decode-pump wall time. MLX's first yield follows a whole
verification round, potentially excluding computation for several tokens in its
first burst, whereas Monolith's prefill emits one token. The additional
`full_steps_ms_per_token` counter records MLX's whole-round time divided by the
tokens emitted within those intervals; raw timestamps and all counters reconcile.

### Plain decode: near parity

Absolute timings shifted between sequential blocks, including Monolith, despite
unchanged packs/cache and identical Monolith output/acceptance sequences on all
24 rerun requests. The primary comparison measured 20.67 / 23.63 ms/token for
Monolith versus 20.89 / 23.50 for MLX. A later control repeated both MLX prefill
sizes and Monolith, with three measured prompts plus warmup per cell:

| Engine/configuration | 4K ms/token | 8K ms/token |
|---|---:|---:|
| Monolith | 18.30 | 21.19 |
| MLX-LM, native prefill 2048 | 18.69 | 21.07 |
| MLX-LM, prefill 64 | 18.68 | 21.09 |

Both MLX prefill sizes converge in this later control; the earlier large change
cannot be attributed to prefill size alone. Monolith's small 4K plain lead and
slight 8K deficit are below the design's 3% promotion margin. **A general
plain-decode advantage is not established.** Do not combine the older Monolith
timing with a slower MLX block to claim a larger lead. The N=7 gap survives even
comparison against the fastest tested MLX block and the terminal-round bound.

All **96 requests** (78 measured, 18 warmups) returned 128 tokens. Exact
cross-engine full-output matches occur on only 1/10 plain and 2/10 speculative
primary prompt pairs; changing MLX prefill size also changes most continuations.
[Raw comparison records, token hashes, counter checks and launches](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/qwen8b-long-context-20260929/mlx-comparison/)
retain the complete evidence. These results establish the N=7 advantage on this
workload at 4K/8K; they do not generalize to every prompt family or longer context.

```bash
git show ade38fb5f81ebdf852a2b65a616703b03f4ec424:tools/bench/results/qwen8b-long-context-20260929/prompts.json > /tmp/qwen8b-long-context-prompts.json
python tools/bench/mlx_spec_step_latency.py --mode n7 \
  --model /path/to/mlx-Qwen3-8B-nvfp4 \
  --drafter /path/to/mlx-community-Qwen3-0.6B-4bit \
  --prompts /tmp/qwen8b-long-context-prompts.json \
  --out /tmp/mlx-long-n7.jsonl
# Use --mode plain without --drafter for plain decode.
# Add --prefill-step-size 64 for the matched-prefill check.
```

## Isolated N=7 target verification

A September 30 follow-up removes drafting and acceptance entirely. N=7 has
**eight target rows**: one anchor plus seven proposals. The timer covers
embedding, all 36 target layers, final normalization and vocabulary projection;
it excludes prefill, sampling, acceptance, rollback and draft generation.

| Prefix context | Monolith wall ms | MLX-LM wall ms | Lower latency | Monolith GPU ms |
|---|---:|---:|---:|---:|
| 4,095 | **26.87** | 50.10 | **46.4%** | 26.54 |
| 8,191 | **33.77** | 64.05 | **47.3%** | 33.41 |

These are medians of 20 fixed-state replays after five warmups per cell, with
one real prefix/batch at each length. Both engines use identical input IDs,
the same target checkpoint and prefill64. Monolith replays the unchanged
speculative ICB's 220 target dispatches, stopping before argmax. MLX evaluates
its native model and materializes all eight logit rows. Cache offsets remain
fixed; output hashes are stable within each engine. Host wall timing includes
submission and synchronization in both engines. The engines run serially in
separate processes; these repeats are not independent prompts.

This isolates the target's batched-forward advantage. Dividing by eight gives
amortized cost per verified position, **not** cost per accepted output token.
[Raw inputs, samples and methodology](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/qwen8b-target-verify-20260930/)
are retained. Reproduce with `tools/bench/target_verify_latency.py`: run
`--engine monolith` with the model/pack/drafter paths and the existing prompts,
then `--engine mlx` with the same model and generated `--inputs` file. Both
require `--out`; default contexts are 4095 and 8191, default repeats are 20.

## Reproduction

Both pack directories must cover 8,704 positions. The local packs for this study
are `/tmp/mpk-long-context/target-pack` and `/tmp/mpk-long-context/draft-pack`.
They preserve every packed weight slab and extend only the BF16 RoPE tables;
every prior table prefix was checked byte-identical. A new pack with a different
physical layout is a different configuration and may need additional tuning.

Copy the archived `autotune.apple-m5-pro.json` into the matching target pack to
reuse the measured choices. Run the screen (`--geometry` adds the grid search),
then fresh generation (using the prompts restored above):

```bash
python tools/bench/long_context_tune.py --mode n7 \
  --model /path/to/mlx-Qwen3-8B-nvfp4 --pack /path/to/target-pack \
  --drafter /path/to/mlx-community-Qwen3-0.6B-4bit --drafter-pack /path/to/draft-pack \
  --prompts /tmp/qwen8b-long-context-prompts.json \
  --outdir /tmp/long-context-screen
# Add --generate --autotune for the full-generation validation.
# Use --mode plain and omit both drafter paths for plain decode.
```

For normal generation, the equivalent existing CLI settings are
`--commute-norm --attention auto --max-context 8704 --prefill-chunk-size 64`;
autotuning is on unless `--no-autotune` is supplied. For N=7 add
`--drafter-kind lm --draft-gamma 7 --verify fixed --verify-length 7` and both draft
paths. The benchmark HTTP adapter differs: it requires explicit `--autotune` to
preserve its old untuned-command semantics, and now accepts the context/attention
options and records their values in each result.

[Raw evidence and methodology](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/qwen8b-long-context-20260929/)
include prompts, every screen sample, timing ranges, choices and full generated
token lists. These are one workload family on one machine, with sequential full
request blocks; no tail-latency or cross-model claim is made.
