# Qwen3 8B: single-request decode on M5 Pro

Measured 2026-09-29 on the 20-core Apple M5 Pro, 24 GB, macOS 26.5.1.
Monolith revision `03a4252` (the input-normalization fusion branch).

The checkout retains compact summaries and methodology. Raw samples, logs, prompts
and tuning caches remain in the [immutable evidence archive](https://github.com/jiazhihao/mpk-apple/tree/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929).
See the [evidence README](../../tools/bench/results/qwen8b-serving-decode-20260929/README.md) to restore the
prompts and cache used by the reproduction commands below.

**Correction: the original Monolith serving runs disabled autotuning, unlike
the earlier layer benchmark and the normal Session default.** Reusing available
cached choices reduces the short-context N=7 round from 53.57 to **35.83 ms**.
The tables below preserve the original untuned measurements; the component audit
below records the correction. The [long-context follow-up](qwen8b-long-context-tuning.md) now records tuned
4K/8K plain and N=7 results.

Plain Monolith has the lowest plain-decode latency in the original screen. Against
vLLM-Metal with the same NVFP4 checkpoint, its median latency is 4.3%, 6.8%,
and 6.9% lower at the three context lengths. llama.cpp and Ollama are close to
each other. With the original untuned Monolith configuration, fixed N=7 speculation
slows all three tested speculative engines on this prompt set; a faster verification layer does not guarantee faster generation.

## Decode results

Median **milliseconds per generated token**, lower is better. Prefill, model
loading, initial compilation, tokenization and HTTP serialization are excluded.
Each request produces 128 tokens; decode covers the 127 tokens after the first
token produced by prefill. Columns are actual input-token counts, not rounded
context labels. These are full-model generation measurements, including the
vocabulary head and sampling, rather than fixed-T layer replays.

| Engine / mode | Target format | 126 | 1,023 | 4,095 |
|---|---|---:|---:|---:|
| Monolith plain (autotune off) | NVFP4 | **17.56** | **18.35** | **20.68** |
| vLLM-Metal plain | NVFP4 | 18.35 | 19.70 | 22.22 |
| llama.cpp Metal plain | Q4_K_M | 20.95 | 21.24 | 23.05 |
| Ollama plain | Q4_K_M | 20.70 | 21.25 | 23.00 |
| Monolith N=7 (autotune off) | NVFP4 | 24.89 | 21.53 | 31.26 |
| vLLM-Metal N=7 | NVFP4 | 27.70 | 26.83 | 41.72 |
| llama.cpp Metal N=7 | Q4_K_M | 39.57 | 41.01 | 44.65 |

Ollama was measured with speculation off. Its bundled llama-server is a different
version from the standalone llama.cpp build.

The two NVFP4 engines use the same checkpoint values; the two GGUF engines use
the **same Q4_K_M file**. Comparisons across these pairs are **not quantization-
or quality-matched**. The tested llama.cpp Metal backend rejects NVFP4 matrix
multiplication, so using NVFP4 there would not provide an equivalent GPU baseline.
The same Qwen3 0.6B affine INT4 drafter is used by Monolith and vLLM-Metal.
llama.cpp uses the official Qwen3 0.6B Q8_0 GGUF drafter, so its speculative
comparison also includes a different draft quantization.

For Monolith with autotuning disabled, N=7 increases median latency by 42%, 17%, and 51%. Across measured
requests, it accepts only 1.22, 1.66, and 1.44 of seven proposed tokens per round
at these contexts. Drafting plus verification costs more than the work saved.
Monolith N=7 is still 10%, 20%, and 25% lower latency than vLLM-Metal N=7 here.
These rates depend on the prompt and generated continuation; they do not establish
that N=7 is generally unhelpful.
Follow-up: [avoid the low-acceptance N=7 slowdown (#133)](https://github.com/jiazhihao/mpk-apple/issues/133).

## Time per N=7 step

A full step means **seven draft proposals, target verification of up to eight
positions, and acceptance/rollback**, including host overhead. It is not one
output token and is not target-verification-only time. Median request-average
milliseconds per full step, with five measured requests per context:

| Engine | 126 | 1,023 | 4,095 |
|---|---:|---:|---:|
| Monolith (autotune off) | **53.57** | **58.18** | **74.92** |
| MLX-LM (direct) | 54.87 | 62.48 | 82.88 |
| vLLM-Metal | 62.65 | 71.07 | 100.00 |
| llama.cpp Metal | 91.38 | 96.37 | 116.43 |

Monolith's step latency is 2.4%, 6.9%, and 9.6% lower than direct MLX-LM's,
and 14.5%, 18.1%, and 25.1% lower than vLLM-Metal's;
it is 41.4%, 39.6%, and 35.7% lower than llama.cpp's. The target/drafter
quantization differences described above still apply. These step costs explain
part of the per-token gap independently of differences in accepted tokens per
round; low acceptance can still make speculation slower than plain decoding.

Monolith reuses the original N=7 samples: divide `decode_wall_ms` by
`len(accepted)`, checking every `verify_len` is seven. vLLM was rerun with the
same settings and additional per-request Prometheus counter deltas for draft
rounds, proposed tokens and accepted tokens. Every measured vLLM round proposed
exactly seven tokens; its decode wall time is divided by the round count.

llama.cpp needs different treatment: it falls back to single-token iterations
near the output limit. Dividing total decode time by full-round count would
charge that tail to the N=7 rounds, while dividing by all iterations mixes
cheaper single-token steps into the average. A fresh run with its built-in
`LLAMA_TRACE=1` records each `accepted x/7` event. The table uses the mean time
between consecutive full-round events within each request, excluding the first
round (no preceding timestamp) and the single-token tail. Trace round counts and
accepted-token sums exactly match the response counters. Instrumentation and
native timer boundaries differ, so this is an engine-cycle comparison rather
than a synchronized GPU-only microbenchmark.

The direct MLX-LM baseline uses MLX 0.32.1 / mlx-lm 0.32.0 from the same pinned
environment as vLLM-Metal, with the same NVFP4 target and affine INT4 drafter.
[Its benchmark](../../tools/bench/mlx_spec_step_latency.py) consumes the
unmodified `speculative_generate_step` generator, using default greedy sampling,
fresh unquantized KV caches and 512-token prefill chunks. It timestamps each
non-draft token yield (the end of a round) and reads the suspended generator's
`num_draft` local to exclude shortened tail rounds. Consecutive full-round
intervals include cache rollback, drafting, target verification, acceptance,
GPU waits and Python overhead. The first round is excluded with prefill;
detokenization is outside the timed loop. This is direct MLX-LM, not an HTTP
server. The native generator source hash is preserved because the instrumentation
depends on its round-end yield and local-variable conventions.

All 18 MLX requests returned 128 tokens. Boundary acceptance counts reconcile
with emitted-token counts, and every interval and mean was recomputed from the
saved timestamps. Request-average MLX ranges were 54.69–54.92, 62.39–62.58, and
82.85–83.05 ms at the three contexts. The short-context advantage is small;
these sequential engine blocks do not establish a universal performance ranking.

The [step evidence](https://github.com/jiazhihao/mpk-apple/blob/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929/steps.jsonl)
retains request averages, iteration counts and llama.cpp's individual acceptance
timestamps. The [raw evidence archive](https://github.com/jiazhihao/mpk-apple/tree/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929) also contains both new runs' responses and counter
snapshots; `methodology.json` records the follow-up launches. Original per-token
results above are unchanged.
The [archived MLX records](https://github.com/jiazhihao/mpk-apple/blob/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929/mlx-lm-n7-steps.jsonl)
separately retain every round-end boundary, timed interval and generated token.

```bash
python tools/bench/mlx_spec_step_latency.py \
  --model /path/to/mlx-Qwen3-8B-nvfp4 \
  --drafter /path/to/mlx-community-Qwen3-0.6B-4bit \
  --prompts /tmp/qwen8b-serving-prompts.json \
  --out /tmp/mlx-n7-steps.jsonl
```

## Why the MLX gap differs from earlier reports

The earlier [normalization-fusion comparison](norm-projection-fusion.md) at
N=7/context 128 reported 493.06 / 914.26 microseconds per target decoder layer
for Monolith / MLX. Across 36 layers that is 17.75 / 32.91 ms, about a 46%
latency reduction. That benchmark excludes drafting, embeddings, the vocabulary
head, sampling and acceptance, and runs fixed-T layers on seeded inputs/cache
prefixes. Its minima cannot be compared directly with this report's median
full-round times of 53.57 / 54.87 ms at a 126-token prompt. Generation also uses
dynamic execution and growing real KV state. Crucially, the serving adapter
hardcoded `autotune=False`, while the layer harness used the default
`autotune=True`. The original explanation omitted this configuration mismatch;
it was not just a difference in timing scope. The full-round result does not
establish that MLX's target-layer kernels suddenly became faster, and subtraction
across these harnesses cannot isolate the cost of drafting or scheduling.

The software environments also differ. To check that confounder, the exact new
MLX step benchmark was rerun with the earlier environment, retaining the same
checkpoints, prompts, 128 outputs, warmup and repetition counts:

| Environment | 126 | 1,023 | 4,095 |
|---|---:|---:|---:|
| Earlier: Python 3.13, MLX 0.32.2, mlx-lm 0.31.3 | 55.32 | 63.11 | 83.65 |
| Serving study: Python 3.12, MLX 0.32.1, mlx-lm 0.32.0 | 54.87 | 62.48 | 82.88 |

Units are median request-average ms/full step. The difference is only 0.8–1.0%,
with all 18 output token sequences identical and an identical speculative-loop
AST. Sequential runs do not isolate package effects from timing drift, but this
check does not support a large environment-driven improvement. The
[old-environment raw records](https://github.com/jiazhihao/mpk-apple/blob/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929/mlx-lm-n7-old-env.jsonl)
and methodology preserve the audit. The following component profile identifies
the much larger Monolith configuration effect.

## Draft-cost and tuning audit

The drafter is Qwen3 0.6B affine INT4: 28 layers, width 1,024 and a
151,936-token vocabulary. Seven sequential autoregressive proposals traverse
196 decoder layers and evaluate the vocabulary head seven times. The target
verifies up to eight positions together, sharing weight reads across positions;
parameter count alone therefore does not predict the draft/verify cost ratio.

The same short-context dispatch stream was profiled with autotuning off, then
with available saved choices reused. Cache misses retained default choices;
**this is a partial cache-reuse probe, not a fresh complete autotuning search**.
Both use unchanged checkpoints, packs, gamma, verification length and prefill.
Approximate GPU milliseconds per round:

| Component | Autotune off | Reuse cached choices |
|---|---:|---:|
| Target embedding and 36 decoder layers | 25.01 | 17.92 |
| Target vocabulary head and sampling | 1.57 | 1.64 |
| Seven draft decoder passes | 23.92 | 13.69 |
| Seven draft heads and sampling | 3.65 | 3.00 |
| Acceptance and selection | 0.02 | 0.02 |
| Unsplit control, GPU time | 53.78 | 35.86 |

Each stage is a contiguous subset of the original compiled dispatches, with the
same buffers and bindings, timed in a separate ICB. Six alternating whole/split
pairs replay four real rounds from a 126-token prompt. All paired committed-token
lists match. Splitting increases total GPU time by about 0.9% / 1.8%, so stage
medians are approximate and do not sum exactly to the unsplit total. Split host
wall time is not used to attribute production latency.

Drafting costs about **27.6 ms untuned, 16.7 ms with cached choices**, or roughly
3.9 / 2.4 ms per proposal. Most of the draft cost is matrix projections, not
acceptance or rollback. Cached choices select better row/K splits and launch
geometries, fuse draft normalization, and restore the target's K-split geometry
that enables the persistent fused projection schedule. The program shrinks from
1,892 to 1,493 dispatches. No kernel implementation changed for this probe.

A separate full-generation check uses the original six short-context prompts,
128 output tokens, one excluded warmup and five measured requests. It measures
**35.83 ms per full round**, down **33.1%** from 53.57; all rounds use N=7.
Median decode latency is 16.63 ms/output token. Only two of five measured output
texts match the untuned run, so per-token gains also include changed acceptance;
this is not an accuracy validation. The 1,023-token tuned case remains unmeasured. Tuned 4K/8K and plain decoding
are covered by the [long-context follow-up](qwen8b-long-context-tuning.md). The historical 17.75 ms number was tuned
fixed-T target decoder work alone, not the target stage of the 53.57 ms run.

[Raw stage samples](https://github.com/jiazhihao/mpk-apple/blob/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929/round-stage-profile.json),
[cached-choice stage samples](https://github.com/jiazhihao/mpk-apple/blob/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929/round-stage-profile-tuned.json),
[full-generation records](https://github.com/jiazhihao/mpk-apple/blob/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929/round-cached-generation.json)
and [summary](../../tools/bench/results/qwen8b-serving-decode-20260929/round-profile-summary.json)
preserve this audit. The [probe](../../tools/bench/profile_spec_round.py) accepts
`--model`, `--pack`, `--drafter`, `--drafter-pack`, `--prompts` and `--out`;
packs must support 4,608 context tokens. Run once without `--cache`, then once
with `--cache /tmp/qwen8b-serving-cached-choices.json`.
The archived cache retains choices and their timing fields, omitting the unused
candidate-variant history. No runtime cache search is performed by this probe.

## Measurement and comparability

One engine was resident at a time, with one serial request. Every shape had one
excluded warmup and five measured prompt variants. A single raw Qwen chat
template disabled thinking. Sampling was greedy, repetition penalty was one
where applicable, and natural EOS remained enabled. Every measured request
returned all 128 tokens and reported the expected input count. Weights were warm,
but prompts were freshly evaluated; there were no prompt-cache hits in the
accepted Ollama results.

Decode counters are taken directly from each engine:

| Engine | Timer used, divided by 127 |
|---|---|
| Monolith | `Generation.decode_wall_ms`, with `decode_tokens == 127` |
| vLLM | Per-request delta of `vllm:request_decode_time_seconds_sum` × 1000; completion-count delta must equal one |
| llama.cpp | `timings.predicted_ms`; its reported per-token time confirms the 127-token denominator |
| Ollama | `eval_duration / 1e6`; the bundled server's native decode timing confirms the denominator |

vLLM defines decode time as last-token minus first-token time. Monolith times its
native decode pump, including host waits; llama.cpp/Ollama expose native decode
timers. These internal boundaries differ slightly and are not client-observed
streaming inter-token latency. Speculative results are amortized per committed
output token, not latency for every individual burst. HTTP wall time is retained
in the raw rows but is not used in the table.

The plain/speculative generated text matches exactly on only 3/15 Monolith,
2/15 vLLM, and 11/15 llama.cpp measured prompts. Shape-dependent arithmetic and
prefill differences have not been isolated here. Consequently these are latency
measurements, **not a token-equivalence or accuracy validation**. All text is
retained so that divergence can be inspected.

The task is a long-form library-design recommendation prompt with request-number
variants, not a representative workload suite. Engine runs were sequential blocks,
not thermally interleaved. Five samples do not support tail-latency claims. Plain
sample ranges do not overlap between Monolith and vLLM at any measured context,
but this is a configuration screen, not an exhaustive tuning or universal ranking.

## Versions and reproduction

- vLLM-Metal 0.30.0, vLLM 0.30.0+cpu, MLX 0.32.1, mlx-lm 0.32.0 at
  `9e6acca691e64d6d8bb808c328fcdea459099cca`. The `+cpu` wheel designation is
  packaging; the measured backend uses MLX on the Apple GPU.
- llama.cpp `b11146`, commit `7fe450e19`, reporting `0.5.0-dev`.
- Ollama 0.34.4, with all 37/37 model layers offloaded to Metal.
- GGUF target: `Qwen/Qwen3-8B-GGUF`, revision
  `7c41481f57cb95916b40956ab2f0b139b296d974`, `Qwen3-8B-Q4_K_M.gguf`.
  SHA256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.

The [benchmark client](../../tools/bench/single_request_latency.py) also provides
a benchmark-only HTTP adapter around Monolith's `Session.generate`.
The [raw evidence archive](https://github.com/jiazhihao/mpk-apple/tree/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929)
contains every 128-token sample (including warmups), exact raw prompts, summary
ranges, compact native log evidence, server commands, environment settings,
model identities and representative packed-weight layouts. The archive normalizes
engine labels and derives `decode_ms_per_token` from the preserved native counters.

Start one server using its command in `methodology.json`, substituting local
installation and model paths. Then run, for example:

```bash
python tools/bench/single_request_latency.py \
  --kind vllm --model qwen3-8b --label vllm-decode-plain \
  --url http://127.0.0.1:18101 \
  --prompts /tmp/qwen8b-serving-prompts.json \
  --counts 128 --out /tmp/vllm-decode.jsonl
```

Important configuration details are recorded rather than silently left to defaults:
vLLM uses one sequence, prefix caching off, BF16 and GPU memory utilization 0.5
(verified 3.94 GiB KV); plain uses default async scheduling and N=7 synchronous
scheduling. llama.cpp uses one slot, full GPU offload, flash attention, no saved
prompt cache, and `cache_prompt=false` per request. Ollama disables saved prompt
caching and receives an untimed unrelated one-token primer before each request
to replace the active slot; nonzero cached-prompt counts abort the client.

All engines allow 4,608 context tokens. The original Monolith table uses commuted
normalization with autotuning disabled, including cached choices; that is not the
normal Session default. The cache-reuse audit above is reported separately. Its prior target/draft packed weights were preserved;
only longer RoPE tables were appended, with byte-identical old prefixes. Plain
prefill chunks are 128 tokens. N=7 uses 64-token prefill chunks because the
128-token speculative prefill arena exceeded the GPU working set before any
sample was produced. This changes prefill configuration, which is excluded from
the latency table, but may affect numerical agreement. Draft gamma and fixed
verification length remain seven. The archive records discarded cache/memory
setup attempts; their samples are excluded.
