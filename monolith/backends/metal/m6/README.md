# Apple M6 (12 GPU cores)

Measured on Apple11, 32 GB RAM, macOS 27.0.1 (2026-10-13). Selection requires
the exact chip name, GPU family and core count; other M6 variants are not
qualified by this profile. All Metal templates come from `kernels/common/`.

The profile enables tensor acceleration with the existing interleaved16 weight
layout. BF16 projections retain the shader path below 16 rows. Decode and
speculative verification retain v1 attention. Qualified long-context prefill
uses the M6 matrix attention variant described below. `config_kernels.json` contains the qualified
per-shape launch choices. No M5 fusion recipe or speculative cost table is used.

Matching dynamic T=8/32/128 attention programs use 24 threadgroups of eight
SIMD groups. The `GQA_CHUNK_SG` macro preserves the original 144-SIMD softmax
partition in both core and merge, independently of the new launch. The policy
is limited to target heads/KV heads/dimension 24/4/256 and seven-proposal draft
32/8/128; other shapes and static programs keep existing launches.

Serving defaults to BF16 KV, 128-token prefill and an 8192-token speculative
context cap. On this machine, BF16 16K generation works but host prefix caching
can induce swapping; BF16 32K with 128-row prefill exceeds GPU residency.

Opting into `--kv-cache-dtype turbo4` or `--kv-cache-dtype q4_0` permits up to 65536 tokens. Automatic
prefill uses 128 rows through 32768 context and 32 rows above it. The 64K policy
requires 32-row prefill; larger chunks exceed the measured memory budget, while
smaller chunks lose scratch reuse. Short requests use the same dynamic graph.
Above 8192, host/disk prefix snapshots are disabled; `--prefix-cache-dir` is
rejected so snapshots cannot consume the remaining RAM. Q8_0 on one side and Q4_0/Turbo4 on the other permits 32768 with 128-row
prefill and no long-prefix snapshots. Symmetric Q8_0 and pairs containing BF16
keep the existing speculative context policy. See [KV quantization](../../../../docs/design/kv-quantization.md)
for rotated-domain attention, numerical checks and full-model benchmarks.

## Validation and performance [M]

The workload was `nvidia/Qwen3.8-27B-NVFP4` with the NVFP4 DSpark draft.
Fresh-process A/B/B/A runs compare the prior 12-by-12 attention launch with
24-by-8, holding weights, projections, softmax partitions and output tokens
constant. Each request uses an 8,064-token prompt and generates 128 tokens,
with prefix caching disabled. Timings exclude compilation and shader validation.

| Run | Prefill GPU time | Decode throughput | Total wall time |
| --- | ---: | ---: | ---: |
| A1 | 65.308 s | 13.175 tok/s | 74.977 s |
| B1 | 63.717 s | 13.604 tok/s | 73.080 s |
| B2 | 63.771 s | 13.581 tok/s | 73.152 s |
| A2 | 65.323 s | 13.122 tok/s | 75.028 s |

There is no short-context speedup claim. At position 128, alternating fixed-state
rounds took 184.39/185.02, 185.04/185.75 and 185.83/186.32 ms (A/B), a small
short-context cost. The original 1,024-token nonstreaming HTTP request completed
in 57.05 seconds (17.95 tok/s), with the same message as the previous version.

Regression coverage checks device selection, serving limits, recipe identity,
core/merge parameters, exact target/draft outputs and cache writes, and the
small-K NVFP4 Markov-head schedule. GPU checks use Metal shader validation.
The backend runtime does not import benchmark tools or research data.

## Matrix attention for prefill

`attention_prefill_mma` enables the shape-scoped M6 path for dynamic target
programs with 32, 64 or 128 compiled tokens, 24 query heads, 4 KV heads, D=256
and Q/K normalization. Both caches must be BF16, Q8_0 or Turbo4 of the same
format. The runtime selects MMA when at least 8 query tokens are active and
the context including those tokens is at least 1024. Complementary predicates
retain v1 for shorter contexts/tails. Decode/verification, draft attention,
Q4_0, mixed formats, other shapes and other chips keep their existing routes.
Exact large-prefill sessions retain v1 for both the large pass and its 128-row
reference graph, preserving their reduction-order contract.
Explicit `--prefill-attention v1` on `generate`, or disabling the profile's
`attention_prefill_mma`, selects the original attention path. The independent
NVFP4 projection policy below is unaffected by that attention override.

The original D=256 MMA uses fixed 32-key softmax partitions and doubles the
partial workspace. The M6 variant instead uses the v1 16/32/64-key partition
rule and GQA_CHUNK_SG=144. Its original deterministic merge/gate remains the
fallback; qualified BF16/Turbo4 workloads use the probability-partial path below.
BF16 uses 16 query/head rows by 32 keys with BF16 operands; quantized caches
use 8 rows by 16 keys with FP32 operands. Both accumulate in FP32. Quantized
blocks are reconstructed only in the threadgroup tile. Turbo4 reuses the
prepared FP32 rotated queries and inverse-rotates only after the final merge.
No full historical BF16 buffer, extra query allocation or larger partial
workspace is introduced. Floating-point matrix reduction order can differ;
existing score/probability/output rounding stages and test thresholds remain.
The matrix library excludes duplicate scalar/prepare/merge entry points; those
operations keep their separate existing pipelines.

The production serving context default remains 8192 and the cache dtype default
remains BF16. Long-context opt-ins and prefix-cache policy are unchanged.
See [M6 matrix prefill qualification](../../../../docs/research/m6-mma-prefill-20261014.md)
for repeated timings, numerical deviations and full-model results.

## Qualified projection and probability partials

`nvfp4_prefill_geometry` selects measured target NVFP4 MLP layouts for dynamic
prefill on this exact device. Gate/up (N=34816, K=5120) uses TM/TN/TK=32/32/128,
24 groups of four SIMD groups for compiled bounds 64/128. Down (N=5120, K=17408)
uses 32/16/128, 48 groups of eight SIMD groups for bounds 32/64/128. The 32-row
gate/up, draft, verification, unknown shapes and custom pack/launch layouts retain
their original geometry. TK and the K reduction order are unchanged.

`attention_probability_partials` applies only to the qualified symmetric BF16
and Turbo4 MMA path. At least 16 active rows and 1024 attended tokens are required.
It writes the existing BF16-rounded probabilities, then reconstructs PV tiles
and merges partitions in their original order, including FP32 correction terms,
Turbo4 inverse rotation, normalization, rounding and gating. Original MMA handles
8–15-row tails; v1 handles shorter rows/contexts. Q8_0 retains the original MMA
partial/merge, while Q4_0 and mixed KV retain v1. Both controls can be disabled
independently in a profile for comparisons.

The probability path reduces logical partial-buffer traffic by 8× at the measured
long-context partition size. It reuses the existing fallback allocation, so peak
managed GPU bytes do not decrease. It adds two predicated operations per target
layer; there remain two active heavy stages. Full-model qualification, memory and
swap, numerical checks, rejected candidates and the measured decode limitation
are recorded in [the projection/partial report](../../../../docs/research/m6-projection-partials-20261014.md).

## Experimental tiled PV and decode audit

The optional profile setting `engine.attention_pv_tiling` is absent/disabled in
the production profile. It requires `attention_probability_partials` and selects
a 16-row, eight-SIMD-group PV tile for symmetric Q8_0. Turbo4 uses that tile only
with at least 64 active query tokens; shorter runtime tails and the 64K/32-row
program retain the original layout. BF16, decode/verification, Q4_0, mixed KV and
other chips keep their existing selection. The wider tile reuses KV scratch for
bounded output readout and adds no device intermediate allocation.

Kernel and full-model outputs matched exactly, but none of the 8K–32K whole-model
current/candidate pairs were swap-free on both sides. The candidate therefore
does not qualify for the default profile. Five alternating old/current decode
pairs per workload and fixed-prefix replays did not reproduce a stable decode
code-path regression, so projection and decode dispatch remain unchanged.
See [the decode/PV investigation](../../../../docs/research/m6-decode-pv-20261015.md)
for stage timings, memory, numerical checks, rejected candidates and the limits
of timestamp-only profiling.
