# M6 matrix attention prefill qualification — 2026-10-14

Apple M6, 12 GPU cores, 32 GiB unified memory. Model: `nvidia/Qwen3.8-27B-NVFP4`, NVFP4 DSpark, 7 proposals. BF16 cache and the production 8192-token context default are unchanged.

## Implementation and dispatch

The original D=256 MMA uses 16 query/head rows, 32-key tiles, eight SIMD groups, BF16 operands and FP32 accumulators. Its fixed 32-key softmax partition doubles partial memory relative to v1. The new M6 kernel preserves v1’s 16/32/64-key partition selection and unchanged core/merge buffer layout. BF16 tiles use 16 rows × 32 keys; Q8_0 and Turbo4 use 8 rows × 16 keys with FP32 operands to avoid adding BF16 rounding to reconstructed KV or rotated Q. Eight SIMD groups per threadgroup, 48 threadgroups. BF16 threadgroup storage is 30 KiB; quantized storage is 28 KiB.

Only the exact M6 target shape (24 query heads, 4 KV heads, D=256, Q/K normalization) with compiled dynamic prefill bounds 32/64/128 qualifies. Runtime MMA requires at least 8 active query tokens and context ≥1024. Complementary predicates retain v1 for shorter requests and prompt tails. Exact large-prefill sessions and their 128-row reference graphs, decode, speculative verification, draft attention, Q4_0, mixed KV formats and all other backend selection policies remain unchanged. `generate --prefill-attention v1` explicitly disables the route. The M6 profile flag `attention_prefill_mma` controls automatic deployment.

Q8_0 and Turbo4 reconstruction is fused into threadgroup tile reads. Turbo4 Q uses the existing FP32 prepared rotation buffer, K/V remain compressed in device memory, and the existing merge performs the single inverse rotation after accumulation. Stored FP16 scales, norm correction, causal masking, score/probability rounding stages and output/gate ordering remain unchanged. Matrix and FP32 denominator reductions reassociate additions; bitwise output identity with v1 is not claimed.

## Kernel timings

Times below are milliseconds for preparation (Turbo4), core and merge together, excluding compilation. Each entry pools 10 samples from each of two opposite-order sweeps after 200 ms warmup. The standalone v1 comparator uses the measured 24-worker × 8-SIMD launch at all row counts; production single-row and 64-row v1 can retain their existing launch. `T` counts new query tokens, before GQA replication.

BF16: **v1 / original MMA / new M6 MMA**, ms.

| T | 2K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|
| 1 | 0.117 / 0.139 / 0.153 | 0.620 / 0.572 / 0.632 | 0.963 / 1.238 / 0.918 | 1.516 / 1.658 / 1.386 |
| 8 | 0.598 / 0.322 / 0.249 | 2.533 / 1.355 / 1.060 | 5.100 / 2.790 / 2.139 | 10.235 / 5.607 / 4.293 |
| 32 | 2.538 / 1.348 / 1.067 | 10.110 / 5.360 / 4.298 | 20.368 / 10.799 / 8.535 | 40.467 / 22.021 / 17.056 |
| 64 | 5.216 / 2.613 / 2.061 | 20.527 / 10.894 / 8.378 | 40.952 / 22.080 / 16.696 | 81.865 / 43.677 / 33.962 |
| 128 | 10.839 / 5.467 / 4.385 | 41.463 / 21.649 / 16.857 | 83.452 / 43.396 / 33.785 | 171.398 / 87.438 / 67.803 |

q8_0: **v1 / new M6 MMA**, ms.

| T | 2K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|
| 1 | 0.126 / 0.107 | 0.394 / 0.323 | 0.803 / 0.568 | 1.562 / 1.137 |
| 8 | 0.658 / 0.402 | 2.657 / 1.642 | 5.365 / 3.312 | 10.732 / 6.798 |
| 32 | 3.012 / 1.810 | 10.809 / 6.689 | 21.584 / 13.250 | 42.762 / 26.271 |
| 64 | 6.840 / 4.043 | 22.547 / 13.714 | 43.524 / 26.773 | 87.298 / 53.944 |
| 128 | 17.762 / 9.615 | 50.350 / 28.688 | 94.842 / 54.619 | 186.535 / 106.267 |

turbo4: **v1 / new M6 MMA**, ms.

| T | 2K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|
| 1 | 0.131 / 0.082 | 0.426 / 0.276 | 0.920 / 0.561 | 1.573 / 1.116 |
| 8 | 0.652 / 0.398 | 2.664 / 1.620 | 5.393 / 3.274 | 10.852 / 6.564 |
| 32 | 2.849 / 1.729 | 10.854 / 6.599 | 21.715 / 13.231 | 43.267 / 26.203 |
| 64 | 6.174 / 3.623 | 22.332 / 13.225 | 44.170 / 26.208 | 87.815 / 52.629 |
| 128 | 14.713 / 8.035 | 47.623 / 26.780 | 93.645 / 52.300 | 184.802 / 103.137 |

At 8–128 query tokens all tested 2K–32K cases improve. BF16 single-row results are inconsistent, including regressions; production decode retains v1. Quantized single-row microbenchmarks improve but are not sufficient to change speculative decode policy.

## Attention stages at 32K / 128 query tokens

Isolated dispatch medians, ms (six observations). Separate command buffers add launch effects, so these columns need not sum to the complete-dispatch timing.

| Cache | Path | Prepare Q | Core | Merge |
|---|---|---:|---:|---:|
| bf16 | v1 | 0.000 | 161.005 | 11.703 |
| bf16 | m6 | 0.000 | 56.176 | 11.409 |
| q8_0 | v1 | 0.000 | 176.695 | 11.388 |
| q8_0 | m6 | 0.000 | 95.012 | 11.625 |
| turbo4 | v1 | 0.050 | 170.821 | 11.522 |
| turbo4 | m6 | 0.050 | 92.967 | 11.293 |

The 1024-context activation boundary was separately repeated for T=8/32/64/128 and all three formats; every measured case improved. Raw threshold timings are in the JSON.

## Workspace and numerical deviations

No additional full-history KV or partial buffers are allocated. At 32K plus 128 new tokens, the standalone partial allocation is 1,629,536,256 bytes for both v1 and the M6 path; the original MMA needs 3,259,072,512 bytes. Turbo4’s existing prepared Q is 3,145,728 bytes. Production uses the actual capacity, including DSpark headroom, rather than this benchmark’s context + T.

| Cache | Max additional relative L2 versus own v1 | Max absolute error |
|---|---:|---:|
| bf16 | 0.02080% | 0.00024414 |
| q8_0 | 0.03135% | 0.00048828 |
| turbo4 | 0.02728% | 0.00024414 |

The quantization error relative to BF16 is distinct from the small additional MMA error; both are recorded in the JSON. Kernel checks retain cosine >0.99999 and maximum error ≤2 BF16 ULPs at output scale, plus the existing quantized attention rtol=0.035/atol=0.004. No existing threshold is weakened.

## Full-model A/B

Each request has context capacity minus 128 prompt tokens, then 128 greedy output tokens, without prefix reuse. Compilation and short warmup are excluded. v1 and MMA use fresh processes, identical local packs and prompt hashes. All 16K cases repeat the full request twice; the table uses medians. Prefill tokens/s uses GPU time; decode tokens/s uses decode wall time. End-to-end is generation request wall time, not process startup or HTTP latency.

| Context | Cache | Path | Prefill GPU s | Prompt tok/s | Decode tok/s | Request s | Accepted/round | Managed GPU GiB | Swap-out MiB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | bf16 | v1 | 63.52 | 126.96 | 13.62 | 72.87 | 2.154 | 22.458 | 0.0 |
| 8K | bf16 | m6 | 53.37 | 151.09 | 14.31 | 62.27 | 2.324 | 22.458 | 0.0 |
| 8K | q8_0 | v1 | 74.43 | 108.34 | 15.13 | 82.86 | 2.559 | 22.421 | 0.0 |
| 8K | q8_0 | m6 | 64.49 | 125.04 | 13.99 | 73.59 | 2.324 | 22.421 | 0.0 |
| 8K | turbo4 | v1 | 71.10 | 113.42 | 15.30 | 79.45 | 2.676 | 22.289 | 0.0 |
| 8K | turbo4 | m6 | 62.46 | 129.10 | 15.11 | 70.89 | 2.676 | 22.289 | 0.0 |
| 16K | bf16 | v1 | 168.61 | 96.41 | 11.33 | 179.87 | 2.237 | 23.188 | 0.0 |
| 16K | bf16 | m6 | 125.10 | 129.95 | 12.98 | 134.93 | 2.667 | 23.188 | 0.0 |
| 16K | q8_0 | v1 | 191.30 | 84.98 | 11.39 | 202.51 | 2.270 | 22.918 | 0.0 |
| 16K | q8_0 | m6 | 156.01 | 104.20 | 12.73 | 166.04 | 2.667 | 22.918 | 0.0 |
| 16K | turbo4 | v1 | 186.97 | 86.94 | 12.73 | 197.00 | 2.788 | 22.648 | 0.0 |
| 16K | turbo4 | m6 | 152.53 | 106.57 | 12.41 | 162.82 | 2.676 | 22.648 | 0.0 |
| 32K | bf16 | v1 | 539.22 | 60.53 | 9.26 | 553.14 | 2.500 | 24.563 | 488.6 |
| 32K | bf16 | m6 | 360.42 | 90.56 | 8.76 | 375.11 | 2.361 | 24.563 | 1480.6 |
| 32K | q8_0 | v1 | 552.06 | 59.12 | 9.28 | 565.89 | 2.606 | 24.620 | 0.0 |
| 32K | q8_0 | m6 | 419.96 | 77.72 | 8.79 | 434.55 | 2.600 | 24.620 | 1022.1 |
| 32K | turbo4 | v1 | 538.34 | 60.63 | 9.04 | 552.51 | 2.647 | 24.077 | 0.0 |
| 32K | turbo4 | m6 | 411.94 | 79.23 | 9.31 | 425.69 | 2.758 | 24.077 | 0.0 |
| 64K | turbo4 | v1 | 1936.40 | 33.78 | 5.82 | 1958.90 | 2.529 | 24.546 | 0.0 |
| 64K | turbo4 | m6 | 1524.34 | 42.91 | 5.84 | 1546.68 | 2.529 | 24.546 | 0.0 |

BF16 32K uses 64-row prefill to fit the working-set budget; Q8_0/Turbo4 through 32K use 128 rows. Turbo4 64K uses 32 rows. GPU memory is the maximum sum of unique managed session buffers at snapshots, excluding opaque driver allocations; peak process RSS is reported separately below and in JSON. Swap counters are system-global deltas, not per-process attribution. Decode kernels are unchanged; changes in measured decode throughput can reflect speculative acceptance changes after prefill.

## Matched prompt-end logits

Captured after the final prompt chunk and before speculative decode, so both paths have exactly the same token history. Matrix reduction changes can accumulate across the full hybrid model; these are separate from the much smaller single-kernel deviations. Existing model cosine >0.999 and a conservative max-absolute ≤reference scale/32 gate are applied, with matching first-token argmax.

| Context | Cache | Cosine | Relative L2 | Max absolute | First token matches |
|---:|---|---:|---:|---:|---|
| 16K | bf16 | 0.999786179 | 2.1239% | 0.243164 | True |
| 16K | q8_0 | 0.999821367 | 1.8963% | 0.250000 | True |
| 16K | turbo4 | 0.999354272 | 3.5936% | 0.656250 | True |
| 32K | bf16 | 0.999490208 | 3.2796% | 0.406250 | True |
| 32K | q8_0 | 0.999741047 | 2.2873% | 0.250000 | True |
| 32K | turbo4 | 0.999062819 | 4.4556% | 0.445312 | True |
| 64K | turbo4 | 0.999218582 | 3.9970% | 0.500000 | True |

16K repeated requests are checked for identical full output tokens and acceptance traces within each path. Free-running v1 and MMA continuations can differ after later tokens; no exact-sequence equivalence or broad model-quality claim is made.


## Validation and limits

- 288 Metal shader-validation tests passed: 60 new matrix attention cases, 3 integrated M6 DSpark/rollback/host-and-disk restore cases, and 225 existing attention/codec regressions. The 3 integration cases were rerun after the final dispatch changes, including the single-token v1 tail fallback. Packed cache bytes match v1 exactly for identical kernel inputs, causal future rows cannot influence earlier outputs, and repeated invocations are bit stable.
- Full contract suite: 1388 passing in the final unrestricted local regression run, 1 skipped, 2 pre-existing Transformers oracle incompatibilities (`Qwen3MoeRotaryEmbedding` no longer accepts `device`). Backend selection and added dispatch tests pass. Other chips retain their existing source branches and policies; no physical M3/M4/M5 device run is claimed.

## Final library memory repeat and 64K limits

The final matrix library excludes unused exported scalar/prepare/merge kernels. The existing separate pipelines remain. A fresh-process BF16 32K repeat produced bit-identical prompt-end logits, all generated tokens and the complete DSpark acceptance trace relative to the first MMA run. Host RSS varies with system paging state; removing exports is not proof of the root cause of the earlier swap events.

| BF16 32K run | Prefill s | Decode tok/s | Peak process RSS GiB | Timed swap-out MiB |
|---|---:|---:|---:|---:|
| v1 | 539.22 | 9.26 | 4.342 | 0.0 |
| MMA initial library | 360.42 | 8.76 | 4.789 | 1480.6 |
| MMA final library | 359.70 | 8.75 | 4.583 | 0.0 |

64K Turbo4, 32-row prefill: fresh v1 1936.40 s / 5.82 tok/s; current MMA 1524.34 s / 5.84 tok/s. The prompt SHA256, packs, token counts and proposal count match. Both paths ran in fresh processes in this session and include matched prompt-logit captures. All 128 generated tokens match; mean accepted proposals per round is 2.529 on both paths, although individual acceptance traces differ. This is one request per path; the 16K comparisons have repeated requests. The older 2026-10-13 v1 run is retained separately in JSON for context.

64K allocation plans at 32-row prefill (before any model GPU allocation): BF16 27.464 GiB, Q8_0 25.639 GiB, Turbo4 24.546 GiB; the device recommends 24.960 GiB. BF16 and Q8_0 were not run at 64K because their plans exceed that budget.

GPU buffer figures exclude opaque driver/compiler allocations. System-global swap-out can include unrelated applications. The first BF16 and Q8_0 32K MMA runs had swap traffic despite identical managed buffers; those observations remain in the tables and JSON.

Kernel quantization error versus BF16 is separate from additional matrix reduction error: the worst relative L2 across the grid is Q8_0 0.754% and Turbo4 11.197% (including quantization); the additional errors versus each format’s v1 implementation are below 0.032%.

## Production HTTP smoke test

A temporary `serve` instance with the original BF16 dtype and 8192-token context default compiled the M6 matrix path, completed a >1024-token non-streaming request and a sampled streaming request, and rejected an overflowing request with HTTP 400. The 64K Turbo4 opt-in also completed a long request with 32-row prefill and prefix snapshots disabled, and rejected context overflow. Both temporary instances were stopped after testing. Detailed response timings and resident-state metadata are in the JSON.

## Remaining costs

Projection/weight reads and hybrid recurrent layers remain substantial, so kernel speedups exceed whole-prefill speedups. The FP32 quantized operand path needs smaller tiles than BF16; tile scheduling and partial-output/merge traffic remain candidates for improvement. Decode/verification matrix routing, Q4_0 and mixed-format MMA require separate end-to-end qualification. These synthetic repeated-note prompts measure performance and numerical stability; they are not a broad downstream model-quality evaluation.

Raw reproducible measurements: [JSON](../../tools/bench/results/m6-mma-prefill-20261014.json). Benchmark entrypoints: [kernels](../../tools/bench/m6_attention.py), [model](../../tools/bench/m6_prefill_model.py).
