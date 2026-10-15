# M6 NVFP4 projection and probability-partial attention qualification

This report follows the [prefill bottleneck investigation](m6-prefill-bottlenecks-20261014.md).
The baseline already includes the qualified M6 matrix attention path. It is **not**
the original scalar v1 baseline. The changes preserve the BF16 cache default,
8192-token production serving limit, and M3/M4/M5 configuration.

The M6 profile enables the qualified projection layout and probability-partial path by default after the long-context full-model checks below. Explicit profile flags retain independent fallback controls.

## Full-model measurements

Apple M6, Apple11, 12 GPU cores, 32 GiB unified memory; target
`nvidia/Qwen3.8-27B-NVFP4`, NVFP4 DSpark draft, seven proposals, greedy decoding,
128 output tokens. Context labels denote capacity; prompt length is capacity
minus 128. Prompt caches are disabled during timing. Compilation, weight load,
and warmup are excluded. Decode throughput uses decode wall time. Request wall
time includes prompt-logit capture but excludes server/network/startup latency.

Every candidate has two uncached full requests. Controls use fresh measurements
plus earlier same-device, same-profile, same-prompt measurements where noted in
the JSON; their token, acceptance and logit bytes were checked before pooling.
Values are mean ± population standard deviation, not confidence intervals.
The 8K–32K matrix used the initial probability-partial policy; full-chunk arithmetic
is unchanged in the final policy, which restores original MMA for 8–15-row tails.
The 64K candidate runs use final source. Additional repeated final-policy 8K
measurements below check the extra predicated operations and restored prefixes.
One model/GPU benchmark ran at a time. An accidentally overlapping short control
and tail probe were excluded, then repeated without GPU contention; no excluded
sample appears below. Performance runs did not enable shader validation.

| Context–KV | Chunk | Control N | Control Prefill s | Projection Prefill s | Projection + partials s | Final time reduction |
| --- | --- | --- | --- | --- | --- | --- |
| 8k-bf16 | 128 | 2 | 53.512 ± 0.139 | 43.742 ± 0.223 | 42.374 ± 0.250 | 20.8% |
| 8k-q8_0 | 128 | 2 | 64.393 ± 0.097 | 54.106 ± 0.025 | Original attention retained | 16.0% |
| 8k-turbo4 | 128 | 2 | 62.608 ± 0.112 | 52.823 ± 0.027 | 50.332 ± 0.043 | 19.6% |
| 16k-bf16 | 128 | 2 | 126.921 ± 0.022 | 106.587 ± 0.308 | 99.220 ± 0.549 | 21.8% |
| 16k-q8_0 | 128 | 2 | 156.021 ± 0.147 | 135.784 ± 0.059 | Original attention retained | 13.0% |
| 16k-turbo4 | 128 | 2 | 152.788 ± 0.263 | 132.004 ± 0.021 | 125.040 ± 0.028 | 18.2% |
| 32k-bf16 | 64 | 3 | 360.008 ± 0.303 | 316.789 ± 0.335 | 295.542 ± 0.242 | 17.9% |
| 32k-q8_0 | 128 | 2 | 421.043 ± 1.087 | 378.391 ± 0.123 | Original attention retained | 10.1% |
| 32k-turbo4 | 128 | 2 | 411.951 ± 0.012 | 374.253 ± 0.211 | 354.686 ± 0.014 | 13.9% |
| 64k-turbo4 | 32 | 2 | 1531.012 ± 6.670 | 1457.656 ± 0.138 | 1381.113 ± 0.048 | 9.8% |

### Decode, latency and acceptance

Cells list control / projection / combined (where applicable). Acceptance is identical for all variants.

| Context–KV | Decode tok/s | Request wall s | Prefill tok/s | Accepted proposals/round |
| --- | --- | --- | --- | --- |
| 8k-bf16 | 14.288 ± 0.026 / 14.205 ± 0.037 / 14.226 ± 0.045 | 62.428 ± 0.155 / 52.711 ± 0.200 / 51.329 ± 0.279 | 150.70 / 184.36 / 190.31 | 2.324324 |
| 8k-q8_0 | 13.971 ± 0.019 / 14.029 ± 0.013 | 73.511 ± 0.084 / 63.187 ± 0.017 | 125.23 / 149.04 | 2.324324 |
| 8k-turbo4 | 15.197 ± 0.097 / 15.142 ± 0.010 / 15.281 ± 0.001 | 70.993 ± 0.165 / 61.236 ± 0.022 / 58.670 ± 0.042 | 128.80 / 152.66 / 160.22 | 2.676471 |
| 16k-bf16 | 12.934 ± 0.071 / 12.819 ± 0.035 / 13.011 ± 0.005 | 136.796 ± 0.025 / 116.540 ± 0.336 / 109.034 ± 0.543 | 128.08 / 152.52 / 163.84 | 2.666667 |
| 16k-q8_0 | 12.698 ± 0.014 / 12.674 ± 0.052 | 166.076 ± 0.156 / 145.855 ± 0.099 | 104.19 / 119.72 | 2.666667 |
| 16k-turbo4 | 12.365 ± 0.027 / 12.402 ± 0.002 / 12.404 ± 0.002 | 163.109 ± 0.287 / 142.291 ± 0.026 / 135.323 ± 0.031 | 106.40 / 123.15 / 130.01 | 2.676471 |
| 32k-bf16 | 8.752 ± 0.004 / 8.678 ± 0.002 / 8.710 ± 0.036 | 374.706 ± 0.303 / 331.629 ± 0.331 / 310.320 ± 0.191 | 90.66 / 103.03 / 110.44 | 2.361111 |
| 32k-q8_0 | 8.787 ± 0.005 / 8.780 ± 0.002 | 435.643 ± 1.096 / 392.987 ± 0.126 | 77.52 / 86.26 | 2.600000 |
| 32k-turbo4 | 9.314 ± 0.000 / 9.219 ± 0.033 / 9.177 ± 0.005 | 425.700 ± 0.011 / 388.322 ± 0.435 / 368.635 ± 0.020 | 79.23 / 87.21 / 92.03 | 2.757576 |
| 64k-turbo4 | 5.840 ± 0.001 / 5.835 ± 0.001 / 5.839 ± 0.001 | 1553.378 ± 6.696 / 1480.025 ± 0.136 / 1403.471 ± 0.043 | 42.72 / 44.87 / 47.36 | 2.529412 |

The 32K Turbo4 comparison records 9.314 → 9.177 decode tok/s (about 1.5%
lower), despite identical tokens and acceptance. Decode dispatch selection is
unchanged; this measurement does not establish its cause. No decode speedup is
claimed. The final-policy 8K recheck also recorded lower Decode throughput;
its fresh-control comparison is included under validation. The end-to-end request still benefits from the larger Prefill reduction.

### Memory and swap

Managed GPU bytes count distinct runtime buffers, including mapped weights and
reused scratch, but excluding opaque driver allocations. Process peak RSS and physical footprint are separate OS counters;
they do not measure total GPU residency and must not be added to buffer bytes.
Peak counters include process compilation/load/warmup and are sampled after
measured requests, before the separate prefix restore check. Physical footprint was not recorded by older controls. Swap
counters are system-wide, not attributable exclusively to this process; the JSON
separates load/warmup, timed requests and the complete process interval.

Cells list control / projection / combined. Timed swap-out is cumulative over the listed repeated requests and source processes; see JSON for individual intervals.

| Context–KV | Managed GPU GiB | Peak RSS GiB | Peak physical footprint GiB | Timed system swap-out GiB |
| --- | --- | --- | --- | --- |
| 8k-bf16 | 22.458 / 22.458 / 22.458 | 2.666 / 2.665 / 2.666 | 2.600 / 2.599 / 2.600 | 0.000 / 0.000 / 0.000 |
| 8k-q8_0 | 22.421 / 22.421 | 2.579 / 2.579 | 2.525 / 2.523 | 0.000 / 0.000 |
| 8k-turbo4 | 22.289 / 22.289 / 22.289 | 2.447 / 2.458 / 2.299 | — / — / 2.393 | 0.000 / 0.000 / 0.000 |
| 16k-bf16 | 23.188 / 23.188 / 23.188 | 3.017 / 3.300 / 3.289 | — / — / 3.441 | 0.000 / 0.000 / 0.000 |
| 16k-q8_0 | 22.918 / 22.918 | 3.020 / 3.031 | — / 3.134 | 0.000 / 0.000 |
| 16k-turbo4 | 22.648 / 22.648 / 22.648 | 2.914 / 2.757 / 2.914 | 2.863 / 2.870 / 2.864 | 0.000 / 0.000 / 0.000 |
| 32k-bf16 | 24.563 / 24.563 / 24.563 | 4.789 / 4.650 / 4.790 | 4.973 / 5.076 / 5.013 | 1.446 / 0.000 / 0.000 |
| 32k-q8_0 | 24.620 / 24.620 | 4.655 / 4.569 | 5.029 / 5.029 | 0.998 / 1.349 |
| 32k-turbo4 | 24.077 / 24.077 / 24.077 | 4.542 / 4.305 / 4.305 | 4.496 / 4.499 / 4.498 | 0.000 / 0.000 / 0.000 |
| 64k-turbo4 | 24.546 / 24.546 / 24.546 | 4.917 / 4.268 / 4.509 | 5.251 / 5.249 / 5.250 | 0.000 / 0.000 / 0.000 |

64K plan-only allocations with 32-row prefill were 27.464 GiB BF16,
25.639 GiB Q8_0, and 24.546 GiB Turbo4, against the device's 24.960 GiB
recommended working-set budget. Only Turbo4 was run at 64K. The BF16/Q8_0
entries are budget-based omissions, not observed out-of-memory failures.

## Projection selection

The optimization reuses `gemm_tile` and `projection_geometry`. TK=128, K product
order, packed codes/scales, BF16 operands, FP32 accumulators, epilogue rounding,
residual/SILU and down-projection norm statistics remain unchanged. No weight
repacking or intermediate allocation is introduced.

| Target projection (N × K) | Compiled prompt bound | TM/TN/TK | Threadgroups × SIMD groups |
| --- | --- | --- | --- |
| Gate/up: 34816 × 5120 | 64, 128 | 32/32/128 | 24 × 4 |
| Down: 5120 × 17408 | 32, 64, 128 | 32/16/128 | 48 × 8 |
| Gate/up: 34816 × 5120 | 32 | Existing layout | Existing 12 × 12 |

Selection requires dynamic target prefill, Apple11/12 cores, tensor acceleration,
and the original qualified pack/layout/launch/epilogue. Decode, draft, unknown
shapes, custom layouts and other chips retain their prior choice. All 64 target
down projections select the qualified layout; all 64 gate/up projections do so
only for compiled bounds 64/128. Draft and verification retain their original geometry. Active tails were checked separately from compiled bounds.

| Real layer-0 projection | Active rows | Control ms | Candidate ms | Speedup |
| --- | --- | --- | --- | --- |
| down | 128 | 2.24877 | 1.63515 | 1.375× |
| down | 32 | 1.07201 | 0.56611 | 1.894× |
| down | 64 | 1.55359 | 0.85846 | 1.810× |
| gate_up | 128 | 5.34903 | 3.43913 | 1.555× |
| gate_up | 32 | 1.19778 | 1.60314 | 0.747× |
| gate_up | 64 | 2.52331 | 1.78256 | 1.416× |

Selected layouts also improved all measured active-tail cases (2–128 rows within
the corresponding compiled bound); the complete samples are in the JSON.

The 32-row gate/up candidate was rejected: approximately 1.198 → 1.603 ms.
Disabling its scale cache and changing the crew was slower. The policy therefore
does not globally replace all NVFP4 MMA launches.

## What the 188 ms merge number meant

At position 32384 with a 128-row chunk, attended context is 32512 and target
attention uses 508 partitions. The reported **188.414 ms is the accumulated
merge time across 16 target and 5 draft layers**, not one layer. Target merges
contribute 185.906 ms and draft merges 2.489 ms (medians of separate sums need
not add to the median of the combined sum). Target MMA partials contribute
1468.360 ms. The sigmoid gate and Turbo4 inverse rotation already run inside
the merge kernel; the separate gate projection contributes 13.002 ms.

The following are GPU timestamp measurements from repeated real-prefix
checkpoints. Counter capture inserts encoder boundaries and perturbs scheduling.
These stage sums are diagnostic and must not be substituted for production
request latency or subtracted from whole-ICB time to infer GPU idle time.

| Attended context | Stage (all target/draft layers) | Control ms | Combined ms |
| --- | --- | --- | --- |
| 8192 | Whole ICB (uninstrumented) | 1200.705 | 975.723 |
| 8192 | Target partial / score | 383.671 | 249.859 |
| 8192 | Target merge / PV+merge+gate | 45.076 | 122.655 |
| 8192 | Q preparation | 0.640 | 0.619 |
| 8192 | Gate projection | 12.935 | 12.703 |
| 8192 | Draft attention core | 15.016 | 14.934 |
| 8192 | Draft merge | 0.614 | 0.599 |
| 16384 | Whole ICB (uninstrumented) | 1626.920 | 1355.970 |
| 16384 | Target partial / score | 740.358 | 491.190 |
| 16384 | Target merge / PV+merge+gate | 91.293 | 248.443 |
| 16384 | Q preparation | 0.618 | 0.617 |
| 16384 | Gate projection | 12.888 | 12.831 |
| 16384 | Draft attention core | 30.721 | 30.727 |
| 16384 | Draft merge | 1.256 | 1.268 |
| 32512 | Whole ICB (uninstrumented) | 2481.316 | 2114.842 |
| 32512 | Target partial / score | 1468.360 | 968.236 |
| 32512 | Target merge / PV+merge+gate | 185.906 | 504.207 |
| 32512 | Q preparation | 0.618 | 0.669 |
| 32512 | Gate projection | 13.002 | 13.033 |
| 32512 | Draft attention core | 61.838 | 61.777 |
| 32512 | Draft merge | 2.489 | 2.620 |

The profiled ICB contains 1660 → 1692 encoded operations and 1055 → 1087 barrier-marked operations. These are program counts, not a measured GPU occupancy or launch-latency counter.

The available public Metal counter set exposes only `GPUTimestamp`, and
`xctrace` is unavailable. No hardware occupancy, cache-miss, DRAM-transaction,
register-spill or GPU-stall claim is made from these data.

### Partial traffic and allocation

| Attended context | Partitions | FP32 partial write or read MiB/layer | Probability write or read MiB/layer | MD write MiB/layer |
| --- | --- | --- | --- | --- |
| 8192 | 128 | 384.000 | 48.000 | 3.000 |
| 16384 | 256 | 768.000 | 96.000 | 6.000 |
| 32512 | 508 | 1524.000 | 190.500 | 11.906 |

These are logical shader accesses, not measured DRAM bytes. At 32512 context,
FP32 partial writes plus reads across 16 target layers total 47.625 GiB; rounded
BF16 probabilities total 5.953 GiB, an 8× reduction. FP32 `(maximum, denominator)`
metadata is retained. Maximum scans are distributed across lanes and each row's
correction is computed once; old replicated metadata requests could already
coalesce or hit cache, so their reduction is not a DRAM bandwidth measurement.

The existing maximum-size partial allocation remains available to the original
fallback. Therefore this change reduces traffic, **not peak workspace allocation**.
The target partial workspace at this capacity is 1,626,365,952 bytes and is
reused across layers, not multiplied by 16 in the memory accounting.

## Probability-partial implementation and rejected alternatives

The first stage retains the original adaptive 16/32/64-key partitions, causal
mask, score rounding, BF16 probability rounding, and unrounded FP32 denominator.
It writes rounded probabilities into the existing partial buffer. The second
stage reads packed V by tile, reconstructs each original MMA PV partial, and
folds partitions in increasing index order using the original scale corrections
and ordered FP32 FMAs. It then inverse-rotates Turbo4 once, normalizes, rounds,
applies the rounded sigmoid gate and product, and uses the original output
permutation. The merge's separate gate buffer and parameter strides are retained.
A device barrier makes both probability and fresh V writes visible to stage two.
No full historical BF16 K/V array is materialized.

Only symmetric BF16 and Turbo4 use probability partials. For a qualified compiled
bound of 32/64/128, runtime rows must be at least 16 and attended context at least
1024. Original MMA handles 8–15 rows; v1 handles fewer rows/shorter context.
Q8_0 keeps its original MMA partial/merge, and Q4_0/mixed KV retain v1. Attention for exact-large
prefill, speculative verification and draft remains unchanged.

There are still two active heavy stages. Complementary runtime predicates add
two encoded inactive operations per target layer; this is not a kernel-launch
reduction. The PV stage uses 256 threads versus the original merge's 32.
Source-declared threadgroup arrays are 30 KiB/28 KiB for BF16/quantized score
and 18.25 KiB/18.125 KiB for PV. Actual register occupancy was not measured.

| KV | Query rows | Historical context | Original chain ms | Probability chain ms | Speedup |
| --- | --- | --- | --- | --- | --- |
| bf16 | 128 | 8192 | 16.719 | 13.997 | 1.194× |
| bf16 | 128 | 16384 | 33.646 | 27.847 | 1.208× |
| bf16 | 128 | 32768 | 68.561 | 50.917 | 1.347× |
| q8_0 | 128 | 8192 | 28.664 | 36.319 | 0.789× |
| q8_0 | 128 | 16384 | 54.737 | 71.072 | 0.770× |
| q8_0 | 128 | 32768 | 106.671 | 140.810 | 0.758× |
| turbo4 | 128 | 8192 | 26.853 | 23.778 | 1.129× |
| turbo4 | 128 | 16384 | 52.544 | 46.601 | 1.128× |
| turbo4 | 128 | 32768 | 103.292 | 93.859 | 1.100× |

Rejected alternatives:

- Q8_0 probability partials: the complete attention chain regressed roughly
  27–32%, even though separately warmed stage measurements looked favorable.
- Probability partials for 8-row tails: complete-chain BF16 and Turbo4 timings
  were slower; 16-row and 24-row tails improved, motivating the runtime guard.
- More SIMD groups per merge, parallel maximum reduction alone, row-major
  partial layout and loop unrolling: no stable complete-chain benefit.
- Removing gating/normalization for diagnostic ablation: changes semantics and
  showed no measurable benefit beyond noise; never used in production.

## Correctness and regression validation

- Full-model candidate/control tokens, prompt logits, accepted/committed proposals,
  verification lengths and prompt hashes match exactly within each KV format.
  Maximum absolute logit deviation is **0**. This includes both 64K repeats.
- **142 distinct specialized attention GPU cases passed** with Metal shader
  validation: 60 original-MMA byte comparisons, 16 runtime-tail dispatch cases,
  60 unchanged v1-oracle checks and 6 speculative/checkpoint/disk-restore session
  cases. The six session cases were also repeated after adding the tail fallback.
  Checks cover causal dependence, partition boundaries, gates, cache bytes,
  repeated overwrite, done predicates, rollback and prefix branching.
- **25 real-weight NVFP4 shader-validation cases passed**, across layers 15, 31,
  and 63 and compiled bounds 32/64/128. Projection outputs and FP32 down-norm
  statistics are byte-identical. Validation timings are excluded from performance.
- Final-source **231 Attention/KV/speculative-rollback shader regressions passed**.
  Existing tolerances were not changed.
- Final default-profile contract suite: **1437 passed, 1 skipped, 2 failed**
  (1440 total). Both failures already occurred before enabling these changes:
  `test_streaming_moe_oracle.py::test_streamed_experts_equal_materialized_hf`
  for False/True. The installed Transformers constructor rejects `device='cpu'`
  in `tools/goldens/moe_streaming_hf.py:84`. That helper is byte-identical to the
  frozen baseline; this unrelated reference-API mismatch is unresolved.
- Real-model prefix restore crosses a **1024-token cached prefix** with a
  1299-token request; output and accepted/committed/verification traces match
  for final BF16/Turbo4 8K and Turbo4 64K. Smaller prefix probes additionally
  cover the other 8K–32K candidate runs except the historical 8K Turbo4
  projection-only sample; final combined Turbo4 is covered by the stronger probe.
- Actual default `lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4` smoke
  passed with local packs: BF16 cache, **8192 context limit**, two identical
  1685-token prompts, identical generated messages. The temporary server stopped.
- **53 non-M6 backend source/config files** match the frozen baseline.
  Shared shader changes are guarded; M3/M4/M5 physical hardware was not tested.
  Python compilation and `git diff --check` passed.

Final-source 8K requests still improve Prefill and total request time. However,
Decode measured **14.182 → 13.512 tok/s for BF16 (−4.7%)** and
**14.994 → 14.563 tok/s for Turbo4 (−2.9%)**. The effect appears in GPU timings,
not just CPU wall time. Recorded decode projection launches are unchanged.
Its cause is not isolated; no decode gain is claimed. These two samples per
variant do not establish behavior for decode-heavy requests with thousands of
output tokens. Total 128-token request latency improves 62.666 → 52.343 s
(BF16) and 70.937 → 59.700 s (Turbo4).


Final-policy 8K repeats (including the original-MMA tail fallback), followed by fresh controls on the same final source:

| KV | N per variant | Control Prefill s | Final Prefill s | Control Decode tok/s | Final Decode tok/s | Restored prefix tokens |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 2 | 53.682 ± 0.062 | 42.910 ± 0.234 | 14.182 ± 0.079 | 13.512 ± 0.021 | 1024 |
| turbo4 | 2 | 62.436 ± 0.006 | 50.952 ± 0.008 | 14.994 ± 0.295 | 14.563 ± 0.010 | 1024 |

All reported exact comparisons are candidate versus baseline **within the same
KV format**. They do not imply quantized KV is identical to BF16. Existing
v1-oracle numerical gates were retained without modification.

## Remaining bottlenecks

The new PV stage still traverses historical V and reconstructs every partition;
score work remains proportional to context. The large fallback partial workspace
still consumes memory despite reduced traffic. Q8_0 needs a different tiling or
scheduling strategy before the new pipeline is useful. FP8 projections, GDN
recurrence and non-attention operations now limit further whole-model gains.
At the final real-prefix point, diagnostic category totals include 388.662 ms
NVFP4 matrix projection, 174.639 ms FP8 matrix projection, 31.976 ms GDN prepare
and 23.594 ms GDN mixer across the model. Decode regressions in the final 8K
recheck and 32K Turbo4 comparison remain unexplained and need a controlled
follow-up before making claims about decode-heavy workloads.
A smaller shared fallback workspace or a safely fused final consumer requires
fresh end-to-end and numerical qualification; it is not enabled here.

## Reproduction

Per-chunk partition counts, logical traffic and measured complete-chunk GPU times
are included in `chunk_statistics` in the results JSON.

Raw measurements are under `build/m6-projection-merge-20261014/`; the checked-in
[results JSON](../../tools/bench/results/m6-projection-partials-20261014.json)
contains samples, source hashes, exactness checks, memory statistics and microbenchmarks.
Use explicit baseline/candidate profiles to avoid comparing a newly changed
default against itself. All GPU commands must run sequentially.

```sh
.venv/bin/python tools/bench/m6_prefill_model.py --paths build/kv-rotated-20261013/paths.json --attention-path m6 --profile build/m6-projection-merge-20261014/profile-combined.json --context 32768 --prefill 128 --kv-cache-dtype turbo4 --repeats 2 --skip-prefix-checkpoint --validate-prefix --out build/recheck.json
.venv/bin/python tools/bench/m6_prefill_profile.py --paths build/kv-rotated-20261013/paths.json --profile build/m6-projection-merge-20261014/profile-combined.json --context 32768 --chunk 128 --kv-cache-dtype turbo4 --positions 8064 16256 32384 --repeats 3 --counter-repeats 3 --out build/recheck-profile.json
MTL_SHADER_VALIDATION=1 .venv/bin/python -m pytest tests/kernels/test_m6_partials.py tests/kernels/test_m6_mma.py
.venv/bin/python -m pytest tests/contract
```
