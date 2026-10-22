# M6 prefill bottlenecks after matrix attention

Follow-up: [8K–64K projection and probability-partial qualification](m6-projection-partials-20261014.md)
records the subsequent implementation, repeated full-model results and production
selection. The measurements and benchmark-only decision below describe this
earlier investigation.

Measured on 2026-10-14, Apple M6, 12 GPU cores, 32 GiB unified memory,
`nvidia/Qwen3.8-27B-NVFP4`, Turbo4 KV, DSpark with seven proposals.
The production matrix-attention path is the baseline for this investigation.

**Result:** NVFP4 projection geometry is the strongest measured non-attention
opportunity. A benchmark-only change reduces repeated 8K prefill from 62.61 s
to 52.82 s, with identical logits, tokens and acceptance traces. Attention
still dominates the final chunks of a 32K prompt. GDN pass-length tuning has
small benefits; CPU submission is not a significant measured bottleneck.

This investigation adds profiling tools and opt-in experimental geometry.
It does **not** enable the experimental geometry in `serve`. BF16 defaults,
serving limits, other backends, numerical tolerances and production shaders
are unchanged by this investigation. Longer-context full-request qualification
of the projection candidate remains necessary before changing dispatch policy.

## Real-prefix stage measurements

Ingest 32,640 actual tokens, stopping at selected intermediate chunks to replay
the existing 128-row ICB. Restore recurrent/control/ring state between probes;
each replay overwrites the same new KV rows without copying historical KV.
Each point has three whole-ICB and three per-dispatch counter measurements,
after a warmup, with alternating measurement order.

| Approximate context | Prefix position | Whole ICB ms | NVFP4 matrix ms | FP8 matrix ms | GDN prepare ms | GDN recurrence ms | Attention core ms | Merge ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 8,064 | 1,200.71 | 557.30 | 174.61 | 30.65 | 26.17 | 398.78 | 45.67 |
| 16K | 16,256 | 1,626.92 | 550.02 | 174.70 | 31.83 | 26.68 | 771.11 | 92.56 |
| Near 32K | 32,384 | 2,481.32 | 551.60 | 174.60 | 31.40 | 26.62 | 1,530.29 | 188.41 |

Whole-ICB time and counter stage times are different measurements, **not
additive**. Counters insert encoder boundaries, perturb scheduling, and their
intervals can overlap. Shares based on those durations are diagnostic rather
than an exact allocation of production latency. The state snapshot is 293.66
MiB of CPU memory; its copies also perturb residency and wall-clock latency.
Do not interpret the probes' wall-minus-GPU time as ordinary serving overhead.

At 16K, NVFP4/FP8 matrices account for approximately 32.3%/10.2% of the counter
sum, GDN prepare plus recurrence 3.4%, and attention plus merge 50.7%. Near
32K, attention plus merge reaches 66.9%, while NVFP4 matrices remain 21.5%.
The smaller whole-model speedup therefore does not establish that attention
has ceased to dominate every long-context workload.

The real-prefix run's generation GPU time, excluding diagnostic replays, is
416.48 s. This is a different repeated-text prompt from the qualification
benchmark and is not an A/B end-to-end speed comparison. Synthetic zero-KV
probes produced similar stage rankings; conclusions above use the real prefix.

## NVFP4 projections

The model already dispatches `gemm_tile`, using MPP `matmul2d` with BF16
operands and FP32 accumulation. Its baseline 128-row MLP projection uses
`TM=32, TN=32, TK=128`, 12 threadgroups per token plane, with 12 SIMD groups
per threadgroup. The four token planes reload/decode their weight tiles.
This is not a scalar-GEMV fallback or absence of matrix acceleration.

Sweep real layer-zero weights and signed random activation operands. Keep
TK, operand rounding, K accumulation order, epilogues and safe math unchanged.
Each candidate has six alternating control/candidate samples, each containing
eight dispatches. Inputs and persistent state are restored for each sample.

| Projection | Candidate geometry (TM/TN/TK; groups × SIMD) | Paired control ms | Candidate ms | Speedup | Max deviation |
| --- | --- | ---: | ---: | ---: | ---: |
| Gate/up, 34,816 × 5,120 | 32/32/128; 24 × 4 | 5.332 | 3.374 | 1.58× | 0, identical bytes |
| Down, 5,120 × 17,408 | 32/16/128; 48 × 8 | 2.244 | 1.615 | 1.39× | 0, identical bytes |

Larger token tiles do not automatically help: gate/up with TM=64/TN=32 and
24 × 8 took 11.67 ms against a 5.34 ms paired control. More decoded-weight
reuse must be balanced against per-thread storage and occupancy. Register
pressure is a plausible explanation, not a measured hardware-counter result.
No tensor-unit utilization percentage is claimed.

### Full-model confirmation, 8K

Fresh control and candidate processes; two uncached requests each. The same
8,064-token prompt generates 128 tokens. Only the two target NVFP4 MLP shapes
above change, only in the 128-row prefill program. Attention, verification and
decode use the existing paths. Compilation/warmup are excluded.

| Metric | Control | NVFP4 geometry candidate |
| --- | ---: | ---: |
| Prefill GPU seconds, runs 1 / 2 | 62.496 / 62.719 | 52.796 / 52.849 |
| Prefill GPU seconds, median | 62.608 | 52.823 |
| Prefill tokens/s | 128.80 | 152.66 |
| End-to-end seconds, median | 70.993 | 61.236 |
| Decode tokens/s, median | 15.197 | 15.142 |
| Nonterminal accepted proposals/round | 2.67647 | 2.67647 |
| Managed GPU buffer allocation | 22.28870 GiB | 22.28870 GiB |
| Peak process RSS | 2.447 GiB | 2.458 GiB |
| System swap-out increase | 0 pages | 0 pages |

Prefill is 1.185× faster, a 15.6% time reduction. End-to-end latency falls
13.7%. Decode is effectively unchanged; this is a prefill-only experiment.
Both repeats have identical output tokens, every accepted/committed count,
and prompt-end logits across both paths. Logit maximum absolute deviation is
zero. Managed allocation counts unique runtime buffers; it is not a complete
Metal driver allocation counter. Process RSS is not total unified GPU memory.

## GDN recurrence

The emitted forward path already separates preparation from recurrence
(`PREPARED=1`, `LOCAL_PREPARE=0`, four columns per SIMD slice). It avoids the
unprepared kernel's repeated convolution replay. Recurrence still advances
sequentially through tokens and, at TP=8, writes and reloads its FP32 state
every eight rows. The source has a clear opportunity to retain it longer.

Prepared GDN microbenchmarks with real weights, nonzero recurrent/conv state
and randomized projections keep arithmetic order and normalization unchanged:

| Pass length | Paired TP=8 control ms | Candidate prepare + recurrence ms | Output / recurrent state |
| ---: | ---: | ---: | --- |
| 16 | 0.5471 | 0.5335 | Identical bytes |
| 32 | 0.5458 | 0.5308 | Identical bytes |
| 64 | 0.5516 | 0.5319 | Identical bytes |
| 128 | 0.5501 | 0.5271 | Identical bytes |

The best pass-length result is only about 4.2% faster for this isolated pair.
Logical state traffic falls substantially, but that is not a measurement of
off-chip traffic: much of the original traffic may be served from cache.
Given its whole-model share, this alone is unlikely to produce a useful large
prefill improvement. Preparation sharing across repeated Q/K heads is a
separate candidate; a parallel scan/MMA recurrence would require a much more
substantial numerical qualification and is not implemented here.

## Attention partials, merge and scheduling

Merge grows from 45.67 to 188.41 ms per profiled chunk between 8K and near 32K.
At the last point, 508 partitions × 128 query tokens × 24 heads × 256 FP32
values imply about 1.49 GiB of partial-output addresses per attention layer.
Across 16 attention layers, a write plus read is about 47.6 GiB of **logical**
traffic per chunk, excluding correction metadata. This is not measured DRAM
traffic, and causal partitions affect actual core writes. The workspace is
reused between layers, so this is traffic rather than peak allocation.

Merge deterministically folds chunks in order, applies softmax correction,
then Turbo4 inverse rotation, BF16 rounding and gating. Reducing partial
traffic without changing this contract is a stronger long-context target than
merely eliminating the 16 merge launches. Fusing or changing partition size
must not silently alter reduction/correction/rounding order.

The prefill ICB contains 1,660 dispatches and 1,055 barrier markers, including
inactive shape/draft/attention alternatives. Hundreds of counter intervals are
under 10 µs, but their combined duration is approximately 1–2 ms per chunk.
This does not measure barrier stalls or establish that launch removal is free.

An independent, uninstrumented 32K request from the preceding qualification
provides a production host-pump control: 255 chunks, 411.940 s GPU and
412.024 s pump wall time; total CPU busy 7.855 ms, encode 2.456 ms and commit
0.967 ms. CPU submission is negligible here. `wait_ms` includes waiting for
the GPU to compute; it is **not GPU idle time**. Device occupancy, pipeline
bubbles and memory stalls need additional GPU counters before attributing
their causes.

## Quantized-prefill specialization gap

`Session.__init__` currently sets `prefill_optimizations=False` for quantized
KV. This also bypasses generic `specialize_prompt`, not just BF16-specific
attention recipes. As a result, intermediate quantized prefill computes the
target LM head even though no output token is needed yet. Its observed matrix
duration is about 49 ms per chunk. Splitting safe prompt-only specialization
from BF16-specific transformations is a concrete next candidate. The opt-in
`prompt-only` probe is available, but it is not qualified or enabled here.

## Validation and next integration boundary

All 20 isolated candidates (8 gate/up, 8 down, 4 GDN) were repeated with
`MTL_SHADER_VALIDATION=1`. No shader errors were reported, and all checked
outputs/state matched their controls exactly. Validation timings were excluded
from performance conclusions. The full-model NVFP4 experiment passed the
four-run output/logit/acceptance comparison above. Python compilation checks
passed for the new tools.

Priority: qualify and integrate the measured NVFP4 geometry first; then study
FP8 projection geometry and partial/merge traffic. Separate quantized prompt
specialization is a small, focused follow-up. GDN pass length and generic
launch fusion have lower measured priority. The projection candidate has not
yet been qualified in full 16K/32K/64K requests, BF16/Q8 full requests, partial
active-row boundary cases, or cache checkpoint/restore regressions; it remains
explicitly opt-in in the benchmark tools. No production-default speedup beyond
the already integrated attention path is claimed.

## Reproduction and evidence

`paths.json` contains local `model`, `pack`, `draft` and `draft-pack` paths.
The commands below record the historical experiment. The checkpoint retains the
profile and full-model tools; the one-off candidate/probe tools were retired
after production integration. Use the profile controls described in
[the benchmark guide](../../tools/bench/README.md) for current comparisons.
Run sequentially on an otherwise idle GPU:

```sh
.venv/bin/python tools/bench/m6_prefill_profile.py --paths paths.json \
  --context 32768 --chunk 128 --kv-cache-dtype turbo4 \
  --positions 8064 16256 32384 --out profile.json
.venv/bin/python tools/bench/m6_prefill_candidates.py --paths paths.json \
  --kind gate_up --out gate-up.json
.venv/bin/python tools/bench/m6_prefill_model.py --attention-path m6 \
  --paths paths.json --context 8192 --kv-cache-dtype turbo4 --prefill 128 \
  --repeats 2 --skip-prefix-checkpoint --out control.json
.venv/bin/python tools/bench/m6_prefill_probe_model.py --experiment nvfp4-crew \
  --attention-path m6 --paths paths.json --context 8192 \
  --kv-cache-dtype turbo4 --prefill 128 --repeats 2 \
  --skip-prefix-checkpoint --out candidate.json
```

[Consolidated measurements](../../tools/bench/results/m6-prefill-bottlenecks-20261014.json)
retain checkpoint metrics, per-request samples, all candidate sweeps, equality
checks, source hashes and raw artifact hashes. Local raw timestamps, complete
operation metadata, validation logs and logits are under
`build/m6-bottlenecks-20261014/`.
