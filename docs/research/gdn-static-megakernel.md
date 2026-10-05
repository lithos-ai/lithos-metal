# Static Gated DeltaNet fusion at N=7

**2026-10-01 follow-up:** the [40-core M5 Max / real 27B checkpoint study](m5max-27b-megakernel.md)
extends the prototype to full attention and NVFP4 MLP, evaluates two kernels per
decoder layer, and compares every layer with native MLX-LM. The M5 Pro experiments
below are historical measurements with their original scope.
The [additional M5 Max tuning study](m5max-27b-megakernel-tuning.md) expands the
configuration knobs and tests the complete-layer effect of narrower fusion.
The [2026-10-02 GDN optimization study](m5max-gdn-mixer-optimization.md)
adds lossless FP8 operand packing and a broader configuration search on M5 Max.

The tested single-kernel schedules do **not** improve latency on the 20-core M5 Pro.
At the time of this experiment the production compiler was unchanged. The later
[M5 Max follow-up](m5max-27b-megakernel-tuning.md#default-selection-after-the-study)
selects mixer-only fusion for its measured fixed-eight-row configuration.
This original experiment covers the **GDN core**:
causal convolution/SiLU, q/k L2 normalization, scalar gates, the eight-token
recurrence, convolution/recurrent state writes, and gated RMSNorm. It excludes
input/output matrix projections, the MLP, drafting, and accept/rollback. N=7
means **eight target rows**, including the anchor.

## Connection to MPK's static compiler

Read [mirage-project/mirage PR #786](https://github.com/mirage-project/mirage/pull/786)
at `c28cac618b98fda1e0f1d590923b5f69b4ef3603`. Its compiler chooses task grids,
derives tensor-slice dependencies, assigns ordered task lists to workers, and
emits a single kernel calling the task bodies. The example is a Kimi K3 MoE
layer, not GDN. The PR description says GPU validation after its cleanup is
still pending; it is not performance evidence for Apple silicon.

Two deliberately small adaptations test the scheduling idea using Monolith's
existing arithmetic, without adding a second production compiler:

* **Static heads:** worker `w` executes heads `w, w+workers, ...`; all stages for
  a head run locally, with threadgroup scratch for preparation and readout.
  Head ownership removes cross-worker dependencies. This uses the existing
  general GDN loop with fused normalization and a fixed worker count.
* **Static stages:** fixed SIMD-group lists execute preparation tasks, then
  state-column tasks, then normalization tasks in one kernel. Two device-wide
  rendezvous separate the stages. Metal `coherent(device)` pointers and device
  fences publish the intermediate values. Monotone per-worker epochs permit
  repeated launches without CPU counter resets. A bounded wait reports failure
  rather than silently accepting an incomplete schedule.

A whole-head dispatch with one threadgroup per head is another fusion control.
The stage schedule also has a three-dispatch control using the same recurrence
slice width and worker geometry. Its preparation/norm grids retain the ordinary
one-SIMD-group-per-task layout; it is a decomposition control, not an isolated
measurement of the barrier instruction cost.

These are fixed schedules for this leaf, not the PR's CP-SAT placement search or
its full MoE implementation. In particular, the stage version uses conservative
whole-stage dependencies, not arbitrary fine-grained dependency counters.
Metal exposes threadgroup IDs, not physical-core affinity; 20 workers means 20
threadgroups, not a guarantee of one pinned worker per core.

## Measurements

Apple M5 Pro, 20 GPU cores, 24 GB; Monolith base
`ade38fb5f81ebdf852a2b65a616703b03f4ec424`, macOS 26.5.1. Shapes match the
Qwen3.8-27B GDN core: 16 key heads, 48 value heads, K=V=128, convolution width 4.
Inputs/weights are seeded synthetic BF16 tensors; recurrent state is nonzero
FP32. These are shape-level measurements, not an inference run of the 27B model.

The current compiler uses 96 threadgroups, 32 SIMD groups each, two state
columns per SIMD group, and a separate gated-norm dispatch. Screening varied
whole-head slice widths 4/8 and static worker counts 10/20/40. Static stage
schedules used 20 workers, 4/8/12 SIMD groups, and slice widths 2/4. Outputs and
both complete state buffers matched the baseline bit for bit.

A warm-cache screen of one layer (64 replays per command buffer, 20 randomized
paired rounds) gave minimum times of 32.76 us for baseline, 36.56 us for the best
whole-head fusion, 39.03 us for 20-worker head fusion, and 48.46 us for the best
static stage schedule. The three-dispatch stage control took 39.78 us.

Confirmation streams **48 distinct state-buffer sets**, two passes per command
buffer, after at least 30 ms GPU warmup per variant; 40 rounds randomize variant
order. Each command buffer's GPU start/end duration is divided by 96 core
invocations. The table is therefore a stack-average core time, not timestamps of
individual layers. Input state is fixed and output written to the opposite slot
on every replay; variants perform identical useful work. State traffic alone is
about 302 MB per 48-layer pass, avoiding the one-layer cache reuse of the screen.

| Schedule | Minimum us/core | Median us/core | Minimum vs current |
|---|---:|---:|---:|
| Current, two dispatches | 42.27 | 49.44 | baseline |
| Whole-head fusion, 48 groups, slice 8 | 52.51 | 59.50 | +24.2% |
| Static head fusion, 20 workers, slice 8 | 56.58 | 63.38 | +33.8% |
| Separate stages, 20 workers × 12 SIMD groups, slice 4 | 53.96 | 61.89 | +27.7% |
| Static stages, same recurrence geometry | 63.36 | 69.96 | +49.9% |

An independent repeat gave minimum times of 43.01 / 52.24 / 55.80 / 53.85 /
63.14 us in the same row order, confirming the result.

Fewer dispatches did not compensate for the loss of independent recurrence
work. Whole-head ownership reduces the current 96 groups to 48 and uses wider
slices; limiting it to 20 workers serializes multiple heads per worker. The
static stage variant retains narrower slices but adds cross-worker publication,
waiting, and a larger combined kernel. These measurements support keeping the
current path; they do not rule out a different schedule that also pipelines the
large matrix projections.

## Correctness and reproduction

The new continuation checks cover two head-count configurations, variable active
lengths (8/3/1/0/6), alternating state slots, `done`, and 64 GPU replays without
resetting counters. The eight new checks plus 74 existing GDN kernel/oracle
checks passed. All eight new checks also passed with `MTL_SHADER_VALIDATION=1`.
No numerical tolerance was relaxed.

```bash
python tools/bench/gdn_static_bench.py --stages --reps 20 \
  --out /tmp/gdn-static-screen.json
python tools/bench/gdn_static_bench.py --confirm --layers 48 --repeat 2 --reps 40 \
  --out /tmp/gdn-static-confirm.json
python -m pytest tests/kernels/test_gdn_static.py tests/kernels/test_gdn_mixer.py
```

Raw timing samples are emitted only to the requested output path. The benchmark
and experimental shader are under `tools/bench/`; neither changes generation
behavior or enables an unproven schedule by default.

## Full GDN block, including matrix projections

Follow-up scope: one megakernel for the entire GDN/attention half of a decoder,
with a separate gated-MLP megakernel planned. The implemented prototype includes
input RMSNorm and permutation, Q/K/V, a/b and z projections, convolution and
recurrence, gated RMSNorm/output permutation, output projection and residual add.
It does **not** implement the full-attention or gated-MLP megakernel yet.

`gdn_block_bench.py` creates a deterministic synthetic fixture with hidden=5120,
16 key heads, 48 value heads, head dimensions 128, and eight target rows. QKV, z
and output weights use FP8 E4M3 with tensor scales; a/b weights are BF16. Each
measurement reads about 116 MB of matrix weights. Both convolution and recurrent
input states are nonzero. The checkpoint/model is not loaded and these results
are not an end-to-end 27B generation measurement.

`gdn_block_static.py` inlines the existing task bodies into a single fixed-worker
kernel, preserving the compiler's barrier-free sibling pair (recurrence and z
projection). The matched control retains separate dispatches with the same
geometry and operand loading as the combined kernel. Code generation remains
experimental and specialized to these GDN task types.

The installed MetalPerformancePrimitives implementation rejects coherent-device
tensor operands. A first workaround copied activation tiles through threadgroup
memory; it regressed substantially (about 3.15 ms split / 4.19 ms fused in an
initial zero-state screen). The selected workaround instead loads activations
through coherent pointers into cooperative register tensors. This API requires
M and N in {16,32}, K in {16,32}, with at least one dimension equal to 32. The
selected matrix tile is M=16, N=16, K=32, with eight active rows. Immutable weights
remain ordinary device reads; cross-worker intermediate buffers are coherent
and fenced. Do not remove coherence just to make the old device-tensor path
compile.

A 16-configuration screen tested 8/12/16/20 workers with 4/8/16/32 SIMD groups.
20 workers × 16 SIMD groups was the strongest candidate. N=32 matrix tiles were
also tried without a clear improvement. Common threadgroup geometry is necessary
because the task bodies contain threadgroup barriers; skipping such barriers
with only a subset of threads would be invalid.

Final confirmation: ICB replay, one full block per command buffer, two in flight;
eight steps per sample, at least 30 ms GPU warmup per variant, 40 randomized
paired rounds. The control and fused output **and both state buffers are
bit-identical**. Against production, output cosine is 0.9999999392 and relative
L2 error is 0.0003486; changed matrix tiling changes floating-point accumulation.

| Variant | Dispatches | Run A min / median, us | Run B min / median, us |
|---|---:|---:|---:|
| Current production | 9 | 857.79 / 935.14 | 838.96 / 922.88 |
| Matched geometry/operand control | 9 | 933.21 / 1025.60 | 958.17 / 1005.45 |
| Full GDN megakernel | 1 | 873.91 / 931.26 | 857.44 / 918.41 |

Fusion helps the matched control, but the full megakernel remains 1.9–2.2% slower
than production by minimum latency; medians are about 0.4–0.5% faster. Treat this
as **near parity, not a demonstrated performance win**. No production defaults
changed. This finding supersedes applying the earlier core-only regression to
the full GDN block: projection fusion materially changes the result.

```bash
python tools/bench/gdn_block_bench.py --out /tmp/gdn-block-a.json
python tools/bench/gdn_block_bench.py --out /tmp/gdn-block-b.json
# Repeat the worker/geometry search on the destination machine:
python tools/bench/gdn_block_bench.py --workers 20 --sgs 16 --out /tmp/gdn-block-candidate.json
MTL_SHADER_VALIDATION=1 python -m pytest tests/kernels/test_gdn_block_static.py
```

The fixture is generated and reused under `/tmp/gdn-full-block-fixture`; no
checkpoint download is required. Build the native extension per CONTRIBUTING.md
and install the oracle dependencies (torch/safetensors) for fixture creation.
The two new full-block checks cover alternating state slots, changing inputs,
continuation, fixture reuse, and 64 replays without counter reset.

Before running on another chip, register a profile for its **actual GPU core count**.
The follow-up adds `apple-m5-max-40c`; re-measure the crew geometry and
bandwidth rather than silently using the M5 Pro profile. That M5 Pro prototype
used fixed-T=8 emission, supported FP8/BF16 cooperative matrix inputs, and rejected
packed 4-bit inputs in that mode. Dynamic speculative row counts, NVFP4 MLP,
full attention/KV state, full-model generations, and direct MLX comparisons are
not validated by the historical experiment above. The linked M5 Max follow-up
validates the fixed-row attention/MLP and MLX comparisons; generation and dynamic
speculative decoding remain outside its scope.
