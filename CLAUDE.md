# CLAUDE.md

Context for anyone (human or agent) picking this repo up on another machine. Project name: **lithos-metal**. The Python import namespace remains
`monolith` for compatibility; use `lithos-metal` for public CLI examples.

## What this is

An LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+, Metal 4). First target:
`nvidia/Qwen3.8-27B-NVFP4` (Qwen3.5-hybrid: 48 Gated-DeltaNet + 16 full-attention layers; NVFP4 MLP + `lm_head`, FP8
attention/GDN projections; ~17.6 GB of weights read per decoded token) with a public DSpark drafter for speculative
decoding (`docs/research/dspark.md`). **v1 is judged on batch-1 decode
latency only** — prefill/TTFT, multi-request serving and energy are explicit non-goals for v1. The engine must stay
general: new models, quantization formats and ops come in through plugins, not engine edits. Stack: C++/Objective-C++
runtime, Python front-end and compiler, generated MSL kernels.

Status (2026-09-24): design, plan, surveys and hardware characterization (M3 Pro and M5 Pro) exist, and the engine
is being built as stacked PRs (roadmap issue #1): skeleton + registries, format plugins, `pack_weights`, the GEMV
harness and M1 study, runtime core v1 (ICB + host pump + token ring + StepState), and the layer library with the
first model package (`monolith/models/qwen3_5`), the M3 kernels, and compiler v0: the 0.8B decodes end to end on the
GPU from one replayed encode and reproduces its HF goldens, prompts of any length fed in chunks of `t_max`
(`python -m monolith.generate`); since then: the fuse pass, chunked prefill through the dynamic-T program,
GPU sampling, per-op tracing and autotuning, model 2 (`monolith/models/qwen3`, Qwen3-8B NVFP4, zero engine edits),
the DSpark drafter as a `Drafter` module verified against DeepSpec's reference, the round's kernels + IR lowering
(#24), and the round inside the target's step program (#38: `python -m monolith.generate --drafter …`; greedy
speculative decode is token-identical to plain decode on the 8B and on the hybrid 0.8B, the host idle). Speed on
this chip: parity with plain decode at best on the shader GEMV path (decode-kernels.md §5) — the M6 gate waits for
the T ≥ 2 GEMM path (M9). Sampling with a drafter is exact speculative sampling (#39); the prompt-set measurement (#40, dspark.md §3) puts
the cost-aware rule at 36 ms per token vs 27 plain — a no-go on the shader path, a projected go on the T ≥ 2 GEMM
path (M9). The barrier pass and the sibling overlap (#29, #35: the gate GEMV beside the mixer core) take the 0.8B
from 6.85 to 6.58 ms per token; the ICB barrier flag orders the flagged command behind all before it (measured;
the field is `barrier_before`). #34 closed with a measured no: the attention v2 (kept as a per-profile option)
is not the long-context win at one threadgroup per core — at two it is the T ≤ 4 win and the profile's default
(`attention: auto`, #113 below); SIMD-group-matrix scoring stays M9's long-context item; fast math buys 1–2 % and
breaks bit-identity, so safe stays. Format 2 (#47) is built: affine INT4 groups (`formats/int4_affine`, MLX / AWQ / GPTQ) as a plugin —
the MLX 4-bit 0.8B decodes token-identical to its oracle; the port needed a per-group bias hook, the quantized-embedding
gather, ragged lane stripes and a package-declared value adapter (mlx_lm folds `1 +` into the zero-centered norms;
porting-log.md); the porting guide (#48, `docs/porting.md`) closes M8. M9's accelerator GEMM is built (#50,
`kernels/common/gemm_tile.metal`): the cooperative right-input fill from the pack words streams NVFP4 at 177 GB/s and FP8
at 253 for 8 or 16 tokens — 0.9–1.1× a T = 1 shader pass, 34–49 % above `p14`'s staged tile; at 32 tokens it is
below `p14` (decode-kernels.md §6). #51 wires it into the step program: with the profile's `accelerator: on`
every T > 1 GEMV runs on the tile as the predicated variant above T = 1, and the DSpark round on the 8B goes from
35.8 to 19.9 ms per token on the prompt set (1.80× the shader path, 1.36× plain: math 1.89×, code 1.60×, text
1.24×, chat 0.99×; the step 94 % bus-bound), tokens equal to the golden (dspark.md §3) — the M6 gate is met on
math and code on this chip. Speculative decoding targets a **DSpark** drafter, not the MTP head.
Go/no-go #2 (#36) is read on this machine: plain decode of the 8B NVFP4 on the weights mlx-lm streams is 0.66× mlx-lm
(41.5 vs 62.9 tok/s; decode-kernels.md §8) — a no-go; the gap is the NVFP4 GEMV's ALU-bound decode (221 GB/s) plus
the pack's 9 % unit padding. The nvfp4 plugin reads MLX's conversions (same codes, `scales` as U8, no tensor scale).
The contingency tasks (#100–#103) run: MLX's half-exponent nibble decode is the plugin's default (#100,
`NVFP4_DECODE = 3`: 2–25 % per shape, the tile's fill +35 %, plain decode of the 8B 0.74× mlx-lm on equal bytes;
gemv-kernel-study.md §3e) — the remaining gap on the wide shapes is the unit padding (#101), on the 4096-row
shapes the occupancy of 256 blocks (the tile's K-split). A checkpoint's BF16 matrices can be re-quantized at pack
time (`pack_weights.py --quantize <fmt>`; the session binds the tree to the pack's formats): the DSpark drafter as
NVFP4 takes the cost-aware round on the MLX 8B pack to 12.6 ms per token, and the tile's K-split (`KSPLIT`: one
row tile per threadgroup of 2 or 4 SIMD-groups, the partials reduced through threadgroup memory — the 4096-row
projections 1.5–1.7× faster at T = 8, autotuned per op) to 10.8, and a wider `x_permute` plus the pruning of the
T = 1 variants a cost-rule program never takes (615 → 469 dispatches) to 10.4, the padding-free pack (#101:
`scale_placement: block` — the block's scales in their own region, the 8B pack 4.66 → 4.30 GB per token, 1.008×
the checkpoint), `StepState.stop_at` (the program stops itself at the request, no over-run) and sub-word lane
units (2 or 4 lanes share a payload word: the DSpark Markov head in NVFP4, 78 → 24 MB), the permutes of un-normed
tile inputs written by their producers (`PERM_OUT`) and a one-step pump to **9.56** against mlx-lm plain's 15.9 —
and against mlx-lm's own speculative decoding with a Qwen3-0.6B 4-bit draft, **9.24 at N = 3**: ours / theirs
1.036–1.055 (the range is the token stream's: near-tie tokens flip with the tile variants' rounding and the block
drafter's acceptance with them; ahead on math, even on code, behind on chat and text), the #103 gate not met then
— it is **met on 2026-09-27 with #113's v3 attention in the verify pass and the small-K GEMV: 9.14 vs 9.25, 0.988**
(math 0.89, code 0.95, text 1.03, chat 1.06; decode-kernels.md §6, §8, §9). The step is at the bus on its GEMVs; the structural lever left is acceptance — an
LM-drafter plugin (a 0.6B Qwen3 step inside the round) projects 6–8 % under mlx-lm's best. Plain decode is at
0.776× mlx-lm. The on-screen frame-pacing check (#7, `p15`) found the display path unaffected by
8–133 ms compute buffers: `max_cb_ms` is a latency knob, not a pacing one.
Model 3 (#46) is built as packages: the MoE ops (`ops/moe.py`: `moe_route`, `moe_gemv` = the GEMV template's pairs
mode addressing expert blocks through the router's ids, `moe_combine`), the `SparseMoE` layer and the `qwen3_moe`
package, proven on a synthetic checkpoint; the real Qwen3-MoE checkpoints did not fit the M5 Pro.
**2026-10-04 M5 Max update:** the real NVFP4 30B-A3B fits the 48 GB / 40-core machine and matches
the repeated 48-token HF continuation. Its long synthetic-KV layer gate remains open on routing
precision; bit-identical expert row/crew tuning is separate from that gate. See
`docs/research/m5max-qwen-llama-audit.md` for the Qwen/Llama audit and bounded searches.
An intermittent model-tier failure (wrong tokens / a hang / an empty generation, never reproducible alone) was three
out-of-bounds stores found with shader validation (#92): the GDN commit pass wrote its read-out through a 16-byte
placeholder, the tile's permute wrote a slab's K into a scratch sized by a narrower input, a drafter appended past
its context cache — fixed, and a `Program` now carries a `context_capacity` the serial ops enforce.
The M5 contingency (#100–#103) took the round on the 8B from 18.4 to 9.56 ms per token wall (V3 NVFP4 decode, the
padding-free pack, the K-split tile, the fused permutes, `stop_at`, the 1 × 2 pump) against mlx-lm's own speculative
decoding at 9.24 — 1.036× then, 0.988× (9.14 vs 9.25) with #113's kernels on 2026-09-27, ahead on math and code,
behind on chat/text (decode-kernels.md §8–§9). The second drafter plugin
(`monolith/spec/lm`, #103): any registered model package built with a `prefix` runs as the classical draft model
inside the round (the same `Qwen3-0.6B-4bit` mlx-lm drafts with; token-identical to plain decode, mlx-lm's
acceptance), with graph activation scopes and per-pass mixer modes; it exposed the small-model step: our 0.6B decodes
at 4.1 ms per token where MLX took 2.1 (row-split GEMV items and the attention's load-ahead brought it from 4.6),
so the LM round cost more than mlx-lm's until the per-layer gap closed (decode-kernels.md §10) — with #113's kernels
the 0.6B step is 2.06 ms and **the LM round is the best path: 8.97 ms per token at N = 5, 0.971 of mlx-lm's 9.24**
(math 0.83, code 0.92, chat 1.04, text 1.06). That comparison
is #113: `tools/bench/layer_vs_mlx.py` measures a decoder layer's cost on both engines as the slope of the step over
the layer count (decode-kernels.md §11) — at T = 1 the 0.6B's layer was 140 µs against MLX's 59 and the 8B's 549
against 400; the attention core at T = 1 over a short context was the largest loss (16–24 blocks of 64 keys over 240
SIMD-groups: the chunk is chosen at run time now, 65 → 24 µs), then the GEMVs on K = 1024 slabs (denser crews as
tuner candidates, the fused norm's fold unrolled, the tuner timing with the program's partial count, shader or tile
per op by measured time), and #34's verdict on the v2 attention was geometry-bound: with two threadgroups per core v2
is 2–3× faster than v1 at T = 1 and at 1024 keys for every T, so the profile's `attention` is `auto` (v2 up to 16
query rows per step). A static-T program emitted the T = 1 shader variant beside the tile without a predicate
(the predication lives behind `STEP_STATE`), streaming every slab twice — the tile alone now; the round's dynamic
program was never affected. A session shares its programs' buffers by name, and a static T = 8 program's params
records collided with the dynamic program's (both compile at T = 8): three o_proj tiles ran another GEMV's record —
out-of-bounds loads and stores, the bench's 16 s steps and stall; params names carry the program kind and `Engine`
never shares a params record now. The INT4 pack keeps its (scale, bias) pairs in the checkpoint's own 16-bit dtype, BF16 or F16 (the
checkpoint's bytes and its exact dequantization; `scale_unit_bytes` / `scale_dtype` in the manifest, older INT4 packs
are refused). The attention at few query rows is a third
kernel (`gqa_decode_v3`, §11.1): core and merge in one dispatch, a 1024-thread threadgroup per query row with the
keys strided over its SIMD-groups and the fold in threadgroup memory — MLX's decode-attention structure — 0.39–0.80×
v2's core + merge at T = 1 from 128 to 8192 keys and 0.42–0.89× at 4–32 query rows (ahead of v1 there too), so the
profile's `auto` is v3 at every row count; v2 and v1 stay as explicit choices.
The small-K GEMVs then got the activation words hoisted ahead of the items (`X_HOIST`: converted and normed once per
SIMD-group where a lane's columns fit 32 floats), one-row items (RG 1, RSPLIT 16) as tuner candidates and a norm fold
that requests 16 partials per round — level with MLX's `quantized_matmul` per shape (11.6 / 6.2 / 17.0 / 8.5 µs on the
0.6B's four against 10.9 / 5.7 / 17.1 / 8.3); the in-program excess over the isolated kernel is the chain's own
(~0.65 µs per pipeline switch, the barriers, ~2 µs per dispatch), not the pack's file mapping or the StepState read.
Per layer at T = 1 over 128 tokens of context the 0.6B is **level with MLX (60–61 vs 59–60 µs across runs; the step
2.06–2.09 vs 2.06–2.07 ms)** and the 8B 1.09× (441 vs 403; the step 17.1 ms, 0.91× mlx-lm); over 1024 tokens 1.29×
and 1.10×; at T = 4 the 0.6B 1.45× (the tile on its small slabs) and the 8B 1.12×; at T = 8 the 0.6B 1.14× (1.03× per step)
and the 8B **0.67×** — the layer gate (every layer strictly faster than MLX's) is open on the 8B's dependency chain
(~2–4 µs per dispatch of switch, barrier and cold start, and an attention nothing overlaps, where MLX's concurrent
stream hides its small kernels: our GEMV kernels are at its rate or better per shape) and on the small model's tile at
T = 4 (~5 µs of fixed cost per dispatch on 1–3.5 MB slabs); a slab prefetch beside the attention (a ~16 MB last-level
cache exists) and rows-per-block attention were measured and rejected; over 1024 tokens MLX's per-layer slope is not
linear in the layer count, so the step ratio is the comparison there (§11.1).

## Read these, in this order

1. `docs/design/design.md` — the design. §0 is the decision table (D1–D14); §7 answers "warp specialization?" (no) and
   "static megakernel?" (static yes, one kernel no).
2. `docs/research/apple-gpu-probes.md` — what was measured on real hardware and what each number implies. §1 is the
   cross-chip table (M3 Pro and M5 Pro filled); §3 is what the M5 Pro confirmed and changed; §4 is the checklist for
   a chip not yet measured (hypotheses H1–H10 and the outcomes that would change the design; written for an M4, whose
   measurement was dropped from the roadmap on 2026-09-25).
3. `plans/implementation-plan.md` — milestones M0–M9 with exit gates and go/no-go points.
4. `docs/research/apple-inference-systems.md` — how MLX, llama.cpp and others work; what to reuse; headroom estimates.
5. `docs/research/dspark.md` — the speculative-decoding method we target, the public drafters for our models, their
   cost on our hardware.
6. `docs/porting.md` — adding a model, a format, an op, a drafter or a chip: the contracts as they are in the tree,
   the CI checks, the golden workflow; `docs/research/porting-log.md` is the evidence it was derived from.

## The design in six lines

* The whole generation loop is one GPU-resident **static program**, compiler-generated from the model graph: one Metal
  dispatch per fused op (~330 per step for Qwen3.8 = 5 all-to-all stages × 64 layers), encoded once into an indirect
  command buffer and replayed; all per-step state lives on the GPU; the host only keeps the queue fed.
* **Not** one long kernel: a dispatch is never preempted, and a dispatch boundary is as cheap a barrier as anything
  in-kernel. Keep dispatches sub-millisecond and command buffers to tens of milliseconds.
* No role-specialized SIMD-groups. Every kernel runs with the crew geometry `gpu_cores × 384 threads`
  (one threadgroup per core, 12 SIMD-groups × 32 lockstep lanes).
* Weights are re-laid-out at load time (block-lane-major packs) so a SIMD-group sweeps one contiguous block; the lane
  order inside the block is a per-chip profile value (lane-interleaved 16-byte words on Apple10, where lane-contiguous
  stripes cap at 70 % of the bus; a tie on Apple9).
* The lever past the memory-bandwidth bound is DSpark speculative decoding (block drafter + Markov head + confidence
  head), run entirely on the GPU with the verify length chosen per step from the confidences and the chip's measured
  cost-per-T table.
* Overlap only an ALU-bound op with a bus-bound sibling (un-barriered dispatches at full geometry, the ALU-bound one
  encoded first on Apple10); never pre-stage weights, never hand-partition cores.

## Working rules

* **Standalone repo (design D15).** Copying files or fragments from MPK/mirage, MLX, llama.cpp, tinygrad, DeepSpec,
  DFlash or gpt-oss is fine and encouraged: keep the license header, add a provenance line (repo, path, commit) and a
  `third_party/NOTICE` entry. Never `import mirage`, never add a submodule or build dependency on MPK, never name
  anything MPK/Mirage. The tree must build and test alone with the Command Line Tools.
* **Model-agnostic by construction (design D16, §5.14).** Model names appear only under `monolith/models/<name>/`.
  Models, layers, formats, ops, drafters and chip profiles are reached through registries; the compiler's coverage
  guard fails a build for an op without a kernel; a model PR that touches `compiler/`, `runtime/` or `kernels/` is
  wrong by definition (CI enforces it). New ops and drafters land as their own packages with oracle tests.

* **Measure before claiming.** Evidence tags in the docs: [M] measured by us, [S] Apple spec, [R] third-party report,
  [H] hypothesis. Microsecond-level numbers move by tens of percent between runs — report ranges, draw conclusions only
  where ranges do not overlap, use paired alternating A/B runs and min-of-N.
* **Every GPU loop must be bounded.** A running dispatch cannot be cancelled and cannot be relied on to be preempted
  (never on Apple9, only sometimes on Apple10); an unbounded spin freezes the display and can trip the watchdog. Keep any single dispatch under ~1.5 s in probes, far less in the engine.
* **Every kernel store must be inside its binding, and shader validation is the test for it.** Buffers are separate
  Metal allocations, so a store past one lands in a neighbour — StepState, a params record, an activation — and
  shows up later as a wrong token, a hang or an empty generation that never reproduces alone. After a kernel or
  emitter change run the GPU tiers under `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`
  (porting.md §0); a `Program` carries its `context_capacity` and the serial ops stop at it (`error = 2`). A
  session's programs share buffers by name (weights, states, StepState, ring, activations); a params record is a
  program's own and is never shared (`Engine`), and its name carries the program kind.
* Correctness may depend only on documented Metal semantics (dispatch ordering, ICB barriers, the MSL memory model).
  Threadgroup→core mapping, in-flight limits and sharing behaviour are per-chip *profile values*, measured by the probes.
* Only bare-metal Macs give meaningful numbers; virtualized macOS (hosted CI runners) exposes a paravirtual GPU.
* Code adapted from other projects keeps its license header and is listed in `third_party/NOTICE` (MPK/Mirage is
  Apache-2.0; MLX, llama.cpp, tinygrad are MIT; gpt-oss Metal is Apache-2.0).
* Numerics contract: weight-only dequantization (W4A16 / W8A16), BF16 residual stream, FP32 accumulators and recurrent
  state. Reference = the HF model run on the dequantized weights. Gates: leaf ops ≤ 2 ULP, layers cos > 0.999, greedy
  tokens equal to the golden, repeated runs bit-identical.

## Probes

`./probes/run_all.sh` builds and runs the 17 probes (~6 min, Xcode Command Line Tools only; shaders compile at runtime,
including the MPP tensor ops of `p14`; `p15` opens a window and needs the screen unlocked) and saves
`probes/results/<chip>_<cores>c_macOS<ver>_<time>.txt`.
`./probes/remote_run.sh user@host` does the same over SSH. Geometry is derived from the GPU core count (`GPU_CORES=<n>`
overrides). Commit every results file. Measured so far: an M3 Pro (2026-09-19, 13 probes) and an M5 Pro (2026-09-22,
all 16, repeats of `p6`/`p6b`/`p12`); the M5 Pro's profile is the writer's (`tools/profile_writer.py`), the M3 Pro's
hand-derived. `p12`–`p14` never ran on the M3 Pro (dropped with the machine, 2026-09-25). `./probes/build/p13_decode_gemv
check` (same for `p14`) compiles every kernel variant without dispatching.

## Next steps

The numbered list below records the September M5 Pro roadmap. Its memory limits
do not apply to the current 48 GB M5 Max: 27B work has proceeded, and the real
30B-A3B MoE is now measured. The current Qwen/Llama correctness limitations and
retained 40-core tuning are in `docs/research/m5max-qwen-llama-audit.md`.

0. **Keep building.** The M5 contingency tasks (#100–#103) are done and closed (2026-09-27): the gate is met at 8.97
   (LM drafter) / 9.14 (DSpark) ms per token against mlx-lm's 9.24. Open on this machine: #113's remaining items —
   the 8B's 9 % at T = 1 (the dispatch chain: switches, barriers, cold starts, and the attention nothing overlaps),
   the small model at T = 4 (the tile's fixed cost on 1–3.5 MB slabs), the attention over long contexts (a per-kv-head
   v3 block with rep rows) — then the staged multi-SIMD-group tile for T ≥ 32 and the K-split for the down projection
   (decode-kernels.md §6), the attention core's SIMD-group-matrix scoring for the long-context rows; and merging the
   #109–#114 stack. The autotuner at install time is built (`tools/profile_writer.py`: it measures the `engine`
   block from the kernel harnesses and merges it into `the selected backend configuration under monolith/backends/metal/`). Intra-op stealing (#44) is
   built, measured and off by default (decode-kernels.md §7). Everything this M5 Pro cannot host was dropped from the
   roadmap and its issues closed as not planned: the M3 Pro and M4 tasks (2026-09-25) and, on 2026-09-27, the 27B
   items (#31's 27B rows, #36's and #40's), #46 (the smallest MoE checkpoint in a format we read,
   `nvidia/Qwen3-30B-A3B-NVFP4`, is ~18.5 GB resident against this machine's ~18–19 GB GPU working set), #42 (a GPU
   box), #43 (Max-class parts), #49's other M5 chips and #52 (macOS 27) — they reopen with the machine.
1. A new chip, if one arrives: run the suite, commit the results, fill its column in the hardware report §1, walk
   H1–H10 in §4, run `tools/profile_writer.py`, and update the design where a hypothesis fails (D4, D5, D6, D8, D14
   are the chip-sensitive decisions).
2. Plan M0 remainder on this machine: the on-screen frame-pacing check (#7). The 27B's baselines and goldens wait for
   a machine that hosts it.
3. Plan M1 (go/no-go): the NVFP4 decode is the problem (ALU-bound at 59 % of nominal on the M5 Pro; FP8 is at 90 %);
   then the comparison against MLX `qmv` on the same machine.
