# Porting guide — adding a model, a format, an op, a drafter or a chip

Derived from the three ports recorded in [`docs/research/porting-log.md`](research/porting-log.md) (a dense model,
a drafter, a quantization format) and their measured time-to-port. Everything here names files that exist; when a
contract in [`design.md §5.14`](design/design.md) and the tree disagree, the tree wins and this guide says so.

The one rule (design D16): **a model is an addition under a registry, never an edit of the engine.** Model names
appear only under `monolith/models/<name>/` — `tools/ci/hygiene.py` fails any other file that names one, in an
identifier *or* a string literal, so a weight-name map cannot leak into the engine. A model PR carries the
`model-pr` label and may change only `monolith/models/`, `tests/` and `docs/` (`tools/ci/extension_check.py`, run
by CI on labelled PRs). Formats, ops and drafters are packages too, reached through registries; each of them is
its own PR with its own oracle test.

## 0. Setup and the test tiers

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,oracle]"          # numpy + pytest; torch, safetensors, transformers for oracles and goldens
pip install nanobind ninja              # the Metal runtime module (runtime/CMakeLists.txt, added by the top-level one)
cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(pwd)/.venv/bin/python && cmake --build build
python -c "import monolith.runtime as r; print(r.is_available())"     # True: the module was written into monolith/runtime/
```

| Tier | Needs | What it proves |
|---|---|---|
| `tests/contract` | nothing but numpy (runs on Linux CI, Python 3.10 and 3.13) | registries, IR lowering, coverage, pack round-trips, format math, adapters — on synthetic checkpoints |
| `tests/kernels`, `tests/runtime` | the Metal module, a bare-metal Mac | every kernel against a numpy or torch oracle; the runtime core |
| `tests/layers`, `tests/models`, `tests/spec` | torch + checkpoints under `~/models` (or `$MONOLITH_MODELS`) | layers vs the HF modules; whole models vs their goldens on the GPU; drafters vs their reference |
| `python tools/ci/hygiene.py` | nothing | standalone repo (no `mirage` import, no MPK/Mirage identifiers), no model names outside `models/` |

Tests that need a checkpoint or torch **skip** when they are missing, so the contract tier stays hermetic. A
virtualized macOS (hosted CI) exposes a paravirtual GPU: only bare-metal numbers mean anything.

After a kernel or emitter change, run the GPU tiers once more under Metal's shader validation — it instruments
every kernel and reports each out-of-bounds device load or store with the kernel, the source line and the buffer's
length (an intermittent wrong-token / hang / empty-generation failure of the speculative programs turned out to be
three such writes, invisible to every other test until the corrupted neighbour happened to matter):

```bash
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 python -m pytest tests/kernels tests/models tests/spec -x 2>&1 | grep -A3 Invalid
```

It works on the ICB replay path (the first violation per kernel is reported); `Engine.run(reencode=True)` reports
every dispatch. Any report is a bug: kernels index by the compiled shapes, so a value narrower than a slab's K, a
placeholder output, or a position past a cache's rows is memory corruption, not a numerical error.

Numerics contract (CLAUDE.md): weight-only dequantization, BF16 residual stream, FP32 accumulators and recurrent
state. The reference is the HF model run on the dequantized weights. Gates: leaf ops ≤ 2 ULP, layers cos > 0.999,
greedy tokens equal to the golden, repeated runs bit-identical.

## 1. Adding a model

Model 2 (dense Qwen3-8B NVFP4) took about an hour from the first line to a running model — 25 minutes for the
package, the rest waiting for the download and the CPU golden — and **zero engine edits**. That is the bar: if the
architecture is a composition of the layer library, the port is three small files and two tests.

### 1.1 Read the HF modeling file first

Decide, before writing anything:

* **The norm convention.** `RMSNorm(one_plus=True)` scales by `1 + w` (Qwen3.5's zero-centered weights);
  `one_plus=False` scales by `w` (Qwen3, Llama). The same flag exists on the per-head q/k norms
  (`GQAAttention(norm_one_plus=)`). Getting this wrong produces a model that runs and is garbage.
* **The attention variant.** `GQAAttention(gate=True)` is the Qwen3.5 hybrid's `[q | gate]` projection with the
  `σ(gate)` output gate; `gate=False` the standard one. `rotary_dim < head_dim` is partial RoPE (the head-dim
  permutation for RoPE is applied at pack time to the q/k rows, the q/k norm weights and the tables — the package
  does not care). Only `rope_type == "default"` is supported; sliding windows are not (the Qwen3 package raises on
  both — copy those guards).
* **The mixer types per layer** (`layer_types` in Qwen3.5: `GatedDeltaNet` or `GQAAttention`), the MLP
  (`GatedMLP`, gated SiLU; the packer interleaves `gate|up` rows in chunks of `pack_rows / 2` so one GEMV yields
  `silu(gate) · up`), tied or untied `lm_head`.
* **Ignored tensors**: vision towers, MTP heads, activation/KV scales of vLLM-style checkpoints. The package's
  `weights.py` names the prefixes it ignores; the weight map claims only what the tree consumes.

### 1.2 The package — `monolith/models/<name>/`

Mirror `monolith/models/qwen3/` (170 lines):

* `config.py` — a dataclass of the fields the tree needs, `from_dict(config.json)` (read `text_config` when the
  checkpoint is multimodal, `rope_parameters` for RoPE, `generation_config.json` for `eos_token_id`), with
  `ValueError` guards for what the library does not do.
* `model.py` — `@register_model("<HF architectures[0]>")` on a `monolith.nn.Model` subclass. The constructor
  composes library modules with two names per module: the checkpoint's `hf_name` / `hf_prefix` (what the weight map
  claims) and the tree's `prefix` (what the pack, the states and the tap buffers are called). Implement:
  `layers()` (the step program follows this order), `state_spec()` (concatenate the mixers' `state_entries()`),
  `feature_taps()` (layers a drafter may read; `range(n_layers)` when all), `tables()` (computed constants such as
  the RoPE tables, BF16), `forward(ids, state, pos)` (the torch oracle: embeddings → layers → norm → head; returns
  logits, every residual stream, the state), `lower(g)` (the same walk in IR: `LowerContext`, states and consts
  registered on the graph, `self.tap_values[layer] = h` after each layer, the sampler last). `from_checkpoint(path)`
  builds the config and calls the package's `bind_checkpoint_formats`.
* `weights.py` — the checkpoint conventions: the text prefix, ignored prefixes, `bind_checkpoint_formats`
  (`nn.pack_plan.bind_formats`: every claimed tensor's storage format is detected from the safetensors dtypes —
  `formats/checkpoint.py detect_format`: NVFP4, FP8-E4M3, affine INT4, BF16, F32 — and set on its module; mixed
  formats in one model are normal) and `load_oracle` (`load_oracle_weights`: the dequantized tensors as BF16 torch
  parameters).
* `__init__.py` — import the config and the model; add the package to the import list in
  `monolith/models/__init__.py` (inside `models/`, so still a model-PR change).

The engine finds the package by `config.json`'s `architectures[0]` (`monolith.models.resolve_model`).

### 1.3 Weights: what the tree declares, what the packer does

Every module's `weight_map()` returns `{local_name: WeightSpec(hf_name, shape, format, transform=, slab=, aux=,
perm=)}`; `Model.full_weight_map()` is the union over the tree. Two kinds of tensor:

* **Slab tensors** (matrices a GEMV streams): declared through `Linear(in_features, [Part(local, hf_name, rows),
  …], row_perm=, epilogue=)`. Consecutive parts that share a storage format become one row-stacked slab and one
  `gemv` op (`q|gate|k|v`, `in_proj_z|qkv|a|b`, `gate|up`); `row_perm` (the RoPE head-dim permutation, the gate/up
  chunk interleave) applies to one format group only. The mixers' `kernel_segments()` tell the kernels where each
  part's rows start.
* **Aux tensors** (norm weights, conv taps, per-head parameters): `aux=True` with an elementwise `transform`
  (`f32`, `bf16_f32` = the BF16-valued parameter widened, `one_plus` = `1 + w` as F32, `neg_exp` = `−exp(w)`;
  `monolith/packs/transforms.py`) and an optional `perm` on the leading axis. They are stored raw in the pack.

`pack_model` (`tools/pack_weights.py --model <ckpt> --out <pack>`) writes `weights.pack` + `manifest.json` from the
tree alone: slabs (`packs.packer.Packer.add_slab`: unpack through the format plugin, stack, permute, pack in the
profile's lane order, per-row tensor scales), aux tensors, tables. `packs.PackFile(pack).dequantize_slab(name)` and
`aux_array(name)` read a pack back for tests.

**Checkpoints written by other tools.** A conversion (mlx_lm, GGUF-to-safetensors, …) is a *package* matter: set
`model.checkpoint_rename` (stored name → the HF name the tree declares) and, when the tool folds a constant into a
tensor, `model.checkpoint_adapt` (`(name, array, info) → array`, keeping the stored dtype). Both are applied by
`SafetensorsDir` for the packer and the oracle alike (`nn.pack_plan.pack_model`, `load_oracle_weights`). The
qwen3_5 package's `mlx_rename` / `mlx_adapt` are the template: mlx_lm's Qwen3.5 port stores every zero-centered
norm as `1 + w`, so the adapter reads the tensors the tree declares `one_plus` back as `bf16(1 + w) − 1`.
**Before anything else, compare the conversion's small tensors against the HF checkpoint** (norm weights, conv
taps, `A_log`, `dt_bias`, q/k norms): `mx == hf`, `mx == hf + 1`, a transposed or reshaped layout — two hours of
bisection were spent finding this after the pack and every kernel had checked out (porting-log.md, format 2).

### 1.4 Lowering and the coverage guard

`lower()` emits IR only (`core.ir.Graph`: `g.input`, `g.state`, `g.const`, `g.op(kind, inputs, outputs,
domain=BlockDomain(...), **attrs)`) — never Metal. The library modules' `lower(g, h, norm, ctx)` do the work; a
model's `lower` is the walk. Before emission the compiler runs `check_coverage(graph, profile)`: an op kind with no
kernel binding for the target profile fails the build with the list of what is missing (`compiler/coverage.py`),
never at run time. Op kinds today: `embed`, `rmsnorm_stat`, `norm_apply`, `gemv`, `lm_head`, `gqa_decode`,
`gqa_merge`, `gdn_mixer`, `gdn_norm`, `gdn_commit`, `argmax`, the samplers, and the speculative round's ops. A model
that needs anything else needs an op first (§3) — a separate PR, merged before the model PR.

### 1.5 Tests

1. **Contract test on a synthetic checkpoint** (`tests/contract/test_<name>_package.py`, copy
   `test_qwen3_package.py`): write a tiny checkpoint with `formats.safetensors_reader.write_safetensors` and a
   `config.json`, then assert the registry resolves the architecture, `full_weight_map()` claims exactly the
   checkpoint's tensors, the state entries, the op-kind histogram of `lower()`, `check_coverage` on a synthetic
   profile (`Profile.from_dict`), and the pack round-trip (`PackFile.dequantize_slab` equals the stacked, permuted
   checkpoint matrices; `aux_array` equals the transformed aux). No torch, no GPU: this is what CI runs.
2. **Goldens.** Dequantize the checkpoint (`python -m monolith.formats.dequant --model <ckpt> --out <bf16 dir>`;
   every plugin format), run `tools/goldens/hf_golden.py --model <bf16 dir> --out tests/models/<name>/goldens/<tag>`
   on the CPU (transformers' own model: greedy tokens, per-layer hidden states, last logits, teacher-forced top-k
   along the continuation — a later token mismatch can be judged against the golden's own margin). Commit the
   `.json` and `.safetensors`.
3. **GPU golden test** (`tests/models/<name>/test_gpu_golden.py`, copy the qwen3 one): pack from the tree, compile,
   replay, compare the greedy tokens and every layer's prefill residual stream read from the program's buffers.
   Skips without the Metal module or the checkpoint under `~/models`.
4. Optional: the torch oracle against the golden without the GPU (`tests/models/qwen3_5/test_golden.py`) and one
   layer of each kind against the HF modules (`tests/layers/test_layers_vs_hf.py`) — worth it for a new mixer.

### 1.6 Run it

```bash
python tools/pack_weights.py --model ~/models/<ckpt> --out /tmp/pack           # slabs, aux, tables from the tree
#   --scale-placement block: the block's scales in their own region (a unit of whole payload words) where that
#   saves bytes — the M5 Pro profile's choice; --quantize <fmt>: BF16 matrices quantized at pack time
python -m monolith.generate --model ~/models/<ckpt> --pack /tmp/pack --prompt "The capital of France is" -n 48
python -m monolith.trace --model ~/models/<ckpt> --pack /tmp/pack --steps 5    # the per-op budget a token is made of
```

`generate` also takes a drafter (`--drafter <dir> --drafter-pack <pack> --drafter-kind dspark`), sampling
(`--temperature --top-k --top-p --min-p --seed`), and the engine knobs (`--barriers`, `--attention`, `--math`,
`--no-autotune`). The first run autotunes the GEMV and GDN geometries per op and caches the choice in the pack
directory. `python tools/bench/layer_vs_mlx.py --model <ckpt> --pack /tmp/pack` compares a decoder layer's cost with
mlx-lm's on the same checkpoint (the slope of the step over the layer count, T = 1 / 4 / 8, two contexts, per-op
profiles on both sides; decode-kernels.md §11) — the measurement a model or kernel change is judged by.

### 1.7 The PR

Label it `model-pr`. The extension check fails the PR if a file outside `monolith/models/`, `tests/`, `docs/`
changed; if the port needed a library generalization (model 2 needed one flag on `GQAAttention`), that is its own
PR *before* the model PR, with its own test. Record the port in `docs/research/porting-log.md`: time, files, what
the library lacked.

## 2. Adding a format

Format 2 (affine INT4 groups, the MLX / AWQ / GPTQ family) took about six hours: one for the plugin, two and a half
for four engine-side extensions the first formats had never exercised, two for the converter's folded norm
convention (§1.3). A format is a plugin, but the *first* format with a new property (a bias, a quantized embedding, a
shape that is not a multiple of 1024) will touch the kernel template once; expect that and write it up.

### 2.1 The plugin — `monolith/formats/<name>.py`

`@register_format("<name>")` on a `Format` subclass (`formats/base.py`), which supplies:

* `bytes_per_weight`, `weights_per_word` (weights in one 16-byte word of the lane-row unit), `scale_group`
  (weights per block scale, 0 = none);
* `unpack(tensors, shape=(N, K)) → DequantSpec` — the checkpoint's raw arrays by role (`weight`, `weight_scale`,
  `scales`, `biases`, …) plus format constants; `dequantize(spec) → float32 [N, K]` — the oracle, the array the
  numerics contract is defined against; `quantize(w)` for synthetic tests and re-quantization (make it bit-exact
  to the ecosystem's converter if you can — read the converter's kernel, not its docs: mlx's rule starts `w_max`
  at 0 and rounds halves away from zero);
* `pack(spec, layout) → (bytes, PackInfo)` and `unpack_pack` — through `formats.blm.pack_blm(payload [N, 32, P],
  scales [N, 32, S], layout, format=, k=, scale_group=)`: the lane-row unit is `[payload | scales | pad16]`, lane ℓ
  holds columns `[ℓ·K/32, (ℓ+1)·K/32)` of the row and the scales of every group its stripe touches. A stripe need not
  be whole words nor start on a group boundary (a *ragged* stripe — K = 3584 with 32 weights per word): the kernels
  derive the tail mask, the scale offset and the group segmentation from `kernels.unit_geometry(info)`; K must be a
  multiple of 256;
* `msl_decode` — the snippet the GEMV and embed templates paste in: `#define WEIGHTS_PER_WORD`, `#define
  SCALE_GROUP`, `decode_word(uint4, thread float*)` (raw codes as floats — block and tensor scales are the
  template's job), `decode_scale(thread const uint*, uint g)` (`g` is the lane-local group index), and for an
  affine format `#define SCALE_BIAS 1` + `decode_bias(...)` (the template adds `bias · Σx` per group).

Register the module in `monolith/formats/__init__.py`. Two engine-side entries are legitimately part of a format
port: `formats/checkpoint.py` — `detect_format` (the safetensors dtype signature that names the format) and
`logical_shape` (packed → logical K) — and `monolith/bench.py random_spec` (a random matrix *in the format's codes*,
so large synthetic shapes are cheap).

### 2.2 Tests and the bench

* `tests/contract/test_formats.py`: add the format to the parametrized round-trip, ULP and geometry tests
  (`expected_unit` bytes for K = 1024), and a test against the converter (skipped without it — `mlx.core` for
  format 2: the same codes, the same values within its arithmetic, the checkpoint grouping detects the triple).
* `tests/kernels/test_gemv_T.py` (every format × lane order × (rows, T)), `test_gemv_fusions.py` (norm on the
  input, residual and `silu_mul` epilogues), `test_embed_stat_argmax.py` if the format can hold the embedding
  table; the ragged case at K = 3584 when the format's word holds more than 8 weights.
* The M1 harness: `python tools/bench/gemv_bench.py --sweep m1 --formats <name> --ts 1,2,4 --out
  tools/bench/results/<chip>_gemv_m1.jsonl` — every point is checked against the oracle (≤ 1 ULP) and reports GB/s
  and % of nominal; add the rows to `docs/research/gemv-kernel-study.md §2` next to the MLX baseline on the same
  shapes (`tools/bench/mlx_baseline.py`).
* A model on the format end to end: `tests/models/qwen3_5/test_mlx_int4.py` is the pattern (convert into a temporary
  directory, pack from the package, greedy tokens equal to the torch oracle on the same dequantized weights, and a
  sanity check that the first token is the right word — a broken convention decodes to noise while every kernel
  test passes).

## 3. Adding an op

The worked example is the mixture-of-experts path (`monolith/ops/moe.py`, porting-log.md's model 3): a routing op,
an indirection mode of the GEMV template (the expert block comes from a GPU-resident ids buffer), a combine op — each
with a numpy oracle — and the `SparseMoE` layer that composes them; the model package (`qwen3_moe`) then reuses it.

This is the one addition that is an engine PR (`ops/`, `kernels/`, `compiler/`), never folded into a model PR.
Today an op needs:

1. `monolith/ops/<op>.py`: `register_op(OpDef(kind, OpClass.MAP | REDUCE | SERIAL, domain_kind).bind("*" or
   "<family>", KernelBinding("<kernel name>", function_constants)))`; the docstring is the op's contract (inputs in
   order, outputs, attrs, what it reads from `StepState`). Import it in `monolith/ops/__init__.py`.
2. `kernels/common/<op>.metal`: one kernel per op, specialized by macros, the crew geometry (one threadgroup per core,
   12 SIMD-groups × 32 lanes) unless the op is serial; **every loop bounded** (a running dispatch cannot be
   cancelled); correctness only from documented Metal semantics. Keep dispatches sub-millisecond.
3. `monolith/kernels.py`: the `*_source`, `*_macros` and `*_params` helpers that assemble the MSL and the parameter
   struct — the same code the tests and the emitter use.
4. `monolith/compiler/emit.py`: a handler in `HANDLERS[kind]` that binds buffers (`ctx.buf(value)`), scratch
   (`ctx.scratch`), its parameter record (`ctx.params`: the program's own bytes — a session shares its programs'
   buffers by name, a params record never), the grid, and declares which bindings the kernel **writes**
   (`ctx.add(..., writes=[...])`) — the barrier pass (`compiler/barriers.py`) places ICB barriers from these; an
   undeclared write is a race.
5. A library module that lowers to it (`monolith/nn/`), with its torch oracle in `monolith/nn/oracle.py`.
6. Tests: the kernel against a numpy model of its contract and against the layer oracle (`tests/kernels/`,
   the DSpark ops in `test_draft_ops.py` are the pattern), the lowering and coverage without a GPU
   (`tests/contract/`), and a bench row (`tools/bench/`) with the measured cost recorded in
   `docs/research/decode-kernels.md`.

The autotuner (`compiler/autotune.py`) times a small set of variants per op instance on synthetic data of the same
shape; add the op's knobs there only when a measurement shows they matter.

## 4. Adding a drafter

Drafter 1 (`Dogacel/Qwen3-8B-DSpark`) took about 1.5 hours for the module and its oracle; the round itself (the
draft ops, the verify/accept kernels, the dynamic-T program) was the engine's work and is shared by every drafter.

`monolith/spec/<name>/` implements `spec.drafter.Drafter` — a `Module` (weights, oracle, lowering) with `gamma`,
`from_checkpoint(path, target_lm_head=)`, `tap_layers()` (the target layers whose residual streams it reads, `-1`
= the embedding), `lower_draft(g, DraftContext, anchor) → DraftBlock` (tokens, confidences, hidden; the context carries the
target's tapped residual streams, the anchor and the step's token rows),
`lower_select(g, block, profile, cost=, threshold=, fixed=)` (the verify length from the profile's cost table or
the drafter's own rule) and `lower_context_update(g, taps, accepted)`. Register with `@register_drafter("<name>")`
and import the package in `monolith/spec/__init__.py`. The target exposes taps through `Model.feature_taps()` and
`tap_values`; `tools/pack_weights.py --drafter-kind <name>` packs the drafter next to the target's pack (a BF16
drafter is re-quantized at pack time with `--quantize nvfp4 --quantize-keep embed_tokens,markov_w1` (the gathered
tables stay; the Markov head's GEMV, K = 256, packs as NVFP4 through sub-word units): the format's
`quantize`, the gathered tables kept as stored, a matrix whose K the format cannot pack — `pack_k_multiple` — kept
too; the session binds the tree to the pack's formats, so the emitted draft ops carry the quantized kernels);
`generate --drafter … --drafter-kind <name>` runs the round.

**An LM drafter** needs no drafter code: `--drafter-kind lm` runs a registered model package as the draft model
(`monolith/spec/lm`; mlx-lm's `draft_model`). The package's tree must take `prefix` in `from_checkpoint` (every slab,
aux entry, table, state and activation name gets it — `monolith/models/qwen3` does; the tables are stored under the
prefix and read by their bare names), expose `embed_tokens`, `layers()`, `norm`, `lm_head` and `tables()`, and use
attention mixers only (the GDN kernels have no ingest/chain modes: `GatedDeltaNet.lower` refuses a pass mode). Pack it
with `tools/pack_weights.py --model <draft ckpt> --out <pack> --drafter-kind lm` (an MLX 4-bit checkpoint packs as it
is) and run `generate --drafter <draft ckpt> --drafter-pack <pack> --drafter-kind lm --draft-gamma N`; the benches'
`fixed:N` modes chain N. The drafter's vocabulary may not exceed the target's (the two must share a tokenizer).

Two shape rules the emitter enforces: a value that feeds a GEMV has exactly the slab's K columns (a drafter whose
block goes through the target's head has the target's hidden width), and a sequence occupies at most the program's
context capacity — the smallest KV cache of the target and, for the drafter, its context cache less the block it
appends after the context (`Program.context_capacity`; `Session.generate` refuses a request past it, and the serial
ops stop the program with `StepState.error = 2` if a step would reach it, so no kernel writes past a cache). Give
the drafter the target's `max_context`.

Tests: a synthetic drafter for the torch-free lowering test and the GPU program test (`tests/dspark_synth.py`,
`tests/contract/test_dspark_lowering.py`, `tests/kernels/test_draft_program.py`), the oracle against the method's
reference implementation (`tests/spec/test_dspark_oracle.py`, skipped without it), and the speculative golden on a
real target: greedy speculative decode must be token-identical to plain decode (`tests/models/qwen3/test_spec_golden.py`).
Then the measurement (`tools/bench/spec_bench.py`, `docs/research/dspark.md §3`): acceptance, tokens per step, ms per
token against plain decode.

## 5. Adding a chip

Add a chip backend under `monolith/backends/metal/<backend>/` and register its
exact chip name, GPU family and core count in `registry.py`. Give distinct core
variants independent configurations and use separate backends when their kernels
or schedules can diverge; the M5 Max 32-core and 40-core variants do this.
Optional source overrides live under `kernels/<backend>/`; shared templates stay
in `kernels/common/`. See the [backend guide](../monolith/backends/metal/README.md)
for lowering, scheduling and fusion hooks.

Start unmeasured chips with conservative native settings and empty cost tables.
Measurements belong to the chip configuration, not to a shared profile layer.
Two tools provide calibration evidence:

1. `python tools/profile_writer.py` (~5 minutes; `--dry-run` prints without writing) — the autotuner at install
   time: it runs the kernel harnesses and writes the `engine` block the compiler reads — `lane_order` and
   `threadgroups_per_core` (the T = 1 GEMV rate), `cost_T` per format (the shader GEMV at T = 1, 2, 4, 8 relative to
   T = 1), the tile's `accelerator_<fmt>` rows and the `accelerator` / `accelerator_min_t` decision, `attention` (v1,
   v2 and v3 over contexts and T; `auto` = v3, which won every point measured here). A new chip without a spec bandwidth gets a measured stand-in (`--nominal-gbps` sets the spec figure).
   The decisions (`monolith/backends/metal/calibration.py`) follow the autotuner's 3 % noise rule; the raw numbers and the
   reason for each go under `writer`.
2. `./probes/run_all.sh` (~5 minutes, Command Line Tools only) writes `probes/results/<chip>_<cores>c_macOS<ver>_<time>.txt`;
   commit every results file. The probe blocks of the profile are derived from them by hand (the README there lists
   which probe each value comes from), as are the two engine values the writer does not measure: `sibling_order`
   (p11's overlap A/B) and `max_cb_ms` (p6/p6b's sharing) — until then the writer leaves the safe defaults.

`family` is the kernel-binding key (an op bound to `"*"` runs on every family, one bound to `"apple10"` only there);
the verify-length rule refuses to extrapolate outside the measured T range of `cost_T`. The engine picks the profile
by chip name, GPU family and core count (`monolith.backends.metal.config_for_device`). On a chip not yet measured, walk the hypotheses
H1–H10 of `docs/research/apple-gpu-probes.md §4`: they say which design decisions a differing measurement would change.

## 6. Checklists

**Any port**

* No `import mirage`, no MPK/Mirage identifiers; copied code keeps its license header, a provenance line and a
  `third_party/NOTICE` entry (`python tools/ci/hygiene.py`).
* Measure before claiming: paired A/B runs, min-of-N, ranges; evidence tags [M] [S] [R] [H] in the docs.
* The port's account in `docs/research/porting-log.md`: time, files touched, what the library or the engine lacked.

**A model PR**

* `monolith/models/<name>/{__init__,config,model,weights}.py`, the import in `models/__init__.py`.
* `tests/contract/test_<name>_package.py` (synthetic checkpoint, no torch); goldens under
  `tests/models/<name>/goldens/`; `tests/models/<name>/test_gpu_golden.py`.
* `model-pr` label; nothing outside `models/`, `tests/`, `docs/` changed.
* A conversion's conventions checked against the HF checkpoint tensor by tensor before the first run; the test's
  prompt ids from the model's own tokenizer.

**A format PR**

* The plugin with `quantize` bit-exact to the converter, the round-trip / ULP / geometry rows in the contract tests,
  the kernel tests parametrized over the format, the M1 sweep rows in the study, a model on the format end to end.

The [Llama-compatible package](llama-models.md) is a second worked example: Llama 3.2
and SmolLM2 use unnormalized Q/K heads, model-owned rotary scaling and BF16 precision
options while sharing the compiler and runtime with the Qwen packages.
