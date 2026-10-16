# Contributing

## Setup (macOS, Command Line Tools; Homebrew only for Python/CMake)

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,oracle]"        # oracle = torch/safetensors/transformers, needed for goldens and kernel oracles
pytest tests/contract                 # no GPU, runs on any machine
cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(pwd)/.venv/bin/python && cmake --build build   # the Metal runtime module
pytest tests/kernels tests/runtime    # GPU kernel and runtime tests (skipped automatically when the module is not built)
pytest tests/layers tests/models      # oracle tiers: layer oracles vs the HF modules, the model oracle vs its golden
                                      # (need torch + the small checkpoint under ~/models or $MONOLITH_MODELS; skip otherwise)
python tools/goldens/hf_golden.py --model ~/models/Qwen3.5-0.8B --out tests/models/qwen3_5/goldens/qwen3_5-0.8b   # regenerate a golden
python tools/pack_weights.py --model ~/models/Qwen3.5-0.8B --out /tmp/pack   # pack a checkpoint from its model package
python tools/bench/gemv_bench.py --format fp8_e4m3 --shape 17408x5120     # a kernel bench (see tools/bench/README.md)
./probes/run_all.sh                   # hardware characterization (Apple GPU, ~5 min)
```

## Release bundles

On an Apple silicon Mac with Python 3.12 and Xcode Command Line Tools, run:

```bash
python3.12 tools/release/build_macos.py
```

This creates an archive in `dist/` containing the compiled wheel and serving dependency wheels,
plus a Homebrew formula with the archive's SHA-256. The
[macOS release workflow](.github/workflows/release.yml) smoke-tests the bundle. Manual runs build one for review; a
`vVERSION` tag matching `pyproject.toml` publishes the bundle the [release gate](docs/ci.md#releasing) built and
measured on the tagged commit, and opens
the pull request that copies the generated formula into `Formula/lithos-metal.rb` in the
[Lithos tap](https://github.com/lithos-ai/homebrew-tap), whose CI validates it with `brew install` and `brew test`.
Regenerate the archive and formula together for each version.

## Test tiers

| Tier | Needs | What it checks |
|---|---|---|
| `tests/contract` | nothing (numpy) | registries, IR, formats, packer, StepState, the layer library's lowering and pack plan on a synthetic checkpoint, hygiene |
| `tests/kernels`, `tests/runtime` | the Metal module | kernels vs their torch-free oracles; the ICB/ring/pump runtime on toy programs |
| `tests/layers` | torch, transformers, the small checkpoint | each layer's `forward()` oracle vs the HF module on real weights (cos > 0.999, max-abs ≤ scale/64) |
| `tests/models` | torch, the small checkpoint (+ the Metal module for `test_gpu_golden`) | the model oracle vs its HF golden: per-layer hidden states (cos > 0.999) and 48 greedy tokens; the same through the compiled step program on the GPU |

After a kernel or emitter change, run the GPU tiers under Metal shader validation:

```bash
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 pytest tests/kernels tests/runtime
```

Validation drops a store outside its binding and logs it on the command buffer, which still completes without an
error. The runtime raises each report where the command buffer completes (`Queue.run`, `Queue.profile`, `Engine.run`,
`Engine.profile`) as `RuntimeError: shader validation: <report>`, so the test that ran the kernel fails whatever it
checks, in any capture mode (`-s` included), and a report in a fixture fails the run as an error.
`MTL_SHADER_VALIDATION=1` alone is enough; `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1` adds each full report on
stderr (otherwise it goes to the unified log). Command buffers the runtime does not commit, such as MLX's in the
mlx-lm comparisons, are not checked. On an M3 Max under macOS 27, validation reported out-of-bounds stores but not
out-of-bounds loads.

CI runs the contract tier on every push and the other tiers on bare-metal Macs when a maintainer comments
`/mac-test` on the pull request, under validation with `/mac-test --shader-validation` ([CI](docs/ci.md)).

Goldens live in `tests/models/<name>/goldens/` (`<tag>.safetensors` + `<tag>.json`), produced by
`tools/goldens/hf_golden.py` from the HF model in BF16 with eager attention on the CPU; the JSON records the
transformers/torch versions and the parameter dtypes the reference held.

## Rules that CI enforces

* **Standalone ([design](docs/design/design.md#portability-and-reuse)).** Copying from MPK/mirage, MLX, llama.cpp, tinygrad, DeepSpec, DFlash or gpt-oss is
  encouraged. Keep the license header, add a provenance line (`# adapted from <repo> <path> @ <commit>`), and add an
  entry to `third_party/NOTICE`. Never `import mirage`; never name anything MPK/Mirage. `tools/ci/hygiene.py` checks.
* **Model-agnostic ([extension contracts](docs/design/extensions.md#models)).** Model names appear only under `monolith/models/<name>/`. A PR labelled
  `model-pr` may change only `monolith/models/`, `tests/`, `docs/` (`tools/ci/extension_check.py`). New ops, formats
  and drafters land as their own packages under their registry, with oracle tests.
* **Measure before claiming.** Every performance number in a PR carries its A/B table (paired alternating runs,
  min-of-N, same machine, same day).

## Branches and PRs

Roadmap tasks are GitHub issues (#1 is the master). One branch per task, `roadmap/<issue>-<slug>`, stacked on the
previous task's branch; PRs target the branch below them and are merged bottom-up. Each PR closes its issue and
states its position in the stack.
