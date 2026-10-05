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

## Test tiers

| Tier | Needs | What it checks |
|---|---|---|
| `tests/contract` | nothing (numpy) | registries, IR, formats, packer, StepState, the layer library's lowering and pack plan on a synthetic checkpoint, hygiene |
| `tests/kernels`, `tests/runtime` | the Metal module | kernels vs their torch-free oracles; the ICB/ring/pump runtime on toy programs |
| `tests/layers` | torch, transformers, the small checkpoint | each layer's `forward()` oracle vs the HF module on real weights (cos > 0.999, max-abs ≤ scale/64) |
| `tests/models` | torch, the small checkpoint (+ the Metal module for `test_gpu_golden`) | the model oracle vs its HF golden: per-layer hidden states (cos > 0.999) and 48 greedy tokens; the same through the compiled step program on the GPU |

Goldens live in `tests/models/<name>/goldens/` (`<tag>.safetensors` + `<tag>.json`), produced by
`tools/goldens/hf_golden.py` from the HF model in BF16 with eager attention on the CPU; the JSON records the
transformers/torch versions and the parameter dtypes the reference held.

## Rules that CI enforces

* **Standalone (design D15).** Copying from MPK/mirage, MLX, llama.cpp, tinygrad, DeepSpec, DFlash or gpt-oss is
  encouraged. Keep the license header, add a provenance line (`# adapted from <repo> <path> @ <commit>`), and add an
  entry to `third_party/NOTICE`. Never `import mirage`; never name anything MPK/Mirage. `tools/ci/hygiene.py` checks.
* **Model-agnostic (design D16, §5.14).** Model names appear only under `monolith/models/<name>/`. A PR labelled
  `model-pr` may change only `monolith/models/`, `tests/`, `docs/` (`tools/ci/extension_check.py`). New ops, formats
  and drafters land as their own packages under their registry, with oracle tests.
* **Measure before claiming.** Every performance number in a PR carries its A/B table (paired alternating runs,
  min-of-N, same machine, same day).

## Branches and PRs

Roadmap tasks are GitHub issues (#1 is the master). One branch per task, `roadmap/<issue>-<slug>`, stacked on the
previous task's branch; PRs target the branch below them and are merged bottom-up. Each PR closes its issue and
states its position in the stack.
