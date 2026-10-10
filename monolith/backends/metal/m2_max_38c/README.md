# Apple M2 Max, 38 GPU cores

Registers `Apple M2 Max` / `Apple8` / 38 cores with its own configuration and
tuning-cache identity. Shared native shaders are used; tensor acceleration,
automatic mixer fusion and cost tables remain disabled.

The direct-encoding policy follows Leo's 30-core M2 contribution in [PR #9](https://github.com/lithos-ai/lithos-metal/pull/9).
The shared `Backend.reencode_default` / `Engine.run` connection is attributed in
`third_party/NOTICE`. Other backends and unregistered Programs keep their ICB
default; explicit encoding overrides remain available. The 30-core test results
are not evidence for this device.

## Local qualification

Tested on a 38-core M2 Max with 64 GB unified memory, macOS 27.0.1, Apple clang
21.0.0, Python 3.13.15, PyTorch 2.14.0 and Transformers 5.17.0.

The BF16 `Qwen/Qwen3.5-0.8B` reference weights were downloaded from ModelScope.
Their SHA-256 is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`,
matching the pinned reference used in PR #9. Configuration, index and tokenizer
files were also checked against ModelScope's published SHA-256 values.

| Check | Result |
| --- | --- |
| CMake/Ninja native build | Passed |
| Contract tier | 1057 passed, 1 optional MLX comparison skipped |
| Runtime and backend contracts, shader validation on | 38 passed; defaults, explicit overrides and Program round-trip covered |
| Real BF16 target, shader validation on | 6 passed; layer oracles, repeated golden tokens, long prefill, sampling, fast math and speculative rollback |
| Synthetic mixed workloads, shader validation on | 18 passed; LM/DSpark drafts, sampling distribution, rollback and prefix reuse |
| Actual serving CLI, shader validation on | Warmup and HTTP passed at 1024 capacity; repeated 384-token request reused 364 prefix tokens, with identical visible text and SSE usage |

Reproduction (after the build in `CONTRIBUTING.md`, with the reference model
under `$MONOLITH_MODELS/Qwen3.5-0.8B`):

```bash
python -m pytest tests/contract
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/runtime tests/contract/test_metal_backends.py
OMP_NUM_THREADS=8 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/models/qwen3_5/test_gpu_golden.py \
  tests/models/qwen3_5/test_spec_rollback.py
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/kernels/test_lm_drafter.py \
  tests/kernels/test_spec_sampling.py tests/kernels/test_spec_rollback.py \
  'tests/kernels/test_large_prefill.py::test_large_prefill_shares_state_with_small_decode[False-False-True-off-lm]' \
  'tests/kernels/test_large_prefill.py::test_large_prefill_shares_state_with_small_decode[False-True-True-off-dspark]'
python -m monolith.serve --model "$MONOLITH_MODELS/Qwen3.5-0.8B" \
  --no-draft --local-files-only --pack /tmp/lithos-m2-38c-serve \
  --max-context 1024 --port 18090
```

## Limits

`validation: unmeasured` refers to performance. No large model/draft pair, 128K
workload, Flash Next adapter or speedup is qualified by the small reference
tests. No hardware probe result from another chip is used as tuning data.
ICB mixed-workload stability on this 38-core device has not been qualified;
direct encoding is the conservative default inherited from the M2 design.
