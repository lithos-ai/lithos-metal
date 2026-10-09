# Apple M2 Max, 30 GPU cores

This backend registers the exact `Apple M2 Max` / `Apple8` / 30-core device,
with an independent configuration and tuning-cache identity. It uses the shared
native GPU shaders, with tensor acceleration, automatic mixer fusion and
unmeasured cost tables disabled. The 38-core variant is not qualified here.

## Command encoding

Mixed ICB speculative workloads produced GPU hangs on this device. The backend
therefore selects the runtime's existing direct serial encoding path by default.
The GPU still computes the model and advances sequence state; the host encodes
its dispatches. The remaining ICB failures are unresolved, and this fallback
makes no latency or speedup claim.

`Backend.reencode_default` and the small `Engine` connection apply this policy
automatically to Programs carrying the M2 backend identity, including after JSON
round-trip. Other backends keep ICB as their default. Standalone Programs with unregistered
backend metadata retain the previous ICB default. Explicit
`Engine.run(reencode=True/False)` overrides remain available for diagnostics.
Device configuration alone cannot select this existing runtime argument.

## Qualification

Validated on Apple M2 Max (30 GPU cores), 64 GB unified memory, macOS 26.7
(`25G229`), SDK 26.2, Apple clang 17.0.0 and Python 3.13.7 on 2026-10-09.
The reference dependencies were PyTorch 2.14.0 and Transformers 5.17.0.
The initial repository baseline for hardware probes was
`236475fa716434807668562cd878ae623c2c1fd5`.

The BF16 reference checkpoint is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. Its safetensors shard has SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The first-MiB fingerprint starts with `667d46579c60b536`, matching the retained
golden metadata.

| Check | Result |
| --- | --- |
| Native runtime CMake/Ninja build | Passed |
| Contract tier | 1062 passed; optional MLX comparison skipped because MLX was absent from this venv |
| Runtime tier plus backend contracts, with shader validation | 43 passed, including backend encoding defaults, explicit overrides and unregistered Program metadata |
| Real-model gate, with shader validation and the M2 default policy | 6 passed: all 24 prefill layer outputs, two identical 48-token golden continuations, long-prompt handoff, reproducible sampling, fast math, and real-target rollback through synthetic drafts |
| Focused mixed-workload gate, with shader validation | 18 passed: LM drafter, sampling distributions, speculative rollback, and large-prefill/prefix reuse with LM and resident DSpark; all 34,956 Engine calls used the M2 default policy |
| Normal serving CLI | Passed warmup, health/model discovery, repeated greedy requests, prefix reuse, matching SSE text/usage, Chat Completions/Responses/Messages and recovery after streaming disconnect; BF16 reference target without a drafter, at 1024-token capacity |

The configuration uses `validation: unmeasured` because performance has not
been benchmarked. Correctness qualification covers the device and workloads
listed above.

## Hardware probes and limits

The [raw probe run](../../../../probes/results/Apple-M2-Max_30c_macOS26.7_20261009-105128.txt)
attempted all 17 probes. Sixteen completed; `p10_claim_protocol` stalled in
`MTLCommandBuffer.waitUntilCompleted` and was terminated with exit 143.
Cross-threadgroup handoffs in `p2` timed out. Neither result is treated as a pass
or used to enable cross-worker fusion on Apple8. The `p7` text hardcodes "M3 Pro";
the run header and detected device identify the actual M2 Max.

The following optional kernel checks failed:

- `test_norm_and_silu_intermediate_rounding[gemm_tile-bf16-4]`
- `test_fused_norm_matches_separate_passes[False-8-False-None-1]`

Both paths are disabled by this backend's default. Experimental kernel
validation is incomplete.

Synthetic drafter checks and the real-target rollback gate do not qualify large
catalogue target/DSpark pairs or a large public MoE checkpoint. No latency,
bandwidth, acceptance or speedup claim is made. A passing bounded run is not an
exhaustive stability guarantee.

## Reproduce

Follow [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for the environment and native
build. Install `.[dev,oracle,serve]`, use the reference dependency versions above,
and download the pinned BF16 checkpoint into `$MONOLITH_MODELS/Qwen3.5-0.8B`.
Run the gates through the backend default, without overriding `Engine.run`:

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
  --no-draft --local-files-only --pack /tmp/lithos-metal-m2-serve \
  --max-context 1024 --port 18089
```

The serving smoke used actual HTTP requests to verify repeated greedy output,
long-system-prefix cache reuse, SSE equivalence and usage, all three protocol
adapters, and a new request after disconnecting an active stream. It qualifies
the target-only reference workload at the stated capacity.
