# Apple M2 Max, 30 GPU cores

This backend registers the exact `Apple M2 Max` / `Apple8` / 30-core device.
It uses the shared native shader implementations, with tensor acceleration and
automatic mixer megakernel fusion disabled. It defaults to direct serial
dispatch encoding: the GPU still advances sequence state and produces tokens,
but the host re-encodes dispatches rather than replaying an ICB. ICB replay is
available through an explicit `Engine.run(reencode=False)` diagnostic override,
and is not qualified for this device. The 38-core M2 Max and other M2
variants remain unregistered; they need their own qualification.

## Qualification

Validated on a bare-metal M2 Max with 64 GB unified memory, macOS 26.7
(`25G229`), SDK 26.2 and Apple clang 17.0.0 on 2026-10-09. Python was 3.13.7;
the reference dependencies matched the retained goldens: PyTorch 2.14.0 and
Transformers 5.17.0. The starting repository revision was
`236475fa716434807668562cd878ae623c2c1fd5`.

The reference checkpoint is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`, in BF16. Its single safetensors
shard has SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The SHA-256 of the first MiB starts with `667d46579c60b536`, matching the
fingerprint recorded in the repository's golden metadata.

| Check | Result |
| --- | --- |
| CMake/Ninja native module build | Passed |
| Wheel build and packaged-runtime smoke test | Passed; the wheel includes the current M2 encoding policy, configuration, native extension and shared shaders |
| Contract tier | 1062 passed after merging the upstream M3 backend; optional MLX comparison skipped because MLX was absent from this venv |
| Runtime tier plus backend contracts, with shader validation | 44 passed after the upstream merge, including encoding policy and the shared-state regression |
| Default encoding model and layer run, with shader validation | 13 passed: GPU goldens, real-model speculative rollback and all layer tests; CPU model goldens also passed in the same 15-test run |
| CPU model golden | Both tests passed |
| Native projection/embedding/GDN checks with shader validation | 382 passed before the first optional fused-norm case failed; full kernel tier is not qualified |
| Mixed state/drafter/MoE/attention/sampling/large-prefill checks, with shader validation | 105 passed, 8 inapplicable resident-prefill combinations skipped; 38,113 Engine calls used the M2 backend's default encoding |
| Real serving entry point, BF16 reference model without a drafter, 1024-token capacity | Passed startup/warmup, health/model discovery, repeated greedy requests, prefix reuse, SSE text/usage, Chat Completions/Responses/Messages and recovery after a streaming disconnect |
| Initial ICB model/layer run | 13 passed before the runtime follow-up; narrower evidence than mixed-workload qualification |
| Initial direct-encoding shader-validation check | Two 48-token continuations exactly matched the retained golden using default launch geometry and no autotuning |

The GPU model checks include all 24 prefill layer outputs, two identical
48-token greedy continuations, the retained 19-token prompt / 32-token
continuation, seed-dependent reproducible sampling, fast math and recurrent
state rollback through rejected synthetic drafts. Synthetic hybrid MoE
checks cover shared experts and repeated continuation; they do not qualify a
large public MoE checkpoint.

The configuration keeps `validation: unmeasured`: performance tuning remains
unmeasured. It has no borrowed cost table, nominal bandwidth claim or fusion
recipe. Qualification of these workloads is separate from tuning or an
end-to-end performance comparison.

## Runtime follow-up

The host pump originally waited for its oldest command buffer and accessed the
shared ring and StepState while later buffers could still be running. It now
waits for the entire pending batch before draining tokens, publishing
`ring_tail`, or observing `done`. A bounded two-buffer regression fails against
the original runtime (the GPU observes an early host-published tail) and passes
with the fix under both ICB replay and direct encoding. This follows Apple's
[shared-storage synchronization requirement](https://developer.apple.com/documentation/metal/mtlstoragemode/shared).
The pump still submits up to `in_flight` buffers per batch.

ICB command barriers are now set before encoding their indirect dispatch, as
required by [the Metal API](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setbarrier%28%29).
These shared-runtime changes are independent of the M2 encoding policy.

The pump fix allowed all 12 speculative sampling/rollback tests to pass in a
normal ICB run. It did not qualify all ICB workloads: an LM-drafter mixed run
hung in a one-buffer prefill, and a resident DSpark decoder with prefix reuse
hung during decode. An LM case also hung with a barrier on every command after
the pump fix. Direct encoding passed the same LM and large-prefill workload
families under shader validation, so the exact M2 backend defaults to that
existing serial path. Other backends retain ICB replay; explicit bool values
on `Engine.run(reencode=...)` take precedence over the backend default.

The cause of the remaining ICB hangs is unresolved. The direct-encoding
fallback is a correctness qualification, not a repair or performance claim
for ICB replay. Its CPU/latency cost has not been measured with paired A/B runs.

The [raw probe run](../../../../probes/results/Apple-M2-Max_30c_macOS26.7_20261009-105128.txt)
attempted all 17 probes. Sixteen completed; `p10_claim_protocol` was terminated
after waiting over two minutes in `MTLCommandBuffer.waitUntilCompleted`.
Its exit 143 is retained in the output, not treated as a passing result.
Cross-threadgroup handoffs in `p2` also timed out. These checks do not qualify
a cross-worker fusion protocol on Apple8. The `p7` scenario text hardcodes
"M3 Pro"; the run header and detected device identify the actual M2 Max.

`p14` returned numerical results for its staged half-operand TensorOps probes;
this does not qualify the engine's BF16 cooperative-tensor layout. The failed
GEMM-tile check below is why acceleration remains off.

The [initial leaf calibration](../../../../probes/results/Apple-M2-Max_30c_calibration_20261009.json)
is a separate candidate produced with `tools/profile_writer.py --quick
--no-accelerator --formats fp8_e4m3,nvfp4`. Its measured decisions are retained
for follow-up work and have not been applied to `config.json`. It is not an
alternating end-to-end A/B study or a large-model qualification.

## Reproduce

Follow the environment setup in [CONTRIBUTING.md](../../../../CONTRIBUTING.md),
then install the reference dependencies and download the pinned checkpoint:

```bash
pip install -e '.[dev,oracle,serve]'
pip install 'torch==2.14.0' 'transformers==5.17.0'
export MONOLITH_MODELS=/tmp/lithos-metal-models
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --include '*.json' --include '*.safetensors' --include '*.txt' \
  --include '*.jinja' --include LICENSE \
  --local-dir "$MONOLITH_MODELS/Qwen3.5-0.8B"
python -m pytest tests/contract
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/runtime tests/contract/test_metal_backends.py
OMP_NUM_THREADS=8 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/models/qwen3_5/test_gpu_golden.py \
  tests/models/qwen3_5/test_spec_rollback.py tests/layers
OMP_NUM_THREADS=8 python -m pytest tests/models/qwen3_5/test_golden.py
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/kernels/test_draft_ops.py tests/kernels/test_moe_ops.py \
  tests/kernels/test_moe_program.py tests/kernels/test_program_sharing.py \
  tests/kernels/test_eos_set.py tests/kernels/test_lm_drafter.py \
  tests/kernels/test_gqa_decode.py::test_matches_kernel_contract \
  tests/kernels/test_gqa_decode.py::test_repeat_runs_are_bit_identical_and_t_active \
  tests/kernels/test_gqa_decode.py::test_matches_layer_oracle \
  tests/kernels/test_spec_sampling.py tests/kernels/test_spec_rollback.py
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/kernels/test_large_prefill.py -k off
# p10 stalled on this device; omit it when reproducing the remaining probes.
GPU_CORES=30 ./probes/run_all.sh \
  p1_limits p2_sync p3_residency p4_core_model p5_bandwidth p5b_access_pattern \
  p6_preemption p6b_interleave p7_dispatch_overhead p8_threadgroup_mem \
  p9_clock_warp p11_interop_overlap p12_stream_geometry p13_decode_gemv \
  p14_tensor_ops p15_frame_pacing
```

## Serving smoke

The normal serving CLI was exercised on the reference checkpoint without
encoding overrides or a custom recipe. Repeated greedy responses were equal,
streamed text matched the non-streaming response, and a repeated long system
prefix reused the cache. After disconnecting a streaming request, the server
accepted another generation. This checks the target-only reference workload;
it does not qualify a large catalogue target/DSpark pair.

```bash
python -m monolith.serve --model "$MONOLITH_MODELS/Qwen3.5-0.8B" \
  --no-draft --local-files-only --pack /tmp/lithos-metal-m2-serve \
  --max-context 1024 --port 18089
# In another terminal after warmup, with the default unauthenticated local setup:
curl --fail http://127.0.0.1:18089/health
curl --fail http://127.0.0.1:18089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-0.8B","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_completion_tokens":16}'
```

## Validation limitations

One repeated shader-validation runtime run exceeded the existing profiling
test's 5 ms timing assertion (11.54 ms). Its isolated recheck and the final
44-test runtime/backend run passed. The assertion was retained unchanged;
these instrumented timings are not performance qualification.

Two experimental paths failed the initial shader-validation sweep:
`test_norm_and_silu_intermediate_rounding[gemm_tile-bf16-4]` and
`test_fused_norm_matches_separate_passes[False-8-False-None-1]`.
Neither path is enabled by this backend's default configuration. This is
not a claim that the entire kernel tier passes on Apple8.

Normal-mode ICB replay remains unqualified for mixed speculative workloads,
as described above. The direct path's synthetic drafter checks and real-model
rollback gate do not qualify a large public target/drafter pair. A single
passing run is not an exhaustive stability guarantee.

Instrumented ICB model replay produced
`kIOGPUCommandBufferCallbackErrorHang`, including in an isolated greedy test.
The same checkpoint passed normal ICB replay and instrumented direct
re-encoding. Apple's [shader validation documentation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage)
requires pipeline and buffer inheritance for ICB validation; the runtime's
per-command ICB bindings do not enable inheritance. Direct re-encoding is
therefore the validation route used here, and the backend's default execution
route; instrumented ICB replay remains unqualified. No shader implementation
changes are included.

To reproduce the instrumented direct-encoding check with the default geometry:

```bash
python tools/pack_weights.py --model "$MONOLITH_MODELS/Qwen3.5-0.8B" \
  --out /tmp/lithos-metal-m2-pack --max-context 512
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 python - <<'PY'
import json, os
from monolith.generate import Session
from monolith.models.qwen3_5 import Qwen3_5Model
from monolith.runtime.engine import Engine

original_run = Engine.run
def reencoded_run(self, *args, **kwargs):
    kwargs['reencode'] = True
    return original_run(self, *args, **kwargs)
Engine.run = reencoded_run
golden = json.load(open('tests/models/qwen3_5/goldens/qwen3_5-0.8b.json'))
model = Qwen3_5Model.from_checkpoint(
    os.path.join(os.environ['MONOLITH_MODELS'], 'Qwen3.5-0.8B'), max_context=512)
session = Session(model, '/tmp/lithos-metal-m2-pack', eos=-1, autotune=False)
for _ in range(2):
    generation = session.generate(golden['prompt_ids'], len(golden['gen_ids']))
    assert generation.tokens == golden['gen_ids']
print('Two instrumented 48-token continuations match the golden')
PY
```
