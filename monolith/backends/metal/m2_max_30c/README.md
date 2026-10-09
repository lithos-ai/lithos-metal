# Apple M2 Max, 30 GPU cores

This backend registers the exact `Apple M2 Max` / `Apple8` / 30-core device.
It uses the shared native shader implementations, with tensor acceleration and
automatic mixer megakernel fusion disabled. The 38-core M2 Max and other M2
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
| Wheel build | Passed; the wheel includes the new backend/configuration and shared Metal sources |
| Contract tier | 1059 passed; optional MLX comparison skipped because MLX was absent from this venv |
| Runtime tier plus backend contracts, with shader validation | 37 passed |
| Normal ICB model and layer run | 13 passed: GPU goldens, real-model speculative rollback and all layer tests |
| CPU model golden | Both tests passed |
| Native projection/embedding/GDN checks with shader validation | 382 passed before the first optional fused-norm case failed; full kernel tier is not qualified |
| Additional normal-mode state/drafter/MoE/attention checks | 79 passed before the DSpark large-prefill case hung; that case passed in isolation |
| Shader validation, direct re-encoding | Two 48-token continuations exactly matched the retained golden using default launch geometry and no autotuning |

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
pytest tests/contract
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  pytest tests/runtime tests/contract/test_metal_backends.py
OMP_NUM_THREADS=8 pytest tests/models/qwen3_5/test_gpu_golden.py \
  tests/models/qwen3_5/test_spec_rollback.py tests/layers
OMP_NUM_THREADS=8 pytest tests/models/qwen3_5/test_golden.py
# p10 stalled on this device; omit it when reproducing the remaining probes.
GPU_CORES=30 ./probes/run_all.sh \
  p1_limits p2_sync p3_residency p4_core_model p5_bandwidth p5b_access_pattern \
  p6_preemption p6b_interleave p7_dispatch_overhead p8_threadgroup_mem \
  p9_clock_warp p11_interop_overlap p12_stream_geometry p13_decode_gemv \
  p14_tensor_ops p15_frame_pacing
```

## Validation limitations

Two experimental paths failed the initial shader-validation sweep:
`test_norm_and_silu_intermediate_rounding[gemm_tile-bf16-4]` and
`test_fused_norm_matches_separate_passes[False-8-False-None-1]`.
Neither path is enabled by this backend's default configuration. This is
not a claim that the entire kernel tier passes on Apple8.

Two normal-mode speculative checks also encountered
`kIOGPUCommandBufferCallbackErrorHang`:

- `tests/kernels/test_spec_sampling.py::test_sampled_drafts_preserve_the_target_distribution[top-k+top-p]`
  failed both in a mixed run and in isolation. It explicitly selects attention
  `v2` and sampled drafts; the registered default remains `v1`.
- `tests/kernels/test_large_prefill.py::test_large_prefill_shares_state_with_small_decode[False-False-False-off-dspark]`
  failed after 79 passing checks in a mixed run, but passed in isolation.
  The equivalent plain and LM-drafter cases passed in the mixed run, covering
  prompt lengths 5, 128, 129, 259, 137 and 265 with the default 128-row chunks.
  The DSpark case uses `v1` with acceleration off, so disabling experimental
  attention alone does not qualify all DSpark workloads.

These failures leave sampled-draft distribution and mixed-workload DSpark
stability unqualified. Passing real-model rollback and synthetic layer checks
are narrower evidence, not a full speculative-serving qualification.

Instrumented ICB model replay produced
`kIOGPUCommandBufferCallbackErrorHang`, including in an isolated greedy test.
The same checkpoint passed normal ICB replay and instrumented direct
re-encoding. Apple's [shader validation documentation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage)
requires pipeline and buffer inheritance for ICB validation; the runtime's
per-command ICB bindings do not enable inheritance. Direct re-encoding is
therefore the validation route used here; instrumented ICB replay remains
unqualified. No runtime or kernel implementation changes are included.

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
