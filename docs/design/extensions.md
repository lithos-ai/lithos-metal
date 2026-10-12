# Extension contracts

lithos-metal separates model composition from the compiler/runtime and isolates hardware policy in chip
backends. Registries connect the components. A new model built from existing layer types should not require
model-name branches in the engine.

Development setup and test commands are in [CONTRIBUTING.md](../../CONTRIBUTING.md).
This document describes the interfaces an extension must satisfy.

## Models

A package under `monolith/models/<name>/` registers a `monolith.nn.Model` subclass under its checkpoint
architecture. It generally contains configuration, graph composition, checkpoint-weight conventions,
and an import that makes the registration available.

| Interface | Responsibility |
| --- | --- |
| `from_checkpoint` | Interpret configuration, validate supported features, and bind formats |
| `layers` and `state_spec` | Define decoder order and persistent state |
| `feature_taps` and `tables` | Expose draft conditioning points and model-owned constants |
| `weight_map` / `full_weight_map` | Declare checkpoint names, shapes, formats, slabs, auxiliary transforms, and permutations |
| `forward` | Independent oracle behavior |
| `lower` | Compose typed IR through shared layer modules |

Models own normalization conventions, gated versus ordinary attention, rotary scaling, EOS interpretation,
ignored checkpoint components, and renaming/adaptation needed by converted checkpoints.
The packer and oracle must apply the same adaptations.

The [coverage guard](../../monolith/compiler/coverage.py) rejects a graph with no implementation for an operation
on the selected backend. A new operation is a shared extension with its own contract, not a hidden special case
inside one model's lowering.

## Weight formats

A `Format` plugin under `monolith/formats` defines payload geometry, scales, quantization/dequantization,
packing constraints, and Metal decoding behavior. It is registered through
[the format registry](../../monolith/formats/registry.py).

Round-trip checks must distinguish exact code/scale preservation from a conversion that changes numerical
values. A format with new grouping, bias, embedding, or alignment requirements may require a shared kernel
extension. Its eligibility and fallback behavior must be explicit.

A packed slab's input/output dimensions must match every consumer binding. Auxiliary tensors and transformed
normalization parameters are part of the same checkpoint contract.

## Operations and layers

An operation defines inference, state updates, input/output dtypes and shapes, and its block domain.
A Metal binding implements that contract for a backend or supported GPU family.

The emitter binds buffers, scratch, parameter records, and launch geometry. It must declare writes accurately
so [barrier placement](../../monolith/compiler/barriers.py) can derive dependencies. An undeclared write is a
correctness error even when a particular execution order happens to hide it.

A shared layer composes operations and supplies an oracle. Kernel tests validate arithmetic and binding bounds;
contract tests validate graph shape, state ownership, and coverage without a GPU.

## Drafters

A [Drafter](../../monolith/spec/drafter.py) is a module with weights, an oracle, and speculative lowering:

- `from_checkpoint` and `bind_target` establish configuration and any shared target modules.
- `tap_layers` identifies the target features used for conditioning.
- `lower_draft` produces proposal tokens, confidences, and optional hidden states.
- `lower_select` chooses the verified prefix under fixed, threshold, or cost-based policy.
- `lower_context_update` preserves the context associated with committed target positions.

`DraftContext` carries target taps, token rows, and the anchor. `DraftBlock` carries proposal outputs.
The shared verifier and state machinery implement acceptance and commit/rollback.

An ordinary language model can serve as an LM drafter when its package satisfies the shared tokenizer,
prefix naming, model-interface, and attention-state requirements. The serving adapter currently exposes
DSpark. See [speculative decoding](speculative-decoding.md).

## Chip backends

A backend is registered for exact chip name, GPU family, and core count. Core-count variants may own
independent configurations and tuning caches. Shared Metal sources live in `kernels/common`;
same-name sources under `kernels/<backend>` override them for that chip.

Backend hooks can specialize operation handlers, final fusion/scheduling, decoder and draft recipes,
prefill policies, command encoding defaults, and full emission. `reencode_default` opts a backend into direct
serial dispatch encoding instead of ICB replay; callers may explicitly override it for validation. The model graph
and runtime Program ABI remain shared.

Unmeasured devices begin with conservative native fallbacks and empty cost tables. Measurements qualify
particular formats, shapes, and schedules; leaf-kernel calibration alone does not validate a fused layer.
Cache identity includes backend configuration, resolved sources, and relevant backend code.

See [backend ownership](../../monolith/backends/metal/README.md) and
[Apple GPU execution](apple-gpu.md#backend-qualification).

## Validation and compatibility

Every extension must preserve its declared numerical and state contracts:

1. Synthetic contract tests cover registration, shapes, weight maps, pack round-trips, and coverage.
2. Kernel tests compare with independent oracles and run with Metal shader validation.
3. Layer tests check the actual normalization, rotary, gating, residual, and state conventions.
4. Model and speculative continuation tests check tokens and persistent state, including rejection and capacity limits.
5. Performance comparisons use matched workloads and include synchronization and materialization costs.

Optimization paths that change rounding must be explicit and validated under that contract.
Correctness cannot depend on undocumented threadgroup-to-core mapping or opportunistic dispatch overlap.
All GPU loops are bounded.

Adapted code retains its license headers and provenance in [third_party/NOTICE](../../third_party/NOTICE).
The repository's hygiene and model-extension checks enforce the standalone and modularity rules.
