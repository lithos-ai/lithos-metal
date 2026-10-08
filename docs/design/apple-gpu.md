# Apple GPU execution model

This document describes the Apple GPU properties that shape lithos-metal's compiler and runtime.
Device-specific measurements belong in backend configurations and benchmark artifacts; the contracts
below must hold independently of one machine's timing results.

## SIMD groups and workers

A Metal SIMD group is the unit of lockstep execution. Cooperative operations require participating
lanes to follow compatible control flow. A threadgroup contains one or more SIMD groups and can share
threadgroup memory and barriers.

A megakernel worker is a threadgroup that processes a bounded sequence of tasks. Its size, number of SIMD
groups, and task geometry are backend parameters. The compiler does not assume that worker IDs identify
physical cores or that an arbitrary number of threadgroups are resident simultaneously.

Cross-worker waits require a schedule whose progress conditions are satisfied. Unsupported geometry is
rejected, and bounded waits report failure rather than spinning indefinitely.

## Inline matrix computation

M5 GPUs expose neural-accelerator matrix operations through Metal Performance Primitives and TensorOps.
The calling SIMD groups participate cooperatively, allowing a shader to combine matrix arithmetic with
normalization, gating, and reductions. Cooperative tensors distribute values across participating threads.

This supports task bodies that keep suitable intermediate values within a worker. It does not remove the
need to publish results consumed by another worker. Backend selection controls accelerator eligibility;
a compatible operation or OS API alone does not qualify an unmeasured chip recipe.

## Memory and locality

Apple silicon has unified CPU/GPU memory. Sharing an allocation still requires correct synchronization;
it does not remove the cost of GPU loads or stores.

Apple GPUs from the M3 generation dynamically allocate on-chip storage across register, threadgroup,
and buffer data. The compiler therefore tunes tile sizes, intermediate lifetimes, and concurrency together.
Keeping more data live can reduce the number of workers that make useful progress.

Weight packs arrange payloads and scales for the consuming lanes and matrix tiles. A layout is specific
to the format, kernel, and backend. Merely reducing the number of dispatches does not guarantee lower
memory traffic: cross-worker intermediates can still require device storage.

## Synchronization and dispatch boundaries

Within a threadgroup, barriers protect shared scratch and stage transitions. Across threadgroups,
publication requires the memory ordering supported by the selected Metal language version, in addition
to a progress protocol.

An ICB dispatch boundary can provide ordering for whole-region dependencies. The compiler chooses between
an in-kernel join and separate dispatches using the complete cost of the region. Global barriers are not
assumed to be cheaper just because they occur inside one kernel.

Each dispatch and command-buffer batch is bounded. GPU sharing and preemption vary with the device and OS,
so the runtime cannot rely on immediate interruption of a long-running kernel. Cancellation stops future
work after the active dispatch completes.

## Independent work

An independent projection can be scheduled beside a recurrence or attention task when their dependencies
permit it. Concurrent work competes for execution units, storage, and memory bandwidth; overlapping two
bandwidth-heavy tasks need not improve throughput.

The compiler records actual reads and writes, while the backend selects scheduling geometry.
Omitting a barrier requires dependency analysis, not a timing observation.

## Backend qualification

A backend owns chip identity, kernel overrides, calibration data, fusion policy, and context recipes.
Qualification should establish:

- Supported launch geometry and the progress conditions of any cross-worker protocol.
- Memory-access patterns and operand layouts for the relevant formats and shapes.
- Native and accelerated matrix paths, including their numerical behavior.
- Attention partitioning, recurrent-state slicing, and whole-region synchronization.
- Prefill/decode state handoff, capacity bounds, cancellation, and deterministic continuation.

Use the [probe suite](../../probes/README.md) to characterize hardware and the
[backend guide](../../monolith/backends/metal/README.md) to register a configuration.
Keep model-level performance conclusions separate from isolated hardware probes.

## References

- [Apple: inline ML operations in Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
- [Apple: GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/)
- [Apple: LLMs and M5 GPU neural accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
