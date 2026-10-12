# Metal chip backends

Each backend owns its configuration, lowering hooks, fusion policy, encoding default and optional
Metal source overrides. Model graphs and the runtime Program ABI stay shared.
M5 Max 32-core and 40-core are independent backends; neither inherits the other's
configuration or autotuning cache.

| Backend | Configuration | Status |
| --- | --- | --- |
| `m2_max_30c` | [30 cores](m2_max_30c/config.json) | Native shaders with direct encoding; [validation and limitations](m2_max_30c/README.md), performance tuning unmeasured |
| `m3_pro` | [18 cores](m3_pro/config.json) | Existing probe-derived settings preserved |
| `m3_max_30c` | [30 cores](m3_max_30c/config.json) | Unmeasured native fallback. On the 36 GB machine, DSpark serving fits through 20480 context; 22528 does not |
| `m3_ultra_60c` | [60 cores](m3_ultra_60c/config.json) | Measured (`profile_writer`, 256 GB): the Apple9 matrix tile from 4 rows |
| `m4_pro` | [16 cores](m4_pro/config-16c.json), [20 cores](m4_pro/config-20c.json) | 20 cores measured (`profile_writer`, 48 GB): the Apple9 matrix tile from 2 rows (NVFP4); 16 cores an unmeasured native fallback |
| `m4_max` | [32 cores](m4_max/config-32c.json), [40 cores](m4_max/config-40c.json) | Unmeasured native fallback |
| `m5_pro` | [20 cores](m5_pro/config.json) | Existing measured settings preserved |
| `m5_max_32c` | [32 cores](m5_max_32c/config.json) | Unmeasured native fallback |
| `m5_max_40c` | [40 cores](m5_max_40c/config.json) | Measured GDN default and explicit attention/MLP/draft recipes preserved |

A configuration can ship measured kernel choices (`kernel_choices`, a file beside it): the autotuner's cache
entries from that chip for its decode programs, plus the prompt pass's 32-row tile geometry measured in the prompt
program. `serve` compiles with them (`Session(measured_choices=True)`); shapes they do not list keep the defaults, and
`python -m monolith.generate --no-autotune` keeps the default geometry everywhere.
Regenerate them when kernels change: `python -m monolith.generate` tunes and caches per pack.

Unmeasured configurations have no cost tables or automatic megakernel fusion,
and keep the tensor accelerator off. They provide a starting point for validation
on those devices, not a performance claim. The 30-core M3 Max, 16-core M4 Pro and
32-core M5 Max have not been GPU-tested by this reorganization, nor the M4 Max on its own hardware.

## Source and compilation ownership

```
monolith/backends/metal/
  config.py, registry.py, context.py, calibration.py
  base.py                 # shared extension interface
  m3_pro/backend.py
  m3_max_30c/backend.py
  m3_ultra_60c/backend.py
  m4_pro/backend.py        # chip-owned Python hooks
  m5_pro/backend.py
  m5_max_32c/backend.py
  m5_max_40c/
    backend.py, scheduling.py, validation.py
    config.json
    recipes/              # selected context and shape configurations
kernels/
  common/                 # shared Metal implementations
  m3_pro/
  m3_max_30c/
  m3_ultra_60c/
  m4_pro/                 # same-name source overrides
  m5_pro/
  m5_max_32c/
  m5_max_40c/
```

`Session` selects by chip name, GPU family **and** core count. A mismatched
explicit chip configuration is rejected. `compile_program` and `emit_program`
enter a compilation-scoped backend context and record the backend/configuration
identity in the resulting Program. Nested compilations and threads restore their
own selection. Kernel templates resolve in the selected chip directory first,
then `kernels/common`. The reorganization moves shared templates without changing
their contents; chip directories initially reuse those implementations.

To customize a chip, override `Backend.handler` for individual operations,
`finalize` for fusion/scheduling after emission, `direct_attention_shape` for
additional explicitly requested direct-cache shapes, `optimize_decoder` or
`optimize_draft` for explicit recipes, `optimize_prefill` for prompt kernels, or `emit`/`compile` to replace the full
lowering strategy. A `.metal` file with the same relative name overrides only
that chip's source. Shared compiler fusion helpers remain reusable. Direct
source-building experiments can use `with using_backend("m5_max_32c"):`.

`Backend.reencode_default` selects the default command encoding for a Program.
It is false for shared/other chip backends (ICB replay), and true for M2 Max
30-core (direct serial encoding while its ICB workloads remain unqualified).
`Engine.run(reencode=True/False)` explicitly overrides the default for diagnostics.
The backend is taken from `Program.backend_id`, including after JSON round-trip.

The 40-core backend's `scheduling.py` owns automatic GDN mixer fusion, the
measured routed-expert crews in `routed.py`, and the two-kernel INT4 MLP
layout in `mlp.py`. The routed policies apply only to NVFP4 top-8 experts:
2048 hidden / 768 intermediate dimensions at static T=1 or T=8, and
2048 hidden / 512 intermediate dimensions at T=8 including speculative rounds;
the latter also fuses expert down projection, weighted reduction, shared-expert
addition and residual in threadgroup-local tasks. Gate/up remains separate.
The MLP policy applies only to static T=8 affine INT4 with 1024 hidden / 3584
intermediate dimensions and BF16 activations/scales. Other shapes and prefill
retain their existing geometry. The GDN recipe requires static T=8,
matching shapes/formats and a complete
normalization boundary. Dynamic/speculative mixer fusion requires a decoder
recipe. Attention, MLP and DSpark context
maps live in [the 40-core recipes directory](m5_max_40c/recipes/); they do not
enable context routing by themselves. The serving setup selects request recipes. See the
[mixer design](../../../docs/design/mixers.md).
The additional direct-cache attention shapes live in `m5_max_40c/attention.py`.
The 40-core `prefill.py` owns large-prompt D=256 attention preparation and
locally reduced 2048-key partitions, plus measured NVFP4/FP8 projection tiles.
The 512-row Qwen path uses contiguous NVFP4 and FP8 operands, shared GDN Q/K
preparation, and masked matrix tails. FP8 stays eight-bit; only native BF16
projections use direct BF16 reads. The compact layouts preserve checkpoint
quantization and leave the eight-row decode recipe independent of prefill tuning.
It runs before the shared scratch-lifetime pass. Prompt specialization skips
intermediate sampling and removes redundant GDN commit replays; persistent
state, input buffers, logits and acceptance logs never alias scratch storage.
`Session(prefill_optimizations=False)` retains the original compiler path for
comparisons. Other chips retain their attention/projection geometry.
The [model adapter design](../../../docs/design/models.md) describes the hybrid MoE and
Llama contracts. Shape-specific task boundaries and configuration choices remain owned by the backend.

Autotuning cache names include backend, core count and a digest of configuration,
resolved Metal sources and backend Python code. Existing caches are left intact;
they are not reused across variants. Native pipeline caches also key on source
and specialization macros.

## Calibration and migration

There is no top-level `profiles/` directory. Its chip files moved beside their
backends, and `profiles/recipes/m5max-27b` moved to `m5_max_40c/recipes`. Use
`config_path("apple-m5-max-40c")`, `load_configs()` or `config_for_device(...)`
from `monolith.backends.metal` instead of constructing a profile path.
`monolith.core.profile` and `monolith.core.profile_writer` retain import aliases
for callers; `profiles_dir()` now returns this backend root and is not a flat
directory of configuration files.

```
python tools/profile_writer.py --dry-run  # measure and print
python tools/profile_writer.py            # refresh this device's registered config
```

The writer preserves existing probe records, backend metadata and layer-fusion
recipes. It saves the prior engine block and measurements under `writer`.
Unregistered devices require `--out` (or `--dry-run`); measurements do not silently
create a chip backend. Leaf calibration does not mark an untested layer recipe
as validated. Store measurements and generated figures with the
[benchmark outputs](../../../tools/bench/README.md), outside the design documentation.
