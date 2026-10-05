# Metal chip backends

Each backend owns its configuration, lowering hooks, fusion policy and optional
Metal source overrides. Model graphs and the runtime Program ABI stay shared.
M5 Max 32-core and 40-core are independent backends; neither inherits the other's
configuration or autotuning cache.

| Backend | Configuration | Status |
| --- | --- | --- |
| `m3_pro` | [18 cores](m3_pro/config.json) | Existing probe-derived settings preserved |
| `m4_pro` | [16 cores](m4_pro/config-16c.json), [20 cores](m4_pro/config-20c.json) | Unmeasured native fallback |
| `m5_pro` | [20 cores](m5_pro/config.json) | Existing measured settings preserved |
| `m5_max_32c` | [32 cores](m5_max_32c/config.json) | Unmeasured native fallback |
| `m5_max_40c` | [40 cores](m5_max_40c/config.json) | Measured GDN default and explicit attention/MLP/draft recipes preserved |

Unmeasured configurations have no cost tables or automatic megakernel fusion,
and keep the tensor accelerator off. They provide a starting point for validation
on those devices, not a performance claim. The M4 Pro and 32-core M5 Max have not
been GPU-tested by this reorganization.

## Source and compilation ownership

```
monolith/backends/metal/
  config.py, registry.py, context.py, calibration.py
  base.py                 # shared extension interface
  m4_pro/backend.py        # chip-owned Python hooks
  m5_pro/backend.py
  m5_max_32c/backend.py
  m5_max_40c/
    backend.py, scheduling.py, validation.py
    config.json
    recipes/              # selected context and shape configurations
kernels/
  common/                 # shared Metal implementations
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
enable automatic context routing. See the
[optimization study](../../../docs/research/m5max-gdn-mixer-optimization.md).
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
The [hybrid MoE study](../../../docs/qwen-hybrid-moe.md#eight-row-moe-task-tuning)
records eight-row expert task tuning, experimental megakernel boundaries and
the seven-proposal DSpark serving default.
The [Qwen/Llama audit](../../../docs/research/m5max-qwen-llama-audit.md) records
their context choices and the remaining numerical gates, including the excluded
INT4 hybrid direct-cache choice at 32K.

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
as validated. Measurements and figures remain in the
[evidence archive](../../../docs/research/m5max-artifacts.md).
