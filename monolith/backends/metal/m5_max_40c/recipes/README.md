# Selected M5 Max 40-core kernel recipes

These are executable configuration inputs for the eight-row 27B decoder and
DSpark draft benchmarks. Measurements, sweep candidates, logs and figures are
kept in the [evidence archive](../../../../../docs/research/m5max-artifacts.md).

- `attention-optimization/selected-contexts.json` maps 128, 4K, 8K, 16K and 32K
  to the five selected attention mixer recipes.
- `mlp-optimization/selected-contexts.json` references the attention map,
  the backend's GDN configuration, and the native and fused MLP alternatives. The native
  MLP uses two kernels and is the selected target configuration.
- `dspark/selected-contexts.json` contains the selected BF16 draft and target
  recipes for all five contexts.
- `dspark/selected-nvfp4-endpoints.json` contains experimental NVFP4 draft
  recipes validated at 128 and 32K only.

Pass the DSpark maps to `tools/bench/dspark_round_latency.py --config` with a
matching `--config-key` and `--contexts` value. These maps do not add automatic
context routing to generation. The GDN default remains in
[`config.json`](../config.json).
