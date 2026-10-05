# M5 Pro fixed-token layer performance — 2026-09-28

[M] Implementation `40a78fd`, Apple M5 Pro (20 GPU cores, 24 GB), macOS 26.5.1,
MLX 0.32.2 / mlx-lm 0.31.3. Verification lengths T=1/4/6/8 and contexts
128/1024 use identical checkpoint weights, BF16 interfaces and seeded inputs.
Speculative acceptance throughput is excluded.

## Results and limits

Seven alternating AB/BA pairs of 64 steps cover every checkpoint layer:

| Format | Individual minimum wins | Every-pair wins | Worst Monolith/MLX minimum ratio |
|---|---:|---:|---:|
| BF16, 24 layers | 192/192 | 192/192 | 0.909986 |
| NVFP4, 36 layers | 288/288 | 287/288 | 0.956222 |
| INT4, 28 layers | 224/224 | 221/224 | 0.977543 |

All **704 individual minima** beat MLX. Companion streaming stacks pass **24/24
minimum-latency and numerical gates**. Isolated replay has different cache
residency and host-overhead amortization; retain both measurements. NVFP4 T=1
streaming wins are narrow (390.23/390.81 µs at context 128, 409.02/409.55 at 1024),
and timing ranges overlap. These are minimum-of-paired-runs results on this
machine, not an every-run or universal hardware claim.

The [final summary](../../tools/bench/results/apple-m5-pro-20c_final_layer_summary_20260928.json)
links all six raw sweeps, including every sample and host stall, at an immutable
evidence commit. The numerical threshold remains 0.999. Direct individual MLX
checks pass 703/704: INT4 layer 24, T=1/context 1024 records cosine 0.998958383
([#124](https://github.com/jiazhihao/mpk-apple/issues/124)). The
[independent CPU audit](https://github.com/jiazhihao/mpk-apple/blob/5c11bb494530f9ad31ccf39be55771a69c40a018/tools/bench/results/apple-m5-pro-20c_int4_layer24_contract_audit_20260928.jsonl)
gives Monolith 0.999966/0.999978 against BF16/FP32-dequantized references, versus
MLX 0.998928/0.998935. The benchmark still exits nonzero for that discrepancy.

## Changes that meet the gate

- **Projection work:** use packed BF16 operands, two-row SIMD projections and a
  small T=4 BF16 kernel; choose measured K-splits at T=6/8. Compact matrix
  partials skip padded token rows. Fixed T=1 NVFP4 uses one-row crews with a
  matching statistic layout; dynamic programs retain their two-row layout.
- **Compile-time constants:** specialize immutable parameter geometry and fixed
  projection lengths. Uniform row scales are constant only after inspecting
  every finite FP32 bit pattern; varying/nonfinite tables retain device loads.
  Position and dynamic active lengths remain runtime data.
- **Weight scales and normalization:** share duplicated narrow INT4 scale runs;
  place eligible NVFP4 scales in physical payload order. Fixed NVFP4 T=1/4
  normalization spreads work over 64/128 SIMD-groups per row with shorter gathers.
  Producer/consumer permutation layout and scratch identity remain unchanged.
- **Mixers:** fuse BF16 projection/convolution at T=1, retaining raw convolution
  history; share GDN preparation, avoid repeated recurrence and fuse output
  normalization. Attention shares queries and paired-key work; adaptive matrix
  attention uses eight query rows and 32/64-key chunks at D=128, replication=2,
  T=4. Other shapes retain their existing paths.

NVFP4 measurements require a normal v3 pack:

```sh
python tools/pack_weights.py --model <checkpoint> --out <pack> \
  --scale-placement block --scale-order payload
```

Codes, scale values and dequantized weights are unchanged. Existing manifests
remain readable; ineligible slabs retain lane order. CPU unpacking, row/matrix
kernels, legacy GEMV, embeddings and autotuning understand the recorded layout.
See [benchmark instructions](../../tools/bench/README.md) to reproduce the gate.

## Native evidence

Public `MTLBinaryArchive` exports exact native code with Command Line Tools.
Available decoders do not reliably decode this M5/compiler combination, so no
instruction counts or decoded assembly are claimed. Raw metadata register fields
are experimental compiler allocations, not occupancy measurements. Their
[interpretation](https://github.com/niklassheth/agx-re/blob/e25433b1d348a809ab86484ce149ff7d459e8cdf/experiments/EXP-M5-21-gpr-machine-model/report.md)
was checked locally with dependent FMA chains; the full calibration is archived.

| Selected change | Native code bytes before → after | Register field before → after |
|---|---:|---:|
| NVFP4 T=1 normalization | 2,500 → 1,558 | 35 → 35 |
| NVFP4 input projection, payload scales | 4,166 → 3,934 | 66 → 61 |
| NVFP4 gate/up, payload scales | 4,778 → 4,540 | 66 → 61 |
| BF16 gate/up, constant scales | 2,778 → 2,714 | 41 → 39 |

Normalization shared memory falls 4→0 bytes and its active row spans 64
threadgroups instead of four. All these specializations report zero local
scratch. A rejected FP4 sign-bit rewrite instead grows to 5,834 bytes and 72
registers and loses about 38 µs/layer. Resource counts alone do not establish
causality; selection used paired layer timings. The
[raw native reports](https://github.com/jiazhihao/mpk-apple/blob/5c11bb494530f9ad31ccf39be55771a69c40a018/tools/bench/results/apple-m5-pro-20c_native_normalization_20260928.jsonl)
preserve the observations.

## Validation and research archive

Metal validation passed all six selected real-model checks (legacy and v3
NVFP4, layer oracles, greedy goldens, BF16 sampling and rejected-step rollback),
227 contract/alternate-tile checks, 180 normalization/scale/barrier checks and
63 payload-layout GPU checks in their respective suites. The existing
[#123](https://github.com/jiazhihao/mpk-apple/issues/123) long-prompt exclusion
is unchanged. No numerical threshold or safe-math requirement was relaxed.

The [full research journal and reproduction commands](https://github.com/jiazhihao/mpk-apple/blob/5c11bb494530f9ad31ccf39be55771a69c40a018/docs/research/m5-native-code.md)
retain rejected experiments, archive-inspection utilities and the CPU audit at
commit `5c11bb494530f9ad31ccf39be55771a69c40a018`. They are omitted from the final
PR diff to keep review focused on the implementation. Raw files can also be read
locally with `git show <evidence-commit>:tools/bench/results/<filename>`.
