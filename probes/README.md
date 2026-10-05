# GPU feasibility probes

Small, bounded Metal programs that measure the Apple-GPU behaviours the design depends on and Apple does not
document. Results and their design consequences: [`docs/research/apple-gpu-probes.md`](../docs/research/apple-gpu-probes.md).

```bash
./run_all.sh                 # build + run everything here (~5 min); output is saved under results/
./run_all.sh p10_claim_protocol p11_interop_overlap   # or a subset
./remote_run.sh user@host    # run on another bare-metal Apple-silicon Mac over SSH and fetch its results file
```

Every probe derives its geometry from the GPU core count (`gpu-core-count` in the IORegistry; override with
`GPU_CORES=<n>`), so results are comparable across a 10-core M4, an 18-core M3 Pro and a 40-core M5 Max. Do not use
virtualized macOS (hosted CI runners): its paravirtual GPU does not schedule like the real one.

| Probe | Question |
|---|---|
| `p1_limits` | SIMD width, threadgroup limits, GPU family, MSL versions |
| `p2_sync` | can units of one dispatch hand off through device atomics? (same SIMD-group / other SIMD-group / other threadgroup) |
| `p3_residency` | how many SIMD-groups are in flight at once; where does physical concurrency saturate |
| `p4_core_model` | threadgroup → core mapping; full-speed SIMD-groups per core |
| `p5_bandwidth` | streaming GB/s vs number of SIMD-groups; a naive NVFP4 decode+dot |
| `p5b_access_pattern` | striped vs interleaved vs block-cooperative lane access |
| `p6_preemption` | does a long dispatch block other GPU work? (runs three ~1.5 s dispatches) |
| `p6b_interleave` | at what granularity is the GPU shared with other clients: dispatch or command buffer? |
| `p7_dispatch_overhead` | CPU encode, GPU per-dispatch overhead, ICB replay, sync round trip |
| `p8_threadgroup_mem` | is threadgroup memory faster than device memory? |
| `p9_clock_warp` | a free-running SIMD-group as an in-kernel clock |
| `p11_interop_overlap` | cores needed to saturate the memory bus; do un-barriered dispatches overlap; does an ALU-bound op hide inside a bus-bound one; can two bus-bound ops overlap usefully (MPK V2-style inter-op pipelining) |
| `p10_claim_protocol` | in-kernel scheduler (own-slice + steal, fenced and unfenced) vs one dispatch per op: exactly-once, cost per barrier, robustness to missing/surplus threadgroups |
| `p12_stream_geometry` | streaming bandwidth vs lane access pattern × load width × loads in flight per lane × SIMD-groups per core × block size; cores needed to saturate the bus with the best one-threadgroup-per-core kernel; ALU-bound op hidden inside a *saturating* bus-bound op (both encode orders); two saturating bus-bound ops overlapped. Added on the M5 Pro, where the p5b crew-geometry sweep reached only 71 % of nominal |
| `p13_decode_gemv` | real FP8-E4M3 and NVFP4 decode GEMV/GEMM (17408 × 5120, 16-byte loads, R-row blocks, T ∈ {1, 2, 4, 8}) in both block-lane-major lane orders, at the crew geometry and at conventional occupancy, checked against a CPU reference — the first half of plan M1. `./build/p13_decode_gemv check` compiles every variant without running |
| `p14_tensor_ops` | M5 neural accelerators: `mpp::tensor_ops::matmul2d` from FP8/NVFP4 tiles dequantized into threadgroup memory (and from half weights directly), TM ∈ {8…64}, checked against a CPU reference; needs only the Command Line Tools. `./build/p14_tensor_ops check` compiles every variant |
| `p15_frame_pacing` | on-screen frame pacing: a window presents one frame per vsync while compute command buffers of 8 / 16 / 33 / 66 / 133 ms (ALU-bound, then bus-bound) run back to back with 3 in flight; presented-time intervals, late frames, fps per buffer length. Needs the screen on and unlocked (it skips otherwise); opens a small floating window for ~70 s |

Safety: an Apple9 GPU does not preempt a running dispatch and an Apple10 GPU does so only sometimes (`p6`), so every
loop here is bounded and the longest single dispatch is ~1.5 s. Expect brief display stalls during `p3`/`p6`/`p6b`. Never add an unbounded spin to a probe.

To characterize a new chip: run everything, commit the results file, add its numbers to the report, and derive a
profile (`cores`, full-speed SIMD-groups per core, in-flight limit, best block size, GB/s, cores needed to saturate the
bus; a hand-derived first cut per chip is in `../monolith/backends/metal/`). `results/` holds the complete M3 Pro reference run and
the M5 Pro runs (11 files: the full suite, repeats of `p6`/`p6b`/`p12`, and `p13`/`p14`).
