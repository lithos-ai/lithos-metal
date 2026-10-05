# Apple GPU execution model — measured facts

Apple documents almost none of what an in-GPU runtime needs to know: how threadgroups map to cores, how many
SIMD-groups run at once, what a dispatch boundary costs, whether the GPU is shared while a dispatch runs, how fast
memory really streams. This report records what we measured, per chip, and what each number means for the
[design](../design/design.md).

| Chip | Status | Results files |
|---|---|---|
| **M3 Pro**, 18-core GPU, 36 GB, Apple9, macOS 26.6.2 | measured 2026-09-19 (on battery, Low Power Mode off); the 13 original probes | [`probes/results/Apple-M3-Pro_18c_…txt`](../../probes/results) |
| **M5 Pro**, 20-core GPU, 24 GB, Apple10, macOS 26.5.1 | measured 2026-09-22 (AC power, display attached); the 13 probes plus three new ones (`p12`–`p14`), with `p6`/`p6b` run 4× and `p12` 3× — §3 | [`probes/results/Apple-M5-Pro_20c_…txt`](../../probes/results) (11 files) |
| **M4 family** | **not measured — next** (§4) | — |

Every number here is an observation on one chip and one OS build — *firmware behaviour, not an API contract*. The design
treats them as per-chip profile values (a first, hand-derived cut of those profiles is in [`chip configurations`](../../monolith/backends/metal)),
and correctness never depends on them.

**Running the suite** (16 probes, ~5 min, Xcode Command Line Tools only — shaders compile at runtime, including the
Metal Performance Primitives tensor ops used by `p14`):

```bash
./probes/run_all.sh                          # on the machine itself; output is saved to probes/results/<chip>_<cores>c_…txt
./probes/remote_run.sh user@host             # on another bare-metal Mac over SSH; fetches the results file
./probes/run_all.sh p10_claim_protocol p11_interop_overlap     # a subset
./probes/build/p13_decode_gemv check         # p13 / p14: compile every kernel variant without running anything
```

Every probe derives its launch geometry from the GPU core count (`gpu-core-count` in the IORegistry; override with
`GPU_CORES=<n>`), so results are comparable across chips. Do not use virtualized macOS (hosted CI runners, Tart/Anka
VMs): a paravirtual GPU does not schedule like the real one. Apple9 GPUs do not preempt a running dispatch and Apple10
GPUs do so only sometimes (P6), so every probe is bounded and the longest single dispatch is ~1.5 s — expect brief
display stalls during `p3`, `p6`, `p6b`. Never add an unbounded spin.

---

## 1. Cross-chip results

Ranges are the spread over repeated runs on the same day. Fill a column per chip; the probe/section that prints each
number is in the first column. The M5 Pro column is read against a nominal 307 GB/s **[S]**.

| Metric (probe) | M3 Pro 18c | M5 Pro 20c | M4 … |
|---|---|---|---|
| GPU cores · nominal GB/s · **GB/s per core** | 18 · 153.6 · **8.5** | 20 · 307 · **15.4** | |
| SIMD width · max threads/threadgroup · threadgroup memory (`p1`) | 32 · 1024 · 32 KB | 32 · 1024 · 32 KB | |
| GPU family · highest MSL that compiles (`p1`) | Apple9 · 4.0 | Apple10 · 4.0 (4.1 needs macOS 27; this machine runs 26.5.1) | |
| Same-SIMD-group handoff deadlocks? · cross-SIMD-group handoff (`p2`) | yes · 195–230 ns | yes · 142–196 ns | |
| SIMD-groups in flight at once (`p3a`) | 1,536 ok, 2,048 not | 1,536 ok (marginal as 48 × 1024 threads: 38 k spins), 2,048 not | |
| Full-speed SIMD-groups per core · threadgroups per core (`p4`) | ~12 (384 threads) · 1 — flat to G = 18, clean 2× at G = 20 | ~12 (k ≤ 14 in one threadgroup) · 1 — flat to G = 20, 2× at G = 22; one SIMD-group's ALU rate is 1.12× the M3 Pro's | |
| Saturated ALU throughput, in full-speed-SIMD-group equivalents (`p3b`) | 205–215 (≈ 11.5–12 per core) | 214–269 (≈ 11–13.5 per core) | |
| Streaming, crew geometry, 64 KB block sweep, **lane-contiguous** sub-ranges, 4 B loads (`p5b`) | **124–134 GB/s (81–87 % of nominal)** | 219 GB/s (**71 %**) — H3 fails as written; see the `p12` row | |
| … lanes on far-apart stripes · lanes interleaved per word (`p5b`) | 59–60 · 84–90 GB/s | 79–80 · 116 GB/s (288 with 16× more SIMD-groups) | |
| … best conventional geometry, 192·C SIMD-groups (`p5`, `p5b`) | 106–126 GB/s | 185–189 (`p5`) / 288 (`p5b`, interleaved) GB/s | |
| **Best raw streaming pattern at the crew geometry (`p12`)** | not run | **blocked, lanes interleaved at 16 B, 4 loads in flight per lane: 286–295 GB/s (93–96 %)**; flat from 12 to 384 SIMD-groups per core and from 16 KB to 1 MB blocks. Lane-contiguous 16 B loads cap at 200–218 GB/s whatever the unrolling, occupancy or block size | |
| Naive NVFP4-style decode + dot (`p5`, fmt 1) | 49–58 GB/s | 68–71 at the crew geometry; 141–157 with 16–64× more SIMD-groups | |
| **Real FP8 decode GEMV**, 17408 × 5120, T = 1 (`p13`) | not run | crew geometry, lane-interleaved pack, R = 16: **275 GB/s (90 %)**; lane-contiguous pack: 226–238; conventional geometry (one block per SIMD-group, 64-thread threadgroups): 277 — **parity** | |
| **Real NVFP4 decode GEMV**, T = 1 (`p13`) | not run | crew geometry: 130–137 GB/s of useful bytes (42–45 %) — **ALU-bound at ~275 G weights/s, the same weights per second as FP8**; 108 SIMD-groups per core: 176–182 (59 %) | |
| Decode-GEMV pass cost vs T, best geometry, relative to T = 1 (`p13`) | not run | FP8: T = 2 ×1.08 · 4 ×1.11 · 8 ×3.6. NVFP4: T = 2 ×1.28 · 4 ×1.79 · 8 ×5.3 | |
| **Neural accelerator (`matmul2d`) from dequantized threadgroup tiles**, TM = 8 (`p14`) | n/a (Apple9 runs MPP on the shader ALUs) | FP8: 169–186 GB/s = 1.5× a T = 1 pass for 8 tokens; NVFP4: 113–119 GB/s useful = 1.5–1.6×; TM = 32: 1.7–1.8×; half weights read directly: 199–210 GB/s; 15.5–16.9 TFLOP/s at TM = 64; accumulation error 1.3–1.6e-6 (FP32 FMA path: 9e-8) | |
| Small dispatch behind ONE long dispatch (`p6`, 4 runs × 3 core counts) | waits 1.24–1.42 s, even with 1/3 of the cores idle | **bimodal**: in 2 of 12 trials every foreign dispatch got in at 0.2–0.4 ms; in 5 the first one waited the whole 1.10–1.11 s; in 5 a later one waited 0.49–1.04 s | |
| Small dispatch behind many short dispatches (`p6b`, 4 runs) | typical 0.3–0.5 ms; **worst case one whole command buffer** (130–140 ms behind ~155 ms buffers; once 1.13 s behind a 1.25 s buffer) | behind an 800-dispatch, 1.15 s buffer: **the whole buffer (1.04–1.05 s) in 3 of 4 runs** (8 quick ones, then the rest, in the 4th); behind ~145 ms buffers: max 44–129 ms; behind ~14.5 ms buffers: max 0.3–1.3 ms | |
| CPU encode · GPU cost per dispatch · ICB replay per 1,240 dispatches · sync round trip (`p7`) | 0.13 µs · 1.3–1.8 µs · 0.01–0.02 ms · 0.12–0.13 ms | 0.18 µs · 1.3–1.6 µs · 0.01 ms · 0.15 ms | |
| Hot read: threadgroup memory vs device memory (`p8`) | 61–64 ns vs 62–64 ns (thread-private array: 116 ns) | 36 vs 37 ns (thread-private: 43) | |
| Clock SIMD-group tick (`p9`) | 47–53 ns, ±1 % within a session, durations linear within 1 % | 38.3–38.6 ns, linear within 1.5 % | |
| Barrier cost per op: **dispatch boundary** vs in-kernel static slices vs in-kernel stealing (`p10`, protocol only) | **1.6–2.7 µs** vs 2.1–2.6 µs (3.1–3.9 µs spec-compliant) vs 4.7–6.1 µs | **1.38–1.40 µs** vs 1.97–2.47 µs vs 4.25–4.78 µs | |
| … same with ~30 µs blocks (`p10`) | 92–96.5 µs/op for every variant | 81.5 (dispatch per op) vs 81.7–84.7 (static) vs 83.5–88.1 (stealing) µs/op | |
| Exactly-once under missing / surplus threadgroups (`p10`) | yes, 0 timeouts | yes, 0 timeouts | |
| One core's uncontended streaming rate (`p11` §1, `p12` §4) | 18.0–18.7 GB/s | 29.2–29.5 GB/s with the lane-contiguous sweep; **71.4 GB/s with the lane-interleaved 16 B sweep** | |
| Cores needed to saturate the bus (`p11` §1, `p12` §4) | 12 of 18 (9 cores already give 114–126 GB/s) | lane-contiguous sweep: 13 of 20, but its ceiling is 215–217; lane-interleaved sweep: **6 of 20 reach 278 GB/s**, 10 reach 286, 13–20 reach 289–294 | |
| ALU-bound op hidden inside a bus-bound op, no barrier, both at full geometry (`p11` §3b, `p12` §5) | 86–114 % hidden — i.e. fully hidden within noise — in either encode order, up to 87 % of the partner's length | with the lane-contiguous streamer (215 GB/s): 106–108 % up to 63 % of the partner, 73 % at 125 %. With the **saturating** streamer (3 runs): **ALU op encoded first → 102–126 % hidden up to 44 % of the partner, 84–93 % at 85 %**; bus op encoded first → only a ≤ 22 % op hides (100–122 %), larger ones 1–29 % | |
| … same pair with cores split by hand (`p11` §3) | 29–32 ms vs 27 ms serial — *slower* | 21.2–23.3 ms vs 17.8 ms serial — *slower* | |
| Two bus-bound ops overlapped (`p11` §4, `p12` §6) | no consistent gain (−3 %…+10 %, within noise) | none: 216 → 212–217 GB/s (`p11`); 292–296 → 295–297 (`p12`) | |
| A last-level cache the GPU re-reads from (a streaming pass over S MB repeated back to back, 2026-09-27, `slc_probe`) | not measured | **4–16 MB re-read at 400–540 GB/s** (8 MB: 29.5 → 21.1 µs, 16 MB: 59.9 → 30.9), 32 MB at the bus (120 → 107), 64 MB no gain (235 → 231): a ~16–24 MB cache. A prefetch of the next GEMV's 9.4 MB slab beside the latency-bound attention (8 µs) cost +27 µs per layer at full crew and more at smaller crews — the slab does not fit the attention's window, and the barrier before the GEMV waits for the prefetch — so D14 stands **[M]** | |

---

## 2. What the M3 Pro numbers mean

| # | Question | Result | Design consequence |
|---|---|---|---|
| P1 | Device limits | SIMD-group = 32 threads; ≤ 1024 threads/threadgroup; threadgroup memory ≤ 32 KB; tier-2 argument buffers; function pointers and dynamic libraries; MSL 4.0 compiles, 4.1 needs macOS 27 | The SIMD-group is the 32-lane lockstep unit — the analogue of an NVIDIA *warp* |
| P2 | Can execution units of **one dispatch** synchronize through device atomics? | Same SIMD-group: **deadlock** (lockstep). Different SIMD-groups, same or different threadgroups: **works, ~200 ns per handoff**, ~2 spins per wait | In-kernel events are feasible; the unit of independent control flow is the SIMD-group |
| P3a | How many SIMD-groups of one dispatch are in flight at once? | 1,536 all see each other within 9 spins; at 2,048 most time out | Anything that waits in-kernel must stay well under ~1,500 SIMD-groups; the crew geometry uses 12 per core |
| P3b/P4 | Physical concurrency and threadgroup → core mapping | Exactly **18 slots = 18 cores**. With threadgroups ≥ 384 threads **one threadgroup runs per core**; a core runs **~12 SIMD-groups at full speed** (16 → 1.5× slower, 32 → 2.6–2.8×). 18 × 384 threads all run at full speed; the 19th+ threadgroup waits for a core | **GPU core ≈ MPK "SM"; threadgroup of 384 = 12 × 32 ≈ MPK "CTA"**. Launch geometry = `cores × 384` (design D4) |
| P5 | Streaming bandwidth | Conventional geometry peaks at 106–126 GB/s; a naive NVFP4 decode+dot kernel is **ALU/load-latency-bound at 49–58 GB/s** | The decode GEMV must be engineered (wide packed loads, activation reuse); it is not automatically bandwidth-bound |
| P5b | Does the lanes' access pattern matter? | Crew geometry: far-apart stripes 59–60 GB/s; interleaved per word 84–90; **64 KB blocks with each lane on a contiguous 2 KB sub-range: 124–134 GB/s** | Block-lane-major weight packs (D8). Advantage over the best conventional geometry: **+5 % to +17 %** depending on the run — to be confirmed with real kernels (plan M1) |
| P6 | Is a running dispatch preempted or shared? | **No.** A 30 µs dispatch on a second queue waited 1.24–1.42 s for one long dispatch, even when it used 12 of 18 cores | A never-returning kernel is not viable; long dispatches freeze every other GPU client |
| P6b | At what granularity *is* the GPU shared? | **Usually per dispatch** (0.3–0.5 ms behind 800 short dispatches in one command buffer, 5 of 6 trials), but **the worst case is a whole command buffer**: 130–140 ms outliers behind ~155 ms buffers, and once the full 1.13 s | Short dispatches **and** short command buffers (design D6): ≤ ~16–33 ms of work per command buffer while a display is attached |
| P6' | Watchdogs | An 11 s dispatch whose threads retired continuously completed without error; others report a ~5 s kill of a non-progressing kernel (MLX #4475) and an *interactivity* kill at ~0.5–1.2 s with the display on (MLX #3267) | Rely on none of it: dispatches are sub-millisecond, command buffers tens of milliseconds |
| P7 | Launch overhead for a 1,240-dispatch "token" | CPU encode 0.13 µs/dispatch; GPU 1.3–1.8 µs/dispatch; ICB replay 0.01–0.02 ms per token; one CPU↔GPU sync 0.12–0.13 ms | ~2 ms/token ≈ 1–2 % of a 27B token here: launch overhead alone does not justify a single-kernel design (design §2) |
| P8 | Is threadgroup memory a fast scratchpad (an SMEM analogue)? | **No**: 61–64 ns per hot read from either; a thread-private array is slower | No software-managed memory level ⇒ no page planner, no loader/storer roles, nothing to pre-stage into |
| P9 | In-kernel clock (MSL has none)? | A dedicated **clock SIMD-group** incrementing an atomic: 47–53 ns/tick, ±1 %, durations linear within 1 % | MPK-style tracing is recoverable at the cost of 1 SIMD-group, in profiling builds |
| P10 | Whole step inside **one** dispatch with in-kernel barriers — does it beat one dispatch per op? | **It works, but it does not win.** Own-slice + steal is exactly-once with zero timeouts even with half the crew missing or 2×/4× surplus threadgroups. Per barrier: dispatch boundary **1.6–2.7 µs** (including the launch of 6,912 threads) vs in-kernel 2.1–2.6 µs with static slices (**3.1–3.9 µs** in the spec-compliant form) and 4.7–6.1 µs with stealing; a global claim cursor is 27 % slower. With 30 µs blocks all variants are within noise | **One dispatch per fused op** (D5). In-kernel sync is never meaningfully cheaper, the spec-compliant form is always dearer, and a multi-op kernel is a long non-preemptible dispatch. Co-residency is a performance property, not a correctness requirement |
| P11 | Can ops overlap (MPK V2-style inter-op pipelining)? | The bus saturates with **half to two-thirds of the cores**. Un-barriered dispatches do overlap: an **ALU-bound op hides (86–114 %, i.e. fully within noise) inside a bus-bound one**; **two bus-bound ops gain nothing**; splitting cores by hand is slower than serial | No weight pre-staging across dependencies; overlap only ALU-bound with bus-bound siblings at full geometry, placement left to the firmware (D14, design §5.12) |

---

## 3. M5 Pro (Apple10): what held, what changed, what is new

Measured 2026-09-22 on an M5 Pro with a 20-core GPU and 24 GB (a configuration that **cannot host the 27B target**:
`recommendedMaxWorkingSetSize` is 19.07 GB and `maxBufferLength` 14.30 GB — it characterizes the GPU and runs kernel
work, nothing more). Verdicts on the hypotheses that §4 (formerly §3) asked the next chip to test:

| Hypothesis | Verdict | Evidence | Consequence |
|---|---|---|---|
| H1 Same SIMD width, limits, lockstep deadlock | **holds** | `p1`, `p2` | none |
| H2 One 384-thread threadgroup per core, ~12 full-speed SIMD-groups | **holds** | `p4`: flat to G = 20, 2× at G = 22; k ≤ 14 SIMD-groups at full speed inside one threadgroup | D4's geometry is right; see N2 for the occupancy caveat |
| H3 Crew-geometry block sweep ≥ 80 % of nominal and ≥ the conventional geometry | **fails as written, holds after one layout change** | lane-contiguous sub-ranges: 219 GB/s (71 %); the same geometry with lanes interleaved at 16 B: 286–295 (93–96 %); a real FP8 kernel: 275 vs 277 for the conventional geometry | D8's intra-block lane order becomes a profile value (N1); the crew-geometry claim shrinks from "+5–17 %" to **parity** for T = 1 |
| H4 A dispatch boundary is no dearer than an in-kernel barrier | **holds, by a wider margin** | 1.38–1.40 µs vs 1.97–2.47 (static) / 4.25–4.78 (stealing); with 30 µs blocks 81.5 vs 81.7–88.1 µs/op | D5 stands; plan M7's M5 re-evaluation is done |
| H5 Sharing: per dispatch, worst case per command buffer, never inside a dispatch | **changed in both directions** | foreign dispatches *sometimes* get in during a dispatch (2 of 12 trials always, 5 of 12 never, 5 of 12 partly); behind a long command buffer they waited for the whole buffer in 3 of 4 runs (M3 Pro: 1 of 6) | in-dispatch preemption exists but cannot be relied on; command-buffer length is the control on both chips and matters *more* here (D6 tightens). **The on-screen check (`p15`, 2026-09-25) then found the display path unaffected**: a window's frames kept arriving every vsync (120 Hz, 0 % late) while ALU-bound compute buffers of 8–133 ms ran back to back with 3 in flight — the compositor is not the foreign work `p6b` measured, so `max_cb_ms` is a latency choice (the pump's over-run), not a pacing constraint |
| H6 The bus needs most of the cores (little spare ALU) | **fails** | 6 of 20 cores stream 278 GB/s; one core streams 71 GB/s, 3.9× an M3 Pro core | the design's "little or none" projection assumed Apple10 cores stream like Apple9 cores; they do not. D14's overlap is worth *more* on the M5 Pro, not less |
| H7 An ALU-bound op is no longer fully hidden | **fails (still hidden) — with a new condition** | hidden 102–126 % when the ALU op is encoded first, 1–29 % when the bus op is first (3 runs) | a per-chip encode-order rule (N3); on the M3 Pro order was irrelevant |
| H8 Threadgroup memory is not faster than device memory | **holds** | 36 vs 37 ns per hot read, both ~40 % faster than on the M3 Pro | D3 stands |

What the three new probes add:

* **N1 — Lane order inside a block decides bandwidth on Apple10 (`p12`).** Whether the SIMD-group sweeps 16 KB or
  1 MB blocks, whether loads are 4 or 16 B wide, whether 1 or 4 loads are in flight per lane, whether 12 or 384
  SIMD-groups sit on a core — none of it moves the lane-contiguous pattern above 200–218 GB/s. Interleaving the lanes
  at 16 B granularity (one load instruction covers a contiguous 512 B span) reaches 286–295 GB/s at the crew geometry
  with nothing else changed. On the M3 Pro the two orders tied (134 vs 132). The rule for the weight packer: *the 32
  lanes' k-th words are adjacent in memory*; the SIMD-group still sweeps one contiguous block, so everything else in
  D8 survives.
* **N2 — Real decode kernels (`p13`).** A first-cut FP8 GEMV with 16 B loads, R-row blocks and FP32 accumulation is
  bus-bound at the crew geometry: 275 GB/s (90 %) with the interleaved pack, 238 with the contiguous one, 277 with the
  conventional many-small-threadgroups geometry — the geometry is at parity, the layout is worth +16 %. The same
  kernel for NVFP4 is **ALU-bound**: 137 GB/s of useful bytes at the crew geometry and 182 at 108 SIMD-groups per core
  — ~275 G weights/s either way, the same weights-per-second as FP8, i.e. the nibble decode is the limiter. More
  SIMD-groups per core help every ALU-heavy variant (NVFP4 T = 1 +33 %, FP8 T = 2 and 4 +19–44 %), so the crew
  geometry's "one threadgroup per core" is a default, not a law. Tail quantization of static slices is visible: with
  240 SIMD-groups, R = 32 leaves 544 blocks → 76 % efficiency, R = 16 → 91 %, R = 4 → 95 %.
* **N3 — Cost of verifying T tokens on the shader ALUs (`p13`).** Relative to a T = 1 pass: FP8 T = 2 ×1.08, T = 4
  ×1.11, T = 8 ×3.6; NVFP4 T = 2 ×1.28, T = 4 ×1.79, T = 8 ×5.3. Speculation that verifies through NVFP4 MLPs (55 %
  of the model's bytes) on the shader path pays only at T ≤ 2–3.
* **N4 — The neural accelerators are reachable and useful from T ≈ 3–8 (`p14`).** With the Command Line Tools alone,
  the runtime Metal compiler accepts `<MetalPerformancePrimitives/…>` and `<metal_tensor>` at MSL 4.0. A block
  dequantizes a [64 × 64] weight tile into threadgroup memory with ordinary loads, wraps it in a `tensor_inline`,
  calls `matmul2d<…, execution_simdgroups<S>>` and accumulates in a cooperative tensor stored once at the end.
  FP8, TM = 8: 186 GB/s = 1.5× a T = 1 pass for 8 tokens; NVFP4: 119 GB/s useful = 1.5×; TM = 32 costs 1.7–1.8× a
  T = 1 pass. Crossover against the shader kernels: FP8 from T ≈ 5–8, NVFP4 from T ≈ 3. The accelerator path's own
  streaming ceiling (half weights read directly) is 200–210 GB/s (65–68 %), so it is not a T = 1 path — consistent
  with every report in the survey. Its accumulation error is 1.3–1.6e-6 (FP32 FMA path: 9e-8), far inside the BF16
  gates.
* **N5 — Encode order (`p12` §5).** With a streamer that saturates the bus, an ALU-bound sibling hides completely
  only when it is encoded *before* the bus-bound op (102–126 % up to 44 % of the partner's length, 84–93 % at 85 %,
  and the bus-bound op hides entirely inside a 1.7× longer ALU op); encoded after it, only a ≤ 22 % op hides. Two
  bus-bound ops still gain nothing (292–296 → 295–297 GB/s).
* **N6 — Everything latency-bound is 25–40 % faster per SIMD-group** (atomic handoff 142–196 ns, hot read 37 ns,
  clock tick 38 ns, ALU 1.12×) while the dispatch overhead is unchanged (1.3–1.6 µs). A 27B token at 291 GB/s is
  ~62 ms, so 330 dispatches cost ~0.5 ms = 0.8 % — the "launch overhead is not the problem" arithmetic of design §2
  gets stronger, not weaker.

---

## 4. Continuing on M4

*Roadmap note (2026-09-25): the M4 measurement was dropped from the roadmap with the M3 Pro tasks; this section stays as
the checklist for any chip not yet measured — the hypotheses and the design decisions each one would change.*

**Do this first:** `./probes/run_all.sh` on the M4, commit the results files, fill the M4 column of §1, then walk the
table below. M4 is the same GPU family as the M3 (Apple9, dynamic caching), so most mechanisms should carry over; the
M5 Pro showed that *the same family label does not guarantee the same memory-system behaviour*, so the layout and
order rules (N1, N5) must be re-measured, not assumed:

| Chip | GPU cores | Nominal GB/s | GB/s per core | Uncontended GB/s per core, measured |
|---|---|---|---|---|
| M3 Pro (measured) | 18 | 153.6 | 8.5 | 18 |
| M5 Pro (measured) | 20 | 307 | 15.4 | 29 (lane-contiguous) / 71 (lane-interleaved) |
| M4 | 10 | 120 | 12.0 | ? |
| M4 Pro | 16 / 20 | 273 | 17.1 / 13.7 | ? |
| M4 Max | 32 / 40 | 410 / 546 | 12.8 / 13.7 | ? |

| Hypothesis for M4 | Probe | If it fails |
|---|---|---|
| H1 Same SIMD width (32), limits, and lockstep deadlock | `p1`, `p2` | kernels assume 32 lanes (BLM stripe count) — make it a profile value |
| H2 One 384-thread threadgroup per core, ~12 SIMD-groups at full speed (`p4`: times flat up to G = cores, 2× just above) | `p4` | crew geometry becomes `cores × (SIMD-groups per core × 32)`; D4 is already profile-driven |
| H3′ Which intra-block lane order streams at ≥ 90 % of nominal at the crew geometry — contiguous (M3 Pro: tie), interleaved (M5 Pro: only this one), or neither | `p12` §1/§3/§4, then `p13` | if neither: adopt MLX's GEMV geometry for that chip (plan M1 fallback) |
| H4 A dispatch boundary is still no dearer than an in-kernel barrier | `p10` (compare "ONE DISPATCH PER OP" with the static-slices and own+steal rows, both builds) | **revisit D5**: multi-op kernels with in-kernel barriers could pay on this chip |
| H5′ Sharing: does a foreign dispatch get in during a dispatch (never on Apple9, sometimes on Apple10), and how often does it wait for a whole command buffer | `p6` and `p6b`, **4 runs each** | tighten or relax the command-buffer length (D6) |
| H6′ How many cores saturate the bus with the best streamer (M3 Pro 12 of 18, M5 Pro 6 of 20) | `p12` §4 | sets how much ALU the overlap rule (D14) can harvest |
| H7′ Is an ALU-bound op hidden inside a saturating bus-bound op, and does encode order matter (M3 Pro no, M5 Pro yes) | `p12` §5 | the sibling-overlap rule and its order become profile values |
| H8 Threadgroup memory is not faster than device memory | `p8` | reconsider staging small hot data (activation stripes) |
| H9 FP8 T = 1 bus-bound at the crew geometry; NVFP4 T = 1 ALU-bound; T-cost curve | `p13` | the M1 kernel study's priorities for that chip |
| H10 MPP `matmul2d` on Apple9 runs on the shader ALUs (survey: 1.05–1.21× over `simdgroup_matrix`) — so `p14` should *lose* to `p13` at every T | `p14` | if it wins anyway, the accelerator path is not M5-only |

Still to write: (a) *done as `p15` (§6)*: the on-screen frame-pacing check found the display path unaffected by
compute command buffers of 8–133 ms on the M5 Pro (ALU-bound; the bus-bound pass needs an unlocked screen and is
pending); (b) `p13` with the NVFP4 decode rewritten around 16-bit packed math and a register LUT — the current
decode is the ALU limiter; (c) `p14` with the tile fill done through a cooperative right-input tensor instead of
threadgroup memory, and with the next tile's dequantization overlapped with the current `matmul2d` (design §5.12's
M5-only pipelining idea).

Practicalities: the streaming probes allocate 2–3 GB buffers (have ~10 GB free; `p13`/`p14` take ~2.1 GB each at a
time). The probes run on any M4; the target model needs ~21 GB of weights plus state, i.e. a ≥ 36 GB machine (M4 Pro
48 GB, M4 Max) — a 16–32 GB base M4, like the 24 GB M5 Pro measured here, can characterize the GPU but cannot host
Qwen3.8-27B.

---

## 5. Details (M3 Pro)

### P2 — in-kernel synchronization (`p2_sync`)

Two designated threads of one dispatch alternately increment a shared `atomic_uint` 20,000 times each, with bounded
spins so the kernel always terminates.

```
same SIMD-group        (tid 0 vs 1)            : A done=1 B done=0   timeouts      <- lockstep: cannot hand off
diff SIMD-group, same threadgroup (0 vs 32)     : 40000 handoffs     218 ns/handoff (avg 2.0 spins/wait, max 4)
diff SIMD-group, same threadgroup (0 vs 512)    : 40000 handoffs     229 ns/handoff
diff threadgroups (2 x 64)                      : 40000 handoffs     195 ns/handoff
diff threadgroups far apart (TG0 vs TG15 of 16) : 40000 handoffs     206 ns/handoff
```

Through MSL 4.0 atomic operations accept only `memory_order_relaxed` (MSL 4.1 adds acquire/release orders) and are
32-bit except `ulong` min/max; that is sufficient for monotone counters and claim cursors.

### P3/P4 — the core model (`p3_residency`, `p4_core_model`)

Fixed ALU work per SIMD-group; GPU time of the dispatch with all SIMD-groups busy (one SIMD-group alone = 15.2 ms):

```
threads/TG   G=1    G=2    G=4    G=9    G=12   G=16   G=18   G=20   G=24   G=36   G=54   G=72
   32        15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2
  128        15.2   15.2   15.2   15.2   15.2   15.4   15.4   15.2   15.2   15.2   19.9   23.6
  256        15.2   15.2   15.2   15.2   15.2   15.2   15.2   23.0   23.2   23.6   31.4   43.2
  384        15.3   15.4   15.6   15.6   15.7   15.7   15.7   31.5   31.5   31.6   47.5   60.0    <- flat to 18, then 2x / 3x / 4x
  512        17.0   23.0   23.4   23.6   23.4   23.6   23.6   38.0   42.9   43.0   60.5   76.7
 1024        39.2   42.8   41.9   43.1   43.3   43.3   43.0   76.7   78.3   79.0  110.5  141.7
```

Inside one 1024-thread threadgroup, 1–14 busy SIMD-groups run at full speed; 16 → ×1.09; 20–28 → ×2.0; 32 → ×2.6.
The firmware hands whole threadgroups to cores: a 384-thread threadgroup fills one core's full-speed capacity and 18 of
them fill the GPU with no time-slicing. Small threadgroups are packed several per core, less efficiently (216
SIMD-groups as 54 × 128 threads reach throughput 160 vs 210 as 18 × 384).

In flight: with every SIMD-group spinning until all have checked in, 1,536 SIMD-groups (as 48 × 1024 or 1536 × 32
threads) all see each other within 9 spins; at 2,048 most time out (the rest are admitted only as residents exit).

### P5/P5b — bandwidth (`p5_bandwidth`, `p5b_access_pattern`)

3.2 GB streamed, raw `uint` sum:

```
crew geometry (18 x 384): lanes on far-apart stripes                  59 - 60 GB/s
crew geometry: lanes interleaved word by word                         84 - 90 GB/s
crew geometry: 64 KB blocks, lane = contiguous 2 KB sub-range        124 - 134 GB/s   <- best
crew geometry: 1 MB blocks, lane = 32 KB sub-range                   113 GB/s
conventional, 3456 SIMD-groups: blocked 64 KB / interleaved / striped 106-117 / 108-126 / 87-92 GB/s
```

A deliberately naive NVFP4-style kernel (byte loads, 16-entry LUT, per-16 block scale, `half` activations) peaks at
49–58 GB/s of weight bytes: ~1.5 memory loads per weight at ~40–60 ns per load per lane (P8 shows the same ~60 ns for a
hot 8 KB working set). The production GEMV must load packed `uint32/uint64` words, keep per-row accumulators in
registers, and keep a lane's activation stripe small and contiguous.

### P6 / P6b — sharing (`p6_preemption`, `p6b_interleave`)

```
small dispatch on an idle GPU                                          : mean 0.46 ms
one long dispatch on 18 / 16 / 12 of 18 cores; small one submitted 150 ms in : waited 1417 / 1402 / 1374 ms

queue A: 1 command buffer  x 800 dispatches of ~1.5 ms | queue B meanwhile: mean 0.32-0.42 ms, max 0.7-1.9 ms  (5 trials)
                                                        |                   mean 1134 ms (n=1)                 (1 trial)
queue A: 8 command buffers x 100 dispatches (~155 ms)   | mean 0.33-3.3 ms, max 0.8 / 1.5 / 10.9 / 130 / 140 ms
queue A: 80 command buffers x 10 dispatches (~15 ms)    | mean 0.33-1.0 ms, max 0.45-8.8 ms
```

Other work is normally scheduled between dispatches, but it can be held behind an entire in-flight command buffer; it
never gets in during a dispatch. Same process, second command queue — the window compositor was not measured directly.

### P15 — on-screen frame pacing under compute command buffers (`p15_frame_pacing`, M5 Pro, 2026-09-25)

A 360 × 220 window presents one trivially rendered frame per vsync through a `CAMetalLayer` on its own queue and
records each frame's presented time; a second queue keeps 3 compute command buffers in flight, each L ms of ~1 ms
ALU dispatches at the crew geometry (`aluwork`, calibrated), for 4 s per L. The display ran at 120 Hz (8.33 ms):

```
L ms   heavy queue                    frames  mean ms  p99 ms  max ms  late >= 2 vsyncs  fps
0      idle                              479     8.35    8.33   16.67   0.2 %           119.7
8      501 buffers of 8.0 ms             481     8.33    8.33    8.33   0.0 %           120.0
16     251 buffers of 16.1 ms            483     8.33    8.33    8.33   0.0 %           120.0
33     123 buffers of 33.2 ms            489     8.33    8.33    8.33   0.0 %           120.0
66      63 buffers of 66.3 ms            501     8.33    8.33    8.34   0.0 %           120.0
133     32 buffers of 133.7 ms           513     8.33    8.33    8.33   0.0 %           120.0
```

Every frame arrived on its vsync at every buffer length — the WindowServer's work is not the "foreign dispatch on a
second queue" of `p6b`, which waited for whole buffers: the display path is scheduled ahead of (or interleaved
with) our compute buffers. Consequences: `max_cb_ms` does not protect frame pacing on this chip; it remains a
latency knob (how far the pump over-runs a request, how soon a stop lands) and stays at 16 ms. Caveats: the window
must be on screen — an occluded window's drawables are never presented, and with the screen locked nothing is
composited (the probe detects the lock and skips) — so the bus-bound pass (the same phases with `p12`'s streaming
kernel, 292 GB/s per dispatch, the engine's own kind of load) is written but still to be run with the screen
unlocked (`./probes/run_all.sh p15_frame_pacing`); and the heavy queue is in the same process as the window, like
`p6b`.

### P7 — launch overhead (`p7_dispatch_overhead`)

```
(A0) 1 cmdbuf, 1 encoder, 1240 dispatches : CPU encode 0.16 ms (0.13 us each) | GPU 1.85-2.28 ms (1.5-1.8 us each)
(A1) concurrent encoder + barriers        : CPU encode 0.47 ms                | GPU 1.85 ms
(B)  ICB built once, replayed             : CPU encode 0.01-0.02 ms           | GPU 1.63-1.68 ms (1.3-1.4 us each)
(C)  the same 1240 ops inside ONE dispatch: CPU encode 0.003-0.008 ms         | GPU 0.002-0.003 ms   (no barriers between ops)
(D)  commit + waitUntilCompleted, 1 tiny dispatch : 0.12-0.13 ms wall         <- every CPU<->GPU sync point
```

Floor numbers for the Metal API itself (same pipeline and buffers rebound each time). Other reports put the GPU cost
per dispatch at 2.6–5 µs on other chips (see the survey).

### P9 — clock SIMD-group (`p9_clock_warp`)

One SIMD-group increments an `atomic_uint` in a tight loop; every other SIMD-group reads it before and after fixed
workloads of 1×–4×. Ticks are converted to time with the command buffer's `GPUStartTime/GPUEndTime`.

```
58.82 ms, 1,240,065 ticks -> 47.4 ns/tick | 1x 14.43  2x 28.99  3x 43.36  4x 58.15 ms
59.73 ms, 1,238,017 ticks -> 48.2 ns/tick | 1x 14.58  2x 29.01  3x 43.76  4x 58.75 ms
66.02 ms, 1,241,089 ticks -> 53.2 ns/tick | 1x 16.11  2x 32.29  3x 48.51  4x 65.28 ms   (reference run, on battery)
```

### P10 — in-kernel barriers vs dispatch boundaries (`p10_claim_protocol`)

One dispatch runs 320 ops in sequence, 544 blocks each (≈ one decode step: 17,408 rows / 32, five barriers per layer).
Lane 0 of each SIMD-group claims a block with a bounded `compare_exchange`, broadcasts it with `simd_broadcast_first`,
all 32 lanes run the block, lane 0 bumps `done[op]`; the op ends when `done == n_blocks` (bounded spin).
`hits[op][block]` must end at exactly 1 everywhere. "Fenced" is the spec-compliant build: MSL 3.2, `coherent(device)`
buffers, `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` before every publish and at every barrier
exit — required by the MSL memory model (§4.8, §6.16) for cross-threadgroup visibility of non-atomic data, although
the unfenced build also worked in every run.

```
                                              protocol only, us/op          with ~30 us blocks, us/op    exactly-once
ONE DISPATCH PER OP (static slices, no sync)   1.6 - 2.7                     92.0 - 94.2                   yes
one kernel, barriers, static slices            2.1 - 2.6   (fenced 3.1-3.9)  92.3 - 95.9                   yes
one kernel, barriers, own slice + steal        4.7 - 5.4   (fenced 5.4-6.1)  93.9 - 96.5                   yes
one kernel, barriers, global claim cursor      87 - 90                       125 - 126                     yes
own + steal, half the crew present             -                             187 - 195  (static re-partitioned: 179-221)   yes
own + steal, 2x / 4x surplus threadgroups      -                             98 - 114 / 121 - 126          yes
```

Victim order matters: with every thief scanning victims in the same order the half-crew case took 264 µs/op; a
per-SIMD-group stride coprime with the crew size brought it to 187–195. Timeouts: 0 in every configuration.

### P11 — inter-op overlap (`p11_interop_overlap`)

A = bus-bound 3.2 GB block sweep; B = ALU-only work, total work held constant across core counts (2.7 ms on 18 cores).

```
(1) A alone on G cores:  1: 18.3 GB/s | 2: 36.8 | 4: 67.8 | 6: 93.9 | 9: 120.4 | 12: 127.9 | 15: 131.6 | 18: 131.2   (other runs: 9 -> 114-126, 18 -> 127-133)
(3) serial A then B (all cores each)                      27.25 ms   (A 24.55 + B 2.71)
    no barrier, A || B, both on all cores                 24.72 ms   <- B fully hidden
    no barrier, A on 15 cores || B on 3 cores             31.17 ms   <- worse than serial
    no barrier, A on 12 cores || B on 6 cores             25.07 ms
    no barrier, B on 6 || A on 12 (B encoded first)       29.04 ms   <- order-sensitive when cores are split
(3b) both on all cores, B encoded first                   25.30 ms
    B = 2x / 4x / 8x work (22 % / 44 % / 87 % of A)       hidden 86-98 % / 88-114 % / 95-100 %
(4) two bus-bound ops: serial 118-131 GB/s | split cores 126-131 | both on all cores 127-130
```

---

## 6. Details (M5 Pro)

Full outputs are in the 11 results files; these are the excerpts the conclusions rest on.

### P6 / P6b — sharing, four runs each (`p6_preemption`, `p6b_interleave`)

The long dispatch is 1.25 s of ALU work on G threadgroups of 384; a second queue submits a ~30 µs dispatch every 20 ms.
"got in" = completed while the long dispatch was still running.

```
run  G=20 (all cores)                          G=18                                   G=13
 1   45 got in, all 0.20-0.36 ms               25 got in: 24 at ~0.2 ms, one 493 ms    14 got in: 13 at ~0.2 ms, one 766 ms
 2   first waited 1114 ms (the whole dispatch) first waited 1096 ms                    22 got in: 21 at ~0.2 ms, one 501 ms
 3   4 at ~0.2 ms, then one waited 992 ms      2 at ~0.2 ms, then one waited 1039 ms   first waited 1107 ms
 4   42 got in, max 1.49 ms                    first waited 1099 ms                    10 got in: 9 at ~0.2 ms, one 859 ms
idle-GPU latency of the small dispatch: mean 0.18-0.36 ms

p6b  1 cmdbuf x 800 dispatches (1.15 s)        8 cmdbufs x 100 (~145 ms each)          80 cmdbufs x 10 (~14.5 ms each)
 1   8 at 0.23 ms, then the 9th ~0.95 s*       n=69 mean 0.85  max 44.5 ms             n=63 mean 0.21  max 0.27 ms
 2   first waited 1040 ms                      n=55 mean 4.48  max 91.8 ms             n=56 mean 0.30  max 1.34 ms
 3   first waited 1048 ms                      n=62 mean 2.82  max 129 ms              n=57 mean 0.21  max 1.23 ms
 4   first waited 1049 ms                      n=67 mean 1.53  max 82.6 ms             n=56 mean 0.30  max 1.14 ms
* inferred: the first run predates the "last sample" column added to p6b; its loop ended after 9 samples in 1152 ms
```

Reading: Apple10's firmware can interleave another queue's threadgroups into a running dispatch, but whether it does
is decided at some point we do not control — the same configuration went from "always" to "never" between runs.
Behind a long command buffer of short dispatches the foreign dispatch usually waits for the entire buffer; ~15 ms
buffers keep it under 1.5 ms. That is D6's rule with a stronger reason.

### P12 — streaming geometry (`p12_stream_geometry`, 3 runs)

3.2 GB, GB/s, min-of-3, 384-thread threadgroups, 64 KB blocks unless stated:

```
(1) SIMD-groups per core:                     12     24     48     96    192    384
  striped 4B                                85.2   82.8   92.5  128.1  197.1  202.4
  interleaved 4B                           113.6  203.2  278.5  228.1  241.7  254.3
  blocked lane-contig 4B (p5b/D8)          199.3  206.7  202.7  204.2  208.1  211.1
  interleaved 16B                          277.5  240.6  274.5  272.2  279.1  280.2
  interleaved 16B x4 in flight             280.7  280.8  274.7  279.1  286.3  278.4
  blocked lane-contig 16B                  211.8  208.9  211.1  208.8  205.3  204.9
  blocked lane-contig 16B, 4 streams/lane  217.7  207.9  206.5  204.2  207.3  200.8
  blocked lane-contig 16B, 64B bursts      214.7  207.7  210.6  209.7  210.8  206.8
  blocked lane-INTERLEAVED 16B x4          290.6  292.0  290.5  284.6  287.5  285.0   <- the pattern for the packer
(3) block size, lane-interleaved 16B x4, 12 SG/core:  16 KB 294.5 | 64 KB 291.3 | 256 KB 292.9 | 1 MB 287.2
    lane-contig 16B, 4 streams/lane, 12 SG/core:      16 KB 269.8 | 64 KB 216.0 | 256 KB 216.3 | 1 MB 204.7   (16 KB blocks with 4 streams approximate interleaving)
(4) one threadgroup per core, cores:            1      2      4      6     10     13     16     20
  blocked lane-contig 4B                     29.2   55.9  104.8  144.8  192.4  209.2  214.6  214.8
  blocked lane-interleaved 16B x4            71.4  134.5  230.8  278.3  286.2  289.3  293.5  286.0
(5) ALU-bound op B beside the saturating streamer A (10.9-11.1 ms, 289-295 GB/s), no barrier, 3 runs, "hidden" = (serial - overlapped) / B alone
  B = 21-22 % of A:  A then B  100 / 121 / 122 %   |  B then A  100 / 126 / 126 %
  B = 41-44 %:       A then B   29 /   4 /  -5 %   |  B then A  121 / 103 / 102 %
  B = 85-87 %:       A then B    3 /   1 /   5 %   |  B then A   93 /  93 /  84 %
  B = 169-174 %:     A then B    2 /  -1 /   0 %   |  B then A   57 /  58 /  61 %   (= A entirely hidden inside B)
(6) two saturating bus-bound ops, 1.6 GB each: serial 292-296 GB/s | overlapped 295-297 GB/s
```

### P13 — real decode GEMV (`p13_decode_gemv`)

y[T][17408] = x[T][5120] · Wᵀ; ~2.1 GB streamed per measurement (identical copies of the packed matrix), min-of-3;
GB/s counts useful weight (+scale) bytes; every variant matches a CPU reference to ≤ 1.2e-7 relative. Geometries: crew
= one 384-thread threadgroup per core with static slices (blocks per SIMD-group and the tail efficiency in brackets);
"2 TG" = two per core; "1 blk" = one block per SIMD-group in 384- or 64-thread threadgroups (the MLX/llama.cpp shape).

```
FP8 E4M3                              crew                     2 TG     1 blk/384  1 blk/64
R=32 T=1 lane-contiguous    203.3 [2.3 blocks/SG, 76 %]     201.5     198.5     219.1
R=16 T=1 lane-contiguous    226.2 [4.5, 91 %]               193.2     197.2     201.2
R=8  T=1 lane-contiguous    237.8 [9.1, 91 %]               232.3     234.9     232.8
R=4  T=1 lane-contiguous    235.7 [18.1, 95 %]              232.1     238.0     239.5
R=16 T=2 lane-contiguous    198.9                           194.0     200.9     201.4
R=8  T=4 lane-contiguous    175.9                           173.1     189.3     197.7
R=8  T=8 lane-contiguous     71.3                            64.4      66.3      69.1
R=32 T=1 lane-interleaved   233.7 [76 %]                    238.3     196.1     221.2
R=16 T=1 lane-interleaved   275.2 [91 %]   <- 90 % of nominal at the crew geometry   265.2  268.6  276.7
R=8  T=1 lane-interleaved   267.8                           263.3     259.8     276.8
R=4  T=1 lane-interleaved   263.1                           269.6     270.8     275.0
R=16 T=2 lane-interleaved   215.1                           255.3     204.3     200.8
R=8  T=4 lane-interleaved   173.4                           235.5     242.1     248.6
R=8  T=8 lane-interleaved    74.0                            67.8      65.4      75.8

NVFP4 (E2M1 + E4M3 per 16), useful bytes
R=32 T=1 lane-contiguous    112.3 [76 %]                    125.0     123.5     131.4
R=16 T=1 lane-contiguous    132.2                           144.8     146.7     156.5
R=8  T=1 lane-contiguous    133.6                           142.6     154.1     165.9
R=4  T=1 lane-contiguous    112.3                           150.7     166.6     172.3
R=16 T=2 lane-contiguous    115.4                           124.8     130.4     138.9
R=8  T=4 lane-contiguous     80.2                            78.8      88.3      94.2
R=8  T=8 lane-contiguous     34.5                            19.8      17.4      17.4
R=32 T=1 lane-interleaved   108.4 [76 %]                    128.5     123.5     156.9
R=16 T=1 lane-interleaved   129.4                           147.9     155.4     156.3
R=8  T=1 lane-interleaved   130.0                           144.9     163.5     176.4
R=4  T=1 lane-interleaved   137.1                           153.3     175.9     182.3   <- 59 % of nominal; ALU-bound
R=16 T=2 lane-interleaved   114.5                           124.6     116.2     142.6
R=8  T=4 lane-interleaved    83.1                            81.2      94.7     101.8
R=8  T=8 lane-interleaved    32.6                            19.6      17.6      17.1
```

Kernel structure: each lane owns a 160-column stripe; per 16 B weight load it decodes 16 (FP8) or 32 (NVFP4) weights
with bit arithmetic (no LUT loads), re-reads the activation chunk from cached device memory for RG = 4 rows (RG = 2 at
T = 8), accumulates in FP32 and `simd_sum`s once per (row, token). The T = 8 collapse and the NVFP4 ceiling are
properties of this first cut (register pressure; ~7 ALU ops per weight for the decode), not of the chip — they set the
agenda for plan M1, they do not close it.

### P14 — neural accelerators (`p14_tensor_ops`)

y[TM][17408] = x[TM][5120] · Wᵀ through `mpp::tensor_ops::matmul2d<desc, execution_simdgroups<S>>`; each
threadgroup owns TN rows and loops over K in TK-wide tiles; `gemm_fp8`/`gemm_nvfp4` dequantize each tile into
threadgroup memory first, `gemm_half` multiplies straight from a device tensor. ~2.1 GB streamed per measurement,
min-of-3; 8 real token rows are checked against the CPU reference (the rest are zero padding).

```
TM  TN  TK  S    gemm_fp8 (89 MB/matrix)        gemm_nvfp4 (50 MB useful)      gemm_half (178 MB, upper bound)
                 ms      GB/s   TFLOP/s         ms      GB/s   TFLOP/s         ms      GB/s   TFLOP/s
 8  64   64 4    0.506  176.2   2.82            0.445  112.7   3.21            0.850  209.7   1.68
 8  64  128 4    0.537  165.8   2.65            0.425  117.9   3.35            0.863  206.6   1.65
 8  32   64 2    0.513  173.6   2.78            0.442  113.3   3.22            0.866  205.8   1.65
 8  64   64 2    0.479  186.0   2.98            0.426  117.6   3.35            0.862  206.7   1.65
 8  64   64 1    0.527  169.1   2.71            0.421  119.0   3.39            0.871  204.7   1.64
16  64   64 4    0.528  168.8   5.40            0.471  106.4   6.05            0.883  201.9   3.23
32  64   64 4    0.559  159.3  10.20            0.545   92.0  10.46            0.911  195.7   6.26
32  64  128 4    0.597  149.3   9.56            0.489  102.4  11.65            0.900  198.0   6.34
32 128   64 4    0.698  127.8   8.18            0.526   95.2  10.84            0.863  206.6   6.61
64  64   64 4    0.734  121.4  15.54            0.674   74.4  16.94            1.027  173.5  11.10
relative error vs the CPU reference: 1.6e-6 (fp8, half), 1.3e-6 (nvfp4); the FP32-FMA kernels of p13: 0.9-1.2e-7
```

Against `p13`'s best shader-ALU passes (FP8 T = 1 0.322 ms, T = 4 0.359, T = 8 1.175; NVFP4 T = 1 0.275, T = 2 0.352,
T = 4 0.492, T = 8 1.452): the accelerator verifies 8 tokens for 1.5× a T = 1 pass in either format, 32 tokens for
1.7–1.8×, and beats the shader kernels from T ≈ 5–8 (FP8) and T ≈ 3 (NVFP4). The half-weight upper bound of
200–210 GB/s says the accelerator path itself streams at 65–68 % of nominal — it is a T > 1 path, not a T = 1 path.

---

## 7. Caveats

* Two chips, one OS build each, one day each; the M3 Pro run was on battery, the M5 Pro run on AC with the display
  attached. The M5 Pro runs macOS 26.5.1, the M3 Pro 26.6.2 — a firmware difference between the two could be OS, not
  silicon. Threadgroup → core mapping, the in-flight limit, sharing granularity and the absence of a watchdog kill are
  observed firmware behaviour. The design stays *correct* if any of them changes (it relies only on Metal's documented
  dispatch ordering and ICB barriers) and only loses performance.
* Microsecond-level numbers move by tens of percent between runs (compare the ranges above); conclusions are drawn only
  where the ranges do not overlap, or from with-work measurements. The M5 Pro's in-dispatch sharing is bimodal, so its
  "sometimes" is a statement about 12 trials, not a rate.
* `p12`–`p14` have run only on the M5 Pro. "Lane order decides bandwidth", "crew geometry = parity" and the T-cost
  curves are M5 Pro facts until the M3 Pro and an M4 have run them. `p13`'s kernels are first cuts (the NVFP4 decode
  is the ALU limiter; the T = 8 variant spills); `p14`'s tile shapes are untuned. They bound what a kernel study can
  gain, they do not replace it (plan M1).
* The 24 GB M5 Pro cannot host the 27B target; baselines against MLX/llama.cpp on the real model still need the 36 GB
  M3 Pro or a larger machine.
* P6/P6b used two queues of one process; cross-process behaviour (the window compositor) was not measured directly.
