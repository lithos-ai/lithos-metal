# basalt's Q5_K head as nvcc compiles it

What NVIDIA's compiler emits at every rounding site of basalt's Q5_K head (`src/kernels/cuda/q5k_head.cu`) and of the
2-bit proxy it includes (`src/kernels/cuda/head_proxy.cuh`), read from the sm_120 PTX and SASS, and the Metal form the
port writes at each site (`kernels/common/basalt_q5k_head.metal`). `q5k_head.cu` builds with `--use_fast_math`
(`CMakeLists.txt:200`). The format follows `matvec.md`, whose method, notation and helpers this page uses.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last.

## Method

- **Sources.** `q5k_head.cu` verbatim with the quoted includes inlined (`basalt/kernels/q5k_head.hpp`, `dp4a.hpp`,
  `pdl.hpp`, `head_proxy.cuh`). Only basalt's public MIT source was sent.
- **Compilers and flags.** Compiler Explorer's public API as in `matvec.md`: NVCC 13.0.0 (`nvcc130`) and 13.3.0
  (`nvcc133`), each with basalt's flags for this file: `-O3 -DNDEBUG -DBASALT_VERSION="0.0.1"
  --generate-code=arch=compute_120,code=[compute_120,sm_120] --use_fast_math`, plus `-Xptxas -v` for the register
  counts.
- **Translation units.** `q5k_head.cu:1-252` (includes, block structs, `q5_load`, `q5_apply`, `q5k_row_dev`, `Top`,
  `q5k_rows_10`, `q5k_head_kernel`), for the top-1 unit also `:253-335` (`q5k_head_top1_kernel`) and `:399-404`
  (`mask_unmapped_kernel`), for the proxy units `:436-498` (`q5k_proxy_pack_kernel`, `q5k_proxy_final_kernel`); then a
  table that takes the address of each kernel instantiation the unit emits (`q5k_head_kernel<1, 10>`, `<2, 10>`,
  `<8, 10>`, `<1, 0>`, `<8, 0>`; `q5k_head_top1_kernel`, `mask_unmapped_kernel`; `q5k_proxy_pack_kernel`;
  `q5k_proxy_final_kernel`; `px::score_kernel`, which every unit emits). The host launchers are left out: a kernel's
  code depends only on its template and the inlined device functions. Nine units per compiler, all compiled (code 0);
  only `q5k_head_kernel<8, 0>`'s SASS listing reached Compiler Explorer's 5,000-line cut (its PTX is complete, and
  `<8, 10>` has the same per-(row, column) templates).
- **Reading a site.** As in `matvec.md`: the PTX shows what NVVM fused and each conversion; the SASS what `ptxas` made
  of it. The dataflow tracer of `matvec.md` prints each floating-point SASS instruction as an expression template;
  the templates below are 13.3's. 13.0 emits the same operations, with the same operands and modifiers, at every
  site this page records; its listings differ only in scheduling and predicate choice (sections 4.2 and 4.3).

## Register and shared-memory use (sm_120, `ptxas -v`, both compilers)

| Kernel | Registers | Shared memory | Barriers |
|---|---|---|---|
| `q5k_head_kernel<1, 10>`, `<2, 10>` | 128 | dynamic (columns) | 1 |
| `q5k_head_kernel<8, 10>` | 156 | dynamic | 1 |
| `q5k_head_kernel<1, 0>` | 63 | dynamic | 1 |
| `q5k_head_kernel<8, 0>` | 128 | dynamic | 1 |
| `q5k_head_top1_kernel` | 126 | 7,777 B | 1 |
| `mask_unmapped_kernel` | 10 | 0 | 0 |
| `q5k_proxy_pack_kernel` | 56 | 0 | 0 |
| `q5k_proxy_final_kernel` | 40 | 8,516 B (`xe`, 7,680 B, and `px::Final`, 836 B) | 1 |
| `px::score_kernel` | 77 | 3,712 B | 1 |

The top-1's grid (`q5k_head.cu:375-386`) is the multiprocessor count times `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
for 256 threads, capped at 4,096. On sm_120 (65,536 registers per multiprocessor, allocation in 8-register units per
thread) 126 registers round to 128: 128 × 256 = 32,768 registers a block, two blocks a multiprocessor; the 7,777 bytes of
shared memory and the 1,536-thread limit allow more. Basalt's measured RTX 5090 has 170 multiprocessors
(`docs/research/basalt-nvcc/dense_tc.md` and the scoping report's decision 4), so its grid is 340 blocks for every row
count above 2,720 and ceil(n_out / 8) below.

## Summary

| Kernel | Rounding sequence nvcc 13.x emits (sm_120) | Metal |
|---|---|---|
| `q5k_head_mmvq` (`q5_apply`, `q5k_head.cu:92-106`; `q5k_row_dev` / `q5k_rows_10`, `:110-223`) | per (row, column) `native_mmvq`'s Q5_K sequence (`matvec.md` section 5): sumf_d = fma(d8_i, f(dot1_i·sc_i), sumf_d), sumf_m = fma(d8_i, f(dot2_i·m_i), sumf_m) from 0; v = fma(dm.x, sumf_d, −fl(dm.y·sumf_m)); p_v = 0 + v, then + v per further block; t = ((p_0 + p_1) + p_2) + p_3; butterfly 16..1; all `.FTZ` | the same, `fma_ftz`, `mul_ftz`, `add_ftz`, `warp_sum_ftz`: `basalt_mmvq.metal`'s Q5_K dot, shared |
| `Top::take` (`:155-164`) | l > m: s = fma(s, e, 1) with e = `MUFU.EX2`(fl((m − l)·fl(log2 e))); else s = s + `MUFU.EX2`(fl((l − m)·fl(log2 e))); all `.FTZ` | `fma_ftz(s, expf_ftz(m - l), 1)`; `add_ftz(s, expf_ftz(l - m))`; `expf_ftz(x)` = `ftz(fast::exp2(mul_ftz(x, fl(log2 e))))` |
| `Top::merge` (`:165-175`) | o.m > m (or empty): s = fma(s, e(m − o.m), o.s); else s = fma(o.s, e(o.m − m), s) | `fma_ftz` with the same `expf_ftz` |
| top-1 output (`:330-332`) | `1.0f / b.s` as `MUFU.RCP` alone | `rcp_ftz(s)` |
| `mask_unmapped_kernel` (`:400-403`) | a store of −inf, no arithmetic | the same |
| top-1 merge (`:303-334`) | an atomic election of the last block | a second dispatch, the same merge sequence (section 3) |
| `q5k_proxy_pack_kernel` (`:438-462`; `head_proxy.cuh:36-87`) | w = fma(FMUL(d, sc), q, −FMUL(dmin, m)); amax · fl(1/1.5); FFMA(t, −0.1, 1); sh = fp16(s); level = F2I.FLOOR(FFMA(w, MUFU.RCP(sh), 2)) clamped; e = FFMA(r, r, e) with r = FFMA(level − 1.5, sh, −w); all `.FTZ` | the same with `rcp_ftz`; NaN level 0, clamp in float (section 4.1) |
| `px::score_kernel` (`head_proxy.cuh:89-167`) | acc = FFMA(scale, FFMA(d8, P, −FMUL(xs, 1.5)), acc); the four SIMD-groups' sums left to right; max and sum butterflies with `__expf`; four argmax rounds | the same (section 4.2) |
| `q5k_proxy_final_kernel` (`:464-496`; `head_proxy.cuh:169-240`) | threshold FADD(m, −5); a warp-aggregated shared-memory atomic orders the list; rescoring by the head's row; FFMA(p.y, e, sum) per tile, FADD(e_ex, −e_lp) per row, butterfly, FMNMX, `MUFU.RCP` | the same arithmetic, the list in ascending candidate order (section 4.3) |

The `[H]` of the scoping report (section 7.2: whether the head resolves `dm.x * sumf_d - dm.y * sumf_m` as
`native_mmvq` does) is settled: the head's SASS rounds the dmin product and fuses the d product with its negation,
`FMUL.FTZ` then `FFMA.FTZ ..., -R`, exactly `native_mmvq.cu:200`'s form (`matvec.md` section 5, Q4_K and Q5_K), in both
compilers.

## 1. `q5k_head_mmvq`

`q5k_head_kernel<1, 10>` (13.3), one row's five blocks for lane L (`kb = {a, a + 8, a + 2, a + 4, a + 6}`, `vt = {0, 0,
1, 2, 3}`, `q5k_head.cu:183-185`):

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `sumf_d += d8 * (dot1 * r.sc[i])` (`:102`), i = 0, 1 | `FFMA.FTZ R28, R31, R28, RZ`; `FFMA.FTZ R116, R35, R33, R28` | `FFMA.FTZ(i, in, FFMA.FTZ(i, in, 0))` | `fma_ftz(d8, float(dot1 * sc), sumf_d)` from 0 |
| `sumf_m += d8 * (dot2 * r.m[i])` (`:103`) | `FFMA.FTZ R30, R31, R30, RZ`; `FFMA.FTZ R118, R35, R34, R30` | the same | `fma_ftz(d8, float(dot2 * m), sumf_m)` |
| `r.dm.x * sumf_d - r.dm.y * sumf_m` (`:105`) | `FMUL.FTZ R45, R118, R113`; `FFMA.FTZ R116, R116, R51, -R45` | `FFMA.FTZ(sumf, in, −FMUL.FTZ(sumf, in))` | `fma_ftz(dmx, sumf_d, -mul_ftz(dmy, sumf_m))` |
| `p[vt[c]][j] += q5_apply(...)` (`:206`) | `FADD.FTZ R3, RZ, R116`; `FADD.FTZ R3, R3, R118` (virtual thread 0's two blocks); `FADD.FTZ R6, RZ, R6` (threads 1..3, one block each) | `FADD.FTZ(0, v)`, `FADD.FTZ(p, v)` | `add_ftz(p, v)` from 0 |
| `t = p[0][j]; t += p[v][j]` (`:209-211`) | `FADD.FTZ R3, R3, R6`; `FADD.FTZ R3, R2, R3`; `FADD.FTZ R3, R4, R3` | `((p0 + p1) + p2) + p3` | `add_ftz` ×3 in virtual-thread order |
| `warp_sum(t)` (`:212`) | `SHFL.BFLY` + `FADD.FTZ` with offsets 16, 8, 4, 2, 1 | `FADD.FTZ(in, shfl)` | `warp_sum_ftz` |

That is `native_mmvq`'s Q5_K kernel per value: thread L + 32v's `tmp` starts at +0 and adds its blocks in ascending
kbx (`FADD.FTZ R20, RZ, R0` there too), warp 0 adds warps 1..3's partials in warp order, then the same butterfly
(`matvec.md` section 4). Which of the two `dm` halves is rounded alone is read from the conversions: as in
`native_mmvq`, the `FMUL.FTZ` takes the high half (dmin) and the `FFMA.FTZ` the low half (d).

The column staging (`:229-240`) precomputes per (q8_1 block, position) `{u0, u1, dot2, d8}`: `dot2` is the same
integer `DP4A(0x01010101, u0, DP4A(0x01010101, u1, 0))` as `q5_q8_dot_impl`'s, and `d8` the exact conversion of the
block's fp16 scale (`__low2float`). Staging is not arithmetic.

## 2. `q5k_head_top1`

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `__expf(m - l)` (`:157`) | `FADD.FTZ R0, R56, -R7`; `FMUL.FTZ R0, R0, 1.4426950216293334961`; `MUFU.EX2 R3, R0` | `MUFU.EX2(FMUL.FTZ(fl(log2 e), FADD.FTZ(−l, m)))` | `expf_ftz(m - l)`: `ftz(fast::exp2(mul_ftz(add_ftz(m, -l), 0x3FB8AA3B)))` |
| `s = s * __expf(m - l) + 1.0f` (`:157`) | `@P1 FFMA.FTZ R8, R8, R3, 1` | `FFMA.FTZ(e, s, 1)` | `fma_ftz(s, e, 1.0f)` |
| `s += __expf(l - m)` (`:161`) | `@!P1 FADD.FTZ R2, -R56, R7`; `FMUL.FTZ`; `MUFU.EX2 R5`; `@!P1 FADD.FTZ R8, R8, R5` | `FADD.FTZ(s, MUFU.EX2(...))` | `add_ftz(s, expf_ftz(l - m))` |
| `s = o.s + s * __expf(m - o.m)` (`:168`) | `FADD.FTZ`; `FMUL.FTZ ..., 1.4426950216293334961`; `MUFU.EX2`; `FFMA.FTZ R5, R2, R5, R6` | `FFMA.FTZ(e, s, o.s)` | `fma_ftz(s, expf_ftz(m - o.m), o.s)` |
| `s += o.s * __expf(o.m - m)` (`:172`) | `FFMA.FTZ R5, R6, R0, R5` | `FFMA.FTZ(e, o.s, s)` | `fma_ftz(o.s, expf_ftz(o.m - m), s)` |
| `*out_prob = 1.0f / b.s` (`:332`) | `MUFU.RCP R13, R6`, stored as is | `MUFU.RCP(s)` | `rcp_ftz(s)` |

The comparisons (`l > m`, `l == m && row < idx`, `o.m > m`, `o.idx < idx`) are exact. The per-block merge of the 8
warps (`:297-299`, thread 0, warp order), the last block's thread-strided merge over the block partials (`:311-317`),
the butterfly over the struct (`:318-324`, all three fields shuffled with the same offset, then `merge`) and the final
warp-order merge (`:327-329`) are the same `merge` sequence at each step. `__expf` is `ex2.approx.ftz.f32` of
`mul.ftz.f32 x, 0f3FB8AA3B` in PTX: `MUFU.EX2` has no documented Metal counterpart, and the contract
(`docs/design/flash-next.md`, invariant 5) maps it to `fast::exp2` of the same product; the tests take its values from
the GPU's own function.

## 3. The cross-threadgroup merge

basalt's top-1 ends in an election (`q5k_head.cu:297-335`): thread 0 of every block stores the block's partial
(m, idx, s) to `part`, calls `__threadfence()` and takes `atomicAdd(count, 1u)`; the block that reads `gridDim.x - 1`
calls `__threadfence()` again, reads every block's partial with `__ldcg` and merges them, and resets the counter. The
elected block must see the partial every other block stored before its fence and its atomic.

What the Metal Shading Language (version 4.1, `2026-06-04` edition) documents:

- Coherence of device memory, section 4.8 (pages 122-123): "Memory in the thread address space has thread coherence,
  and memory in the threadgroup address space has threadgroup coherence. By default, memory in the device address
  space has threadgroup coherence." and "Metal 3.2 and later support the coherent(device) qualifiers for buffers ...
  to indicate that the object has device coherence, that is, memory operations are visible across threads on the
  device if you properly synchronize them." A plain `device` partials buffer is therefore not even in scope; it would
  need `coherent(device)`.
- Ordering of atomics, section 6.16.1 (page 299): "For atomic operations other than atomic_thread_fence,
  memory_order_relaxed is the only enumeration value. With memory_order_relaxed, there are no synchronization or
  ordering constraints; the operation only requires atomicity." (Page 300 adds that Metal 4.1 accepts other orders on
  atomic operations.)
- Fences, section 6.16.1 (page 300): "In Metal 3.2 and later, you can use memory_order_seq_cst on
  atomic_thread_fence to indicate that everything that happens before a store operation in one thread becomes a
  visible side effect in the thread that performs the load, and establishes a single total modification order of all
  tagged atomic operations." Section 6.16.3 (page 302): "an atomic_thread_fence with memory_order_seq_cst ordering prevents
  all preceding writes from moving past all subsequent stores within that scope."
- Release-acquire, section 6.16.1.2 (page 300): "The synchronization is established only between the threads
  releasing and acquiring the same atomic variable. Other threads can see a different order of memory accesses than
  either or both synchronized threads."

Each guarantee is stated for one store and the one load that reads it. The specification has no release-sequence
rule: nothing says that a read-modify-write that reads the value of another read-modify-write carries the release of
the threads earlier in the counter's modification order. With 340 blocks, the elected block's RMW reads the value
written by the one RMW just before it in that order; that the other 338 blocks' partials are visible to it is not
stated. So the guarantee
basalt relies on is not explicit, and the port runs the merge as a second dispatch: `basalt_q5k_head_top1` writes
each threadgroup's partial (thread 0 after the 8-SIMD-group merge, as `q5k_head.cu:297-302`), and
`basalt_q5k_head_top1_merge`, one threadgroup of 256 threads, applies the last block's sequence unchanged: thread t
merges partials t, t + 256, ... in order (`:309-317`), the XOR butterfly 16..1 over the struct (`:318-324`), thread 0
the SIMD-groups in order (`:325-329`), then the id and `rcp_ftz(s)` (`:330-332`). The arithmetic and its order are
basalt's; the ordering between the two is the dispatch boundary (a serial compute encoder runs its dispatches in
order; in an indirect command buffer the second command needs the barrier flag, `barrier_before`, which the
program-binding PR sets), and there is no counter to reset between replays of an indirect command buffer.

The other cross-threadgroup dependencies of these kernels:

| Kernel | Dependency | Port |
|---|---|---|
| `q5k_head_mmvq`, `mask_unmapped_kernel` | none: each row is one SIMD-group's (one thread's), written once | as basalt |
| `q5k_proxy_pack_kernel` | none: each group is one thread's | as basalt |
| `px::score_kernel` then `q5k_proxy_final_kernel` | the final kernel reads every tile's candidates and partial; basalt launches the two in order with programmatic dependent launch (`q5k_head.cu:525-528`; `pdl_wait` before the reads, `head_proxy.cuh:109`, `q5k_head.cu:474`) | two dispatches in order: the dispatch boundary |
| `q5k_head_top1_kernel` and the score's `pdl_wait` | the column written by the kernel before (the q8_1 quantizer) | the dispatch before |

Inside a threadgroup every dependency goes through threadgroup memory and `threadgroup_barrier(mem_flags::
mem_threadgroup)`, as basalt's `__syncthreads()`.

## 4. The 2-bit proxy

The copy holds, per tile of 32 rows and per 64 weights, a 16-byte word per row (`lane = row & 31`), and after
`tiles(n_cap) * 40 * 32 * 16` bytes the fp16 scales, two per word (`head_proxy.cuh:17-28, 72-87`); the port keeps the
layout byte for byte, and the score and final kernels take `n_cap` to find the scales and the scratch's partials.

### 4.1 `q5k_proxy_pack_kernel` (`q5k_head.cu:438-462`) with `fit_group<false>` and `pack_group<false>`

The PTX divides with `div.approx.ftz.f32` (`w / sh` and `amax / 1.5f`), floors with `cvt.rmi.ftz.f32.f32` and converts
with `cvt.rzi.ftz.s32.f32`; `ptxas` makes the constant division a multiply, the division by `sh` a reciprocal shared by
the 32 weights, fuses that multiply with the `+ 2.0f`, and folds the floor into the conversion. Both compilers' listings
of this kernel give the same template multiset.

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `d = dm.x * (float) sci`, `mn = dm.y * (float) mi` (`:454`) | `FMUL.FTZ R9, R49, R9`; `FMUL.FTZ R8, R51, R52` | `FMUL.FTZ(h, i)` | `mul_ftz(h2f(d), float(sc))` |
| `w[l] = d * (float) q - mn` (`:459`) | `FFMA.FTZ R10, R9, R10, -R8` | `FFMA.FTZ(d, i, -mn)` | `fma_ftz(d, float(q), -mn)` |
| `amax = fmaxf(amax, fabsf(w[l]))` from `1e-9f` (`head_proxy.cuh:44-45`) | `FMNMX.FTZ R14, \|R10\|, 9.9999997171806853657e-10, !PT`, then `FMNMX.FTZ R38, \|R7\|, R14, !PT`, ... | `FMNMX.FTZ(\|w\|, amax)` | `max_ftz(amax, abs_ftz(w))` (exact; order-free) |
| `amax / 1.5f` (`:46`, `:48`) | `FMUL.FTZ R9, R9, 0.6666666865348815918` | `FMUL.FTZ(fl(1 / 1.5), amax)` | `mul_ftz(amax, 0x3F2AAAAB)` |
| `1.0f - 0.1f * (float) t` (`:48`) | `UI2FP.F32.U32 UR5, UR4`; `UFFMA.FTZ UR5, UR5, -UR6, 1` (`UR6 = 0x3dcccccd`) | `FFMA.FTZ(t, -0.1f, 1)` | `fma_ftz(float(t), -0.1f, 1.0f)` |
| `s = amax / 1.5f * (...)` (`:48`) | `FMUL.FTZ R41, R9, UR5` | `FMUL.FTZ` | `mul_ftz(a15, f)` |
| `sh = __half2float(__float2half(s))` (`:49`) | `F2FP.F16.F32.PACK_AB R45, RZ, R41`; `HADD2.F32 R45, -RZ, R45.H0_H0` | RNE to fp16, exact back | `float(half(s))` (round to nearest even, fp16 subnormals kept: `test_basalt_platform.py`) |
| `level(w, sh)`: `(int) floorf(w / sh + 2.0f)` clamped to 0..3 (`:36`) | `MUFU.RCP R43, R45`; `FFMA.FTZ R0, R10, R43, 2`; `F2I.FTZ.FLOOR.NTZ R0, R0`; then the integer clamp | `F2I.FLOOR(FFMA.FTZ(w, MUFU.RCP(sh), 2))` | `px_level`: `floor(fma_ftz(w, rcp_ftz(sh), 2))`, NaN to 0, clamped in float |
| `r = ((float) level - 1.5f) * sh - w[l]` (`:52`) | `FADD.FTZ R0, R44, -1.5`; `FFMA.FTZ R0, R45, R0, -R10` | `FFMA.FTZ(FADD.FTZ(-1.5, i), sh, -w)` | `fma_ftz(add_ftz(float(lv), -1.5f), sh, -w)` |
| `e += r * r` from 0 (`:50-53`) | `FFMA.FTZ R0, R0, R0, RZ`, then `FFMA.FTZ R47, R47, R47, R0`, ... in weight order | `FFMA.FTZ(r, r, e)` | `fma_ftz(r, r, e)` from `0.0f` |
| `if (e < best_e)` (`:55`) | `FSETP.GEU.FTZ.AND P0, PT, R49, R42, PT`; `FSEL R42, R49, R42, !P0`; `FSEL R40, R41, R40, !P0` | strict, first best kept | `e < best_e` |
| stored scale and the packed levels (`:57-59`, `head_proxy.cuh:76-86`) | `F2FP.F16.F32.PACK_AB R0, RZ, R40`; `HADD2.F32 R3, -RZ, R0.H0_H0`; `MUFU.RCP R3, R3`; `FFMA.FTZ R9, R10, R3, 2`; `F2I.FTZ.FLOOR.NTZ R9, R9` | as above | `half(best_s)`, `px_level` with `rcp_ftz(float(hs))` |

`level`'s conversion is where the 2-bit copy can see a NaN: a group whose weights are all zero (both 6-bit scale
fields zero, about one group in 4,096 of random blocks) has `amax = 1e-9`, `s` near `6.7e-10`, `sh = 0` in fp16,
`MUFU.RCP` gives `+inf` and `0 * inf + 2` is NaN. The PTX ISA (`cvt`): "In float-to-integer conversions, depending upon
conversion types, NaN input results in following value: Zero if source is not .f64 and destination is not .s64, .u64."
and "For float-to-integer conversions, the result is clamped to the destination range by default". MSL takes NaN to 0
as well (section 8.6: "When converting a floating-point type to an integer type, if the floating-point value is NaN,
the resulting integer is 0.") but converts "with no saturation" (section 2.23), so `px_level` tests the NaN and clamps
in float before converting; an `s` past fp16's range (65,520 and up) gives `sh = inf`, a reciprocal of 0 and level 2
everywhere, through the same code. `test_proxy_pack_emulation_bit_for_bit` plants both.

### 4.2 `px::score_kernel` (`head_proxy.cuh:89-167`)

The PTX computes the group term as `mul.ftz.f32` (d8 times P), `mul.ftz.f32` (xs times `0f3FC00000`), `sub.ftz.f32`,
then `fma.rn.ftz.f32` into the accumulator; `ptxas` fuses the first product into the subtraction, keeping the second
rounded, in every one of the 20 groups. 13.0's listing differs from 13.3's only in scheduling: the masked exponential
is a predicated `MUFU.EX2` instead of an `FSEL` after it.

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `xs[tid] = d8 * (float) sum` (`:118`) | `FMUL.FTZ R42, R49, R42` | `FMUL.FTZ(h, i)` | `mul_ftz(d8, float(sum))` |
| `P = DP4A(...)` (`:129-138`) | `IDP.4A.S8.S8` chains | exact integer | `dp4a` |
| `acc += sg * (xdd[g] * (float) P - 1.5f * xs[g])` (`:140`) | `FMUL.FTZ R51, R40, 1.5`; `FFMA.FTZ R44, R44, R49, -R51`; `FFMA.FTZ R44, R37, R44, RZ`, then `FFMA.FTZ R37, R37, R45, R44` | `FFMA.FTZ(scale, FFMA.FTZ(xdd, P, -FMUL.FTZ(xs, 1.5)), acc)` | `fma_ftz(scale, fma_ftz(xdd, float(P), -mul_ftz(xs, 1.5f)), acc)` from 0, j = 0..9, h = 0, 1 |
| `red[0] + red[1] + red[2] + red[3]` (`:146`) | `FADD.FTZ R5, R2, R5`; `FADD.FTZ R4, R5, R4`; `FADD.FTZ R4, R4, R7` | left to right | `add_ftz` ×3 |
| `m = fmaxf(m, shfl)` (`:150`) | `FMNMX.FTZ R10, R7, R10, !PT`, offsets 16..1 | `FMNMX.FTZ` | `max_ftz` butterfly |
| `e = v > -inf ? __expf(v - m) : 0` (`:151`) | `FADD.FTZ R11, R7, -R10`; `FMUL.FTZ R11, R11, 1.4426950216293334961`; `MUFU.EX2 R11, R11`; `FSETP.NE.FTZ.AND P4, PT, R7, -INF, PT`; `FSEL R7, R11, RZ, P4` | `MUFU.EX2(FMUL.FTZ(log2 e, FADD.FTZ(v, -m)))` | `expf_ftz(add_ftz(v, -m))` |
| `e += shfl` (`:153`) | `FADD.FTZ R16, R7, R16`, offsets 16..1 | butterfly | `warp_sum_ftz` |
| the kPer argmax rounds (`:155-166`) | `FSETP.GEU.FTZ` / `FSETP.NEU.FTZ` and `FSEL` | exact | `ov > bv \|\| (ov == bv && oi < bi)` |

### 4.3 `q5k_proxy_final_kernel` (`q5k_head.cu:464-496`) with `collect` and `pick`

13.0's listing of this kernel has the same operations at every site. It differs from 13.3's in scheduling, in the
predicate registers its `FSEL`s test, and in the tracer's leaf classes where a register was loaded outside the traced
block.

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `m = fmaxf(m, cand[i].x)`, butterfly, `wmax` (`head_proxy.cuh:185-191`) | `FMNMX.FTZ` chains | exact, order-free | `max_ftz` |
| `c.x >= m - kDelta` (`:199`) | `FADD.FTZ R11, R18, -5`; `FSETP.GE.FTZ.AND P1, PT, R8, R11, PT` | `FADD.FTZ(m, -5)` | `add_ftz(m, -5.0f)` |
| `atomicAdd(&f.cnt, 1)` (`:200`) | `VOTEU.ANY UR8, UPT, PT`; `FLO.U32 R13, UR8`; `POPC R12, UR8`; `POPC R15, R14` (the mask and `SR_LTMASK`); `@P0 ATOMS.ADD R12, [UR4+0x340], R12`; `SHFL.IDX PT, R10, R12, R13, 0x1f`; `IADD R10, R10, R15` | per warp request: one shared-memory add of the requesting lanes' count by the highest of them, each lane at the base plus its rank among them | the port's fixed order, below |
| rescoring (`q5k_head.cu:488-492`) | `q5k_row_dev<1, 10>`'s sequence (section 1) | the head's bits | `q5k_row<1>` |
| `sum += p.y * __expf(p.x - bv)` where `p.y > 0` (`:228`) | `@P0 FADD.FTZ R0, -R3, R10`; `@P0 FMUL.FTZ R0, R0, 1.4426950216293334961`; `@P0 MUFU.EX2 R0, R0`; `@P0 FFMA.FTZ R4, R11, R0, R4` | `FFMA.FTZ(p.y, e, sum)` | `fma_ftz(p.y, expf_ftz(add_ftz(p.x, -bv)), sum)` |
| `sum += __expf(ex - bv) - __expf(lp - bv)` (`:230`) | `MUFU.EX2` pairs; `FADD.FTZ R7, R8, -R7`; `FADD.FTZ R7, R7, R4` | `FADD.FTZ(sum, FADD.FTZ(e_ex, -e_lp))` | `add_ftz(sum, add_ftz(expf_ftz(...), -expf_ftz(...)))` |
| `sum += shfl` (`:232`) | `FADD.FTZ R3, R3, R4`, offsets 16..1 | butterfly | `warp_sum_ftz` |
| `sum = fmaxf(sum, one)`; `1.0f / sum` (`:236-238`) | `FMNMX.FTZ R0, R0, R3, !PT`; `MUFU.RCP R9, R0` | | `rcp_ftz(max_ftz(sum, one))` |

The list's order. basalt's `collect` appends every qualifying candidate with `atomicAdd(&f.cnt, 1)` on shared memory,
and nvcc compiles it as a warp-aggregated atomic: within one warp's request the slots go to the requesting lanes in
lane order, and the requests take their slots in the order they reach the shared-memory unit, which CUDA leaves
unspecified (each warp's requests in its own program order: its loop's candidates `i`, `i + 512`, ...). The order
matters: `pick` gives lane L the listed rows L and L + 32, so the order sets which lane adds which exact-minus-
approximate term before the butterfly, and so the probability's last bits; and when more than 64 candidates qualify,
it decides which 64 are rescored, and so possibly the pick. The scoping report's inference (section 7.4) that on CUDA
these vary from run to run follows. A threadgroup atomic would port the mechanism with documented Metal semantics, but
its order is just as unspecified, so neither the tests nor repeated runs could rely on the bits. The port fixes the
arrival order to ascending candidate index: pass k of the 512-strided loop, SIMD-groups 0..15 in order, lanes in order,
each slot `cnt + (the counts of the SIMD-groups before) + simd_prefix_exclusive_sum(take)` (the lane's rank among the
requesting lanes, as `POPC(mask & SR_LTMASK)`), the counts through threadgroup memory between two barriers per pass.
That is one of the orders basalt's kernel produces (every warp's request of pass k arriving before any of pass k + 1,
in warp order), with basalt's arithmetic on it; the list, the rescored set past 64 candidates and the probability then
repeat run to run, and the emulation checks them bit for bit. On the test's 400 trials, between 1 and 78 candidates
qualified (12.8 on average); the trials above 64 were checked bit for bit like the others.

Every value these kernels compare (`FSETP.*.FTZ`, `FMNMX.FTZ`) is the result of an `.FTZ` operation or a stored fp32
value written by one, so it is never subnormal, and the port's plain comparisons are the flushed ones.
