# basalt's dense_tc kernels as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of `dense_tc` and `dense_tc_pair`, the window projections basalt
runs for every window width on this pack's decode path, read from the sm_120 PTX and SASS, and the Metal form the port
writes at each site. This page is the authority the Metal kernels follow (`docs/design/flash-next.md`, invariant 5),
as `matvec.md` is for `native_mmvq` and the q8_1 quantizers: each weight type's fp32 terms (`half_sb_terms`,
`dense_tc.cu:100-330`), every combining step of each summation order, and the rule that picks the order.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last.

## Method

- **Sources.** `src/kernels/cuda/dense_tc.cu` with the headers it includes (`include/basalt/kernels/dense_tc.hpp`,
  `pdl.hpp`, and `include/basalt/ggml/ggml_common.h` under `GGML_COMMON_DECL_CUDA` / `GGML_COMMON_IMPL_CUDA`). Only
  basalt's public MIT source was sent.
- **Compilers.** Compiler Explorer's public API, as in `matvec.md`: NVCC 13.0.0 (`nvcc130`) and 13.3.0 (`nvcc133`),
  PTX and the `SASS (sm_120)` listing.
- **Flags.** `dense_tc.cu` is not in the `--use_fast_math` list (`CMakeLists.txt:185-202`), so it builds with nvcc's
  defaults (IEEE division, no flush to zero, contraction on): `-O3 -DNDEBUG -DBASALT_VERSION="0.0.1"
  --generate-code=arch=compute_120,code=[compute_120,sm_120]`, exactly, in both versions.
- **Translation units.** Each unit is basalt's file verbatim through the end of its device code (`dense_tc.cu:1-960`:
  the includes, `Fmt`, the helpers, `half_sb_terms`, the four kernel templates and `dense_tc_l2_kernel`), local
  includes inlined under `#line` directives, then a table that takes the address of one kernel instantiation. For each
  of the six types of this pack (Q4_K 12, Q5_K 13, Q6_K 14, IQ4_XS 23, Q3_K 11, Q2_0 42): `dense_tc_one_kernel<T, 24>`,
  `dense_tc_tile_kernel<T>`, `dense_tc_kernel<T, 3, 1, 1>` and `dense_tc_kernel<T, 3, 1, 2>`, the four kernels the
  variant rule can launch (`dense_tc.cu:1040-1059`); and `dense_tc_pair_kernel<true, DB, LQ>` for DB and LQ each false
  and true (the decode path's n_in 2560 always takes the register-fragment build, `xr`, `:1126`).
- **The pair kernel's listing.** Compiler Explorer's device listing stops at 5,000 lines (`[truncated; too many
  lines]`), and so does its PTX assembler's. `dense_tc_pair_kernel` holds all seven types' branches and runs past it;
  its PTX arrives whole (6,098 to 7,866 lines). So each pair unit adds one entry point that calls basalt's
  `dense_tc_pair_block<true, DB, LQ>` verbatim with both matrices' type set to one type T: the kernel as it runs on a
  tile of type T, its type switch (`:870-906`) folded to that branch. Units: the four types of the non-LQ build and the
  six of the LQ build, at DB false and true.
- **Reading a site.** As in `matvec.md`: the PTX shows what the front end fused (`fma.rn.f32`); the SASS shows what
  `ptxas` made of the rest. A PTX `mul.f32` / `sub.f32` without a rounding modifier may still be fused by `ptxas`, which
  it does in Q4_K and Q5_K below.
- **Version check.** The dataflow tracer of `matvec.md` (each SASS float result printed as an expression template over
  the leaves `h0` / `h1`, the low or high fp16 half of a word converted by `HADD2.F32`; `i` an int32 converted by
  `I2FP`, marked by its writer: `imma` the tensor-core dot, `ishfl` a shuffled integer, `iother` any other integer
  arithmetic; `shm` a shared-memory load; `in` a value from before the basic block; `0` the zero register) gives the
  same multiset of templates in 13.0 and 13.3 for 43 of the 44 comparable units (all 24 single-type kernels but one,
  all 20 per-type pair units). The exception, `dense_tc_kernel<42, 3, 1, 1>`, differs only in where a d8 load sits:
  13.0 places it before a branch, so its leaf reads `in` instead of `h0`; with the loaded leaves (`h0`, `h1`, `in`)
  merged into one class the multisets are equal.

## Notation

As in `matvec.md`: fl(e) rounds once to fp32, to nearest even; fma(a, b, c) rounds a·b + c once; h(x) is an fp16 value
converted to fp32 (exact); f(n) an int32 converted to fp32. dot is the exact int32 dot of a 32-value block (or a
16-value half) of weight codes with the 32 (16) q8_1 codes of a column, which `mma.sync` computes for 16 rows by 8
columns; d8 is the column's q8_1 scale, h(the low half of the block's `ds`). Every integer converted here is below
2^24 = 16,777,216 in magnitude, so f(n) is exact. The largest is Q6_K's sc0·e0 + sc1·e1, int8 scales (|sc| ≤ 128)
on 16-value dots of codes in [−32, 31]: at most 2·128·16·32·127 = 16,646,144; next is IQ4_XS's dot·(ls − 32), at
most 32·127·127·32 = 16,516,096.

**No flush to zero.** No float add, multiply or fma in any unit carries `.FTZ` (the only `.FTZ` instructions are
`F2I.FTZ.U32.TRUNC.NTZ` in the index arithmetic), and the PTX has no `.ftz`: every float operation is IEEE's.

**No subnormal can occur.** This GPU flushes subnormal inputs and results of add, multiply and fma in
`MTLMathModeSafe` (`tests/kernels/test_basalt_platform.py`), where these IEEE instructions keep them; no value here is
ever subnormal. Each weight scale (d, dmin, dd, dw) and each d8 is an fp16 value, an integer multiple of 2^-24. The
products with integers (f1, f2, d8·f(n)) and every fma or add of them are then integer multiples of 2^-24 (inner) or
2^-48 (part, acc, the warp sums and the output), and a sum of multiples of 2^-K rounds to a multiple of 2^-K: below
2^(24−K) such a value is exact, above it the fp32 grid spacing is itself a multiple of 2^-K. So every result is 0 or at
least 2^-48 in magnitude, and Metal's `*`, `+` and `fma()` give the IEEE result.

## Summary

| Type | Per block i of a half super-block, per (row, column) as nvcc 13.x emits it (sm_120) | Metal |
|---|---|---|
| Q4_K, Q5_K | f1 = fl(dm.x·f(sc)), f2 = fl(dm.y·f(m)); inner = fma(f1, f(dot), −fl(f2·f(isum))); acc = fma(inner, d8, acc) | the same, `*` and `fma()` |
| Q6_K, Q3_K | part = fma(f(sc0·dot0 + sc1·dot1), d8, part) from +0, i = 0..3; then acc = fma(dd, part, acc) | the same |
| IQ4_XS | part = fma(f(dot·ls), d8, part) from +0, i = 0..3; then acc = fma(dw, part, acc) | the same |
| Q2_0 | per Q2_0 block b = 0, 1: part = fma(f(dot), d8, part) from +0 over its two 32-value blocks; then acc = fma(dw_b, part, acc) | the same |

| Order | Warps a tile | Warp w = 4g + hb | Final sum per (row, column), from the SASS |
|---|---|---|---|
| one | 24 | half super-block hb of the chunks c ≡ g (mod 6) | ((((((w0 + w4) + w8) + w12) + w16) + w20) + (((((w1 + w5) + w9) + w13) + w17) + w21)) + ((((((w2 + w6) + w10) + w14) + w18) + w22) + (((((w3 + w7) + w11) + w15) + w19) + w23)) |
| tile, pair | 8 | the chunks c ≡ g (mod 2) | ((w0 + w4) + (w1 + w5)) + ((w2 + w6) + (w3 + w7)) |
| s, KS 1 | 4 | every chunk | ((w0 + w1) + w2) + w3 |
| s, KS 2 | 8 | the 512-value chunks c ≡ g (mod 2) | (((w0 + w1) + w2) + w3) + (((w4 + w5) + w6) + w7) |

A warp's sum w is its lanes' acc after its chunks in ascending order, from +0. Every add of the tables is a plain IEEE
`FADD` in both versions. The `[H]` of the scoping report (which product of Q4_K's and Q5_K's `a·b − c·d` nvcc fuses) is
settled by the SASS: the d product, as in `native_mmvq` (`matvec.md` section 5).

## 1. The terms per type (`half_sb_terms`, `dense_tc.cu:100-330`)

Each lane holds four (row, column) accumulators acc[q], for rows g, g + 8 and columns 2t, 2t + 1 (lane = 4g + t), and
runs the four blocks of its half super-block in ascending order. The raw lines are from `dense_tc_tile_kernel<T>`, the
first instance of each site; the other kernels and the pair units have the same templates (Method).

### Q4_K (`:106-156`; `Fmt<12>`, 144-byte blocks)

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| f1 | `:143` `dm[h].x * (float) sc` | `cvt.rn.f32.u32`; `mul.f32 %r40, %r138, %r161` | `FMUL R64, R35, R22` | `FMUL R66, R39, R22` | `dm.x * float(sc)` |
| f2 | `:144` `dm[h].y * (float) m` | `mul.f32 %r181, %r139, %r41` | `FMUL R63, R38, R21` | `FMUL R63, R38, R63` | `dm.y * float(m)` |
| inner | `:155` `f1 * (float) d[q] - f2 * (float) isum` | `mul.f32 %r194, %r40, %r193`; `mul.f32 %r196, %r181, %r195`; `sub.f32 %r197, %r194, %r196` | `FMUL R71, R63, R68`; `FFMA R47, R64, R47, -R71` | `FMUL R71, R63, R68`; `FFMA R47, R64, R47, -R71` | `fma(f1, float(dot), -(f2 * float(isum)))` |
| acc | `:155` `acc[q] += x.d8[i][q & 1] * (...)` | `fma.rn.f32 %r42, %r449, %r197, %r482` | `FFMA R47, R47, R61, R42` | `FFMA R47, R47, R61, R42` | `fma(inner, d8, acc)` |

`h0` / `h1` trace the operands: f1 takes the `.H0_H0` half of the `dm` word (d), f2 the `.H1_H1` half (dmin), and
the template is `FFMA(FFMA(FMUL(h0, i), imma, -FMUL(FMUL(h1, i), ishfl)), h0, acc)` in both versions. The front end
fuses only the outer `+=` (PTX: two multiplies and a subtract); `ptxas` rounds the dmin product and fuses the d product
with its negation, the choice it makes in `native_mmvq`'s Q4_K and Q5_K returns (`matvec.md` section 5). isum is the
column's code sum (`:149-152`): `__dp4a` with `0x01010101`, two XOR shuffles and an indexed shuffle, all integer.

### Q5_K (`:106-156`; `Fmt<13>`, 176-byte blocks)

Q4_K's arithmetic line for line, the fifth bit ORed into the codes (`:131-134`) before the dot: `FMUL R69, R40, R69`
(f1) and `FMUL R70, R41, R72` (f2) in 13.0, `FMUL R44, R70, R45` (the dmin product), `FFMA R75, R69, R4, -R44`
(inner), `FFMA R22, R75, R67, R22` (acc); 13.3 `FMUL R5, R69.reuse, R72`, `FFMA R5, R68.reuse, R77, -R5`, `FFMA R22,
R5, R67.reuse, R22`. Metal: Q4_K's forms.

### Q6_K (`:157-195`; `Fmt<14>`, 210-byte blocks)

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| part, i = 0..3 | `:192` `part[q] += x.d8[i][q & 1] * (float) (sc0 * e0 + sc1 * e1)` | `fma.rn.f32 %r261, %r457, %r260, 0f00000000`, then `fma.rn.f32 %r301, %r461, %r300, %r261`, ... | `FFMA R50, R51, R44.reuse, RZ`; `FFMA R42, R42, R44, R51`; `FFMA R17, R17, R53, R50`; `FFMA R26, R26, R46, R15` | `FFMA R50, R50, R49.reuse, RZ`; `FFMA R47, R47, R41.reuse, R22`; `FFMA R47, R20, R40, R47`; `FFMA R28, R28, R42.reuse, R47` | `part = fma(float(sc0 * e0 + sc1 * e1), d8, part)` from +0 |
| acc | `:195` `acc[q] += dd[q >> 1] * part[q]` | `fma.rn.f32` | `FFMA R28, R13.reuse, R17, R28` | `FFMA R36, R9, R28, R36` | `fma(dd, part, acc)` |

e0 and e1 are the two 16-value halves' dots (`mma.sync ... m16n8k16`, `:188-189`); sc0·e0 + sc1·e1 is integer
arithmetic. Template: `FFMA(h0, FFMA(i, h0, FFMA(i, h0, FFMA(i, h0, FFMA(i, h0, 0)))), acc)`.

### Q3_K (`:196-234`; `Fmt<11>`, 110-byte blocks)

Q6_K's sequence with the codes `__vsub4(...) - 4` and the scales sc − 32 (`:219-224`): 13.0 `FFMA R29, R20,
R17.reuse, RZ`, `FFMA R29, R22, R27, R29`, `FFMA R44, R44, R16, R29`, `FFMA R19, R19, R17.reuse, R44` (part), `FFMA
R32, R49, R21, R32` (acc); 13.3 `FFMA R22, R49, R16.reuse, RZ` ... `FFMA R37, R9.reuse, R30, R37`. PTX `fma.rn.f32`
at every site. Metal: Q6_K's forms.

### IQ4_XS (`:297-329`; `Fmt<23>`, 136-byte blocks)

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| part, i = 0..3 | `:325` `part[q] += x.d8[i][q & 1] * (float) (d[q] * ls[q >> 1])` | `fma.rn.f32 %r206, %r432, %r205, 0f00000000`, then `fma.rn.f32 %r260, %r436, %r259, %r206`, ... | `FFMA R52, R17, R39, RZ`; `FFMA R51, R51, R53, R16`; `FFMA R51, R16, R21, R51`; `FFMA R20, R20, R16.reuse, R51` | `FFMA R16, R16, R39, RZ`; `FFMA R46, R45, R53, R16`; `FFMA R45, R22, R38, R45`; `FFMA R23, R23, R21.reuse, R46` | `part = fma(float(dot * ls), d8, part)` from +0 |
| acc | `:328` `acc[q] += dw[q >> 1] * part[q]` | `fma.rn.f32` | `FFMA R12, R33.reuse, R20, R12` | `FFMA R12, R17, R23, R12` | `fma(dw, part, acc)` |

The codes come from `iq4_lookup` (`:79-84`), the codebook's `__byte_perm` lookup; ls = the 6-bit scale minus 32.

### Q2_0 (`:235-270`; `Fmt<42>`, four 18-byte blocks per 256 values)

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| part, k = 0, 1 | `:266` `part[q] += x.d8[i][q & 1] * (float) d[q]` | `fma.rn.f32 %r183, %r374, %r182, 0f00000000`, then `fma.rn.f32 %r223, %r378, %r222, %r183` | `FFMA R40, R12, R29.reuse, RZ`; `FFMA R29, R29, R14.reuse, R40` | `FFMA R43, R12, R29.reuse, RZ`; `FFMA R31, R16, R14, R43` | `part = fma(float(dot), d8, part)` from +0 |
| acc, per Q2_0 block | `:269` `acc[q] += dw[q >> 1] * part[q]` | `fma.rn.f32` | `FFMA R23, R17.reuse, R16, R23`, then `FFMA R22, R13, R12, R22` | `FFMA R24, R15.reuse, R31, R24`, then `FFMA R26, R13.reuse, R14, R26` | `fma(dw, part, acc)` |

A half super-block holds Q2_0 blocks 2n and 2n + 1 (`:238`); part restarts at +0 for each. The codes are
`__vsub4((q | q << 6 | q << 12 | q << 18) & 0x03030303u, 0x01010101u)` (`:261`), code − 1.

## 2. The summation orders and their warp-sum trees

A tile is 16 rows; each warp leaves its lanes' acc in shared memory as `red[warp][row][column]` (16 · 8 floats, 0x200
bytes a warp), and one thread per (row, column) adds them (`dense_tc.cu:407-427`, `:552-560`, `:667-681`,
`:919-928`). The SASS loads each warp's value at the offset (warp · 0x200 + base); mapping the offsets back to warps
gives the trees of the Summary in both versions and for all six types (raw lines, Q6_K):

| Order | SASS 13.0 | SASS 13.3 |
|---|---|---|
| one (`dense_tc_one_kernel<14, 24>`) | `FADD R6, R6, R15`; ... `FADD R23, R6, R23`; `FADD R23, R4, R23`; `FADD R43, R10, R43`; `FADD R8, R8, R43`; `FADD R23, R23, R8`: four chains h_q = w_q + w_4+q + ... + w_20+q (24 loads from base 0x80), then (h0 + h1) + (h2 + h3) | the same lines |
| tile (`dense_tc_tile_kernel<14>`) | `LDS R12, [R6+0x80]`, `LDS R13, [R6+0x880]`, ...; `FADD R12, R12, R13` (w0 + w4); `FADD R15, R14, R15` (w1 + w5); `FADD R12, R12, R15`; `FADD R16, R16, R17` (w2 + w6); `FADD R19, R18, R19` (w3 + w7); `FADD R19, R16, R19`; `FADD R19, R12, R19` | `FADD R8, R8, R9`; `FADD R17, R16, R17`; `FADD R8, R8, R17`; `FADD R18, R18, R19`; `FADD R21, R20, R21`; `FADD R21, R18, R21`; `FADD R21, R8, R21` |
| pair (per-type unit, Q6_K, LQ build) | `FADD R32, R32, R35`; `FADD R37, R34, R37`; `FADD R32, R32, R37`; `FADD R36, R36, R39`; `FADD R41, R38, R41`; `FADD R41, R36, R41`; `FADD R41, R32, R41` (loads at 0x80, 0x880, 0x280, 0xa80, ...) | the same tree |
| s, KS 1 (`dense_tc_kernel<14, 3, 1, 1>`) | `LDS R13, [R76+UR6+0x5700]` (w0), `0x5900` (w1), `0x5b00` (w2), `0x5d00` (w3); `FADD R13, R13, R14`; `FADD R13, R13, R16`; `FADD R13, R13, R18` | `FADD R17, R16, R17`; `FADD R18, R17, R18`; `FADD R19, R18, R19` |
| s, KS 2 (`dense_tc_kernel<14, 3, 1, 2>`) | loads w0..w7 from 0xa500; `FADD R8, R8, R9`; `FADD R10, R10, R15`; `FADD R8, R8, R11`; `FADD R10, R10, R17`; `FADD R13, R8, R13`; `FADD R10, R10, R19`; `FADD R13, R13, R10` | the same lines |

For KS 2, `v = q == 0 ? vq : v + vq` (`:423`) is the last add: the first group's sum plus the second's. A warp whose
chunks are all past nchunk (the one order when n_in / 512 < 6) leaves acc at +0 and its zero takes part in the adds,
as in basalt.

What each warp sums, from the source (the order of the per-warp loop is not a compiler choice):

- one (`:589-595`, `:639`, `:653`): warp w takes hb = w % 4 and the 512-value chunks c = w / 4, w / 4 + 6, ... (NG = 6).
- tile (`:461`, `:519`) and pair (`pair_terms_xr`, `:796-801`): hb = w & 3, chunks w >> 2, w >> 2 + 2, ...
- s (`:343-346`, `:380`): hb = w & 3; KS 1: every chunk; KS 2: the 1024-value chunk c's half ks = w / 4, which is the
  512-value chunk 2c + ks, so warp w takes the same chunks as the tile order's warp w, in the same order.

## 3. The variant rule (`dense_tc.cu:1017-1059`)

`launch<TYPE>` picks the order from the shape, the weights' alignment and the device's multiprocessor count, never
the column count:

1. one, when the weights are 16-byte aligned, a row is a whole number of 16-byte units ((n_in / 256)·SB % 16 == 0),
   the 16-row tiles number at most the multiprocessors, and `one_smem_bytes<TYPE, 24>(n_in)` = 128 + max(24·16·8·4,
   16·one_row_stride(row bytes)) ≤ 99 · 1024 (`one_ok`, `:1018-1022`; `one_row_stride` rounds the row up to an odd
   number of 16-byte units, `:52-54`);
2. else tile, when aligned and two staged tiles fit: 2·16·(n_in / 256)·SB ≤ 90 · 1024 (`:1051`);
3. else s with KS 2 when n_in % 1024 == 0 and the tiles number fewer than the multiprocessors (`:1055-1056`);
4. else s with KS 1.

`dense_tc_pair` always takes the pair kernel (the tile order; `:1107-1146`), for every type pair it supports.

The multiprocessor count is the device's (`sm_count()`, `:967-975`), so on basalt's hardware the order, and with it
the bits, follow the GPU model. The port fixes it at 170, the count of basalt's measured RTX 5090
(`docs/precision.md:6`; 170 SMs per KitGuru's review, quoted in the scoping report). At 170, every native weight is
its own `cudaMalloc` (aligned, `native_dense.cpp:215-216`), and this pack's projections take:

| Projection | n_in → n_out | Tiles | Order at 170 | Why |
|---|---|---|---|---|
| ssm_out, attn_output | 6144 → 2560 | 160 | one | rows of 24·SB bytes are whole 16-byte units for every type; `one_smem_bytes` 28,032 (Q2_0) to 80,768 (Q6_K) for the six types |
| attn_q | 2560 → 12288 | 768 | tile | 768 > 170; two tiles take 23,040 (Q2_0) to 67,200 (Q6_K) bytes of the six types |
| GDN qkv + z | 2560 → 10240 + 6144 | | pair | `dense_tc_pair` |

## 4. Metal

`kernels/common/basalt_dense_tc.metal` holds `basalt_dense_tc` (the one, tile and s orders, chosen by macro) and
`basalt_dense_tc_pair`. Each starts with `#pragma STDC FP_CONTRACT OFF` (through `basalt_fp.metal`) and writes
`fma()` exactly at the `FFMA` sites above, plain `*` at the `FMUL` sites and `+` at the `FADD` sites; nothing
flushes, because no subnormal reaches them.

- **The integer dots.** `mma.sync`'s only role is the exact int32 dot of a block (or half block) for 16 rows by 8
  columns. Lane 4g + t holds the same fragments as on CUDA; it fetches column c's code words for its positions
  4t.. and 16 + 4t.. from lane 4c + t (`simd_shuffle`), forms its partial dots for rows g and g + 8 against all eight
  columns with `dp4a`, and the four lanes of group g add their partials (integer adds, exact in any order), each lane
  keeping columns 2t and 2t + 1: the D fragment basalt's lane holds.
- **Geometry.** A threadgroup is one 16-row tile with basalt's warps as SIMD-groups (24, 8, 4 or 8), so warp w's
  chunks and the tree are basalt's; the pair's threadgroups take the first matrix's tiles, then the second's.
- **Staging.** Basalt stages whole tiles in up to 99 KiB of shared memory; Apple9 has 32 KiB of threadgroup memory.
  Staging carries no arithmetic, so the Metal kernels read the weights from device memory (two 16-bit loads a word:
  Q6_K, Q3_K and Q2_0 blocks are only 2-byte aligned) and keep only the warp sums in threadgroup memory. Rows past
  n_out read row n_out − 1 (basalt reads stale staging bytes there) and are never stored; columns at or past ncols get
  zero codes and d8 = 0, as in basalt (`:381-393`), and are never stored.

`tests/kernels/test_basalt_dense_tc.py` checks every order bit for bit against the numpy emulation of this page
(`tests/kernels/basalt_emu.py`: `dense_tc`), ports `dense_tc_parity`'s cases and pairs, and adds a float64 reference
per type.
