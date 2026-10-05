# GEMV kernel study — the M1 harness on the M5 Pro

Status: measured 2026-09-24 on the M5 Pro (20-core GPU, 24 GB, macOS 26.5.1, AC power) with
[`tools/bench/gemv_bench.py`](../../tools/bench/gemv_bench.py) (plan M1, issues #9 and #10). Every point is min-of-3
over ≥ 2 GB streamed, checked against the exact format oracle (≤ 2 BF16 ULP at the output's magnitude, accumulation
noise < 1e-4); the raw JSON lines are in [the archived results](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results). Nominal = 307 GB/s.

## 1. The kernel

`kernels/common/gemv_T.metal`: static slices over the block-lane-major pack (design D8), 16-byte weight loads in either lane
order, the format plugin's decode snippet, in-word block scales, the per-row tensor scale from the pack, BF16
activations converted once per word and reused across `RG` rows, FP32 accumulation, one `simd_sum` per (row, token).
Knobs: rows per block `R`, tokens `T`, row group `RG`, lane order, and the geometry — the crew (12 SIMD-groups per
core, `×n` = n threadgroups of 384 per core) or one block per SIMD-group in small threadgroups (the MLX/llama.cpp
shape, "1blk/SG tg64").

## 2. M1 sweep (reference NVFP4 decode, R = 16, RG = 2) — `apple-m5-pro-20c_gemv_m1.jsonl`, 378 points

Best geometry per shape and T; the crew ×1 number in the last column:

| format | shape | T = 1 | T = 2 | T = 4 | crew ×1, T = 1 |
|---|---|---|---|---|---|
| FP8 | 17408×5120 (gate/up) | 260 GB/s (85 %) | 248 (81 %) | 109 (36 %) | 254 |
| FP8 | 5120×17408 (down) | 279 (91 %) | 238 (77 %) | 124 (41 %) | 201 |
| FP8 | 10240×5120 (GDN qkv) | 277 (90 %) | 234 (76 %) | 118 (39 %) | 250 |
| FP8 | 12288×5120 (q + gate) | 264 (86 %) | 255 (83 %) | 104 (34 %) | 229 |
| FP8 | 6144×5120 (z) | 231 (75 %) | 255 (83 %) | 89 (29 %) | 231 |
| FP8 | 5120×6144 (o / out) | 273 (89 %) | 242 (79 %) | 124 (40 %) | 196 |
| FP8 | 248320×5120 (lm_head) | 291 (95 %) | 288 (94 %) | 125 (41 %) | 274 |
| NVFP4 | 17408×5120 | 157 (51 %) | 110 (36 %) | 78 (26 %) | 118 |
| NVFP4 | 5120×17408 | 143 (47 %) | 110 (36 %) | 80 (26 %) | 88 |
| NVFP4 | 248320×5120 | 184 (60 %) | 127 (41 %) | 89 (29 %) | 131 |
| INT4 affine | 17408×5120 | 242 (79 %) | 136 (44 %) | 88 (29 %) | 201 |
| INT4 affine | 5120×17408 | 255 (83 %) | 151 (49 %) | 96 (31 %) | 158 |
| INT4 affine | 10240×5120 | 231 (75 %) | 144 (47 %) | 93 (30 %) | 196 |
| INT4 affine | 12288×5120 | 243 (79 %) | 131 (43 %) | 85 (28 %) | 180 |
| INT4 affine | 6144×5120 | 247 (80 %) | 114 (37 %) | 71 (23 %) | 179 |
| INT4 affine | 5120×6144 | 254 (83 %) | 153 (50 %) | 99 (32 %) | 159 |
| INT4 affine | 248320×5120 | 261 (85 %) | 158 (51 %) | 97 (32 %) | 220 |

Also at the crew geometry, T = 1, 17408×5120: BF16 284 GB/s (93 %), INT8 266 GB/s (87 %).

The INT4 affine rows (format 2, #47, 2026-09-24, the same sweep: 126 points, every point ≤ 1 ULP of the oracle) are
the plugin's own decode — nibble → float, one (scale, bias) pair per group of 64 (FP32 when this was measured, the
checkpoint's 16-bit dtype — BF16 or F16 — since #113: 0.5625 bytes per weight instead of 0.625), the bias folded in as `bias · Σx` per group. At T = 1 it
streams at 75–85 % of nominal, a third above NVFP4's
LUT decode (47–60 %) and within 10 % of FP8; the best geometry at T = 1 was one block per SIMD-group on every shape
(the crew ×1 column is 158–220). At T = 2 and 4 it falls to the ALU bound like the other formats (37–51 %, 23–32 %).
MLX's `quantized_matmul` in affine-4 (§3c's baseline file, `affine4_g64`) streams the same shapes at 247–285 GB/s
(82–93 %) at T = 1: ours is 0.85–0.98× MLX — the same gap as NVFP4, the same T ≥ 2 remedy (M9).

## 3. NVFP4 decode study — `apple-m5-pro-20c_nvfp4_decode.jsonl`

Three exact decodes of the E2M1 nibbles (`monolith/formats/nvfp4.py`, macro `NVFP4_DECODE`):

* **V0** — per nibble, float bit construction with two selects (the `p13` decode; ~14 ALU ops per weight);
* **V1** — nibble pairs decoded into a `half2` with packed 16-bit integer arithmetic (both halves of a 32-bit word
  at once), then converted to `float2`;
* **V2** — the eight magnitudes as small integers (`value × 2`) in one 32-bit constant, four bits each, an
  `int → float` conversion and a sign select; the `× 0.5` folds into the block scale.

T = 1, `interleaved16`:

| shape | geometry | V0 | V1 | V2 |
|---|---|---|---|---|
| 17408×5120 | crew ×1, R = 16 | 118 GB/s (39 %) | 150 (49 %) | 161 (52 %) |
| 17408×5120 | crew ×4, R = 16 | 146 (47 %) | 179 (58 %) | 211 (69 %) |
| 17408×5120 | 1blk/SG tg64, R = 8 | 168 (55 %) | 203 (66 %) | 243 (79 %) |
| 17408×5120 | 1blk/SG tg64, R = 4, RG = 4 | — | — | **252 (82 %)** |
| 5120×17408 | 1blk/SG tg64, R = 4, RG = 4 | — | — | 226 (74 %) |
| 248320×5120 | 1blk/SG tg64, R = 4 | — | 246 (80 %) | **273 (89 %)** |

T > 1 does not benefit from the decode (V2, 1blk/SG: T = 2 130–151 GB/s, T = 4 89–99): once the decode is cheap the
FMAs per weight dominate, which is the accelerator path's territory (design §5.6, `p14`).

## 3b. The remaining knobs (#11) — `apple-m5-pro-20c_kernel_knobs.jsonl`, 108 points

* **Rows per block at T = 1** (one block per SIMD-group / crew ×1, GB/s): FP8 R = 2: 287/273, 4: 287/270, 8:
  285/259, 16: 275/259, 32: 280/222; NVFP4 (V2) R = 2: 256/175, 4: 249/169, 8: 242/163, 16: 225/161, 32: 231/136;
  INT8 R = 2: 282/281 … 32: 281/243; BF16 flat at 287–293. Small blocks win for the decoded formats; the crew ×1
  geometry pays 5–10 % on FP8 and ~30 % on NVFP4, more at large R (tail quantization: 17408/32 = 544 blocks over
  240 SIMD-groups is 76 % efficient).
* **T > 1 on the shader ALUs** (best of R ∈ {4, 8, 16} × RG ∈ {2, 4, 8}, one block per SIMD-group): FP8 T = 2 278
  GB/s (91 %), T = 4 139 (45 %), T = 8 88 (29 %); NVFP4 T = 2 156 (51 %), T = 4 102 (33 %), T = 8 19 (6 %; register
  spills). RG = 4 is the right row group for T ≥ 2 (the M1 sweep's RG = 2 was 20 % worse at T = 4).
* **`safe` vs `fast` math**: identical throughput (284.6 vs 284.7 GB/s FP8; 251.9 vs 251.7 NVFP4) and identical
  outputs — the kernel has no transcendental or division; stay in `safe`.

## 3c. Against MLX on the same shapes (#12) — `apple-m5-pro-20c_mlx_baseline.jsonl`

MLX 0.32.2 `quantized_matmul` (`mode="nvfp4"`, group 16: the same bytes per weight as our pack; affine 4-bit group 64;
BF16 matmul), wall-clock over ≥ 2 GB of weight copies, min-of-5 (`tools/bench/mlx_baseline.py`; the table is printed
by `tools/bench/m1_gate_table.py`):

| shape | T | ours FP8 | ours NVFP4 (V2) | MLX nvfp4 | MLX affine-4 | MLX bf16 |
|---|---|---|---|---|---|---|
| 248320×5120 | 1 | 291 (95 %) | 273 (89 %) | 286 (93 %) | 285 | 288 |
| 17408×5120 | 1 | 287 (94 %) | 256 (83 %) | 281 (91 %) | 279 | 272 |
| 5120×17408 | 1 | 279 (91 %) | 226 (74 %) | 284 (92 %) | 282 | 287 |
| 17408×5120 | 2 | 278 (91 %) | 156 (51 %) | 280 (91 %) | 279 | 282 |
| 17408×5120 | 4 | 139 (45 %) | 102 (33 %) | 262 (85 %) | 253 | 285 |
| 6144×5120 | 4 | 89 (29 %) | 62 (20 %) | 245 (80 %) | 235 | 284 |

(The narrow-shape NVFP4 rows at T = 1 are being re-measured with V2; the table in `m1_gate_table.py` is the record.)

Readings:

1. **T = 1, NVFP4: 0.80–0.95× MLX.** MLX's `qmv` streams NVFP4 at 91–93 % of nominal on every shape; our V2 kernel is
   at 83–89 % on the wide shapes and 74 % on `down` (K = 17408). The M1 gate asked for ≥ 1.10× MLX; **that is not
   met**, and the honest reading is that a 4-bit GEMV on this chip is a solved problem at ~92 % — the remaining lever
   for plain decode is fusion and GPU autonomy, as the design's §2 ledger already said, not the GEMV geometry.
   FP8 (which MLX does not have) is at 91–95 %.
2. **T = 2–4: MLX is 2–4× faster than our shader kernels.** MLX runs T > 1 through its `qmm_t` path — dequantize a
   tile and multiply with `simdgroup_multiply_accumulate` (SIMD-group 8×8 matrix FMAs, available on every Apple GPU
   since Apple7) — and stays at 85–91 % of nominal at T = 2–4 on the 5120-K shapes. Our T > 1 kernels issue one FMA
   per weight per token on the shader ALUs and are ALU-bound from T = 2 (NVFP4) or T = 4 (FP8), and `p14`'s MPP
   `matmul2d` path (61 % at T = 8) is also behind MLX at T = 4. **Consequence for the design:** the verify pass of
   speculative decoding (design §5.8) must use a SIMD-group-matrix kernel for T ≥ 2 — with it, a T = 4 pass should
   cost ~1.1× a T = 1 pass on this chip instead of the ×1.8–2.7 the profile's `cost_T` table currently records from
   the shader kernels, which makes DSpark's block of 4–7 drafts pay much better than the FMA numbers suggested.

## 3d. The fusions (#19/#20) — `apple-m5-pro-20c_gemv_fusions.jsonl`, 5 alternating rounds

`gemv_T` gained the fusions of design §5.1/§5.6 as macros: `NORM` (the RMSNorm scaling applied on the activation
load, `x = bf16(h·r·(1+w))`, with `r` from a statistic buffer of 1 or `n_blocks` partial sums), `EPILOGUE=1`
(residual add before the single rounding), `EPILOGUE=2` (`silu(gate)·up` over chunk-interleaved rows) and
`STAT_OUT` (per-block partial Σy² of the BF16 outputs for the next norm). A standalone `norm_apply` kernel (one
SIMD-group per token, writes the BF16 normalized activation) is the alternative to `NORM`; `rmsnorm_stat` is the
standalone statistic. All variants pass the ≤ 2 ULP gates against the layer oracles (`tests/kernels/test_gemv_fusions.py`).

Cost, useful GB/s, best of 5 alternating rounds (median in parentheses; the machine was noisy during this run —
medians move up to 2× between rows, so read only the within-row ratios, which the alternation protects):

| format | shape | T | plain RG2 | plain RG8 | NORM fused RG8 | NORM+residual+STAT_OUT RG8 | norm_apply + plain RG8 |
|---|---|---|---|---|---|---|---|
| nvfp4 | 17408×5120 | 1 | 157 (144) | 161 (124) | 131 (114) | 120 (106) | 163 (162) |
| nvfp4 | 17408×5120 | 4 | 73 (71) | 83 (60) | 68 (51) | 65 (51) | 82 (61) |
| nvfp4 | 5120×17408 | 1 | 121 (84) | 128 (82) | 118 (77) | 119 (78) | 120 (78) |
| nvfp4 | 5120×17408 | 4 | 53 (53) | 60 (60) | 51 (50) | 50 (50) | 59 (59) |
| nvfp4 | 6144×5120 | 1 | 89 (85) | 92 (88) | 87 (84) | 87 (84) | 90 (85) |
| nvfp4 | 6144×5120 | 4 | 49 (49) | 56 (54) | 48 (46) | 47 (45) | 54 (53) |
| nvfp4 | 12288×5120 | 1 | 88 (85) | 95 (89) | 88 (86) | 90 (85) | 88 (87) |
| nvfp4 | 12288×5120 | 4 | 65 (51) | 74 (55) | 60 (60) | 58 (58) | 72 (72) |
| fp8_e4m3 | 17408×5120 | 1 | 138 (127) | 127 (119) | 116 (115) | 114 (114) | 115 (114) |
| fp8_e4m3 | 17408×5120 | 4 | 70 (66) | 84 (80) | 72 (72) | 72 (71) | 87 (79) |
| fp8_e4m3 | 5120×17408 | 1 | 112 (109) | 103 (100) | 102 (97) | 99 (97) | 98 (97) |
| fp8_e4m3 | 5120×17408 | 4 | 68 (55) | 92 (66) | 78 (60) | 78 (58) | 90 (65) |
| fp8_e4m3 | 6144×5120 | 1 | 231 (209) | 198 (185) | 183 (130) | 186 (118) | 153 (123) |
| fp8_e4m3 | 6144×5120 | 4 | 82 (63) | 108 (105) | 92 (68) | 91 (67) | 105 (104) |
| fp8_e4m3 | 12288×5120 | 1 | 227 (118) | 200 (110) | 162 (107) | 189 (107) | 175 (109) |
| fp8_e4m3 | 12288×5120 | 4 | 83 (82) | 109 (109) | 94 (93) | 92 (92) | 107 (107) |

What it says **[M]**:

1. **Fusing the scaling into the GEMV loses.** At equal geometry (RG = 8) the `NORM` form costs 5–19 % at T = 1 and
   13–18 % at T = 4: the chunk is re-scaled (2 multiplies + a BF16 rounding per element) once per row group, on top
   of an NVFP4 decode that is already ALU-bound. RG = 2, the plain T = 1 default, made it 40–60 % (first run, not
   tabulated). The `norm_apply` dispatch costs ~2 µs and 20–40 KB of traffic instead: within 0–3 % of the plain GEMV
   at T = 4 for both formats and at T = 1 for NVFP4; the FP8 T = 1 rows are inside the noise (66–95 %) and need a
   quiet re-run. **Default:** the statistic is hoisted (`STAT_OUT`, free within noise) and the scaling is its own
   dispatch; the fused `NORM` variant stays available to the autotuner. Design §5.1's "norm never costs a separate
   dispatch" holds for the *statistic* (the all-to-all part); the elementwise scaling is cheaper as a dispatch than
   as ALU work inside an ALU-bound kernel — +128 dispatches ≈ 0.25 ms per token for the 64-layer model versus
   5–18 % of ~7 ms of GEMV time.
2. **RG = 8 at T ≥ 2** for both formats (+13 % NVFP4, +20–34 % FP8 vs RG = 2); at T = 1 FP8 keeps RG = 2 (RG = 8 is
   8 % slower) and NVFP4 is indifferent. `gemv_macros` now picks 2 at T = 1 and 8 above.
3. **Residual epilogue and `STAT_OUT` are free** (`NORM+residual+STAT_OUT` = `NORM` within noise), so a layer
   boundary is two dispatches — the producer with the residual add and the statistic, the consumer — plus the
   scaling dispatch, and no separate reduction.

## 3e. MLX's decode as `NVFP4_DECODE = 3` (#100) — `apple-m5-pro-20c_nvfp4_v3_shapes.jsonl`

Go/no-go #2 (decode-kernels.md §8) left the NVFP4 GEMV at 221 GB/s in the 8B's step against mlx-lm's 268 over
its whole step, the decode the suspect. V3 is MLX's `fp4.h` decode (MIT, `third_party/NOTICE`): a nibble's three
magnitude bits placed straight into a half's exponent field, `as_type<half>(ushort((c & 7) << 9))` — the E2M1
value × 2⁻¹⁴ exactly (e = 0 lands in the half subnormals, m · 2⁻¹⁵) — the sign a select, half → float, the 2¹⁴
folded into `decode_scale`. It is bit-identical to V2 (the same products and sums up to an exact power of two;
`test_nvfp4_decode_variants_are_exact`) and the shortest of the four per weight: shift, mask, select, convert.
Measured 2026-09-26 [M] on the 8B's shapes (`tools/bench/decode_shapes.py`: R = 16, T = 1, the three geometries,
min-of-3 over ≥ 2 GB, the oracle checked; `tools/bench/mlx_baseline.py`'s `qmv` the same day; GB/s of the
weights' bytes on both sides — our pack streams 1.11× that at K = 4096 and 1.04× at K = 12288, the lane-row
unit's padding of #101):

| shape | MLX `qmv` | V2, best geometry | V3, best geometry | V3 / `qmv` | V3 crew ×1 / crew ×2 / 1 blk per SIMD-group |
|---|---|---|---|---|---|
| gate\|up 24576×4096 | 290 | 227 (crew ×2) | **244** (1 blk/SG) | 0.84 | 195 / 236 / 244 |
| down 4096×12288 | 272 | 180 (1 blk/SG) | **224** (1 blk/SG) | 0.82 | 124 / 192 / 224 |
| qkv 6144×4096 | 252 | 205 (1 blk/SG) | **243** (1 blk/SG) | 0.97 | 171 / 238 / 243 |
| o_proj 4096×4096 | 232 | 170 (1 blk/SG) | **208** (1 blk/SG) | 0.90 | 118 / 182 / 208 |
| lm_head 151936×4096 | 294 | 258 (1 blk/SG) | **262** (1 blk/SG) | 0.89 | 211 / 257 / 262 |
| M1 17408×5120 | 283 | 224 (crew ×2) | **248** (1 blk/SG) | 0.88 | 198 / 238 / 248 |

V3 gains 2–25 % per shape (most on the K = 12288 down projection and the 4096-row o_proj, where V2's ALU cost bit
hardest) and moves every shape to the one-block-per-SIMD-group geometry, which the autotuner picks. R = 8 packs
gain nothing (gate|up 252 vs 246, down 218 vs 235 GB/s, best RG each): R = 16 stays. The tile (`gemm_tile`,
decode-kernels.md §6) takes the same snippet in its cooperative fill: at TM = 8, gate|up 172 → 232 GB/s and down
92 → 141; the profile writer (#49) re-measured the engine block with V3 — `accelerator_nvfp4` 1.35 → 1.04 of a
T = 1 pass at 8 rows, 1.36 → 1.06 at 16 — and per shape the tile against the shader is:

| shape | shader T = 1 | tile TM = 8 | tile TM = 16 | shader T = 2 / 4 / 8 |
|---|---|---|---|---|
| gate\|up 24576×4096 | 0.232 ms (244 GB/s) | 0.245 (231, ×1.06) | 0.250 (227, ×1.08) | ×1.54 / ×2.42 / ×7.52 |
| down 4096×12288 | 0.126 ms (224 GB/s) | 0.201 (141, ×1.59) | 0.203 (139, ×1.61) | ×1.46 / ×2.30 / ×11.01 |
| qkv 6144×4096 | 0.058 ms (244 GB/s) | 0.069 (205, ×1.19) | 0.070 (203, ×1.20) | ×1.93 / ×3.08 / ×8.67 |
| o_proj 4096×4096 | 0.045 ms (207 GB/s) | 0.066 (143, ×1.45) | 0.067 (142, ×1.46) | ×1.39 / ×2.24 / ×10.80 |
| lm_head 151936×4096 | 1.337 ms (262 GB/s) | 1.405 (249, ×1.05) | 1.440 (243, ×1.08) | ×1.53 / ×2.46 / ×7.38 |

The tile is within 8 % of a T = 1 shader pass on gate|up and lm_head and 1.2–1.6× on the three 4096–6144-row
shapes: 256–384 row tiles cannot occupy the crew's 480 SIMD-groups (the K-split of decode-kernels.md §6 is the
next item; the shader's T = 2 pass costs 1.4–1.9×, so the tile stays the T ≥ 2 path throughout).

**Against #100's gate** (within 5 % of `qmv` per shape): not met by the decode alone — V3 is 0.82–0.97 of `qmv`
in weight bytes. The bus, though, carries the pack's padding: at K = 4096 / 5120 the V3 shader's bus rate is
0.93–1.07 of `qmv`'s (gate|up 271 GB/s, qkv 270, o_proj 231, lm_head 291, M1 264), and 0.85 on the down projection
(232). So the remaining gap on the wide shapes is the bytes, not the ALU — #101 (the unit without its padding) is
the item that closes it; on the 4096-row down projection it is occupancy ([H]: 256 blocks of 16 rows against
`qmv`'s 1024 SIMD-groups of 4 rows each, K = 12288 at 8 bytes per thread per row; the R = 8 pack did not help at
RG = 2). `qmv`'s structure — two SIMD-groups per threadgroup, four rows per SIMD-group, 16 weights per thread per
row with one E4M3 scale byte each, the activations loaded once per 512 columns and reused over the four rows — is
the same activation reuse and, after V3, the same instruction count per weight as ours. In the step
(decode-kernels.md §8): plain decode of the 8B on the MLX pack 41.5 → 46.4 tokens/s, 0.66 → 0.74× mlx-lm.

## 4. What it says

1. **FP8 is bus-bound at T = 1** (85–95 % of nominal); the crew geometry is at parity on the wide shapes and 5–30 %
   behind on the narrow ones (fewer blocks per SIMD-group: 6144-row matrices give 1.6 blocks per crew SIMD-group, so
   tail quantization and the lack of latency hiding both bite). One block per SIMD-group in 64-thread threadgroups is
   the safer default; the crew geometry with 2–4 threadgroups per core recovers most of the gap.
2. **NVFP4 was ALU-bound with the reference decode and is close to bus-bound with V2**: 82 % on gate/up, 89 % on
   lm_head, 74 % on down (K = 17408: 34 scale bytes per lane-row spill into a third scale word and the unit pads 306 →
   320 bytes). V2 is the default. Remaining ideas for the last 10–20 %: fold the scale bytes into the payload words for
   long K, a `half`-domain dot for the 16-weight group (a numerics-gate question), and the tail-quantization-aware
   block count.
3. **Small R wins for NVFP4** (R = 4 > 8 > 16 at one block per SIMD-group): fewer registers per row group and finer
   slices; for FP8 R = 8–16 is flat. `RG = 2` beats 4 for FP8 at the crew geometry (254 vs 223 GB/s) and RG = 1 is
   worse everywhere.
4. **T = 2 costs ×1.05–1.15 for FP8 and ×1.6–1.8 for NVFP4** relative to the same shape at T = 1 (best geometries);
   T = 4 costs ×2.3 (FP8) and ×2.7 (NVFP4) with RG = 2 — worse than `p13`'s ×1.11 / ×1.79 at RG = 4, so the T > 1
   kernels need their own tuning (issue #11) and the accelerator path from T ≈ 3–5.

## 5. Against plan M1's gate

| gate | status on the M5 Pro |
|---|---|
| FP8 shapes ≥ 100 GB/s effective | met (231–291 GB/s) |
| NVFP4 T = 1 ≥ 80 % of nominal | met on gate/up (82 %) and lm_head (89 %); 74 % on down — the K = 17408 layout item above |
| NVFP4 T = 1 ≥ 1.10× MLX's kernel on the same machine | **not met**: 0.80–0.95× (MLX `qmv` is at 91–93 % of nominal; ours 74–89 %) — §3c |
| outputs within 2 ULP (BF16) of the oracle | met on every point (max 0.0 ULP at the output's magnitude, max relative error 1.8e-7) |
