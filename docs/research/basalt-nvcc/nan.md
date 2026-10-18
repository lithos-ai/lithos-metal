# NaN bit patterns: what sm_120 stores and what this GPU stores

Which bits a NaN carries in every float, fp16 and packed word the merged basalt port kernels store: whether a NaN can
reach the store, which bits basalt's sm_120 code stores there, and which bits this GPU's operations give. Where the
two differ, the Metal kernel rewrites a NaN to sm_120's encoding at the store (`cuda_nan`, `cuda_nan_f16` in
`kernels/common/basalt_fp.metal`), so its outputs equal basalt's bit for bit for non-finite values too. This page
extends the per-site pages of `docs/research/basalt-nvcc/` (`docs/design/flash-next.md`, invariant 5) and uses their
records of each store's last instruction.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last. Evidence tags on this page: [M] measured by us, [S] a
vendor specification (here NVIDIA's PTX ISA; `CLAUDE.md` keeps [S] for Apple's), [H] inferred.

## Method

- **NVIDIA hardware.** Compiler Explorer's public API runs CUDA programs on one GPU, a Tesla T4 (sm_75); no sm_120
  GPU is available to us. The probe is a single-thread kernel with one `__noinline__` device function per PTX
  instruction, written as inline `asm volatile` (`add.rn.f32`, `add.rn.ftz.f32`, `mul.rn.f32`, `fma.rn.f32`,
  `div.rn.f32`, `div.approx.ftz.f32`, `rcp.approx.ftz.f32`, `ex2.approx.f32`, `max.f32`, `abs.f32`, `neg.f32`,
  `cvt.rn.f16.f32`, `cvt.f32.f16`, `cvt.rzi.s32.f32`, ...), plus `__float2half`, `__floats2half2_rn`,
  `make_half2(__float2half(a), __float2half(b))` (basalt's `q8_1_ds`) and `roundf`. Its inputs come from device
  memory so nothing folds: made NaNs (inf − inf, inf · 0, fma(inf, 0, 1), 0 / 0) and passed ones (0x7fc00000, the
  signaling 0x7fa5a5a5, 0x7fd2d2d2, the negative 0xffc00001, 0x7fffffff, 0xffffffff, 0x7f800001; the fp16 NaNs 0x7e00,
  0x7d55, 0xfe01, 0x7fff). Built with NVCC 13.3.0 (`nvcc133`, `-O3`, code for sm_75 and sm_120) and run there; the
  same source built for sm_120 alone with basalt's flags (`--generate-code=arch=compute_120,code=[compute_120,sm_120]`)
  gives the SASS each instruction becomes on basalt's target. A host program prints NVIDIA's own NaN constants and the
  host-side fp16 conversion.
- **This GPU.** The same operations in MSL, `MTLMathModeSafe` (the repository default), on the M3 Max (Apple family
  9, macOS 27.0.1), with `basalt_fp.metal`'s helpers.
- **The stores.** Each store's value is traced back through basalt's source to the instruction that wrote it, as the
  per-site pages record it (`matvec.md`, `dequant.md`, `experts.md`, `bf16.md`, `dense_tc.md`, `q5k_head.md`,
  `s2_experts.md`), and the inputs that can make a NaN reach it are worked out from the same arithmetic.

## What each operation gives a NaN

The PTX ISA (9.4, section 9.7.3, Floating-Point Instructions) leaves the single-precision NaN unspecified [S]:
"Single-precision instructions return an unspecified NaN. Note that future implementations may support NaN payloads for
single-precision instructions, so PTX programs should not rely on the specific single-precision NaNs being
generated." Its float-to-integer rule is specified [S] (section 9.7.10.24, `cvt`): "In float-to-integer conversions,
depending upon conversion types, NaN input results in following value: Zero if source is not .f64 and destination is
not .s64, .u64." The only encoding it names is the double-precision approximations' "canonical NaN with encoding
0x7fffffff00000000" (`rcp.approx.ftz.f64`, `rsqrt.approx.ftz.f64`).

| Operation (PTX → sm_120 SASS) | NaN in or made | sm_75 [M] | sm_120 | This GPU [M] |
|---|---|---|---|---|
| `add.f32`, `sub.f32`, `add.ftz.f32`, `add.rz.ftz.f32` → `FADD`, `FADD.FTZ`, `FADD.FTZ.RZ` | inf − inf; any NaN operand, every payload and sign above | 0x7fffffff | 0x7fffffff [H] | `+`, `add_ftz`: 0x7fc00000 |
| `mul.f32`, `mul.ftz.f32` → `FMUL`, `FMUL.FTZ` | inf · 0; a NaN operand | 0x7fffffff | 0x7fffffff [H] | `*`, `mul_ftz`: 0x7fc00000 |
| `fma.rn.f32`, `fma.rn.ftz.f32` → `FFMA`, `FFMA.FTZ` | fma(inf, 0, 1); a NaN operand | 0x7fffffff | 0x7fffffff [H] | `fma`, `fma_ftz`: 0x7fc00000 |
| `div.rn.f32` → `MUFU.RCP`, `FCHK`, `FFMA` ×5, slow path | 0 / 0; a NaN operand | 0x7fffffff | 0x7fffffff [H] | `/`: 0x7fc00000 |
| `div.approx.ftz.f32`, `div.full.ftz.f32` → `MUFU.RCP`, `FMUL.FTZ` | 0 / 0; a NaN operand | 0x7fffffff | 0x7fffffff [H] | `mul_ftz(a, rcp_ftz(b))`: 0x7fc00000 |
| `rcp.approx.ftz.f32` → `MUFU.RCP`; `rcp.rn.f32` → `MUFU.RCP`, `FFMA` | a NaN | 0x7fffffff | 0x7fffffff [H] | `1.0f / x`, `rcp_ftz`: 0x7fc00000 |
| `ex2.approx.ftz.f32` → `MUFU.EX2`; `ex2.approx.f32` → `MUFU.EX2` with the halving guard | a NaN | 0x7fffffff | 0x7fffffff [H] | `fast::exp2`, `expf_ftz`, `expf_neg`, `precise::exp`: 0x7fc00000 |
| `max.f32`, `max.ftz.f32` → `FMNMX`, `FMNMX.FTZ` | one NaN operand | the other operand | the other operand [H] | `fmax`, `max_ftz`: the other operand |
| `max.f32`, `max.ftz.f32`, `min.f32` → `FMNMX`, `FMNMX.FTZ` | two NaNs | 0x7fffffff | 0x7fffffff [H] | one operand's bits (0x7fd2d2d2 or 0xffc00001 here) |
| `abs.f32`, `neg.f32` → `FADD \|x\|, -RZ`, `FADD -x, -RZ` | a NaN | 0x7fffffff | 0x7fffffff [H] | `fabs`, `-x`, `ftz`: the payload kept (`fabs` clears the sign, `-` flips it) |
| `roundf` → `FADD.RZ`, `FRND.TRUNC`; `cvt.rzi.f32.f32` → `FRND.TRUNC` | a NaN | 0x7fffffff | 0x7fffffff [H] | `round`: the payload kept; `floor`, `trunc`: 0x7fc00000 |
| `cvt.rn.f16.f32` → `F2F.F16.F32` (sm_75; sm_120 for a lone conversion) | any NaN, either sign | 0x7fff | 0x7fff [H] | `half(x)`, `f16_bits`: 0x7e00 |
| two `cvt.rn.f16.f32` packed → `F2FP.F16.F32.PACK_AB` (sm_120; sm_75 emits `F2F` ×2 and `PRMT`) | any NaN | (not on sm_75) | 0x7fff [H] | as above |
| `cvt.f32.f16` → `HADD2.F32 -RZ, x.H0_H0` | an fp16 NaN | 0x7fffffff | 0x7fffffff [H] | `h2f`: 0x7fc00000 |
| `cvt.rzi.s32.f32`, `cvt.rzi.ftz.s32.f32`, `cvt.rmi.ftz.s32.f32` → `F2I.TRUNC.NTZ`, `F2I.FTZ.TRUNC.NTZ`, `F2I.FTZ.FLOOR.NTZ` | a NaN | 0 | 0 [S] | `int(x)`: 0 (MSL section 8.6) |
| `q8_1_finite` (`q8_1_finite.hpp:18-20`) → `FSETP.GT.FTZ \|v\|, 65504`, `LOP3.LUT`, `FSEL` | a NaN | | the NaN unchanged (the compare is false) | the same expression: the NaN unchanged |
| `selp`, moves, shuffles, `__int_as_float` → `FSEL`, `MOV`, `SHFL` | a NaN | | the bits unchanged | the bits unchanged |

NVIDIA's constants and host conversion (CUDA 13.3) [M]: `CUDART_NAN_F` is `__int_as_float(0x7fffffffU)` (stored as
0x7fffffff), `CUDART_NAN_FP16` is 0x7fff, and the host `__float2half` and `__floats2half2_rn` (`cuda_fp16.hpp`'s
software conversion) give 0x7fff for each of the NaNs above, the negative one included.

**sm_120's encoding.** The sm_75 measurements give NVIDIA's canonical NaN for every arithmetic instruction class
basalt's stores end in, whatever NaN came in: 0x7fffffff in fp32 and 0x7fff in fp16, with the sign dropped. The sm_120
SASS uses the same instruction classes (`FADD`, `FMUL`, `FFMA`, `MUFU.RCP`, `MUFU.EX2`, `FMNMX`, `F2F`, `HADD2.F32`,
`F2I`); NVIDIA's NaN constants and its host emulation of the fp16 conversion carry the same encodings. This page takes
them as sm_120's, as PR #18 did for its kernels (`s2_experts.md`, Summary); they are not measured on an sm_120 GPU,
and the PTX ISA does not promise them.

**The fp16 conversion on sm_120.** basalt's q8_1 scale and sum go through `F2FP.F16.F32.PACK_AB` on sm_120
(`matvec.md` sections 1 and 3), which sm_75 does not emit, so no run measured it. The NaN that reaches it is always
the canonical 0x7fffffff: d and the sum are results of `FADD`, `FMUL`, `FMNMX` of two NaNs or the `div.rn.f32`
sequence, and `q8_1_finite`'s select passes them unchanged. 0x7fffffff converts to 0x7fff both under the rule `F2F`
follows on sm_75 (every NaN to 0x7fff) and under a conversion that keeps the leading payload bits (sign 0, exponent
all ones, the ten leading mantissa bits all ones), so the stored half is 0x7fff unless `F2FP` produced a third
encoding, which no NVIDIA conversion measured here or documented does.

**This GPU's encoding.** Its add, multiply, fma, division, `fast::exp2`, `precise::exp`, `floor` and `trunc` give
0x7fc00000 for a made or a passed NaN; `fabs`, negation, `round` and `ftz` keep a NaN's payload; `fmax` and `fmin`
return one operand of two NaNs; the conversion to half gives 0x7e00 for every NaN; the conversion back gives
0x7fc00000. A port kernel's stored NaN therefore differs from sm_120's wherever it is not rewritten.

## The stores

"NaN reaches it" names inputs that make a NaN arrive at the store in basalt's arithmetic (and so in the port's). An
infinity alone never reaches a q8_1 half: `q8_1_finite` clamps it to ±65504 (`q8_1_finite.hpp:1-9`). A NaN in a
weight scale comes only from a damaged file, in an activation only from upstream; the kernels store it either way.
Each row's sm_120 cell names the row of the operation table above it rests on (the store's last instruction, [M] on
sm_75 and [H] for sm_120, whose PTX ISA leaves the single-precision NaN unspecified, section 9.7.3; an integer result
rests on the `cvt` rule, section 9.7.10.24 [S]). The SASS lines are the per-site pages' (13.3 unless marked);
where a page records only an instruction pair and a count, the row says so.

| Metal kernel: store | basalt | Last instruction on sm_120 | NaN reaches it | sm_120 | This GPU, unrewritten | Port |
|---|---|---|---|---|---|---|
| `native_quantize_q8_1`, `native_swiglu_quantize_q8_1`: codes | `native_mmvq.cu:157`, `:175` | `F2I.FTZ.TRUNC.NTZ` (`matvec.md` section 1) | never: a NaN quotient converts to 0 | 0 [S] (`cvt` rule) | 0 (`q8_1_code`) | no change |
| the same: fp16 d and sum | `native_mmvq.cu:158`, `:176`; `q8_1_finite.hpp:29-31` | `F2FP.F16.F32.PACK_AB R13, R13, R0` of `FMUL.FTZ` (d) and `FADD.FTZ` (sum) through `FSEL` (section 1) | sum: a NaN lane, or +inf and −inf in one block (`FADD.FTZ` butterfly); d: every lane NaN (`FMNMX.FTZ` of NaNs, then `FMUL.FTZ`); the SwiGLU's value is NaN for a NaN gate, a −inf gate (`__fdividef(−inf, inf)` is −inf · `MUFU.RCP`(inf) = −inf · 0), a NaN up, or inf · 0 in the `__fmul_rn` product (`native_mmvq.cu:170`) | 0x7fff [H] (`F2FP` row; the NaN in is `FMUL.FTZ`'s or `FADD.FTZ`'s 0x7fffffff) | 0x7e00 | `cuda_nan_f16` in `q8_1_put` |
| `quantize_q8_1`: codes | `iq_kernels.cu:1892` | `F2I.TRUNC.NTZ` (section 3) | never | 0 [S] (`cvt` rule) | 0 | no change |
| `quantize_q8_1`: fp16 d and sum | `iq_kernels.cu:1893` | `F2FP.F16.F32.PACK_AB` of the `div.rn.f32` sequence (d) and `FADD` (sum) (section 3 records no registers) | as above; d = NaN / 127 for an all-NaN block | 0x7fff [H] (`F2FP` row; the NaN in is `div.rn.f32`'s or `FADD`'s 0x7fffffff) | 0x7e00 | `cuda_nan_f16` in `q8_1_put` |
| `basalt_mmvq`: y, one and several columns | `native_mmvq.cu:281`, `:347`, `:435`, `:517`, `:615`, `:689`, `:800`, `:878`, `:1148` | `SHFL.BFLY` + `FADD.FTZ` ×5, the last add stored (`matvec.md` sections 4, 6 record the pair, no register line) | a weight's fp16 d or dmin inf against a zero integer dot or a zero d8, or NaN; a NaN d8, or a NaN s8 in Q4_0 and Q5_0 | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_dequant_f32`: y | `iq_kernels.cu:2020-2021`, `:2043`, `:2054-2055`, `:2067` | `FMUL R9, R0.reuse, R5` (Q2_0), `FMUL R7, R6.reuse, R7` (IQ4_NL), `FMUL R21, R22.reuse, R21` (Q3_K), `FMUL R9, R6.reuse, R9` (IQ4_XS), 13.0 (`dequant.md`) | the block's d NaN, or inf times a zero factor: Q2_0's code 1, Q3_K's code 0 or scale 32, IQ4_XS's scale 32 (IQ4_NL's codebook has no 0) | 0x7fffffff [H] (`FMUL` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_expert_gu`: gate and up, `GU2` 1 | `gu2.cuh:296-297` | `SHFL.BFLY` + `FADD` ×5, `warp_sum`'s last add stored (`experts.md` section 1 records the pair, no register line) | a gate or up row's d2 inf against a zero sumi or d8, or NaN; a NaN d8 | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_expert_gu`: gate and up, `GU2` 0 | `iq_kernels.cu:1153`, `:1158`, `:1167` | `SHFL.BFLY` + `FADD` ×5 (section 1) | as above | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_expert_swiglu_q8_1`: codes, fp16 d and sum | `gu2.cuh:388`; `iq_kernels.cu:1892-1893` | as `quantize_q8_1` (`experts.md` section 2) | a NaN gate or up (a NaN row), a gate of −inf (`div.rn.f32` of −inf by inf), inf · 0 in the product | codes 0 [S]; halves 0x7fff [H] (`F2FP` row) | codes 0; halves 0x7e00 | `cuda_nan_f16` in `q8_1_put` |
| `basalt_expert_down`: out, `SUB16` 1 | `iq_kernels.cu:1803`, `:1808`, `:1817` | `FADD R35, R16, R17` (s + s16), then `SHFL.BFLY` + `FADD` ×4, the last add stored (`experts.md` section 3) | a down row's d2 inf or NaN; a NaN q8_1 d of the intermediate | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_expert_down`: out, `SUB16` 0 | `iq_kernels.cu:1874` | `SHFL.BFLY` + `FADD` ×5 (section 3) | as above | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_mmvf`: y | `native_bf16.cu:72`, `:121` | the block sum's `FADD.FTZ` ×5 ending `FADD.FTZ R7, R4, R7`, then `STG.E desc[UR8][R2.64], R7` (13.2; `bf16.md` section 1 and its 13.3 note); the multi kernel's `STG.E` per token (section 2) | a NaN weight or activation; inf · 0; inf − inf | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_dense_tc`: y; `basalt_dense_tc_pair`: y0, y1 | `dense_tc.cu:425`, `:559`, `:680`, `:926` | the tree's last add, 13.0 (`dense_tc.md` section 2): `FADD R23, R23, R8` (one), `FADD R19, R12, R19` (tile), `FADD R13, R13, R18` (s, KS 1), `FADD R13, R13, R10` (s, KS 2), `FADD R41, R32, R41` (pair) | a weight's fp16 scale inf against a zero term, or NaN; a NaN d8 | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_head`: y | `q5k_head.cu:139`, `:218` | `SHFL.BFLY` + `FADD.FTZ` with offsets 16..1 (`q5k_head.md` section 1 records the pair, no register line) | as `basalt_mmvq`'s Q5_K | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_head_mask_unmapped`: y | `q5k_head.cu:402` | the constant −inf | never | | | no change |
| `basalt_q5k_head_top1`: y | `q5k_head.cu:213-218` | the head's `FADD.FTZ` butterfly, or the constant −inf of a masked row | as the head | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_head_top1`: part m | `q5k_head.cu:300` | a logit through `Top::take` and `merge` (`:155-175`): m changes only to l with l > m, or to a nonempty partial's m | never: `l > m` is false for a NaN l | | | no change |
| `basalt_q5k_head_top1`: part idx | `q5k_head.cu:301` | an int's bits (`__int_as_float`; 0x7fffffff is the empty index) | not a float value | | | no change |
| `basalt_q5k_head_top1`: part s | `q5k_head.cu:302` | `@P1 FFMA.FTZ R8, R8, R3, 1` or `@!P1 FADD.FTZ R8, R8, R5` (`take`), `FFMA.FTZ R5, R2, R5, R6` or `FFMA.FTZ R5, R6, R0, R5` (`merge`; `q5k_head.md` section 2) | a NaN logit (`s += __expf(NaN − m)`); a second +inf logit (`inf − inf`) | 0x7fffffff [H] (`FFMA`, `FADD` rows) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_head_top1_merge`: out_prob | `q5k_head.cu:332` | `MUFU.RCP R13, R6`, stored as is (`q5k_head.md` section 2) | the merged s NaN | 0x7fffffff [H] (`rcp.approx` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_proxy_pack`: fp16 scale | `head_proxy.cuh:86` | `F2FP.F16.F32.PACK_AB R0, RZ, R40` (`q5k_head.md` section 4.1) | never: amax starts at 1e-9 and `FMNMX.FTZ` drops a NaN weight (`:44-45`), so amax and s = amax · fl(1/1.5) · fl(1 − 0.1t) (a positive factor) are finite or +inf | | | no change |
| `basalt_q5k_proxy_pack`: levels | `head_proxy.cuh:76-85` | `F2I.FTZ.FLOOR.NTZ R9, R9`, integer clamp (`q5k_head.md` section 4.1) | never: a NaN level converts to 0 | 0 [S] (`cvt` rule) | 0 (`px_level`) | no change |
| `basalt_q5k_proxy_score`: part m | `head_proxy.cuh:154` | `FMNMX.FTZ R10, R7, R10, !PT`, offsets 16..1 (`q5k_head.md` section 4.2) | every lane's score NaN | 0x7fffffff [H] (`FMNMX` two-NaN row) | an operand's 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_proxy_score`: part e | `head_proxy.cuh:154` | `FADD.FTZ R16, R7, R16`, offsets 16..1 (section 4.2) | a +inf score (`__expf(inf − inf)`) | 0x7fffffff [H] (`FADD` row) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_proxy_score`: cand score | `head_proxy.cuh:164` | a score's bits through `SHFL` and `FSEL` (`FSETP.GEU.FTZ` / `FSETP.NEU.FTZ`, section 4.2): `FADD.FTZ R4, R4, R7` (`:146`) or the constant −inf (`:147`) | a NaN score: no compare selects a NaN over a number, and lane 0 keeps its own NaN | 0x7fffffff [H] (`FADD` row; `FSEL` keeps the bits) | 0x7fc00000 | `cuda_nan` |
| `basalt_q5k_proxy_score`: cand row | `head_proxy.cuh:164` | an int's bits | not a float value | | | no change |
| `basalt_q5k_proxy_final`: out_prob | `head_proxy.cuh:194`, `:236-238` | the constant 1, or `MUFU.RCP` of `FMNMX.FTZ(sum, one)`, one 0 or 1 (section 4.3) | never: `FMNMX` drops a NaN sum | | | no change |
| `basalt_s2_gu`, `basalt_s2_swiglu`, `basalt_s2_down`: gate/up, out | `s2_expert_grouped.cu:963-964`, `:170`, `:1022-1023` | `FADD`, `FMUL` (`s2_experts.md`) | `s2_experts.md`, Summary | 0x7fffffff [H] (`FADD`, `FMUL` rows) | 0x7fc00000 | `cuda_nan` (PR #18) |
| `basalt_quantize_q8_0_scaled`: fp32 scale, fp16 d, codes | `quantize_act.cu:91-100` | `div.rn.f32` under `amax > 0`; `f16_from_f32` (integer); `F2I.TRUNC.NTZ` | scale and d: never, s is 0 for an all-NaN block (`s2_experts.md` section 1); codes: a NaN x, which converts to 0 | codes 0 [S] (`cvt` rule) | codes 0 (`q8_1_code`) | no change |

The integer outputs (token ids, the head's and the proxy's `out_id`, the experts' plans) carry no float encoding.

**A forward note on `f16_from_f32`.** basalt's software fp16 conversion (`include/basalt/kernels/f16_bits.hpp:36-65`)
maps a NaN to `sign | 0x7E00`: for the positive canonical NaN sm_120's arithmetic makes, 0x7E00, the bits this GPU's
`half()` gives and not the hardware conversion's 0x7fff; for a negative NaN, 0xFE00, where `half()` drops the sign. `f16_bits` stands for it in the drafter quantizer, where no NaN arrives. A port of the fp16 KV append
(`src/kernels/cuda/qsa.cu:134-135`), which writes K and V through `f16_from_f32`, stores a NaN; that port needs
`sign | 0x7E00` there.

## Metal

`basalt_fp.metal` holds the two rewrites, integer operations on the bits:

- `cuda_nan(x)`: a float NaN (any sign or payload) becomes 0x7fffffff; every other value keeps its bits. It moves
  here from `basalt_q8_1.metal`, where PR #18 introduced it.
- `cuda_nan_f16(h)`: an fp16 NaN becomes 0x7fff; every other half keeps its bits. It stands for the hardware
  conversion (`__float2half`, `F2F` and `F2FP`) after `f16_bits`, and is applied in `q8_1_put`, the store every q8_1
  quantizer shares; `f16_bits` itself stays the conversion `f16_from_f32` needs for finite values.

Each kernel applies them at the stores the table marks, so a finite value is stored as before. `tests/kernels/
test_basalt_nan.py` plants inputs that make a NaN reach every rewritten store, requires the stored bits to be
sm_120's, and runs the same kernels without the rewrite as the negative control.
