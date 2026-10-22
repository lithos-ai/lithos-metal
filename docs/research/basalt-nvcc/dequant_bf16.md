# basalt's GGUF block dequantizers to fp16, BF16 and fp32 as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of `src/kernels/cuda/dequant_bf16.cu` (one thread per 32-value
group, GGUF blocks to fp16, BF16 or fp32), read from the sm_120 SASS, and the Metal form the port writes at each
site. The prompt path runs the fp16 entry, `dequant_f16`, ahead of every `Gemm::native` product
(`src/prefill/gemm.cu:169-185`); the BF16 and fp32 entries run only in tests. This page is the authority the Metal
kernel follows (`docs/design/flash-next.md`, invariant 5).

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/dequant_bf16.cu` unless another file is named.

## Method

- **Date.** October 18, 2026.
- **Sources.** `src/kernels/cuda/dequant_bf16.cu` with the basalt headers it includes inlined under `#line` markers:
  `include/basalt/kernels/dequant_bf16.hpp` and `include/basalt/kernels/iq_kernels.hpp`. System headers stay
  `#include`s. Only basalt's public MIT source was sent.
- **Compiler.** Compiler Explorer's public API (`POST https://godbolt.org/api/compiler/nvcc130/compile`,
  `produceDevice`): NVCC 13.0.0, `Cuda compilation tools, release 13.0, V13.0.48` (`Build
  cuda_13.0.r13.0/compiler.36260728_0`). Both the PTX and the `SASS (sm_120)` listings were read.
- **Flags.** The file is in the `kernels` library's source list (`CMakeLists.txt:159`) and not in the
  `--use_fast_math` list (`:186-202`); the project sets no `CMAKE_CUDA_FLAGS`, defaults to Release (`:12-14`) and
  builds for compute capability 12.0 only (`:3-4`, `:26-31`). nvcc's defaults hold: `-ftz=false -prec-div=true
  -prec-sqrt=true -fmad=true`. userArguments, exactly: `-arch=sm_120 -lineinfo -O3 -DNDEBUG
  '-DBASALT_VERSION="0.0.1"'`.
- **Translation units.** The whole file's SASS passes Compiler Explorer's 5,000-line listing cap, so it was sent as
  seven units, each the whole file with the line numbering kept, the host launcher (`:207-240`) and the public entries
  (`:254-276`) blanked, and explicit instantiations of `dequant_kernel<TYPE, T, VEC>` appended: for each output type
  T (fp16 `H16`, BF16 `uint16_t`, fp32 `float`) one unit with the vector store (`VEC = true`) for types 2, 6, 7, 8,
  20, 42 (Q4_0, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_0) and one for types 11, 12, 13, 14, 23 (Q3_K, Q4_K, Q5_K, Q6_K,
  IQ4_XS), and a seventh with the scalar store (`VEC = false`) for fp16 and types 8, 13, 14, 23, the types
  `dequant_f16_parity` runs through both stores. A kernel's code depends only on its own body and the device functions
  it calls, so the split changes no instruction. Every listing is complete.
- **Reading.** Every float instruction of each instantiation was matched to its source line (from `-lineinfo`) and,
  where an expression has several products or terms, to its operands through the registers of the full listing.

## Summary

- **Use.** The prompt path runs `dequant_f16` only, inside `Gemm::native` (`src/prefill/gemm.cu:178`, `:183`), for
  every weight matrix of a native type it multiplies; the fp32 and BF16 entries run only in tests. Each call converts
  a slice of a matrix's rows, so its arguments are the model's, not per-step values (invariant 4).
- **No approximate operation.** Each value is a product of an fp16 scale and small integers: `FMUL` (and `FFMA` for
  Q4_K, Q5_K and Q5_1's `d1 * q - m1` and `q * d + m`), exact conversions (`HADD2.F32` for the fp16 scale,
  `I2FP`/`I2F` for the integers), then the stored image: `F2FP.F16.F32.PACK_AB` for fp16, integer rounding for BF16,
  nothing for fp32. No `.FTZ`, no `MUFU` but the index divisions'.
- **Exact by construction.** Every product is exact and none is subnormal, for every finite fp16 scale [M] (section 2),
  so the fp32 values are d · P exactly, and Q4_K's, Q5_K's and Q5_1's one rounding is the same with or without the
  contraction. Plain Metal arithmetic therefore gives CUDA's values bit for bit, with `fma()` where the SASS has `FFMA`.
  The stored images follow section 1: `cuda_nan_f16(f16_bits(v))` for fp16, `f2bf(cuda_nan(v))` for BF16 (this
  file's `f2bf` keeps a NaN a NaN, unlike `kernels.cu`'s `bf`), `cuda_nan(v)` for fp32.
- **Invariance.** One thread per 32-value group; a value depends only on its block's bytes, never on the slice, the
  grid or the store form (`dequant_f16_parity` checks the vector and scalar stores bitwise).
- **Reuse.** `h2f`, `f16_bits`, `cuda_nan`, `cuda_nan_f16` (`kernels/common/basalt_fp.metal`). The IQ4_NL codebook
  and the Q4_K and Q5_K scale unpacking are the port's own copies under basalt's names, `kv_iq4nl` and `scale_min_k4`
  in `kernels/common/basalt_dequant16.metal`: `kernels/common/basalt_dequant.metal` defines its codebook beside
  decode's kernel, whose `#error` refuses six of the ten types, so no other source includes it. For IQ4_NL, Q3_K,
  IQ4_XS and Q2_0 the fp32 values equal `basalt_dequant.metal`'s (the same exact products of `iq_kernels.cu`'s
  formulas); only the layout differs (one thread per 32-value group here, one threadgroup per 256-value superblock
  there).

## 1. Loads and output conversions

`dequant_kernel<TYPE, T, VEC>` (`:170-188`) gives each thread one 32-value group: g = blockIdx.x · 256 + threadIdx.x
over rows · groups_per_row (`:173-174`), the row `row0 + g / groups_per_row` of the packed blocks, and the group's
32 outputs at `out + g * 32`. With `VEC` (the launcher's choice when `out` is 16-byte aligned, `:207-240`) the 32
values are built in registers (`T tmp[32]`) and written as 16-byte stores (`:177-184`); without it `put` stores
each value. The values do not depend on the store: `dequant_f16_parity` checks the two forms bitwise (section 3).
Every group is independent, so no value depends on the slice (`row0`, `rows`) or the grid. The row and group indices
(`:175`, `g / groups_per_row` and `g % groups_per_row`) use the exact integer division of `__cuda_sm20_div_s64`
(`I2F.U64.RP`, `MUFU.RCP`, `F2I.U64.TRUNC`, its body charged to `:188`) and of its 32-bit fast path (`UI2F.U32.RP`,
`MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ`): index arithmetic, exact.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:17-19` | `h2f(p)`: `__half2float(__ushort_as_half((uint16_t) (p[0] \| (p[1] << 8))))` | two `LDG.E.U8.CONSTANT` and `IMAD h, hi, 0x100, lo`; `HADD2.F32 d, -RZ, h.H0_H0` (`cuda_fp16.hpp:708`): −0 + h, the exact fp16-to-fp32 conversion | `h2f(ushort(p[0] \| (p[1] << 8)))` (`basalt_fp.metal:115`): `float(as_type<half>(h))`, exact, fp16 subnormals kept. A NaN scale's payload need not survive: every use of d is a multiply, whose NaN result is `cuda_nan`'s (below) |
| integer factors | `(float) (q - 1)`, `(float) (int8_t) b`, `(float) kv_iq4nl[..]`, `(float) sc`, … | `I2FP.F32.S32`, `I2FP.F32.U32`, `I2F.S8`, `I2F.S16`, `I2F.U16` | `float(n)`: exact, every factor is at most 128 in magnitude |
| `:30` | `put(H16*)`: `o[i].v = __half_as_ushort(__float2half_rn(v))` | `F2FP.F16.F32.PACK_AB h, b, a` (two values per instruction, `:184` with `VEC`; `F2FP.F16.F32.PACK_AB h, RZ, a`, `cuda_fp16.hpp:566`, without) | `cuda_nan_f16(f16_bits(v))` (`basalt_fp.metal`): round to nearest even, ±inf past 65504 (at 65520 and up), fp16 subnormals kept, a NaN as 0x7fff (`nan.md`) |
| `:20-27`, `:28` | `put(uint16_t*)`: `f2bf(v)` | `LOP3.LUT a, u, 0x7fffffff, RZ, 0xc0`; `ISETP.GT.U32.AND P, PT, a, 0x7f800000, PT`; NaN: `PRMT` (u >> 16) and `LOP3.LUT h, h, 0x40, RZ, 0xfc`; else `SHF.R.U32.HI t, RZ, 0x10, u`; `LOP3.LUT t, t, 0x1, RZ, 0xc0`; `IADD3 u, u, 0x7fff, t`; the high half | `f2bf(cuda_nan(v))`, a float argument as in basalt's signature (`:20`), then the same integer steps on its bits u: `(u & 0x7fffffffu) > 0x7f800000u ? ushort((u >> 16) \| 64u) : ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16)`. v is a product, so CUDA's NaN is 0x7fffffff and its image 0x7fff; `cuda_nan` gives Metal the same, where this GPU's own NaN 0x7fc00000 would give 0x7fc0 |
| `:31` | `put(float*)`: `o[i] = v` | a plain store | `cuda_nan(v)` stored |

Unlike `src/prefill/kernels.cu:31-35`'s `bf` (ported as `bf16_rne_nonan` in `kernels/common/basalt_fp.metal`), this
`f2bf` keeps a NaN a NaN (`:22-24`: "as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`"), so the two files' BF16
images of a NaN differ (0x7fff here, 0x8000 there); the port keeps both as basalt has them.

## 2. Per-type arithmetic

`group32<TYPE, T>` (`:41-166`) for each type. d, dmin, d_all and m are fp16 scales read by `h2f`; every other
factor is a small integer. This pack's projections use every type below but Q5_1 (`flash-next.md`, "The model file"; the
IQ-only types 16, 17, 18, 21, 22 and 29 go to `iq_dequant_f16` instead, `:259-267`, and this pack has none).

| Type (id) | Lines | Value | SASS | Metal |
|---|---|---|---|---|
| Q2_0 (42): 64 values per 18-byte block | `:43-52` | `(float) (q - 1) * d`, q = `(b[2 + e / 4] >> ((e % 4) * 2)) & 3` (`:50-51`) | `IADD3 q, q, -0x1`; `I2FP.F32.S32 f, q`; `FMUL v, d, f` | `float(q - 1) * d` |
| Q4_0 (2) | `:53-59` | `(float) ((b[2 + j] & 0x0F) - 8) * d` (`:57`), the high nibbles for j + 16 (`:58`) | `I2FP.F32.S32`; `FMUL` | `float(n - 8) * d` |
| Q5_0 (6) | `:60-69` | `(float) (((b[6 + j] & 0x0F) \| xh0) - 16) * d` (`:67`), `xh1` for j + 16 (`:68`) | `I2FP.F32.S32`; `FMUL` | `float(n - 16) * d` |
| Q5_1 (7), not in this pack | `:70-80` | `(float) ((b[8 + j] & 15) \| xh0) * d + m` (`:78-79`) | `I2FP.F32.U32`; `FFMA v, d, f, m`: the front end contracts the expression | `fma(d, float(n), m)` |
| Q8_0 (8) | `:81-84` | `(float) (int8_t) b[2 + j] * d` (`:84`) | `I2F.S8`; `FMUL` | `float(char(b)) * d` |
| IQ4_NL (20) | `:85-91` | `d * (float) kv_iq4nl[b[2 + j] & 0xf]`, `[b[2 + j] >> 4]` (`:89-90`) | `I2F.S16` of the codebook entry; `FMUL` | `d * float(kv_iq4nl[i])` |
| Q3_K (11): 256 values per 110-byte block | `:92-113` | `dl = d_all * (float) (scales[is] - 32)` (`:111`); `dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4))` (`:112`) | `I2FP.F32.S32`; `FMUL dl, d_all, s` (twice per group: t < 16 and t ≥ 16); `I2FP.F32.S32`; `FMUL v, dl, k` | `(d_all * float(s - 32)) * float(k)` |
| Q4_K (12): 144-byte blocks | `:114-122` | `d1 = d * (float) sc, m1 = dmin * (float) m` (`:120`); `d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1` (`:122`) | `I2F.U16`; `FMUL d1`, `FMUL m1`; `I2F.U16`; `FFMA v, d1, f, -m1` | `fma(d1, float(n), -m1)` |
| Q5_K (13): 176-byte blocks | `:123-136` | as Q4_K (`:129`), the value `d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1` (`:135`) | `I2F.U16`; two `FMUL`s; `I2FP.F32.U32`; `FFMA v, d1, f, -m1` | `fma(d1, float(n), -m1)` |
| Q6_K (14): 210-byte blocks | `:137-152` | `d * (float) sc[is + 2 * qu] * (float) (q - 32)` (`:151`) | `I2F.S8 s, sc`; `FMUL t, d, s`; `I2FP.F32.S32 f, q - 32`; `FMUL v, t, f`: left to right | `(d * float(sc)) * float(q - 32)` |
| IQ4_XS (23): 136-byte blocks | `:153-165` | `dl = d * (float) (ls - 32)` (`:159`); `dl * (float) kv_iq4nl[qs[j] & 0xf]`, `[qs[j] >> 4]` (`:162-163`) | `I2FP.F32.S32`; `FMUL dl`; `I2F.S16`; `FMUL v, dl, k` | `(d * float(ls - 32)) * float(kv_iq4nl[i])` |

The Q4_K and Q5_K rows' `FFMA` is the fp16 and BF16 units' form at all 32 unrolled sites of `:122` and of `:135`; the
fp32 unit fuses one site of each line and keeps `FMUL p, d1, f` then `FADD v, -m1, p` at the other 31 (the PTX has
`mul.f32` and `sub.f32` at every site, so the fusion is ptxas's choice), which gives the same bits because d1 · q is
exact (below), so one `fma(d1, float(n), -m1)` serves all three outputs.

**Every product is exact.** d has at most 11 significant bits and the integer factors at most 7 and 5 (Q6_K:
|sc| ≤ 128, |q − 32| ≤ 32; IQ4_XS: |ls − 32| ≤ 32, |kv| ≤ 127), so no product needs more than 23 bits. A host check
over every finite fp16 scale, subnormals and zeros included, and every factor product of each type
(`d · P` in binary64, compared with its binary32 rounding) confirms it [M]: every product is exact and none is
subnormal (the smallest nonzero one is 2^-24). The fp32 value is therefore d · P exactly, whatever the multiply order,
for every type but Q4_K, Q5_K and Q5_1, whose one rounding is the difference d1 · q − m1 (or the sum q · d + m): there
`FFMA` and a multiply followed by a subtraction give the same bits, since the product is exact; Metal writes `fma()`
where the SASS has `FFMA`, as the repository's ports do. These differences are multiples of 2^-24 too, so they are
zero or at least 2^-24: no fp32 subnormal occurs anywhere in this file, and this GPU's flushing changes no value.

So plain Metal arithmetic gives CUDA's fp32 values bit for bit. What remains is the stored image (section 1): a NaN
(a NaN scale, or an infinite one times a zero factor) is CUDA's 0x7fffffff and becomes 0x7fff in fp16 and in BF16;
an infinite scale times a nonzero factor is ±inf; a finite value of at least 65520 in magnitude becomes ±inf in fp16.
The largest finite magnitude, 65504 · 4096, is far below fp32's range.

The header of `kernels/common/basalt_dequant.metal` (`:16-17`), the port of the same formulas in `iq_kernels.cu` for
IQ4_NL, Q3_K, IQ4_XS and Q2_0, says "the products are not exact in general"; for those four types they are exact
too, by the check above. Its Metal form is unaffected: Metal's multiply is correctly rounded either way.

## 3. Test coverage

Four basalt tests call this file's entries directly. `dequant_f16_parity` and `dequant_bf16_test`, below, are not
registered with `ctest` and are run by hand. `elementwise_parity`, registered (`CMakeLists.txt:489-492`), checks
`dequant_bf16` on two Q8_0 blocks (`src/kernels/elementwise_parity.cpp:438-467`): with a NaN scale every BF16 value
must be a NaN, with scale 1 each must be ggml's BF16 of its integer. `q5_projection_parity`
(`src/kernels/q5_projection_parity.cpp`, built at `CMakeLists.txt:791-793`, run by hand on a real shard) checks
`dequant_f32` and `dequant_bf16` on Q5_1 only, which this pack does not use and the port refuses. The port runs the
first three in `tests/kernels/test_basalt_prompt_dequant.py`, `elementwise_parity`'s check verbatim.

**`dequant_f16_parity`** (`src/kernels/dequant_f16_parity.cpp`, built at `CMakeLists.txt:515-517`). For IQ4_XS,
Q8_0, Q5_K and Q6_K (`:37-38`) on random blocks of 10240 × 2560 values (`:35`, `std::mt19937 rng(7)`, `:40`), with
every block's scale overwritten by the normal fp16 value `0.001f + (float) (rng() % 1000) * 1e-4f` (`:49`) and Q5_K's
dmin by 0.01 (`:51`), so that no NaN or infinite scale occurs. It counts three kinds of mismatch:

```
            if (__half_as_ushort(__float2half_rn(f[(size_t) i])) != h[(size_t) i]) ++bad_f32;
            if (h[(size_t) i] != h2[(size_t) i]) ++bad_scalar;
...
                    if (e > 1e-3 * std::fabs((double) v) + 1e-6) ++bad_cpu;
...
        const bool ok = bad_f32 == 0 && bad_scalar == 0 && bad_cpu == 0;
```

(`:73`, `:74`, `:84`, `:98`): the fp16 output of the vector store against the fp32 path rounded to fp16, bitwise; the
same output through the scalar store (`out` misaligned by one element, `:64`), bitwise; and the first 64 rows (`:76`) against
the artifact's CPU dequantizers, within 10^-3 relative plus 10^-6. On Metal the first two hold by construction
(sections 1 and 2: the fp32 values are exact, and the store form changes none), and the port keeps both checks as
bit checks; the third needs a host reference of the four types, here `tests/contract/gguf_reference.py`'s float32
values, bit for bit basalt's CPU dequantizers (`include/basalt/artifact/dequant.hpp`), compared in binary64 at
basalt's bound. The prompt path converts this pack's other six types too (Q2_0, Q3_K, Q4_K, IQ4_NL, Q4_0, Q5_0;
`flash-next.md`, "The model file"); the type list is the test's data, so the port runs the same three checks for all ten.

**`dequant_bf16_test`** (`src/kernels/dequant_bf16_test.cpp`, `CMakeLists.txt:804-807`: "not a ctest: needs the full
model"). For each type `dequant_bf16_supported` accepts, the first four rows of the first two-dimensional tensor of
that type in the model's GGUF shards, against the artifact's CPU dequantizers:

```
                const double tol = 1e-6 * std::fabs((double) ref[i]) + 1e-30;
                if (e > tol) ++bad_f;
...
                if (eh > std::ldexp(std::fabs((double) ref[i]), -8) + 1e-30) ++bad_h;
...
            const bool ok = bad_f == 0 && bad_h == 0;
```

(`:91-92`, `:95`, `:99`): the fp32 path within 10^-6 relative (its header, `:6-7`: "the products are the reference's,
reordered at most") and the BF16 path within 2^-8 of the reference's magnitude. With every product exact (section 2)
the fp32 path equals any reference that forms the same products, whatever its order or contraction; only Q4_K's
and Q5_K's one rounding of d1 · q − m1 must be the reference's too, and it is the same rounding in either form. The
port runs these criteria on the committed pack fixture (`tests/contract/fixtures/basalt_pack.safetensors`), not on
the shards: the fixture's tensor of each of the ten types, rows 0 and n − 1, at most their first 1,024 bytes in whole
blocks (where basalt reads four whole rows); and on `dequant_f16_parity`'s blocks above.
