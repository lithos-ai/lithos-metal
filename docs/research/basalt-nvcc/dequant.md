# basalt's dequantizers as nvcc compiles them

What NVCC 13.0 and 13.3 emit for the floating-point steps of basalt's GGUF dequantizers for this pack's types that
`iq_parity` checks: `dq_iq4_nl`, `dq_q3_k`, `dq_iq4_xs` and `dq_q2_0` (`src/kernels/cuda/iq_kernels.cu:2012-2069`). They
run under `dequant_flat_kernel<float>` (`:2216-2220`, launched by `iq_dequant_f32`, `:2496-2500`), and the Q3_K token
embedding's gather shares them (`:2478-2483`). The page also gives the Metal form, `kernels/common/basalt_dequant.metal`.
Its conventions are those of `matvec.md`: basalt commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`, fl(e) one
rounding to nearest even, h(x) the exact fp16-to-fp32 conversion and f(n) the exact int-to-fp32 conversion.

## Method

- **Unit.** `iq_kernels.cu` lines 1-33 (its includes and namespaces), 1923-1926 (`cvt`) and 2012-2069 (the four
  functions), verbatim, with the quoted local includes inlined. Then comes one kernel per type that calls its function
  as `dequant_flat_kernel` does: superblock `blockIdx.x`, thread `threadIdx.x`, `y + i * QK_K`.
  `dequant_flat_kernel` selects the function with a runtime `switch` on the type, so each case is the same inlined
  body.
- **Flags.** `iq_kernels.cu` builds with nvcc's defaults (no `--use_fast_math`, `CMakeLists.txt:185-202`): `-O3
  -DNDEBUG -DBASALT_VERSION="0.0.1" --generate-code=arch=compute_120,code=[compute_120,sm_120]`, on Compiler
  Explorer's `nvcc130` and `nvcc133`.

## Result

Each value is one or two IEEE multiplies, with nothing fused and nothing flushed. The PTX has `mul.f32` (round to
nearest, no `.ftz`). The SASS has `FMUL` without `.FTZ` and no `FFMA` or `FADD`. Its conversions are `HADD2.F32
R, -RZ, R.H0_H0` (fp16 to fp32) and `I2F.S8` / `I2F.S16` / `I2FP.F32.S32` (integer to fp32). The `HFMA2 R, -RZ, RZ,
0, 0` lines only zero registers. The two versions emit the same instructions, except that 13.3 converts the codebook
bytes with `I2F.S16` where 13.0 uses `I2F.S8`, which gives the same values.

| Function | basalt | PTX (13.0 / 13.3) | SASS (13.0 / 13.3) | Value | Metal |
|---|---|---|---|---|---|
| `dq_iq4_nl` | `:2020-2021` `d * kvalues_iq4nl[q]` | `mul.f32 %f3, %f1, %f2` / `mul.f32 %r8, %r1, %r7` | `FMUL R7, R6.reuse, R7` / the same | fl(h(d) · f(k)) | `d * float(KVALUES_IQ4NL[q])` |
| `dq_q3_k` | `:2039` `(float) x->d * (us - 32)`; `:2043` `dl * (code)` | `mul.f32 %f4, %f2, %f3`; `mul.f32 %f6, %f4, %f5` / `mul.f32 %r21, %r11, %r20`; `mul.f32 %r32, %r21, %r31` | `FMUL R22, R17, R22`; `FMUL R21, R22.reuse, R21` / `FMUL R24, R19, R24`; `FMUL R23, R24.reuse, R23` | fl(fl(h(d) · f(us − 32)) · f(code)) | `h2f(d) * float(us - 32)`, then `dl * float(q)` |
| `dq_iq4_xs` | `:2052` `(float) x->d * (ls - 32)`; `:2054-2055` `d * kvalues_iq4nl[q]` | `mul.f32 %f3, %f1, %f2`; `mul.f32 %f5, %f3, %f4` / `mul.f32 %r25, %r1, %r24`; `mul.f32 %r27, %r25, %r26` | `FMUL R6, R6, R3`; `FMUL R9, R6.reuse, R9` / `FMUL R3, R3, R4`; `FMUL R7, R3.reuse, R7` | fl(fl(h(d) · f(ls − 32)) · f(k)) | `h2f(d) * float(ls - 32)`, then `d * float(KVALUES_IQ4NL[q])` |
| `dq_q2_0` | `:2067` `d * (float) (code - 1)` | `mul.f32 %f3, %f1, %f2` / `mul.f32 %r12, %r1, %r11` | `FMUL R9, R0.reuse, R5` / the same | fl(h(d) · f(code − 1)) | `d * float(code - 1)` |

**No subnormal arises.** Every factor is an fp16 scale (|h(d)| ≥ 2^-24 when nonzero, normal in fp32), a small
nonzero integer, or a product of the two. So no product is subnormal, and this GPU's flushing of subnormal multiply
results in safe mode (`tests/kernels/test_basalt_platform.py`) cannot change a value. The Metal multiplies with
contraction off are therefore CUDA's operations. `tests/kernels/test_basalt_iq_parity.py` measures a relative error
of 0 against gguf-py's dequantizers on basalt's fixtures, as basalt reports for its own kernels
(`iq_parity.cpp:10`).
