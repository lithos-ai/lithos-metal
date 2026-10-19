# basalt's CPU Q2_0 expert kernels as x86 compilers compile them

What GCC and Clang emit at every rounding site of basalt's two CPU expert kernels, `act_quant_q8_1` and
`q2_0_gguf_rows_multi`, and the ARM NEON form the port writes at each site. basalt computes an expert on the CPU
when the GPU cache misses it, so these two kernels carry the CPU tier's whole arithmetic. This page is the
authority `runtime/src/basalt_cpu.cpp` follows, as `docs/research/basalt-nvcc/matvec.md` is for the GPU kernels.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b
there, and a bare `:N` is a line of the file named last. Tags: **[M]** measured here, **[S]** a vendor
specification or a compiler's own header, **Inference:** a conclusion drawn from reading code.

## The sources

`src/kernels/cpu/expert.cpp:143-148` (`unpack64_q2_0`), `:336-370` (`act_quant_q8_1`, the AVX-512 block; the
function returns at `:369`, so the scalar loop at `:371-397` is unreachable) and `:503-556` (`q2g_row_multi`,
`q2g_rows`, `q2_0_gguf_rows_multi`), with `include/basalt/kernels/cpu/expert.hpp:25-61` for `ActQ` and the
geometry. `src/kernels/cpu/expert_layout.cpp:124-133` routes this pack to them wherever AVX-512 is present.

## 1. Which compiler, and the values that depend on the answer

basalt pins no compiler: `README.md:35-43` asks for "CMake 3.24 or newer and a C++20 compiler" and builds with a
bare `cmake -S . -B build`; `CMakeLists.txt` sets no `CMAKE_CXX_COMPILER` and only `-Wall -Wextra` (`:45-46`). What
it does pin is the build type and the ISA: `:12-14` forces `CMAKE_BUILD_TYPE=Release` when the caller names none,
so these kernels compile at CMake's `CMAKE_CXX_FLAGS_RELEASE`, `-O3 -DNDEBUG` — the flags come from CMake's own
default, not from basalt's file — and `:673-674` gives `expert.cpp` alone
`-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma`. Both matter to what follows:
the optimisation level changes which FMA form GCC emits (section 6).
`_mm512_reduce_add_ps` and `_mm512_reduce_max_ps` are header-defined, so their lane trees come from whichever
compiler builds basalt. Both candidates were read, and both were cross-compiled and their x86 output inspected [M]:

- **GCC** `gcc/config/i386/avx512fintrin.h` (`releases/gcc-14`): the float `__MM512_REDUCE_OP` at `:16297-16306`,
  used by `_mm512_reduce_add_ps` at `:16308-16312`, and the max form at `:16339-16350`, used by
  `_mm512_reduce_max_ps` at `:16359-16364` [S].
- **Clang** `avx512fintrin.h:9448-9450` and `:9579-9581`: `__builtin_ia32_reduce_fadd_ps512(-0.0f, __W)` and
  `__builtin_ia32_reduce_fmax_ps512(__V)` [S], which lower to the same shape.

Both expand to the same four-level pairwise tree, so the **grouping is compiler-independent**:

```
R_add(v[0..15]):  u[j] = v[j] + v[j+8]  (j=0..7);  t[j] = u[j] + u[j+4]  (j=0..3)
                  a = t[0] + t[2];  b = t[1] + t[3];  res = a + b
R_max(v[0..15]):  u[j] = maxps(v[j+8], v[j]);       t[j] = maxps(u[j+4], u[j])
                  p = maxps(t[0], t[2]);  q = maxps(t[1], t[3]);  res = maxps(p, q)
                  where maxps(a, b) = (a > b) ? a : b      [S] Intel SDM, MAXPS Operation
```

The sum is identical under both because the **grouping** is, not because addition commutes: a reduction's bits
follow its association. Clang's `reassoc llvm.vector.reduce.fadd` licenses any association [M], so that agreement
is a measurement to re-check rather than a guarantee. The grouping is all that a **numeric** result depends on;
the per-node operand order decides a **NaN** one, and that order is not even fixed within one binary — GCC treats
`+` as commutative and reassigns the add tree's registers per token and per `NT`, so a row that makes a NaN has a
different answer in token slot 0 of an `NT = 2` job than in a single-token job, and a different one again under
another GCC version (section 6, [M] `pr21-x86-ground-truth.md:258-342`). So `amax` is **not** the only
compiler-dependent value, as this section first claimed: the add tree's operand order is a second, and
`p = d * scale`'s a third (GCC emits `SRC1 = dd`, Clang `SRC1 = scale`). The other two are invisible to a NaN
activation, since `act_quant_q8_1` never writes a NaN; they are reached by a NaN fp16 block scale or by a forged
`ActQ`. `amax` is the one a NaN activation reaches:
GCC's first two levels take `src1 = hi` as written above, Clang's
take `src1 = lo`, and MAXPS returns SRC2 on an unordered compare. The port writes GCC's order, since basalt is
Linux-only (`AGENTS.md:3`, `README.md:33`), so its default `c++` is g++ (**Inference**); `runtime/src/basalt_cpu.cpp`
records both orders, and `tests/kernels/test_basalt_cpu_q2.py` case A6 asserts one while printing the other, so the
choice flips in one line if basalt's build is ever shown to be Clang. Clang's `fmax` reduction carries `nnan`, so
under Clang that value is undefined for a NaN chunk rather than merely differently ordered, and the low-half tree
is the measured realisation of it [M].

## 2. The one output whose bits may differ

`res[t]` can differ where the kernel **generates** a NaN from operands that are not NaN. An infinite activation
element gives its chunk `scale = +inf`, and then `d * scale` with `d = ±0`, the accumulator's fused multiply-add on
a lane whose four codes are all zero, the same on the correction, or `inf - inf` in the final subtraction each
produce one. AArch64 writes its default NaN, `0x7fc00000` [M]; x86 writes the floating-point indefinite,
`0xffc00000`, with the sign bit set ([S] Intel SDM Vol. 1 4.8.3.7 and Table 11-1; **[M]** on a Xeon Platinum 8488C
running basalt's own kernels, `pr21-x86-ground-truth.md:22-32`, after this note was first written without an
AVX-512 machine). `act_quant_q8_1` cannot emit one, since
`amax > 0` is an ordered compare and `s = inf` requires an infinite lane, whose `sum <= -127` makes
`hx = s * sum = -inf` rather than `inf * 0`. The port's comparison was therefore bitwise everywhere and NaN-ness on
`res`; **section 6 closes that carve-out**, since the x86 bits are fixed at every site, and the port now computes
them — `runtime/src/basalt_cpu.cpp`'s `x86_row()` recomputes any NaN `res[t]` under x86's rules, and the tests
compare all 32 bits.

## 3. The floating-point environment

`grep -rn "setcsr|FLUSH_ZERO|DENORMALS_ZERO|ffast-math|Ofast|ffp-contract|crtfastmath|_MM_SET_"` over basalt's
`CMakeLists.txt`, `src` and `include` returns nothing [M], so basalt runs on the default MXCSR: round to nearest
even, every exception masked, flush-to-zero off, denormals-are-zero off (**Inference** from that absence plus the
x86-64 ABI default). macOS on arm64 leaves FPCR at 0 (round to nearest, flush-to-zero off), so subnormals survive on
both sides and the subnormal-scale case of section 4 is reachable here too.

## 4. The quantizer's reachable edge cases

`s = amax > 0 ? amax / 127 : 0` and `inv = s > 0 ? 1 / s : 0`, then `t = x * inv`, `r = t + copysign(0.5, t)` and
`VCVTTPS2DQ` (`expert.cpp:352-362`). The conversion returns the **integer indefinite** `0x80000000` for a NaN, for
either infinity and for anything outside int32 ([S] Intel SDM Vol. 1, Table 11-1 and D.4.2.2), which the clamp to
[-127, 127] then takes to **−127**, never +127. Three consequences, all reachable from finite input:

- A chunk whose `s` falls at or below `2^-128` has `inv = +inf`, so its zeros give `0 * inf = NaN` and its nonzeros
  `±inf`: **every code is −127**, zeros included.
- An infinite element gives `s = inf`, `inv = 0` and `t = inf * 0 = NaN`: that element is −127 while the finite
  ones, including `−0`, take the `>= 0` branch and give 0.
- `amax` at or below `63 * 2^-149` rounds `s` to zero, so `inv = 0` and every code is 0. From `64 * 2^-149` the
  quotient rounds up to the subnormal `s = 2^-149`, which falls in the `inv = +inf` band above [M].

The +127 side of the clamp needs a NaN in the chunk, which lets the `reduce_max` tree swallow the true peak:
`x = 0.1` everywhere with `x[4] = 0.11` and `x[16] = NaN` gives `amax = 0.1`, so `x[4]` scales to `r = 140.2` and
clamps to `+127` [M]. Without a NaN it is inert: wherever `inv` is finite, `|x| <= amax` holds `r` at 127.50003 or
below [M], measured at the exhaustive worst case in case A8. ARM's `FCVTZS` **saturates** instead (NaN gives 0, `+inf` gives
`INT32_MAX`), which after the clamp would give code 0 and +127 where x86 gives −127 for both, so the port selects on
`vcagtq_f32` against `2^31` and substitutes `INT32_MIN` itself.

## 5. Where NEON is not x86

| x86 | NEON | Why they differ |
|---|---|---|
| `_mm512_and_ps(x, 0x7fffffff)` | `vabsq_f32` | identical |
| `_mm512_max_ps`, every node of the trees | `vbslq_f32(vcgtq_f32(a, b), a, b)` | `vmaxq_f32` returns the NaN and `vmaxnmq_f32` the number; MAXPS returns SRC2 on any unordered compare and on `(±0, ∓0)`. The select is MAXPS exactly, and it keeps the operand order of section 1 visible |
| `_mm512_reduce_add_ps`, `_mm512_reduce_max_ps` | the explicit four-level trees | `vaddvq_f32` gives `(t0+t1)+(t2+t3)` where x86 gives `(t0+t2)+(t1+t3)`, and `FMAXV` has the wrong NaN rule |
| `_mm512_mul_ps` then `_mm512_add_ps` | `vmulq_f32` then `vaddq_f32`, built `-ffp-contract=off` | two roundings; a contracted `fma` skips the rounding of `t` and changes codes. Apple clang keeps them apart at its default while they are separate statements and fuses them under `fast` [M], so the flag pins the guarantee |
| `_mm512_fmadd_ps` (`acc`, `corr`) | `vfmaq_f32` | one rounding; fusion is required here, which is why contraction is disabled on the file and not globally |
| `_mm512_cvttps_epi32` | `vbslq_s32(vcagtq_f32(2^31, r), vcvtq_s32_f32(r), INT32_MIN)` | section 4 |
| `_mm512_dpbusd_epi32` | `vdotq_s32` | SDOT is signed by signed, VPDPBUSD unsigned by signed, equal here because codes are 0..3. NEON vector `j` lane `k` is the x86 dword lane `4j + k` |
| `_mm512_multishift_epi64_epi8` | `vqtbl1q_u8`, `vshlq_u8`, `vandq_u8` | the port decodes the documented rule `code m = (qs[m / 4] >> 2 * (m % 4)) & 3` (`ggml/src/ggml-quants.c:220-222`), which is what VPMULTISHIFTQB produces |
| `_mm256_cvtph_ps` | `vcvt_f32_f16` | both exact for every half including subnormals at the defaults of section 3 [M] |
| `_mm512_cvtepi32_epi8`, `_mm512_reduce_add_epi32` | `vmovn_s32` with `vmovn_s16`, `vaddvq_s32` | exact after the clamp; the integer sum is order-independent (`abs(sum) <= 4064`) |

## 6. Which NaN each site returns, and where x86 and AArch64 part

This section supersedes three claims made earlier here: section 1's "only compiler-dependent value" (the operand
order of `p = d * scale` and of the add tree are two more, reachable not from a NaN activation but from a NaN fp16
block scale or a forged `ActQ`), section 2's carve-out that the port's comparison is NaN-ness rather than bits on
`res`, and section 2's "[M, both measured]" on the x86 indefinite. The emitted form governs, not the written one:
GCC's `avx512fintrin.h:16297-16306` writes the add tree's level 3 as `__T6 + __T7`, and GCC emits `vaddps` with
`__T7` as SRC1.

**How the output was obtained.** This section was first written without an AVX-512 machine, so its x86 *results*
were **[S]**, the SDM's rule applied to an instruction form that was itself **[M]**: one translation unit,
`include/basalt/kernels/cpu/expert.hpp` followed by `src/kernels/cpu/expert.cpp`, the two functions byte-identical
to the originals (`diff` over `:336-400` and `:500-560`), through the Compiler Explorer API
(`POST godbolt.org/api/compiler/g142/compile`, and `g144`) with **`-O2`** `-std=c++20 -Wall -Wextra` plus basalt's
per-file flags (`CMakeLists.txt:673-674`), Intel syntax, so the first register after the destination is SRC1.

They are now **[M]**. The PR #21 reviewer ran basalt's own `act_quant_q8_1` and `q2_0_gguf_rows_multi`, verbatim,
on a Xeon Platinum 8488C (AVX-512) at basalt's Release flags — **`-O3 -DNDEBUG`** plus the `-m` list — under GCC
12.5, 13.4, 14.3, 14.4, 15.2 and 15.3, with per-instruction operand probes, end-to-end fixtures, a ~700,000-case
differential and a lane-level scalar model that reproduces the real kernel with **0 mismatches in 62,374 outputs**
on each of four GCC versions: `pr21-x86-ground-truth.md` (`.context/basalt-port/pr21-ground-truth/`, cited below
as `GT:<line>`). Where that measurement and the `-O2` inference below disagree, the measurement governs and the
row says so. The AArch64 column was measured on an
Apple M3 Max, macOS 27.0.1, FPCR = 0 (DN = 0, FZ = 0) [M], with single-instruction probes and with
`reduce_add_ps` replayed verbatim from `runtime/src/basalt_cpu.cpp`. Probe inputs must be `volatile`:
LLVM's constant folder does not follow FMLA's NaN rule and folds case A below to x86's answer [M].

**x86's two rules** [S]. For the two-source instructions (`VMULPS`, `VADDPS`, `VSUBPS`, `VADDSS`, `VSUBSS`,
`VMULSS`), Intel SDM Vol. 1 Table 4-7, "Rules for Handling NaNs": two QNaNs give the "First source operand", two
SNaNs or a mixed pair give the first source operand converted to a QNaN, and the table's note fixes which that is
— "For AVX instructions, the first source operand is usually the 2nd operand in a non-destructive source syntax".
`MAXPS` is the documented exception (section 5). For the FMA instructions, Vol. 1 §14.5.1 maps the mnemonic's three
digits onto the arithmetic equation — first digit the multiplicand, second the multiplier, third the addend, each
naming an operand position — and Table 14-17 gives the priority as **x (multiplicand), then y (multiplier), then z
(addend)**: `Q(x)` whenever x is a NaN whatever y and z are, `Q(y)` when only y and z can be, `Q(z)` only when
neither multiplicand nor multiplier is a NaN. When a NaN is **created** rather than propagated — `INF * 0`, or
`x*y` and `z` infinite with opposite signs — Table 14-17 gives `QNaNIndefinite`, defined there as "the QNaN which
has the sign bit equal to 1, the most significand field equal to 1, and the remaining significand field bits equal
to 0", i.e. `0xffc00000`. For the two-source instructions the definitive source is Table 11-1, "Masked Responses of
SSE/SSE2/SSE3 Instructions to Invalid Arithmetic Operations" (§11.5.2.1, and Intel AVX follows SSE here per
§4.8.3.5): "Addition of opposite signed infinities or subtraction of like-signed infinities → Return the QNaN
floating-point Indefinite" and "Multiplication of infinity by zero → Return the QNaN floating-point Indefinite"
— §4.8.3.7 names that value, and Table 14-17's quoted definition gives its bits. The same table fixes `MAXPS`
("Return the source 2 operand value") and the integer indefinite of section 4. The indefinite's bits do not
depend on the compiler, the instruction form or the operand order.

**AArch64's two rules** [M, this machine]. `FADD`/`FSUB`/`FMUL` return op1's NaN when both sources are NaNs
(`fadd A, B` → A, `fadd B, A` → B), the same first-source rule as x86. Vector `FMLA Vd, Vn, Vm` (`vfmaq_f32`)
returns the **addend** first, then Vn, then Vm (`fmla d=A n=B m=C` → A; `fmla d=1 n=A m=B` → A; `fmla d=1 n=B m=A`
→ B) — the reverse of x86's ranking. Every created NaN is the default NaN `0x7fc00000`, sign clear (`0*inf`,
`inf-inf`, `inf+-inf`). One extra AArch64 case has no x86 counterpart: when the product is `0 * Inf`, a QNaN
addend is **discarded** for the default NaN (`fmla d=A n=0 m=inf` → `7fc00000`), where x86's Table 14-17 returns
`Q(z)`, the addend. SNaN quieting agrees on both (`0x7f800005` → `0x7fc00005`).

| Site | GCC emits (basalt's Release `-O3` build) | x86 returns | AArch64 / the port returns |
|---|---|---|---|
| `p = d * scale` (`:518`) | `vmulps p, dd, scale` — SRC1 = `dd`, in all six GCC versions [M] `GT:156` | both NaN → **`d`**; `0 * ±inf` → `0xffc00000` [M] `GT:204-217` | `fmul` op1 = `d` → **`d`**, same; created → `0x7fc00000`. The port's `x86_row()` writes the x86 rule explicitly |
| `corr = fmadd(dd, hx, corr)` (`:519`) | `vfmadd231ps corr, dd, hx` at **every** NT and token, in all six versions [M] `GT:157`. (The `-O2` run below read `vfmadd213ps` at NT ≥ 3; that was an artefact of its constant-propagated clones, and the ranking was the same either way) | **`dd` > `hx` > `corr`**; `dd * hx = 0 * inf` → `0xffc00000`, unless `corr` is a NaN, which Table 14-17 then returns [M] `GT:221-241` | `FMLA`: **`corr` first**; created → `0x7fc00000`, *including* when `corr` is a NaN and the product is `0 * inf`. `x86_row()` restores the x86 ranking |
| `acc = fmadd(perm(p), cvt(dot), acc)` (`:528`) | `vfmadd231ps acc, P, D` — priority `P`, `D`, `acc`, and `D` is a `vcvtdq2ps` result, never a NaN [M] `GT:158`. GCC 15.2/15.3 swap `P` and `D` at NT = 5, tokens 0-3, which cannot change the answer because `D` is never a NaN | effective **`P` > `acc`** [M] `GT:245-254` | **`acc` > `P`**; `x86_row()` restores it |
| add tree, levels 1–2 (`:532`; the `acc` and `corr` trees are separate reductions) | `vaddps u, hi, lo` or `lo, hi` — **the side varies with NT, the token slot and the GCC version** [M] `GT:159`, `:268-342`; the one-token job is `hi` at both levels in every version | the chosen half's NaN | `vaddq_f32(v[2], v[0])`, op1 = high → the one-token job's side. `x86_row()` pins that order for every slot |
| add tree, level 3 | `vpermilps t7, t6, 78` then `vaddps a, t7, t6` — SRC1 = the **permuted** vector, in every slot of every version [M] `GT:342` | lane 0 returns **`t[2]`**, lane 1 `t[3]` | `vaddq_f32(t, vextq_f32(t, t, 2))` emits `ext.16b v2, v1, v1, #8` then `fadd.2s v1, v1, v2`, op1 = `t` → returns **`t[0]`** [M] — the order mismatch the vector path keeps and `x86_row()` corrects |
| add tree, level 4 | `vaddss res, T8[0], T8[1]` — lane 0 or lane 1, varying with the slot as levels 1–2 do [M] `GT:159`, `:268-342`; lane 0 in the one-token job | the chosen lane's NaN | `vgetq_lane(p,0) + vgetq_lane(p,1)`, op1 = lane 0 → the one-token job's lane |
| `res = reduce(acc) - reduce(corr)` (`:532`) | `vsubss res, acc_sum, corr_sum` — SRC1 = the `acc` sum, all six versions [M] `GT:160` | the **`acc`** sum's NaN; `inf - inf` → `0xffc00000` [M] `GT:346-353` | `fsub` op1 = the `acc` sum → same; created → `0x7fc00000`, which `x86_row()` replaces with the indefinite |
| `amax` max tree (`:352`) | `vmaxps` per section 1 | `MAXPS` returns SRC2 on any unordered compare | the `maxps()` select, same — and a NaN `amax` then dies at `vcomiss`: `amax > 0.f` is false, so `s = +0` |
| `t = x * inv` (`:356`) | `vmulps t, x, inv` — SRC1 = `x` | `x`'s NaN; `0 * inf` → `0xffc00000` | op1 = `x` → same; created → `0x7fc00000` |
| `r = t ± 0.5` (`:357-358`) | `vaddps r, t, blend` — SRC1 = `t` | `t`'s NaN | `vaddq_f32(t0, c)`, op1 = `t` → same |
| `hx = s * (float) sum` (`:367`) | `vmulss hx, (float) sum, s` — GCC **swaps** the written order, SRC1 = `(float) sum` | inert: `sum` is a `vcvtsi2ss` result, so at most one operand is ever a NaN | same |

The last four rows never reach an output. A NaN `amax` is killed by the ordered compare; a NaN `t` or `r` is killed
by `VCVTTPS2DQ`, which returns the integer indefinite that the clamp takes to −127 (section 4); and `s * sum` with
`s = +inf` needs `sum = 0`, which an infinite lane's −127 forbids (section 2). So `act_quant_q8_1` writes no NaN,
and the only NaN-valued input the row kernel can see from basalt's own pipeline is a row's own fp16 block scale.

**Is the choice compiler-dependent?** (a) The FMA priority is not: all six measured GCC versions make `dd` the
first multiplicand and `corr` the addend, and `P` the first multiplicand with `acc` the addend [M] `GT:154-160`,
and Clang 21 the same [M, here]. The *form* does vary with the optimisation level — the `-O2` clones emit `vfmadd213ps`,
basalt's Release `-O3` build `vfmadd231ps` everywhere — without changing the ranking, since each form's own
formula names the same operand first. (b) `p = d * scale` **is** compiler-dependent: GCC emits SRC1 = `dd`, Clang
`vmulps p, scale, dd`, SRC1 = `scale` [M], so under a Clang-built basalt an `a->scale` NaN would beat the block
scale's. (c) The add tree's operand order is compiler-dependent **and** slot-dependent: GCC 12.5 differs from 14.3
in 19 of the 36 (NT, token) slots, 13.4 in 15, 15.2 and 15.3 in 6, and 14.4 in none [M] `GT:326-340`; Clang mirrors
levels 1–2 and takes the unpermuted vector at level 3 [M]. Level 3 alone is fixed: the shuffle is SRC1 in every
slot of every version [M] `GT:342`. basalt's build is g++ (section 1's inference), so the port follows the GCC
column, at the one-token slot. (d) The subtraction and the created-NaN bits are fixed by the ISA and depend on
nothing [M] `GT:346-353`.

**The inputs that separate them.** The port side was measured here by replaying its own operations [M]; the x86
side is now measured too, each case through basalt's own kernels on the AVX-512 host [M]. Codes `0x55` make every
weight code 1, so each `VPDPBUSD` dword lane holds 4 * 127 = 508.

| Case | Input | x86 | the port, before `x86_row()` |
|---|---|---|---|
| A — the `acc` FMA | `nblocks = 2`, fp16 block scales `0x7e01` then `0x7e02` (→ `0x7fc02000`, `0x7fc04000` [M]), codes `0x55`, 128 activations of `1.0` (`s = 1/127`, `hx = 32`) | `0x7fc04000` — block 1's scale is the first multiplicand [M] `GT:40`, fixture (b), every (nt, t) | `0x7fc02000` — the accumulator is the addend |
| B — add tree level 3 | finite block scales, every `a->scale` zero, `a->hx` with two payloads four lanes apart; a forged `ActQ`, since `act_quant_q8_1` writes no NaN, so this one is latent rather than reachable | the shuffle's, i.e. lane 2's [M] `GT:313`, probe P2, every (nt, t) | lane 0's |
| C — any created NaN | 64 copies of `2^124` give `scale = 1.674618e+35` (`0x7a010204`) and `hx = +inf` with no infinity in the input; one block, `d = 1.0`, codes `0x55`, so each `acc` lane stays finite at 8.51e37 and **`acc`'s reduction** overflows to `+inf` against `corr = +inf`: `res = inf - inf` | `0xffc00000` [M] `GT:29`, fixture (a), every (nt, t) | `0x7fc00000` |
| D — the reverse direction | a QNaN `acc` lane meeting an `inf * 0` product: block 1's codes are all zero while its chunk scale is `+inf`, against an `acc` already holding a QNaN | the QNaN survives — Table 14-17's `Q(z)`, the addend [M] `GT:53`, fixture (c1), every (nt, t) | `0x7fc00000` — the addend discarded |

**The smallest correct rule.** NaN-*ness* never differs: both ISAs make a NaN at the same sites from the same
inputs, so `isnan(res[t])` is true on exactly the same rows — measured over 21,635 NaN outputs of the large
differential, where every non-NaN output and every NaN *position* agreed and only payloads differed [M]
`GT:101-103`. Only the bits differ, in three ways — the created-NaN
payload (every site, always `0xffc00000` against `0x7fc00000`), the FMA's ranking (cases A and D), and the add
tree's operand order (level 3 in every slot, case B; levels 1, 2 and 4 wherever x86's slot order is not the
one-token one). The third is *not* repairable by swapping the two arguments in the C source: LLVM's LangRef,
"Behavior of Floating-Point NaN values", gives a NaN result "a non-deterministic sign" with "the quiet bit and
payload ... non-deterministically chosen" between the preferred all-zero payload and a copy from any NaN input
[S], so no NEON expression written in C is entitled to a particular payload at all — the AArch64 column above is
this compiler's and this chip's measured behaviour, not a contract. Only explicit bit tests are. That gives a
two-case rule, which `runtime/src/basalt_cpu.cpp` implements as one scalar replay at the single write of `res[t]`:

1. If no NaN enters the row — none among its `nblocks` fp16 block scales (one per 18-byte block) and none among
   the `2 * nblocks` `ActQ` `scale` and `hx` values it reads —
   then any NaN in `res[t]` was created, and every created NaN on x86 is the indefinite; a propagated indefinite
   keeps its bits, so the answer is exactly `0xffc00000`. Measured on all 2,506 such outputs of the model run,
   identical under GCC 12.5, 13.4, 14.3 and 15.2 [M] `GT:375-378`; it covers the whole case reachable from
   `act_quant_q8_1`'s own output (section 2). The blanket rewrite "`0x7fc00000` →
   `0xffc00000`" is **not** equivalent and is wrong on an ordinary input: an fp16 block scale of `0x7e00`, the
   canonical half NaN, converts to `0x7fc00000` [M] and must survive unchanged, while `0xfe00` propagates to
   `0xffc00000` [M] and must not be read as a created one.
2. If a NaN does enter, the row's result needs x86's rankings, which the NEON path cannot give at the two FMA
   sites — and x86 has **no single answer** there: the two halves of `acc` can end on different NaNs, and which
   one the add tree keeps moves with the token slot, the token count and the GCC version (1,103 of the model's
   kind-0 outputs differ between slots of the 14.3 build, 309 between versions) [M] `GT:379-389`, so basalt's own
   output contradicts its promise that "Every token's output is bitwise the single-token job's"
   (`include/basalt/kernels/cpu/pool.hpp:54-55`). The port therefore pins **GCC 14.3's one-token order for every
   slot**, which is basalt 14.3's single-token job bit for bit and keeps the promise. That is a second
   build-dependent value beside `amax`.
   The replay is **lane-exact**, not a single-accumulator loop: 16 lanes; lane `L` of the
   group at `b0` takes block `b0 + L/2`'s `dd` and `a->hx[2*b0 + L]`; `acc` lanes 0-7 and 8-15 take `p[2i]` and
   `p[2i+1]`; and both trees use the pairings and SRC1 sides of the table. A single-accumulator loop gets case B
   wrong, and every row whose `corr` lanes hold different blocks' NaNs with it. It selects NaNs by
   **explicit bit tests** at each site, never by letting
   native float arithmetic choose: `fmaf` lowers to `FMADD`, which is addend-first with the default NaN, so a
   replay written in plain C reproduces exactly the two divergences it is there to remove. It runs on no finite
   row, so it costs nothing, and it subsumes case 1.

The port in that form reproduces the reviewer's lane-level model — the one validated against basalt's real kernel
with 0 mismatches in 62,374 outputs — bit for bit on all three activation kinds, at `-O0`, `-O2` and `-O3` [M], and
matches every recorded fixture and probe at `nt = 1` (`tests/kernels/test_basalt_cpu_q2.py`, group C). What remains
different from a particular basalt binary is only case 2 at `nt > 1`, or under a GCC other than 14.3/14.4.

## 7. The supported input range

A decision of the engineering review. basalt's `act_quant_q8_1` writes no NaN `scale` or `hx` for any float input
(**Inference** from `expert.cpp:352-367`). A NaN `amax` fails the ordered `amax > 0.f`, so `s = +0` (`:353`) and
`hx = +0 * sum` is a zero (`:367`). A finite `amax` gives a finite `s`, so `hx` is finite or overflows to an infinity.
`amax = +inf` needs an infinite lane, whose `t = inf * (1.f / s) = inf * 0` (`:354`, `:356`) is a NaN that `VCVTTPS2DQ`
and the clamp take to −127 while every finite lane's `t = ±0` gives 0 (`:357-361`, section 4), so `sum <= -127` (`:364`)
and `hx = +inf * sum = -inf`. Case A11 asserts the exact outputs [M]: a single `+inf` gives `scale = +inf`,
`sum = -127`, `hx = -inf` at every position; a single NaN gives −127 at its own position, and where it also reaches
`amax` (position 19 alone, under GCC's tree) `scale = +0`, `sum = -127`, `hx = -0.0`; `+inf` with `-inf` gives
`sum = -254` and `hx = -inf`. The port at `-O0` and at the Release flags wrote no NaN `scale` or `hx` over 3,681,564
chunks drawn with NaN, ±inf, huge, subnormal and zero elements, and every `+inf` scale came with `sum <= -127` and
`hx = -inf` [M]. A NaN therefore enters a row only through a NaN fp16 block scale in the weights, which a valid model
file does not hold. The pinned GCC 14.3 single-token rule of section 6's case 2 decides only rows of a corrupt weight
file and test inputs (a NaN block scale, or a NaN planted in `scale` or `hx` through the binding); every NaN generated
from non-NaN inputs, an infinite activation's included, is `0xffc00000` under every build measured — basalt's under GCC
12.5, 13.4, 14.3 and 15.2 [M] `GT:375-378`, the port's at `-O0` and at the Release flags, all 316,278 NaN outputs of
316,366 from rows with no NaN block scale fed those activations [M].
