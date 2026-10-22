# basalt's CPU SwiGLU: which `expf`, which order, which NaN

`runtime/src/basalt_cpu_expf.cpp` is the one piece of arithmetic basalt's CPU expert pool performs itself:
`src/kernels/cpu/pool.cpp:434`, `res = (g / (1.f + std::exp(-g))) * u`. Every other value the pool produces comes
out of the Q2_0 kernels. The port has to return the same bits basalt does, because a token's output has to be the
same number whichever tier produced it, so three things have to be settled rather than assumed: which
implementation of `expf` runs, in which order the three arithmetic instructions execute, and which NaN each one
writes. This records the answers and where they came from.

Evidence tags: [M] measured, [S] vendor specification, [R] third-party report.

## 1. Which `expf`

basalt runs on Linux, so `std::exp` of a float at pool.cpp:434 is glibc's `expf`. Apple's is a different
implementation; neither is correctly rounded, and the two differ at 1,930,129 of the 2,218,786,818 float32
arguments in [-40, 40] [M, this repository against this machine's own libm, Apple clang 21.0.0 on an M3 Max, so
the count is a property of Apple's `expf` and moves with macOS], so calling the platform's cannot be bit-exact and
the algorithm is transcribed instead.

glibc's generic `expf` has been Arm's `optimized-routines` `math/expf.c` since 2017
(`sysdeps/ieee754/flt-32/e_expf.c`) [R]. That source is MIT, where glibc's copy is LGPL-2.1-or-later, so the
transcription is taken from Arm's at `f67e7daa5457646fad9f75580263dba654c988aa` and keeps this tree's licence
position (`third_party/NOTICE`, `third_party/ARM-OPTIMIZED-ROUTINES-LICENSE`).

### The variant

glibc's x86-64 multiarch holds a second build of that same source, `__expf_fma`, compiled with `-mfma -mavx2`, and
an ifunc picks it on any host with FMA and AVX2 [M, `expf@GLIBC_2.27` through the PLT resolving to `libm+0x7dc40`,
glibc 2.39-0ubuntu8.9, Xeon Platinum 8375C and 8488C]. Every host that can run basalt's AVX-512 expert kernels has
both features, so `__expf_fma` is the variant basalt runs and the one the port reproduces.

The two variants are the same function except at the rounding step. Arm writes every multiply-add with operators;
the contraction sites are the three of the polynomial and, once `z = InvLn2N * xd` is substituted, `z + SHIFT` and
`z - kd`. Contracting the polynomial changes nothing at any of the 2^32 arguments [M], because the double carries
29 bits more than the float it rounds back to. Contracting the rounding step changes two: `0x4202422f` and
`0xc27c65d9` [M], because the fused form subtracts `kd` from the unrounded product where the written form
subtracts it from the rounded `z` and is itself exact. `__expf_fma` fuses them; `__expf_sse2` does not and returns
the correctly rounded value there [M, the same hosts with `GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA`]. The port
writes those two steps as explicit `std::fma` so that `-ffp-contract` cannot decide which function it is.

Arm's other rounding branch, the `round`/`lround` one, is not glibc's anywhere: glibc 2.39 carries a single
`sysdeps/ieee754/flt-32/math_config.h`, which defaults `TOINT_INTRINSICS` to 0 and defines it nowhere else, so
every architecture's glibc compiles the shift branch the port transcribes [M, lines 40-44 of that file]. It would
be the same function here in any case: `InvLn2N * (double) x` is never exactly a tie at any float32 argument
[M, swept over all 2^32], so no argument reaches the one place the two branches round differently.

## 2. The Codegen For Line 434

x86-64 gcc 14.3, `-std=c++20 -O3 -DNDEBUG`, no `-m` flags, which is how basalt builds `pool.cpp` [M, Compiler
Explorer, with pool.cpp:425-434 kept structurally intact]:

```
movss   xmm1, [g]            ; g
movaps  xmm0, xmm1
xorps   xmm0, [.LC27]        ; -g, a sign flip (0x80000000), exact
movss   [rsp+8], xmm1        ; spill g across the call
call    expf                 ; e = expf(-g), one scalar call
addss   xmm0, [.LC28]        ; d = e + 1.0f
movss   xmm1, [rsp+8]
divss   xmm1, xmm0           ; q = g / d
mulss   xmm1, [u]            ; res = q * u
movss   [res], xmm1
```

Three single-precision instructions, no fused multiply-add (there is no multiply feeding an add in the
expression), one scalar `expf` call. With `-march=x86-64-v3` the same sequence appears as the VEX encodings, so
the bits do not depend on which of basalt's build configurations compiles the pool [M].

The operand order is the compiler's, not the source's, and it decides the NaN. In Intel syntax the destination is
SRC1: the add takes `e` as SRC1 and the constant as SRC2, which is the source's operands swapped and is exact
since addition commutes and `1.f` is never a NaN; the divide takes `g`, which does not commute; the multiply takes
the quotient, which is the one that could have gone either way.

## 3. Which NaN

x86 writes the floating-point indefinite `0xffc00000`, sign bit set, where AArch64 writes the default NaN
`0x7fc00000` [S, Intel SDM Vol. 1 4.8.3.7; S, Arm ARM A1.5.3]. An SSE scalar arithmetic instruction handed a NaN
returns the first of its sources that is one, quieted, and otherwise the indefinite when the operation itself
creates one. So:

* `g` = -inf gives `e` = +inf, `d` = +inf and `-inf / inf` at the divide: `0xffc00000`.
* The add never creates one. Its only invalid form is `inf + -inf`, and `e` is `expf`'s own return, which is
  never negative at any of the 2^32 arguments [M]. It is written as the instruction for fidelity to what basalt
  executes, and the recorded vectors cannot tell it from the C operator.
* `0 * inf` and `inf * 0` at the multiply: `0xffc00000`. Several recorded rows reach it, including
  `g` = `0xc2b17218` (just past the overflow threshold, where `g / inf` is a signed zero) with `u` = +inf.
* A NaN argument comes back quieted with its payload and sign: `expf` returns `x + x`, which quiets on both
  architectures, and then the divide returns its first source. `0x7f812345` gives `0x7fc12345`.
* With a NaN `g` and a NaN `u` the multiply keeps the quotient's payload, not `u`'s.

`expf` itself needs no help, since `x + x` is the same on both. The add, the divide and the multiply are written
as the x86 instructions rather than as C operators (`x86_add`, `x86_div`, `x86_mul` in the kernel).

One corner of the SSE rule is outside the recorded corpus. SSE returns the first source operand that is a NaN,
where an AArch64 instruction prefers a signalling NaN wherever it sits; the two part only when SRC1 is a quiet NaN
and SRC2 a signalling one, and no recorded row holds that pair. Case D3 asserts it [S] against the two manuals
rather than against a measurement, and says so.

## 4. The Fixtures

Two recorded files under `tests/kernels/fixtures/`, both measured on Compiler Explorer's x86-64 executors (Xeon
Platinum 8375C and 8488C, AVX-512) under glibc 2.39 = Ubuntu 24.04 libc6 2.39-0ubuntu8.9, libm build-id
`d53fcbff9f855463606b25cfbe2a2e10856ea3a7`, MXCSR control `0x1f80` [M, 2026-10-18]. Each carries its own
provenance in its header comment.

`swiglu_vectors.txt`, 4,226 rows of `g u res` as binary32 bit patterns: basalt's whole line, built as basalt
builds it. 57 special `g` against two `u` (the thresholds, the two arguments of section 1, zeros, infinities,
quiet and signalling NaNs, subnormals), 16 special-`u` pairs, and 4,096 random pairs with `|g| < 40` and most of
their mass near zero. Six runs across the two CPUs agree, and every run's own digest of the `res` column is
`dfdc90ecf37deba2`. Test case A2 holds every row.

`expf_digests.txt`, 4,096 digests: `expf` at every one of the 4,294,967,296 float32 arguments. Block k covers bit
patterns `k << 20 .. (k << 20) | 0xfffff` in increasing order, and its digest is the FNV-1a of the result bits,
`h = 0xcbf29ce484222325` then `h = (h ^ bits) * 0x100000001b3 mod 2^64` per argument. The trailer carries the two
blocks `__expf_sse2` digests differently, `0x420` and `0xc27`, which are the blocks holding the two arguments of
section 1. Test case B4 holds every block, through `_native.basalt_expf_digest`.

## 5. What The Port Reproduces

[M, this repository on a 48 GB 40-core M3 Max, Apple clang 21.0.0 at `-O2 -ffp-contract=off`]:

* All 4,096 blocks of `expf_digests.txt`, so `expf_glibc` is `__expf_fma` at every float32 argument.
* All 4,226 rows of `swiglu_vectors.txt`, including every NaN row, and the digest of its own `res` column is
  `dfdc90ecf37deba2`. Blocks 0x7f8-0x7ff and 0xff8-0xfff of the digest sweep are `expf`'s own output at all
  16,777,214 NaN arguments, so glibc's NaN bits are pinned too.
* Built with the rounding step unfused instead, it fails exactly blocks `0x420` and `0xc27`, and its digests at
  those two blocks are the trailer's `__expf_sse2` digests exactly. The port is therefore either variant, and the
  only thing that separates them is the two `std::fma` calls.
