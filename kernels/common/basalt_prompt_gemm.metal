// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/gemm.cu:122-185 and src/prefill/kernels.cu:31-40, :818-860
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
//
// basalt's prompt-path products: Gemm::bf16 (gemm.cu:122-132), BF16 x BF16, and Gemm::native's second stage, Gemm::f16
// (:158-167) on the FP16 image of a GGUF matrix (:169-185), FP16 x FP16. Each is one cublasGemmEx with an fp32 output
// and compute type, alpha 1 and beta 0; gemm.cu has no device code (docs/research/basalt-nvcc/prefill_kernels.md,
// Method), so cuBLAS picks the summation order and this port fixes its own: every output is one fp32 fma chain from
// +0 in increasing k,
//
//     acc = 0; for k in 0 .. K - 1: acc = fma(x[t][k], w[n][k], acc);  y[t][n] = cuda_nan(acc)
//
// on the operands widened exactly (h2f; a BF16 field as the high half of a float). Metal's fma reads a subnormal BF16
// operand as the zero of its sign and flushes a partial sum whose exact value is below 2^-126 (a lone product
// included): the order is the port's, not basalt code. Widened FP16 operands, their fp32 products and the partial sums
// are never subnormal in fp32 (nonzero ones are multiples of 2^-24 and 2^-48). A NaN is stored as sm_120's 0x7fffffff
// (cuda_nan). Each output's bits depend only on its x row, its w row and K: not on the chunk's rows, the tile or the
// slice that computes it.
//
// basalt_prompt_gemm: x [B][K] (bf16 or fp16 bits), w [N][K] (the same type; a BF16 tensor in place or the FP16
// image), y [B][N] floats. Rows come from StepState through prompt_rows (monolith/kernels.py's PROMPT_ROWS, ahead of
// this file with basalt_fp.metal and the StepState struct): R = prompt_rows(st). A threadgroup computes a 64 x 64 tile
// of y, rows row0 + 64 * gy and columns col0 + 64 * gx onward (row0 and col0 are the slice's, from its params record):
// 256 threads, each 4 x 4 outputs (rows 4 tr + i, columns 4 tc + j), K in steps of 16 staged as floats in one 8,192 B
// threadgroup array (quads of 4 rows as float4, their slots XOR-swizzled by kk), the next step's 16-bit operands
// loaded into registers while this step's fmas run: the speed a dispatch needs to stay under 1 ms at K 10240 (C4),
// with every output's chain unchanged. A tile at or past R returns before its first barrier; loads past R or N stage
// zeros (never read past x's B rows or w's N rows); a thread none of whose outputs is stored skips its fmas but meets
// every barrier; stores need t < R and n < N. Grid: (ceil(ncols / 64), nrows / 64) threadgroups of 256 per slice.
// K % 16 == 0.
//
// basalt_prompt_convert: to_f16 (kernels.cu:818-821, hf :40) at CV_BF16 0 and to_bf16 without ylo (:826-833, bf
// :31-35) at CV_BF16 1, x [B][n] floats to y [B][n] fp16 or bf16 bits. One thread per element of the chunk's rows,
// i < prompt_rows(st) * n (basalt's launchers, :846-850 and :856-860, run a grid-stride loop of
// min(ceil(count / 256), 4096) blocks of 256 over the element count; every element is independent, so the geometry
// changes no bit).
// hf is __float2half_rn: round to nearest even, a finite overflow to inf, fp16 subnormals kept, an fp32 subnormal to
// the zero of its sign, every NaN to sm_120's 0x7fff (f16_bits, then cuda_nan_f16). bf is integer arithmetic on the
// loaded bits (bf16_rne_nonan): round to nearest even, a finite value from 0x7f7f8000 in magnitude to inf, an fp32
// subnormal rounded by its bits (a load and as_type keep them), and no NaN case, so a NaN's bits round like any
// others, with 32-bit wrap-around: sm_120's 0x7fffffff gives 0x8000 (-0), 0xffffffff 0x0000, 0x7fc00000 0x7fc0. x
// is only loaded, so it keeps its producer's NaN (a ported kernel stores cuda_nan's) and takes no cuda_nan here.
// Grid: ceil(B * n / 256) threadgroups of 256.
//
// Macros: GEMM_BF16 (1: BF16 operands, 0: FP16), CV_BF16 (0: to_f16, 1: to_bf16), and PROMPT_ROWS' T_SRC, ROWS_MAX
// and T_STATIC_ROWS.
#pragma STDC FP_CONTRACT OFF

#ifndef GEMM_BF16
#define GEMM_BF16 0
#endif
#ifndef CV_BF16
#define CV_BF16 0
#endif

// ponytail: one geometry for every shape (a 64 x 64 tile, 4 x 4 outputs per thread, K steps of 16, one single-buffered
// threadgroup array beside a register prefetch); a per-shape geometry macro, gated unchanged by GM-EMU and GM-ROWS
// (tests/kernels/test_basalt_prompt_gemm.py), when prompt speed is measured to matter.
#define PG_TILE 64u
#define PG_BK 16u

struct PromptGemmParams { uint n, k, row0, col0; };

static inline float pg_widen(ushort b) {
#if GEMM_BF16
    return as_type<float>(uint(b) << 16);
#else
    return h2f(b);
#endif
}

kernel void basalt_prompt_gemm(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                               device float* y [[buffer(2)]], constant PromptGemmParams& p [[buffer(3)]],
                               device const StepState* st [[buffer(15)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    // [0][kk][q ^ kk]: x[t0 + 4q + 0..3][k0 + kk]; [1][kk][q ^ kk]: w[n0 + 4q + 0..3][k0 + kk] (8,192 B)
    threadgroup float4 s[2][PG_BK][PG_TILE / 4u];
    const uint t0 = p.row0 + PG_TILE * tg.y, n0 = p.col0 + PG_TILE * tg.x;
    const uint R = prompt_rows(st);
    if (st->done || t0 >= R) return;                      // the whole threadgroup, before any barrier
    const uint tr = tid % 16u, tc = tid / 16u;            // outputs: rows t0 + 4 tr + i, columns n0 + 4 tc + j
    const uint sq = tid / 16u, skk = tid % 16u;           // staging: quad sq (rows or columns 4 sq + c) at kk = skk
    const bool live = t0 + 4u * tr < R && n0 + 4u * tc < p.n;  // else none of the thread's 16 outputs is stored
    device const ushort* xs[4];
    device const ushort* ws[4];
    bool xin[4], win[4];
    for (uint c = 0; c < 4u; ++c) {                       // a row past R or N stages 0 and points at row 0 (in bounds)
        const uint t = t0 + 4u * sq + c, n = n0 + 4u * sq + c;
        xin[c] = t < R; win[c] = n < p.n;
        xs[c] = x + ulong(xin[c] ? t : 0u) * p.k + skk;
        ws[c] = w + ulong(win[c] ? n : 0u) * p.k + skk;
    }
    ushort rx[4], rw[4];                                  // the step's 16-bit operands, loaded a step ahead
    for (uint c = 0; c < 4u; ++c) { rx[c] = xin[c] ? xs[c][0] : ushort(0); rw[c] = win[c] ? ws[c][0] : ushort(0); }
    float acc[4][4];
    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < p.k; k0 += PG_BK) {
        s[0][skk][sq ^ skk] = float4(pg_widen(rx[0]), pg_widen(rx[1]), pg_widen(rx[2]), pg_widen(rx[3]));
        s[1][skk][sq ^ skk] = float4(pg_widen(rw[0]), pg_widen(rw[1]), pg_widen(rw[2]), pg_widen(rw[3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint k1 = k0 + PG_BK;
        if (k1 < p.k)
            for (uint c = 0; c < 4u; ++c) { rx[c] = xin[c] ? xs[c][k1] : ushort(0); rw[c] = win[c] ? ws[c][k1] : ushort(0); }
        if (live)
            for (uint k2 = 0; k2 < PG_BK; k2 += 4u) {     // kk = k2 + u increasing: every chain sees k in order
                float4 a[4], b[4];
                for (uint u = 0; u < 4u; ++u) { a[u] = s[0][k2 + u][tr ^ (k2 + u)]; b[u] = s[1][k2 + u][tc ^ (k2 + u)]; }
                for (uint u = 0; u < 4u; ++u)
                    for (uint i = 0; i < 4u; ++i)
                        for (uint j = 0; j < 4u; ++j) acc[i][j] = fma(a[u][i], b[u][j], acc[i][j]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // every thread has read the step before the next store
    }
    for (uint i = 0; i < 4u; ++i) {
        const uint t = t0 + 4u * tr + i;
        for (uint j = 0; j < 4u; ++j) {
            const uint n = n0 + 4u * tc + j;
            if (t < R && n < p.n) y[ulong(t) * p.n + n] = cuda_nan(acc[i][j]);
        }
    }
}

struct PromptConvertParams { uint n; };

kernel void basalt_prompt_convert(device const float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                                  constant PromptConvertParams& p [[buffer(2)]],
                                  device const StepState* st [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if (st->done || i >= prompt_rows(st) * p.n) return;
#if CV_BF16
    y[i] = bf16_rne_nonan(x[i]);
#else
    y[i] = cuda_nan_f16(f16_bits(x[i]));
#endif
}
