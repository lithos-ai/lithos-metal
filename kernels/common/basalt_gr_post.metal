// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_gr_postops.cu:35-41, :65-74, :107-114 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which adapts ggml-org/llama.cpp
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh} @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT;
// third_party/NOTICE), written in the rounding sequence nvcc compiles for it
// (docs/research/basalt-nvcc/native_gr_postops.md).
//
// basalt's hyper-connection write, native_gr_post, over T tokens at once: stream c of token t gets the weight
// w = 2 sigmoid(inject[t][c] / hc) and out[t][c][d] = fma(block_out[t][d], w, r[t][c][d]), in the sequence of
// native_gr_postops.md section 1. The file is --use_fast_math in basalt, so every operation is .FTZ, here through
// basalt_fp.metal's helpers (needed ahead of this file); MUFU.EX2 and MUFU.RCP, the two approximate operations, are
// expf_ftz and rcp_ftz (contract invariant 5), and a NaN is stored as sm_120's (cuda_nan). One thread per element in
// threadgroups of 256 (basalt's THREADS); the threads past the last element return before any load. Nothing is
// reduced and no thread reads another's values, so a token's bits depend neither on T nor on its row. out may be r's
// buffer (basalt's in-place call: each thread loads its own r[i] before it stores out[i], and no other thread loads
// r[i]); block_out and inject overlap neither.
#pragma STDC FP_CONTRACT OFF

// p: n_embd, hc, T (the program's width), 0. n_embd * hc <= INT_MAX and T * hc * n_embd < 2^32 (basalt_gr_post_params),
// so the 32-bit indices are exact.
kernel void basalt_gr_post(device const float* r [[buffer(0)]],            // residual [T][hc][n_embd]
                           device const float* block_out [[buffer(1)]],    // [T][n_embd]
                           device const float* inject [[buffer(2)]],       // [T][hc], the read's inject_out
                           device float* out [[buffer(3)]],                // [T][hc][n_embd]
                           constant uint4& p [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
    const uint n_embd = p.x, hc = p.y, row = hc * n_embd;
    if (i >= p.z * row) return;
    const uint t = i / row, c = (i % row) / n_embd, d = i % n_embd;
    const float x = fma_ftz(inject[t * hc + c], 1.0f / float(hc), 0.0f);    // FFMA.FTZ R0, R0, UR4, RZ; UR4 = 1 / hc
    const float g = rcp_ftz(add_ftz(expf_ftz(-x), 1.0f));                    // FMUL.FTZ, MUFU.EX2, FADD.FTZ, MUFU.RCP
    const float w = fma_ftz(g, 2.0f, 0.0f);                                  // FFMA.FTZ R5, R0, 2, RZ
    out[i] = cuda_nan(fma_ftz(block_out[t * n_embd + d], w, r[i]));          // FFMA.FTZ R5, R5, R2, R8
}
