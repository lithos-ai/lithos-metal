// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_gr_norm.cu:37-100
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which adapts ggml-org/llama.cpp
// ggml/src/ggml-cuda/{norm.cu,common.cuh} @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's grouped RMSNorm, native_gr_rms_norm_weighted (weighted_rms_norm<NORM_BLOCK>): row r of x (n_cols floats)
// becomes y[c] = (scale * x[c]) * gamma[c], scale = rsqrtf(sum of the row's squares / n_cols + eps), with row
// r % gamma_rows of gamma (gamma_rows = n_rows is basalt's gamma += row_offset; the n-gram block's three norms pass
// 4 rows of 2560 with 4 gamma rows). Needs basalt_fp.metal ahead of it.
//
// Thread t = 32 * sg + lane is basalt's threadIdx.x, so its columns and its slot do not depend on how Metal orders the
// SIMD-groups: it squares columns t, t + NORM_BLOCK, ... in order, fma(v, v, partial) from +0; each SIMD-group reduces
// with basalt's XOR butterfly; lane 0 of each stores its sum into slot sg; after the barrier every SIMD-group (not only
// the first) loads the slots (lane < NORM_BLOCK / 32; +0 above) and runs the butterfly again, so every thread holds
// the same sum and computes the scale itself: rsqrtf(fma(sum, 1 / n_cols, eps)), the division fused with the add as
// ptxas fuses it. Every float step is .FTZ (fma_ftz, add_ftz, mul_ftz, rcp_ftz for MUFU.RCP, rsqrtf_ftz for
// MUFU.RSQ); a NaN is stored as sm_120's (cuda_nan; docs/research/basalt-nvcc/nan.md).
//
// Macros: NORM_BLOCK threads per threadgroup, basalt's choice of 256 below 1024 columns and 1024 from there
// (native_gr_norm.cu:93-96; monolith/kernels.py's basalt_gr_norm_macros). The block size fixes the reduction order,
// so the pipeline must accept NORM_BLOCK threads. Grid: n_rows threadgroups of NORM_BLOCK threads.
//
// The rounding sequence nvcc compiles for this kernel: docs/research/basalt-nvcc/ple.md
#pragma STDC FP_CONTRACT OFF

#ifndef NORM_BLOCK
#define NORM_BLOCK 1024
#endif

struct GrNormParams { uint n_cols; uint gamma_rows; uint eps_bits; uint pad; };

kernel void basalt_gr_norm(device const float* x [[buffer(0)]], device const float* gamma [[buffer(1)]],
                           device float* y [[buffer(2)]], constant GrNormParams& p [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint t = 32u * uint(sg) + uint(lane);                   // basalt's threadIdx.x: warp t / 32, lane t % 32
    device const float* in = x + ulong(r) * p.n_cols;
    device const float* g = gamma + ulong(r % p.gamma_rows) * p.n_cols;
    device float* out = y + ulong(r) * p.n_cols;
    float partial = 0.0f;
    for (uint c = t; c < p.n_cols; c += NORM_BLOCK) {
        const float v = in[c];
        partial = fma_ftz(v, v, partial);
    }
    threadgroup float sums[32];
    partial = warp_sum_ftz(partial);
    if (lane == 0) sums[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial = lane < NORM_BLOCK / 32 ? sums[lane] : 0.0f;
    partial = warp_sum_ftz(partial);
    const float scale = rsqrtf_ftz(fma_ftz(partial, rcp_ftz(float(p.n_cols)), as_type<float>(p.eps_bits)));
    for (uint c = t; c < p.n_cols; c += NORM_BLOCK) out[c] = cuda_nan(mul_ftz(mul_ftz(scale, in[c]), g[c]));
}
