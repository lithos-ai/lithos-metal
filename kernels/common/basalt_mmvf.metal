// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_bf16.cu:37-135
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which adapts ggml-org/llama.cpp
// ggml/src/ggml-cuda/{mmvf.cu,common.cuh} @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's BF16-weight, fp32-activation GEMV: y[k * ldy + o] = row o of a BF16 matrix (n_out rows of n_in values,
// row-major, n_in even) times the fp32 activation row x + k * ldx, for k < n_tok. Needs basalt_fp.metal ahead of it.
//
// One kernel for basalt's one-row kernel (NT 1: bf16_f32_mmvf_kernel) and its multi-row kernel (NT 4 or 8:
// bf16_f32_mmvf_multi_kernel), which give every output the same operations in the same order (native_bf16.cu:75-77):
// thread t of threadgroup o takes the bf16 pairs t, t + BLOCK, ... of row o in order and, for each row k < n_tok of x,
// adds fma(bf(w_lo), x.x, acc) then fma(bf(w_hi), x.y, acc) from +0; each SIMD-group reduces with basalt's XOR
// butterfly; with more than one SIMD-group, lane 0 of each stores its sum into a zeroed 32-slot array and SIMD-group 0
// runs the butterfly over all 32 slots; thread 0 writes. The float steps are the ones nvcc compiles
// (docs/research/basalt-nvcc/bf16.md): FFMA.FTZ and FADD.FTZ, written as fma_ftz and add_ftz.
//
// Macros: BLOCK threads per threadgroup (basalt's mmvf_block_size: 32, 64, ..., 256); NT accumulator rows (1, 4 or
// 8). monolith/kernels.py's basalt_mmvf_macros picks both as basalt's launchers do. Grid: n_out threadgroups of BLOCK
// threads.
#pragma STDC FP_CONTRACT OFF

#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef NT
#define NT 1
#endif

struct MmvfParams { uint n_in; uint n_out; uint n_tok; uint ldx; uint ldy; uint pad0; uint pad1; uint pad2; };

kernel void basalt_mmvf(device const float2* x [[buffer(0)]], device const uint* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant MmvfParams& p [[buffer(3)]],
                        uint o [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint t = 32u * uint(sg) + uint(lane);                   // basalt's threadIdx.x: warp t / 32, lane t % 32
    const uint pairs = p.n_in / 2u;
    device const uint* row = w + ulong(o) * pairs;                // row o as words of two bf16 values, low half first
#if BLOCK > 32
    threadgroup float partials[NT][32];
    if (sg == 0)
        for (int k = 0; k < NT; ++k) partials[k][lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    float acc[NT];
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (uint pair = t; pair < pairs; pair += BLOCK) {
        const uint word = row[pair];
        const float w0 = as_type<float>(word << 16), w1 = as_type<float>(word & 0xffff0000u);
        for (uint k = 0; k < NT; ++k) {
            if (k < p.n_tok) {
                const float2 in = x[ulong(k) * (p.ldx / 2u) + pair];
                acc[k] = fma_ftz(w0, in.x, acc[k]);
                acc[k] = fma_ftz(w1, in.y, acc[k]);
            }
        }
    }
    for (int k = 0; k < NT; ++k) acc[k] = warp_sum_ftz(acc[k]);
#if BLOCK > 32
    if (lane == 0)
        for (int k = 0; k < NT; ++k) partials[k][sg] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (int k = 0; k < NT; ++k) acc[k] = warp_sum_ftz(partials[k][lane]);
#endif
    if (t == 0)
        for (uint k = 0; k < NT; ++k)
            if (k < p.n_tok) y[ulong(k) * p.ldy + o] = acc[k];
}
