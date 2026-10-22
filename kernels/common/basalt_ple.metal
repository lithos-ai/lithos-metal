// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_ple_postops.cu:37-89, src/kernels/cuda/ple.cu:46-52,
// src/kernels/cuda/elementwise.cu:222-227 and src/kernels/cuda/verify_kernels.cu:506-512
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE); native_ple_postops.cu's arithmetic adapts
// ggml-org/llama.cpp ggml/src/ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu} @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's n-gram (PLE) block after its projections, one token (native_ple_postops), between its grouped norms
// (basalt_gr_norm.metal): the gate, the broadcast and the causal convolution with its SiLU and residual; then the
// history's three moves in a verify window: the advance after each row, the snapshot after it and the commit's
// restore. The constants are basalt's (native_ple_postops.cu:36; include/basalt/kernels/ngram.hpp:30-41): N the stream
// width, H the hyper-connection streams, D = H * N their channels, HIST the history's rows, HS = HIST * D its words
// (src/core/verify.cpp:939), SNAP_SLOTS the snapshots, one per row of the largest window (kVerifyMaxT,
// include/basalt/kernels/verify_kernels.hpp:21). Needs basalt_fp.metal ahead of it.
//
// - basalt_ple_gate (gate_kernel): threadgroup b of 512 threads reduces stream b's N products of the normalized key
//   and query. Thread t = 32 * sg + lane (basalt's threadIdx.x) adds product t + 512 j into partial j from +0, then the
//   eight partials in order into a sum from +0 (the zero adds kept: +0 + -0 is +0); basalt's XOR butterfly per
//   SIMD-group; lane 0 of each stores its sum into slot sg; after the barrier every SIMD-group runs the butterfly over
//   the 16 slots, lanes 16 to 31 adding +0; thread 0 scales the sum (an fma with a +0 addend), clamps its magnitude at
//   1e-6f, takes its square root, signs it ((s > 0) - (s < 0): 0 for +-0 and a NaN) and stores the sigmoid
//   1 / (1 + expf(-y)), which is never NaN. The scale is the host's 1.0f / std::sqrt(float(N)) (params).
// - basalt_ple_broadcast (broadcast_kernel): channel i of stream i / N is value[i % N] times gate[i / N].
// - basalt_ple_conv (conv_residual_kernel): channel c's taps are history rows 0, 3 and 6 (the history is
//   channel-major, row r of channel c at c * HIST + r, row 8 the newest) and this token's normalized[c], each times its
//   fp16 weight taps[c * 4 + k], summed in k order from the first term; then the SiLU sum / (1 + expf(-sum)), stored
//   as conv, and result = hidden + (gated + activation). The history is only read; result may be bound onto hidden
//   (each thread reads its hidden[c] before it stores result[c]).
// - basalt_ple_advance (history_advance_kernel; verify.cpp:1118): channel c's HIST words drop row 0 and take
//   normalized[c] as row 8, in place, each row read before it is overwritten (no other thread touches the column, and
//   normalized lies outside the history: the emitter refuses an alias, as ple_history_advance does, ple.cu:207-209).
// - basalt_ple_snapshot (copy_from_mapped_kernel; verify.cpp:1123): the history's HS words into one snapshot slot,
//   bound at the slot (slot r after row r), as 16-byte words in basalt's grid-stride order.
// - basalt_ple_restore (copy_indexed_kernel; verify.cpp:2137): the history from snapshot slot index[1], the commit
//   record's word 1 (n_keep - 1, basalt's commit_ + 1), which every thread reads at run time. A negative index copies
//   nothing (basalt's); an index past the last slot restores the last one (the port's bound: basalt has none on the
//   device, and its host refuses an n_keep outside 1 to the window's rows, verify.cpp:2732).
// The moves do no arithmetic (their files are IEEE-built, and no rounding rule applies to a move): every binding is
// words (uint, uint4), so every bit pattern, subnormals and NaN payloads included, moves unchanged.
//
// Every float step is .FTZ: mul_ftz and add_ftz for basalt's __fmul_rn and __fadd_rn (never fused), fma_ftz for the
// explicit __fmaf_rn, max_ftz and abs_ftz for fmaxf and fabsf, and for the fast-math library calls sqrtf_ftz
// (MUFU.SQRT), expf_ftz (MUFU.EX2 of the product with log2 e; expf_ftz(-x) has the bits of the SASS product with
// -log2 e) and rcp_ftz (MUFU.RCP; a / b is mul_ftz(a, rcp_ftz(b))). A NaN is stored as sm_120's (cuda_nan;
// docs/research/basalt-nvcc/nan.md) at the gated, conv and result stores.
//
// Grids: the gate H threadgroups of 512 threads (one per stream), the broadcast, the convolution and the advance
// D / 256 threadgroups of 256 threads, the snapshot and the restore 64 threadgroups of 256 threads in grid-stride
// loops of two and six passes (monolith/kernels.py's PLE_GRIDS). The gate's block size fixes its reduction order, so
// its pipeline must accept 512 threads.
//
// The rounding sequence nvcc compiles for these kernels: docs/research/basalt-nvcc/ple.md
#pragma STDC FP_CONTRACT OFF

constant constexpr uint N = 2560u;
constant constexpr uint H = 4u;
constant constexpr uint D = N * H;
constant constexpr uint HIST = 9u;
constant constexpr uint HS = HIST * D;
constant constexpr uint SNAP_SLOTS = 8u;

struct PleGateParams { uint scale_bits; uint pad[3]; };

kernel void basalt_ple_gate(device const float* key [[buffer(0)]], device const float* query [[buffer(1)]],
                            device float* gate [[buffer(2)]], constant PleGateParams& gp [[buffer(3)]],
                            uint b [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint t = 32u * uint(sg) + uint(lane);                   // basalt's threadIdx.x: warp t / 32, lane t % 32
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint j = 0; j < 8u; ++j) {
        const uint d = t + j * 512u;
        const float p = d < N ? mul_ftz(key[b * N + d], query[b * N + d]) : 0.0f;
        sums[j] = add_ftz(sums[j], p);
    }
    float sum = 0.0f;
    for (uint j = 0; j < 8u; ++j) sum = add_ftz(sum, sums[j]);
    threadgroup float partials[16];
    sum = warp_sum_ftz(sum);
    if (lane == 0) partials[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum_ftz(sum);
    if (t == 0) {
        const float s = fma_ftz(as_type<float>(gp.scale_bits), sum, 0.0f);   // ggml SCALE's zero bias
        const float mag = sqrtf_ftz(max_ftz(abs_ftz(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[b] = rcp_ftz(add_ftz(1.0f, expf_ftz(-mul_ftz(sign, mag))));
    }
}

kernel void basalt_ple_broadcast(device const float* value [[buffer(0)]], device const float* gate [[buffer(1)]],
                                 device float* gated [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < D) gated[i] = cuda_nan(mul_ftz(value[i % N], gate[i / N]));
}

kernel void basalt_ple_conv(device const float* history [[buffer(0)]], device const float* normalized [[buffer(1)]],
                            device const ushort* taps [[buffer(2)]], device const float* hidden [[buffer(3)]],
                            device const float* gated [[buffer(4)]], device float* conv [[buffer(5)]],
                            device float* result [[buffer(6)]], uint c [[thread_position_in_grid]]) {
    if (c >= D) return;
    float sum = 0.0f;
    for (uint k = 0; k < 4u; ++k) {
        const float x = k == 3u ? normalized[c] : history[c * HIST + 3u * k];
        const float term = mul_ftz(x, h2f(taps[c * 4u + k]));
        sum = k == 0u ? term : add_ftz(sum, term);
    }
    const float act = mul_ftz(sum, rcp_ftz(add_ftz(1.0f, expf_ftz(-sum))));
    conv[c] = cuda_nan(act);
    result[c] = cuda_nan(add_ftz(hidden[c], add_ftz(gated[c], act)));
}

kernel void basalt_ple_advance(device uint* history [[buffer(0)]], device const uint* normalized [[buffer(1)]],
                               uint c [[thread_position_in_grid]]) {
    if (c >= D) return;
    device uint* column = history + c * HIST;
    for (uint row = 0; row + 1u < HIST; ++row) column[row] = column[row + 1u];
    column[HIST - 1u] = normalized[c];
}

kernel void basalt_ple_snapshot(device uint4* dst [[buffer(0)]], device const uint4* src [[buffer(1)]],
                                uint i0 [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    for (uint i = i0; i < HS / 4u; i += n) dst[i] = src[i];
}

kernel void basalt_ple_restore(device uint* dst [[buffer(0)]], device const uint* src [[buffer(1)]],
                               device const int* index [[buffer(2)]], uint i0 [[thread_position_in_grid]],
                               uint n [[threads_per_grid]]) {
    int idx = index[1];
    if (idx < 0) return;
    idx = min(idx, int(SNAP_SLOTS) - 1);
    for (uint i = i0; i < HS; i += n) dst[i] = src[uint(idx) * HS + i];
}
