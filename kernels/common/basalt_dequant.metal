// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/iq_kernels.cu:2012-2069 and :2216-2220
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which transcribes ggml-org/llama.cpp
// ggml/src/ggml-cuda/dequantize.cuh and convert.cu (and ggml/src/ggml-common.h's kvalues_iq4nl) @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's dequantizers for the in-pack types iq_parity checks: dq_iq4_nl, dq_q3_k, dq_iq4_xs and dq_q2_0 under
// dequant_flat_kernel<float> (threadgroup i = the 256-value superblock i, 32 threads, each thread writing basalt's
// values). GGUF blocks as the file stores them, in; fp32 values, out. Macro DQ_TYPE: the ggml type id (11 Q3_K,
// 20 IQ4_NL, 23 IQ4_XS, 42 Q2_0). Grid: n / 256 threadgroups of 32 threads. Needs basalt_fp.metal ahead of it.
//
// Arithmetic (docs/research/basalt-nvcc/dequant.md): iq_kernels.cu builds with nvcc's IEEE defaults, and each value
// is one or two correctly rounded multiplies, FMUL in SASS, no .FTZ: fl(h(d) * f(k)) (IQ4_NL, Q2_0) and
// fl(fl(h(d) * f(s - 32)) * f(k)) (Q3_K, IQ4_XS). Every factor is an fp16 value or a small integer, so no product is
// subnormal (the smallest nonzero one is 2^-24) and this GPU's flushing changes nothing; the products are not exact
// in general, so the rounding is the multiply's own, as CUDA's. A NaN (a NaN scale, or an infinite one times a zero
// factor) is stored as sm_120's (cuda_nan; docs/research/basalt-nvcc/nan.md).
#pragma STDC FP_CONTRACT OFF

#ifndef DQ_TYPE
#define DQ_TYPE 11
#endif

constant char KVALUES_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

kernel void basalt_dequant_f32(device const ushort* vx [[buffer(0)]], device float* y [[buffer(1)]],
                               uint ibs [[threadgroup_position_in_grid]], ushort tid_ [[thread_index_in_threadgroup]]) {
    const int tid = int(tid_);
    device float* yy = y + 256u * ibs;
#if DQ_TYPE == 20
    // dq_iq4_nl (:2012-2023): 8 blocks of 18 bytes (d, qs[16]) per superblock
    const int il = tid / 8, ib = tid % 8;
    device const ushort* x = vx + 9u * (8u * ibs + uint(ib));
    device float* yo = yy + 32 * ib + 4 * il;
    const float d = h2f(x[0]);
    for (int j = 0; j < 4; ++j) {
        const uint q = uint(w_u8(x, 2 + 4 * il + j));
        yo[j + 0] = cuda_nan(d * float(KVALUES_IQ4NL[q & 0xfu]));
        yo[j + 16] = cuda_nan(d * float(KVALUES_IQ4NL[q >> 4]));
    }
#elif DQ_TYPE == 11
    // dq_q3_k (:2025-2045): one 110-byte block (hmask[32], qs[64], scales[12], d) per superblock
    device const ushort* x = vx + 55u * ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint m = 1u << (4 * n + j);
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const uint sc = 96u;                                            // scales[] at byte 96
        const int us = is < 4  ? int((w_u8(x, sc + is) & 0xFu) | (((w_u8(x, sc + is + 8) >> 0) & 3u) << 4)) :
                       is < 8  ? int((w_u8(x, sc + is) & 0xFu) | (((w_u8(x, sc + is + 4) >> 2) & 3u) << 4)) :
                       is < 12 ? int((w_u8(x, sc + is - 8) >> 4) | (((w_u8(x, sc + is) >> 4) & 3u) << 4)) :
                                 int((w_u8(x, sc + is - 8) >> 4) | (((w_u8(x, sc + is - 4) >> 6) & 3u) << 4));
        const float dl = h2f(x[54]) * float(us - 32);                  // (float) x->d * (us - 32)
        device float* yo = yy + 128 * n + 32 * j;
        for (int l = l0; l < l0 + 4; ++l) {
            const int q = int((w_u8(x, 32u + uint(32 * n + l)) >> shift) & 3u) - ((w_u8(x, uint(l)) & m) ? 0 : 4);
            yo[l] = cuda_nan(dl * float(q));
        }
    }
#elif DQ_TYPE == 23
    // dq_iq4_xs (:2046-2057): one 136-byte block (d, scales_h, scales_l[4], qs[128]) per superblock
    device const ushort* x = vx + 68u * ibs;
    const int il = tid / 8, ib = tid % 8;
    device float* yo = yy + 32 * ib + 4 * il;
    const int ls = int(((w_u8(x, 4u + uint(ib / 2)) >> (4 * (ib % 2))) & 0xfu) | (((uint(x[1]) >> (2 * ib)) & 3u) << 4));
    const float d = h2f(x[0]) * float(ls - 32);
    for (int j = 0; j < 4; ++j) {
        const uint q = uint(w_u8(x, 8u + uint(16 * ib + 4 * il + j)));
        yo[j + 0] = cuda_nan(d * float(KVALUES_IQ4NL[q & 0xfu]));
        yo[j + 16] = cuda_nan(d * float(KVALUES_IQ4NL[q >> 4]));
    }
#elif DQ_TYPE == 42
    // dq_q2_0 (:2058-2069): 4 blocks of 18 bytes (d, qs[16] of 2-bit codes) per superblock, value d * (code - 1)
    const int b = tid / 8, part = tid % 8;
    device const ushort* x = vx + 9u * (4u * ibs + uint(b));
    const float d = h2f(x[0]);
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = int((w_u8(x, 2u + uint(i / 4)) >> ((i % 4) * 2)) & 3u);
        yy[b * 64 + i] = cuda_nan(d * float(code - 1));
    }
#else
#error "basalt_dequant: unsupported DQ_TYPE"
#endif
}
