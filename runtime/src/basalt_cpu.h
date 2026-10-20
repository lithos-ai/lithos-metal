// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cpu/expert.cpp:143-148, :336-370, :503-556 and
// include/basalt/kernels/cpu/expert.hpp:25-61 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// which adapt ggml-org/llama.cpp ggml/src/ggml-common.h (block_q2_0) and ggml/src/ggml-quants.c
// (dequantize_row_q2_0's code rule) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's two CPU Q2_0 kernels, transcribed to ARM NEON operation for operation: the activation quantizer the CPU
// expert tier feeds its dot products, and the Q2_0 GGUF row dot. basalt's signatures, unchanged, so its call sites
// (src/kernels/cpu/pool.cpp:423-444) transcribe one for one when the pool follows.
#pragma once

#include <cstddef>
#include <cstdint>

namespace monolith::basalt_cpu {

// basalt's geometry (expert.hpp:25-36). H and MAXC bound the ActQ buffers; both kernels take their widths at run
// time (n, nblocks, row_bytes), so nothing here pins a model.
inline constexpr int H = 2560;          // n_embd: the widest reduction an ActQ holds
inline constexpr int FF = 640;          // expert intermediate width: the second width the pool's callers quantize
inline constexpr int QK = 64;           // QK2_0: weights per fp16 block scale
inline constexpr int QKA = 32;          // activation elements per q8_1 scale
inline constexpr int BB = 18;           // bytes per Q2_0 block: fp16 scale, then 16 code bytes
inline constexpr int MAXC = H / QKA;    // 80 activation chunks
inline constexpr int MAXT = 8;          // tokens per job (expert.hpp:122)

/// One quantized activation, field for field as expert.hpp:55-61 declares it.
/// hx[k] = scale[k] * (float) sum[k] is the weight-independent correction the Q2_0 identity needs.
struct ActQ {
    alignas(64) int8_t q[H];
    float scale[MAXC];
    int32_t sum[MAXC];
    float hx[MAXC];
    int nchunks;
};

/// expert.cpp:336-370 (the AVX-512 block; the scalar loop after it is unreachable, and the --cpu-oracle-q8-0
/// branch at :337-340 is not ported). Writes q[0, n), and scale / sum / hx over [0, n / QKA), and sets nchunks.
/// n must be a positive multiple of QKA, at most H; the kernel throws std::invalid_argument otherwise.
void act_quant_q8_1(const float* x, int n, ActQ& a);

/// expert.cpp:544-556 through q2g_rows / q2g_row_multi. Row r is w + r * row_bytes; out[t] is indexed by the
/// ABSOLUTE row r, and rows outside [r0, r1) are left alone. Reads a[t]->q[0, 64 * nblocks),
/// a[t]->scale[0, 2 * nblocks) and a[t]->hx[0, 2 * nblocks), and nothing else - never nchunks, never sum.
/// nt must be 1..MAXT and nblocks 1..H / QK: the per-token accumulators are MAXT wide and the reads above run to
/// 64 * nblocks of an H-element array. basalt bounds nt by instantiating q2g_rows<8> for every nt above 8
/// (expert.cpp:554); the caller here bounds it, and the kernel throws std::invalid_argument on either, in a form
/// that survives the Release build's NDEBUG.
void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1);

}  // namespace monolith::basalt_cpu
