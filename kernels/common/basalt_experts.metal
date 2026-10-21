// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/iq_kernels.cu:864-1014, :1087-1171, :1648-1921 and
// src/kernels/cuda/gu2.cuh:223-390 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which adapt
// ggml-org/llama.cpp ggml/src/ggml-cuda/{vecdotq.cuh,quantize.cu} @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT;
// third_party/NOTICE).
//
// basalt's native_expert_grouped for Q2_0 gate/up and Q2_0 down (iq_kernels.cu:2553-2603): three dispatches in order,
// gate/up -> SwiGLU + q8_1 -> down. Needs basalt_fp.metal and basalt_q8_1.metal ahead of it.
//
// The plan (basalt's, read on the device): n_groups[0] groups; group g's expert blob starts at grp_ptr[g] (basalt's
// grp_ptr: a table of 64-bit GPU addresses, one a group, each 2-byte aligned and inside a resident buffer, so the
// blobs may sit in several buffers) and owns the entries [grp_start[g], grp_start[g + 1]); entry e reads token
// ent_tok[e]'s q8_1 row (n_embd / 32 blocks) and writes row ent_dst[e] of `out` (n_embd floats). A blob is basalt's
// native_expert_layout(42, 42, n_embd, n_ff): n_ff gate rows, n_ff up rows (n_embd / 64 Q2_0 blocks each, up at
// up_off) and n_embd down rows (n_ff / 64 blocks each, at down_off).
// Scratch, basalt's: gate and up (entry e's n_ff floats at e * n_ff) and hq (entry e's n_ff / 32 q8_1 blocks).
//
// Threadgroup row y takes groups y, y + gy, ... (gy = threadgroups in y; basalt's group stride: the result does not
// depend on gy). Every float step is the one nvcc compiles for iq_kernels.cu (nvcc's IEEE defaults;
// docs/research/basalt-nvcc/experts.md): fma() exactly at its FFMA sites, plain multiplies and adds elsewhere. No
// subnormal can reach the matrix-vector parts (every value is a multiple of 2^-48), so the GPU's operations are
// CUDA's there; the SwiGLU product can be subnormal and goes through ieee_mul, then the IEEE q8_1 store. A NaN is
// stored as sm_120's (cuda_nan, and the q8_1 store's cuda_nan_f16; docs/research/basalt-nvcc/nan.md).
//
// - basalt_expert_gu: gate/up. GU2 1: native_gu2_kernel<42> (gu2.cuh:338-357, n_embd 2560 only): 16 rows a
//   threadgroup, rows 16b + 2w and + 1 on SIMD-group w. GU2 0: native_gu_multi_kernel<42> (iq_kernels.cu:1125-1171):
//   8 rows a threadgroup, one a SIMD-group. Grid (ceil(2 n_ff / rows), gy) threadgroups of 256.
// - basalt_expert_swiglu_q8_1: swiglu_q8_1_entries_pdl_kernel (gu2.cuh:375-390) over the call's own entries
//   [grp_start[0], grp_start[n_groups]); grid ceil(cap_entries * n_ff / 256) threadgroups of 256, grid-strided.
// - basalt_expert_down: native_down_multi_kernel<42> (iq_kernels.cu:1688-1878). SUB16 1 (n_ff 640): 16 rows of 16
//   lanes a threadgroup, row_dot_20_sub16 on the entries' q8_1 rows staged in threadgroup memory; SUB16 0: 8 rows a
//   threadgroup, one a SIMD-group, row_dot_multi from device memory. Grid (ceil(n_embd / rows), gy) of 256.
#pragma STDC FP_CONTRACT OFF

#ifndef GU2
#define GU2 0
#endif
#ifndef SUB16
#define SUB16 0
#endif
#define GRP_NC 6   // entries a pass (iq_kernels.cu:1123)

struct ExpertParams { uint n_embd; uint n_ff; uint gu_row; uint d_row; uint up_off; uint down_off; uint pad0; uint pad1; };

// Split<42>::apply into s: s = fma(fl(d2 * d8), f(sumi), s), the contracted `s += r.d2 * d8 * sumi` (FFMA in SASS)
template <typename P> static inline float q2_0_acc(thread const Q20W& w, P x, int b, float s) {
    return fma(w.d2 * q8_d(x, b), float(q2_0_sumi(w, x, b)), s);
}

// row_dot_multi<42, NC> (iq_kernels.cu:898-919): one row against the n <= NC activations x + off[c] (offsets in q8_1
// blocks), the whole SIMD-group: lane L takes calls k = L, L + 32, ... (block k / 2, half k % 2, q8_1 block k), then
// the butterfly. EXACT_N: every c < NC is an entry.
template <int NC, bool EXACT_N, typename P>
static inline void row_dot_multi(device const ushort* row, P x, thread const int* off, int n, int nb, int lane,
                                 thread float* s) {
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * 2; k += 32) {
        const Q20W w = q2_0_load(row + 9 * (k / 2), k % 2);
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] = q2_0_acc(w, x + 9 * off[c], k, s[c]);
    }
    for (int c = 0; c < NC; ++c)
        if (EXACT_N || c < n) s[c] = warp_sum(s[c]);
}

// ---- gate / up ---------------------------------------------------------------------------------------------------

#if GU2
// gu2::rows2<42, NC> (gu2.cuh:254-300): rows r, r + 1 (row A at rowA, row B gu_row bytes after it) for the n <= NC
// entries from e. Lane L sums calls L and L + 32 of both rows, then call 64 + (L & 15) of row A (lanes 0..15) or of
// row B (lanes 16..31), whose scale product and integer sum move down 16 lanes before lanes 0..15 add them.
template <int NC>
static inline void rows2(device const ushort* rowA, uint row_halves, int r, int e, int n, device const int* ent_tok,
                         device const uint* xq, device float* dst, int n_ff, int lane) {
    device const ushort* rowB = rowA + row_halves;
    device const uint* xc[NC];
    for (int c = 0; c < NC; ++c) xc[c] = xq + 9 * 80 * ent_tok[e + min(c, n - 1)];   // 2560 values: 80 blocks
    float sA[NC], sB[NC];
    for (int c = 0; c < NC; ++c) sA[c] = sB[c] = 0.0f;
    for (int it = 0; it < 2; ++it) {
        const int k = lane + 32 * it;
        const Q20W wa = q2_0_load(rowA + 9 * (k >> 1), k & 1);
        const Q20W wb = q2_0_load(rowB + 9 * (k >> 1), k & 1);
        for (int c = 0; c < NC; ++c) {
            if (c < n) {
                sA[c] = fma(wa.d2 * q8_d(xc[c], k), float(q2_0_sumi(wa, xc[c], k)), sA[c]);
                sB[c] = fma(wb.d2 * q8_d(xc[c], k), float(q2_0_sumi(wb, xc[c], k)), sB[c]);
            }
        }
    }
    const int k2 = 64 + (lane & 15);
    const Q20W w2 = q2_0_load((lane < 16 ? rowA : rowB) + 9 * (k2 >> 1), k2 & 1);
    for (int c = 0; c < NC; ++c) {
        if (c < n) {
            const float d = w2.d2 * q8_d(xc[c], k2), f = float(q2_0_sumi(w2, xc[c], k2));
            const float dB = as_type<float>(simd_shuffle_down(as_type<uint>(d), ushort(16)));
            const float fB = as_type<float>(simd_shuffle_down(as_type<uint>(f), ushort(16)));
            if (lane < 16) {
                sA[c] = fma(d, f, sA[c]);
                sB[c] = fma(dB, fB, sB[c]);
            }
            sA[c] = warp_sum(sA[c]);
            sB[c] = warp_sum(sB[c]);
            if (lane == 2 * c) dst[(e + c) * n_ff + r] = cuda_nan(sA[c]);
            if (lane == 2 * c + 1) dst[(e + c) * n_ff + r + 1] = cuda_nan(sB[c]);
        }
    }
}
#endif

kernel void basalt_expert_gu(device const GrpPtr* grp_ptr [[buffer(1)]],
                             device const int* grp_start [[buffer(2)]], device const int* n_groups [[buffer(3)]],
                             device const int* ent_tok [[buffer(4)]], device const uint* xq [[buffer(5)]],
                             device float* gate [[buffer(6)]], device float* up [[buffer(7)]],
                             constant ExpertParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint2 ntg [[threadgroups_per_grid]], ushort lane [[thread_index_in_simdgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int ng = n_groups[0];
    if (int(tg.y) >= ng) return;
    const int n_ff = int(p.n_ff);
#if GU2
    const int row = int(tg.x) * 16 + 2 * int(sg);                  // rows row, row + 1: both gate or both up
    if (row >= 2 * n_ff) return;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - n_ff : row;
    const ulong w_off = ulong(is_up ? p.up_off : 0u) + ulong(r) * p.gu_row;
    device float* dst = is_up ? up : gate;
    for (int g = int(tg.y); g < ng; g += int(ntg.y)) {
        device const ushort* wr = grp_ptr[g].p + w_off / 2;
        const int e1 = grp_start[g + 1];
        for (int e = grp_start[g]; e < e1;) {                     // gu2::pair (gu2.cuh:303-330)
            const int n = e1 - e;
            if (n == 1) {
                rows2<1>(wr, p.gu_row / 2, r, e, 1, ent_tok, xq, dst, n_ff, lane);
                e += 1;
            } else if (n == 2) {
                rows2<2>(wr, p.gu_row / 2, r, e, 2, ent_tok, xq, dst, n_ff, lane);
                e += 2;
            } else if (n == 3) {
                rows2<3>(wr, p.gu_row / 2, r, e, 3, ent_tok, xq, dst, n_ff, lane);
                e += 3;
            } else {
                const int m = min(n, GRP_NC);
                rows2<GRP_NC>(wr, p.gu_row / 2, r, e, m, ent_tok, xq, dst, n_ff, lane);
                e += m;
            }
        }
    }
#else
    const int row = int(tg.x) * 8 + int(sg);
    if (row >= 2 * n_ff) return;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - n_ff : row;
    const ulong w_off = ulong(is_up ? p.up_off : 0u) + ulong(r) * p.gu_row;
    const int nb = int(p.n_embd) / 64, xb = int(p.n_embd) / 32;
    device float* dst = is_up ? up : gate;
    for (int g = int(tg.y); g < ng; g += int(ntg.y)) {
        device const ushort* wr = grp_ptr[g].p + w_off / 2;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (n == 1) {
                const int off1[1] = {ent_tok[e] * xb};
                float s1[1];
                row_dot_multi<1, true>(wr, xq, off1, 1, nb, lane, s1);
                if (lane == 0) dst[e * n_ff + r] = cuda_nan(s1[0]);
            } else if (n == 2) {
                const int off2[2] = {ent_tok[e] * xb, ent_tok[e + 1] * xb};
                float s2[2];
                row_dot_multi<2, true>(wr, xq, off2, 2, nb, lane, s2);
                if (lane < 2) dst[(e + int(lane)) * n_ff + r] = cuda_nan(s2[lane]);
            } else {
                int off[GRP_NC];
                for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
                float s[GRP_NC];
                row_dot_multi<GRP_NC, false>(wr, xq, off, n, nb, lane, s);
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && int(lane) == c) dst[(e + c) * n_ff + r] = cuda_nan(s[c]);
            }
        }
    }
#endif
}

// ---- SwiGLU + q8_1 -----------------------------------------------------------------------------------------------

// h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]), then q8_1_store (gu2.cuh:386-388); n_ff is a multiple of 32, so a
// SIMD-group is one q8_1 block and the bounds are SIMD-group uniform.
kernel void basalt_expert_swiglu_q8_1(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                                      device const int* grp_start [[buffer(2)]], device const int* n_groups [[buffer(3)]],
                                      device uchar* hq [[buffer(4)]], constant ExpertParams& p [[buffer(5)]],
                                      uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    const uint lo = uint(grp_start[0]) * p.n_ff, hi = uint(grp_start[n_groups[0]]) * p.n_ff;
    for (uint i = lo + tg * 256u + tid; i < hi; i += ntg * 256u) {
        const float g = gate[i];
        const float s = g / (1.0f + expf_neg(g));                 // div.rn.f32: correctly rounded
        q8_1_store_ieee(ieee_mul(s, up[i]), hq, i);
    }
}

// ---- down --------------------------------------------------------------------------------------------------------

#if SUB16
// row_dot_20_sub16<42, NC> (iq_kernels.cu:980-1014): 20 calls on 16 lanes, t = lane & 15: call t, then call t + 16
// for t < 4 into s16, s + s16, then the butterfly 8, 4, 2, 1 inside the 16 lanes (row_dot_multi's tree on 32 lanes
// with lanes 20..31 at zero).
template <int NC, bool EXACT_N>
static inline void row_dot_20_sub16(device const ushort* row, threadgroup const uint* x, thread const int* off, int n,
                                    int t, thread float* s) {
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const Q20W w = q2_0_load(row + 9 * (t / 2), t % 2);
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] = q2_0_acc(w, x + 9 * off[c], t, s[c]);
    }
    float s16[NC];
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16;
        const Q20W w = q2_0_load(row + 9 * (k / 2), k % 2);
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] = q2_0_acc(w, x + 9 * off[c], k, s16[c]);
    }
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = s[c] + s16[c];                                    // __fadd_rn
            for (ushort o = 8; o > 0; o >>= 1) s[c] = s[c] + shfl_xor(s[c], o);
        }
    }
}
#endif

kernel void basalt_expert_down(device const GrpPtr* grp_ptr [[buffer(1)]],
                               device const int* grp_start [[buffer(2)]], device const int* n_groups [[buffer(3)]],
                               device const int* ent_dst [[buffer(4)]], device const uint* hq [[buffer(5)]],
                               device float* out [[buffer(6)]], constant ExpertParams& p [[buffer(7)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],
                               ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int ng = n_groups[0];
    if (int(tg.y) >= ng) return;
    const int n_embd = int(p.n_embd), hb = int(p.n_ff) / 32;
#if SUB16
    threadgroup uint s_hq[GRP_NC * 20 * 9];                        // six entries' 20 q8_1 blocks (iq_kernels.cu:1726)
    const int r = int(tg.x) * 16 + int(tid >> 4), t = int(tid & 15);
    const bool valid_r = r < n_embd;
    const ulong w_off = ulong(p.down_off) + ulong(valid_r ? r : 0) * p.d_row;
    for (int g = int(tg.y); g < ng; g += int(ntg.y)) {
        device const ushort* wr = grp_ptr[g].p + w_off / 2;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            device const uint* src = hq + 9 * 20 * e;
            const int words = n * (20 * 9);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = int(tid); i < words; i += 256) s_hq[i] = src[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (valid_r) {
                if (n == 1) {
                    const int off1[1] = {0};
                    float s1[1];
                    row_dot_20_sub16<1, true>(wr, s_hq, off1, 1, t, s1);
                    if (t == 0) out[ent_dst[e] * n_embd + r] = cuda_nan(s1[0]);
                } else if (n == 2) {
                    const int off2[2] = {0, 20};
                    float s2[2];
                    row_dot_20_sub16<2, true>(wr, s_hq, off2, 2, t, s2);
                    if (t < 2) out[ent_dst[e + t] * n_embd + r] = cuda_nan(t == 0 ? s2[0] : s2[1]);
                } else {
                    int off[GRP_NC];
                    for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                    float s[GRP_NC];
                    row_dot_20_sub16<GRP_NC, false>(wr, s_hq, off, n, t, s);
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n && t == c) out[ent_dst[e + c] * n_embd + r] = cuda_nan(s[c]);
                }
            }
        }
    }
#else
    const int r = int(tg.x) * 8 + int(sg);
    const bool valid_r = r < n_embd;
    const ulong w_off = ulong(p.down_off) + ulong(valid_r ? r : 0) * p.d_row;
    const int nb = int(p.n_ff) / 64;
    for (int g = int(tg.y); g < ng; g += int(ntg.y)) {
        device const ushort* wr = grp_ptr[g].p + w_off / 2;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if (valid_r) {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                int off[GRP_NC];
                for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
                float s[GRP_NC];
                row_dot_multi<GRP_NC, false>(wr, hq, off, n, nb, lane, s);
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && int(lane) == c) out[ent_dst[e + c] * n_embd + r] = cuda_nan(s[c]);
            }
        }
    }
#endif
}
