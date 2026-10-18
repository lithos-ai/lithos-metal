// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:179-1172
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which adapts ggml-org/llama.cpp
// ggml/src/ggml-cuda/{vecdotq.cuh,mmvq.cu,common.cuh} and ggml/src/ggml-common.h @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's native_mmvq: y[j][r] = row r of a GGUF weight matrix (blocks as in the file, rows of n_in / DIV blocks)
// times column j of q8_1 activations (basalt_q8_1.metal's layout: column j's blocks start at block j * n_in / 32),
// for j < NCOLS; y[j] starts at j * n_out. Needs basalt_fp.metal ahead of it.
//
// One kernel for basalt's one-column kernels and its multi-column kernel (native_mmvq_multi_kernel), which give every
// (row, column) value the same operations in the same order (native_mmvq.cu:882-896): each thread takes the weight
// blocks kbx = tid / T, tid / T + BPI', ... of its rows in ascending order at sub-position kqs, `load` decodes the
// block once and `apply` runs the format's dot product per column; SIMD-groups 1..NW-1 store their partial sums, SIMD-
// group 0 adds them in SIMD-group order and reduces with basalt's XOR butterfly; lane i writes row row0 + i. The float
// steps are the ones nvcc compiles (docs/research/basalt-nvcc/matvec.md): fma() exactly where SASS has FFMA.FTZ, the
// rest .FTZ adds and multiplies. Integer steps are basalt's, as exact int32 arithmetic.
//
// Macros: MMVQ_TYPE the ggml type id (2 Q4_0, 6 Q5_0, 8 Q8_0, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 20 IQ4_NL,
// 23 IQ4_XS, 42 Q2_0); NCOLS 1..8; ROWS rows per threadgroup; NW SIMD-groups per threadgroup (BPI' = BPI * NW / 4).
// basalt's layouts (monolith/kernels.py basalt_mmvq_macros): one column NW 4, ROWS 1 (4 for small K); several
// columns, exact (the default) NW 4, ROWS 1 (2 for small K); the upstream layout (mmvq_multi_parity's negative
// control only) NW 4 for <= 4 columns else 2, ROWS 2. Grid: ceil(n_out / ROWS) threadgroups of 32 * NW threads.
#pragma STDC FP_CONTRACT OFF

#ifndef MMVQ_TYPE
#define MMVQ_TYPE 8
#endif
#ifndef NCOLS
#define NCOLS 1
#endif
#ifndef ROWS
#define ROWS 1
#endif
#ifndef NW
#define NW 4
#endif

struct MmvqParams { uint n_in; uint n_out; uint pad0; uint pad1; };

// q8_d, q8_s, q8_w (a q8_1 block's scale, sum and code words) and iq4_table_lookup are basalt_fp.metal's.

#if MMVQ_TYPE == 13
// Q5_K: basalt_fp.metal's q5k_load and q5k_dot (native_mmvq.cu:180-240, 897-937), shared with the Q5_K head
#define DIV 256
#define T_PER 16
#define KBY 8
#define BPI 8
#define BLOCK_BYTES 176
static inline int kqs_of(int tid) { return 2 * (tid % 16); }
typedef Q5KW Wt;
static inline Wt wload(device const ushort* blk, int iqs) { return q5k_load(blk, iqs); }
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    return add_ftz(tmp, q5k_dot(r, x, iqs));
}

#elif MMVQ_TYPE == 12
// Q4_K (144-byte blocks: dm, scales[12], qs[128]); native_mmvq.cu:938-975 (Q4KTraits), dot :522-574
#define DIV 256
#define T_PER 16
#define KBY 8
#define BPI 8
#define BLOCK_BYTES 144
static inline int kqs_of(int tid) { return 2 * (tid % 16); }
struct Wt { int vl[2]; uchar sc[2]; uchar m[2]; float dmx, dmy; int bq8_offset; };
static inline Wt wload(device const ushort* blk, int iqs) {
    Wt r;
    r.bq8_offset = 2 * ((iqs / 2) / 4);
    const int ql = (16 + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4)) / 4;
    r.vl[0] = load_int_b2(blk, ql);
    r.vl[1] = load_int_b2(blk, ql + 4);
    const int j = r.bq8_offset / 2;
    const int jm = j & 1;
    const uint s0 = blk[2 + jm], s2 = blk[2 + jm + 2], s4 = blk[2 + jm + 4];
    const uint hi = uint(-int(j >= 2));
    const uint aux0 = ((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi);
    const uint aux1 = ((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi);
    r.sc[0] = uchar(aux0); r.sc[1] = uchar(aux0 >> 8);
    r.m[0] = uchar(aux1); r.m[1] = uchar(aux1 >> 8);
    r.dmx = h2f(blk[0]);
    r.dmy = h2f(blk[1]);
    return r;
}
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    float sumf_d = 0.0f, sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int b = r.bq8_offset + i;
        const float d8 = q8_d(x, b);
        const int u0 = q8_w(x, b, (iqs / 2) % 4), u1 = q8_w(x, b, (iqs / 2) % 4 + 4);
        const int v0i = (r.vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (r.vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = dp4a(v1i, u1, dp4a(v0i, u0, 0));
        const int dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
        sumf_d = fma_ftz(d8, float(dot1 * int(r.sc[i])), sumf_d);
        sumf_m = fma_ftz(d8, float(dot2 * int(r.m[i])), sumf_m);
    }
    // dm.x * sumf_d - dm.y * sumf_m: ptxas rounds the dmin product and fuses the d product (matvec.md section 5)
    return add_ftz(tmp, fma_ftz(r.dmx, sumf_d, -mul_ftz(r.dmy, sumf_m)));
}

#elif MMVQ_TYPE == 42
// Q2_0 (18-byte blocks: d, qs[16] of 2-bit codes, value = d * (code - 1)); Q20Traits, native_mmvq.cu:976-1007
#define DIV 64
#define T_PER 2
#define KBY 2
#define BPI 64
#define BLOCK_BYTES 18
static inline int kqs_of(int tid) { return tid % 2; }
typedef Q20W Wt;                                                        // basalt_fp.metal's Q2_0 decode
static inline Wt wload(device const ushort* blk, int iqs) { return q2_0_load(blk, iqs); }
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    return fma_ftz(mul_ftz(r.d2, q8_d(x, iqs)), float(q2_0_sumi(r, x, iqs)), tmp);   // d2 * d8 * sumi into tmp +=
}

#elif MMVQ_TYPE == 11
// Q3_K (110-byte blocks: hmask[32], qs[64], scales[12], d); Q3KTraits, native_mmvq.cu:1008-1033, dot :360-398
#define DIV 256
#define T_PER 16
#define KBY 8
#define BPI 8
#define BLOCK_BYTES 110
static inline int kqs_of(int tid) { return tid % 16; }
struct Wt { int vl; int vh; float d; device const ushort* blk; int scale_offset; int bq8_offset; };
static inline Wt wload(device const ushort* blk, int iqs) {
    Wt r;
    r.bq8_offset = 4 * (iqs / 8);
    r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    r.d = h2f(blk[54]);
    r.vl = load_int_b2(blk + 16, iqs);
    r.vh = ~load_int_b2(blk, iqs % 8) >> r.bq8_offset;
    r.blk = blk;
    return r;
}
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    float sumf = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int isc = r.scale_offset + 2 * i;
        const int sc_low = (int(w_u8(r.blk, 96 + isc % 8)) >> (4 * (isc / 8))) & 0xf;
        const int sc_high = ((int(w_u8(r.blk, 96 + 8 + isc % 4)) >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (r.vl >> (2 * i)) & 0x03030303;
        const int vih = ((r.vh >> i) << 2) & 0x04040404;
        const int vi = vsubss4(vil, vih);
        const int b = r.bq8_offset + i;
        sumf = fma_ftz(q8_d(x, b), float(dp4a(vi, q8_w(x, b, iqs % 8), 0) * sc), sumf);
    }
    return fma_ftz(r.d, sumf, tmp);                                     // d3 * sumf into tmp +=
}

#elif MMVQ_TYPE == 14
// Q6_K (210-byte blocks: ql[128], qh[64], scales[16] int8, d); Q6KTraits, native_mmvq.cu:1034-1060, dot :621-651
#define DIV 256
#define T_PER 32
#define KBY 8
#define BPI 4
#define BLOCK_BYTES 210
static inline int kqs_of(int tid) { return tid % 32; }
struct Wt { int vl; int vh; float d; int sc[2]; int bq8_offset; };
static inline Wt wload(device const ushort* blk, int iqs) {
    Wt r;
    r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    r.vl = load_int_b2(blk, iqs);
    r.vh = load_int_b2(blk + 64, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    r.sc[0] = int(as_type<char>(w_u8(blk, 192 + scale_offset)));
    r.sc[1] = int(as_type<char>(w_u8(blk, 192 + scale_offset + 4)));
    r.d = h2f(blk[104]);
    return r;
}
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    float sumf = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int vil = (r.vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((r.vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = vsubss4(vil | vih, 0x20202020);
        const int b = r.bq8_offset + 2 * i;
        sumf = fma_ftz(q8_d(x, b), float(dp4a(vi, q8_w(x, b, iqs % 8), 0) * r.sc[i]), sumf);
    }
    return fma_ftz(r.d, sumf, tmp);                                     // d * sumf into tmp +=
}

#elif MMVQ_TYPE == 23
// IQ4_XS (136-byte blocks: d, scales_h, scales_l[4], qs[128]); IQ4XSTraits, native_mmvq.cu:1061-1087, dot :462-479
#define DIV 256
#define T_PER 8
#define KBY 8
#define BPI 16
#define BLOCK_BYTES 136
static inline int kqs_of(int tid) { return 4 * (tid % 8); }
struct Wt { int2 v[4]; int ls; float dw; };
static inline Wt wload(device const ushort* blk, int iqs) {
    Wt r;
    for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(load_int_b2(blk + 4, iqs + j));
    r.ls = ((int(w_u8(blk, 4 + iqs / 8)) >> (iqs & 0x04)) & 0x0f) | (((int(blk[1]) >> (iqs / 2)) & 0x03) << 4);
    r.dw = h2f(blk[0]);
    return r;
}
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        sumi = dp4a(r.v[j].x, q8_w(x, iqs / 4, j), sumi);
        sumi = dp4a(r.v[j].y, q8_w(x, iqs / 4, j + 4), sumi);
    }
    sumi *= r.ls - 32;
    return fma_ftz(mul_ftz(r.dw, q8_d(x, iqs / 4)), float(sumi), tmp);   // d * sumi into tmp +=
}

#elif MMVQ_TYPE == 2 || MMVQ_TYPE == 6 || MMVQ_TYPE == 20 || MMVQ_TYPE == 8
// The 32-value formats (SmallTraits, native_mmvq.cu:1088-1098; small_q8_dot :696-763): Q4_0 and IQ4_NL (18 bytes:
// d, qs[16]), Q5_0 (22: d, qh[4], qs[16]), Q8_0 (34: d, qs[32]); Qi 4, or 8 for Q8_0
#define DIV 32
#define KBY 1
#define QI_S (MMVQ_TYPE == 8 ? 8 : 4)
#define T_PER (QI_S / 2)
#define BPI (2 * 4 * 32 / QI_S)
#define BLOCK_BYTES (MMVQ_TYPE == 8 ? 34 : MMVQ_TYPE == 6 ? 22 : 18)
static inline int kqs_of(int tid) { return 2 * (tid % (QI_S / 2)); }
struct Wt { device const ushort* blk; };
static inline Wt wload(device const ushort* blk, int) { return Wt{blk}; }
static inline float apply_acc(float tmp, thread const Wt& r, device const uint* x, int iqs) {
    device const ushort* w = r.blk;
    int sumi = 0;
#if MMVQ_TYPE == 8
    for (int i = 0; i < 2; ++i) sumi = dp4a(load_int_b2(w + 1, iqs + i), q8_w(x, 0, iqs + i), sumi);
    return fma_ftz(mul_ftz(h2f(w[0]), q8_d(x, 0)), float(sumi), tmp);   // d0 * d1 * float(sumi) into tmp +=
#elif MMVQ_TYPE == 20
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w + 1, iqs + i));
        sumi = dp4a(v.x, q8_w(x, 0, iqs + i), sumi);
        sumi = dp4a(v.y, q8_w(x, 0, iqs + i + 4), sumi);
    }
    return fma_ftz(mul_ftz(h2f(w[0]), q8_d(x, 0)), float(sumi), tmp);   // d * sumi into tmp +=
#else
    for (int i = 0; i < 2; ++i) {
#if MMVQ_TYPE == 2
        const int v = load_int_b2(w + 1, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
#else
        const int vl = load_int_b2(w + 3, iqs + i);
        const int vh = load_int_b2(w + 1, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
#endif
        sumi = dp4a(vi0, q8_w(x, 0, iqs + i), sumi);
        sumi = dp4a(vi1, q8_w(x, 0, iqs + i + 4), sumi);
    }
    // d * (sumi * ds.x - k * ds.y), k = 4 (Q4_0) or 8 (Q5_0): fma(sumi, d8, fl(-k * s8)), then fma into tmp
    const float k = MMVQ_TYPE == 2 ? -4.0f : -8.0f;
    return fma_ftz(h2f(w[0]), fma_ftz(float(sumi), q8_d(x, 0), mul_ftz(q8_s(x, 0), k)), tmp);
#endif
}
#else
#error "basalt_mmvq: unsupported MMVQ_TYPE"
#endif

kernel void basalt_mmvq(device const ushort* w [[buffer(0)]], device const uint* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant MmvqParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int tid = 32 * int(sg) + int(lane);
    const int row0 = ROWS * int(tg);
    const int n_in = int(p.n_in), n_out = int(p.n_out);
    const int blocks_per_row = n_in / DIV;
    const int x_stride = n_in / 32;                                     // q8_1 blocks per activation column
    float tmp[NCOLS][ROWS];
    for (int j = 0; j < NCOLS; ++j)
        for (int i = 0; i < ROWS; ++i) tmp[j][i] = 0.0f;
    for (int kbx = tid / T_PER; kbx < blocks_per_row; kbx += BPI * NW / 4) {
        const int kby = kbx * KBY;
        const int kqs = kqs_of(tid);
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                device const ushort* blk = w + (uint(row0 + i) * uint(blocks_per_row) + uint(kbx)) * (BLOCK_BYTES / 2);
                const Wt wv = wload(blk, kqs);                         // once per (row, block)
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] = apply_acc(tmp[j][i], wv, x + 9 * (j * x_stride + kby), kqs);
            }
        }
    }
    threadgroup float partial[NW > 1 ? NW - 1 : 1][NCOLS][ROWS][32];
    if (sg > 0) {
        for (int j = 0; j < NCOLS; ++j)
            for (int i = 0; i < ROWS; ++i) partial[sg - 1][j][i][lane] = tmp[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg > 0) return;
    for (int j = 0; j < NCOLS; ++j) {
        for (int i = 0; i < ROWS; ++i) {
            float t = tmp[j][i];
            for (int l = 0; l < NW - 1; ++l) t = add_ftz(t, partial[l][j][i][lane]);
            t = warp_sum_ftz(t);
            if (int(lane) == i && row0 + i < n_out) y[j * n_out + row0 + i] = t;
        }
    }
}
