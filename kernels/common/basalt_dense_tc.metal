// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/dense_tc.cu:32-931
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), whose block formats are ggml-org/llama.cpp
// ggml/src/ggml-common.h's @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's dense_tc and dense_tc_pair: y[j][r] = row r of a GGUF weight matrix (rows of n_in / 256 super-blocks of
// Fmt<TYPE>::SB bytes) times column j of q8_1 activations (basalt_q8_1.metal's layout: column j's blocks start at
// block j * n_in / 32), for j < ncols <= 8; y[j] starts at j * n_out. Needs basalt_fp.metal ahead of it.
//
// A threadgroup is one 16-row tile. basalt's warps of the tile are virtual SIMD-groups (24 for the one order, 8 for
// tile, pair and s with KS 2, 4 for s with KS 1), run by the threadgroup's SIMD-groups in turn: warp w takes half
// super-block w & 3 of the 512-value chunks w >> 2, w >> 2 + NG, ... (NG = warps / 4) in ascending order, each lane
// adding its four (row, column) terms (half_sb_terms, dense_tc.cu:100-330) from +0; the tile's warp sums are then
// added by the order's tree (docs/research/basalt-nvcc/dense_tc.md section 2). mma.sync's only role there is the exact
// int32 dot of a block (or half block) for 16 rows by 8 columns: lane 4g + t holds basalt's A and B fragments,
// fetches column c's code words from lane 4c + t, forms its partial dots with dp4a, and the four lanes of group g add
// them (integer adds, exact in any order), each keeping columns 2t and 2t + 1, the D fragment basalt's lane holds. The
// float steps are the ones nvcc compiles (dense_tc.md section 1): fma() exactly where SASS has FFMA, plain * and + at
// FMUL and FADD, nothing flushed (no subnormal reaches them). Basalt stages the tile in shared memory; staging is not
// arithmetic, so the weights are read from device memory (two 16-bit loads a word where a block is only 2-byte
// aligned). Rows past n_out read row n_out - 1 and columns at or past ncols get zero codes and d8 = 0; neither is
// stored. A NaN is stored as sm_120's (cuda_nan; docs/research/basalt-nvcc/nan.md).
//
// Macros: DTC_TYPE the ggml type id (11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 23 IQ4_XS, 42 Q2_0); DTC_ORDER 0 one, 1 tile,
// 2 s with KS 1, 3 s with KS 2 (monolith/kernels.py basalt_dense_tc_macros); DTC_TYPE0 / DTC_TYPE1 the pair's
// matrices' types. Grid: one threadgroup per 16-row tile (the pair: the first matrix's tiles, then the second's), of
// at least 4 SIMD-groups.
#pragma STDC FP_CONTRACT OFF

#ifndef DTC_TYPE
#define DTC_TYPE 12
#endif
#ifndef DTC_ORDER
#define DTC_ORDER 1
#endif
#ifndef DTC_TYPE0
#define DTC_TYPE0 DTC_TYPE
#endif
#ifndef DTC_TYPE1
#define DTC_TYPE1 DTC_TYPE0
#endif
#define DTC_NWV (DTC_ORDER == 0 ? 24 : DTC_ORDER == 2 ? 4 : 8)   // basalt's warps per tile

struct DtcParams { uint n_in; uint n_out; uint ncols; uint n_out1; };

// Fmt<TYPE>::SB (dense_tc.cu:35-42): the bytes of 256 values (Q2_0: four 64-value blocks)
static constexpr int dtc_sb(int t) {
    return t == 12 ? 144 : t == 13 ? 176 : t == 14 ? 210 : t == 23 ? 136 : t == 11 ? 110 : 72;
}

static inline uint ld16(device const uchar* p) { return uint(*(device const ushort*)p); }
static inline uint ld32(device const uchar* p) { return *(device const uint*)p; }                  // 4-byte aligned
static inline uint ld32_b2(device const uchar* p) { return ld16(p) | (ld16(p + 2) << 16); }         // 2-byte aligned
static inline int byte_of(uint w, int b) { return int((w >> (8 * b)) & 0xffu); }
static inline int sbyte_of(uint w, int b) { return int(as_type<char>(uchar(w >> (8 * b)))); }
static inline uint vsub4u(uint a, uint b) { return as_type<uint>(vsub4(as_type<int>(a), as_type<int>(b))); }

// The quad's sums of each lane's partial dots p0 (row g) and p1 (row g + 8) for columns 0..3 (lo) and 4..7 (hi): an
// exchange over XOR 2 leaves lanes t = 0, 1 columns 0..3 and lanes 2, 3 columns 4..7, one over XOR 1 leaves lane t
// columns 2t and 2t + 1. Returns the D fragment: (g, 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1).
static inline int4 quad_sum(int4 p0lo, int4 p0hi, int4 p1lo, int4 p1hi, ushort t) {
    const bool b1 = (t & 2) != 0, b0 = (t & 1) != 0;
    const int4 s0 = (b1 ? p0hi : p0lo) + simd_shuffle_xor(b1 ? p0lo : p0hi, ushort(2));
    const int4 s1 = (b1 ? p1hi : p1lo) + simd_shuffle_xor(b1 ? p1lo : p1hi, ushort(2));
    const int2 u0 = (b0 ? s0.zw : s0.xy) + simd_shuffle_xor(b0 ? s0.xy : s0.zw, ushort(1));
    const int2 u1 = (b0 ? s1.zw : s1.xy) + simd_shuffle_xor(b0 ? s1.xy : s1.zw, ushort(1));
    return int4(u0, u1);
}

// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 with C = 0 (dense_tc.cu:60-65): a0 / a1 rows g / g + 8 values
// 4t.., a2 / a3 values 16 + 4t..; b0 / b1 column g's codes 4t.. and 16 + 4t...
static inline int4 mma_k32(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, ushort t) {
    int p0[8], p1[8];
    for (ushort c = 0; c < 8; ++c) {
        const int x0 = simd_shuffle(as_type<int>(b0), ushort(4 * c + t));
        const int x1 = simd_shuffle(as_type<int>(b1), ushort(4 * c + t));
        p0[c] = dp4a(as_type<int>(a2), x1, dp4a(as_type<int>(a0), x0, 0));
        p1[c] = dp4a(as_type<int>(a3), x1, dp4a(as_type<int>(a1), x0, 0));
    }
    return quad_sum(int4(p0[0], p0[1], p0[2], p0[3]), int4(p0[4], p0[5], p0[6], p0[7]),
                    int4(p1[0], p1[1], p1[2], p1[3]), int4(p1[4], p1[5], p1[6], p1[7]), t);
}

// mma.sync.aligned.m16n8k16 (dense_tc.cu:66-70): a0 / a1 rows g / g + 8 values 4t.. of a 16-value half, b column g's.
static inline int4 mma_k16(uint a0, uint a1, uint b, ushort t) {
    int p0[8], p1[8];
    for (ushort c = 0; c < 8; ++c) {
        const int x = simd_shuffle(as_type<int>(b), ushort(4 * c + t));
        p0[c] = dp4a(as_type<int>(a0), x, 0);
        p1[c] = dp4a(as_type<int>(a1), x, 0);
    }
    return quad_sum(int4(p0[0], p0[1], p0[2], p0[3]), int4(p0[4], p0[5], p0[6], p0[7]),
                    int4(p1[0], p1[1], p1[2], p1[3]), int4(p1[4], p1[5], p1[6], p1[7]), t);
}

// The column data of a warp's four blocks (XFrag, dense_tc.cu:93-98): column g's codes 4t.. (b0) and 16 + 4t.. (b1),
// and the q8_1 scales of columns 2t and 2t + 1.
struct XFrag { uint b0[4]; uint b1[4]; float d8[4][2]; };

// Half super-block n (blocks 4n..4n + 3) of the super-blocks at p0 (row g) and p1 (row g + 8): the four (row, column)
// terms added to acc (half_sb_terms, dense_tc.cu:100-330; the float sites of dense_tc.md section 1).
template <int TYPE>
static inline void half_sb_terms(thread float4& acc, device const uchar* p0, device const uchar* p1, int n, ushort t,
                                 thread const XFrag& x) {
    device const uchar* p[2] = {p0, p1};
    if (TYPE == 12 || TYPE == 13) {                                    // :106-156
        uint qw[2][4], hw[2][2] = {{0u, 0u}, {0u, 0u}}, sw[2][3];
        float dmx[2], dmy[2];
        for (int h = 0; h < 2; ++h) {
            device const uchar* qs = p[h] + (TYPE == 12 ? 16 : 48) + 64 * n;
            for (int i = 0; i < 4; ++i) qw[h][i] = ld32(qs + 16 * i + 4 * t);   // pairs 2n, 2n + 1: words 4t, 16 + 4t
            if (TYPE == 13) {
                hw[h][0] = ld32(p[h] + 16 + 4 * t);
                hw[h][1] = ld32(p[h] + 32 + 4 * t);
            }
            for (int i = 0; i < 3; ++i) sw[h][i] = ld32(p[h] + 4 + 4 * i);
            dmx[h] = h2f(ushort(ld16(p[h])));
            dmy[h] = h2f(ushort(ld16(p[h] + 2)));
        }
        for (int i = 0; i < 4; ++i) {
            const int j = 4 * n + i, pp = i >> 1, sh = 4 * (i & 1);
            uint lo[2], hi[2];
            float f1[2], f2[2];
            for (int h = 0; h < 2; ++h) {
                lo[h] = (qw[h][2 * pp] >> sh) & 0x0f0f0f0fu;
                hi[h] = (qw[h][2 * pp + 1] >> sh) & 0x0f0f0f0fu;
                if (TYPE == 13) {
                    lo[h] |= ((hw[h][0] >> j) & 0x01010101u) << 4;
                    hi[h] |= ((hw[h][1] >> j) & 0x01010101u) << 4;
                }
                int sc, m;                                              // get_scale_min_k4
                if (n == 0) {
                    sc = byte_of(sw[h][0], i) & 63;
                    m = byte_of(sw[h][1], i) & 63;
                } else {
                    sc = (byte_of(sw[h][2], i) & 0xF) | ((byte_of(sw[h][0], i) >> 6) << 4);
                    m = (byte_of(sw[h][2], i) >> 4) | ((byte_of(sw[h][1], i) >> 6) << 4);
                }
                f1[h] = dmx[h] * float(sc);                              // FMUL
                f2[h] = dmy[h] * float(m);                               // FMUL
            }
            const int4 d = mma_k32(lo[0], lo[1], hi[0], hi[1], x.b0[i], x.b1[i], t);
            // the code sums of columns 2t and 2t + 1 (column g's from lanes 4g..4g + 3)
            int is = dp4a(0x01010101, as_type<int>(x.b0[i]), dp4a(0x01010101, as_type<int>(x.b1[i]), 0));
            is += simd_shuffle_xor(is, ushort(1));
            is += simd_shuffle_xor(is, ushort(2));
            const int isum[2] = {simd_shuffle(is, ushort(8 * t)), simd_shuffle(is, ushort(8 * t + 4))};
            for (int q = 0; q < 4; ++q) {
                // f1 * dot - f2 * isum: ptxas rounds the dmin product and fuses the d product (dense_tc.md, Q4_K)
                const float inner = fma(f1[q >> 1], float(d[q]), -(f2[q >> 1] * float(isum[q & 1])));
                acc[q] = fma(inner, x.d8[i][q & 1], acc[q]);
            }
        }
    } else if (TYPE == 14) {                                           // :157-195
        uint lw[2][4], hw[2][2], sw[2][2];
        float dd[2];
        for (int h = 0; h < 2; ++h) {
            device const uchar* ql = p[h] + 64 * n;
            device const uchar* qh = p[h] + 128 + 32 * n;
            for (int i = 0; i < 4; ++i) lw[h][i] = ld32_b2(ql + 16 * i + 4 * t);
            hw[h][0] = ld32_b2(qh + 4 * t);
            hw[h][1] = ld32_b2(qh + 16 + 4 * t);
            sw[h][0] = ld32_b2(p[h] + 192 + 8 * n);
            sw[h][1] = ld32_b2(p[h] + 196 + 8 * n);
            dd[h] = h2f(ushort(ld16(p[h] + 208)));
        }
        float4 part = 0.0f;
        for (int i = 0; i < 4; ++i) {   // block 4n + i: ql bytes 32 (i & 1) + .., nibble i >> 1, qh bits 2i
            const int sh = 4 * (i >> 1), sq = 2 * i, li = 2 * (i & 1);
            uint lo[2], hi[2];
            int sc0[2], sc1[2];
            for (int h = 0; h < 2; ++h) {
                lo[h] = vsub4u(((lw[h][li] >> sh) & 0x0f0f0f0fu) | (((hw[h][0] >> sq) & 0x03030303u) << 4), 0x20202020u);
                hi[h] = vsub4u(((lw[h][li + 1] >> sh) & 0x0f0f0f0fu) | (((hw[h][1] >> sq) & 0x03030303u) << 4),
                               0x20202020u);
                sc0[h] = sbyte_of(sw[h][i >> 1], 2 * (i & 1));
                sc1[h] = sbyte_of(sw[h][i >> 1], 2 * (i & 1) + 1);
            }
            const int4 e0 = mma_k16(lo[0], lo[1], x.b0[i], t);
            const int4 e1 = mma_k16(hi[0], hi[1], x.b1[i], t);
            for (int q = 0; q < 4; ++q)
                part[q] = fma(float(sc0[q >> 1] * e0[q] + sc1[q >> 1] * e1[q]), x.d8[i][q & 1], part[q]);
        }
        for (int q = 0; q < 4; ++q) acc[q] = fma(dd[q >> 1], part[q], acc[q]);
    } else if (TYPE == 11) {                                           // :196-234; Q3_K: hmask[32], qs[64], scales[12], d
        uint qw[2][2], hw[2][2], sc[2][2];
        float dd[2];
        for (int h = 0; h < 2; ++h) {
            qw[h][0] = ld32_b2(p[h] + 32 + 32 * n + 4 * t);           // values 4t.. of every block of the half
            qw[h][1] = ld32_b2(p[h] + 48 + 32 * n + 4 * t);           // values 16 + 4t..
            hw[h][0] = ld32_b2(p[h] + 4 * t);
            hw[h][1] = ld32_b2(p[h] + 16 + 4 * t);
            const uint a0 = ld32_b2(p[h] + 96), a1 = ld32_b2(p[h] + 100), a2 = ld32_b2(p[h] + 104);
            // the half's eight 6-bit scales (ggml's aux[2n], aux[2n + 1])
            sc[h][0] = ((n ? a0 >> 4 : a0) & 0x0f0f0f0fu) | (((a2 >> (4 * n)) & 0x03030303u) << 4);
            sc[h][1] = ((n ? a1 >> 4 : a1) & 0x0f0f0f0fu) | (((a2 >> (4 * n + 2)) & 0x03030303u) << 4);
            dd[h] = h2f(ushort(ld16(p[h] + 108)));
        }
        float4 part = 0.0f;
        for (int i = 0; i < 4; ++i) {   // block 4n + i: 2-bit plane i of the half's qs, hmask bit 4n + i
            const int j = 4 * n + i;
            uint lo[2], hi[2];
            int sc0[2], sc1[2];
            for (int h = 0; h < 2; ++h) {
                lo[h] = vsub4u(((qw[h][0] >> (2 * i)) & 0x03030303u) | (((hw[h][0] >> j) & 0x01010101u) << 2),
                               0x04040404u);
                hi[h] = vsub4u(((qw[h][1] >> (2 * i)) & 0x03030303u) | (((hw[h][1] >> j) & 0x01010101u) << 2),
                               0x04040404u);
                sc0[h] = byte_of(sc[h][i >> 1], 2 * (i & 1)) - 32;
                sc1[h] = byte_of(sc[h][i >> 1], 2 * (i & 1) + 1) - 32;
            }
            const int4 e0 = mma_k16(lo[0], lo[1], x.b0[i], t);
            const int4 e1 = mma_k16(hi[0], hi[1], x.b1[i], t);
            for (int q = 0; q < 4; ++q)
                part[q] = fma(float(sc0[q >> 1] * e0[q] + sc1[q >> 1] * e1[q]), x.d8[i][q & 1], part[q]);
        }
        for (int q = 0; q < 4; ++q) acc[q] = fma(dd[q >> 1], part[q], acc[q]);
    } else if (TYPE == 42) {                                           // :235-270; four 18-byte blocks d, qs[16]
        for (int b = 0; b < 2; ++b) {   // Q2_0 block 2n + b: the 32-value blocks 4n + 2b, 4n + 2b + 1
            float dw[2];
            uint qb[2][2][2];
            for (int h = 0; h < 2; ++h) {
                device const uchar* bp = p[h] + 18 * (2 * n + b);
                dw[h] = h2f(ushort(ld16(bp)));
                for (int k = 0; k < 2; ++k) {
                    qb[h][k][0] = bp[2 + 8 * k + t];
                    qb[h][k][1] = bp[6 + 8 * k + t];
                }
            }
            float4 part = 0.0f;
            for (int k = 0; k < 2; ++k) {
                const int i = 2 * b + k;
                uint v[2][2];
                for (int h = 0; h < 2; ++h)
                    for (int e = 0; e < 2; ++e) {
                        const uint q = qb[h][k][e];
                        v[h][e] = vsub4u((q | q << 6 | q << 12 | q << 18) & 0x03030303u, 0x01010101u);
                    }
                const int4 d = mma_k32(v[0][0], v[1][0], v[0][1], v[1][1], x.b0[i], x.b1[i], t);
                for (int q = 0; q < 4; ++q) part[q] = fma(float(d[q]), x.d8[i][q & 1], part[q]);
            }
            for (int q = 0; q < 4; ++q) acc[q] = fma(dw[q >> 1], part[q], acc[q]);
        }
    } else {                                                           // :297-329; IQ4_XS
        uint qw[2][4];
        int sl[2], shh[2];
        float dw[2];
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 4; ++i) qw[h][i] = ld32(p[h] + 8 + 16 * (4 * n + i) + 4 * t);
            sl[h] = int(ld16(p[h] + 4 + 2 * n));
            shh[h] = int(ld16(p[h] + 2));
            dw[h] = h2f(ushort(ld16(p[h])));
        }
        float4 part = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int j = 4 * n + i;
            uint lo[2], hi[2];
            int ls[2];
            for (int h = 0; h < 2; ++h) {
                const int2 v = iq4_table_lookup(as_type<int>(qw[h][i]));   // dense_tc.cu's iq4_lookup: the same bytes
                lo[h] = as_type<uint>(v.x);
                hi[h] = as_type<uint>(v.y);
                ls[h] = (((sl[h] >> (4 * i)) & 0xf) | (((shh[h] >> (2 * j)) & 3) << 4)) - 32;
            }
            const int4 d = mma_k32(lo[0], lo[1], hi[0], hi[1], x.b0[i], x.b1[i], t);
            for (int q = 0; q < 4; ++q) part[q] = fma(float(d[q] * ls[q >> 1]), x.d8[i][q & 1], part[q]);
        }
        for (int q = 0; q < 4; ++q) acc[q] = fma(dw[q >> 1], part[q], acc[q]);
    }
}

// The warp sums of the tile at row0 of a matrix with NWV warps a tile, into red[warp][row][column] (16 x 8 floats a
// warp): warp w's chunks and its terms as in dense_tc_one_block / dense_tc_tile_block / dense_tc_block
// (dense_tc.cu:519-540, 639-659, 375-406).
template <int TYPE, int NWV>
static inline void warp_sums(threadgroup float* red, device const uchar* w, int row0, int n_out, device const uint* x,
                             int n_in, int ncols, ushort sg, ushort nsg, ushort lane) {
    const int g = lane >> 2;
    const ushort t = lane & 3;
    const int nchunk = n_in / 512, nb32 = n_in / 32;
    const uint row_bytes = uint(n_in / 256) * uint(dtc_sb(TYPE));
    device const uchar* rowA = w + uint(min(row0 + g, n_out - 1)) * row_bytes;
    device const uchar* rowB = w + uint(min(row0 + g + 8, n_out - 1)) * row_bytes;
    device const uint* xg = x + 9u * uint(g) * uint(nb32);
    device const uint* xc0 = x + 9u * uint(2 * t) * uint(nb32);
    device const uint* xc1 = x + 9u * uint(2 * t + 1) * uint(nb32);
    for (int vw = sg; vw < NWV; vw += nsg) {
        const int hb = vw & 3;
        float4 acc = 0.0f;
        for (int c = vw >> 2; c < nchunk; c += NWV / 4) {
            XFrag xf;
            for (int k4 = 0; k4 < 4; ++k4) {
                const int k = 16 * c + 4 * hb + k4;                    // the 32-value block of the columns
                xf.b0[k4] = g < ncols ? uint(q8_w(xg, k, t)) : 0u;
                xf.b1[k4] = g < ncols ? uint(q8_w(xg, k, 4 + t)) : 0u;
                xf.d8[k4][0] = 2 * t < ncols ? q8_d(xc0, k) : 0.0f;
                xf.d8[k4][1] = 2 * t + 1 < ncols ? q8_d(xc1, k) : 0.0f;
            }
            const uint off = uint(2 * c + (hb >> 1)) * uint(dtc_sb(TYPE));
            half_sb_terms<TYPE>(acc, rowA + off, rowB + off, hb & 1, t, xf);
        }
        threadgroup float* rw = red + 128 * vw;
        rw[g * 8 + 2 * t] = acc[0];
        rw[g * 8 + 2 * t + 1] = acc[1];
        rw[(g + 8) * 8 + 2 * t] = acc[2];
        rw[(g + 8) * 8 + 2 * t + 1] = acc[3];
    }
}

// A (row, column)'s value from the warp sums rr[128 * w] (dense_tc.md section 2): one (:667-681), tile and pair
// (:552-560, :919-928), s with KS 1 and 2 (:407-427).
template <int ORDER>
static inline float tree(threadgroup const float* rr) {
    if (ORDER == 0) {
        float h[4];
        for (int q = 0; q < 4; ++q) {
            float v = rr[128 * q];
            for (int j = 1; j < 6; ++j) v = v + rr[128 * (4 * j + q)];
            h[q] = v;
        }
        return (h[0] + h[1]) + (h[2] + h[3]);
    } else if (ORDER == 1) {
        return ((rr[0] + rr[512]) + (rr[128] + rr[640])) + ((rr[256] + rr[768]) + (rr[384] + rr[896]));
    } else if (ORDER == 2) {
        return ((rr[0] + rr[128]) + rr[256]) + rr[384];
    } else {
        const float v = ((rr[0] + rr[128]) + rr[256]) + rr[384];
        return v + (((rr[512] + rr[640]) + rr[768]) + rr[896]);
    }
}

kernel void basalt_dense_tc(device const uchar* w [[buffer(0)]], device const uint* x [[buffer(1)]],
                            device float* y [[buffer(2)]], constant DtcParams& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float red[DTC_NWV * 128];
    const int n_out = int(p.n_out), ncols = int(p.ncols), row0 = 16 * int(tg);
    warp_sums<DTC_TYPE, DTC_NWV>(red, w, row0, n_out, x, int(p.n_in), ncols, sg, nsg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tid = 32 * int(sg) + int(lane), row = tid >> 3, col = tid & 7;
    if (tid < 128 && col < ncols && row0 + row < n_out)
        y[col * n_out + row0 + row] = cuda_nan(tree<DTC_ORDER>(red + row * 8 + col));
}

// dense_tc_pair (dense_tc.cu:704-931): two matrices of one input, the first matrix's tiles (n_out rows) then the
// second's (n_out1 rows), each in its own format with the tile order's arithmetic.
kernel void basalt_dense_tc_pair(device const uchar* w0 [[buffer(0)]], device const uchar* w1 [[buffer(1)]],
                                 device const uint* x [[buffer(2)]], device float* y0 [[buffer(3)]],
                                 device float* y1 [[buffer(4)]], constant DtcParams& p [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float red[8 * 128];
    const int nt0 = (int(p.n_out) + 15) / 16, ncols = int(p.ncols);
    const bool second = int(tg) >= nt0;
    const int n_out = second ? int(p.n_out1) : int(p.n_out), row0 = 16 * (second ? int(tg) - nt0 : int(tg));
    if (second)
        warp_sums<DTC_TYPE1, 8>(red, w1, row0, n_out, x, int(p.n_in), ncols, sg, nsg, lane);
    else
        warp_sums<DTC_TYPE0, 8>(red, w0, row0, n_out, x, int(p.n_in), ncols, sg, nsg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tid = 32 * int(sg) + int(lane), row = tid >> 3, col = tid & 7;
    if (tid < 128 && col < ncols && row0 + row < n_out)
        (second ? y1 : y0)[col * n_out + row0 + row] = cuda_nan(tree<1>(red + row * 8 + col));
}
