// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/s2_expert_grouped.cu:34-46, :104-108, :165-171, :215-279,
// :736-1083 and src/kernels/cuda/quantize_act.cu:74-101 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
// third_party/NOTICE).
//
// The drafter's routed Q2_0 experts (src/core/mtp.cpp:931-935): basalt_s2_group_resident builds the plan on the device,
// basalt_quantize_q8_0_scaled quantizes the activations, then moe_grouped_s2 (s2_expert_grouped.cu:1085-1122) is four
// dispatches in order: basalt_s2_gu, basalt_s2_swiglu, basalt_quantize_q8_0_scaled over the intermediate and
// basalt_s2_down.
// Needs basalt_fp.metal and basalt_q8_1.metal ahead of it.
//
// Both files build with nvcc's IEEE defaults: every float operation is IEEE's, subnormals kept
// (docs/research/basalt-nvcc/s2_experts.md). This GPU flushes subnormals, and here the activation scale is the
// quantizer's fp32 s, any non-negative float, so a subnormal can occur at every rounding site: each goes through
// basalt_q8_1.metal's IEEE helpers (ieee_div127, ieee_rcp, ieee_mul, ieee_fma, ieee_div, warp_sum_ieee), whose
// hardware path is the plain operation (fma() where SASS has FFMA), and every compare that a subnormal could reach is
// on bits. Contraction is off (basalt_fp.metal's pragma). A NaN can come only from FFMA, FADD, FMUL and div.rn.f32 (or
// pass through them), each of which gives sm_120's canonical 0x7fffffff, so every float the gate/up, SwiGLU and down
// kernels store goes through cuda_nan (this GPU's NaN is 0x7fc00000).
//
// The plan (basalt's): n_groups[0] groups; group g's blob starts at grp_ptr[g] (a table of 64-bit GPU addresses, one a
// group, each 2-byte aligned and inside a buffer the dispatch makes resident, so the blobs may sit in several buffers)
// and owns the entries [grp_start[g], grp_start[g + 1]), of which the kernels take at most GMAX = 8 (basalt's
// min(..., GMAX)). Entry e reads token ent_tok[e]'s activation row (80 blocks of 34 bytes, x 4-byte aligned) with its
// fp32 scales, and writes row ent_dst[e] of `out` (2560 floats). The blob (cpu/expert.hpp, restated at
// s2_expert_grouped.cu:34-46): 1280 gate/up code rows of 640 bytes (row slot 2r is gate row r, 2r + 1 up row r; value
// 4j + k in bits 2k of byte j), 2560 down code rows of 160 bytes, the gate/up fp16 scales (40 a row), the down scales
// (10 a row); a weight is d_w * (code - 1).
// Scratch, basalt's (one buffer, 4-byte aligned, bound at these offsets): gate/up (entry e's gate row at e * 640, its
// up row at (cap + e) * 640 floats), the quantized intermediate (cap * 20 blocks of 34 bytes, from the gate/up bytes
// rounded up to 16), its fp32 scales (cap * 20, from the next multiple of 16).
#pragma STDC FP_CONTRACT OFF

#define S2_H 2560
#define S2_FF 640
#define S2_GMAX 8                                   // entries a group the kernels take (s2_expert_grouped.cu:742)
#define S2_ROW_GU (S2_H / 4)                        // bytes of a gate/up code row
#define S2_ROW_D (S2_FF / 4)                        // bytes of a down code row
#define S2_SC_GU (S2_H / 64)                        // fp16 scales of a gate/up row
#define S2_SC_D (S2_FF / 64)                        // of a down row
#define S2_O_D_CODES (2 * S2_FF * S2_ROW_GU)
#define S2_O_GU_SCALES (S2_O_D_CODES + S2_H * S2_ROW_D)
#define S2_O_D_SCALES (S2_O_GU_SCALES + 2 * S2_FF * S2_SC_GU * 2)

struct Q80Params { ulong n_blocks; ulong pad0; };                 // n_blocks: basalt's long long (quantize_act.cu:79)
struct S2Params { uint cap_entries; uint pad0; uint pad1; uint pad2; };
struct S2ResidentParams { int n; int k_per_tok; uint blob; uint pad0; };

// ---- quantize_q8_0_scaled ---------------------------------------------------------------------------------------

// quantize_q8_0_scaled_warp_kernel (quantize_act.cu:78-101; s2_experts.md section 1): SIMD-group = 34-byte block,
// lane = value; grid ceil(n_blocks * 32 / 256) threadgroups of 256 (whole SIMD-groups past the end return). The block
// index and the offsets are 64-bit, as basalt's long long (:80-83, :89): past 2^32 values the grid's 32-bit
// thread_position_in_grid would wrap. amax is the largest |x| by its bits (fmaxf drops a NaN; an all-NaN block has
// amax NaN and s 0); s = amax / 127 and inv = 1 / s correctly rounded, subnormals kept; t = fl(x * inv), r = fl(t +-
// 0.5) (FMUL then FADD: not fused); the code is (int) r clamped to +-127, NaN to 0. The block's fp16 d is
// f16_from_f32(s) (round to nearest even), the fp32 scale s.
kernel void basalt_quantize_q8_0_scaled(device const float* x [[buffer(0)]], device uchar* blocks [[buffer(1)]],
                                        device float* scales [[buffer(2)]], constant Q80Params& p [[buffer(3)]],
                                        uint tg [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]]) {
    const ulong b = (ulong(tg) * tpg + tid) >> 5;
    const uint lane = tid & 31u;
    if (b >= p.n_blocks) return;
    const float xv = x[b * 32ul + lane];
    int key = isnan(xv) ? -1 : int(as_type<uint>(xv) & 0x7fffffffu);
    for (ushort o = 16; o > 0; o >>= 1) key = max(key, simd_shuffle_xor(key, o));
    const float s = key > 0 ? ieee_div127(as_type<float>(uint(key))) : 0.0f;   // amax > 0.f, by its bits
    const float inv = as_type<uint>(s) != 0u ? ieee_rcp(s) : 0.0f;              // s > 0.f: s is never negative or NaN
    device uchar* out = blocks + b * 34ul;
    if (lane == 0u) {
        scales[b] = s;
        const ushort d16 = f16_bits(s);
        out[0] = uchar(d16 & 0xffu);
        out[1] = uchar(d16 >> 8);
    }
    const float t = ieee_mul(xv, inv);
    const float r = t + (t >= 0.0f ? 0.5f : -0.5f);    // a subnormal t gives code 0 whichever way the compare goes
    out[2u + lane] = as_type<uchar>(q8_1_code(r));
}

// ---- the plan: moe_group_resident --------------------------------------------------------------------------------

// group_resident_kernel (s2_expert_grouped.cu:1033-1073): one threadgroup of 128 over the n <= 128 routed ids; a group
// per distinct expert in first-appearance order, its entries in routing order; group g's blob at base + id * blob
// bytes (base: the resident experts' buffer, bound at its first blob), ent_dst = the routing index, ent_tok = that
// index / k_per_tok; counts = {groups, entries}. Integer work only.
kernel void basalt_s2_group_resident(device const int* ids [[buffer(0)]], device const uchar* base [[buffer(1)]],
                                     device GrpPtr* grp_ptr [[buffer(2)]], device int* grp_start [[buffer(3)]],
                                     device int* counts [[buffer(4)]], device int* ent_dst [[buffer(5)]],
                                     device int* ent_tok [[buffer(6)]], constant S2ResidentParams& p [[buffer(7)]],
                                     ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129];
    const int i = int(tid), n = p.n;
    const int e = i < n ? ids[i] : -1;
    e_s[i] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) {
                if (first == i) first = j;
                ++rank;
            }
        if (first == i)
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi].p = (device const ushort*) (base + ulong(long(e_s[j])) * ulong(p.blob));
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = i;
        ent_tok[at] = i / p.k_per_tok;
    }
}

// ---- the grouped kernels' integer parts --------------------------------------------------------------------------

// load_x_chunk (s2_expert_grouped.cu:240-270): the 32 int8 of the block_q8_0 at byte xb (64-bit, as basalt's pointer)
// of the 4-byte aligned row words xw as the eight words X[4h + f] = {x[16h + f], x[16h + 4 + f], x[16h + 8 + f],
// x[16h + 12 + f]}, read as nine aligned words (the ninth, inside the row's next block, only when the codes start 2
// bytes into a word: an even chunk, never a row's last), and their sum hx.
static inline int s2_load_chunk(device const uint* xw, ulong xb, thread int* X) {
    const ulong q = xb + 2ul;
    const uint off = uint(q & 3ul), sh = off * 8u;
    device const uint* w = xw + (q - off) / 4ul;
    uint v[9];
    for (int j = 0; j < 8; ++j) v[j] = w[j];
    v[8] = off != 0u ? w[8] : 0u;
    int n[8];
    int hx = 0;
    for (int j = 0; j < 8; ++j) {
        n[j] = int(uint(((ulong(v[j + 1]) << 32) | ulong(v[j])) >> sh));
        hx = dp4a(0x01010101, n[j], hx);
    }
    for (int h = 0; h < 2; ++h) {
        const int t0 = byte_perm(n[4 * h], n[4 * h + 1], 0x5140);
        const int t1 = byte_perm(n[4 * h], n[4 * h + 1], 0x7362);
        const int t2 = byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);
        const int t3 = byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);
        X[4 * h + 0] = byte_perm(t0, t2, 0x5410);
        X[4 * h + 1] = byte_perm(t0, t2, 0x7632);
        X[4 * h + 2] = byte_perm(t1, t3, 0x5410);
        X[4 * h + 3] = byte_perm(t1, t3, 0x7632);
    }
    return hx;
}

// stage_chunk (:895-902): chunk i = k * NC + c of a group, its words at xs_w[j * STRIDE + i] and (dx as bits, hx) at
// xs_dh[i].
template <int STRIDE>
static inline void s2_stage(device const uint* xw, ulong xb, float dx, threadgroup int* xs_w, threadgroup int2* xs_dh,
                            int i) {
    int X[8];
    const int hx = s2_load_chunk(xw, xb, X);
    for (int j = 0; j < 8; ++j) xs_w[j * STRIDE + i] = X[j];
    xs_dh[i] = int2(as_type<int>(dx), hx);
}

// expand_codes (:221-231): a chunk's 8 code bytes (four 16-bit reads of a 2-byte aligned row: CUDA's uint2 load, the
// same bits) as the eight dp4a operands m[4h + f], byte b the code of value 16h + 4b + f.
static inline void s2_expand(device const ushort* cw, thread int* m) {
    const uint x = uint(cw[0]) | (uint(cw[1]) << 16), y = uint(cw[2]) | (uint(cw[3]) << 16), M = 0x03030303u;
    m[0] = int(x & M);
    m[1] = int((x >> 2) & M);
    m[2] = int((x >> 4) & M);
    m[3] = int((x >> 6) & M);
    m[4] = int(y & M);
    m[5] = int((y >> 2) & M);
    m[6] = int((y >> 4) & M);
    m[7] = int((y >> 6) & M);
}

// chunk_s (:273-278): sum of code * x over the chunk
static inline int s2_chunk_s(thread const int* m, thread const int* X) {
    int s = 0;
    for (int j = 0; j < 8; ++j) s = dp4a(m[j], X[j], s);
    return s;
}

// ---- gate / up -----------------------------------------------------------------------------------------------------

// gu_grouped_t_kernel (:904-968; s2_experts.md section 2): grid (40, cap_groups) threadgroups of 256; threadgroup
// (bx, g) takes row slots [32 bx, 32 bx + 32) of group g: SIMD-group w the (gate r, up r) pairs w and w + 8. Per
// (row, entry), lane L: the chunks c = L, L + 32, L + 64 (< 80) in order, acc = fma(fl(dw * dx), f(s - hx), acc) from
// +0 (FFMA), then basalt's warp_sum (shift-down; lane 0 is the XOR butterfly's lane 0), lane 0 stores. Activation
// offsets and gate_up indices are 64-bit where basalt's are size_t (:921-922, :963-964).
kernel void basalt_s2_gu(device const GrpPtr* grp_ptr [[buffer(0)]], device const int* grp_start [[buffer(1)]],
                         device const int* n_groups [[buffer(2)]], device const int* ent_tok [[buffer(3)]],
                         device const uint* x_q8_0 [[buffer(4)]], device const float* x_scales [[buffer(5)]],
                         device float* gate_up [[buffer(6)]], constant S2Params& p [[buffer(7)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]], ushort warp [[simdgroup_index_in_threadgroup]]) {
    constexpr int NC = S2_H / 32;
    threadgroup int xs_w[8 * S2_GMAX * NC];         // 20 KB: word j of entry k's chunk c at [j][k * NC + c]
    threadgroup int2 xs_dh[S2_GMAX * NC];           // 5 KB: (dx as bits, hx)
    const int g = int(tg.y);
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, S2_GMAX);
    for (int i = int(tid); i < ne * NC; i += 256) {
        const int k = i / NC, c = i - k * NC;
        const int tok = ent_tok[e0 + k];
        s2_stage<S2_GMAX * NC>(x_q8_0, ulong(tok) * NC * 34 + ulong(c) * 34, x_scales[ulong(tok) * NC + ulong(c)], xs_w,
                               xs_dh, i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort* blob = grp_ptr[g].p;
    const int row0 = int(tg.x) * 32;                  // even: a pair is always (gate r, up r)
    for (int pp = int(warp); pp < 16; pp += 8) {
        const int i = row0 + 2 * pp;
        device const ushort* codes = blob + i * (S2_ROW_GU / 2);
        device const ushort* scales = blob + S2_O_GU_SCALES / 2 + i * S2_SC_GU;
        int m0[3][8], m1[3][8];
        float dw0[3], dw1[3];
        for (int q = 0; q < 3; ++q) {
            const int c = int(lane) + 32 * q;
            if (c < NC) {
                s2_expand(codes + 4 * c, m0[q]);
                s2_expand(codes + S2_ROW_GU / 2 + 4 * c, m1[q]);
                dw0[q] = h2f(scales[c >> 1]);
                dw1[q] = h2f(scales[S2_SC_GU + (c >> 1)]);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int q = 0; q < 3; ++q) {
                const int c = int(lane) + 32 * q;
                if (c >= NC) break;
                const int at = k * NC + c;
                int X[8];
                for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (S2_GMAX * NC) + at];
                const int2 dh = xs_dh[at];
                const float dx = as_type<float>(dh.x);
                acc0 = ieee_fma(ieee_mul(dw0[q], dx), float(s2_chunk_s(m0[q], X) - dh.y), acc0);
                acc1 = ieee_fma(ieee_mul(dw1[q], dx), float(s2_chunk_s(m1[q], X) - dh.y), acc1);
            }
            const float s0 = warp_sum_ieee(acc0), s1 = warp_sum_ieee(acc1);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                gate_up[ulong(e) * S2_FF + ulong(r)] = cuda_nan(s0);
                gate_up[ulong(p.cap_entries) * S2_FF + ulong(e) * S2_FF + ulong(r)] = cuda_nan(s1);
            }
        }
    }
}

// ---- SwiGLU ---------------------------------------------------------------------------------------------------------

// swiglu_kernel (:165-171; s2_experts.md section 3): gate_up[i] = (g / (1 + __expf(-g))) * u in place over the gate
// half, every capacity entry (i < cap * 640); grid ceil(cap * 640 / 256) threadgroups of 256; the pair count and i are
// 64-bit, as basalt's long long (:165-166). __expf(-g) is ex2.approx.f32 of fl(g * -log2 e) (expf_neg), the add plain
// (1 + a subnormal e is 1 either way), the division and the product IEEE's.
kernel void basalt_s2_swiglu(device float* gate_up [[buffer(0)]], constant S2Params& p [[buffer(1)]],
                             uint tg [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const ulong n_pairs = ulong(p.cap_entries) * S2_FF, i = ulong(tg) * tpg + tid;
    if (i >= n_pairs) return;
    const float g = gate_up[i], u = gate_up[n_pairs + i];
    gate_up[i] = cuda_nan(ieee_mul(ieee_div(g, expf_neg(g) + 1.0f), u));
}

// ---- down -----------------------------------------------------------------------------------------------------------

// down_grouped_t_kernel (:970-1027; s2_experts.md section 5): grid (40, cap_groups) threadgroups of 256; 64 rows a
// threadgroup, SIMD-group w the row pairs (r, r + 1) w, w + 8, w + 16, w + 24; lane c < 20 holds chunk c: acc =
// fma(fl(dw * dx), f(s - hx), +0), the other lanes +0; warp_sum; lane 0 stores rows r and r + 1 of ent_dst[e].
// Intermediate offsets and out indices are 64-bit where basalt's are size_t (:986-987, :1021).
kernel void basalt_s2_down(device const GrpPtr* grp_ptr [[buffer(0)]], device const int* grp_start [[buffer(1)]],
                           device const int* n_groups [[buffer(2)]], device const int* ent_dst [[buffer(3)]],
                           device const uint* h_q8_0 [[buffer(4)]], device const float* h_scales [[buffer(5)]],
                           device float* out [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                           ushort warp [[simdgroup_index_in_threadgroup]]) {
    constexpr int NC = S2_FF / 32;
    threadgroup int hs_w[8 * S2_GMAX * NC];
    threadgroup int2 hs_dh[S2_GMAX * NC];
    const int g = int(tg.y);
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, S2_GMAX);
    for (int i = int(tid); i < ne * NC; i += 256) {
        const int k = i / NC, c = i - k * NC;
        const ulong e = ulong(e0 + k);
        s2_stage<S2_GMAX * NC>(h_q8_0, e * NC * 34 + ulong(c) * 34, h_scales[e * NC + ulong(c)], hs_w, hs_dh, i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort* blob = grp_ptr[g].p;
    const int row0 = int(tg.x) * 64, c = int(lane);
    for (int pp = int(warp); pp < 32; pp += 8) {
        const int r = row0 + 2 * pp;
        device const ushort* codes = blob + S2_O_D_CODES / 2 + r * (S2_ROW_D / 2);
        device const ushort* scales = blob + S2_O_D_SCALES / 2 + r * S2_SC_D;
        int m0[8], m1[8];
        float dw0 = 0.0f, dw1 = 0.0f;
        if (c < NC) {
            s2_expand(codes + 4 * c, m0);
            s2_expand(codes + S2_ROW_D / 2 + 4 * c, m1);
            dw0 = h2f(scales[c >> 1]);
            dw1 = h2f(scales[S2_SC_D + (c >> 1)]);
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            if (c < NC) {
                const int at = k * NC + c;
                int X[8];
                for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (S2_GMAX * NC) + at];
                const int2 dh = hs_dh[at];
                const float dx = as_type<float>(dh.x);
                acc0 = ieee_fma(ieee_mul(dw0, dx), float(s2_chunk_s(m0, X) - dh.y), acc0);
                acc1 = ieee_fma(ieee_mul(dw1, dx), float(s2_chunk_s(m1, X) - dh.y), acc1);
            }
            const float s0 = warp_sum_ieee(acc0), s1 = warp_sum_ieee(acc1);
            if (lane == 0) {
                const ulong o = ulong(ent_dst[e0 + k]) * S2_H + ulong(r);
                out[o] = cuda_nan(s0);
                out[o + 1] = cuda_nan(s1);
            }
        }
    }
}
