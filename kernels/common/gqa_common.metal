// Shared by the attention kernels (gqa_decode.metal v1 and gqa_decode_v2.metal): the params record, the BF16
// helpers, the per-head norm + RoPE of one row held lane-per-dim (D/32 dims per lane, rotary partner on lane ^ 16).
#ifndef STEP_STATE
#define STEP_STATE 0                 // 1: position and T come from the bound StepState (the step program); 0: from params
#endif
#ifndef DRAFT
#define DRAFT 0
#endif
#ifndef STEAL
#define STEAL 0                      // 1: gqa_decode claims its blocks own-slice-first-then-steal through the cursors of buffer 10 (#44)
#endif
#ifndef STEAL_HITS
#define STEAL_HITS 0                 // 1 (tests): count the claims per block in buffer 12
#endif
#ifndef CH
#define CH 64u
#endif
#ifndef RBMAX
#define RBMAX 8u
#endif
#ifndef NSG
#define NSG 12u                      // SIMD-groups per threadgroup (the crew geometry)
#endif
#ifndef QK_NORM
#define QK_NORM 1                  // disable for attention without per-head query/key RMSNorm
#endif
#define DL (D / 32u)

#ifndef LM_MODE
#define LM_MODE 0                    // an LM drafter's attention (design §5.8): 1 = its ingest pass (T = n_inject rows ending at position),
#endif                               // 2 = chain step CHAIN_I (T = n_chain rows at position + CHAIN_I), 3 = the first chain step
                                     // (n_inject ingest rows then the anchor: T = n_inject + n_chain from position - n_inject); 0 = the target's
#ifndef CHAIN_I
#define CHAIN_I 0u
#endif
#ifndef PV_UNROLL
#define PV_UNROLL 8u                 // keys whose values the P·V pass loads ahead of consuming them (divides 32)
#endif

// The keys per chunk this step (the core and the merge compute the same value): the smallest chunk down to 16 keys
// whose blocks of a T = 1 step still fit one wave of the crew — a short context on a small model is 16–24 blocks of
// 64 keys over 240 SIMD-groups, each walking its keys alone, and 16-key chunks put four times the SIMD-groups on it
// (decode-kernels.md §11); a context whose 64-key blocks already exceed a wave keeps them (halving there only adds a
// tail wave and merge work: measured slower at 1024 keys). Bounded by the partial workspace's chunk count;
// independent of T, so every T's rows sum in the same order (a drafter's chain row and the target's verify row of
// one position agree to the bit).
#define GQA_SMALL_CONTEXT 256u
static inline uint pick_chunk(uint ctx, uint kv_heads, uint rep, uint n_sg, uint n_chunks_max) {
#if ADAPTIVE_CHUNK
  return ctx <= GQA_SMALL_CONTEXT ? 32u : CH;
#endif
#if FIXED_CHUNK
  return CH;
#endif
  const uint n_rg1 = (rep + RBMAX - 1u) / RBMAX;
  uint ch = CH;
  while (ch > 16u) {
    const uint ch2 = ch / 2u, n_half = (ctx + ch2 - 1u) / ch2;
    if (kv_heads * n_rg1 * n_half > n_sg || n_half > n_chunks_max) break;
    ch = ch2;
  }
  return ch;
}

struct GqaParams {
  uint heads; uint kv_heads; uint t_active; uint position;
  uint n_sg; uint q_off; uint gate_off; uint k_off;
  uint v_off; uint in_stride; uint out_stride; uint ctx_max;
  float eps; float scaling; uint has_gate; uint n_chunks_max;
  uint rows_max; uint pad0; uint pad1; uint nominal_sg;                     // pad0 / pad1: the DRAFT variant's n_new and second row stride;
};                                                                         // nominal_sg: the crew the STEAL variant's slices are cut for

static inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline ushort bf16bits(float v) { return ushort(as_type<uint>(round_bf16(v)) >> 16); }

static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }

// a lane's DL BF16 values (its slice of a D-vector: rows are D-strided, so the slice is DL·2-byte aligned) as one
// vector access — the scalar form issued DL loads per key, and the P·V pass over a 32-key chunk was 4× the
// instructions it needed (decode-kernels.md §11)
static inline void load_dl(device const ushort* p, thread float* f) {
#if DL == 8
  const uint4 q = *(device const uint4*)p;
  f[0] = bf16lo(q.x); f[1] = bf16hi(q.x); f[2] = bf16lo(q.y); f[3] = bf16hi(q.y);
  f[4] = bf16lo(q.z); f[5] = bf16hi(q.z); f[6] = bf16lo(q.w); f[7] = bf16hi(q.w);
#elif DL == 4
  const uint2 q = *(device const uint2*)p;
  f[0] = bf16lo(q.x); f[1] = bf16hi(q.x); f[2] = bf16lo(q.y); f[3] = bf16hi(q.y);
#elif DL == 2
  const uint q = *(device const uint*)p;
  f[0] = bf16lo(q); f[1] = bf16hi(q);
#else
  for (uint e = 0; e < DL; e++) f[e] = bf16f(p[e]);
#endif
}
static inline uint pack_bf16x2_bits(float lo, float hi) { return uint(bf16bits(lo)) | (uint(bf16bits(hi)) << 16); }
static inline void store_dl(device ushort* p, const thread float* f) {   // the slice, rounded to BF16, as one vector store
#if DL == 8
  *(device uint4*)p = uint4(pack_bf16x2_bits(f[0], f[1]), pack_bf16x2_bits(f[2], f[3]), pack_bf16x2_bits(f[4], f[5]), pack_bf16x2_bits(f[6], f[7]));
#elif DL == 4
  *(device uint2*)p = uint2(pack_bf16x2_bits(f[0], f[1]), pack_bf16x2_bits(f[2], f[3]));
#elif DL == 2
  *(device uint*)p = pack_bf16x2_bits(f[0], f[1]);
#else
  for (uint e = 0; e < DL; e++) p[e] = bf16bits(f[e]);
#endif
}
static inline void copy_dl(device ushort* dst, device const ushort* src) {   // a BF16 slice copied as it is
#if DL == 8
  *(device uint4*)dst = *(device const uint4*)src;
#elif DL == 4
  *(device uint2*)dst = *(device const uint2*)src;
#elif DL == 2
  *(device uint*)dst = *(device const uint*)src;
#else
  for (uint e = 0; e < DL; e++) dst[e] = src[e];
#endif
}
static inline void load_part(device const float* p, thread float* f) {   // a lane's DL FP32 partials
#if DL % 4 == 0
  for (uint e = 0; e < DL; e += 4) { const float4 q = *(device const float4*)(p + e); f[e] = q.x; f[e + 1] = q.y; f[e + 2] = q.z; f[e + 3] = q.w; }
#else
  for (uint e = 0; e < DL; e++) f[e] = p[e];
#endif
}
static inline void store_part(device float* p, const thread float* f) {
#if DL % 4 == 0
  for (uint e = 0; e < DL; e += 4) *(device float4*)(p + e) = float4(f[e], f[e + 1], f[e + 2], f[e + 3]);
#else
  for (uint e = 0; e < DL; e++) p[e] = f[e];
#endif
}
#define MERGE_UNROLL 4u              // chunks whose partials the merge requests before folding any (the fold order is kept)

// per-head RMSNorm (1 + w) and RoPE of one D-vector held DL-per-lane; the reference's rounding order
static inline void norm_rope(thread float* f, device const float* nw, device const ushort* cos_row, device const ushort* sin_row,
                             float eps, uint lane) {
#if QK_NORM
  float ss = 0.0f;
  for (uint e = 0; e < DL; e++) ss = fma(f[e], f[e], ss);
  ss = simd_sum(ss);
  const float rstd = rsqrt(ss / float(D) + eps);
  for (uint e = 0; e < DL; e++) f[e] = round_bf16(f[e] * rstd * nw[lane * DL + e]);
#endif
  const bool lo = lane < 16u;
  float r[DL];
  for (uint e = 0; e < DL; e++) {
    const float partner = simd_shuffle_xor(f[e], 16u);
    const float c = bf16f(cos_row[lane * DL + e]), s = bf16f(sin_row[lane * DL + e]);
    const float a = round_bf16(f[e] * c), b = round_bf16(partner * s);
    r[e] = round_bf16(lo ? a - b : a + b);
  }
  for (uint e = 0; e < DL; e++) f[e] = r[e];
}

// the chunk (keys per SIMD-group partial) of the v2 kernels for a context: the finest (32) unless larger chunks
// still give every threadgroup a (kv head, batch of NSG chunks) block — both kernels compute it from the same inputs
static inline uint chunk_of(uint ctx, uint kv, uint n_tg) {
  uint ch = 32u;
  if (kv * ((ctx + NSG * 64u - 1u) / (NSG * 64u)) >= n_tg) ch = 64u;
  if (kv * ((ctx + NSG * 128u - 1u) / (NSG * 128u)) >= n_tg) ch = 128u;
  return ch;
}
