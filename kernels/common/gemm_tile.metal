// gemm_tile: y[t][row] = sum_k dequant(W[row][k]) * x[t][k], tiled in TM-token blocks through the M5 neural accelerators
// (mpp::tensor_ops::matmul2d — design §5.6 / §5.12, plan M9, #50), reading the same block-lane-major pack as gemv_T.
//
// The Python side prepends `#include <metal_stdlib>`, the MPP header and a FORMAT SNIPPET (WEIGHTS_PER_WORD,
// SCALE_GROUP, decode_word, decode_scale[, decode_bias]) and sets the macros below. One SIMD-group multiplies one
// [TN rows x TK columns] weight tile (4096 weights: 16 x 256 up to 16 tokens, 32 x 128 at 32 — measured,
// docs/research/decode-kernels.md §6) at a time into a cooperative destination tensor; the right operand is a
// *cooperative* tensor filled straight from the pack words — no threadgroup staging, no barrier, no exchange:
//
//   * the accelerator's register layout (measured; tests/kernels/test_gemm_tile.py checks it against
//     get_multidimensional_index) gives thread `lane` the reduction slots 4*(bit0 + 2*bit3) + 16*jump + q for the rows
//     (bits 1,2,4) + 8*slot, i.e. a quarter of the tile's columns, in runs of 4, for TN/8 rows;
//   * the reduction index is free, so the tile's columns are permuted: quad member (bit0, bit3) owns TK/4 *consecutive*
//     pack columns of each of its rows — one contiguous half-word (4-bit formats at TK = 64), whole 16-byte words
//     otherwise — which it loads and decodes alone with the format snippet, every weight decoded once, one block
//     scale per 16-column chunk, and lands in its operand registers as BF16 (the reference's dequantized dtype);
//   * a tile's row piece is TK/WPW adjacent lanes' word j of the pack — adjacent 16-byte words in the interleaved
//     layout, a whole 128-byte cache line at TK = 256 (then lane group q runs outer, word j inner, and a thread's
//     block-scale words stay in registers across the lane group's words: Q_OUTER, SCALE_CACHE);
//   * the activation rows are read through a device tensor in the same order (x' from x_permute: pack order —
//     word j of lane l -> columns [j*32*WPW + l*WPW, +WPW) — then the slot order inside each tile), so the left
//     operand of a tile is one contiguous slice of x'.
//
// Geometry: the crew (n_sg SIMD-groups take static slices of the row tiles; S SIMD-groups per threadgroup share
// nothing; more SIMD-groups per core hide more latency), or with KSPLIT > 1 one row tile per threadgroup of KSPLIT
// SIMD-groups, each streaming a contiguous K / KSPLIT slice of the tile and the partial tiles reduced through
// threadgroup memory (the register layout is the same in every SIMD-group; SIMD-group 0 adds and runs the
// epilogue) — the remedy for the shapes whose row tiles cannot occupy the crew (256 tiles of a 4096-row projection
// against 480 SIMD-groups: decode-kernels.md §6). Macros: K, R (rows per pack block, divides TN), TM (token
// rows: 8 leaves half of the accelerator's 16-row minimum unused, so 16 costs the same), TN, TK, LANE_ORDER,
// UNIT_WORDS, PAYLOAD_WORDS, SCALE_W0, SCALE_UOFF, SCALE_WORDS (kernels.unit_geometry), Q_OUTER, SCALE_CACHE, OUT_BF16,
// plus the snippet's. Requires K % (32*WPW) == 0 and K % TK == 0; grid.y covers ceil(T_act / TM) (padding is zero in x'
// and are not written). Measured on the M5 Pro at 17408 x 5120: NVFP4 177 GB/s, FP8 253, INT4 204 at 8 or 16 tokens
// — 0.9-1.1x a T = 1 GEMV pass — against p14's staged tile (119 / 186); at 32 tokens the un-overlapped fill and
// the activation traffic leave it below the staged tile (a multi-SIMD-group staged variant is the follow-up).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef OUT_BF16
#define OUT_BF16 0
#endif
#ifndef STEP_STATE
#define STEP_STATE 0                 // 1: the row count comes from StepState (buffer 15) and the dispatch is a per-T variant
#endif
#ifndef T_SRC
#define T_SRC 0                      // with STEP_STATE: 0 = t_this_step, 1 = n_inject, 2 = the T_STATIC_ROWS macro, 3 = n_chain, 4 = n_inject + n_chain
#endif
#ifndef T_STATIC_ROWS
#define T_STATIC_ROWS 1u
#endif
#ifndef T_LO
#define T_LO 0                       // with T_HI: the variant runs only when T_LO < T_act <= T_HI (design §5.7)
#endif
#ifndef EPILOGUE
#define EPILOGUE 0                   // 1: y = bf16(v + residual) (EPILOGUE_ROUND: v rounded to BF16 first); 2: silu(gate)·up over
#endif                               //    chunk-interleaved rows (block b = gate rows [0, CHUNK) | up rows [CHUNK, R)), N/2 outputs
#ifndef EPILOGUE_ROUND
#define EPILOGUE_ROUND 0
#endif
#ifndef STAT_OUT
#define STAT_OUT 0                   // 1: per-block partial sums of squares of the BF16-rounded outputs, stat_out[t * n_blocks + block]
#endif
#ifndef POST_NORM
#define POST_NORM 0                  // x already contains gamma*h; apply the RMS scalar after the product
#endif
#ifdef ROW_SCALE_BITS
#define GEMM_ROW_SCALE(i) as_type<float>(uint(ROW_SCALE_BITS))
#else
#define GEMM_ROW_SCALE(i) row_scale[(i)]
#endif
#if R != 16 && R != 8
#error "gemm_tile: the epilogues index pack blocks of 8 or 16 rows"
#endif
#define CHUNK (R / 2u)
#define PAIR_XOR ((R == 16u) ? 8u : 1u)                    // the lane holding the partner row of a silu_mul pair
#ifndef Q_OUTER
#define Q_OUTER 0                    // 1: lane group outer, word inner (a tile's row piece is a whole cache line)
#endif
#ifndef SCALE_CACHE
#define SCALE_CACHE 0                // 1: keep the scale words of a thread's rows in registers across a lane group's words (needs Q_OUTER)
#endif
#if SCALE_CACHE && !Q_OUTER
#error "gemm_tile: SCALE_CACHE needs Q_OUTER"
#endif
#ifndef EXP_MODE
#define EXP_MODE 0                   // bench experiments (tools/bench/gemm_bench.py --macro EXP_MODE=n): 2 = the matmul with one
                                     // fill (no loads, no decode); 5 = the fill from synthetic words (no loads); 6 = the
                                     // activation slice pinned to the first tile (A traffic served from L1)
#endif
#ifndef SCALE_BIAS
#define SCALE_BIAS 0
#endif
#ifndef SCALE_W0
#define SCALE_W0 PAYLOAD_WORDS
#define SCALE_UOFF 0u
#endif
#ifndef KSPLIT
#define KSPLIT 1u                                         // SIMD-groups per row tile, each a contiguous K slice (1 … 16)
#endif
#ifndef TN
#define TN 64u                                            // rows per tile (16, 32 or 64)
#endif
#ifndef TK
#define TK 64u                                            // columns per tile (64, 128 or 256): TN * TK = 4096
#endif
#define KL (K / 32u)
#define WPW WEIGHTS_PER_WORD
#define LPT (TK / WPW)                                    // lanes (pack words) per K tile
#define KT (K / TK)                                       // K tiles per row
#define NB (TN / R)                                       // pack blocks per row tile
#define CT (TK / 4u)                                      // a thread's consecutive columns per row (its quad's quarter)
#define NW ((CT >= WPW) ? (CT / WPW) : 1u)                // words holding them (one half-word when CT < WPW)
#define NCH (CT / 16u)                                    // 16-column chunks of them (one block scale each)
#if SCALE_GROUP > 0 && (SCALE_GROUP % 16) != 0
#error "gemm_tile: the block scale group must be a multiple of 16 columns"
#endif
#define NS_B (TN / 8u)                                    // row slots per thread in the right operand
#define NB_C ((TM > 16u) ? (TM / 16u) : 1u)               // 16-row blocks of the destination (its element order is
                                                          // q, slot (2), jump (TN/16), block — the right operand's is q, slot (8), jump)
#define C_CAP (NB_C * TN / 2u)                            // destination elements per thread (16 · NB_C rows × TN over 32 lanes)
// Opt-in after whole-layer measurement: compacting scratch can also change
// register allocation. The hardware accumulator covers at least 16 token rows.
#ifndef COMPACT_PARTIALS
#define COMPACT_PARTIALS 0
#endif
#ifdef T_HI
#define PART_TOKENS T_HI
#else
#define PART_TOKENS TM
#endif
#if COMPACT_PARTIALS && PART_TOKENS <= 8 && TM <= 16
#define PART_CAP (C_CAP / 2u)
#define PART_INDEX(i) (((i) / 4u) * 8u + (i) % 4u)
#else
#define PART_CAP C_CAP
#define PART_INDEX(i) (i)
#endif
#if COMPACT_PARTIALS && PART_TOKENS <= 4
#define PART_LANES 16u
#else
#define PART_LANES 32u
#endif
#define KT_S (KT / KSPLIT)                                // K tiles per slice
#if (KT % KSPLIT) != 0
#error "gemm_tile: KSPLIT must divide the K tiles"
#endif
#if SCALE_CACHE && (KT_S % PAYLOAD_WORDS) != 0
#error "gemm_tile: with the scale cache a K slice must be whole lane groups (KSPLIT divides 32 / LPT)"
#endif
#if SCALE_GROUP > 0
#if (KL % SCALE_GROUP) == 0
#define LANE_OFF 0u
#else
#define LANE_OFF ((ln * KL) % SCALE_GROUP)
#endif
#endif

struct GemmParams { uint n_rows; uint n_tiles; uint n_sg; uint t_active; float out_scale; uint tile0; uint n_blocks; uint pad; };
// n_rows / n_tiles / n_blocks / tile0 describe a row range of the slab (a whole slab: N / all tiles / all blocks / 0);
// outputs and stat partials are range-relative, the weights and row scales are addressed by the slab tile.

static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline float silu_f(float g) { return g / (1.0f + exp(-g)); }

static inline uint unit_word(uint lane, uint r, uint j) {
#if LANE_ORDER == 0
  return (lane * R + r) * UNIT_WORDS + j;
#else
  return (r * UNIT_WORDS + j) * 32u + lane;
#endif
}
#ifdef LANES_PER_WORD
#error "sub-word units are the shader GEMV's: the tile and the gather read whole-word units"
#endif
#ifndef SCALE_LANE_DIVISOR
#define SCALE_LANE_DIVISOR 1u
#endif
#ifndef SCALE_PLACEMENT
#define SCALE_PLACEMENT 0            // 1: the block's scales in their own region after its payload words (blm.py, #101):
#endif                               //    lane ln's row r scales start (ln * SCALE_RUN) % 16 bytes into word SCALE_WORD(ln, r, 0)
#if SCALE_PLACEMENT
#define SCALE_BASE (R * 32u * PAYLOAD_WORDS)
#define SCALE_WORD(ln, r, s) (SCALE_BASE + ((r) * (32u / SCALE_LANE_DIVISOR) * SCALE_RUN + ((ln) / SCALE_LANE_DIVISOR) * SCALE_RUN) / 16u + (s))
#define SCALE_SOFF(ln) (((((ln) / SCALE_LANE_DIVISOR) * SCALE_RUN) % 16u) / SCALE_UNIT_BYTES)
#else
#define SCALE_WORD(ln, r, s) unit_word((ln), (r), SCALE_W0 + (s))
#define SCALE_SOFF(ln) 0u
#endif
// Load a short, uint-aligned scale run directly instead of a uint4 followed
// by lane-dependent extraction (affine pairs or byte-sized FP4 scales).
#if SCALE_PLACEMENT && SCALE_RUN <= 8 && ((SCALE_BIAS && SCALE_UNIT_BYTES == 4) || (SCALE_UNIT_BYTES == 1 && SCALE_RUN % 4 == 0))
#define NARROW_SCALE_RUN 1
#define SCALE_REG_OFFSET(ln) 0u
static inline uint narrow_scale_word(device const uint4* wb, uint ln, uint r, uint g) {
  return reinterpret_cast<device const uint*>(wb + SCALE_BASE)[
      (r * (32u / SCALE_LANE_DIVISOR) + ln / SCALE_LANE_DIVISOR) * (SCALE_RUN / 4u) + g];
}
#else
#define NARROW_SCALE_RUN 0
#define SCALE_REG_OFFSET(ln) SCALE_SOFF(ln)
#endif
#if SCALE_PLACEMENT
#define BLOCK_WORDS (R * 32u * UNIT_WORDS + SCALE_REGION_WORDS)       // a block: its payload words then its scale region
#else
#define BLOCK_WORDS (R * 32u * UNIT_WORDS)
#endif

constexpr constant auto desc = matmul2d_descriptor(int(TM), int(TN), int(TK), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
using tA_t = tensor<device bfloat, dextents<int, 2>, tensor_inline>;

kernel void gemm_tile(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
                      device bfloat* xp [[buffer(2)]],
#if OUT_BF16
                      device ushort* y [[buffer(3)]],
#else
                      device float* y [[buffer(3)]],
#endif
                      constant GemmParams& p [[buffer(4)]],
#if POST_NORM
                      device const float* norm_stat [[buffer(5)]],
#endif
#if NORM_OUT
                      device const float* norm_weight [[buffer(13)]], device ushort* norm_x [[buffer(14)]],
#endif
#if EPILOGUE == 1
                      device const ushort* residual [[buffer(7)]],
#endif
#if STAT_OUT
                      device float* stat_out [[buffer(8)]],
#endif
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint3 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]],
                      uint3 group [[threadgroup_position_in_grid]]) {
  const uint sg = gid.x / sw;
#if KSPLIT > 1
  const uint slice = sg % KSPLIT, sg_tile = sg / KSPLIT, n_tg = p.n_sg / KSPLIT;   // a threadgroup is the KSPLIT SIMD-groups of one tile
  threadgroup float part[KSPLIT - 1][PART_LANES][PART_CAP];                            // the partial tiles of slices 1 … KSPLIT-1
#else
  const uint sg_tile = sg, n_tg = p.n_sg;
#endif
#if STEP_STATE
  if (st->done) return;                                                     // uniform over the threadgroup: no barrier is skipped
  uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step)));
  if (T_act == 0u) return;                                                  // no rows this step (an LM drafter's chain in a prefill chunk)
#ifdef T_HI
  if (T_act > T_HI || T_act <= T_LO) return;
#endif
#else
  uint T_act = p.t_active;
#endif
  // Each grid.y plane processes at most TM tokens. Predication above uses the
  // full step length; all following addressing and guards are local to this tile.
  const uint token0 = group.y * TM;
  if (token0 >= T_act) return;
  T_act = min(T_act - token0, uint(TM));
  xp += (ulong)token0 * K;
  const uint output_cols = (EPILOGUE == 2) ? p.n_rows / 2u : p.n_rows;
  y += (ulong)token0 * output_cols;
#if EPILOGUE == 1
  residual += (ulong)token0 * output_cols;
#endif
#if STAT_OUT
  stat_out += (ulong)token0 * p.n_blocks;
#endif
  matmul2d<desc, execution_simdgroup> op;
  tA_t tA(xp, dextents<int, 2>(int(K), int(TM)));
  const uint c0b = 4u * ((lane & 1u) + 2u * ((lane >> 3) & 1u));      // this thread's column-run base
  const uint c1b = ((lane >> 1) & 3u) + 4u * ((lane >> 4) & 1u);      // this thread's row-slot base
  const uint mq = (lane & 1u) | (((lane >> 3) & 1u) << 1);            // its member id in the quad sharing those rows
#if POST_NORM
  float norm_r = 0.f;
#endif
  for (uint tile = p.tile0 + sg_tile; tile < p.tile0 + p.n_tiles; tile += n_tg) {
    auto bT = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cT = op.get_destination_cooperative_tensor<tA_t, decltype(bT), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++) cT[i] = 0.0f;
#if SCALE_CACHE
    uint scc[NS_B][NW][SCALE_WORDS * 4];                                // the scale words of this thread's rows and lanes,
#endif                                                                  // kept across the PAYLOAD_WORDS tiles of a lane group
#if KSPLIT > 1
    for (uint kt = slice * KT_S; kt < (slice + 1u) * KT_S; kt++) {      // this SIMD-group's K slice (whole lane groups under Q_OUTER)
#else
    for (uint kt = 0; kt < KT; kt++) {
#endif
      // a tile's row piece is LPT adjacent lanes' word j, LPT*16 bytes of a 128-byte line: when that is the whole line
      // (Q_OUTER) lane group q runs outer and word j inner, so the scale words can be kept across j; otherwise j runs
      // outer so the lane groups sharing a line are consecutive tiles
#if Q_OUTER
      const uint q = kt / PAYLOAD_WORDS, j = kt % PAYLOAD_WORDS;        // pack word j of lanes [q*LPT, q*LPT + LPT)
#else
      const uint j = kt / (32u / LPT), q = kt % (32u / LPT);
#endif
#if EXP_MODE == 2
      if (kt == 0) {
#endif
      // the reduction index is permuted so that this thread's 16 slots of each row are 16 *consecutive* pack columns,
      // [16*mq, 16*mq + 16) of the tile (x_permute orders x' the same way): one contiguous half-word (4-bit formats),
      // a whole word (8-bit) or two words (bf16) per row, loaded by this thread alone — no exchange, no redundancy
      // beyond the other half of a 4-bit word, one block scale per row
#pragma clang loop unroll(full)
      for (uint s = 0; s < NS_B; s++) {
        const uint n = c1b + 8u * s;                                     // row inside the tile
        // A final TN-row tile can extend beyond the last packed block. Those
        // outputs are masked below, but their cooperative loads must stay valid.
        const uint last_block = (p.tile0 * TN + p.n_rows - 1u) / R;
        const uint b = min(tile * NB + n / R, last_block), r = n % R;
        device const uint4* wb = w + (ulong)b * BLOCK_WORDS;
        const uint c0 = CT * mq;                                         // this thread's first tile column
        const uint lw0 = c0 / WPW, e0 = c0 % WPW;                        // its first word inside the tile and code offset
        const uint ln0 = q * LPT + lw0;                                  // the lane holding that word
        uint4 words[NW];
#pragma clang loop unroll(full)
        for (uint i = 0; i < NW; i++) {
#if EXP_MODE == 5
          words[i] = uint4(lane * 2654435761u + kt * 40503u + i, n * 97u + kt, i * 7u + lane, kt);
#else
          words[i] = wb[unit_word(ln0 + i, r, j)];
#endif
        }
#if CT < WPW
        words[0] = (e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0];   // the half this thread owns
#endif
        float wv[NW * WPW];
#pragma clang loop unroll(full)
        for (uint i = 0; i < NW; i++) decode_word(words[i], wv + i * WPW);
#if SCALE_GROUP > 0
        float scv[NCH];                                                  // one block scale per 16-column chunk
#if SCALE_BIAS
        float bv[NCH];
#endif
#pragma clang loop unroll(full)
        for (uint i = 0; i < NW; i++) {
#if SCALE_PAYLOAD_ORDER
          const uint off = r * 32u * SCALE_RUN + j * 64u + (ln0 + i) * 2u;
          uint sw = reinterpret_cast<device const ushort*>(wb + SCALE_BASE)[off / 2u];
#pragma clang loop unroll(full)
          for (uint ch = 0; ch < ((CT < WPW) ? 1u : (WPW / 16u)); ch++)
            scv[i * (WPW / 16u) + ch] = decode_scale(&sw, e0 / 16u + ch);
#else
#if NARROW_SCALE_RUN
          uint scw[SCALE_RUN / 4u];
#pragma clang loop unroll(full)
          for (uint sc = 0; sc < SCALE_RUN / 4u; sc++)
            scw[sc] = narrow_scale_word(wb, ln0 + i, r, sc);
#else
#if SCALE_CACHE
          if (j == 0u) {
#pragma clang loop unroll(full)
            for (uint sc = 0; sc < SCALE_WORDS; sc++) {
#if EXP_MODE == 5
              const uint4 v4 = uint4(0x38383838u + lane, 0x38383838u, 0x38383838u + kt, 0x38383838u);
#else
              const uint4 v4 = wb[SCALE_WORD(ln0 + i, r, sc)];
#endif
              scc[s][i][4 * sc] = v4.x; scc[s][i][4 * sc + 1] = v4.y; scc[s][i][4 * sc + 2] = v4.z; scc[s][i][4 * sc + 3] = v4.w;
            }
          }
          thread const uint* scw = scc[s][i];
#else
          uint scw[SCALE_WORDS * 4];
#pragma clang loop unroll(full)
          for (uint sc = 0; sc < SCALE_WORDS; sc++) {
#if EXP_MODE == 5
            const uint4 v4 = uint4(0x38383838u + lane, 0x38383838u, 0x38383838u + kt, 0x38383838u);
#else
            const uint4 v4 = wb[SCALE_WORD(ln0 + i, r, sc)];
#endif
            scw[4 * sc] = v4.x; scw[4 * sc + 1] = v4.y; scw[4 * sc + 2] = v4.z; scw[4 * sc + 3] = v4.w;
          }
#endif
#endif
          const uint ln = ln0 + i;                                       // (LANE_OFF is a function of ln)
#pragma clang loop unroll(full)
          for (uint ch = 0; ch < ((CT < WPW) ? 1u : (WPW / 16u)); ch++) {
            const uint g = (LANE_OFF + j * WPW + e0 + 16u * ch) / SCALE_GROUP;
            scv[i * (WPW / 16u) + ch] = decode_scale(scw + SCALE_UOFF, SCALE_REG_OFFSET(ln) + g);
#if SCALE_BIAS
            bv[i * (WPW / 16u) + ch] = decode_bias(scw + SCALE_UOFF, SCALE_REG_OFFSET(ln) + g);
#endif
          }
#endif // SCALE_PAYLOAD_ORDER
        }
#endif
#pragma clang loop unroll(full)
        for (uint jump = 0; jump < TK / 16u; jump++)
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) {
            float v = wv[4u * jump + qq];
#if SCALE_GROUP > 0
            v *= scv[jump / 4u];
#if SCALE_BIAS
            v += bv[jump / 4u];
#endif
#endif
#if BF16_STORAGE
            // Preserve the stored BF16 bits instead of expanding each element
            // to FP32 and rounding it back for the cooperative matrix operand.
            const uint e = 4u * jump + qq;
            bT[uint16_t(((jump * NS_B + s) << 2) | qq)] = as_type<bfloat2>(words[e / 8u][(e % 8u) / 2u])[e % 2u];
#else
            bT[uint16_t(((jump * NS_B + s) << 2) | qq)] = bfloat(v);
#endif
          }
      }
#if EXP_MODE == 2
      }
#endif
      // the activation slice of this tile: x' is in pack order (word j major, lane minor), whatever the loop order
      const uint kp = j * (32u / LPT) + q;
#if EXP_MODE == 6
      auto sA = tA.slice<int(TK), int(TM)>(0, 0);                         // a pinned slice: the A traffic without its bytes
#else
      auto sA = tA.slice<int(TK), int(TM)>(int(kp * TK), 0);
#endif
#if EXP_MODE != 1
      op.run(sA, bT, cT);
#else
      if (kt == KT - 1u) op.run(sA, bT, cT);                          // fill-only timing: one run so the fill is not dead
#endif
    }
#if KSPLIT > 1
    // the slices' partial tiles meet in threadgroup memory: slices 1 … KSPLIT-1 write theirs, slice 0 adds them into
    // its own and runs the epilogue alone. Compact scratch omits padded token rows;
    // PART_INDEX maps its slots back to the hardware accumulator. Only a subsequent
    // tile needs the second barrier to keep its writes behind this tile's reads.
    if (slice != 0u && lane < PART_LANES) {
#pragma clang loop unroll(full)
      for (uint16_t i = 0; i < PART_CAP; i++) part[slice - 1u][lane][i] = cT[PART_INDEX(i)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice == 0u && lane < PART_LANES) {
#pragma clang loop unroll(full)
      for (uint s2 = 1; s2 < KSPLIT; s2++)
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < PART_CAP; i++) cT[PART_INDEX(i)] += part[s2 - 1u][lane][i];
    }
#if COMPACT_PARTIALS
    if (tile + n_tg < p.tile0 + p.n_tiles) threadgroup_barrier(mem_flags::mem_threadgroup);
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    if (slice != 0u) continue;
#endif
#if POST_NORM
#if TM > 8
#error POST_NORM requires a short tile
#endif
#if POST_NORM_ONCE
    // A persistent crew reuses the same input across its output tiles.
    // Fold after the first MMA so the first tile keeps its short live range.
    if (tile == p.tile0 + sg_tile) {
#endif
    // Adjacent quads load one token's partials; shuffle its reciprocal RMS
    // into the accumulator's token layout. Keep this after MMA to shorten
    // register lifetimes and skip the non-writing K slices.
    const uint norm_row = token0 + min(lane / 4u, T_act - 1u), q = lane % 4u;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (uint b = q; b < POST_NORM_PARTS; b += 64u) {
      float v[16];
      for (uint u = 0; u < 16; u++) v[u] = b + 4u * u < POST_NORM_PARTS ? norm_stat[norm_row * POST_NORM_PARTS + b + 4u * u] : 0.f;
      for (uint u = 0; u < 16; u += 4) { s0 += v[u]; s1 += v[u+1]; s2 += v[u+2]; s3 += v[u+3]; }
    }
    float ssq = (s0 + s1) + (s2 + s3);
    ssq += simd_shuffle_xor(ssq, ushort(1));
    ssq += simd_shuffle_xor(ssq, ushort(2));
    norm_r = simd_shuffle(rsqrt(ssq / float(K) + POST_NORM_EPS), ushort(4u * c1b));
#if POST_NORM_ONCE
    }
#endif
#endif
    // epilogue over the destination: element ((blk*(TN/16) + jump)*2 + s2) << 2 | q holds row n = c0b + 16*jump + q
    // and token m = 16*blk + c1b + 8*s2; a lane's 4 rows lie in one pack block, the block's other rows in the lanes
    // differing in bit 0 (and bit 3 when R = 16); rows beyond t_active and the range are not written
#pragma clang loop unroll(full)
    for (uint jump = 0; jump < TN / 16u; jump++) {
      const uint n = c0b + 16u * jump;                                   // this lane's 4 consecutive rows of the tile
      const uint row = tile * TN + n;                                    // the slab row (weights, row scales)
      const uint rrow = (tile - p.tile0) * TN + n;                       // the range-relative row (outputs)
      const uint bb = (tile - p.tile0) * (TN / R) + (n / R);             // the range-relative pack block
      float rs[4];
#pragma clang loop unroll(full)
      for (uint qq = 0; qq < 4u; qq++) rs[qq] = (rrow + qq < p.n_rows) ? GEMM_ROW_SCALE(min(row + qq, p.tile0 * TN + p.n_rows - 1u)) * p.out_scale : 0.0f;   // in bounds even when hoisted
#pragma clang loop unroll(full)
      for (uint blk = 0; blk < NB_C; blk++)
#pragma clang loop unroll(full)
        for (uint s2 = 0; s2 < 2u; s2++) {
          const uint m = 16u * blk + c1b + 8u * s2;
          float v[4];
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) v[qq] = cT[uint16_t((((blk * (TN / 16u) + jump) * 2u + s2) << 2) | qq)] * rs[qq];
#if POST_NORM
          for (uint qq = 0; qq < 4u; qq++) v[qq] *= norm_r;
#endif
#if EPILOGUE == 2
          float pv[4];                                                   // the partner rows: up for a gate lane, gate for an up lane
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) pv[qq] = simd_shuffle_xor(v[qq], ushort(PAIR_XOR));
          const bool gate_lane = (lane & PAIR_XOR) == 0u;               // rows [0, CHUNK) of the block
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) v[qq] = silu_mul(v[qq], pv[qq]);
          const uint orow0 = bb * CHUNK + (n % R), n_out = p.n_rows / 2u;
          const bool writer = gate_lane && m < T_act;
#else
          const uint orow0 = rrow, n_out = p.n_rows;
          const bool writer = m < T_act;
#endif
          // the rows m >= T_act of the 16-row destination are never written, but a load behind `continue` may still be
          // issued ahead of the branch (the compiler hoists a side-effect-free load): every address below stays inside
          // its binding whether or not the lane writes — shader validation caught the residual read at rows 8–15 of
          // an 8-row value (#113), a fault waiting for an unmapped neighbour
          const uint m_in = min(m, T_act - 1u);
          float ssq = 0.0f;
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) {
            if (!writer || rrow + qq >= p.n_rows) continue;
            float vv = v[qq];
#if EPILOGUE == 1
#if EPILOGUE_ROUND
            vv = round_bf16(vv);
#endif
            vv += as_type<float>(uint(residual[(ulong)m_in * n_out + min(orow0 + qq, n_out - 1u)]) << 16);
#endif
            const float vr = round_bf16(vv);
            ssq = fma(vr, vr, ssq);
#if NORM_OUT
            norm_x[(ulong)(token0 + m) * NORM_K + norm_dest(orow0 + qq)] =
                ushort(as_type<uint>(round_bf16(vr * norm_weight[orow0 + qq])) >> 16);
#endif
#if PERM_OUT
            y[(ulong)m * PERM_K + perm_dest(orow0 + qq)] = ushort(as_type<uint>(vr) >> 16);   // the consumer tile's x' (its K = n_out)
#elif OUT_BF16
            y[(ulong)m * n_out + orow0 + qq] = ushort(as_type<uint>(vr) >> 16);
#else
            y[(ulong)m * n_out + orow0 + qq] = vv;
#endif
          }
#if STAT_OUT
          // the block's sum over its rows: the lanes differing in bit 0 (and bit 3 when a block is 16 rows)
          ssq += simd_shuffle_xor(ssq, ushort(1));
#if R == 16
          ssq += simd_shuffle_xor(ssq, ushort(8));
#endif
          if (m < T_act && (lane & ((R == 16u) ? 9u : 1u)) == 0u && rrow < p.n_rows) stat_out[m * p.n_blocks + bb] = ssq;
#endif
        }
    }
  }
}

// x_permute: x [T, K] BF16 (rows t >= T_act ignored) -> x' [TM, K] in gemm_tile's reduction order, zero beyond
// T_act; with PERM_NORM the RMSNorm scaling is applied on the way (x'[t][i] = bf16(h[t][c(i)] · r[t] · norm_w[c(i)]),
// r[t] = rsqrt(Σ stat[t·stat_parts ..] / K + eps) — the normalized activation the reference's norm produces).
// Pack order first (word j of lane l -> columns [j*32*WPW + l*WPW, +WPW)), then inside each TK-column tile the
// accelerator's slot order: slot 4*(bit0 + 2*bit3) + 16*jump + q of quad member (bit0, bit3) holds tile column
// (TK/4)*(bit0 + 2*bit3) + 4*jump + q, so a thread's TK/4 slots are TK/4 consecutive pack columns. PERM_SG
// SIMD-groups per output row, each a K/PERM_SG slice, the strides compile-time (K, TK, WPW) and the gathers issued
// PERM_UNROLL at a time — a step runs one of these per GEMV input, so its latency is the whole cost.
// With STEP_STATE the same per-T predicate as the tile it feeds.
#ifndef PERM_NORM
#define PERM_NORM 0
#endif
#ifndef PERM_SG
#define PERM_SG 4u
#endif
#ifndef PERM_UNROLL
#define PERM_UNROLL 4u
#endif
#ifndef PERM_FOLD_LOADS
#define PERM_FOLD_LOADS 16u          // the norm fold's loads requested per round (divides by 4)
#endif
struct XPermParams { uint k; uint t_active; uint tm; uint wpw; uint tk; uint stat_parts; float eps; uint pad; };

static inline uint perm_source(uint i) {                                  // the original column of slot i
  const uint kt = i / TK, slot = i % TK;
  const uint mq = ((slot >> 2) & 1u) + 2u * ((slot >> 3) & 1u), jump = slot >> 4, qq = slot & 3u;
  const uint c = kt * TK + (TK / 4u) * mq + 4u * jump + qq;               // the pack-order column
  const uint span = 32u * WPW;
  const uint j = c / span, rem = c % span, l = rem / WPW, e = rem % WPW;
  return l * KL + j * WPW + e;
}

#ifndef PERM_GROUPS
#define PERM_GROUPS 1u
#endif
kernel void x_permute(device const ushort* x [[buffer(0)]],
#if PERM_NORM
                      device const float* stat [[buffer(1)]], device const float* norm_w [[buffer(2)]],
#endif
                      device ushort* xp [[buffer(3)]], constant XPermParams& p [[buffer(4)]],
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / PERM_SG, slice = sg % PERM_SG;
  if (t >= p.tm) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step)));
  if (T_act == 0u) return;                                                  // no rows this step (an LM drafter's chain in a prefill chunk)
#ifdef T_HI
  if (T_act > T_HI || T_act <= T_LO) return;
#endif
#else
  const uint T_act = p.t_active;
#endif
  const uint k0 = slice * (K / PERM_SG), k1 = k0 + K / PERM_SG;          // this SIMD-group's slice of the row
  device ushort* out = xp + (ulong)t * K;
  if (t >= T_act) {
    for (uint i = k0 + lane; i < k1; i += 32u) out[i] = 0;
    return;
  }
#if PERM_NORM
#if PERM_GROUPS > 1
  threadgroup float shared_r;
  if (slice % PERM_GROUPS == 0u) {
#endif
  // the statistic's partials folded FOLD_LOADS loads per round (as gemv_T's fold: a row-split producer leaves up to 2048
  // per token, one load latency per 32 of them was the dispatch's critical path), four accumulators in a fixed order
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  device const float* sp = stat + t * p.stat_parts;
  for (uint base = lane; base < p.stat_parts; base += 32u * PERM_FOLD_LOADS) {
    float v[PERM_FOLD_LOADS];
    for (uint u = 0; u < PERM_FOLD_LOADS; u++) { const uint i = base + 32u * u; v[u] = (i < p.stat_parts) ? sp[i] : 0.0f; }
    for (uint u = 0; u < PERM_FOLD_LOADS; u += 4u) { s0 += v[u]; s1 += v[u + 1u]; s2 += v[u + 2u]; s3 += v[u + 3u]; }
  }
  const float ssq = simd_sum((s0 + s1) + (s2 + s3));
  const float r = rsqrt(ssq / float(K) + p.eps);
#if PERM_GROUPS > 1
  if (lane == 0u) shared_r = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = shared_r;
#endif
#endif
  device const ushort* row = x + (ulong)t * K;
  for (uint i = k0 + lane; i < k1; i += 32u * PERM_UNROLL) {
    uint src[PERM_UNROLL];
    ushort v[PERM_UNROLL];
#pragma clang loop unroll(full)
    for (uint u = 0; u < PERM_UNROLL; u++) {                                // independent gathers (a slice may be shorter than the unroll)
      src[u] = (i + 32u * u < k1) ? perm_source(i + 32u * u) : 0u;
      v[u] = (i + 32u * u < k1) ? row[src[u]] : ushort(0);
    }
#pragma clang loop unroll(full)
    for (uint u = 0; u < PERM_UNROLL; u++) {
      if (i + 32u * u >= k1) continue;
#if PERM_NORM
      const float f = round_bf16(norm_scale(as_type<float>(uint(v[u]) << 16), r, norm_w[src[u]]));
      v[u] = ushort(as_type<uint>(f) >> 16);
#endif
      out[i + 32u * u] = v[u];
    }
  }
}

// coop_layout: the register layout the fill assumes, read back from the API for the test (one SIMD-group).
kernel void coop_layout(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  matmul2d<desc, execution_simdgroup> op;
  auto bT = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cT = op.get_destination_cooperative_tensor<tA_t, decltype(bT), float>();
  device int* o = out + lane * (4 + 3 * 1024);                       // [cap_b, cap_c, -, -, then (valid, c0, c1) per element]
  o[0] = int(bT.get_capacity()); o[1] = int(cT.get_capacity());
  int q = 4;
  for (uint16_t i = 0; i < bT.get_capacity() && i < 512; i++) { auto c = bT.get_multidimensional_index(i); o[q++] = bT.is_valid_element(i); o[q++] = c[0]; o[q++] = c[1]; }
  for (uint16_t i = 0; i < cT.get_capacity() && i < 512; i++) { auto c = cT.get_multidimensional_index(i); o[q++] = cT.is_valid_element(i); o[q++] = c[0]; o[q++] = c[1]; }
}
