// gemv_T: y[t][row] = sum_k dequant(W[row][k]) * x[t][k] over a block-lane-major pack (design D8, §5.6).
//
// The Python side prepends `#include <metal_stdlib>` and a FORMAT SNIPPET (monolith.formats.<fmt>.msl_decode) that
// defines WEIGHTS_PER_WORD, SCALE_GROUP, decode_word(uint4, thread float*) and decode_scale(thread const uint*, uint),
// then sets the macros below. One kernel = one op of the step program: `n_sg` SIMD-groups take static slices of the
// `n_blocks` row blocks (block b -> rows [b*R, b*R+R)); the 32 lanes own 32 column stripes of K/32 columns; every
// weight load is one 16-byte word of the lane-row unit; activations are BF16 and are re-read per RG-row group from
// cached device memory; accumulation is FP32; one simd_sum per (row, token); the per-row scale (the source matrix's
// per-tensor scale, from the pack's row-scale table) is applied once per output.
//
// Macros: K (columns), R (rows per block), T (tokens), RG (rows per activation reuse group, divides R; 1 with X_HOIST),
//         LANE_ORDER (0 contiguous, 1 interleaved16), UNIT_WORDS (16-byte words per lane-row unit),
//         PAYLOAD_WORDS (weight words per lane-row; the last one is partial — K_TAIL columns — when K/32 is not
//         whole words, a ragged stripe), SCALE_W0 / SCALE_UOFF / SCALE_WORDS (the row's scale bytes start SCALE_UOFF
//         uints into word SCALE_W0, the tail of a partial payload word, and span SCALE_WORDS words; 0 = none),
//         GROUP_SEG (columns per in-word scale segment: gcd(WPW, K/32, SCALE_GROUP), so a segment never straddles a
//         scale group even when a lane's stripe starts inside one — LANE_OFF, the stripe's offset in its first group),
//         OUT_BF16 (1: write bf16 outputs, 0: float), X_PRECONVERT (1: convert the activation chunk to float once
//         per word and reuse it across the row group; 0: keep it as bf16 words and convert per row — for large T*WPW)
//
// Fusions (design §5.1, §5.6), each a macro so a pipeline variant is selected by (format, T, fusions):
//   NORM=1      the input is the raw residual stream h: x[t][k] = bf16(h[t][k] · r[t] · norm_w[k]) with
//               r[t] = rsqrt(Σ stat[t·stat_parts .. +stat_parts) / K + eps) — the RMSNorm scaling applied on load,
//               its statistic read from a stat buffer (1 part from rmsnorm_stat, or the producing op's per-block
//               partial sums of squares, stat_parts = its n_blocks). Rounded to BF16 like the reference's norm output.
//   EPILOGUE=1  residual add before the single rounding: y = bf16(acc·scale + residual[t][row]).
//   EPILOGUE=2  silu·mul over chunk-interleaved rows: block b holds gate rows [b·R, b·R+CHUNK) and up rows
//               [b·R+CHUNK, b·R+R) (pack_weights' interleave_chunks with chunk = CHUNK = R/2); output
//               y[t][b·CHUNK + i] = bf16(silu(gate_i) · up_i) over n_rows/2 outputs; RG must divide CHUNK.
//   STAT_OUT=1  the epilogue also writes per-block partial sums of squares of the BF16-rounded outputs,
//               stat_out[t·n_blocks + b], for the next op's NORM (the hoisted norm statistic).
#ifndef NORM
#define NORM 0
#endif
#ifndef EPILOGUE_ROUND
#define EPILOGUE_ROUND 0             // with EPILOGUE=1: round the product to BF16 before the residual add (two roundings, the
#endif                               // reference's separate linear + add; the fused layers keep the single rounding)
#ifndef LOCAL_GATE_CACHE
#define LOCAL_GATE_CACHE 0
#endif
#ifndef EPILOGUE
#define EPILOGUE 0
#endif
#ifndef STAT_OUT
#define STAT_OUT 0
#endif
#ifndef CHUNK
#define CHUNK (R / 2u)
#endif
#ifndef X_HOIST
#define X_HOIST 0                    // 1: the activation words are converted once per SIMD-group ahead of the items (X_PRECONVERT, small K, T = 1)
#endif
#ifndef FOLD_LOADS
#define FOLD_LOADS 16u              // the norm fold's loads requested per round (divides by 4)
#endif
#ifndef RG
#define RG 4
#endif
#ifndef OUT_BF16
#define OUT_BF16 0
#endif
#ifndef X_PRECONVERT
#define X_PRECONVERT 1
#endif
#ifndef T_STATIC
#define T_STATIC 0
#endif
#ifndef STEP_STATE
#define STEP_STATE 0                 // 1: T comes from the bound StepState (buffer 15) and the kernel returns at once after `done`
#endif
#ifndef T_SRC
#define T_SRC 0                      // with STEP_STATE: 0 = t_this_step (the target's T), 1 = n_inject (the drafter's context rows), 2 = the T macro (a draft block)
#endif
#ifndef T_LO
#define T_LO 0                       // with T_HI: a predicated per-T variant (design §5.7) — it runs only when T_LO < T_act <= T_HI and
#endif                               // returns at once otherwise, so a step's ALU work follows its actual T, not the program's T_max
#ifndef SCALE_BIAS
#define SCALE_BIAS 0                 // a format with a per-group bias (affine INT4: w = scale·code + bias): acc += bias · Σ x over the group
#endif
#ifndef PAIRS
#define PAIRS 0                      // 1: the MoE expert mode (design §5.11) — T = 1 per item; the work items are (token, slot, block) over
#endif                               // T_act tokens × K_TOPK slots × EXPERT_BLOCKS blocks of the slot's expert (ids[t·K_TOPK + slot], buffer 9):
                                     // the item streams slab block e·EXPERT_BLOCKS + b against row t of x into columns slot·n_out + … of row t
#ifndef PAIRS_X_SLOT
#define PAIRS_X_SLOT 0               // with PAIRS: 0 = the item's activation row is the token's (x [T, K]: the gate|up input, shared by the
#endif                               // token's slots); 1 = the (token, slot) pair's (x [T·K_TOPK, K]: the down projection reads the slot's activation)
#if PAIRS && ((PAIRS == 1 && T != 1) || NORM || STAT_OUT || EPILOGUE == 1 || X_HOIST)
#error "gemv_T PAIRS: ungrouped tasks need T=1; norm, statistics, residual and activation hoisting are unsupported"
#endif
#ifndef GROUP_STRIDE
#define GROUP_STRIDE (T + 2u)
#endif
#define KL (K / 32u)                                   // columns per lane
#define WPW WEIGHTS_PER_WORD
#define XW (WPW / 8u)                                  // uint4 words of bf16 activations per weight word
#define K_TAIL (KL % WPW)                              // valid columns of the last payload word (0: whole words)
#ifndef SCALE_W0
#define SCALE_W0 PAYLOAD_WORDS                         // the scale bytes start on the word after the payload …
#define SCALE_UOFF 0u                                  // … at its first uint (a ragged stripe puts them in the tail word)
#endif
#if SCALE_GROUP > 0
#ifndef GROUP_SEG
#define GROUP_SEG ((WPW >= SCALE_GROUP) ? SCALE_GROUP : WPW)
#endif
#define WPG GROUP_SEG                                  // weights per in-word scale segment (one group's run of columns)
#define GPW (WPW / GROUP_SEG)                          // segments per word
#if (KL % SCALE_GROUP) == 0
#define LANE_OFF 0u                                    // every lane stripe starts on a group boundary
#else
#define LANE_OFF ((lane * KL) % SCALE_GROUP)           // the stripe's start inside its first group (per lane)
#endif
#define GROUP_OF(j, g) ((LANE_OFF + (j) * WPW + (g) * WPG) / SCALE_GROUP)   // a segment's group, lane-local index
#else
#define GPW 1u
#define WPG WPW
#endif

// block0: the first slab block of the dispatch — a row range of one slab as its own dispatch (a mixer's gate
// projection emitted as an un-barriered sibling of the mixer core, design §5.12): rows [block0·R, block0·R + n_rows)
// of the slab, outputs (and STAT_OUT partials) relative to the range.
struct GemvParams { uint n_rows; uint n_blocks; uint n_sg; uint t_active; float out_scale; float eps; uint stat_parts; uint block0; };

#ifndef LANES_PER_WORD
#define LANES_PER_WORD 1u                              // 2 or 4: a sub-word unit — lanes share one payload word (blm.py), interleaved order only
#endif
#ifndef RSPLIT
#define RSPLIT 1u                                      // work items per block: item i streams rows [i·R/RSPLIT, (i+1)·R/RSPLIT) of its block
#endif                                                 // (silu_mul: that share of the gate rows and their up partners) — a narrow slab's blocks alone
                                                       // leave most of the crew idle (design §5.5: 64 blocks over 240 SIMD-groups); RG divides the share
#define RR (R / RSPLIT)                                // rows per item
#if (R % RSPLIT) != 0 || (RR % RG) != 0
#error "gemv_T: RSPLIT must divide R and RG must divide R / RSPLIT"
#endif
#if EPILOGUE == 2 && ((CHUNK % RSPLIT) != 0 || ((CHUNK / RSPLIT) % RG) != 0)
#error "gemv_T silu_mul: RSPLIT must divide CHUNK and RG must divide CHUNK / RSPLIT"
#endif
static inline uint unit_word(uint lane, uint r, uint j) {
#if LANES_PER_WORD > 1
#if LANE_ORDER == 0
#error "gemv_T: sub-word units need the interleaved lane order"
#endif
  return (r * UNIT_WORDS + j) * (32u / LANES_PER_WORD) + lane / LANES_PER_WORD;   // the word LANES_PER_WORD lanes share
#elif LANE_ORDER == 0
  return (lane * R + r) * UNIT_WORDS + j;
#else
  return (r * UNIT_WORDS + j) * 32u + lane;
#endif
}
// a lane's part of a shared payload word, moved to the front (the rest zero: their columns are past K_TAIL)
static inline uint4 sub_word(uint4 q, uint lane) {
#if LANES_PER_WORD == 2
  return (lane & 1u) ? uint4(q.z, q.w, 0u, 0u) : uint4(q.x, q.y, 0u, 0u);
#elif LANES_PER_WORD == 4
  const uint s = lane & 3u;
  return uint4((s == 0u) ? q.x : (s == 1u) ? q.y : (s == 2u) ? q.z : q.w, 0u, 0u, 0u);
#else
  return q;
#endif
}
#ifndef SCALE_LANE_DIVISOR
#define SCALE_LANE_DIVISOR 1u
#endif
#ifndef SCALE_PLACEMENT
#define SCALE_PLACEMENT 0            // 1: the block's scales in their own region after its payload words (blm.py, #101):
#endif                               //    lane ln's row r scales start (ln * SCALE_RUN) % 16 bytes into word SCALE_WORD(ln, r, 0)
#if SCALE_PLACEMENT
#define SCALE_BASE (R * 32u * PAYLOAD_WORDS / LANES_PER_WORD)          // the block's payload words (sub-word units share words)
#define SCALE_WORD(ln, r, s) (SCALE_BASE + ((r) * (32u / SCALE_LANE_DIVISOR) * SCALE_RUN + ((ln) / SCALE_LANE_DIVISOR) * SCALE_RUN) / 16u + (s))
#define SCALE_SOFF(ln) (((((ln) / SCALE_LANE_DIVISOR) * SCALE_RUN) % 16u) / SCALE_UNIT_BYTES)
#else
#define SCALE_WORD(ln, r, s) unit_word((ln), (r), SCALE_W0 + (s))
#define SCALE_SOFF(ln) 0u
#endif
// Reconstruct a lane's scale words for the legacy GEMV/gather interfaces.
// Matrix and row-specialized kernels read coalesced scale pairs directly.
#if SCALE_PAYLOAD_ORDER
#undef SCALE_SOFF
#define SCALE_SOFF(ln) 0u
static inline uint4 payload_scale_word(device const uint4* wb, uint ln, uint r, uint s) {
  uint4 result = uint4(0);
#pragma clang loop unroll(full)
  for (uint b = 0; b < 16u; b++) {
    const uint g = s * 16u + b;
    if (g < SCALE_RUN) {
      const uint off = r * 32u * SCALE_RUN + (g / 2u) * 64u + ln * 2u + g % 2u;
      const uint v = reinterpret_cast<device const uchar*>(wb + SCALE_BASE)[off];
      result[b / 4u] |= v << ((b % 4u) * 8u);
    }
  }
  return result;
}
#define LOAD_SCALE_WORD(wb, ln, r, s) payload_scale_word((wb), (ln), (r), (s))
#else
#define LOAD_SCALE_WORD(wb, ln, r, s) ((wb)[SCALE_WORD((ln), (r), (s))])
#endif

// Load a short, uint-aligned scale run directly instead of a uint4 followed
// by lane-dependent extraction (affine pairs or byte-sized FP4 scales).
#if !SCALE_PAYLOAD_ORDER && SCALE_PLACEMENT && SCALE_RUN <= 8 && ((SCALE_BIAS && SCALE_UNIT_BYTES == 4) || (SCALE_UNIT_BYTES == 1 && SCALE_RUN % 4 == 0))
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
#define BLOCK_WORDS (R * 32u * UNIT_WORDS / LANES_PER_WORD + SCALE_REGION_WORDS)   // a block: its payload words then its scale region
#else
#define BLOCK_WORDS (R * 32u * UNIT_WORDS / LANES_PER_WORD)
#endif

static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }
static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline uint pack_bf16x2(float lo, float hi) {
  uint a = as_type<uint>(lo); a += 0x7FFFu + ((a >> 16) & 1u);
  uint b = as_type<uint>(hi); b += 0x7FFFu + ((b >> 16) & 1u);
  return (a >> 16) | (b & 0xFFFF0000u);
}

kernel void gemv_T(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
                   device const ushort* x [[buffer(2)]],
#if OUT_BF16
                   device ushort* y [[buffer(3)]],
#else
                   device float* y [[buffer(3)]],
#endif
                   constant GemvParams& p [[buffer(4)]],
#if NORM
                   device const float* stat [[buffer(5)]], device const float* norm_w [[buffer(6)]],
#endif
#if EPILOGUE == 1
                   device const ushort* residual [[buffer(7)]],
#endif
#if STAT_OUT
                   device float* stat_out [[buffer(8)]],
#endif
#if PAIRS
                   device const int* ids [[buffer(9)]],
#endif
#if STEP_STATE
                   device const StepState* st [[buffer(15)]],
#endif
                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T : st->t_this_step)));
  if (T_act == 0u) return;                                     // no rows this step (an LM drafter's chain in a prefill chunk): no weights streamed   // dynamic T (≤ the compiled T), design §5.7
#ifdef T_HI
  if (T_act > T_HI || T_act <= T_LO) return;
#endif
#elif T_STATIC
  const uint T_act = T;                        // compile-time T (plain decode programs, benches)
#else
  const uint T_act = p.t_active;               // <= T; tokens beyond it are skipped
#endif
#if NORM
  float rn[T];                                 // the per-token RMSNorm scale, from the statistic's partial sums
  for (uint t = 0; t < T; t++) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    if (t < T_act) {
      // lane-parallel, four accumulators fed from FOLD_LOADS loads requested together: a row-split producer leaves up
      // to 2048 partials per token, and a round of four loads per latency was 16 latencies — 3.6 µs of a 6144 × 4096
      // GEMV (decode-kernels.md §11.1); the elements reach the same accumulators in the same order as before
      device const float* sp = stat + t * p.stat_parts;
      for (uint base = lane; base < p.stat_parts; base += 32u * FOLD_LOADS) {
        float v[FOLD_LOADS];
        for (uint u = 0; u < FOLD_LOADS; u++) { const uint i = base + 32u * u; v[u] = (i < p.stat_parts) ? sp[i] : 0.0f; }
        for (uint u = 0; u < FOLD_LOADS; u += 4u) { s0 += v[u]; s1 += v[u + 1u]; s2 += v[u + 2u]; s3 += v[u + 3u]; }
      }
    }
    const float ssq = simd_sum((s0 + s1) + (s2 + s3));
    rn[t] = rsqrt(ssq / float(K) + p.eps);
  }
#endif
#if X_HOIST
  // the activation words converted (and normed) once per SIMD-group, ahead of the items: a small-K slab's items
  // re-read and re-converted them per item — the 0.6B's 4096 × 1024 projection is 2048 items of two rows over
  // 960 SIMD-groups (decode-kernels.md §11.1: 13.1 → 12.7 µs, and with one-row items 11.6)
  float xh[PAYLOAD_WORDS][T][WPW];
#if SCALE_BIAS
  float xsh[PAYLOAD_WORDS][T][GPW];
#endif
  for (uint j = 0; j < PAYLOAD_WORDS; j++) {
    const uint col = lane * KL + j * WPW;
#if K_TAIL
    const uint nvalid = (j + 1u == PAYLOAD_WORDS) ? K_TAIL : WPW;
#else
    const uint nvalid = WPW;
#endif
        // convert the activation chunk once per word and reuse it across the RG rows (T*WPW floats of registers)
#if NORM
        float nwv[WPW];
        for (uint e = 0; e < WPW; e += 4) {
          float4 q = (e < nvalid) ? *(device const float4*)(norm_w + col + e) : float4(0.0f);
          nwv[e] = q.x; nwv[e + 1] = q.y; nwv[e + 2] = q.z; nwv[e + 3] = q.w;
        }
#endif
        for (uint t = 0; t < T; t++) {
          if (t < T_act) {
            device const uint4* xp = (device const uint4*)(x + t * K + col);
            for (uint v = 0; v < XW; v++) { uint4 q = (8u * v < nvalid) ? xp[v] : uint4(0u);
              xh[j][t][8 * v] = bf16lo(q.x); xh[j][t][8 * v + 1] = bf16hi(q.x); xh[j][t][8 * v + 2] = bf16lo(q.y); xh[j][t][8 * v + 3] = bf16hi(q.y);
              xh[j][t][8 * v + 4] = bf16lo(q.z); xh[j][t][8 * v + 5] = bf16hi(q.z); xh[j][t][8 * v + 6] = bf16lo(q.w); xh[j][t][8 * v + 7] = bf16hi(q.w); }
#if NORM
            for (uint e = 0; e < WPW; e++) xh[j][t][e] = round_bf16(norm_scale(xh[j][t][e], rn[t], nwv[e]));
#endif
          } else { for (uint e = 0; e < WPW; e++) xh[j][t][e] = 0.0f; }
        }
#if SCALE_BIAS
                                       // Σ x over each scale group of the word, for the bias term
        for (uint t = 0; t < T; t++) for (uint g = 0; g < GPW; g++) { float s = 0.0f; for (uint e = 0; e < WPG; e++) s += xh[j][t][g * WPG + e]; xsh[j][t][g] = s; }
#endif
  }
#endif
#if PAIRS == 2
  // One expert's selected tokens share its decoded weights. The table builder
  // partitions large groups into at most T pairs; stores retain original slots.
  const uint n_items = min(uint(ids[0]), uint(MAX_PAIRS)) * EXPERT_BLOCKS * RSPLIT;
  for (uint it = sg; it < n_items; it += p.n_sg) {
    const uint item = it / RSPLIT, part = it % RSPLIT;
    const uint group = item / EXPERT_BLOCKS, bb = item % EXPERT_BLOCKS;
    device const int* entry = ids + 1u + group * GROUP_STRIDE;
#if GROUP_MIN
    if (uint(entry[1]) < GROUP_MIN) continue;
#endif
#if GROUP_MAX
    if (uint(entry[1]) > GROUP_MAX) continue;
#endif
    const uint b = uint(entry[0]) * EXPERT_BLOCKS + bb;
    const uint T_act = min(uint(entry[1]), uint(T));
    uint pairs[T];
    for (uint t=0; t<T; t++) pairs[t] = t < T_act ? uint(entry[2u+t]) : 0u;
    device const ushort* xrow = x;
#define X_ROW(t) (PAIRS_X_SLOT ? pairs[t] : pairs[t] / K_TOPK)
#elif PAIRS
  const uint n_items = T_act * K_TOPK * EXPERT_BLOCKS * RSPLIT;
  for (uint it = sg; it < n_items; it += p.n_sg) {
    const uint item = it / RSPLIT, part = it % RSPLIT;
    const uint t_tok = item / (K_TOPK * EXPERT_BLOCKS), slot = (item / EXPERT_BLOCKS) % K_TOPK, bb = item % EXPERT_BLOCKS;
    const uint b = uint(ids[t_tok * K_TOPK + slot]) * EXPERT_BLOCKS + bb;    // the slab block of the slot's expert
    device const ushort* xrow = x + (ulong)(PAIRS_X_SLOT ? (t_tok * K_TOPK + slot) : t_tok) * K;   // the item's activation row (T = 1 below)
    const uint ocol0 = slot * (p.n_rows / (EPILOGUE == 2 ? 2u : 1u));      // the slot's columns of the output row
#else
  for (uint it = sg; it < p.n_blocks * RSPLIT; it += p.n_sg) {
    const uint bb = it / RSPLIT, part = it % RSPLIT;                 // the range-relative block and the item's share of its rows
    const uint b = bb + p.block0;                                    // the slab block
    device const ushort* xrow = x;
    const uint t_tok = 0u, ocol0 = 0u;
#endif
#ifndef X_ROW
#define X_ROW(t) (t)
#endif
    device const uint4* wb = w + (ulong)b * BLOCK_WORDS;
#if STAT_OUT
    float ssq_out[T];
    for (uint t = 0; t < T; t++) ssq_out[t] = 0.0f;
#endif
#if EPILOGUE == 2
#define CR (CHUNK / RSPLIT)
#if LOCAL_GATE_CACHE
    // A row-split item only consumes its own gate rows. Relative indices let
    // the compiler keep a single row group's values in registers.
    float gate_v[CR][T];
#define GATE_SLOT (ri + i)
#else
    float gate_v[CHUNK][T];
#endif
    // the item's gate rows [part·CR, (part+1)·CR) then their up partners CHUNK + the same range (the pairs stay in one item)
    for (uint side = 0; side < 2u; side++)
#if LOCAL_GATE_CACHE
    for (uint ri = 0; ri < CR; ri += RG) {
      const uint r0 = side * CHUNK + part * CR + ri;
#else
    for (uint r0 = side * CHUNK + part * CR; r0 < side * CHUNK + (part + 1u) * CR; r0 += RG) {
#endif
#else
    for (uint r0 = part * RR; r0 < (part + 1u) * RR; r0 += RG) {
#endif
      float acc[RG][T];
      for (uint i = 0; i < RG; i++) for (uint t = 0; t < T; t++) acc[i][t] = 0.0f;
#if SCALE_GROUP > 0
#if NARROW_SCALE_RUN
      uint scw[RG][SCALE_RUN / 4u];
      for (uint i = 0; i < RG; i++) for (uint g = 0; g < SCALE_RUN / 4u; g++)
        scw[i][g] = narrow_scale_word(wb, lane, r0 + i, g);
#else
      uint scw[RG][SCALE_WORDS * 4];
      for (uint i = 0; i < RG; i++) for (uint s = 0; s < SCALE_WORDS; s++) {
        uint4 q = LOAD_SCALE_WORD(wb, lane, r0 + i, s);
        scw[i][4 * s] = q.x; scw[i][4 * s + 1] = q.y; scw[i][4 * s + 2] = q.z; scw[i][4 * s + 3] = q.w;
      }
#endif
#endif
      for (uint j = 0; j < PAYLOAD_WORDS; j++) {
        const uint col = lane * KL + j * WPW;
#if K_TAIL
        const uint nvalid = (j + 1u == PAYLOAD_WORDS) ? K_TAIL : WPW;   // a ragged stripe: the last word is partial
#else
        const uint nvalid = WPW;
#endif
#if X_PRECONVERT
#if X_HOIST
        thread float (&xf)[T][WPW] = xh[j];                       // this word's converted activation, hoisted
#if SCALE_BIAS
        thread float (&xs)[T][GPW] = xsh[j];
#endif
#else
        // convert the activation chunk once per word and reuse it across the RG rows (T*WPW floats of registers)
        float xf[T][WPW];
#if NORM
        float nwv[WPW];
        for (uint e = 0; e < WPW; e += 4) {
          float4 q = (e < nvalid) ? *(device const float4*)(norm_w + col + e) : float4(0.0f);
          nwv[e] = q.x; nwv[e + 1] = q.y; nwv[e + 2] = q.z; nwv[e + 3] = q.w;
        }
#endif
        for (uint t = 0; t < T; t++) {
          if (t < T_act) {
            device const uint4* xp = (device const uint4*)(xrow + X_ROW(t) * K + col);
            for (uint v = 0; v < XW; v++) { uint4 q = (8u * v < nvalid) ? xp[v] : uint4(0u);
              xf[t][8 * v] = bf16lo(q.x); xf[t][8 * v + 1] = bf16hi(q.x); xf[t][8 * v + 2] = bf16lo(q.y); xf[t][8 * v + 3] = bf16hi(q.y);
              xf[t][8 * v + 4] = bf16lo(q.z); xf[t][8 * v + 5] = bf16hi(q.z); xf[t][8 * v + 6] = bf16lo(q.w); xf[t][8 * v + 7] = bf16hi(q.w); }
#if NORM
            for (uint e = 0; e < WPW; e++) xf[t][e] = round_bf16(norm_scale(xf[t][e], rn[t], nwv[e]));
#endif
          } else { for (uint e = 0; e < WPW; e++) xf[t][e] = 0.0f; }
        }
#if SCALE_BIAS
        float xs[T][GPW];                                       // Σ x over each scale group of the word, for the bias term
        for (uint t = 0; t < T; t++) for (uint g = 0; g < GPW; g++) { float s = 0.0f; for (uint e = 0; e < WPG; e++) s += xf[t][g * WPG + e]; xs[t][g] = s; }
#endif
#endif
#else
        uint4 xq[T][XW];
        for (uint t = 0; t < T; t++) {
          if (t < T_act) { device const uint4* xp = (device const uint4*)(xrow + X_ROW(t) * K + col); for (uint v = 0; v < XW; v++) xq[t][v] = (8u * v < nvalid) ? xp[v] : uint4(0u); }
          else { for (uint v = 0; v < XW; v++) xq[t][v] = uint4(0); }
        }
#if NORM
        for (uint v = 0; v < XW; v++) {
          if (8u * v >= nvalid) continue;                             // the padded columns of a partial word stay zero
          float4 n0 = *(device const float4*)(norm_w + col + 8 * v), n1 = *(device const float4*)(norm_w + col + 8 * v + 4);
          for (uint t = 0; t < T; t++) {
            if (t >= T_act) continue;
            uint4 q = xq[t][v];
            q.x = pack_bf16x2(norm_scale(bf16lo(q.x), rn[t], n0.x), norm_scale(bf16hi(q.x), rn[t], n0.y));
            q.y = pack_bf16x2(norm_scale(bf16lo(q.y), rn[t], n0.z), norm_scale(bf16hi(q.y), rn[t], n0.w));
            q.z = pack_bf16x2(norm_scale(bf16lo(q.z), rn[t], n1.x), norm_scale(bf16hi(q.z), rn[t], n1.y));
            q.w = pack_bf16x2(norm_scale(bf16lo(q.w), rn[t], n1.z), norm_scale(bf16hi(q.w), rn[t], n1.w));
            xq[t][v] = q;
          }
        }
#endif
#if SCALE_BIAS
        float xs[T][GPW];
        for (uint t = 0; t < T; t++) for (uint g = 0; g < GPW; g++) {
          float s = 0.0f;
          for (uint e = 0; e < WPG; e++) { const uint ee = g * WPG + e; const uint word = xq[t][ee >> 3][(ee >> 1) & 3]; s += (ee & 1u) ? bf16hi(word) : bf16lo(word); }
          xs[t][g] = s;
        }
#endif
#endif
        for (uint i = 0; i < RG; i++) {
          uint4 q = sub_word(wb[unit_word(lane, r0 + i, j)], lane);
          float wv[WPW];
          decode_word(q, wv);
          for (uint t = 0; t < T; t++) {
#if !X_PRECONVERT
            uint xw[XW * 4];
            for (uint v = 0; v < XW; v++) { xw[4 * v] = xq[t][v].x; xw[4 * v + 1] = xq[t][v].y; xw[4 * v + 2] = xq[t][v].z; xw[4 * v + 3] = xq[t][v].w; }
#endif
            for (uint g = 0; g < GPW; g++) {
              if (g * WPG >= nvalid) break;                        // a partial word's padding segments: no scale of theirs is read
              float part = 0.0f;
              for (uint e = 0; e < WPG; e++) {
                const uint ee = g * WPG + e;
#if X_PRECONVERT
                const float xv = xf[t][ee];
#else
                const float xv = (ee & 1u) ? bf16hi(xw[ee >> 1]) : bf16lo(xw[ee >> 1]);
#endif
                part = fma(wv[ee], xv, part);
              }
#if SCALE_GROUP > 0
              const float s = decode_scale(scw[i] + SCALE_UOFF, SCALE_REG_OFFSET(lane) + GROUP_OF(j, g));
              acc[i][t] = fma(part, s, acc[i][t]);
#if SCALE_BIAS
              acc[i][t] = fma(decode_bias(scw[i] + SCALE_UOFF, SCALE_REG_OFFSET(lane) + GROUP_OF(j, g)), xs[t][g], acc[i][t]);
#endif
#else
              acc[i][t] += part;
#endif
            }
          }
        }
      }
      for (uint i = 0; i < RG; i++) {
        const uint r = r0 + i;
        const uint row = b * R + r;                                  // the slab row (weights, row scales)
        const uint rrow = bb * R + r;                                // the range-relative row (outputs)
        const float rs = (rrow < p.n_rows) ? row_scale[row] * p.out_scale : 0.0f;
        for (uint t = 0; t < T; t++) {
          float v = simd_sum(acc[i][t]) * rs;
#if EPILOGUE == 2
          if (r < CHUNK) {
#if LOCAL_GATE_CACHE
            gate_v[GATE_SLOT][t] = v;
#else
            gate_v[r][t] = v;
#endif
            continue;
          }
          const uint orow = bb * CHUNK + (r - CHUNK), n_out = p.n_rows / 2u;
#if LOCAL_GATE_CACHE
          v = silu_mul(gate_v[GATE_SLOT][t], v);
#else
          v = silu_mul(gate_v[r - CHUNK][t], v);
#endif
#else
          const uint orow = rrow, n_out = p.n_rows;
#if EPILOGUE == 1
#if EPILOGUE_ROUND
          v = round_bf16(v);
#endif
          if (orow < n_out && t < T_act) v += as_type<float>(uint(residual[t * n_out + orow]) << 16);
#endif
#endif
          if (orow < n_out && t < T_act) {
            const float vr = round_bf16(v);
#if STAT_OUT
            ssq_out[t] = fma(vr, vr, ssq_out[t]);
#endif
            if (lane == 0) {
#if PAIRS == 2
              const uint yrow = pairs[t] / K_TOPK, ycol = (pairs[t] % K_TOPK) * n_out + orow, ystride = K_TOPK * n_out;
#elif PAIRS
              const uint yrow = t_tok, ycol = ocol0 + orow, ystride = K_TOPK * n_out;
#else
              const uint yrow = t, ycol = orow, ystride = n_out;
#endif
#if OUT_BF16
              y[yrow * ystride + ycol] = ushort(as_type<uint>(vr) >> 16);
#else
              y[yrow * ystride + ycol] = v;
#endif
            }
          }
        }
      }
    }
#if STAT_OUT
    if (lane == 0) for (uint t = 0; t < T; t++) if (t < T_act) stat_out[t * p.n_blocks * RSPLIT + it] = ssq_out[t];   // one partial per item
#endif
  }
}
