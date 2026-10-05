// argmax over BF16 logits [T][vocab] in two dispatches (a REDUCE whose partials are combined in block order, so the
// result is deterministic; ties → the lowest index, like torch.argmax):
//   argmax_partial: SIMD-group sg scans static slices of 256-logit spans (32 lanes × 8 contiguous logits = one
//                   coalesced 512-byte read per span) for each token and writes its (max, index) partial;
//   argmax_final:   one SIMD-group per token folds the n_sg partials in order and writes token[t].
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef ARGMAX_LAST
#define ARGMAX_LAST 0                // 1: only the last of the T_act rows is reduced (an LM drafter's first chain step) into token[0]
#endif
#ifndef T_SRC
#define T_SRC 0                      // with STEP_STATE: 0 = t_this_step, 1 = n_inject, 2 = the static T_STATIC_ROWS
#endif
#ifndef T_STATIC_ROWS
#define T_STATIC_ROWS 1u
#endif
struct ArgmaxParams { uint vocab; uint t_active; uint n_sg; uint n_spans; };

static inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }

static inline void better(thread float& best, thread uint& bi, float v, uint i) {
  if (v > best || (v == best && i < bi)) { best = v; bi = i; }
}

static inline void simd_argmax(thread float& best, thread uint& bi) {
  for (uint off = 16u; off > 0u; off >>= 1u) {
    float ov = simd_shuffle_down(best, off);
    uint oi = simd_shuffle_down(bi, off);
    better(best, bi, ov, oi);
  }
  best = simd_broadcast_first(best);
  bi = simd_broadcast_first(bi);
}

kernel void argmax_partial(device const ushort* logits [[buffer(0)]], device float* part_val [[buffer(1)]], device uint* part_idx [[buffer(2)]],
                           constant ArgmaxParams& p [[buffer(3)]],
#if STEP_STATE
                           device const StepState* st [[buffer(15)]],
#endif
                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step)));
#else
  const uint T_act = p.t_active;
#endif
#if ARGMAX_LAST
  const uint t_first = (T_act > 0u) ? T_act - 1u : 0u;              // only the last row: its partials and token land in slot 0
#else
  const uint t_first = 0u;
#endif
  for (uint t = t_first; t < T_act; t++) {
    float best = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    device const ushort* row = logits + (ulong)t * p.vocab;
    for (uint s = sg; s < p.n_spans; s += p.n_sg) {
      const uint base = s * 256u + lane * 8u;
      if (base + 8u <= p.vocab) {
        uint4 q = *(device const uint4*)(row + base);
        uint w[4] = {q.x, q.y, q.z, q.w};
        for (uint v = 0; v < 4; v++) {
          better(best, bi, as_type<float>(w[v] << 16), base + 2u * v);
          better(best, bi, as_type<float>(w[v] & 0xFFFF0000u), base + 2u * v + 1u);
        }
      } else {
        for (uint e = 0; e < 8u; e++) if (base + e < p.vocab) better(best, bi, bf16f(row[base + e]), base + e);
      }
    }
    simd_argmax(best, bi);
    if (lane == 0) { part_val[(t - t_first) * p.n_sg + sg] = best; part_idx[(t - t_first) * p.n_sg + sg] = bi; }
  }
}

kernel void argmax_final(device const float* part_val [[buffer(0)]], device const uint* part_idx [[buffer(1)]], device int* token [[buffer(2)]],
                         constant ArgmaxParams& p [[buffer(3)]],
#if STEP_STATE
                         device const StepState* st [[buffer(15)]],
#endif
                         uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
#if STEP_STATE
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step)));
  if (st->done || t >= T_act) return;
#else
  const uint T_act = p.t_active;
  if (t >= T_act) return;
#endif
#if ARGMAX_LAST
  if (t != 0u) return;                                                 // the last row's partials sit in slot 0, its token goes to token[0]
#endif
  float best = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint i = lane; i < p.n_sg; i += 32u) better(best, bi, part_val[t * p.n_sg + i], part_idx[t * p.n_sg + i]);
  simd_argmax(best, bi);
  if (lane == 0) token[t] = int(bi);
}
