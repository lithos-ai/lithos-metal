// gdn_mixer + gdn_norm: the Gated-DeltaNet mixer for T tokens (design §5.6; issue #22), in the reference's order and
// rounding (transformers' modeling_qwen3_5.py: causal_conv1d → l2norm q/k → σ(b), −exp(A_log)·softplus(a + dt_bias)
// → torch_recurrent_gated_delta_rule → RMSNormGated).
//
// Block = (value head h, group of SPB state-column slices of SL columns); a SIMD-group takes static slices of the
// Hv × (DV / (SL·SPB)) blocks. The recurrence is column-separable, so a block owns its columns' state outright and
// writes its columns of the read-out o (FP32) to a workspace; gdn_norm (one SIMD-group per (token, head)) applies
// the gated RMSNorm over the whole head afterwards. Lane ℓ owns k-rows {ℓ, ℓ+32, …} of the [DK, DV] FP32 state and
// channels {ℓ, ℓ+32, …} of the head's q, k (its key head = h / (Hv/Hk), HF's repeat_interleave) and v.
//   1. conv + SiLU for the lane's channels over the window [conv_state | x_0 .. x_{T-1}] (FP32 taps of the BF16
//      weights, BF16 rounding, SiLU in FP32 → BF16); the last CW-1 inputs become the new conv state (q/k channels
//      are written by the first value head of the key head, v channels by their own head);
//   2. per token: β = bf16(σ(b)), g = −exp(A_log)·softplus(a + dt_bias) in FP32, q/k L2-normalized over DK
//      (rsqrt(Σx² + 1e-6)), q /= √DK;
//   3. the recurrence is column-separable, so the head is processed in DV/SL column slices with the slice's state
//      (KR × SL floats per lane) in registers: S ← S·e^g; kv = kᵀS (simd_sum per column); Δ = (v − kv)·β;
//      S += k ⊗ Δ; o = qᵀS; S is stored back once per slice. Tokens go TP at a time (registers), each pass reads
//      and writes the state once; the block's o columns go to the workspace o_part [T][Hv·DV];
//   4. gdn_norm, per token and head over DV: o → BF16, rstd = rsqrt(mean(o²) + eps),
//      y = bf16(bf16(bf16(o·rstd)·w)·silu(z)).
// Macros: DK, DV (multiples of 32), CW (conv width), SL (columns per state slice), SPB (slices per block), TP (tokens
// per pass). The conv state is written by the block with slice group 0 of each head (q/k channels only by the
// first value head of the key head).
//
// State slots (SLOTS=2, needs STEP_STATE): the two states are double-buffered by step parity — the step's pass reads
// slot (step & 1) and writes the other, so the writer of a step never aliases the window its readers replay (with
// one slot, a fast head could overwrite the conv window a slower block of the same head is still reading). In a
// speculative program (design §5.8) the same kernel with COMMIT=1 runs after the accept scan (which advanced
// `step`): it recomputes the recurrence for the committed tokens (StepState.checkpoint_index) from the slot the step's pass read and
// overwrites the slot the pass wrote — the rejected positions never reach the state — and writes no read-out: its
// o_part binding is a placeholder the kernel never touches (an early version stored the read-out through it, past
// the end of a 16-byte value: the source of an intermittent wrong-token / hang / empty-generation failure of the
// speculative programs, found with MTL_SHADER_VALIDATION=1). The emitter compiles every
// program with SLOTS=2 (the advance op alternates `step` too): with one slot and several value heads per key head
// (hv > hk), the v-head blocks sharing a key head's conv window race on it — a block that finishes first overwrites
// the window a slower sibling is still reading (seen as a flaky oracle test under GPU contention). SLOTS=1 is a
// bench-only mode for hv == hk.
#ifndef FUSED_NORM
#define FUSED_NORM 0
#endif
#if FUSED_NORM && (!PREPARED || SPB != 1 || COMMIT)
#error "fused GDN normalization needs one prepared state slice per SIMD group"
#endif
#ifndef LOCAL_PREPARE
#define LOCAL_PREPARE 0
#endif
#if LOCAL_PREPARE && (!PREPARED || COMMIT || SPB != 1)
#error "local preparation needs one prepared state slice per SIMD group"
#endif
#ifndef PREPARED
#define PREPARED 0
#endif
#if PREPARED
#define TREG 1u
#define TI 0u
#else
#define TREG TP
#define TI t
#endif
#ifndef PRECONVOLVED
#define PRECONVOLVED 0
#endif
#if PRECONVOLVED && (!PREPARED || COMMIT)
#error "preconvolved rows require a prepared forward recurrence"
#endif
#ifndef SINGLE_PASS
#define SINGLE_PASS 0
#endif
#if SINGLE_PASS && (!LOCAL_PREPARE || COMMIT)
#error "single-pass recurrence requires local preparation and one state slice per SIMD group"
#endif
#define PREP_STRIDE (2u * DK + DV + 2u)
#ifndef SL
#define SL 8u
#endif
#ifndef SPB
#define SPB 1u
#endif
#ifndef TP
#define TP 4u
#endif
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef SLOTS
#define SLOTS 1u
#endif
#ifndef COMMIT
#define COMMIT 0
#endif
#if (SLOTS == 2u || COMMIT) && !STEP_STATE
#error "state slots and the commit pass read StepState"
#endif
#define KR (DK / 32u)
#define VR (DV / 32u)
#define NSL (DV / SL)
#define NSG (NSL / SPB)
#ifndef LOCAL_GROUPS
#define LOCAL_GROUPS NSG
#endif
#if LOCAL_PREPARE && (NSG % LOCAL_GROUPS != 0 || LOCAL_GROUPS > 32)
#error "local preparation groups must divide one head and fit a threadgroup"
#endif
#if LOCAL_PREPARE && FUSED_NORM && LOCAL_GROUPS != NSG
#error "fused normalization requires a whole head in one threadgroup"
#endif

struct GdnParams {
  uint hv; uint hk; uint t_active; uint q_off;
  uint k_off; uint v_off; uint z_off; uint a_off;
  uint b_off; uint in_stride; uint ab_stride; uint ab_separate;
  uint out_stride; uint n_sg; uint key_dim; uint pad0;
  float eps; float pad1; float pad2; float pad3;
};

static inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline ushort bf16bits(float v) { return ushort(as_type<uint>(round_bf16(v)) >> 16); }
static inline float silu_f(float x) { return x / (1.0f + exp(-x)); }
static inline float softplus_f(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// conv + SiLU of channel c for tokens t0 .. t0+n-1 (window = conv_state row, then the projection's inputs)
static inline void conv_channel(device const ushort* proj, uint in_stride, device const ushort* conv_state, device const ushort* conv_w,
                                uint c, uint t0, uint n, thread float* y) {
  float w[CW], win[CW - 1u];
  for (uint j = 0; j < CW; j++) w[j] = bf16f(conv_w[c * CW + j]);
  for (uint j = 0; j < CW - 1u; j++) win[j] = bf16f(conv_state[c * (CW - 1u) + j]);
  for (uint t = 0; t < t0 + n; t++) {                       // replay the step's inputs up to the pass's tokens
    const float x = bf16f(proj[t * in_stride + c]);
    if (t >= t0) {
      float acc = 0.0f;
      for (uint j = 0; j < CW - 1u; j++) acc = fma(w[j], win[j], acc);
      acc = fma(w[CW - 1u], x, acc);
      y[t - t0] = round_bf16(silu_f(round_bf16(acc)));
    }
    for (uint j = 0; j + 1u < CW - 1u; j++) win[j] = win[j + 1u];
    win[CW - 2u] = x;
  }
}

static inline void conv_state_update(device const ushort* proj, uint in_stride, device const ushort* src, device ushort* dst, uint c, uint T) {
#if PRECONVOLVED
  return;
#endif
  // Only the final CW-1 inputs survive; copy them directly, including any
  // retained prefix when the step is shorter than the convolution window.
  for (uint j = 0; j < CW - 1u; j++) {
    const uint pos = T + j;
    dst[c * (CW - 1u) + j] = pos < CW - 1u ? src[c * (CW - 1u) + pos]
                                                   : proj[(pos - (CW - 1u)) * in_stride + c];
  }
}

static inline float pick(thread const float* arr, uint i) {         // arr[i] with a compile-time-indexed body
  float r = arr[0];
  for (uint k = 1; k < VR; k++) if (i == k) r = arr[k];
  return r;
}

// A prepared token only needs the convolution window ending at that token.
static inline float conv_at(device const ushort* proj, uint stride, device const ushort* state,
                            device const ushort* weight, uint channel, uint token) {
#if PRECONVOLVED
  return bf16f(proj[token * stride + channel]);
#endif
  float acc = 0.0f;
#pragma clang loop unroll(full)
  for (uint tap = 0; tap < CW; tap++) {
    const uint pos = token + tap;
    const float x = pos < CW - 1u ? bf16f(state[channel * (CW - 1u) + pos])
                                        : bf16f(proj[(pos - (CW - 1u)) * stride + channel]);
    acc = fma(bf16f(weight[channel * CW + tap]), x, acc);
  }
  return round_bf16(silu_f(round_bf16(acc)));
}

// The same arithmetic feeds either a device workspace or the whole head's
// threadgroup workspace. Only the last token writes the opposite state slot.
template<typename Output>
static inline void prepare_token(device const ushort* proj, device const ushort* proj_ab,
                                 device const ushort* conv_state, device ushort* conv_dst,
                                 device const ushort* conv_w, device const float* neg_exp_a_log,
                                 device const float* dt_bias, constant GdnParams& p,
                                 uint t, uint T, uint h, uint kh, uint kind, uint lane, Output dst) {
  device const ushort* pq = proj + p.q_off;
#if DK == DV
  // Equal head dimensions share the convolution body; only q/k need L2 normalization.
  float vec[KR];
  const uint base = kind < 2u ? kind * p.key_dim + kh * DK : 2u * p.key_dim + h * DV;
  for (uint i = 0; i < KR; i++)
    vec[i] = conv_at(pq, p.in_stride, conv_state, conv_w, base + lane + 32u * i, t);
  if (kind < 2u) {
    float ss = 0.f;
    for (uint i = 0; i < KR; i++) ss = fma(vec[i], vec[i], ss);
    const float inv = rsqrt(simd_sum(ss) + 1e-6f);
    for (uint i = 0; i < KR; i++) {
      vec[i] *= inv;
      if (kind == 0u) vec[i] /= sqrt(float(DK));
    }
  }
  for (uint i = 0; i < KR; i++) dst[kind * DK + lane + 32u * i] = vec[i];
#else
  if (kind < 2u) {
    float vec[KR], ss = 0.0f;
    const uint base = kind * p.key_dim + kh * DK;
    for (uint i = 0; i < KR; i++) {
      vec[i] = conv_at(pq, p.in_stride, conv_state, conv_w, base + lane + 32u * i, t);
      ss = fma(vec[i], vec[i], ss);
    }
    const float inv = rsqrt(simd_sum(ss) + 1e-6f);
    for (uint i = 0; i < KR; i++) {
      float v = vec[i] * inv;
      if (kind == 0u) v /= sqrt(float(DK));
      dst[kind * DK + lane + 32u * i] = v;
    }
  } else {
    for (uint i = 0; i < VR; i++)
      dst[2u * DK + lane + 32u * i] = conv_at(pq, p.in_stride, conv_state, conv_w, 2u * p.key_dim + h * DV + lane + 32u * i, t);
  }
#endif
  if (kind == 2u && lane == 0) {
    device const ushort* ab = p.ab_separate ? proj_ab : proj;
    const uint stride = p.ab_separate ? p.ab_stride : p.in_stride;
    const float a = bf16f(ab[t * stride + p.a_off + h]), b = bf16f(ab[t * stride + p.b_off + h]);
    dst[2u * DK + DV] = round_bf16(1.0f / (1.0f + exp(-b)));
    dst[2u * DK + DV + 1u] = exp(neg_exp_a_log[h] * softplus_f(a + dt_bias[h]));
  }
#if SLOTS == 2u
  // The last token's preparation owns the final window. Readers use the other
  // slot, so q/k/v groups can write independently without a cross-group race.
  // Keeping these address calculations out of the recurrence reduces its live registers.
  if (t + 1u == T && (kind == 2u || h % (p.hv / p.hk) == 0u)) {
    const uint base = kind < 2u ? kind * p.key_dim + kh * DK : 2u * p.key_dim + h * DV;
    for (uint i = 0; i < (kind < 2u ? KR : VR); i++)
      conv_state_update(pq, p.in_stride, conv_state, conv_dst, base + lane + 32u * i, T);
  }
#endif
}

// Compute the convolution and normalization once per (token, value head),
// rather than once per state-column block. The recurrence keeps the same FP32 order.
kernel void gdn_prepare(device const ushort* proj [[buffer(0)]], device const ushort* proj_ab [[buffer(1)]],
                        device ushort* conv_state [[buffer(2)]], device const ushort* conv_w [[buffer(4)]],
                        device const float* neg_exp_a_log [[buffer(5)]], device const float* dt_bias [[buffer(6)]],
                        device float* prepared [[buffer(8)]], constant GdnParams& p [[buffer(9)]],
#if STEP_STATE
                        device const StepState* st [[buffer(15)]],
#endif
                        uint sg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if (st->done) return;
  const uint T = st->t_this_step;
#else
  const uint T = p.t_active;
#endif
  const uint group = sg / 3u, kind = sg % 3u;
  const uint t = group / p.hv, h = group % p.hv, kh = h / (p.hv / p.hk);
  if (t >= T) return;
#if SLOTS == 2u
  device ushort* conv_dst = conv_state + ((st->step & 1u) ^ 1u) * (2u * p.key_dim + p.hv * DV) * (CW - 1u);
  conv_state += (st->step & 1u) * (2u * p.key_dim + p.hv * DV) * (CW - 1u);
#else
  device ushort* conv_dst = conv_state;
#endif
  device float* dst = prepared + (t * p.hv + h) * PREP_STRIDE;
  prepare_token(proj, proj_ab, conv_state, conv_dst, conv_w, neg_exp_a_log, dt_bias,
                p, t, T, h, kh, kind, lane, dst);
}

kernel void gdn_mixer(device const ushort* proj [[buffer(0)]], device const ushort* proj_ab [[buffer(1)]],
                      device ushort* conv_state [[buffer(2)]], device float* rec_state [[buffer(3)]],
                      device const ushort* conv_w [[buffer(4)]], device const float* neg_exp_a_log [[buffer(5)]],
                      device const float* dt_bias [[buffer(6)]], device float* o_part [[buffer(7)]],

#if PREPARED && !LOCAL_PREPARE
                      device const float* prepared [[buffer(8)]],
#endif
                      constant GdnParams& p [[buffer(9)]],
#if FUSED_NORM
                      constant GdnParams& np [[buffer(11)]],
                      device const float* norm_w [[buffer(12)]],
                      device const ushort* z [[buffer(13)]],
                      device ushort* final_out [[buffer(14)]],
#endif
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
#if FUSED_NORM
  // One whole head per threadgroup: state slices publish each token for its
  // gated RMSNorm, preserving the standalone norm's lane and summation order.
  threadgroup float readout[TP][DV];
#endif
#if LOCAL_PREPARE
  threadgroup float local_prep[TP * PREP_STRIDE];
#endif
  const uint sg = gid / sw;
  const uint rep = p.hv / p.hk;
#if STEP_STATE
  if (st->done) return;
#if COMMIT
  // Intermediate prefill accepts every input row. The forward recurrence has
  // already written the new slot, and accept_scan advanced the parity to it.
  // Only speculative rejection needs a replay of the committed prefix.
  if (st->prefill_left > 0u) return;
  const uint T = st->checkpoint_index;                       // the committed tokens of the step that just ended (n_inject is the drafter's row count)
#else
  const uint T = st->t_this_step;
#endif
#else
  const uint T = p.t_active;
#endif
#if SLOTS == 2u
#if COMMIT
  const uint rd = (st->step + 1u) & 1u, wr = st->step & 1u;  // the accept scan advanced `step`: re-read the pass's input slot
#else
  const uint rd = st->step & 1u, wr = (st->step + 1u) & 1u;
#endif
  const ulong rec_stride = (ulong)p.hv * DK * DV;
  const uint conv_stride = (2u * p.key_dim + p.hv * DV) * (CW - 1u);
  device const ushort* conv_in = conv_state + rd * conv_stride;
  device ushort* conv_out = conv_state + wr * conv_stride;
  device const float* rec_in = rec_state + rd * rec_stride;
  device float* rec_out = rec_state + wr * rec_stride;
#else
  device const ushort* conv_in = conv_state;
  device ushort* conv_out = conv_state;
  device const float* rec_in = rec_state;
  device float* rec_out = rec_state;
#endif
  const uint n_blocks = p.hv * NSG;
  // conv channels are indexed relative to the q|k|v columns (q at q_off, k at q_off + key_dim, v at q_off + 2·key_dim):
  // the projection may carry other rows ahead of them
  device const ushort* pq = proj + p.q_off;
#if SINGLE_PASS
  // The emitter launches every state slice, with compiled T <= TP. No grid or
  // token-pass replay is needed; partial active lengths still use live state.
  const uint b = sg;
  if (b < n_blocks) {
#else
  for (uint b = sg; b < n_blocks; b += p.n_sg) {
#endif
    const uint h = b / NSG, grp = b % NSG;
    const uint kh = h / rep;
#if SINGLE_PASS
    const uint t0 = 0;
    if (T > 0) {
#else
    for (uint t0 = 0; t0 < T; t0 += TP) {
#endif
      const uint n = min(TP, T - t0);
#if LOCAL_PREPARE
      // A threadgroup owns a whole head or a disjoint subset of its columns.
      // Each subset shares preparation; only the first writes the conv state.
      for (uint job = grp % LOCAL_GROUPS; job < 3u * n; job += LOCAL_GROUPS)
        prepare_token(proj, proj_ab, conv_in, conv_out, conv_w, neg_exp_a_log, dt_bias,
                      p, t0 + job / 3u, grp < LOCAL_GROUPS ? T : 0u, h, kh, job % 3u, lane,
                      local_prep + (job / 3u) * PREP_STRIDE);
      threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
      // 1. conv + SiLU of this lane's channels for the pass's tokens
      float qv[TREG][KR], kv[TREG][KR], vv[TREG][VR];
      float beta[TREG], eg[TREG];
#if !PREPARED
      for (uint i = 0; i < KR; i++) {
        float y[TP];
        conv_channel(pq, p.in_stride, conv_in, conv_w, kh * DK + lane + 32u * i, t0, n, y);
        for (uint t = 0; t < TP; t++) qv[t][i] = y[t];
        conv_channel(pq, p.in_stride, conv_in, conv_w, p.key_dim + kh * DK + lane + 32u * i, t0, n, y);
        for (uint t = 0; t < TP; t++) kv[t][i] = y[t];
      }
      for (uint i = 0; i < VR; i++) {
        float y[TP];
        conv_channel(pq, p.in_stride, conv_in, conv_w, 2u * p.key_dim + h * DV + lane + 32u * i, t0, n, y);
        for (uint t = 0; t < TP; t++) vv[t][i] = y[t];
      }
      // 2. per-token scalars and the q/k L2 norms
      for (uint t = 0; t < TP; t++) {
        if (t < n) {
          const uint tt = t0 + t;
          device const ushort* ab = p.ab_separate ? proj_ab : proj;
          const uint abs_ = p.ab_separate ? p.ab_stride : p.in_stride;
          const float a = bf16f(ab[tt * abs_ + p.a_off + h]), b = bf16f(ab[tt * abs_ + p.b_off + h]);
          beta[t] = round_bf16(1.0f / (1.0f + exp(-b)));
          eg[t] = exp(neg_exp_a_log[h] * softplus_f(a + dt_bias[h]));
          float sq = 0.0f, sk = 0.0f;
          for (uint i = 0; i < KR; i++) { sq = fma(qv[t][i], qv[t][i], sq); sk = fma(kv[t][i], kv[t][i], sk); }
          sq = simd_sum(sq); sk = simd_sum(sk);
          const float rq = rsqrt(sq + 1e-6f), rk = rsqrt(sk + 1e-6f);
          const float scale = sqrt(float(DK));
          for (uint i = 0; i < KR; i++) { qv[t][i] = (qv[t][i] * rq) / scale; kv[t][i] = kv[t][i] * rk; }
        } else {
          beta[t] = 0.0f; eg[t] = 1.0f;
          for (uint i = 0; i < KR; i++) { qv[t][i] = 0.0f; kv[t][i] = 0.0f; }
        }
      }
#endif
      // 3. the recurrence over this block's state slices
      for (uint s = grp * SPB; s < (grp + 1u) * SPB; s++) {
        float S[KR][SL];
        for (uint i = 0; i < KR; i++) {                        // the first pass reads the step's input slot, later passes the slot the block writes
          device const float* state = (t0 == 0u) ? rec_in : (device const float*)rec_out;
          for (uint j = 0; j < SL; j++) S[i][j] = state[((ulong)(h * DK + lane + 32u * i)) * DV + s * SL + j];
        }
        for (uint t = 0; t < TP; t++) {
          if (t >= n) break;
#if PREPARED
#if LOCAL_PREPARE
          threadgroup const float* src = local_prep + t * PREP_STRIDE;
#else
          device const float* src = prepared + ((t0 + t) * p.hv + h) * PREP_STRIDE;
#endif
          for (uint i = 0; i < KR; i++) {
            qv[0][i] = src[lane + 32u * i]; kv[0][i] = src[DK + lane + 32u * i];
          }
          beta[0] = src[2u * DK + DV]; eg[0] = src[2u * DK + DV + 1u];
#endif
          for (uint i = 0; i < KR; i++) for (uint j = 0; j < SL; j++) S[i][j] *= eg[TI];
          float delta[SL];
          for (uint j = 0; j < SL; j++) {
            float part = 0.0f;
            for (uint i = 0; i < KR; i++) part = fma(S[i][j], kv[TI][i], part);
            const float kvm = simd_sum(part);
            const uint v = s * SL + j;
#if PREPARED
            const float vt = src[2u * DK + v];
#else
            const float vt = simd_shuffle(pick(vv[TI], v / 32u), ushort(v % 32u));
#endif
            delta[j] = (vt - kvm) * beta[TI];
          }
          for (uint i = 0; i < KR; i++) for (uint j = 0; j < SL; j++) S[i][j] = fma(kv[TI][i], delta[j], S[i][j]);
#if !COMMIT
          for (uint j = 0; j < SL; j++) {                        // the read-out: the step's pass only (the commit pass
            float part = 0.0f;                                   // advances the state and binds no output of its own)
            for (uint i = 0; i < KR; i++) part = fma(S[i][j], qv[TI][i], part);
            const float o = simd_sum(part);
            if (lane == 0) {
#if FUSED_NORM
              readout[t][s * SL + j] = o;
#else
              o_part[(t0 + t) * p.out_stride + h * DV + s * SL + j] = o;
#endif
            }
          }
#endif
        }
        for (uint i = 0; i < KR; i++) {
          for (uint j = 0; j < SL; j++) rec_out[((ulong)(h * DK + lane + 32u * i)) * DV + s * SL + j] = S[i][j];
        }
      }
#if LOCAL_PREPARE && !FUSED_NORM && !SINGLE_PASS
      if (t0 + TP < T) threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
#if FUSED_NORM
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint t = sg % NSG; t < n; t += NSG) {
        float ob[VR], ss = 0.0f;
        for (uint i = 0; i < VR; i++) {
          ob[i] = round_bf16(readout[t][lane + 32u * i]);
          ss = fma(ob[i], ob[i], ss);
        }
        const float rstd = rsqrt(simd_sum(ss) / float(DV) + np.eps);
        for (uint i = 0; i < VR; i++) {
          const uint v = lane + 32u * i;
          float y = round_bf16(ob[i] * rstd);
          y = round_bf16(norm_w[v] * y);
          const float zz = bf16f(z[(t0 + t) * np.in_stride + h * DV + v]);
          y = round_bf16(y * silu_f(zz));
#if PERM_OUT
          final_out[(t0 + t) * p.out_stride + perm_dest(h * DV + v)] = bf16bits(y);
#else
          final_out[(t0 + t) * p.out_stride + h * DV + v] = bf16bits(y);
#endif
        }
      }
      if (t0 + TP < T) threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }
    // the new conv state: the last CW-1 inputs of the step (q/k channels once per key head, v channels per head),
    // written by the head's first slice group
#if !PREPARED || SLOTS != 2u
    if (grp == 0u) {
      if (h % rep == 0u) {
        for (uint i = 0; i < KR; i++) {
          conv_state_update(pq, p.in_stride, conv_in, conv_out, kh * DK + lane + 32u * i, T);
          conv_state_update(pq, p.in_stride, conv_in, conv_out, p.key_dim + kh * DK + lane + 32u * i, T);
        }
      }
      for (uint i = 0; i < VR; i++) conv_state_update(pq, p.in_stride, conv_in, conv_out, 2u * p.key_dim + h * DV + lane + 32u * i, T);
    }
#endif
  }
}

kernel void gdn_norm(device const float* o_part [[buffer(0)]], device const ushort* proj [[buffer(1)]], device const float* norm_w [[buffer(2)]],
                     device ushort* out [[buffer(3)]], constant GdnParams& p [[buffer(4)]],
#if STEP_STATE
                     device const StepState* st [[buffer(15)]],
#endif
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / p.hv, h = sg % p.hv;
#if STEP_STATE
  if (st->done || t >= st->t_this_step) return;
#else
  if (t >= p.t_active) return;
#endif
  float ob[VR], ss = 0.0f;
  for (uint i = 0; i < VR; i++) { ob[i] = round_bf16(o_part[t * p.out_stride + h * DV + lane + 32u * i]); ss = fma(ob[i], ob[i], ss); }
  ss = simd_sum(ss);
  const float rstd = rsqrt(ss / float(DV) + p.eps);
  for (uint i = 0; i < VR; i++) {
    const uint v = lane + 32u * i;
    float y = round_bf16(ob[i] * rstd);
    y = round_bf16(norm_w[v] * y);
    const float z = bf16f(proj[t * p.in_stride + p.z_off + h * DV + v]);
    y = round_bf16(y * silu_f(z));
#if PERM_OUT
    out[t * p.out_stride + perm_dest(h * DV + v)] = bf16bits(y);
#else
    out[t * p.out_stride + h * DV + v] = bf16bits(y);
#endif
  }
}
