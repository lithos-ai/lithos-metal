// Copyright © 2023-2024 Apple Inc.
// Adapted from ml-explore/mlx mlx/backend/metal/kernels/gemv.h @
// 1f8e74e3f12f31365464a6867c6579f0e9b29d85 (MIT; third_party/NOTICE).
// Four-vector prefetch/reuse and per-row shuffle reduction follow GemvWide.
// MPK's packed weights, permuted input, row ranges, epilogues and state guards
// are handled here. One threadgroup owns a pack block: 16 lanes per row.
// Include after gemm_tile.metal for its macros, parameters and helpers.
static inline uint packed_slot4(uint i) {
  // The matrix input's within-tile permutation, in four-BF16 vectors.
  return (i / (TK / 4u)) * (TK / 4u) + (i % (TK / 4u)) / (TK / 16u) + 4u * (i % (TK / 16u));
}

kernel void gemv_bf16_small(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
                      device bfloat* xp [[buffer(2)]],
#if OUT_BF16
                      device ushort* y [[buffer(3)]],
#else
                      device float* y [[buffer(3)]],
#endif
                      constant GemmParams& p [[buffer(4)]],
#if SHARED_NORM
                      device const float* stat [[buffer(5)]], device const float* norm_w [[buffer(6)]],
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
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
#if STEP_STATE
  if (st->done) return;                                                     // uniform over the threadgroup: no barrier is skipped
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step)));
  if (T_act == 0u) return;                                                  // no rows this step (an LM drafter's chain in a prefill chunk)
#ifdef T_HI
  if (T_act > T_HI || T_act <= T_LO) return;
#endif
#else
  const uint T_act = p.t_active;
#endif

#if SHARED_NORM
  if (T_act == 0u || T_act > 4u) return;
#endif
#define VECTORS 4
#define KLANES 16u
#define RPS (32u/KLANES)
#if EPILOGUE == 2
  threadgroup float scratch[R][VECTORS];
#endif
#if STAT_OUT
  threadgroup float stats[R][VECTORS];
#endif
  const uint block = gid/(KLANES*R), rr=(sg%(R/RPS))*RPS+lane/KLANES;
  const uint klane = lane%KLANES;
  const uint row=p.tile0*TN+block*R+rr, rrow=block*R+rr;
  device const bfloat4 *w4=(device const bfloat4*)w+(ulong)min(row,p.tile0*TN+p.n_rows-1u)*(K/4u);
  // Share one normalized input tile across all weight rows in this block.
#if SHARED_NORM
  threadgroup bfloat4 local_x[4u * K / 4u];
  // Distribute the four token norms across SIMD-groups instead of repeating
  // all four reductions in every group. R=8/16 gives one/two groups per token.
  const uint nv = (gid / 32u) % 4u;
  {
    const uint active_v = min(nv, T_act - 1u);
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (uint b = lane; b < STAT_PARTS; b += 512u) {
      float v[16];
      for (uint u = 0; u < 16; u++)
        v[u] = b + 32u * u < STAT_PARTS ? stat[active_v * STAT_PARTS + b + 32u * u] : 0.f;
      for (uint u = 0; u < 16; u += 4) { s0 += v[u]; s1 += v[u+1]; s2 += v[u+2]; s3 += v[u+3]; }
    }
    const float rn = rsqrt(simd_sum((s0 + s1) + (s2 + s3)) / float(K) + EPS);
    for (uint i = lane + 32u * ((gid / 32u) % (R / 2u) / 4u); i < K / 4u; i += 32u * (R / 8u)) {
      const uint phys = i * 4u, ln = (phys % (32u * WPW)) / WPW, j = phys / (32u * WPW);
      const uint col = ln * KL + j * WPW + phys % WPW;
      const float4 raw = float4(((device const bfloat4*)xp)[col / 4u + active_v * (K / 4u)]);
      const float4 nw = *(device const float4*)(norm_w + col);
      bfloat4 val;
      for (uint e = 0; e < 4; e++) val[e] = bfloat(round_bf16(norm_scale(raw[e], rn, nw[e])));
      local_x[nv * (K / 4u) + packed_slot4(i)] = val;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const bfloat4 *x4 = local_x;
#else
  device const bfloat4 *x4 = (device const bfloat4*)xp;
#endif
  for (uint vc = 0; vc < T_act; vc += VECTORS) {
    float result[VECTORS] = {0};
    for (uint base = 0; base < K/4u; base += KLANES*8u) {
      float4 wf[8];
#pragma clang loop unroll(full)
      for (uint j = 0; j < 8; j++) wf[j]=base+j*KLANES+klane<K/4u ? float4(w4[base+j*KLANES+klane]):float4(0);
#pragma clang loop unroll(full)
      for (uint v = 0; v < VECTORS; v++) {
        float acc = 0;
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8; j++) if (base+j*KLANES+klane<K/4u) acc+=dot(wf[j],float4(x4[min(vc+v,T_act-1u)*(K/4u)+packed_slot4(base+j*KLANES+klane)]));
        result[v]+=acc;
      }
    }
#pragma clang loop unroll(full)
    for (uint v = 0; v < VECTORS; v++) {
      for (ushort offset = KLANES/2u; offset > 0; offset >>= 1) result[v]+=simd_shuffle_down(result[v],offset);
      result[v]=result[v]*row_scale[min(row,p.tile0*TN+p.n_rows-1u)]*p.out_scale;
#if EPILOGUE == 2
      if (klane==0) scratch[rr][v]=result[v];
#endif
    }
#if EPILOGUE == 2
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
#pragma clang loop unroll(full)
    for (uint v = 0; v < VECTORS; v++) {
      float value=result[v];
#if EPILOGUE == 2
      value=silu_mul(value, scratch[(rr+R/2u)%R][v]);
      const uint orow=block*(R/2u)+rr, nout=p.n_rows/2u;
      const bool writer=rr<R/2u;
#else
      const uint orow=rrow,nout=p.n_rows;
      const bool writer=true;
#endif
#if EPILOGUE == 1
#if EPILOGUE_ROUND
      value=round_bf16(value);
#endif
      value+=as_type<float>(uint(residual[min(vc+v,T_act-1u)*nout+min(orow,nout-1u)])<<16);
#endif
      float vr=round_bf16(value);
      if (klane==0 && writer && vc+v<T_act && orow<nout) {
#if PERM_OUT
        y[(vc+v)*PERM_K+perm_dest(orow)]=ushort(as_type<uint>(vr)>>16);
#else
#if OUT_BF16
        y[(vc+v)*nout+orow]=ushort(as_type<uint>(vr)>>16);
#else
        y[(vc+v)*nout+orow]=value;
#endif
#endif
      }
#if STAT_OUT
      if (klane==0) stats[rr][v]=writer && orow<nout ? vr*vr:0.0f;
#endif
    }
#if STAT_OUT
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg%(R/RPS)==0) {
#pragma clang loop unroll(full)
      for (uint v = 0; v < VECTORS; v++) {
        float ss=simd_sum(lane<R ? stats[lane][v]:0.0f);
        if (lane==0 && vc+v<T_act) stat_out[(vc+v)*p.n_blocks+block]=ss;
      }
    }
#endif
#if EPILOGUE == 2 || STAT_OUT
    if (vc + VECTORS < T_act) threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
  }
}

