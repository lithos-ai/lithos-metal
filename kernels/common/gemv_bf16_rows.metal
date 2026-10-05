// Copyright © 2023-2024 Apple Inc.
// Adapted from ml-explore/mlx mlx/backend/metal/kernels/gemv.h @
// 1f8e74e3f12f31365464a6867c6579f0e9b29d85 (MIT; third_party/NOTICE).
// Reuse a four-element activation vector across two output rows, with MPK's
// packed weights, optional RMS normalization and fused epilogues.
// Include after gemm_tile.metal. One threadgroup owns one R-row pack block.
#ifndef BF_ROWS
#define BF_ROWS 2u
#endif
#ifndef PROJ_CONV
#define PROJ_CONV 0
#endif
#if PROJ_CONV && (!OUT_BF16 || !STEP_STATE || EPILOGUE != 0 || CONV_WIDTH < 2)
#error "projection convolution requires a plain BF16 projection and two state slots"
#endif
static inline uint bf16_slot4(uint i) {
 return (i/(TK/4u))*(TK/4u)+(i%(TK/4u))/(TK/16u)+4u*(i%(TK/16u));
}
kernel void gemv_bf16_rows(device const bfloat4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
 device const bfloat4* xp [[buffer(2)]],
#if OUT_BF16
 device ushort* y [[buffer(3)]],
#else
 device float* y [[buffer(3)]],
#endif
 constant GemmParams& p [[buffer(4)]],
#if PROJ_CONV
 device ushort* conv_state [[buffer(10)]], device const ushort* conv_w [[buffer(11)]],
#endif
#if DIRECT_NORM
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
 uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
 if(st->done) return;
 const uint ta=(T_SRC==1)?st->n_inject:((T_SRC==3)?st->n_chain:((T_SRC==4)?st->n_inject+st->n_chain:((T_SRC==2)?T_STATIC_ROWS:st->t_this_step)));
#else
 const uint ta=p.t_active;
#endif
 if(ta!=1u)return;
#ifdef T_HI
 if(ta > T_HI || ta <= T_LO)return;
#endif
 const uint sg=gid/32u,block=sg/(R/BF_ROWS),r0=(sg%(R/BF_ROWS))*BF_ROWS,row0=p.tile0*TN+block*R+r0;
#if DIRECT_NORM
 float s0=0,s1=0,s2=0,s3=0;
 for(uint b=lane;b<STAT_PARTS;b+=512u){
  float v[16];for(uint u=0;u<16;u++)v[u]=b+32u*u<STAT_PARTS?stat[b+32u*u]:0.f;
  for(uint u=0;u<16;u+=4){s0+=v[u];s1+=v[u+1];s2+=v[u+2];s3+=v[u+3];}
 }
 const float rn=rsqrt(simd_sum((s0+s1)+(s2+s3))/float(K)+EPS);
#endif
 float acc[BF_ROWS]={0};
#pragma clang loop unroll_count(2)
 for(uint b=0;b<K/4u;b+=32u) {
  const uint k=b+lane;
#if DIRECT_NORM
  const uint phys=k*4u,ln=(phys%(32u*WPW))/WPW,j=phys/(32u*WPW),col=ln*KL+j*WPW+phys%WPW;
  const float4 raw=float4(xp[col/4u]),nw=*(device const float4*)(norm_w+col);
  float4 x;
  for(uint e=0;e<4;e++)x[e]=round_bf16(norm_scale(raw[e], rn, nw[e]));
#else
  const float4 x=float4(xp[bf16_slot4(k)]);
#endif
#pragma clang loop unroll(full)
  for(uint r=0;r<BF_ROWS;r++){
   float4 v=float4(w[(ulong)min(row0+r,p.tile0*TN+p.n_rows-1u)*(K/4u)+k]);
   acc[r]+=dot(x,v);
  }
 }
#if EPILOGUE == 2
 threadgroup float vals[R];
#endif
#if STAT_OUT
 threadgroup float stats[R];
#endif
#pragma clang loop unroll(full)
 for(uint r=0;r<BF_ROWS;r++){
  acc[r]=simd_sum(acc[r])*GEMM_ROW_SCALE(min(row0+r,p.tile0*TN+p.n_rows-1u))*p.out_scale;
#if EPILOGUE == 2
  if(lane==0) vals[r0+r]=acc[r];
#endif
 }
#if EPILOGUE == 2
 threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
#pragma clang loop unroll(full)
 for(uint r=0;r<BF_ROWS;r++){
  uint rr=r0+r;
#if EPILOGUE == 2
  const uint o=block*(R/2u)+rr,nout=p.n_rows/2u;
  bool writer=rr<R/2u;
  float v=silu_mul(acc[r], vals[(rr+R/2u)%R]);
#else
  const uint o=block*R+rr,nout=p.n_rows;
  bool writer=true;
  float v=acc[r];
#endif
#if EPILOGUE == 1
#if EPILOGUE_ROUND
  v=round_bf16(v);
#endif
  v+=as_type<float>(uint(residual[min(o,nout-1u)])<<16);
#endif
#if OUT_BF16
  float vr=round_bf16(v);
#else
  float vr=v;
#endif
#if PROJ_CONV
  if (o >= CONV_START && o < CONV_START + CONV_DIM) {
   const uint c = o - CONV_START;
   const uint stride = CONV_DIM * (CONV_WIDTH - 1u);
   device const ushort* src = conv_state + (st->step & 1u) * stride + c * (CONV_WIDTH - 1u);
   device ushort* dst = conv_state + ((st->step + 1u) & 1u) * stride + c * (CONV_WIDTH - 1u);
   float sum = 0;
#pragma clang loop unroll(full)
   for (uint j = 0; j < CONV_WIDTH - 1u; j++)
    sum = fma(as_type<float>(uint(conv_w[c * CONV_WIDTH + j]) << 16),
              as_type<float>(uint(src[j]) << 16), sum);
   sum = fma(as_type<float>(uint(conv_w[c * CONV_WIDTH + CONV_WIDTH - 1u]) << 16), vr, sum);
   if (lane == 0) {
    for (uint j = 0; j < CONV_WIDTH - 2u; j++) dst[j] = src[j + 1u];
    // Preserve the raw BF16 projection for the next step's convolution window.
    dst[CONV_WIDTH - 2u] = ushort(as_type<uint>(vr) >> 16);
   }
   vr = round_bf16(silu_f(round_bf16(sum)));
  }
#endif
  if(lane==0&&writer&&o<nout){
#if PERM_OUT
   const uint dst=perm_dest(o);
#else
   const uint dst=o;
#endif
#if OUT_BF16
   y[dst]=ushort(as_type<uint>(vr)>>16);
#else
   y[dst]=vr;
#endif
  }
#if STAT_OUT
  if(lane==0)stats[rr]=writer&&o<nout?vr*vr:0.f;
#endif
 }
#if STAT_OUT
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg%(R/BF_ROWS)==0){float s=simd_sum(lane<R?stats[lane]:0.f);if(lane==0)stat_out[block]=s;}
#endif
}
