// Copyright © 2025-2026 Apple Inc.
// Adapted from ml-explore/mlx mlx/backend/metal/kernels/fp_quantized.h @
// 1f8e74e3f12f31365464a6867c6579f0e9b29d85 (MIT; third_party/NOTICE).
// Quantization-group dot products reuse four activation vectors across two rows.
// MPK's physical pack addressing, input permutation, fused epilogues and guards
// are implemented here. Include after gemm_tile.metal for shared helpers.
// SIMD-groups reduce NV_ROWS rows each. A threadgroup owns NV_SG groups;
// gated epilogues keep the two slab halves in the same threadgroup.
#ifndef NV_ROWS
#define NV_ROWS 2u
#endif
#ifndef NV_SG
#define NV_SG (R / NV_ROWS)
#endif
#ifndef NV_UNROLL
#define NV_UNROLL 2
#endif
#if R % (NV_SG * NV_ROWS) != 0 || (EPILOGUE == 2 && NV_SG * NV_ROWS != R)
#error "NVFP4 row groups must divide a slab block and keep gate/up partners together"
#endif
// The physical weight-column order maps directly to the matrix input tile.
// Avoid a round trip through the pack's logical (lane, word, element) columns.
static inline uint nvfp4_slot4(uint p) {
 const uint kt=p/TK,r=p%TK,mq=r/(TK/4u),r2=r%(TK/4u);
 return (kt*TK+(r2&3u)+((mq&1u)<<2)+((mq>>1)<<3)+((r2>>2)<<4))/4u;
}
kernel void gemv_nvfp4_rows(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
 device const bfloat4* xp [[buffer(2)]],
#if OUT_BF16
 device ushort* y [[buffer(3)]],
#else
 device float* y [[buffer(3)]],
#endif
 constant GemmParams& p [[buffer(4)]],
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
 const uint sg=gid/32u,block=sg/(R/NV_ROWS),r0=(sg%(R/NV_ROWS))*NV_ROWS,row0=p.tile0*TN+block*R+r0;
 // A 512-column step advances whole input tiles; its lane offset is invariant.
 const uint xslot=nvfp4_slot4(lane*16u);
 float acc[NV_ROWS]={0};
 device const uint4* wb=w+(ulong)(p.tile0*TN/R+block)*BLOCK_WORDS;
#pragma clang loop unroll_count(NV_UNROLL)
 for(uint base=0;base<K;base+=32u*16u) {
  const uint phys=base+lane*16u,ln=(phys%(32u*WPW))/WPW,j=phys/(32u*WPW),sub=(phys%WPW)/16u;
  const uint local_g=j*2u+sub;
  float4 x[4];
#pragma clang loop unroll(full)
  for(uint v=0;v<4;v++)x[v]=float4(xp[base/4u+xslot+4u*v]);
#pragma clang loop unroll(full)
  for(uint r=0;r<NV_ROWS;r++) {
   const uint rr=r0+r;
   uint2 q=reinterpret_cast<device const uint2*>(wb)[64u*rr*UNIT_WORDS+base/16u+lane];
   float wv[32];decode_word(uint4(q.x,q.y,0,0),wv);
#if SCALE_PLACEMENT
#if SCALE_PAYLOAD_ORDER
   const uint off=rr*32u*SCALE_RUN+base/16u+lane;
#else
   const uint off=(rr*32u+ln)*SCALE_RUN+local_g;
#endif
   uint sw=reinterpret_cast<device const uint*>(wb+SCALE_BASE)[off/4u];
   const float scale=decode_scale(&sw,off%4u);
#else
   uint sw=reinterpret_cast<device const uint*>(wb)[4u*unit_word(ln,rr,SCALE_W0+local_g/16u)+(local_g%16u)/4u];
   const float scale=decode_scale(&sw,local_g%4u);
#endif
   float sum=0;
#pragma clang loop unroll(full)
   for(uint v=0;v<4;v++)sum+=dot(x[v],float4(wv[v*4],wv[v*4+1],wv[v*4+2],wv[v*4+3]));
   acc[r]=fma(scale,sum,acc[r]);
  }
 }
#if EPILOGUE == 2
 threadgroup float vals[R];
#endif
#if STAT_OUT
 threadgroup float stats[NV_SG * NV_ROWS];
#endif
#pragma clang loop unroll(full)
 for(uint r=0;r<NV_ROWS;r++){
  acc[r]=simd_sum(acc[r])*GEMM_ROW_SCALE(min(row0+r,p.tile0*TN+p.n_rows-1u))*p.out_scale;
#if EPILOGUE == 2
  if(lane==0) vals[r0+r]=acc[r];
#endif
 }
#if EPILOGUE == 2
 threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
#pragma clang loop unroll(full)
 for(uint r=0;r<NV_ROWS;r++){
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
  if(lane==0)stats[(sg%NV_SG)*NV_ROWS+r]=writer&&o<nout?vr*vr:0.f;
#endif
 }
#if STAT_OUT
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg%NV_SG==0){float s=simd_sum(lane<NV_SG*NV_ROWS?stats[lane]:0.f);if(lane==0)stat_out[sg/NV_SG]=s;}
#endif
}
