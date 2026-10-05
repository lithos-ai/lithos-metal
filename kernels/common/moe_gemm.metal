// Grouped NVFP4 expert projection. Each SIMD task owns one expert/output tile;
// the gathered activation rows retain their original token/slot destinations.
using namespace mpp;
using namespace mpp::tensor_ops;
#ifndef NATIVE_A
#define NATIVE_A 0
#endif
constexpr constant auto moe_desc = matmul2d_descriptor(NATIVE_A ? 8 : 16, int(TN), int(TK), false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);

static inline float moe_weight(device const uint4* w, uint row, uint col) {
  const uint block=row/R, r=row%R, ln=col/KL, within=col%KL;
  device const uint4* wb=w+(ulong)block*BLOCK_WORDS;
  uint4 word=sub_word(wb[unit_word(ln,r,within/32u)],ln);
  const uint nibble=(word[(within%32u)/8u]>>((within%8u)*4u))&15u;
  // Independent E2M1 decode (the scalar GEMV snippet can fold a scale factor).
  const float values[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  uint4 scale=LOAD_SCALE_WORD(wb,ln,r,0u);
  uint raw[4]={scale.x,scale.y,scale.z,scale.w};
  const uint at=SCALE_SOFF(ln)+within/16u;
  float sc=fp8_e4m3_scale((raw[at/4u]>>((at%4u)*8u))&255u);
  return ((nibble&8u)?-values[nibble&7u]:values[nibble&7u])*sc;
}

kernel void moe_gemm(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
                     device const ushort* x [[buffer(2)]], device ushort* y [[buffer(3)]],
                     constant GemvParams& p [[buffer(4)]], device const int* groups [[buffer(9)]],
#if STEP_STATE
                     device const StepState* st [[buffer(15)]],
#endif
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if(st->done || st->t_this_step==0u) return;
#endif
  const uint tiles=p.n_rows/TN, count=min(uint(groups[0]),uint(MAX_PAIRS))*tiles;
  matmul2d<moe_desc,execution_simdgroup> op;
#if NATIVE_A
  threadgroup bfloat inputs[MATRIX_SGS][8u*TK];
  using tA_t=tensor<threadgroup bfloat,dextents<int,2>,tensor_inline>;
#endif
  for(uint job=gid/32u;job<count;job+=p.n_sg) {
    const uint group=job/tiles, tile=job%tiles;
    device const int* entry=groups+1u+group*(T+2u);
    const uint expert=uint(entry[0]), active=min(uint(entry[1]),uint(T));
#if !NATIVE_A
    auto aT=op.get_left_input_cooperative_tensor<bfloat,bfloat,float>();
#endif
    auto bT=op.get_right_input_cooperative_tensor<bfloat,bfloat,float>();
#if NATIVE_A
    auto cT=op.get_destination_cooperative_tensor<tA_t,decltype(bT),float>();
#else
    auto cT=op.get_destination_cooperative_tensor<decltype(aT),decltype(bT),float>();
#endif
    for(uint16_t i=0;i<cT.get_capacity();i++) cT[i]=0.f;
    for(uint k=0;k<K;k+=TK) {
#if NATIVE_A
      for(uint at=lane;at<8u*TK;at+=32u) {
        const uint token=at/TK;
        const uint pair=uint(entry[2u+min(token,active-1u)]);
        const uint xr=PAIRS_X_SLOT?pair:pair/K_TOPK;
        inputs[(gid/32u)%MATRIX_SGS][at]=token<active?as_type<bfloat>(x[xr*K+k+at%TK]):bfloat(0);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      tA_t aT(inputs[(gid/32u)%MATRIX_SGS],dextents<int,2>(int(TK),8));
#else
#pragma clang loop unroll(full)
      for(uint16_t i=0;i<aT.get_capacity();i++) {
        auto c=aT.get_multidimensional_index(i);
        uint pair=uint(entry[2u+min(uint(c[1]),active-1u)]);
        uint xr=PAIRS_X_SLOT?pair:pair/K_TOPK;
        aT[i]=c[1]<active?as_type<bfloat>(x[xr*K+k+c[0]]):bfloat(0);
      }
#endif
      const uint c1b=((lane>>1)&3u)+4u*((lane>>4)&1u);
      const uint member=(lane&1u)|(((lane>>3)&1u)<<1);
#pragma clang loop unroll(full)
      for(uint s=0;s<TN/8u;s++) {
#if PACKED_MOE
        const uint row=expert*p.n_rows+tile*TN+s*8u+c1b;
        const ulong tile_id=(ulong)expert*tiles+tile;
        const ulong at=((tile_id*(K/TK)+k/TK)*(TN/8u)+s)*32u+lane;
#endif
#pragma clang loop unroll(full)
        for(uint jump=0;jump<TK/16u;jump++) {
#if PACKED_MOE
          uint word=reinterpret_cast<device const ushort*>(w)[at*(TK/16u)+jump];
          uint sc=reinterpret_cast<device const uchar*>(w)[PACKED_SCALE_BASE+(ulong)row*(K/16u)+k/16u+jump];
          float scale=fp8_e4m3_scale(sc)*16384.f;
#else
          uint row=expert*p.n_rows+tile*TN+s*8u+c1b, col=k+jump*16u+member*4u;
          uint ln=col/KL, within=col%KL;
          device const uint4* wb=w+(ulong)(row/R)*BLOCK_WORDS;
          uint4 payload=sub_word(wb[unit_word(ln,row%R,within/32u)],ln);
          uint word=payload[(within%32u)/8u]>>((within%8u)*4u);
          uint4 scales=LOAD_SCALE_WORD(wb,ln,row%R,0u);
          uint at=SCALE_SOFF(ln)+within/16u;
          float scale=fp8_e4m3_scale((scales[at/4u]>>((at%4u)*8u))&255u)*16384.f;
#endif
#pragma clang loop unroll(full)
          for(uint v=0;v<4u;v++) {
            uint code=(word>>(v*4u))&15u;
            half tiny=as_type<half>(ushort((code&7u)<<9));
            bT[uint16_t(((jump*(TN/8u)+s)<<2)|v)]=bfloat(float((code&8u)?-tiny:tiny)*scale);
          }
        }
      }
      op.run(aT,bT,cT);
#if NATIVE_A
      simdgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }
#pragma clang loop unroll(full)
    for(uint16_t i=0;i<cT.get_capacity();i++) {
      auto c=cT.get_multidimensional_index(i);
      uint row=tile*TN+c[0], token=c[1];
      float value=cT[i]*row_scale[expert*p.n_rows+row]*p.out_scale;
#if EPILOGUE == 2
      float partner=simd_shuffle_xor(value,ushort(8));
      value=value/(1.f+exp(-value))*partner;
      const bool store=row%R<CHUNK;
      row=row/R*CHUNK+row%R;
      const uint width=p.n_rows/2u;
#else
      const bool store=true;
      const uint width=p.n_rows;
#endif
      if(store && token<active) {
        uint pair=uint(entry[2u+token]);
        y[pair*width+row]=ushort(as_type<uint>(round_bf16(value))>>16);
      }
    }
  }
}
