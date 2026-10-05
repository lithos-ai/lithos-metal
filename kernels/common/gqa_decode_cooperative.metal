// Independent SIMD attention tasks with small cooperative matrix operands.
// Q/K are prepared once by gqa_prepare_mma. Ordinary loads carry the fused
// program's coherent device qualification; no mutable device tensor views.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#define KN CH
#ifndef QM
#define QM 16
#endif
#define AQ ((QM < 16) ? 16 : QM)
#define CK 32u
#define VN 32u
constexpr constant auto score_desc = matmul2d_descriptor(AQ, 32, CK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
constexpr constant auto value_desc = matmul2d_descriptor(AQ, VN, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);

kernel void gqa_decode_mma(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                          device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                          device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                          device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
#if ATTENTION_CACHED_PREFIX
#if DRAFT
                          device const ushort* prefix_k [[buffer(12)]], device const ushort* prefix_v [[buffer(13)]],
#else
                          device const ushort* prefix_k [[buffer(11)]], device const ushort* prefix_v [[buffer(12)]],
#endif
#endif
                          device const StepState* st [[buffer(15)]],
                          uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat score_memory[ATTENTION_SG * QM * KN];
  if (st->done) return;
#if DRAFT
  const uint T=p.t_active, position=st->drafter_ctx_len;
  const uint qpos0=position+st->n_inject, ctx=qpos0+T;
#else
  const uint T=st->t_this_step, position=st->position, ctx=position+T;
#endif
  if (T==0u) return;
  const uint rep=p.heads/p.kv_heads, rows=T*rep, chunks=(ctx+KN-1u)/KN;
  const uint groups=(rows+QM-1u)/QM;
  const uint blocks=p.kv_heads*chunks*groups, jobs=(blocks+ATTENTION_SG-1u)/ATTENTION_SG;
  threadgroup bfloat* score=score_memory+sgi*QM*KN;
  threadgroup bfloat* prob=reinterpret_cast<threadgroup bfloat*>(score);
  for (uint block = tgid; block < jobs; block += p.n_sg) {
    const uint tile=block*ATTENTION_SG+sgi;
    if (tile>=blocks) continue;
    const uint j=tile/(chunks*groups), c=(tile/groups)%chunks, r0=(tile%groups)*QM;
    for (uint sk=0;sk<KN;sk+=32u) {
    matmul2d<score_desc,execution_simdgroup> score_op;
    auto q=score_op.get_left_input_cooperative_tensor<bfloat,bfloat,float>();
    auto k=score_op.get_right_input_cooperative_tensor<bfloat,bfloat,float>();
    auto sc=score_op.get_destination_cooperative_tensor<decltype(q),decltype(k),float>();
    for (uint16_t i=0;i<sc.get_capacity();i++) if(sc.is_valid_element(i)) sc[i]=0.0f;
    for (uint dk=0;dk<D;dk+=CK) {
      for (uint16_t i=0;i<q.get_capacity();i++) if(q.is_valid_element(i)) {
        auto x=q.get_multidimensional_index(i);
        const uint row=r0+x[1], t=row/rep, h=j*rep+row%rep;
        q[i]=x[1]<QM && row<rows ? bfloat(bf16f(qkvg[t*p.in_stride+p.q_off+h*D+dk+x[0]])) : bfloat(0);
      }
      for (uint16_t i=0;i<k.get_capacity();i++) if(k.is_valid_element(i)) {
        auto x=k.get_multidimensional_index(i);
        const uint key=c*KN+sk+x[1];
        k[i]=
#if ATTENTION_CACHED_PREFIX
             key<position ? bfloat(bf16f(prefix_k[(key*p.kv_heads+j)*D+dk+x[0]])) :
#endif
#if DRAFT
             key<qpos0 ? bfloat(bf16f(k_cache[(key*p.kv_heads+j)*D+dk+x[0]])) :
             key<ctx ? bfloat(bf16f(qkvg[(key-qpos0)*p.in_stride+p.k_off+j*D+dk+x[0]])) : bfloat(0);
#else
             key<ctx ? bfloat(bf16f(k_cache[(key*p.kv_heads+j)*D+dk+x[0]])) : bfloat(0);
#endif
      }
      score_op.run(q,k,sc);
    }
    for (uint16_t i=0;i<sc.get_capacity();i++) if(sc.is_valid_element(i)) {
      auto x=sc.get_multidimensional_index(i); if(x[1]<QM) score[x[1]*KN+sk+x[0]]=bfloat(sc[i]);
    }
    } // 32-key score tiles
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Scores are rounded to BF16 before scaling, exactly as in the staged
    // path. Scores and probabilities can therefore share a BF16 tile.
    for (uint r=0;r<QM;r++) {
      const uint row=r0+r, t=row/rep;
      float values[KN/32];
      float m=-INFINITY;
      for(uint u=0;u<KN/32;u++) {
        const uint key=c*KN+lane+u*32;
        const bool visible=row<rows && key<ctx
#if !DRAFT
                           && key<=position+t
#endif
                           ;
        values[u]=visible ? round_bf16(round_bf16(float(score[r*KN+lane+u*32]))*p.scaling) : -INFINITY;
        m=max(m,values[u]);
      }
      m=simd_max(m); float den=0;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for(uint u=0;u<KN/32;u++) {
        const float v=values[u]==-INFINITY ? 0.0f : exp(values[u]-m);
        prob[r*KN+lane+u*32]=bfloat(v); den+=v;
      }
      den=simd_sum(den);
      if(lane==0 && row<rows) {
        const uint base=((j*p.n_chunks_max+c)*p.rows_max+row)*2;
        part_md[base]=m; part_md[base+1]=den;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    matmul2d<value_desc,execution_simdgroup> value_op;
    auto pr=value_op.get_left_input_cooperative_tensor<bfloat,bfloat,float>();
    auto v=value_op.get_right_input_cooperative_tensor<bfloat,bfloat,float>();
    auto out=value_op.get_destination_cooperative_tensor<decltype(pr),decltype(v),float>();
    for(uint d0=0;d0<D;d0+=VN) {
      for(uint16_t i=0;i<out.get_capacity();i++) if(out.is_valid_element(i)) out[i]=0.0f;
      for(uint sk=0;sk<KN;sk+=32u) {
        for(uint16_t i=0;i<pr.get_capacity();i++) if(pr.is_valid_element(i)) {
          auto x=pr.get_multidimensional_index(i); pr[i]=x[1]<QM ? prob[x[1]*KN+sk+x[0]] : bfloat(0);
        }
        for(uint16_t i=0;i<v.get_capacity();i++) if(v.is_valid_element(i)) {
          auto x=v.get_multidimensional_index(i); const uint key=c*KN+sk+x[1];
          v[i]=d0+x[0]>=D ? bfloat(0) :
#if ATTENTION_CACHED_PREFIX
               key<position ? bfloat(bf16f(prefix_v[(key*p.kv_heads+j)*D+d0+x[0]])) :
#endif
#if DRAFT
               key<qpos0 ? bfloat(bf16f(v_cache[(key*p.kv_heads+j)*D+d0+x[0]])) :
               key<ctx ? bfloat(bf16f(qkvg[(key-qpos0)*p.in_stride+p.v_off+j*D+d0+x[0]])) : bfloat(0);
#else
               key<ctx ? bfloat(bf16f(v_cache[(key*p.kv_heads+j)*D+d0+x[0]])) : bfloat(0);
#endif
        }
        value_op.run(pr,v,out);
      }
      for(uint16_t i=0;i<out.get_capacity();i++) if(out.is_valid_element(i)) {
        auto x=out.get_multidimensional_index(i);const uint row=r0+x[1],dim=d0+x[0];
        if(x[1]<QM && row<rows && dim<D) part_o[((j*p.n_chunks_max+c)*p.rows_max+row)*D+dim]=out[i];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
