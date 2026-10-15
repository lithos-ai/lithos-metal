// Diagnostic only: deliberately materializes FP32 PV partials to separate stages.
// Extra writes/reads and encoder boundaries change timing; this is NOT a
// production alternative or an additive decomposition of the fused kernel.
static_assert(KN * sizeof(M6Operand) >= QM * sizeof(float), "PV readout must fit the KV tile");
kernel void gqa_pv_split_m6(device const ushort* part_prob [[buffer(0)]],
    device const float* part_md [[buffer(1)]], device const ushort* qkvg [[buffer(2)]],
    device ushort* output [[buffer(3)]], constant GqaParams& p [[buffer(4)]],
    device const ushort* v_cache [[buffer(5)]], device float* pv_partials [[buffer(6)]],
#if STEP_STATE
    device const StepState* st [[buffer(15)]],
#endif
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup M6Operand kv_tile[KN * D], prob[QM * CH];
#if STEP_STATE
  if (st->done) return;
  const uint T = st->t_this_step, position = st->position;
#else
  const uint T = p.t_active, position = p.position;
#endif
  if (!T) return;
#if M6_MMA_DISPATCH
  if (T < M6_PV_MIN_ROWS || position + T < 1024u) return;
#endif
  const uint rep = p.heads/p.kv_heads, rows = T*rep, ctx = position+T;
  const uint ch = pick_chunk(ctx, p.kv_heads, rep, p.n_sg, p.n_chunks_max);
  const uint chunks = (ctx+ch-1u)/ch, groups = (rows+QM-1u)/QM;
  const uint j = group/groups, r0 = (group%groups)*QM;
  if (j>=p.kv_heads) return;
  matmul2d<m6_value_desc, execution_simdgroups<MMA_SG>> value_op;
  for (uint c = 0;c<chunks; ++c) {
    for (uint r = sgi;r<QM; r += MMA_SG) {
      const uint row = r0+r, base = (j*p.n_chunks_max+c)*p.rows_max+row;
      for (uint kk = lane;kk<CH; kk += 32u)
        prob[r*CH+kk] = row<rows?M6Operand(bf16f(part_prob[base*CH+kk])):M6Operand(0);
    }
    auto partial = value_op.get_destination_cooperative_tensor<M6Tile, M6Tile, float>();
    for (uint16_t i = 0;i<partial.get_capacity(); ++i) if (partial.is_valid_element(i)) partial[i] = 0.f;
    for (uint offset = 0;offset<ch; offset += KN) {
      for (uint kk = sgi;kk<KN; kk += MMA_SG) {
        const uint key = c*ch+offset+kk;
        float v[DL];
        for (uint e = 0;e<DL; ++e) v[e] = 0.f;
        if (offset+kk<ch && key<ctx) kv_attention_load<V_CACHE_TYPE>(v_cache, key*p.kv_heads+j, v, lane);
        for (uint e = 0;e<DL; ++e) kv_tile[kk*D+lane*DL+e] = M6Operand(v[e]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      M6Tile pt_all(prob, dextents<int, 2>(CH, QM)), vt(kv_tile, dextents<int, 2>(D, KN));
      auto pt = pt_all.slice<KN, QM>(offset, 0);
      value_op.run(pt, vt, partial);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0;i<partial.get_capacity(); ++i) if (partial.is_valid_element(i)) {
      auto ix = partial.get_multidimensional_index(i);
      const uint row = r0 + ix[1];
      if (row < rows) pv_partials[((j*p.n_chunks_max+c)*p.rows_max+row)*D+ix[0]] = partial[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Each query/head keeps the production sequential partition FMA order.
kernel void gqa_partition_split_m6(device const float* partials [[buffer(0)]],
    device const float* md [[buffer(1)]], device float* merged [[buffer(2)]],
    constant GqaParams& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rep=p.heads/p.kv_heads, rows=p.t_active*rep;
  const uint j=group/rows, row=group%rows;
  if (j>=p.kv_heads) return;
  const uint ch=pick_chunk(p.position+p.t_active,p.kv_heads,rep,p.n_sg,p.n_chunks_max);
  const uint chunks=(p.position+p.t_active+ch-1u)/ch;
  float m=-INFINITY;
  for(uint c=lane;c<chunks;c+=32u) m=max(m,md[((j*p.n_chunks_max+c)*p.rows_max+row)*2u]);
  m=simd_max(m);
  float o[DL], den=0.f;
  for(uint e=0;e<DL;++e) o[e]=0.f;
  for(uint c=0;c<chunks;++c) {
    const uint base=(j*p.n_chunks_max+c)*p.rows_max+row;
    const float cm=md[base*2u];
    if(cm!=-INFINITY) {
      const float w=exp(cm-m);
      den=fma(md[base*2u+1u],w,den);
      for(uint e=0;e<DL;++e) o[e]=fma(w,partials[base*D+lane*DL+e],o[e]);
    }
  }
  const uint dst=(j*p.rows_max+row)*(D+1u);
  for(uint e=0;e<DL;++e) merged[dst+lane*DL+e]=o[e];
  if(!lane) merged[dst+D]=den>0.f?1.f/den:0.f;
}

kernel void gqa_finalize_split_m6(device const float* merged [[buffer(0)]],
    device ushort* raw [[buffer(3)]], constant GqaParams& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint rep=p.heads/p.kv_heads, rows=p.t_active*rep;
  const uint j=group/rows, row=group%rows, t=row/rep, h=j*rep+row%rep;
  if(j>=p.kv_heads) return;
  const uint src=(j*p.rows_max+row)*(D+1u);
  float o[DL];
  for(uint e=0;e<DL;++e) o[e]=merged[src+lane*DL+e];
#if V_TURBO4_ROTATED
  kv_rotate_head(o,lane,true);
#endif
  for(uint e=0;e<DL;++e) raw[t*p.out_stride+h*D+lane*DL+e]=bf16bits(round_bf16(o[e]*merged[src+D]));
}

kernel void gqa_gate_split_m6(device const ushort* raw [[buffer(0)]],
    device const ushort* qkvg [[buffer(2)]], device ushort* output [[buffer(3)]],
    constant GqaParams& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint t=group/p.heads,h=group%p.heads;
  if(t>=p.t_active) return;
  for(uint e=0;e<DL;++e) {
    float y=bf16f(raw[t*p.out_stride+h*D+lane*DL+e]);
    if(p.has_gate) {
      const float g=bf16f(qkvg[t*p.in_stride+p.gate_off+h*D+lane*DL+e]);
      y=round_bf16(y*round_bf16(1.f/(1.f+exp(-g))));
    }
#if PERM_OUT
    output[t*PERM_K+perm_dest(h*D+lane*DL+e)]=bf16bits(y);
#else
    output[t*p.out_stride+h*D+lane*DL+e]=bf16bits(y);
#endif
  }
}
