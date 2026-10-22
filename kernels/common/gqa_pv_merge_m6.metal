// M6 second stage: consume BF16-rounded partition probabilities,
// reconstruct each original PV partial, and fold partitions in original order.
// This replaces large FP32 partial writes/reads without changing the partition.
// Requires a device barrier after the score stage: it also fills fresh V rows.
#ifndef PV_QM
#define PV_QM QM
#endif
#ifndef PV_SG
#define PV_SG MMA_SG
#endif
#if PV_QM == QM && PV_SG == MMA_SG
#define M6_PV_VALUE_DESC m6_value_desc
static_assert(KN * sizeof(M6Operand) >= QM * sizeof(float), "PV readout must fit the KV tile");
#else
#define M6_PV_VALUE_DESC m6_tiled_value_desc
constexpr constant auto m6_tiled_value_desc = matmul2d_descriptor(PV_QM, D, KN, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
constexpr constant uint PV_READOUT_ROWS = (PV_QM < KN * sizeof(M6Operand) / sizeof(float) ? PV_QM : KN * sizeof(M6Operand) / sizeof(float));
static_assert(PV_QM % PV_READOUT_ROWS == 0, "PV readout batches must divide query rows");
#endif
kernel void gqa_pv_merge_m6(device const ushort* part_prob [[buffer(0)]],
    device const float* part_md [[buffer(1)]], device const ushort* qkvg [[buffer(2)]],
    device ushort* output [[buffer(3)]], constant GqaParams& p [[buffer(4)]],
    device const ushort* v_cache [[buffer(5)]],
#if STEP_STATE
    device const StepState* st [[buffer(15)]],
#endif
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup M6Operand kv_tile[KN * D], prob[PV_QM * CH];
  threadgroup float maxima[PV_QM], denominators[PV_QM], weights[PV_QM];
  threadgroup uint live[PV_QM];
#if STEP_STATE
  if (st->done) return;
  const uint T = st->t_this_step, position = st->position;
#else
  const uint T = p.t_active, position = p.position;
#endif
  if (!T) return;
#if PV_ACTIVE_MIN
  if (T < PV_ACTIVE_MIN) return;
#endif
#if PV_ACTIVE_MAX
  if (T >= PV_ACTIVE_MAX) return;
#endif
#if M6_MMA_DISPATCH
  if (T < M6_PV_MIN_ROWS || position + T < 1024u) return;
#endif
  const uint rep = p.heads/p.kv_heads, rows = T*rep, ctx = position+T;
  const uint ch = pick_chunk(ctx, p.kv_heads, rep, p.n_sg, p.n_chunks_max);
  const uint chunks = (ctx+ch-1u)/ch, groups = (rows+PV_QM-1u)/PV_QM;
  const uint j = group/groups, r0 = (group%groups)*PV_QM;
  if (j>=p.kv_heads) return;
  for (uint r = sgi;r<PV_QM; r += PV_SG) {
    const uint row = r0+r;
    float m = -INFINITY;
    if (row<rows) for (uint c = lane;c<chunks;c+=32u)
      m = max(m, part_md[((j*p.n_chunks_max+c)*p.rows_max+row)*2u]);
    m = simd_max(m);
    if (!lane) { maxima[r] = m; denominators[r] = 0.f; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  matmul2d<M6_PV_VALUE_DESC, execution_simdgroups<PV_SG>> value_op;
  auto accumulated = value_op.get_destination_cooperative_tensor<M6Tile, M6Tile, float>();
  for (uint16_t i = 0;i<accumulated.get_capacity(); ++i) if (accumulated.is_valid_element(i)) accumulated[i] = 0.f;
  for (uint c = 0;c<chunks; ++c) {
    for (uint r = sgi;r<PV_QM; r += PV_SG) {
      const uint row = r0+r, base = (j*p.n_chunks_max+c)*p.rows_max+row;
      for (uint kk = lane;kk<CH; kk += 32u)
        prob[r*CH+kk] = row<rows?M6Operand(bf16f(part_prob[base*CH+kk])):M6Operand(0);
      if (!lane) {
        const float m = row<rows?part_md[base*2u]:-INFINITY;
        live[r] = m!=-INFINITY;
        const float w = live[r]?exp(m-maxima[r]):0.f;
        weights[r] = w;
        if (live[r]) denominators[r] = fma(part_md[base*2u+1u],w, denominators[r]);
      }
    }
    auto partial = value_op.get_destination_cooperative_tensor<M6Tile, M6Tile, float>();
    for (uint16_t i = 0;i<partial.get_capacity(); ++i) if (partial.is_valid_element(i)) partial[i] = 0.f;
    for (uint offset = 0;offset<ch; offset += KN) {
      for (uint kk = sgi;kk<KN; kk += PV_SG) {
        const uint key = c*ch+offset+kk;
        float v[DL];
        for (uint e = 0;e<DL; ++e) v[e] = 0.f;
        if (offset+kk<ch && key<ctx) kv_attention_load<V_CACHE_TYPE>(v_cache, key*p.kv_heads+j, v, lane);
        for (uint e = 0;e<DL; ++e) kv_tile[kk*D+lane*DL+e] = M6Operand(v[e]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      M6Tile pt_all(prob, dextents<int, 2>(CH, PV_QM)), vt(kv_tile, dextents<int, 2>(D, KN));
      auto pt = pt_all.slice<KN, PV_QM>(offset, 0);
      value_op.run(pt, vt, partial);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0;i<partial.get_capacity(); ++i) if (partial.is_valid_element(i)) {
      auto ix = partial.get_multidimensional_index(i);
      if (live[ix[1]]) accumulated[i] = fma(weights[ix[1]],partial[i],accumulated[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#if PV_QM == QM && PV_SG == MMA_SG
  // KV scratch has enough bytes for the FP32 readout in both operand formats.
  threadgroup float* readout = reinterpret_cast<threadgroup float*>(kv_tile);
  for (uint16_t i = 0;i<accumulated.get_capacity(); ++i) if (accumulated.is_valid_element(i)) {
    auto ix = accumulated.get_multidimensional_index(i);
    readout[ix[1]*D+ix[0]] = accumulated[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = sgi;r<QM; r += MMA_SG) {
    const uint row = r0+r, t = row/rep, h = j*rep+row%rep;
    if (row>=rows) continue;
    float o[DL];
    for (uint e = 0;e<DL; ++e) o[e] = readout[r*D+lane*DL+e];
    const float inv = denominators[r]>0.f?1.f/denominators[r]:0.f;
#if V_TURBO4_ROTATED
    kv_rotate_head(o, lane, true);
#endif
    for (uint e = 0;e<DL; ++e) {
      float y = round_bf16(o[e]*inv);
      if (p.has_gate) {
        const float g = bf16f(qkvg[t*p.in_stride+p.gate_off+h*D+lane*DL+e]);
        y = round_bf16(y*round_bf16(1.f/(1.f+exp(-g))));
      }
#if PERM_OUT
      output[t*PERM_K+perm_dest(h*D+lane*DL+e)] = bf16bits(y);
#else
      output[t*p.out_stride+h*D+lane*DL+e] = bf16bits(y);
#endif
    }
  }
#else
  // Reuse the KV tile for bounded readout batches, without larger shared allocation.
  threadgroup float* readout = reinterpret_cast<threadgroup float*>(kv_tile);
  for (uint rb = 0; rb < PV_QM; rb += PV_READOUT_ROWS) {
    for (uint16_t i = 0;i<accumulated.get_capacity(); ++i) if (accumulated.is_valid_element(i)) {
      auto ix = accumulated.get_multidimensional_index(i);
      if (ix[1] >= rb && ix[1] < rb + PV_READOUT_ROWS)
        readout[(ix[1]-rb)*D+ix[0]] = accumulated[i];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = rb+sgi;r<rb+PV_READOUT_ROWS; r += PV_SG) {
    const uint row = r0+r, t = row/rep, h = j*rep+row%rep;
    if (row>=rows) continue;
    float o[DL];
    for (uint e = 0;e<DL; ++e) o[e] = readout[(r-rb)*D+lane*DL+e];
    const float inv = denominators[r]>0.f?1.f/denominators[r]:0.f;
#if V_TURBO4_ROTATED
    kv_rotate_head(o, lane, true);
#endif
    for (uint e = 0;e<DL; ++e) {
      float y = round_bf16(o[e]*inv);
      if (p.has_gate) {
        const float g = bf16f(qkvg[t*p.in_stride+p.gate_off+h*D+lane*DL+e]);
        y = round_bf16(y*round_bf16(1.f/(1.f+exp(-g))));
      }
#if PERM_OUT
      output[t*PERM_K+perm_dest(h*D+lane*DL+e)] = bf16bits(y);
#else
      output[t*p.out_stride+h*D+lane*DL+e] = bf16bits(y);
#endif
    }
  }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#endif
}
