// Fixed worker lists: prepare tasks -> state-column tasks -> gated-norm tasks.
// A worker is a threadgroup; Metal has no API for binding one to a physical core.
// Coherent pointers and device fences publish intermediate data across groups.
static inline bool stage_barrier(coherent(device) device atomic_uint* flags,
                                 uint worker, uint tid, threadgroup uint& ok) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0) {
    ok = 1;
    const uint epoch = atomic_fetch_add_explicit(flags + worker, 1u, memory_order_relaxed) + 1u;
    for (uint i = 0; i < WORKERS; i++) {
      uint spins = 0;
      while (atomic_load_explicit(flags + i, memory_order_relaxed) < epoch) {
        if (++spins == 1000000u) { ok = 0; atomic_store_explicit(flags + WORKERS, 1u, memory_order_relaxed); break; }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return ok;
}
kernel void gdn_static(
  coherent(device) device const ushort* proj [[buffer(0)]],
  coherent(device) device const ushort* ab [[buffer(1)]],
  coherent(device) device ushort* conv [[buffer(2)]],
  coherent(device) device float* state [[buffer(3)]],
  coherent(device) device const ushort* cw [[buffer(4)]],
  coherent(device) device const float* neg [[buffer(5)]],
  coherent(device) device const float* dt [[buffer(6)]],
  coherent(device) device float* part [[buffer(7)]],
  coherent(device) device float* prep [[buffer(8)]],
  constant GdnParams& p [[buffer(9)]],
  coherent(device) device atomic_uint* flags [[buffer(10)]],
  constant GdnParams& np [[buffer(11)]],
  coherent(device) device const float* nw [[buffer(12)]],
  coherent(device) device const ushort* z [[buffer(13)]],
  coherent(device) device ushort* out [[buffer(14)]],
  coherent(device) device const StepState* st [[buffer(15)]],
  uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
  uint worker [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ok;
  if (st->done) return;
  const uint sg = gid / 32u;
  for (uint task = sg; task < 3u * p.hv * st->t_this_step; task += WORKERS * SGS)
    gdn_prepare(proj, ab, conv, cw, neg, dt, prep, p, st, task, lane);
  if (!stage_barrier(flags, worker, tid, ok)) return;
  gdn_mixer(proj, ab, conv, state, cw, neg, dt, part, prep, p, st, gid, lane, 32u);
  if (!stage_barrier(flags, worker, tid, ok)) return;
  for (uint task = sg; task < p.hv * st->t_this_step; task += WORKERS * SGS)
    gdn_norm(part, z, nw, out, np, st, task * 32u + lane, lane, 32u);
}
