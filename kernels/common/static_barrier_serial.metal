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
