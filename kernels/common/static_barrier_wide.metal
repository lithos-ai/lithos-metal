// Bounded device barrier with several SIMD groups polling arrival flags.
// All device fences are retained. A final threadgroup broadcast makes timeout
// handling uniform even when a different worker reports the failure first.
static inline bool stage_barrier(coherent(device) device atomic_uint* flags,
                                 uint worker, uint tid, threadgroup uint& ok) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0)
    ok = atomic_fetch_add_explicit(flags + worker, 1u, memory_order_relaxed) + 1u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32u * POLL_SGS) {
    const uint epoch = ok;
    bool success = true;
    for (uint i = tid; i < WORKERS; i += 32u * POLL_SGS) {
      uint spins = 0;
      while (atomic_load_explicit(flags + i, memory_order_relaxed) < epoch) {
        if (++spins == 1000000u) { success = false; break; }
      }
    }
    success = simd_all(success);
    if (tid % 32u == 0u && !success)
      atomic_store_explicit(flags + WORKERS, 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (tid == 0)
    ok = uint(atomic_load_explicit(flags + WORKERS, memory_order_relaxed) == 0u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return ok != 0;
}
