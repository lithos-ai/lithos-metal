// Bounded, coherent all-worker barrier. One SIMD group polls the arrival flags
// in parallel; all threads still publish/observe through device-scope fences.
static inline bool stage_barrier(coherent(device) device atomic_uint* flags,
                                 uint worker, uint tid, threadgroup uint& ok) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0)
    ok = atomic_fetch_add_explicit(flags + worker, 1u, memory_order_relaxed) + 1u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32u) {
    const uint epoch = ok;
    bool success = true;
    for (uint i = tid; i < WORKERS; i += 32u) {
      uint spins = 0;
      while (atomic_load_explicit(flags + i, memory_order_relaxed) < epoch) {
        if (++spins == 1000000u) { success = false; break; }
      }
    }
    success = simd_all(success);
    if (tid == 0) {
      ok = uint(success);
      if (!success) atomic_store_explicit(flags + WORKERS, 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return ok != 0;
}
