#include <metal_stdlib>
using namespace metal;
// USE_FENCE=1 builds the spec-compliant variant (MSL >= 3.2): shared buffers are coherent(device) and every publish /
// barrier exit is fenced, as the MSL memory model requires for cross-threadgroup visibility of non-atomic data.
#ifdef USE_FENCE
#define COH coherent(device)
#define FENCE() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)
#else
#define COH
#define FENCE()
#endif
// Claim-based SPMD runtime in miniature. One dispatch runs N_OPS ops in sequence. Every simd-group (SG) runs the same
// program: for each op, lane 0 claims blocks with a bounded CAS on a shared cursor and broadcasts the block id to its 31
// lockstep siblings; all 32 lanes execute the block; lane 0 bumps a monotone `done` counter. When the cursor is exhausted
// the SG waits (bounded spin) until done == n_blocks, i.e. an in-kernel barrier, then moves to the next op.
// No work is statically assigned, so ANY subset of resident SGs completes the program; late arrivals fall through.
struct P { uint n_ops; uint n_blocks; uint work; uint max_spin; uint mode; uint n_sg; uint nominal_sg; uint one_op; };
// mode 2 (recommended): every NOMINAL simd-group v owns the contiguous slice [v*n/NOM, (v+1)*n/NOM) behind its own cursor.
// An SG drains its own slice first (CAS is uncontended in the common case), then steals from the other slices.
// SGs beyond the nominal crew (late / surplus threadgroups) own nothing and only steal. Progress never depends on any
// particular SG being resident.
kernel void spmd(COH device atomic_uint* cursor [[buffer(0)]],   // [n_ops] next unclaimed block
                 COH device atomic_uint* done   [[buffer(1)]],   // [n_ops] completed blocks
                 COH device atomic_uint* hits   [[buffer(2)]],   // [n_ops * n_blocks] execution count per block (must end == 1)
                 device atomic_uint* stats  [[buffer(3)]],   // [0]=barrier timeouts [1]=total barrier spins [2]=blocks run [3]=SGs that ran >=1 block
                 COH device float* sink [[buffer(4)]], constant P& p [[buffer(5)]], device const uint* strides [[buffer(6)]],
                 uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; uint my_blocks = 0; float x = 1.0f + float(lane) * 1e-3f;
  if (p.mode == 3) {                         // ---- one op per dispatch: static slice, no in-kernel synchronization at all
    uint op = p.one_op;
    for (uint b = sg; b < p.n_blocks; b += p.n_sg) {
      for (uint i = 0; i < p.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
      if (lane == 0) { atomic_fetch_add_explicit(&hits[op * p.n_blocks + b], 1, memory_order_relaxed); my_blocks++; }
    }
    sink[gid] = x; if (lane == 0) atomic_fetch_add_explicit(&stats[2], my_blocks, memory_order_relaxed);
    return;
  }
  for (uint op = 0; op < p.n_ops; op++) {
    if (p.mode == 0) {                       // ---- dynamic claiming
      while (true) {
        uint b = 0xFFFFFFFFu;
        if (lane == 0) { uint cur = atomic_load_explicit(&cursor[op], memory_order_relaxed);
          while (cur < p.n_blocks) { if (atomic_compare_exchange_weak_explicit(&cursor[op], &cur, cur + 1, memory_order_relaxed, memory_order_relaxed)) { b = cur; break; } } }
        b = simd_broadcast_first(b);         // lane 0's claim reaches all 32 lanes (they are lockstep anyway)
        if (b == 0xFFFFFFFFu) break;         // op exhausted
        for (uint i = 0; i < p.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
        sink[gid] = x; FENCE();               // block output written, then released
        if (lane == 0) { atomic_fetch_add_explicit(&hits[op * p.n_blocks + b], 1, memory_order_relaxed); atomic_fetch_add_explicit(&done[op], 1, memory_order_relaxed); my_blocks++; }
      }
    } else if (p.mode == 2) {                // ---- own slice first, then steal
      uint NOM = p.nominal_sg;
      for (uint k = 0; k < NOM; k++) {
        // victim order is a per-SG permutation: the host supplies 64 strides coprime with NOM, so every SG visits every
        // victim exactly once and thieves fan out over different victims instead of herding onto the same cursor.
        uint v = (sg + k * strides[sg % 64u]) % NOM; if (k == 0 && sg >= NOM) continue;   // surplus SGs own no slice
        uint lo = v * p.n_blocks / NOM, hi = (v + 1) * p.n_blocks / NOM; uint ci = p.n_ops + op * NOM + v;
        while (true) {
          uint b = 0xFFFFFFFFu;
          if (lane == 0) { uint cur = atomic_load_explicit(&cursor[ci], memory_order_relaxed);
            while (lo + cur < hi) { if (atomic_compare_exchange_weak_explicit(&cursor[ci], &cur, cur + 1, memory_order_relaxed, memory_order_relaxed)) { b = lo + cur; break; } } }
          b = simd_broadcast_first(b);
          if (b == 0xFFFFFFFFu) break;
          for (uint i = 0; i < p.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
          sink[gid] = x; FENCE();               // block output written, then released
        if (lane == 0) { atomic_fetch_add_explicit(&hits[op * p.n_blocks + b], 1, memory_order_relaxed); atomic_fetch_add_explicit(&done[op], 1, memory_order_relaxed); my_blocks++; }
        }
        if (atomic_load_explicit(&done[op], memory_order_relaxed) >= p.n_blocks) break;   // nothing left anywhere: stop scanning
      }
    } else {                                 // ---- static partition (reference): SG s owns blocks s, s+n_sg, ...
      for (uint b = sg; b < p.n_blocks; b += p.n_sg) {
        for (uint i = 0; i < p.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
        sink[gid] = x; FENCE();               // block output written, then released
        if (lane == 0) { atomic_fetch_add_explicit(&hits[op * p.n_blocks + b], 1, memory_order_relaxed); atomic_fetch_add_explicit(&done[op], 1, memory_order_relaxed); my_blocks++; }
      }
    }
    uint s = 0;                              // ---- barrier: all blocks of this op complete (bounded)
    while (atomic_load_explicit(&done[op], memory_order_relaxed) < p.n_blocks) { if (++s >= p.max_spin) { if (lane == 0) atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed); break; } }
    FENCE();                                 // barrier exit: acquire everything published before done reached the target
    if (lane == 0) atomic_fetch_add_explicit(&stats[1], s, memory_order_relaxed);
  }
  sink[gid] = x;
  if (lane == 0) { atomic_fetch_add_explicit(&stats[2], my_blocks, memory_order_relaxed); if (my_blocks) atomic_fetch_add_explicit(&stats[3], 1, memory_order_relaxed); }
}
