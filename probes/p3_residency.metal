#include <metal_stdlib>
using namespace metal;
struct C { uint S; uint max_spin; uint work; uint active; };
// (a) Full residency: EVERY simd-group (lane 0 acts; its 31 siblings are lockstep-bound to it) checks in and
// waits (bounded) until all S simd-groups of the dispatch have checked in. No thread returns early.
kernel void residency(device atomic_uint* ctr [[buffer(0)]], device uint* res [[buffer(1)]], constant C& c [[buffer(2)]],
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; uint s = 0; uint seen = 0; bool ok = false;
  if (lane == 0) atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
  // all 32 lanes spin (keeps the whole simd-group genuinely busy, no divergence tricks)
  while (s < c.max_spin) { seen = atomic_load_explicit(ctr, memory_order_relaxed); if (seen >= c.S) { ok = true; break; } s++; }
  if (lane == 0) { res[sg*2+0] = ok ? s : 0xFFFFFFFFu; res[sg*2+1] = seen; }
}
// (b) Physical concurrency: each of the first `active` simd-groups does a fixed amount of pure ALU work (no memory traffic).
// Wall time stays flat while active <= hardware concurrency, then grows linearly.
kernel void aluwork(device float* out [[buffer(0)]], constant C& c [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; if (sg >= c.active) return;
  float x = float(gid & 1023) * 1e-3f + 1.0f; 
  for (uint i = 0; i < c.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
  out[gid] = x;
}
