#include <metal_stdlib>
using namespace metal;
struct C { uint work; uint n_workers_sg; uint max_ticks; uint pad; };
// SG 0 of TG 0 = "clock warp": free-runs incrementing an atomic tick counter until all workers are done (bounded).
// Every other simd-group does a fixed ALU workload and timestamps its begin/end by reading the tick counter.
kernel void clocked(device atomic_uint* tick [[buffer(0)]], device atomic_uint* done [[buffer(1)]], device uint* stamps [[buffer(2)]],
                    constant C& c [[buffer(3)]], device float* sink [[buffer(4)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw;
  if (sg == 0) { if (lane != 0) return;   // (siblings are lockstep-bound anyway)
    for (uint t = 0; t < c.max_ticks; t++) { atomic_fetch_add_explicit(tick, 1, memory_order_relaxed);
      if ((t & 1023u) == 0 && atomic_load_explicit(done, memory_order_relaxed) >= c.n_workers_sg) break; }
    return; }
  uint t0 = atomic_load_explicit(tick, memory_order_relaxed);
  float x = float(gid & 1023) * 1e-3f + 1.0f; uint w = c.work * (1u + (sg & 3u));          // 4 workload classes: 1x..4x
  for (uint i = 0; i < w; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
  sink[gid] = x;
  uint t1 = atomic_load_explicit(tick, memory_order_relaxed);
  if (lane == 0) { stamps[sg*2] = t0; stamps[sg*2+1] = t1; atomic_fetch_add_explicit(done, 1, memory_order_relaxed); }
}
