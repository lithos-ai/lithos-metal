#include <metal_stdlib>
using namespace metal;
struct C { uint work; uint active_per_tg; uint active_tgs; uint pad; };
// Fixed ALU work for the first `active_per_tg` simd-groups of the first `active_tgs` threadgroups.
kernel void aluwork(device float* out [[buffer(0)]], constant C& c [[buffer(1)]],
                    uint gid [[thread_position_in_grid]], uint tg [[threadgroup_position_in_grid]],
                    uint lt [[thread_position_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
  if (tg >= c.active_tgs || (lt / sw) >= c.active_per_tg) return;
  float x = float(gid & 1023) * 1e-3f + 1.0f;
  for (uint i = 0; i < c.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
  out[gid] = x;
}
