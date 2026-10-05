#include <metal_stdlib>
using namespace metal;
struct P { uint a_tid; uint b_tid; uint rounds; uint max_spin; };
// Ping-pong: global thread a_tid handles even counter values, b_tid handles odd ones. Bounded spins => always terminates.
kernel void pingpong(device atomic_uint* ctr [[buffer(0)]], device uint* res [[buffer(1)]], constant P& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid != p.a_tid && gid != p.b_tid) return;
  uint parity = (gid == p.a_tid) ? 0u : 1u;
  uint slot = parity * 4;
  uint done = 0, total_spins = 0, max_seen = 0, timeouts = 0;
  for (uint r = 0; r < p.rounds; r++) {
    uint want = 2*r + parity; uint s = 0; bool ok = false;
    while (s < p.max_spin) { if (atomic_load_explicit(ctr, memory_order_relaxed) == want) { ok = true; break; } s++; }
    total_spins += s; max_seen = max(max_seen, s);
    if (!ok) { timeouts++; break; }
    atomic_store_explicit(ctr, want + 1, memory_order_relaxed); done++;
  }
  res[slot+0] = done; res[slot+1] = total_spins; res[slot+2] = max_seen; res[slot+3] = timeouts;
}
// Check-in: lane 0 of every threadgroup increments a counter, then waits (bounded) until all G groups have checked in.
struct C { uint G; uint max_spin; };
kernel void checkin(device atomic_uint* ctr [[buffer(0)]], device uint* res [[buffer(1)]], constant C& c [[buffer(2)]],
                    uint tg [[threadgroup_position_in_grid]], uint lt [[thread_position_in_threadgroup]]) {
  if (lt != 0) return;
  uint order = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
  uint s = 0; uint seen = 0;
  while (s < c.max_spin) { seen = atomic_load_explicit(ctr, memory_order_relaxed); if (seen >= c.G) break; s++; }
  res[tg*3+0] = order; res[tg*3+1] = s; res[tg*3+2] = seen;
}
