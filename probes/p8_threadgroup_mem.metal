#include <metal_stdlib>
using namespace metal;
struct C { uint iters; uint n; uint mode; uint pad; };
// Repeatedly dot a small "activation vector" (n halfs, like the 5120-wide hidden state = 10 KB) against a pseudo-weight stream,
// reading the activations from: mode 0 = device buffer (shared by all), mode 1 = threadgroup memory (copied in once per TG),
// mode 2 = thread-private stack copy of a 256-element window (register/private memory).
kernel void hot(device const half* act [[buffer(0)]], device float* out [[buffer(1)]], constant C& c [[buffer(2)]],
                threadgroup half* tg [[threadgroup(0)]],
                uint gid [[thread_position_in_grid]], uint lt [[thread_position_in_threadgroup]], uint tpt [[threads_per_threadgroup]]) {
  if (c.mode == 1) { for (uint i = lt; i < c.n; i += tpt) tg[i] = act[i]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float acc = 0; uint idx = gid * 7u;
  if (c.mode == 0)      for (uint it = 0; it < c.iters; it++) { idx = (idx + 13u) & (c.n - 1u); acc += float(act[idx]) * 1.0001f; }
  else if (c.mode == 1) for (uint it = 0; it < c.iters; it++) { idx = (idx + 13u) & (c.n - 1u); acc += float(tg[idx]) * 1.0001f; }
  else { half priv[256]; for (uint i = 0; i < 256; i++) priv[i] = act[i]; for (uint it = 0; it < c.iters; it++) { idx = (idx + 13u) & 255u; acc += float(priv[idx]) * 1.0001f; } }
  out[gid] = acc;
}
