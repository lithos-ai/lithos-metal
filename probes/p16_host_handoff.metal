#include <metal_stdlib>
using namespace metal;

// One thread advances a counter and logs a host-written word: log[v] = a[1] where v is the counter before the bump.
kernel void bump(device uint* a [[buffer(0)]], device uint* log [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i == 0) { uint v = a[0]; log[v & 4095u] = a[1]; a[0] = v + 1u; }
}

// A bounded ALU loop: keeps the GPU busy for a controllable time.
kernel void spin_work(device float* a [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  float x = a[i & 1023u];
  for (uint k = 0; k < n; k++) x = fma(x, 1.0000001f, 1e-7f);
  a[i & 1023u] = x;
}
