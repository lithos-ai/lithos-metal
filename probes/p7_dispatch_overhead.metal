#include <metal_stdlib>
using namespace metal;
struct A { uint n; uint k; };
// tiny elementwise op standing in for the many small per-layer ops of a decode step (5120-wide vectors)
kernel void tiny(device float* x [[buffer(0)]], device float* y [[buffer(1)]], device float* z [[buffer(2)]],
                 constant A& a [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < a.n) z[i] = x[i] * 0.999f + y[i]; }
// the same N ops done as ONE dispatch with an in-kernel loop (what a megakernel does with small ops)
kernel void fusedloop(device float* x [[buffer(0)]], device float* y [[buffer(1)]], device float* z [[buffer(2)]],
                 constant A& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.n) return; float v = z[i]; for (uint k = 0; k < a.k; k++) v = x[i] * 0.999f + y[i] + v * 0.0f; z[i] = v; }
