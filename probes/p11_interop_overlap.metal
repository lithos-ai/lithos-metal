#include <metal_stdlib>
using namespace metal;
// A: bus-bound streaming (block-cooperative sweep, 64 KB blocks - the GEMV-like access pattern from p5b).
struct SA { uint n_sg; uint bytes_per_sg; uint chunk; uint pad; };
kernel void stream(device const uint* w [[buffer(0)]], device float* out [[buffer(1)]], constant SA& c [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; if (sg >= c.n_sg) return;
  uint words_per_sg = c.bytes_per_sg / 4; const device uint* p = w + (size_t)sg * words_per_sg; uint s = 0;
  uint cw = c.chunk / 4, sub = cw / sw, nblk = words_per_sg / cw;
  for (uint b = 0; b < nblk; b++) { const device uint* q = p + b * cw + lane * sub; for (uint i = 0; i < sub; i++) s += q[i]; }
  out[sg] = float(simd_sum(s) & 0xffff);
}
// B: ALU-bound work with no memory traffic (stands in for a token mixer / sampling / small op).
struct SB { uint work; uint pad0; uint pad1; uint pad2; };
kernel void alu(device float* out [[buffer(0)]], constant SB& c [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  float x = float(gid & 1023) * 1e-3f + 1.0f;
  for (uint i = 0; i < c.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
  out[gid] = x;
}
