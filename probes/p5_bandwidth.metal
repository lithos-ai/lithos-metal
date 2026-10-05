#include <metal_stdlib>
using namespace metal;
struct C { uint n_sg; uint bytes_per_sg; uint fmt; uint pad; };
// Streaming-read bandwidth: each simd-group owns a disjoint contiguous slice of a big buffer; the 32 lanes split the slice
// (like a GEMV whose 32 lanes each own a column stripe), then simd_sum. fmt 0: read as uint (raw bytes/4), fmt 1: NVFP4-like
// decode (2 nibbles/byte via 16-entry LUT, x fp8-ish block scale every 16 elems) dotted with an activation vector.
constant float LUT[16] = {0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6};
kernel void stream(device const uchar* w [[buffer(0)]], device float* out [[buffer(1)]], constant C& c [[buffer(2)]],
                   device const half* act [[buffer(3)]], device const uchar* scales [[buffer(4)]],
                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; if (sg >= c.n_sg) return;
  uint per_lane = c.bytes_per_sg / sw; uint base = sg * c.bytes_per_sg + lane * per_lane;
  float acc = 0;
  if (c.fmt == 0) { const device uint* p = (const device uint*)(w + base); uint s = 0; for (uint i = 0; i < per_lane / 4; i++) s += p[i]; acc = float(s & 0xffff); }
  else { for (uint b = 0; b < per_lane; b += 8) {            // 8 bytes = 16 fp4 weights = one scale block
      float sc = float(scales[(base + b) >> 3]) * (1.0f/64.0f); float t = 0;
      for (uint j = 0; j < 8; j++) { uchar q = w[base + b + j]; uint k = ((b + j) * 2) & 4095; t += LUT[q & 15] * float(act[k]) + LUT[q >> 4] * float(act[k + 1]); }
      acc += sc * t; } }
  out[sg] = simd_sum(acc);
}
