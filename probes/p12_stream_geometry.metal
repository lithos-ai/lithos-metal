#include <metal_stdlib>
using namespace metal;
// Streaming-read bandwidth vs (lane access pattern) x (load width) x (loads in flight per lane) x (SIMD-groups per core).
// Each SIMD-group owns one contiguous `bytes_per_sg` slice; blocked modes sweep it `chunk` bytes at a time (the GEMV row-block
// pattern of p5b/D8). Written for the M5 Pro, where p5b showed the crew geometry (12 SIMD-groups per core, lane-contiguous 4 B
// loads) streaming at 71 % of nominal while 192 SIMD-groups per core with interleaved lanes reached 94 %: does the gap close
// with wider loads and more loads in flight per lane (ILP), or only with more SIMD-groups (TLP)?
struct C { uint n_sg; uint bytes_per_sg; uint mode; uint chunk; };
kernel void stream(device const uint* w [[buffer(0)]], device float* out [[buffer(1)]], constant C& c [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; if (sg >= c.n_sg) return;
  const uint words = c.bytes_per_sg / 4; const device uint* p = w + (size_t)sg * words; uint s = 0;
  const device uint4* p4 = (const device uint4*)p; const uint w4 = words / 4;
  const uint cw = c.chunk / 4, sub = cw / sw, nblk = words / cw;            // blocked: words per chunk, words per lane per chunk
  const uint cw4 = cw / 4, sub4 = sub / 4;
  uint4 s4 = 0, t4 = 0, u4 = 0, v4 = 0;
  switch (c.mode) {
    case 0: { uint per = words / sw; const device uint* q = p + lane * per; for (uint i = 0; i < per; i++) s += q[i]; } break;          // striped 4 B
    case 1: { uint per = words / sw; for (uint i = 0; i < per; i++) s += p[i * sw + lane]; } break;                                    // interleaved 4 B
    case 2: for (uint b = 0; b < nblk; b++) { const device uint* q = p + b * cw + lane * sub; for (uint i = 0; i < sub; i++) s += q[i]; } break;   // blocked, lane-contiguous 4 B (p5b mode 2)
    case 3: { uint per = w4 / sw; for (uint i = 0; i < per; i++) s4 += p4[i * sw + lane]; } break;                                   // interleaved 16 B
    case 4: { uint per = w4 / sw; uint i = 0; for (; i + 4 <= per; i += 4) { s4 += p4[i * sw + lane]; t4 += p4[(i + 1) * sw + lane]; u4 += p4[(i + 2) * sw + lane]; v4 += p4[(i + 3) * sw + lane]; }
              for (; i < per; i++) s4 += p4[i * sw + lane]; } break;                                                                 // interleaved 16 B, 4 loads in flight
    case 5: for (uint b = 0; b < nblk; b++) { const device uint4* q = p4 + b * cw4 + lane * sub4; for (uint i = 0; i < sub4; i++) s4 += q[i]; } break;   // blocked, lane-contiguous 16 B
    case 6: for (uint b = 0; b < nblk; b++) { const device uint4* q = p4 + b * cw4 + lane * sub4; uint qtr = sub4 / 4;             // blocked, lane-contiguous 16 B, 4 streams per lane
              for (uint i = 0; i < qtr; i++) { s4 += q[i]; t4 += q[qtr + i]; u4 += q[2 * qtr + i]; v4 += q[3 * qtr + i]; } } break;
    case 7: for (uint b = 0; b < nblk; b++) { const device uint4* q = p4 + b * cw4 + lane * sub4;                                     // blocked, lane-contiguous 16 B, 64 B bursts
              for (uint i = 0; i + 4 <= sub4; i += 4) { s4 += q[i]; t4 += q[i + 1]; u4 += q[i + 2]; v4 += q[i + 3]; } } break;
    case 8: for (uint b = 0; b < nblk; b++) { const device uint4* q = p4 + b * cw4;                                                    // blocked, lane-INTERLEAVED 16 B within the block, 4 in flight
              for (uint i = 0; i + 4 * sw <= cw4; i += 4 * sw) { s4 += q[i + lane]; t4 += q[i + sw + lane]; u4 += q[i + 2 * sw + lane]; v4 += q[i + 3 * sw + lane]; } } break;
  }
  s4 += t4 + u4 + v4; s += s4.x + s4.y + s4.z + s4.w;
  out[sg] = float(simd_sum(s) & 0xffff);
}
// B: ALU-bound work with no memory traffic (same as p11), for the overlap sections (5)/(6) with the saturating streamer as A.
struct SB { uint work; uint p0; uint p1; uint p2; };
kernel void alu(device float* out [[buffer(0)]], constant SB& c [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  float x = float(gid & 1023) * 1e-3f + 1.0f;
  for (uint i = 0; i < c.work; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
  out[gid] = x;
}
