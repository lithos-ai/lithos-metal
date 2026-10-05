#include <metal_stdlib>
using namespace metal;
struct C { uint n_sg; uint bytes_per_sg; uint mode; uint chunk; };
// mode 0: STRIPED   - lane L streams its own far-apart contiguous stripe of the simd-group's slice (p5 behaviour)
// mode 1: INTERLEAVED - the 32 lanes walk the slice together; lane L takes word (i*32 + L): all lanes share pages/cache lines
// mode 2: BLOCKED   - the slice is cut into `chunk`-byte blocks; within a block lane L owns a contiguous 1/32 sub-range
//                     (like a GEMV where a simd-group sweeps one row-block at a time, lanes = column stripes of that block)
kernel void stream(device const uint* w [[buffer(0)]], device float* out [[buffer(1)]], constant C& c [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  uint sg = gid / sw; if (sg >= c.n_sg) return;
  uint words_per_sg = c.bytes_per_sg / 4; const device uint* p = w + (size_t)sg * words_per_sg; uint s = 0;
  if (c.mode == 0) { uint per = words_per_sg / sw; const device uint* q = p + lane * per; for (uint i = 0; i < per; i++) s += q[i]; }
  else if (c.mode == 1) { uint per = words_per_sg / sw; for (uint i = 0; i < per; i++) s += p[i * sw + lane]; }
  else { uint cw = c.chunk / 4, sub = cw / sw, nblk = words_per_sg / cw;
         for (uint b = 0; b < nblk; b++) { const device uint* q = p + b * cw + lane * sub; for (uint i = 0; i < sub; i++) s += q[i]; } }
  out[sg] = float(simd_sum(s) & 0xffff);
}
