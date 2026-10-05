#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
// p5b (M5 Pro): the crew geometry's lane-contiguous 64 KB block sweep streams 219 GB/s (71 % of nominal); 16x more SIMD-groups
// with per-word interleaved lanes stream 288 GB/s (94 %). This probe separates the three candidate causes - load width,
// loads in flight per lane (ILP), SIMD-groups per core (TLP) - and re-asks p11 (1) "how many cores saturate the bus?" with
// the best one-threadgroup-per-core kernel. Every number is min-of-3 over 3.2 GB.
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p12_stream_geometry.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&e]; if (!ps) { printf("pso: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> psB = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"alu"] error:&e]; if (!psB) { printf("pso alu: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLBuffer> outB = [d newBufferWithLength:NC*384*4 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q = [d newCommandQueue];
  const size_t TOTAL = (size_t)3 << 30; id<MTLBuffer> w = [d newBufferWithLength:TOTAL options:MTLResourceStorageModeShared]; memset(w.contents, 0x5a, TOTAL);
  id<MTLBuffer> out = [d newBufferWithLength:(1<<20)*4 options:MTLResourceStorageModeShared];
  const char* names[] = { "striped 4B", "interleaved 4B", "blocked lane-contig 4B (p5b/D8)", "interleaved 16B", "interleaved 16B x4 in flight",
                          "blocked lane-contig 16B", "blocked lane-contig 16B, 4 streams/lane", "blocked lane-contig 16B, 64B bursts", "blocked lane-interleaved 16B x4" };
  // run one configuration: returns GB/s (min-of-3)
  double (^run)(uint32_t, uint32_t, uint32_t, int, size_t) = ^double(uint32_t mode, uint32_t nsg, uint32_t chunk, int tpg, size_t bytes) {
    size_t per = bytes / nsg; per = chunk ? per / chunk * chunk : per / 2048 * 2048; if (per > 0xFFFFFFFFu || per == 0) return 0;
    struct { uint32_t n_sg, bytes_per_sg, mode, chunk; } C;
    C.n_sg = nsg; C.bytes_per_sg = (uint32_t)per; C.mode = mode; C.chunk = chunk; double best = 1e9;
    for (int rep = 0; rep < 3; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:ps]; [en setBuffer:w offset:0 atIndex:0]; [en setBuffer:out offset:0 atIndex:1]; [en setBytes:&C length:sizeof(C) atIndex:2];
      [en dispatchThreadgroups:MTLSizeMake(((size_t)nsg * 32 + tpg - 1) / tpg,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double t = cb.GPUEndTime - cb.GPUStartTime; if (t < best) best = t; }
    return (double)per * nsg / 1e9 / best; };
  printf("Streaming 3.2 GB on %s (%d cores; crew geometry = %d SIMD-groups). Cells: GB/s, min-of-3. Blocked modes use 64 KB blocks.\n", d.name.UTF8String, NC, 12*NC);
  int mult[] = {12, 24, 48, 96, 192, 384};   // SIMD-groups per core; 12 = one 384-thread threadgroup per core
  printf("\n=== (1) pattern x load width x ILP x SIMD-groups per core, threadgroups of 384 ===\n%-44s", "SIMD-groups per core:"); for (int mi = 0; mi < 6; mi++) printf(" %6d", mult[mi]); printf("\n");
  for (uint32_t m = 0; m < 9; m++) { printf("  %-42s", names[m]); for (int mi = 0; mi < 6; mi++) printf(" %6.1f", run(m, mult[mi] * NC, m >= 2 && m != 3 && m != 4 ? 65536 : 0, 384, TOTAL)); printf("\n"); }
  printf("\n=== (2) threadgroup size at 96 SIMD-groups per core (8 x 384 / 3 x 1024 / 12 x 256 per core) ===\n");
  int tpgs[] = {256, 384, 512, 1024}; for (uint32_t m = 2; m < 9; m++) { printf("  %-42s", names[m]); for (int ti = 0; ti < 4; ti++) printf(" TG=%4d: %6.1f", tpgs[ti], run(m, 96 * NC, m >= 2 && m != 3 && m != 4 ? 65536 : 0, tpgs[ti], TOTAL)); printf("\n"); }
  printf("\n=== (3) block size for the blocked modes, at 12 and 48 SIMD-groups per core ===\n");
  uint32_t chunks[] = {16384, 65536, 262144, 1048576};
  for (uint32_t m = 5; m < 9; m++) for (int mi = 0; mi < 2; mi++) { int sgpc = mi ? 48 : 12; printf("  %-42s %3d SG/core:", names[m], sgpc); for (int ci = 0; ci < 4; ci++) printf("  %4u KB: %6.1f", chunks[ci] / 1024, run(m, sgpc * NC, chunks[ci], 384, TOTAL)); printf("\n"); }
  printf("\n=== (4) p11 (1) again: cores needed to saturate the bus with ONE 384-thread threadgroup per core, best ILP kernels ===\n");
  int Gs[] = {1, 2, 4, NC/3, NC/2, (2*NC)/3, (5*NC)/6, NC}; int nG = tidy(Gs, 8); uint32_t ms[] = {2, 6, 7, 8};
  printf("  %-42s", "cores:"); for (int i = 0; i < nG; i++) printf(" %6d", Gs[i]); printf("\n");
  for (int k = 0; k < 4; k++) { printf("  %-42s", names[ms[k]]); for (int i = 0; i < nG; i++) printf(" %6.1f", run(ms[k], 12 * Gs[i], 65536, 384, TOTAL)); printf("\n"); }
  // ---- (5)/(6): p11 (3b) and (4) again with the SATURATING streamer as the bus-bound op A (mode 8, crew geometry, 64 KB blocks)
  typedef struct { uint32_t n_sg, bytes_per_sg, mode, chunk; } SA; typedef struct { uint32_t work, p0, p1, p2; } SB;
  #define ENC_A(en, off, bytes, outoff) do { SA a = { (uint32_t)(12*NC), (uint32_t)((((bytes) / (12*NC)) / 65536) * 65536), 8, 65536 }; \
      [en setComputePipelineState:ps]; [en setBuffer:w offset:(off) atIndex:0]; [en setBuffer:out offset:(outoff) atIndex:1]; [en setBytes:&a length:sizeof(a) atIndex:2]; \
      [en dispatchThreadgroups:MTLSizeMake(NC,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; } while (0)
  #define ENC_B(en, wk) do { SB b = { (uint32_t)(wk), 0, 0, 0 }; [en setComputePipelineState:psB]; [en setBuffer:outB offset:0 atIndex:0]; [en setBytes:&b length:sizeof(b) atIndex:1]; \
      [en dispatchThreadgroups:MTLSizeMake(NC,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; } while (0)
  #define RUNMS(CONCURRENT, BODY) ({ double best = 1e9; for (int rep = 0; rep < 4; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; \
      id<MTLComputeCommandEncoder> en = (CONCURRENT) ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder]; \
      BODY; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; if (t < best) best = t; } best; })
  const uint32_t WK = 170000;   // p11's B: ~10 % of p11's A on an M3 Pro
  double tA = RUNMS(0, ENC_A(en, 0, TOTAL, 0));
  printf("\n=== (5) p11 (3b) again: ALU-bound op B hidden inside the SATURATING bus-bound op A (mode 8, crew geometry), no barrier, both on all cores ===\n");
  printf("  A alone: %.2f ms (%.1f GB/s)\n", tA, TOTAL/1e9/(tA/1e3));
  for (int m = 1; m <= 8; m *= 2) { double tb = RUNMS(0, ENC_B(en, WK*m)); double ser = RUNMS(0, { ENC_A(en, 0, TOTAL, 0); ENC_B(en, WK*m); }); double ov = RUNMS(1, { ENC_A(en, 0, TOTAL, 0); ENC_B(en, WK*m); }); double ov2 = RUNMS(1, { ENC_B(en, WK*m); ENC_A(en, 0, TOTAL, 0); });
    printf("  B = %dx work (alone %5.2f ms = %3.0f%% of A): serial %6.2f ms | no barrier A,B %6.2f ms (hidden %3.0f%% of B) | B,A %6.2f ms (hidden %3.0f%%)\n", m, tb, 100*tb/tA, ser, ov, 100*(ser-ov)/tb, ov2, 100*(ser-ov2)/tb); }
  printf("\n=== (6) p11 (4) again: two SATURATING bus-bound ops, 1.6 GB each ===\n");
  double h = RUNMS(0, { ENC_A(en, 0, TOTAL/2, 0); ENC_A(en, TOTAL/2, TOTAL/2, 65536); });
  printf("  serial    : A1 then A2, all cores each                  %6.2f ms -> %6.1f GB/s\n", h, TOTAL/1e9/(h/1e3));
  double hc = RUNMS(1, { ENC_A(en, 0, TOTAL/2, 0); ENC_A(en, TOTAL/2, TOTAL/2, 65536); });
  printf("  no barrier: A1 || A2, all cores each (2x oversubscribed) %6.2f ms -> %6.1f GB/s\n", hc, TOTAL/1e9/(hc/1e3));
}}
