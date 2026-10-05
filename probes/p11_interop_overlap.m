#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
// Can two ops overlap on an Apple GPU? MPK V2 overlaps op k+1's weight streaming with op k's compute. The only Metal
// mechanism for inter-op overlap is two dispatches with NO barrier between them in a concurrent encoder (or ICB).
// (1) how many cores does it take to saturate the memory bus?  (2) do un-barriered dispatches really run concurrently,
// (3) and does a bus-bound op overlap for free with an ALU-bound op / with another bus-bound op?
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int C = gpu_cores(); const int g56 = (5*C)/6, g16 = C - g56, g23 = (2*C)/3, g13 = C - g23, gh = C/2;
  NSString* src = [NSString stringWithContentsOfFile:@"p11_interop_overlap.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> psA = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&e], psB = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"alu"] error:&e];
  id<MTLCommandQueue> q = [d newCommandQueue];
  const size_t TOTAL = (size_t)3 << 30, CHUNK = 65536;
  id<MTLBuffer> w = [d newBufferWithLength:TOTAL options:MTLResourceStorageModeShared]; memset(w.contents, 0x5a, TOTAL);
  id<MTLBuffer> outA = [d newBufferWithLength:4096*4 options:MTLResourceStorageModeShared], outA2 = [d newBufferWithLength:4096*4 options:MTLResourceStorageModeShared], outB = [d newBufferWithLength:C*384*4 options:MTLResourceStorageModeShared];
  typedef struct { uint32_t n_sg, bytes_per_sg, chunk, pad; } SA; typedef struct { uint32_t work, p0, p1, p2; } SB;
  // encode helpers ---------------------------------------------------------------------------------------------------
  #define ENC_A(en, G, off, bytes, outbuf) do { SA a = { (uint32_t)((G)*12), (uint32_t)((((bytes) / ((G)*12)) / CHUNK) * CHUNK), (uint32_t)CHUNK, 0 }; \
      [en setComputePipelineState:psA]; [en setBuffer:w offset:(off) atIndex:0]; [en setBuffer:(outbuf) offset:0 atIndex:1]; [en setBytes:&a length:sizeof(a) atIndex:2]; \
      [en dispatchThreadgroups:MTLSizeMake((G),1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; } while (0)
  #define ENC_B(en, G, wk) do { SB b = { (uint32_t)(wk), 0, 0, 0 };  /* work per thread */ [en setComputePipelineState:psB]; [en setBuffer:outB offset:0 atIndex:0]; [en setBytes:&b length:sizeof(b) atIndex:1]; \
      [en dispatchThreadgroups:MTLSizeMake((G),1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; } while (0)
  #define RUN(CONCURRENT, BODY) ({ double best = 1e9; for (int rep = 0; rep < 4; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; \
      id<MTLComputeCommandEncoder> en = (CONCURRENT) ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder]; \
      BODY; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; if (t < best) best = t; } best; })
  printf("device: %s, %d GPU cores\n=== (1) memory-bus saturation: stream 3.2 GB (64 KB block sweep) on G of the %d cores ===\n", d.name.UTF8String, C, C);
  int Gs[] = {1, 2, 4, C/3, gh, g23, g56, C}; double tA[257] = {0}; int nG = tidy(Gs, 8);
  for (int i = 0; i < nG; i++) { int G = Gs[i]; double t = RUN(0, ENC_A(en, G, 0, TOTAL, outA)); tA[G] = t; printf("  G=%2d cores: %6.1f ms -> %6.1f GB/s  (%.1f GB/s per core)\n", G, t, TOTAL/1e9/(t/1e3), TOTAL/1e9/(t/1e3)/G); }
  const uint32_t WK = 170000;   // B on all cores is ~10% of A on an M3 Pro, like a token-mixer stage. Total ALU work is held constant across core counts.
  double tB18 = RUN(0, ENC_B(en, C, WK)), tB6 = RUN(0, ENC_B(en, g13, (uint64_t)WK*C/g13)), tB3 = RUN(0, ENC_B(en, g16, (uint64_t)WK*C/g16));
  printf("\n=== (2) the ALU-bound op B alone (same total work): %d cores %.2f ms | %d cores %.2f ms | %d cores %.2f ms ===\n", C, tB18, g13, tB6, g16, tB3);
  printf("\n=== (3) op A (bus-bound) and op B (ALU-bound) in ONE encoder ===\n");
  double s1 = RUN(0, { ENC_A(en, C, 0, TOTAL, outA); ENC_B(en, C, WK); });
  printf("  serial    : A on all cores, then B on all cores (the design today)      %6.2f ms   (A %.2f + B %.2f)\n", s1, tA[C], tB18);
  double c1 = RUN(1, { ENC_A(en, C, 0, TOTAL, outA); ENC_B(en, C, WK); });
  printf("  no barrier: A on all cores || B on all cores (both want every core)     %6.2f ms\n", c1);
  double c2 = RUN(1, { ENC_A(en, g56, 0, TOTAL, outA); ENC_B(en, g16, (uint64_t)WK*C/g16); });
  printf("  no barrier: A on %2d cores || B on %2d cores                              %6.2f ms   (A alone: %.2f, B alone: %.2f)\n", g56, g16, c2, tA[g56], tB3);
  double c3 = RUN(1, { ENC_A(en, g23, 0, TOTAL, outA); ENC_B(en, g13, (uint64_t)WK*C/g13); });
  printf("  no barrier: A on %2d cores || B on %2d cores                              %6.2f ms   (A alone: %.2f, B alone: %.2f)\n", g23, g13, c3, tA[g23], tB6);
  double c4 = RUN(1, { ENC_B(en, g13, (uint64_t)WK*C/g13); ENC_A(en, g23, 0, TOTAL, outA); });
  printf("  no barrier: B on %2d cores || A on %2d cores (encoded in the other order) %6.2f ms\n", g13, g23, c4);
  printf("\n=== (3b) robustness at full crew geometry: encode order, and how much ALU work can hide ===\n");
  double o2 = RUN(1, { ENC_B(en, C, WK); ENC_A(en, C, 0, TOTAL, outA); });
  printf("  no barrier: B on all cores || A on all cores (B encoded first)          %6.2f ms\n", o2);
  for (int m = 2; m <= 8; m *= 2) { double tb = RUN(0, ENC_B(en, C, WK*m)); double ser = RUN(0, { ENC_A(en, C, 0, TOTAL, outA); ENC_B(en, C, WK*m); }); double ov = RUN(1, { ENC_A(en, C, 0, TOTAL, outA); ENC_B(en, C, WK*m); });
    printf("  B = %dx work (alone %.2f ms = %2.0f%% of A): serial %6.2f ms | no barrier %6.2f ms  -> hidden %.0f%% of B\n", m, tb, 100*tb/tA[C], ser, ov, 100*(ser-ov)/tb); }
  printf("\n=== (4) two BUS-bound ops: can overlapping them create bandwidth? ===\n");
  double h = RUN(0, { ENC_A(en, C, 0, TOTAL/2, outA); ENC_A(en, C, TOTAL/2, TOTAL/2, outA2); });
  printf("  serial    : A1 (1.6 GB) on all cores, then A2 (1.6 GB) on all cores     %6.2f ms -> %6.1f GB/s\n", h, TOTAL/1e9/(h/1e3));
  double hc = RUN(1, { ENC_A(en, gh, 0, TOTAL/2, outA); ENC_A(en, gh, TOTAL/2, TOTAL/2, outA2); });
  printf("  no barrier: A1 on half the cores || A2 on the other half                %6.2f ms -> %6.1f GB/s\n", hc, TOTAL/1e9/(hc/1e3));
  double hc2 = RUN(1, { ENC_A(en, C, 0, TOTAL/2, outA); ENC_A(en, C, TOTAL/2, TOTAL/2, outA2); });
  printf("  no barrier: A1 on all cores || A2 on all cores (2x oversubscribed)      %6.2f ms -> %6.1f GB/s\n", hc2, TOTAL/1e9/(hc2/1e3));
}}
