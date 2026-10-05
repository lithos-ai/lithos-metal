#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p4_core_model.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> al = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"aluwork"] error:&e];
  id<MTLCommandQueue> q = [d newCommandQueue];
  id<MTLBuffer> out = [d newBufferWithLength:1024*1024*4 options:MTLResourceStorageModeShared];
  struct { uint32_t work, active_per_tg, active_tgs, pad; } C;
  #define RUN(G, TPG, APT, ATG) ({ C.work = 1000000; C.active_per_tg = (APT); C.active_tgs = (ATG); double best = 1e9; \
    for (int rep = 0; rep < 3; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; \
      [en setComputePipelineState:al]; [en setBuffer:out offset:0 atIndex:0]; [en setBytes:&C length:sizeof(C) atIndex:1]; \
      [en dispatchThreadgroups:MTLSizeMake((G),1,1) threadsPerThreadgroup:MTLSizeMake((TPG),1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; \
      double t = (cb.GPUEndTime-cb.GPUStartTime)*1e3; if (t < best) best = t; } best; })
  double base = RUN(1, 1024, 1, 1);
  printf("device: %s, %d GPU cores\nbaseline: 1 simd-group alone = %.2f ms\n\n", d.name.UTF8String, NC, base);
  printf("=== (1) ONE threadgroup of 1024 threads; k busy simd-groups inside it ===\n");
  for (int k = 1; k <= 32; k += (k < 16 ? 1 : 4)) { double t = RUN(1, 1024, k, 1); printf("  k=%2d busy simd-groups: %7.2f ms  x%.2f   (effective parallel groups = %.1f)\n", k, t, t/base, k*base/t); }
  printf("\n=== (2) G threadgroups, ALL simd-groups busy; vary threadgroup size (does 1 TG == 1 core?) ===\n");
  int tpgs[] = {32, 64, 128, 256, 384, 512, 1024};
  int Gs[12] = {1, 2, 4, NC/2, (2*NC)/3, NC-2, NC, NC+2, (4*NC)/3, 2*NC, 3*NC, 4*NC};   // the step is expected right after G = C (one threadgroup per core)
  printf("  %-8s", "TG size"); for (int j = 0; j < 12; j++) printf(" G=%-5d", Gs[j]); printf("   (cells: ms; baseline %.1f)\n", base);
  for (int i = 0; i < 7; i++) { printf("  %-8d", tpgs[i]); for (int j = 0; j < 12; j++) { double t = RUN(Gs[j], tpgs[i], 9999, 9999); printf(" %7.1f", t); } printf("\n"); }
  printf("\n=== (3) same, expressed as total throughput in full-speed simd-group equivalents ===\n");
  printf("  %-8s", "TG size"); for (int j = 0; j < 12; j++) printf(" G=%-5d", Gs[j]); printf("\n");
  for (int i = 0; i < 7; i++) { printf("  %-8d", tpgs[i]); for (int j = 0; j < 12; j++) { double t = RUN(Gs[j], tpgs[i], 9999, 9999); printf(" %7.1f", (double)Gs[j]*(tpgs[i]/32)*base/t); } printf("\n"); }
}}
