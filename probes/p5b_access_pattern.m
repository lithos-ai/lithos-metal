#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p5b_access_pattern.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&e];
  id<MTLCommandQueue> q = [d newCommandQueue];
  const size_t TOTAL = (size_t)3 << 30; id<MTLBuffer> w = [d newBufferWithLength:TOTAL options:MTLResourceStorageModeShared]; memset(w.contents, 0x5a, TOTAL);
  id<MTLBuffer> out = [d newBufferWithLength:65536*4 options:MTLResourceStorageModeShared];
  struct { uint32_t n_sg, bytes_per_sg, mode, chunk; } C;
  printf("Raw streaming bandwidth vs lane access pattern, 3.2 GB, %s. Crew geometry = %d SGs (%d TG x 384).\n", d.name.UTF8String, 12*NC, NC);
  struct { const char* name; uint32_t mode, chunk; } modes[] = { {"striped (lane = far-apart stripe)", 0, 0}, {"interleaved (lanes share cache lines)", 1, 0},
    {"blocked 64KB (lane = 2KB sub-range)", 2, 65536}, {"blocked 1MB (lane = 32KB sub-range)", 2, 1<<20}, {"blocked 16MB (lane = 512KB sub-range)", 2, 1<<24} };
  uint32_t sgs[] = {(uint32_t)(12*NC), (uint32_t)(192*NC)};   // crew geometry (one TG per core) vs 16x more, smaller slices
  for (int m = 0; m < 5; m++) for (int si = 0; si < 2; si++) { uint32_t nsg = sgs[si]; if (modes[m].mode == 2 && TOTAL / nsg < modes[m].chunk) continue;
    C.n_sg = nsg; C.mode = modes[m].mode; C.chunk = modes[m].chunk; size_t per = TOTAL / nsg; if (C.chunk) per = per / C.chunk * C.chunk; else per &= ~(size_t)127; C.bytes_per_sg = (uint32_t)per;
    double best = 1e9; for (int rep = 0; rep < 3; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:ps]; [en setBuffer:w offset:0 atIndex:0]; [en setBuffer:out offset:0 atIndex:1]; [en setBytes:&C length:sizeof(C) atIndex:2];
      [en dispatchThreadgroups:MTLSizeMake(nsg*32/384,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double t = cb.GPUEndTime - cb.GPUStartTime; if (t < best) best = t; }
    printf("  %-40s SG=%5u: %6.1f ms -> %6.1f GB/s\n", modes[m].name, nsg, best*1e3, (double)per*nsg/1e9/best); }
}}
