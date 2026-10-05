#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p5_bandwidth.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&e]; if (!ps) { printf("pso: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [d newCommandQueue];
  const size_t TOTAL = (size_t)3 << 30;   // 3 GB of "weights" (bigger than any cache)
  id<MTLBuffer> w = [d newBufferWithLength:TOTAL options:MTLResourceStorageModeShared]; memset(w.contents, 0x5a, TOTAL);
  id<MTLBuffer> sc = [d newBufferWithLength:TOTAL/8 options:MTLResourceStorageModeShared]; memset(sc.contents, 0x40, TOTAL/8);
  id<MTLBuffer> act = [d newBufferWithLength:8192 options:MTLResourceStorageModeShared]; uint16_t* a = act.contents; for (int i = 0; i < 4096; i++) a[i] = 0x3c00;
  id<MTLBuffer> out = [d newBufferWithLength:65536*4 options:MTLResourceStorageModeShared];
  struct { uint32_t n_sg, bytes_per_sg, fmt, pad; } C;
  printf("Streaming %.1f GB through the GPU of %s (%d cores). TG = threads per threadgroup, SG = simd-groups used.\n", TOTAL/1e9, d.name.UTF8String, NC);
  for (int fmt = 0; fmt < 2; fmt++) {
    printf("\n=== fmt=%d: %s ===\n", fmt, fmt==0 ? "raw u32 sum (pure bandwidth)" : "NVFP4-like decode+dot (LUT nibble decode, per-16 block scale, half activations)");
    int tpgs[] = {384, 1024, 32};
    for (int ti = 0; ti < 3; ti++) { int tpg = tpgs[ti];
      int sgs[] = {12, 3*NC, 6*NC, 12*NC, 24*NC, 48*NC, 96*NC, 192*NC, 768*NC};   // 12*NC = one 384-thread threadgroup per core
      for (int si = 0; si < 9; si++) { uint32_t nsg = sgs[si]; if (tpg == 32 && (si % 2)) continue;
        C.n_sg = nsg; C.bytes_per_sg = (uint32_t)((TOTAL / nsg) & ~(size_t)(32*8 - 1)); C.fmt = fmt; double best = 1e9;
        for (int rep = 0; rep < 2; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
          [en setComputePipelineState:ps]; [en setBuffer:w offset:0 atIndex:0]; [en setBuffer:out offset:0 atIndex:1]; [en setBytes:&C length:sizeof(C) atIndex:2]; [en setBuffer:act offset:0 atIndex:3]; [en setBuffer:sc offset:0 atIndex:4];
          size_t threads = (size_t)nsg * 32; [en dispatchThreadgroups:MTLSizeMake((threads + tpg - 1) / tpg,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
          double t = cb.GPUEndTime - cb.GPUStartTime; if (t < best) best = t; }
        double gb = (double)C.bytes_per_sg * nsg / 1e9; printf("  TG=%4d SG=%5u (%6.1f MB/SG): %7.1f ms  -> %6.1f GB/s%s\n", tpg, nsg, C.bytes_per_sg/1e6, best*1e3, gb/best, fmt==1 ? "  (weight bytes; +12.5% scale bytes)" : ""); } } }
}}
