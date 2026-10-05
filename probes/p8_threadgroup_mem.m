#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p8_threadgroup_mem.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"hot"] error:&e]; if (!ps) { printf("pso: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [d newCommandQueue];
  id<MTLBuffer> act = [d newBufferWithLength:65536 options:MTLResourceStorageModeShared]; uint16_t* a = act.contents; for (int i = 0; i < 32768; i++) a[i] = 0x3c00;
  id<MTLBuffer> out = [d newBufferWithLength:NC*384*4 options:MTLResourceStorageModeShared];
  struct { uint32_t iters, n, mode, pad; } C; const char* names[] = {"device buffer", "threadgroup memory", "thread-private array (256 elems)"};
  printf("Hot small-working-set reads (8,000,000 strided reads/thread), %d TG x 384 threads, %s:\n", NC, d.name.UTF8String);
  uint32_t ns[] = {4096, 16384};
  for (int ni = 0; ni < 2; ni++) for (int m = 0; m < 3; m++) { if (m == 2 && ni == 1) continue; C.iters = 8000000; C.n = ns[ni]; C.mode = m; double best = 1e9;
    for (int rep = 0; rep < 3; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:ps]; [en setBuffer:act offset:0 atIndex:0]; [en setBuffer:out offset:0 atIndex:1]; [en setBytes:&C length:sizeof(C) atIndex:2]; [en setThreadgroupMemoryLength:32768 atIndex:0];
      [en dispatchThreadgroups:MTLSizeMake(NC,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double t = cb.GPUEndTime - cb.GPUStartTime; if (t < best) best = t; if (cb.error) printf("ERR %s\n", cb.error.localizedDescription.UTF8String); }
    printf("  working set %5u halfs (%5.1f KB) from %-34s: %7.2f ms  (%.2f ns/read)\n", C.n, C.n*2/1024.0, names[m], best*1e3, best*1e9/C.iters); }
}}
