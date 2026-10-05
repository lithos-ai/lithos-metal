#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int C = gpu_cores(); const int NSG = 12*C;
  NSString* src = [NSString stringWithContentsOfFile:@"p9_clock_warp.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"clocked"] error:&e];
  id<MTLCommandQueue> q = [d newCommandQueue];
  id<MTLBuffer> tick = [d newBufferWithLength:4 options:MTLResourceStorageModeShared], done = [d newBufferWithLength:4 options:MTLResourceStorageModeShared], st = [d newBufferWithLength:NSG*2*4 options:MTLResourceStorageModeShared]; id<MTLBuffer> sink = [d newBufferWithLength:C*384*4 options:MTLResourceStorageModeShared];
  struct { uint32_t work, n_workers_sg, max_ticks, pad; } P = { 1000000, (uint32_t)(NSG-1), 40000000, 0 };   // class k takes ~15.2*k ms
  printf("Clock-warp probe on %s: 1 ticking simd-group + %d worker simd-groups (%d TG x 384). Workload classes 1x..4x.\n", d.name.UTF8String, NSG-1, C);
  for (int rep = 0; rep < 4; rep++) { *(uint32_t*)tick.contents = 0; *(uint32_t*)done.contents = 0;
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:ps]; [en setBuffer:tick offset:0 atIndex:0]; [en setBuffer:done offset:0 atIndex:1]; [en setBuffer:st offset:0 atIndex:2]; [en setBytes:&P length:sizeof(P) atIndex:3]; [en setBuffer:sink offset:0 atIndex:4];
    [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; uint32_t total = *(uint32_t*)tick.contents; uint32_t* s = st.contents;
    double ns_per_tick = gpu_ms * 1e6 / total; double sum[4] = {0}, mn[4] = {1e18,1e18,1e18,1e18}, mx[4] = {0}; int cnt[4] = {0};
    for (int g = 1; g < NSG; g++) { int k = g & 3; double dt = (double)(s[g*2+1] - s[g*2]) * ns_per_tick / 1e6; sum[k] += dt; cnt[k]++; if (dt < mn[k]) mn[k] = dt; if (dt > mx[k]) mx[k] = dt; }
    printf(" run %d: gpu %.2f ms, %u ticks -> %.1f ns/tick | measured per class (ms) mean[min..max]:", rep, gpu_ms, total, ns_per_tick);
    for (int k = 0; k < 4; k++) { int kk = (k + 1) & 3; printf("  %dx: %.2f[%.2f..%.2f]", kk + 1, sum[kk]/cnt[kk], mn[kk], mx[kk]); } printf("\n"); }
  printf(" expected: 1x=15.2  2x=30.4  3x=45.7  4x=60.9 ms (from the ALU probe)\n");
}}
