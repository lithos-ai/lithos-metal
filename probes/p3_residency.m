#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, NSString* n) { NSError* e=nil; id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:n] error:&e]; if(!p){printf("PSO %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(1);} return p; }
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil;
  NSString* src = [NSString stringWithContentsOfFile:@"p3_residency.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [d newCommandQueue];
  id<MTLComputePipelineState> rs = pso(d, lib, @"residency"), al = pso(d, lib, @"aluwork");
  id<MTLBuffer> ctr = [d newBufferWithLength:4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> res = [d newBufferWithLength:8192*2*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> out = [d newBufferWithLength:8192*32*4 options:MTLResourceStorageModeShared];
  struct { uint32_t S, max_spin, work, active; } C;
  printf("=== (a) FULL RESIDENCY: all S simd-groups (x32 lanes, all spinning) must be alive at once; max_spin=60000 ===\n");
  uint32_t Ss[] = {32, 128, 512, 1024, 1536, 2048};   // capped: beyond the in-flight limit each extra round costs a full spin timeout, and the GPU is NOT preemptible
  for (int i = 0; i < 6; i++) { for (int tpg = 1024; tpg >= 32; tpg = (tpg == 1024 ? 32 : 0)) {
    C.S = Ss[i]; C.max_spin = 60000; C.work = 0; C.active = 0; if (C.S*32 < (uint32_t)tpg) continue;
    *(uint32_t*)ctr.contents = 0; memset(res.contents, 0, res.length);
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:rs]; [en setBuffer:ctr offset:0 atIndex:0]; [en setBuffer:res offset:0 atIndex:1]; [en setBytes:&C length:sizeof(C) atIndex:2];
    [en dispatchThreadgroups:MTLSizeMake(C.S*32/tpg,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    uint32_t* r = (uint32_t*)res.contents; uint32_t ok=0,to=0,maxs=0; for (uint32_t g=0; g<C.S; g++) { if (r[g*2]==0xFFFFFFFFu) to++; else { ok++; if (r[g*2]>maxs) maxs=r[g*2]; } }
    printf("S=%5u simd-groups (%6u threads, TG=%4d): ok=%5u timeout=%5u max_spins_ok=%7u gpu=%8.2f ms %s\n", C.S, C.S*32, tpg, ok, to, maxs, (cb.GPUEndTime-cb.GPUStartTime)*1e3, cb.error?"ERROR":""); if (tpg == 32) break; } }
  printf("\n=== (b) PHYSICAL CONCURRENCY: fixed ALU work per active simd-group (work=3,000,000 iters), dispatch of 4096 simd-groups, TG=1024 ===\n");
  uint32_t As[] = {1, 8, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
  double base = 0;
  const int NC = gpu_cores();
  for (int i = 0; i < 15; i++) { if ((int)As[i] > 200 * NC) continue;   /* keeps the longest dispatch under ~1 s on any chip */ C.S = 4096; C.max_spin = 0; C.work = 3000000; C.active = As[i];
    double best = 1e9; for (int rep = 0; rep < 2; rep++) {
      id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:al]; [en setBuffer:out offset:0 atIndex:0]; [en setBytes:&C length:sizeof(C) atIndex:2];
      [en dispatchThreadgroups:MTLSizeMake(4096*32/1024,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double t = (cb.GPUEndTime-cb.GPUStartTime)*1e3; if (t < best) best = t; }
    if (i == 0) base = best;
    printf("active simd-groups=%5u (%6u threads): gpu=%8.2f ms  (x%.2f vs 1 group; throughput = %.0f group-equivalents)\n", As[i], As[i]*32, best, best/base, As[i]*base/best); }
}}
