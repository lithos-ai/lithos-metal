#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
#include <mach/mach_time.h>
#include <pthread.h>
static double now_ms() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int C = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p4_core_model.metal" encoding:NSUTF8StringEncoding error:&e];   // reuse aluwork
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e];
  id<MTLComputePipelineState> al = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"aluwork"] error:&e];
  id<MTLCommandQueue> qLong = [d newCommandQueue], qSmall = [d newCommandQueue];
  id<MTLBuffer> out = [d newBufferWithLength:1024*1024*4 options:MTLResourceStorageModeShared], out2 = [d newBufferWithLength:4096*4 options:MTLResourceStorageModeShared];
  struct { uint32_t work, active_per_tg, active_tgs, pad; } CL, CS = { 2000, 9999, 9999, 0 };   // small: ~30us of work on 1 TG
  // baseline latency of the small dispatch on an idle GPU
  double base[50]; for (int i = 0; i < 50; i++) { double t0 = now_ms(); id<MTLCommandBuffer> cb = [qSmall commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:al]; [en setBuffer:out2 offset:0 atIndex:0]; [en setBytes:&CS length:sizeof(CS) atIndex:1]; [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; base[i] = now_ms() - t0; }
  double bsum = 0, bmax = 0; for (int i = 5; i < 50; i++) { bsum += base[i]; if (base[i] > bmax) bmax = base[i]; }
  printf("Small dispatch on idle GPU: mean %.3f ms, max %.3f ms\n\n", bsum/45, bmax);
  int Gs[] = {C, C-2, (2*C)/3};
  for (int gi = 0; gi < 3; gi++) { int G = Gs[gi];
    CL.work = 90000000; CL.active_per_tg = 9999; CL.active_tgs = 9999;   // ~1.4 s per simd-group at 15.2 ns/iter
    id<MTLCommandBuffer> lcb = [qLong commandBuffer]; id<MTLComputeCommandEncoder> len = [lcb computeCommandEncoder];
    [len setComputePipelineState:al]; [len setBuffer:out offset:0 atIndex:0]; [len setBytes:&CL length:sizeof(CL) atIndex:1]; [len dispatchThreadgroups:MTLSizeMake(G,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [len endEncoding];
    double tl0 = now_ms(); [lcb commit];
    usleep(150000);   // let the long dispatch get going
    double lat[400]; double when[400]; int n = 0;
    while (lcb.status != MTLCommandBufferStatusCompleted && lcb.status != MTLCommandBufferStatusError && n < 400) {
      double t0 = now_ms(); id<MTLCommandBuffer> cb = [qSmall commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:al]; [en setBuffer:out2 offset:0 atIndex:0]; [en setBytes:&CS length:sizeof(CS) atIndex:1]; [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double t1 = now_ms(); when[n] = t0 - tl0; lat[n] = t1 - t0; n++; usleep(20000); }
    [lcb waitUntilCompleted]; double ltot = now_ms() - tl0; double lgpu = (lcb.GPUEndTime - lcb.GPUStartTime) * 1e3;
    double s = 0, mx = 0; int during = 0; for (int i = 0; i < n; i++) if (when[i] + lat[i] <= ltot + 1) { s += lat[i]; if (lat[i] > mx) mx = lat[i]; during++; }
    printf("LONG dispatch on G=%2d threadgroups x 384 (of %d cores): long GPU time %.0f ms%s\n", G, C, lgpu, lcb.error ? "  [ERROR]" : "");
    printf("   %d small dispatches submitted while it ran; %d completed before it ended: mean latency %.3f ms, max %.3f ms\n", n, during, during ? s/during : 0, mx);
    printf("   first 6 latencies (ms): "); for (int i = 0; i < n && i < 6; i++) printf("%.2f ", lat[i]); printf("\n\n"); }
}}
