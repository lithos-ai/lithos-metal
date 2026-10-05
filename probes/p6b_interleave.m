#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
#include <mach/mach_time.h>
// P6 showed that ONE long dispatch blocks all other GPU work until it returns. Here the same ~1.2 s of GPU work is issued as
// MANY short dispatches, either in ONE command buffer or split across several, while a second queue submits small dispatches.
// Question: at what granularity can other GPU clients (e.g. the window compositor) get in - dispatch or command buffer?
static double now_ms() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int C = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p4_core_model.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e];
  id<MTLComputePipelineState> al = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"aluwork"] error:&e];
  id<MTLCommandQueue> qA = [d newCommandQueue], qB = [d newCommandQueue];
  id<MTLBuffer> out = [d newBufferWithLength:C*384*4 options:MTLResourceStorageModeShared], out2 = [d newBufferWithLength:4096*4 options:MTLResourceStorageModeShared];
  struct { uint32_t work, active_per_tg, active_tgs, pad; } CL = { 100000, 9999, 9999, 0 }, CS = { 2000, 9999, 9999, 0 };   // long-side dispatch ~1.5 ms each
  struct { const char* name; int n_cb; int disp_per_cb; } cases[] = { {"1 command buffer  x 800 dispatches of ~1.5 ms", 1, 800}, {"8 command buffers x 100 dispatches", 8, 100}, {"80 command buffers x 10 dispatches (~15 ms each)", 80, 10} };
  for (int ci = 0; ci < 3; ci++) {
    NSMutableArray* cbs = [NSMutableArray array]; double t0 = now_ms();
    for (int c = 0; c < cases[ci].n_cb; c++) { id<MTLCommandBuffer> cb = [qA commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:al]; [en setBuffer:out offset:0 atIndex:0];
      for (int i = 0; i < cases[ci].disp_per_cb; i++) { [en setBytes:&CL length:sizeof(CL) atIndex:1]; [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; }
      [en endEncoding]; [cb commit]; [cbs addObject:cb]; }
    id<MTLCommandBuffer> last = cbs.lastObject; usleep(100000);
    double lat[200]; int n = 0;
    while (last.status != MTLCommandBufferStatusCompleted && last.status != MTLCommandBufferStatusError && n < 200) { double s0 = now_ms();
      id<MTLCommandBuffer> cb = [qB commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:al]; [en setBuffer:out2 offset:0 atIndex:0]; [en setBytes:&CS length:sizeof(CS) atIndex:1];
      [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; lat[n++] = now_ms() - s0; usleep(10000); }
    [last waitUntilCompleted]; double total = now_ms() - t0;
    double s = 0, mx = 0; int k = n > 1 ? n - 1 : n; for (int i = 0; i < k; i++) { s += lat[i]; if (lat[i] > mx) mx = lat[i]; }   // drop the last sample (may straddle the end)
    printf("%-52s total %.0f ms | small dispatches on a 2nd queue meanwhile: n=%d  mean %.2f ms  max %.2f ms | last sample (may straddle the end): %.1f ms\n", cases[ci].name, total, k, k ? s/k : 0, mx, n ? lat[n-1] : 0); }
}}
