#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
static double now_ms() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil;
  NSString* src = [NSString stringWithContentsOfFile:@"p7_dispatch_overhead.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = [lib newFunctionWithName:@"tiny"]; pd.supportIndirectCommandBuffers = YES;
  id<MTLComputePipelineState> tiny = [d newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
  id<MTLComputePipelineState> fused = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fusedloop"] error:&e];
  if (!tiny || !fused) { printf("pso: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [d newCommandQueue];
  const int N = 1240, W = 5120;   // dispatches per token; vector width
  id<MTLBuffer> x = [d newBufferWithLength:W*4 options:MTLResourceStorageModeShared], y = [d newBufferWithLength:W*4 options:MTLResourceStorageModeShared], z = [d newBufferWithLength:W*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> ab = [d newBufferWithLength:16 options:MTLResourceStorageModeShared]; uint32_t* ap = ab.contents; ap[0] = W; ap[1] = N;
  struct { uint32_t n, k; } A = { W, N };
  MTLSize tg = MTLSizeMake(384,1,1), grid = MTLSizeMake((W+383)/384,1,1);
  printf("Scenario: %d small dispatches per 'token' (width %d), M3 Pro. Times are per token, median of 15.\n\n", N, W);
  double enc[15], wall[15], gpu[15];
  #define MEDIAN(a) ({ for (int i_=0;i_<15;i_++) for (int j_=i_+1;j_<15;j_++) if (a[j_]<a[i_]) { double t_=a[i_]; a[i_]=a[j_]; a[j_]=t_; } a[7]; })
  // (A) conventional: one command buffer, one encoder, N dispatches, serial dispatch type
  for (int mode = 0; mode < 2; mode++) {
    for (int r = 0; r < 15; r++) { double t0 = now_ms();
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> en = mode == 0 ? [cb computeCommandEncoder] : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      for (int i = 0; i < N; i++) { [en setComputePipelineState:tiny]; [en setBuffer:x offset:0 atIndex:0]; [en setBuffer:y offset:0 atIndex:1]; [en setBuffer:z offset:0 atIndex:2]; [en setBytes:&A length:sizeof(A) atIndex:3];
        [en dispatchThreadgroups:grid threadsPerThreadgroup:tg]; if (mode == 1) [en memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
      [en endEncoding]; double t1 = now_ms(); [cb commit]; [cb waitUntilCompleted]; double t2 = now_ms();
      enc[r] = t1 - t0; wall[r] = t2 - t0; gpu[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; }
    double me = MEDIAN(enc), mw = MEDIAN(wall), mg = MEDIAN(gpu);
    printf("(A%d) 1 cmdbuf, 1 encoder (%s): CPU encode %.2f ms (%.2f us/dispatch) | GPU %.2f ms (%.1f us/dispatch) | wall %.2f ms\n", mode, mode==0?"serial":"concurrent+barriers", me, me*1e3/N, mg, mg*1e3/N, mw); }
  // (B) ICB: encode once, replay per token
  MTLIndirectCommandBufferDescriptor* id_ = [MTLIndirectCommandBufferDescriptor new]; id_.commandTypes = MTLIndirectCommandTypeConcurrentDispatch; id_.inheritBuffers = NO; id_.inheritPipelineState = NO; id_.maxKernelBufferBindCount = 4;
  id<MTLIndirectCommandBuffer> icb = [d newIndirectCommandBufferWithDescriptor:id_ maxCommandCount:N options:0];
  double tb0 = now_ms();
  for (int i = 0; i < N; i++) { id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:i]; [c setComputePipelineState:tiny]; [c setKernelBuffer:x offset:0 atIndex:0]; [c setKernelBuffer:y offset:0 atIndex:1]; [c setKernelBuffer:z offset:0 atIndex:2]; [c setKernelBuffer:ab offset:0 atIndex:3];
    [c concurrentDispatchThreadgroups:grid threadsPerThreadgroup:tg]; [c setBarrier]; }
  printf("(B)  ICB one-time build of %d commands: %.2f ms\n", N, now_ms() - tb0);
  for (int r = 0; r < 15; r++) { double t0 = now_ms(); id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en useResource:x usage:MTLResourceUsageRead]; [en useResource:y usage:MTLResourceUsageRead]; [en useResource:z usage:MTLResourceUsageWrite]; [en useResource:ab usage:MTLResourceUsageRead];
    [en executeCommandsInBuffer:icb withRange:NSMakeRange(0, N)]; [en endEncoding]; double t1 = now_ms(); [cb commit]; [cb waitUntilCompleted]; double t2 = now_ms();
    enc[r] = t1 - t0; wall[r] = t2 - t0; gpu[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; }
  { double me = MEDIAN(enc), mw = MEDIAN(wall), mg = MEDIAN(gpu);
    printf("(B)  ICB replay:                       CPU encode %.2f ms (%.3f us/dispatch) | GPU %.2f ms (%.1f us/dispatch) | wall %.2f ms\n", me, me*1e3/N, mg, mg*1e3/N, mw); }
  // (C) megakernel-style: the same N small ops inside ONE dispatch
  for (int r = 0; r < 15; r++) { double t0 = now_ms(); id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:fused]; [en setBuffer:x offset:0 atIndex:0]; [en setBuffer:y offset:0 atIndex:1]; [en setBuffer:z offset:0 atIndex:2]; [en setBytes:&A length:sizeof(A) atIndex:3];
    [en dispatchThreadgroups:grid threadsPerThreadgroup:tg]; [en endEncoding]; double t1 = now_ms(); [cb commit]; [cb waitUntilCompleted]; double t2 = now_ms();
    enc[r] = t1 - t0; wall[r] = t2 - t0; gpu[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; }
  { double me = MEDIAN(enc), mw = MEDIAN(wall), mg = MEDIAN(gpu);
    printf("(C)  ONE dispatch, in-kernel loop:     CPU encode %.3f ms | GPU %.3f ms (%.2f us per in-kernel op) | wall %.2f ms\n", me, mg, mg*1e3/N, mw); }
  // (D) commit+wait round trip of a near-empty command buffer (CPU<->GPU sync cost, e.g. per spec-decode round)
  for (int r = 0; r < 15; r++) { A.k = 1; double t0 = now_ms(); id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:tiny]; [en setBuffer:x offset:0 atIndex:0]; [en setBuffer:y offset:0 atIndex:1]; [en setBuffer:z offset:0 atIndex:2]; [en setBytes:&A length:sizeof(A) atIndex:3];
    [en dispatchThreadgroups:grid threadsPerThreadgroup:tg]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; wall[r] = now_ms() - t0; gpu[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; enc[r] = 0; }
  { double mw = MEDIAN(wall), mg = MEDIAN(gpu); printf("(D)  commit+waitUntilCompleted round trip for 1 tiny dispatch: wall %.3f ms (GPU %.3f ms)  <- cost of each CPU<->GPU sync point\n", mw, mg); }
}}
