#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <unistd.h>
#include "probe_common.h"
// Probe: host handoffs between GPU passes through MTLSharedEvent, the mechanism of the host-step runtime. A pass
// signals an event and the CPU observes it and does its work; the GPU work that needs the CPU's result either waits
// in the same command buffer for a second event the CPU signals (encodeWaitForEvent), or sits in a second command
// buffer the CPU commits once its work is done. Passes are ordered by one MTLFence (ICB-referenced resources made
// resident through a residency set are not hazard-tracked).
//   1. visibility: a GPU write before the signal is seen by the CPU once it observes the signal; a CPU write is seen
//      by the GPU work after it (1,000 steps, ICB ranges): a. the in-buffer wait, two command buffers in flight;
//      b. the second command buffer, committed after the CPU write
//   2. the GPU idle gap across one handoff with no CPU work (stage-boundary timestamps): the in-buffer wait with a
//      spinning CPU and with waitUntilSignaledValue, the second command buffer encoded before the signal (the host
//      only commits it) and encoded after it, and a plain pass boundary. Encoded after the signal is a lower bound on
//      the host-step runtime's handoff, which also reads StepState, runs the task and encodes a whole segment of
//      passes; tools/bench/host_handoff.py measures that handoff itself
//   3. a command buffer parked at an in-buffer wait after its first pass while the CPU sleeps 2 to 6 s: does it
//      finish without an error (the GPU watchdog), and do another queue's command buffers keep completing meanwhile?
//   4. the same with the wait as the command buffer's first command (no pass has run before it)
//   5. a command buffer whose last command is a signal: how long after its status reads Completed is the signal
//      visible to the CPU (the runtime's grace before it fails a point whose command buffer finished without it)
static double now_ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }
static int cmp_d(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, NSString* n, BOOL icb) {
  NSError* e = nil; MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [l newFunctionWithName:n]; pd.supportIndirectCommandBuffers = icb;
  id<MTLComputePipelineState> p = [d newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
  if (!p) { printf("PSO %s: %s\n", n.UTF8String, e.localizedDescription.UTF8String); exit(1); }
  return p;
}
// bounded host wait for a GPU signal: spin (mode 0) or waitUntilSignaledValue (mode 1); 0 on timeout
static int host_wait(id<MTLSharedEvent> ev, uint64_t v, int mode, double timeout_ms) {
  if (mode == 1) return [ev waitUntilSignaledValue:v timeoutMS:(uint64_t)timeout_ms] ? 1 : 0;
  double t0 = now_ms();
  while (ev.signaledValue < v) if (now_ms() - t0 > timeout_ms) return 0;
  return 1;
}
// part 2's pass B: one small dispatch behind the fence, sampled at stage boundaries 2 and 3
static void encode_pass_b(id<MTLCommandBuffer> cb, id<MTLCounterSampleBuffer> sb, id<MTLFence> fence,
                          id<MTLComputePipelineState> spin, id<MTLBuffer> work) {
  uint32_t one = 1;
  MTLComputePassDescriptor* p2 = [MTLComputePassDescriptor computePassDescriptor];
  p2.sampleBufferAttachments[0].sampleBuffer = sb; p2.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2; p2.sampleBufferAttachments[0].endOfEncoderSampleIndex = 3;
  id<MTLComputeCommandEncoder> en = [cb computeCommandEncoderWithDescriptor:p2];
  [en waitForFence:fence]; [en setComputePipelineState:spin]; [en setBuffer:work offset:0 atIndex:0]; [en setBytes:&one length:4 atIndex:1];
  [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [en updateFence:fence]; [en endEncoding];
}
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil;
  NSString* src = [NSString stringWithContentsOfFile:@"p16_host_handoff.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e];
  if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  const int cores = gpu_cores();
  id<MTLComputePipelineState> bump = pso(d, lib, @"bump", YES), spin = pso(d, lib, @"spin_work", NO);
  id<MTLBuffer> a = [d newBufferWithLength:64 options:MTLResourceStorageModeShared];
  id<MTLBuffer> logb = [d newBufferWithLength:4096 * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> work = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
  memset(a.contents, 0, 64); memset(logb.contents, 0xff, 4096 * 4); memset(work.contents, 0, 4096);
  MTLIndirectCommandBufferDescriptor* desc = [MTLIndirectCommandBufferDescriptor new];
  desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch; desc.inheritBuffers = NO; desc.inheritPipelineState = NO;
  desc.maxKernelBufferBindCount = 2;
  id<MTLIndirectCommandBuffer> icb = [d newIndirectCommandBufferWithDescriptor:desc maxCommandCount:3 options:0];
  for (int c = 0; c < 3; c++) {
    id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:c];
    [cmd setComputePipelineState:bump]; [cmd setKernelBuffer:a offset:0 atIndex:0]; [cmd setKernelBuffer:logb offset:0 atIndex:1];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [cmd setBarrier];
  }
  MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new]; rd.initialCapacity = 8;
  id<MTLResidencySet> rs = [d newResidencySetWithDescriptor:rd error:&e];
  [rs addAllocation:a]; [rs addAllocation:logb]; [rs addAllocation:work]; [rs commit]; [rs requestResidency];
  id<MTLCommandQueue> q = [d newCommandQueue]; [q addResidencySet:rs];
  id<MTLSharedEvent> gpu_ev = [d newSharedEvent], host_ev = [d newSharedEvent];
  id<MTLFence> fence = [d newFence];
  uint64_t v = 0;
  printf("gpu cores: %d\n", cores);

  // ---- 1. visibility, two command buffers in flight -----------------------------------------------------------
  {
    const int N = 1000; int seen_bad = 0, log_bad = 0, timeouts = 0; BOOL fence_live = NO;
    volatile uint32_t* av = (volatile uint32_t*)a.contents;
    id<MTLCommandBuffer> inflight[2] = {nil, nil}; uint64_t vals[2] = {0, 0};
    for (int it = 0; it <= N; it++) {
      if (it < N) {   // encode step `it`: pass [0,1), signal, wait, pass [1,3)
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        if (fence_live) [en waitForFence:fence];
        [en executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)]; [en updateFence:fence]; [en endEncoding]; fence_live = YES;
        [cb encodeSignalEvent:gpu_ev value:v + 1]; [cb encodeWaitForEvent:host_ev value:v + 1];
        en = [cb computeCommandEncoder]; [en waitForFence:fence];
        [en executeCommandsInBuffer:icb withRange:NSMakeRange(1, 2)]; [en updateFence:fence]; [en endEncoding];
        [cb commit];
        inflight[it & 1] = cb; vals[it & 1] = ++v;
      }
      if (it >= 1) {  // serve step it-1's host point, then retire its command buffer
        int s = (it - 1) & 1;
        if (!host_wait(gpu_ev, vals[s], 0, 2000)) { timeouts++; host_ev.signaledValue = v; break; }
        if (av[0] != (uint32_t)(3 * (it - 1) + 1)) seen_bad++;   // the first pass of step it-1 bumped a[0]
        av[1] = 100000 + (it - 1);                              // the CPU write the step's next pass logs
        host_ev.signaledValue = vals[s];
        [inflight[s] waitUntilCompleted];
        if (inflight[s].error) printf("  command buffer error: %s\n", inflight[s].error.localizedDescription.UTF8String);
        if (((uint32_t*)logb.contents)[(3 * (it - 1) + 1) & 4095] != (uint32_t)(100000 + (it - 1))) log_bad++;
      }
    }
    printf("1a. in-buffer wait, %d steps, 2 in flight: GPU->CPU misses %d, CPU->GPU misses %d, host timeouts %d\n", N, seen_bad, log_bad, timeouts);
  }
  {   // 1b: a segment is [pass [1,3) of step it-1, pass [0,1) of step it, signal]; the CPU commits the next segment
      // after it has checked the signalled step and written the word that step's [1,3) pass logs
    const int N = 1000; int seen_bad = 0, log_bad = 0, timeouts = 0;
    volatile uint32_t* av = (volatile uint32_t*)a.contents;
    memset(a.contents, 0, 64); memset(logb.contents, 0xff, 4096 * 4);
    id<MTLCommandBuffer> last = nil;
    for (int it = 0; it <= N; it++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en waitForFence:fence];
      if (it > 0) { [en executeCommandsInBuffer:icb withRange:NSMakeRange(1, 2)]; [en updateFence:fence]; [en endEncoding];
                    en = [cb computeCommandEncoder]; [en waitForFence:fence]; }
      if (it < N) [en executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
      [en updateFence:fence]; [en endEncoding];
      if (it < N) [cb encodeSignalEvent:gpu_ev value:v + 1];
      [cb commit]; last = cb;
      if (it == N) break;
      v++;
      if (!host_wait(gpu_ev, v, 0, 2000)) { timeouts++; break; }
      if (av[0] != (uint32_t)(3 * it + 1)) seen_bad++;           // step it's first pass bumped a[0]
      av[1] = 200000 + it;                                       // the word step it's [1,3) pass logs
    }
    [last waitUntilCompleted];
    if (last.error) printf("  command buffer error: %s\n", last.error.localizedDescription.UTF8String);
    for (int it = 0; it < N; it++) if (((uint32_t*)logb.contents)[(3 * it + 1) & 4095] != (uint32_t)(200000 + it)) log_bad++;
    printf("1b. second command buffer, %d steps: GPU->CPU misses %d, CPU->GPU misses %d, host timeouts %d\n", N, seen_bad, log_bad, timeouts);
  }

  // ---- 2. GPU idle gap across a host round trip ----------------------------------------------------------------
  {
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> cs in d.counterSets) if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = cs;
    if (!ts || ![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { printf("2. skipped: no stage-boundary timestamps\n"); }
    else {
      MTLCounterSampleBufferDescriptor* sd = [MTLCounterSampleBufferDescriptor new];
      sd.counterSet = ts; sd.storageMode = MTLStorageModeShared; sd.sampleCount = 4;
      id<MTLCounterSampleBuffer> sb = [d newCounterSampleBufferWithDescriptor:sd error:&e];
      uint32_t busy = 20000;   // pass A runs ~0.5 ms so the command buffer is executing when it signals
      const char* names[5] = {"plain pass boundary (fence only)", "in-buffer wait, spinning CPU", "in-buffer wait, waitUntilSignaledValue",
                              "second buffer encoded before the signal", "second buffer encoded after the signal"};
      for (int round = 0; round < 2; round++)
        for (int mode = -1; mode <= 3; mode++) {
          const int N = 200; double gaps[200];
          MTLTimestamp c0, g0, c1, g1; [d sampleTimestamps:&c0 gpuTimestamp:&g0];
          for (int it = 0; it < N; it++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            MTLComputePassDescriptor* p1 = [MTLComputePassDescriptor computePassDescriptor];
            p1.sampleBufferAttachments[0].sampleBuffer = sb; p1.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0; p1.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoderWithDescriptor:p1];
            [en waitForFence:fence]; [en setComputePipelineState:spin]; [en setBuffer:work offset:0 atIndex:0]; [en setBytes:&busy length:4 atIndex:1];
            [en dispatchThreadgroups:MTLSizeMake(cores, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [en updateFence:fence]; [en endEncoding];
            if (mode == 0 || mode == 1) { [cb encodeSignalEvent:gpu_ev value:v + 1]; [cb encodeWaitForEvent:host_ev value:v + 1]; }
            if (mode >= 2) [cb encodeSignalEvent:gpu_ev value:v + 1];
            if (mode == 2) {           // the continuation is encoded before the signal; the host only commits it
              id<MTLCommandBuffer> cb2 = [q commandBuffer];
              encode_pass_b(cb2, sb, fence, spin, work);
              [cb commit]; v++;
              if (!host_wait(gpu_ev, v, 0, 2000)) printf("  host wait timed out\n");
              [cb2 commit]; [cb2 waitUntilCompleted];
            } else if (mode == 3) {    // the continuation is encoded once the signal is seen (the runtime's lower bound)
              [cb commit]; v++;
              if (!host_wait(gpu_ev, v, 0, 2000)) printf("  host wait timed out\n");
              id<MTLCommandBuffer> cb2 = [q commandBuffer];
              encode_pass_b(cb2, sb, fence, spin, work);
              [cb2 commit]; [cb2 waitUntilCompleted];
            } else {
              encode_pass_b(cb, sb, fence, spin, work);
              [cb commit];
              if (mode >= 0) {
                v++;
                if (!host_wait(gpu_ev, v, mode, 2000)) { printf("  host wait timed out\n"); host_ev.signaledValue = v; }
                host_ev.signaledValue = v;
              }
            }
            [cb waitUntilCompleted];
            [d sampleTimestamps:&c1 gpuTimestamp:&g1];
            NSData* data = [sb resolveCounterRange:NSMakeRange(0, 4)];
            const MTLCounterResultTimestamp* r = (const MTLCounterResultTimestamp*)data.bytes;
            double scale = (g1 > g0) ? (double)(c1 - c0) / (double)(g1 - g0) : 1.0;   // ns per GPU tick
            gaps[it] = (double)(r[2].timestamp - r[1].timestamp) * scale / 1e3;
          }
          qsort(gaps, N, sizeof(double), cmp_d);
          printf("2. round %d, %-40s: GPU gap end of pass A -> start of pass B: min %.1f  p10 %.1f  median %.1f  p90 %.1f  max %.1f us\n",
                 round + 1, names[mode + 1], gaps[0], gaps[N / 10], gaps[N / 2], gaps[9 * N / 10], gaps[N - 1]);
        }
    }
  }

  // ---- 5. the last signal of a command buffer whose status reads Completed --------------------------------------
  {
    const int N = 2000; double lag[2000]; int missing = 0;
    for (int it = 0; it < N; it++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en waitForFence:fence]; [en executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)]; [en updateFence:fence]; [en endEncoding];
      [cb encodeSignalEvent:gpu_ev value:v + 1];
      [cb commit]; v++;
      const double t0 = now_ms();
      while (cb.status < MTLCommandBufferStatusCompleted && now_ms() - t0 < 2000) {}
      const double tc = now_ms();
      while (gpu_ev.signaledValue < v && now_ms() - tc < 100) {}
      lag[it] = (now_ms() - tc) * 1e3;
      if (gpu_ev.signaledValue < v) missing++;
    }
    qsort(lag, N, sizeof(double), cmp_d);
    printf("5. %d command buffers ending in a signal: the signal visible after the status reads Completed: min %.2f  median %.2f  "
           "p99 %.2f  max %.2f us; not within 100 ms: %d\n", N, lag[0], lag[N / 2], lag[99 * N / 100], lag[N - 1], missing);
  }

  // ---- 3. and 4. a command buffer parked at a wait (P16_PARKS=0 skips these: ~37 s, deliberate GPU timeouts) ------
  if (!getenv("P16_PARKS") || atoi(getenv("P16_PARKS")) != 0) {
    id<MTLCommandQueue> other = [d newCommandQueue];
    // every case runs on a fresh queue: once a queue's buffer has hit the watchdog, Metal ignores that queue's later
    // submissions (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored), which would poison the next case
    const double parks_s[8] = {2, 4, 4.5, 5, 5.5, 6, 4, 6};
    for (int k = 0; k < 8; k++) {
      const int at_start = k >= 6;                   // part 4: the wait is the first command of the buffer
      const double park = parks_s[k];
      id<MTLCommandQueue> cq = [d newCommandQueue]; [cq addResidencySet:rs];
      id<MTLCommandBuffer> cb = [cq commandBuffer];
      id<MTLComputeCommandEncoder> en;
      if (!at_start) {
        en = [cb computeCommandEncoder];
        [en executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)]; [en updateFence:fence]; [en endEncoding];
        [cb encodeSignalEvent:gpu_ev value:v + 1];
      }
      [cb encodeWaitForEvent:host_ev value:v + 1];
      en = [cb computeCommandEncoder]; if (!at_start) [en waitForFence:fence];
      [en executeCommandsInBuffer:icb withRange:NSMakeRange(1, 2)]; [en updateFence:fence]; [en endEncoding];
      [cb commit]; v++;
      int ok = at_start ? 1 : host_wait(gpu_ev, v, 1, 5000);
      if (at_start) gpu_ev.signaledValue = v;        // keep the GPU event's value in step with v
      // while the command buffer is parked, another queue keeps submitting small command buffers
      double t0 = now_ms(), worst = 0; int done_other = 0, err_other = 0;
      while (now_ms() - t0 < park * 1e3) {
        id<MTLCommandBuffer> o = [other commandBuffer];
        id<MTLComputeCommandEncoder> oe = [o computeCommandEncoder];
        uint32_t n = 2000;
        [oe setComputePipelineState:spin]; [oe setBuffer:work offset:0 atIndex:0]; [oe setBytes:&n length:4 atIndex:1];
        [oe dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [oe endEncoding];
        double s0 = now_ms(); [o commit]; [o waitUntilCompleted]; double lat = now_ms() - s0;
        if (lat > worst) worst = lat;
        if (o.error) err_other++; else done_other++;
        usleep(50000);
      }
      MTLCommandBufferStatus mid = cb.status;
      double r0 = now_ms();
      host_ev.signaledValue = v;
      [cb waitUntilCompleted];
      printf("%d. parked %4.1f s %s: publish seen %s; status while parked %d; finished %.2f ms after the host signal; error: %s; "
             "GPU time %.0f ms; other queue: %d command buffers completed (%d errors), worst latency %.2f ms\n",
             at_start ? 4 : 3, park, at_start ? "at the start" : "after a pass", ok ? "yes" : "NO", (int)mid, now_ms() - r0,
             cb.error ? cb.error.localizedDescription.UTF8String : "none", (cb.GPUEndTime - cb.GPUStartTime) * 1e3, done_other, err_other, worst);
    }
  }
}
return 0; }
