#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "probe_common.h"
#include <mach/mach_time.h>
// P15: on-screen frame pacing while compute command buffers of L ms run back to back on another queue (plan M0 #7,
// design D6). A small window with a CAMetalLayer presents one trivially rendered frame per vsync on its own queue and
// records each frame's presented time (addPresentedHandler): the intervals the screen actually saw. Meanwhile a heavy
// queue keeps 3 command buffers in flight (the host pump's pattern), each L ms of ~1 ms ALU dispatches (aluwork of
// p4, calibrated first). Per L in {0 (baseline), 8, 16, 33, 66, 133} for ~4 s: mean / p99 / max frame interval,
// the share of frames that arrived >= 2 vsyncs late (dropped), the achieved fps, and the heavy queue's own rate.
// Our drawable goes through the WindowServer like every window's, so a stall here is a stall on screen. The heavy
// kernel is run twice: ALU-bound (aluwork) and bus-bound (p12's streaming read over a 1 GB buffer, ~0.8 ms of ~290 GB/s
// per dispatch — the engine's own kind of work, which competes with the compositor for the bus too). Bounded:
// dispatches ~1 ms, buffers <= 133 ms, ~70 s total. Skips (exit 0) when there is no display.
static double now_ms() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
#define NL 6
static const double lens[NL] = { 0, 8, 16, 33, 66, 133 };

int main() { @autoreleasepool {
  setvbuf(stdout, NULL, _IOLBF, 0);                                       // line-buffered: the results survive a crash when piped
  CFDictionaryRef sess = CGSessionCopyCurrentDictionary();
  if (CGMainDisplayID() == 0 || !sess) { printf("p15: no display / window session — skipped\n"); return 0; }
  CFBooleanRef locked = CFDictionaryGetValue(sess, CFSTR("CGSSessionScreenIsLocked"));
  if (locked && CFBooleanGetValue(locked)) { printf("p15: the screen is locked — no window is composited, nothing to measure; unlock and re-run\n"); CFRelease(sess); return 0; }
  CFRelease(sess);
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int C = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p4_core_model.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:nil error:&e];
  id<MTLComputePipelineState> al = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"aluwork"] error:&e];
  id<MTLCommandQueue> qHeavy = [d newCommandQueue], qFrame = [d newCommandQueue];
  id<MTLBuffer> out = [d newBufferWithLength:C * 384 * 4 options:MTLResourceStorageModeShared];
  // calibrate the ALU dispatch to ~1 ms at the crew geometry
  struct { uint32_t work, active_per_tg, active_tgs, pad; } CL = { 50000, 9999, 9999, 0 };
  double per_ms = 0;
  for (int it = 0; it < 3; it++) {
    id<MTLCommandBuffer> cb = [qHeavy commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:al]; [en setBuffer:out offset:0 atIndex:0];
    for (int i = 0; i < 20; i++) { [en setBytes:&CL length:sizeof(CL) atIndex:1]; [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; }
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    per_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 20.0;
    CL.work = (uint32_t)(CL.work * 1.0 / per_ms);           // aim at 1 ms per dispatch
  }
  printf("p15: %d cores; ALU dispatch calibrated to %.2f ms (work=%u)\n", C, per_ms, CL.work);
  // the bus-bound dispatch: p12's streaming kernel (mode 8: blocked, lane-interleaved 16 B, 4 loads in flight — the
  // pattern that streams ~290 GB/s at the crew geometry) over a 1 GB buffer, one ~240 MB window per dispatch (4 rotate)
  NSString* src12 = [NSString stringWithContentsOfFile:@"p12_stream_geometry.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib12 = [d newLibraryWithSource:src12 options:nil error:&e];
  id<MTLComputePipelineState> st = [d newComputePipelineStateWithFunction:[lib12 newFunctionWithName:@"stream"] error:&e];
  const size_t BIG = (size_t)1 << 30, WIN = (size_t)1 << 28;
  id<MTLBuffer> big = [d newBufferWithLength:BIG options:MTLResourceStorageModeShared];
  memset(big.contents, 1, BIG);
  struct { uint32_t n_sg, bytes_per_sg, mode, chunk; } CS = { (uint32_t)(C * 12), (uint32_t)(((WIN / (C * 12)) / 65536) * 65536), 8, 65536 };
  double stream_ms = 0;
  { id<MTLCommandBuffer> cb = [qHeavy commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:st];
    for (int i = 0; i < 8; i++) { [en setBuffer:big offset:(i % 4) * WIN atIndex:0]; [en setBuffer:out offset:0 atIndex:1]; [en setBytes:&CS length:sizeof(CS) atIndex:2];
      [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; }
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; stream_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 8.0; }
  printf("p15: streaming dispatch of %.0f MB: %.2f ms (%.0f GB/s)\n", (double)CS.bytes_per_sg * CS.n_sg / 1e6, stream_ms, (double)CS.bytes_per_sg * CS.n_sg / 1e6 / stream_ms);

  // the window: a CAMetalLayer we present into every vsync
  [NSApplication sharedApplication];
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  NSRect r = NSMakeRect(80, 80, 360, 220);
  NSWindow* w = [[NSWindow alloc] initWithContentRect:r styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable) backing:NSBackingStoreBuffered defer:NO];
  w.title = @"p15 frame pacing"; w.level = NSFloatingWindowLevel; w.releasedWhenClosed = NO;   // ARC owns the window (close would release it once more)                    // stays visible: an occluded window's drawables are never presented
  NSView* v = [[NSView alloc] initWithFrame:r]; v.wantsLayer = YES;
  CAMetalLayer* layer = [CAMetalLayer layer]; layer.device = d; layer.pixelFormat = MTLPixelFormatBGRA8Unorm; layer.framebufferOnly = YES;
  layer.maximumDrawableCount = 3; layer.displaySyncEnabled = YES; layer.frame = v.bounds; layer.drawableSize = CGSizeMake(360, 220);
  v.layer = layer; w.contentView = v; [w makeKeyAndOrderFront:nil]; [NSApp activateIgnoringOtherApps:YES];
  [NSApp finishLaunching];

  const double phase_s = 4.0;
  __block int phase = -1; __block int pass = 0; __block bool running = true;
  static double pres[2][NL][4000]; static int npres[2][NL], nunpres[2][NL]; memset(npres, 0, sizeof(npres)); memset(nunpres, 0, sizeof(nunpres));
  static double heavy_ms[2][NL]; static int heavy_n[2][NL];
  // the frame thread: present a frame whenever a drawable is free (vsync-paced), record its presented time
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{ uint32_t f = 0;
    while (running) { @autoreleasepool {
      id<CAMetalDrawable> dr = [layer nextDrawable]; if (!dr) { usleep(1000); continue; }
      id<MTLCommandBuffer> cb = [qFrame commandBuffer];
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = dr.texture; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      rp.colorAttachments[0].clearColor = MTLClearColorMake((f % 60) / 60.0, 0.3, 1.0 - (f % 60) / 60.0, 1.0);
      id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp]; [en endEncoding];
      int ph = phase, ps = pass;
      [dr addPresentedHandler:^(id<MTLDrawable> p) { if (ph >= 0 && ph < NL) { if (p.presentedTime <= 0) nunpres[ps][ph]++; else if (npres[ps][ph] < 4000) pres[ps][ph][npres[ps][ph]++] = p.presentedTime * 1e3; } }];
      [cb presentDrawable:dr]; [cb commit]; f++;
    } } });
  // the heavy thread: two passes (ALU-bound, bus-bound); per phase, buffers of L ms with 3 in flight
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    for (int ps = 0; ps < 2; ps++) { pass = ps; const double unit = ps == 0 ? per_ms : stream_ms;
    for (int p = 0; p < NL; p++) {
      usleep(700000); phase = p; double t0 = now_ms(); int n = 0; double gpu = 0; int di = 0;
      if (lens[p] == 0) { while (now_ms() - t0 < phase_s * 1e3) usleep(20000); }
      else {
        int k = (int)(lens[p] / unit + 0.5); if (k < 1) k = 1; NSMutableArray* pending = [NSMutableArray array];
        while (now_ms() - t0 < phase_s * 1e3) { @autoreleasepool {
          id<MTLCommandBuffer> cb = [qHeavy commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
          if (ps == 0) { [en setComputePipelineState:al]; [en setBuffer:out offset:0 atIndex:0];
            for (int i = 0; i < k; i++) { [en setBytes:&CL length:sizeof(CL) atIndex:1]; [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; } }
          else { [en setComputePipelineState:st];
            for (int i = 0; i < k; i++, di++) { [en setBuffer:big offset:(di % 4) * WIN atIndex:0]; [en setBuffer:out offset:0 atIndex:1]; [en setBytes:&CS length:sizeof(CS) atIndex:2];
              [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; } }
          [en endEncoding]; [cb commit]; [pending addObject:cb];
          if (pending.count >= 3) { id<MTLCommandBuffer> old = pending[0]; [old waitUntilCompleted]; gpu += (old.GPUEndTime - old.GPUStartTime) * 1e3; n++; [pending removeObjectAtIndex:0]; }
        } }
        for (id<MTLCommandBuffer> cb in pending) { [cb waitUntilCompleted]; gpu += (cb.GPUEndTime - cb.GPUStartTime) * 1e3; n++; }
      }
      heavy_ms[ps][p] = n ? gpu / n : 0; heavy_n[ps][p] = n; phase = -1;
    } }
    running = false; });
  // pump the window's events until the phases are done
  while (running) { @autoreleasepool {
    NSEvent* ev = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:[NSDate dateWithTimeIntervalSinceNow:0.05] inMode:NSDefaultRunLoopMode dequeue:YES];
    if (ev) [NSApp sendEvent:ev]; } }
  usleep(300000);
  // the baseline phase's median interval is the vsync period
  double base_med = 16.7; { int n = npres[0][0]; if (n > 2) { double iv[4000]; for (int i = 1; i < n; i++) iv[i-1] = pres[0][0][i] - pres[0][0][i-1]; qsort(iv, n - 1, sizeof(double), cmpd); base_med = iv[(n - 1) / 2]; } }
  printf("vsync period from the baseline: %.2f ms (%.0f Hz)\n", base_med, 1000.0 / base_med);
  for (int ps = 0; ps < 2; ps++) {
    printf("\n=== heavy queue: %s dispatches, 3 command buffers in flight ===\n", ps == 0 ? "ALU-bound (aluwork ~1 ms)" : "bus-bound (streaming read of ~240 MB per dispatch)");
    printf("%-8s %-26s %8s %8s %8s %8s %10s %10s\n", "L ms", "heavy queue", "frames", "mean ms", "p99 ms", "max ms", "late>=2vs", "fps");
    for (int p = 0; p < NL; p++) {
      int n = npres[ps][p]; if (n < 3) { printf("%-8.0f (no presented frames recorded; %d unpresented — the window was occluded)\n", lens[p], nunpres[ps][p]); continue; }
      double iv[4000]; int m = n - 1; for (int i = 1; i < n; i++) iv[i-1] = pres[ps][p][i] - pres[ps][p][i-1];
      double sum = 0, mx = 0; int late = 0; for (int i = 0; i < m; i++) { sum += iv[i]; if (iv[i] > mx) mx = iv[i]; if (iv[i] >= 2.0 * base_med - 1.0) late++; }
      double span = pres[ps][p][n-1] - pres[ps][p][0]; qsort(iv, m, sizeof(double), cmpd);
      char hv[64]; if (lens[p] == 0) snprintf(hv, sizeof hv, "idle"); else snprintf(hv, sizeof hv, "%d buffers, %.1f ms each", heavy_n[ps][p], heavy_ms[ps][p]);
      printf("%-8.0f %-26s %8d %8.2f %8.2f %8.2f %8.1f%%  %8.1f%s\n", lens[p], hv, m, sum / m, iv[(int)(0.99 * (m - 1))], mx, 100.0 * late / m, 1000.0 * m / span, nunpres[ps][p] ? "  (+unpresented)" : "");
    }
  }
  [w close];
  return 0;
}}
