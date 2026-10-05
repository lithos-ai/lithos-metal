#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int C = gpu_cores(); const uint32_t NOM = (uint32_t)(12*C);
  NSString* src = [NSString stringWithContentsOfFile:@"p10_claim_protocol.metal" encoding:NSUTF8StringEncoding error:&e];
  for (int fence = 0; fence < 2; fence++) {
  MTLCompileOptions* co = [MTLCompileOptions new]; co.languageVersion = (MTLLanguageVersion)((3 << 16) | 2);   // MSL 3.2: coherent(device) + atomic_thread_fence
  if (fence) co.preprocessorMacros = @{ @"USE_FENCE" : @1 };
  id<MTLLibrary> lib = [d newLibraryWithSource:src options:co error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spmd"] error:&e]; if (!ps) { printf("pso: %s\n", e.localizedDescription.UTF8String); return 1; }
  printf("\n######## %s ########\n", fence ? "SPEC-COMPLIANT: coherent(device) buffers + atomic_thread_fence(seq_cst, device scope) on every publish and barrier exit" : "relaxed atomics on plain device pointers (no fences)");
  id<MTLCommandQueue> q = [d newCommandQueue];
  const uint32_t N_OPS = 320, N_BLOCKS = (uint32_t)(30*C + 4);   // ~ one decode step: 320 barriers; ~2.5 blocks per simd-group (544 = 17408 rows / 32 on an 18-core part)
  id<MTLBuffer> cursor = [d newBufferWithLength:(N_OPS + N_OPS*NOM)*4 options:MTLResourceStorageModeShared], done = [d newBufferWithLength:N_OPS*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> hits = [d newBufferWithLength:N_OPS*N_BLOCKS*4 options:MTLResourceStorageModeShared], stats = [d newBufferWithLength:16 options:MTLResourceStorageModeShared], sink = [d newBufferWithLength:4*C*384*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> strides = [d newBufferWithLength:64*4 options:MTLResourceStorageModeShared]; { uint32_t* sp = strides.contents; int n = 0;
    for (uint32_t v = 5; n < 64; v++) { uint32_t a = v, b = NOM; while (b) { uint32_t t = a % b; a = b; b = t; } if (a == 1) sp[n++] = v; } }
  struct { uint32_t n_ops, n_blocks, work, max_spin, mode, n_sg, nominal_sg, one_op; } P;
  printf("Claim-protocol probe on %s (%d cores, nominal crew %u simd-groups): %u ops x %u blocks per dispatch.\n", d.name.UTF8String, C, NOM, N_OPS, N_BLOCKS);
  printf("%-44s %5s %9s %9s %10s %9s %8s %s\n", "case", "TGs", "gpu ms", "us/op", "exactly1x", "timeouts", "SGs used", "avg barrier spins/SG/op");
  struct { const char* name; uint32_t mode, work; int G; } cases[] = {
    {"protocol only (empty blocks), global cursor", 0, 0, C}, {"protocol only (empty blocks), static slices", 1, 0, C},
    {"~30us blocks, global cursor",                 0, 2000, C}, {"~30us blocks, static slices",                 1, 2000, C},
    {"protocol only (empty blocks), own+steal", 2, 0, C}, {"~30us blocks, own slice + steal", 2, 2000, C},
    {"~30us blocks, global cursor, half the crew", 0, 2000, C/2}, {"~30us blocks, own+steal, half the crew", 2, 2000, C/2},
    {"~30us blocks, static slices, half the crew", 1, 2000, C/2},
    {"~30us blocks, own+steal, 2x surplus TGs", 2, 2000, 2*C}, {"~30us blocks, own+steal, 4x surplus TGs", 2, 2000, 4*C} };
  for (int ci = 0; ci < 11; ci++) { P.n_ops = N_OPS; P.n_blocks = N_BLOCKS; P.work = cases[ci].work; P.max_spin = 3000000; P.mode = cases[ci].mode; P.n_sg = cases[ci].G * 12; P.nominal_sg = NOM; double best = 1e9; uint32_t bad = 0, to = 0, used = 0; double spins = 0;
    for (int rep = 0; rep < 3; rep++) { memset(cursor.contents, 0, cursor.length); memset(done.contents, 0, done.length); memset(hits.contents, 0, hits.length); memset(stats.contents, 0, 16);
      id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:ps];
      [en setBuffer:cursor offset:0 atIndex:0]; [en setBuffer:done offset:0 atIndex:1]; [en setBuffer:hits offset:0 atIndex:2]; [en setBuffer:stats offset:0 atIndex:3]; [en setBuffer:sink offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; [en setBuffer:strides offset:0 atIndex:6];
      [en dispatchThreadgroups:MTLSizeMake(cases[ci].G,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; if (t < best) best = t; if (cb.error) printf("   ERROR: %s\n", cb.error.localizedDescription.UTF8String);
      uint32_t* h = hits.contents; bad = 0; for (uint32_t i = 0; i < N_OPS*N_BLOCKS; i++) if (h[i] != 1) bad++; uint32_t* st = stats.contents; to = st[0]; used = st[3]; spins = (double)st[1] / (P.n_sg) / N_OPS; }
    printf("%-44s %5d %9.2f %9.2f %10s %9u %8u %.1f\n", cases[ci].name, cases[ci].G, best, best*1e3/N_OPS, bad ? "FAIL" : "yes", to, used, spins); }
  // ---- the alternative to in-kernel barriers: the SAME work as N_OPS separate dispatches in one serial encoder (dispatch boundary = barrier)
  for (int wi = 0; wi < 2; wi++) { uint32_t work = wi ? 2000 : 0; double best = 1e9; uint32_t bad = 0;
    for (int rep = 0; rep < 3; rep++) { memset(hits.contents, 0, hits.length); memset(stats.contents, 0, 16);
      id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:ps];
      [en setBuffer:cursor offset:0 atIndex:0]; [en setBuffer:done offset:0 atIndex:1]; [en setBuffer:hits offset:0 atIndex:2]; [en setBuffer:stats offset:0 atIndex:3]; [en setBuffer:sink offset:0 atIndex:4]; [en setBuffer:strides offset:0 atIndex:6];
      for (uint32_t op = 0; op < N_OPS; op++) { P.n_ops = N_OPS; P.n_blocks = N_BLOCKS; P.work = work; P.max_spin = 0; P.mode = 3; P.n_sg = NOM; P.nominal_sg = NOM; P.one_op = op;
        [en setBytes:&P length:sizeof(P) atIndex:5]; [en dispatchThreadgroups:MTLSizeMake(C,1,1) threadsPerThreadgroup:MTLSizeMake(384,1,1)]; }
      [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3; if (t < best) best = t;
      uint32_t* h = hits.contents; bad = 0; for (uint32_t i = 0; i < N_OPS*N_BLOCKS; i++) if (h[i] != 1) bad++; }
    printf("%-44s %5d %9.2f %9.2f %10s %9s %8s %s\n", work ? "~30us blocks, ONE DISPATCH PER OP (static)" : "protocol only, ONE DISPATCH PER OP (static)", C, best, best*1e3/N_OPS, bad ? "FAIL" : "yes", "-", "all", "(no in-kernel sync; 320 dispatches)"); }
  }
}}
