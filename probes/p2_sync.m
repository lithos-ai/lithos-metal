#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
// Probe: in-kernel synchronization between execution units of ONE dispatch via device-memory atomics.
static NSString* SRC = nil;
static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, NSString* n) { NSError* e=nil; id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:n] error:&e]; if(!p){printf("PSO %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(1);} return p; }
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil;
  SRC = [NSString stringWithContentsOfFile:@"p2_sync.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLLibrary> lib = [d newLibraryWithSource:SRC options:nil error:&e]; if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLCommandQueue> q = [d newCommandQueue];
  id<MTLComputePipelineState> pp = pso(d, lib, @"pingpong"), ck = pso(d, lib, @"checkin");
  id<MTLBuffer> ctr = [d newBufferWithLength:4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> res = [d newBufferWithLength:4096*3*4 options:MTLResourceStorageModeShared];
  struct { uint32_t a,b,rounds,max_spin; } P;
  struct { const char* name; uint32_t tpg; uint32_t ngroups; uint32_t a; uint32_t b; } cases[] = {
    {"same SIMD-group        (tid 0 vs 1,   one TG of 64)",   64, 1, 0, 1},
    {"diff SIMD-group, same TG (tid 0 vs 32, one TG of 64)",  64, 1, 0, 32},
    {"diff SIMD-group, same TG (tid 0 vs 512, one TG of 1024)",1024,1, 0, 512},
    {"diff threadgroups       (tid 0 vs 64,  two TGs of 64)",  64, 2, 0, 64},
    {"diff threadgroups       (tid 0 vs 32,  two TGs of 32)",  32, 2, 0, 32},
    {"diff threadgroups far   (TG0 vs TG15, 16 TGs of 384)",  384,16, 0, 15*384},
  };
  printf("=== PING-PONG handoff via device atomic (rounds=20000 per side, max_spin=200000) ===\n");
  for (int i = 0; i < 6; i++) {
    *(uint32_t*)ctr.contents = 0; memset(res.contents, 0, 64);
    P.a = cases[i].a; P.b = cases[i].b; P.rounds = 20000; P.max_spin = 200000;
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:pp]; [en setBuffer:ctr offset:0 atIndex:0]; [en setBuffer:res offset:0 atIndex:1]; [en setBytes:&P length:sizeof(P) atIndex:2];
    [en dispatchThreadgroups:MTLSizeMake(cases[i].ngroups,1,1) threadsPerThreadgroup:MTLSizeMake(cases[i].tpg,1,1)]; [en endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    uint32_t* r = (uint32_t*)res.contents; double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime)*1e3;
    uint32_t handoffs = r[0] + r[4];
    printf("%-58s: A done=%u B done=%u timeouts=%u/%u  gpu=%.2f ms", cases[i].name, r[0], r[4], r[3], r[7], gpu_ms);
    if (handoffs > 1000 && r[3]+r[7]==0) printf("  => %.0f ns/handoff (avg spins/wait A=%.1f B=%.1f, max=%u)", gpu_ms*1e6/handoffs, (double)r[1]/r[0], (double)r[5]/r[4], r[2]>r[6]?r[2]:r[6]);
    if (cb.error) printf("  ERROR: %s", cb.error.localizedDescription.UTF8String);
    printf("\n");
  }
  printf("\n=== CHECK-IN: how many threadgroups of one dispatch are concurrently resident? (max_spin=3,000,000) ===\n");
  struct { uint32_t G; uint32_t tpg; } cc[] = { {2,32},{4,32},{8,32},{16,32},{32,32},{64,32},{128,32},{256,32},{512,32},{8,384},{16,384},{32,384},{64,384},{8,1024},{16,1024},{32,1024},{64,1024} };
  for (int i = 0; i < 17; i++) {
    struct { uint32_t G, max_spin; } C = { cc[i].G, 3000000 };
    *(uint32_t*)ctr.contents = 0; memset(res.contents, 0, 4096*3*4);
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:ck]; [en setBuffer:ctr offset:0 atIndex:0]; [en setBuffer:res offset:0 atIndex:1]; [en setBytes:&C length:sizeof(C) atIndex:2];
    [en dispatchThreadgroups:MTLSizeMake(C.G,1,1) threadsPerThreadgroup:MTLSizeMake(cc[i].tpg,1,1)]; [en endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    uint32_t* r = (uint32_t*)res.contents; uint32_t ok = 0, to = 0; uint64_t maxs = 0;
    for (uint32_t g = 0; g < C.G; g++) { if (r[g*3+2] >= C.G) { ok++; if (r[g*3+1] > maxs) maxs = r[g*3+1]; } else to++; }
    printf("G=%4u threadgroups x %4u threads (%6u threads): all-present seen by %4u, timed out %4u, max spins among ok=%llu, gpu=%.2f ms%s\n",
      C.G, cc[i].tpg, C.G*cc[i].tpg, ok, to, maxs, (cb.GPUEndTime-cb.GPUStartTime)*1e3, cb.error ? "  ERROR" : "");
  }
}}
