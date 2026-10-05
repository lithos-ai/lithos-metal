#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main() { @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  printf("device: %s\n", d.name.UTF8String);
  printf("hasUnifiedMemory=%d  recommendedMaxWorkingSetSize=%.2f GB  maxBufferLength=%.2f GB\n",
    d.hasUnifiedMemory, d.recommendedMaxWorkingSetSize/1e9, d.maxBufferLength/1e9);
  printf("maxThreadgroupMemoryLength=%lu bytes\n", (unsigned long)d.maxThreadgroupMemoryLength);
  MTLSize m = d.maxThreadsPerThreadgroup; printf("maxThreadsPerThreadgroup=%lu x %lu x %lu\n", m.width, m.height, m.depth);
  printf("argumentBuffersSupport tier=%lu  maxArgumentBufferSamplerCount=%lu\n", (unsigned long)d.argumentBuffersSupport, (unsigned long)d.maxArgumentBufferSamplerCount);
  printf("supportsFunctionPointers=%d supportsDynamicLibraries=%d supports32BitFloatFiltering=%d\n", d.supportsFunctionPointers, d.supportsDynamicLibraries, d.supports32BitFloatFiltering);
  for (int f = 1001; f <= 1012; f++) printf("  MTLGPUFamilyApple%d: %d\n", f-1000, [d supportsFamily:(MTLGPUFamily)f]);
  printf("  MTLGPUFamilyMetal3(5001): %d   Metal4(5002): %d\n", [d supportsFamily:(MTLGPUFamily)5001], [d supportsFamily:(MTLGPUFamily)5002]);
  // MSL language version probe: compile trivial kernel at increasing versions
  NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] = 1; }";
  unsigned long vers[] = { (3<<16)|0, (3<<16)|1, (3<<16)|2, (4<<16)|0, (4<<16)|1 };
  for (int i = 0; i < 5; i++) { MTLCompileOptions* o = [MTLCompileOptions new]; o.languageVersion = (MTLLanguageVersion)vers[i]; NSError* e = nil;
    id<MTLLibrary> l = [d newLibraryWithSource:src options:o error:&e];
    NSString* msg = l ? @"OK" : [[e.localizedDescription componentsSeparatedByString:@"\n"] firstObject];
    printf("  MSL %lu.%lu: %s\n", vers[i]>>16, vers[i]&0xffff, msg.UTF8String); }
  // pipeline properties
  NSError* e = nil; id<MTLLibrary> l = [d newLibraryWithSource:src options:nil error:&e];
  id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k"] error:&e];
  printf("threadExecutionWidth=%lu maxTotalThreadsPerThreadgroup=%lu staticThreadgroupMemoryLength=%lu\n",
    (unsigned long)p.threadExecutionWidth, (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)p.staticThreadgroupMemoryLength);
}}
