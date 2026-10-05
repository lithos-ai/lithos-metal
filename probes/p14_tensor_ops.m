#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
#include <math.h>
// M5 neural accelerators (mpp::tensor_ops::matmul2d) for T > 1 verification GEMMs against a CPU reference; compiles with the Command Line
// Tools only (the runtime Metal compiler ships the MPP headers). ./build/p14_tensor_ops check   compiles every variant without running.
static const uint32_t N = 17408, K = 5120;
static float fp4(uint32_t q) { uint32_t e = (q >> 1) & 3, m = q & 1; float v = e ? ldexpf(1.0f + m / 2.0f, (int)e - 1) : m * 0.5f; return (q & 8) ? -v : v; }
static float fp8(uint32_t q) { uint32_t e = (q >> 3) & 15, m = q & 7; float v = e ? ldexpf(1.0f + m / 8.0f, (int)e - 7) : ldexpf((float)m, -9); return (q & 0x80) ? -v : v; }
static float h2f(uint16_t h) { uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v = e ? ldexpf(1.0f + m / 1024.0f, (int)e - 15) : ldexpf((float)m, -24); return s ? -v : v; }
static uint16_t f2h(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t s = u >> 31, e = (u >> 23) & 255, m = u & 0x7FFFFF; if (e == 0) return (uint16_t)(s << 15); int he = (int)e - 127 + 15; if (he <= 0) return (uint16_t)(s << 15); return (uint16_t)((s << 15) | (he << 10) | (m >> 13)); }
int main(int argc, char** argv) { @autoreleasepool {
  bool check = argc > 1 && !strcmp(argv[1], "check");
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  printf("matmul2d probe on %s (%d cores, Apple10 = %d): y[TM][%u] = x[TM][%u] . W^T; TM = T tokens (padded), TN rows of W per threadgroup, TK-wide K tiles, S SIMD-groups per threadgroup.\n", d.name.UTF8String, NC, [d supportsFamily:(MTLGPUFamily)1010], N, K);
  NSString* src = [NSString stringWithContentsOfFile:@"p14_tensor_ops.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLCommandQueue> q = [d newCommandQueue];
  srand(777);
  uint8_t* wq = malloc((size_t)N * K); for (size_t i = 0; i < (size_t)N * K; i++) wq[i] = (uint8_t)(rand() & 0xBF);
  uint8_t* nib = malloc((size_t)N * K / 2); for (size_t i = 0; i < (size_t)N * K / 2; i++) nib[i] = (uint8_t)(rand() & 0xFF);          // two E2M1 codes per byte, low nibble first
  uint8_t* bsc = malloc((size_t)N * K / 16); for (size_t i = 0; i < (size_t)N * K / 16; i++) bsc[i] = (uint8_t)(0x28 + (rand() % 32));  // E4M3 block scales in [0.25, 2)
  uint16_t* xh = calloc((size_t)32 * K, 2); for (size_t i = 0; i < (size_t)8 * K; i++) xh[i] = f2h((float)(rand() % 2001 - 1000) / 1000.0f);   // 8 real token rows, the rest zero padding
  const int TREAL = 8;
  double* yref8 = malloc((size_t)TREAL * N * 8); double* yref4 = malloc((size_t)TREAL * N * 8);
  if (!check) for (uint32_t n = 0; n < N; n++) for (int t = 0; t < TREAL; t++) { double s = 0, s4 = 0;
    for (uint32_t k = 0; k < K; k++) { double xv = h2f(xh[t * K + k]); s += (double)fp8(wq[(size_t)n * K + k]) * xv;
      uint32_t q = (nib[((size_t)n * K + k) / 2] >> ((k & 1) * 4)) & 15; s4 += (double)fp4(q) * fp8(bsc[((size_t)n * K + k) / 16]) * xv; }
    yref8[(size_t)t * N + n] = s; yref4[(size_t)t * N + n] = s4; }
  id<MTLBuffer> xb = [d newBufferWithBytes:xh length:(size_t)32 * K * 2 options:MTLResourceStorageModeShared];
  id<MTLBuffer> yb = [d newBufferWithLength:(size_t)32 * N * 4 options:MTLResourceStorageModeShared];
  struct { int TM, TN, TK, S; } cfg[] = { {8,64,64,4}, {8,64,128,4}, {8,32,64,2}, {8,64,64,2}, {8,64,64,1}, {16,64,64,4}, {32,64,64,4}, {32,64,128,4}, {32,128,64,4}, {64,64,64,4} };
  const int NCFG = sizeof(cfg) / sizeof(cfg[0]);
  for (int fmt = 0; fmt < 3; fmt++) {   // 0 = FP8 staged, 1 = NVFP4 staged, 2 = half direct
    const size_t MB = fmt == 0 ? (size_t)N * K : fmt == 1 ? (size_t)N * K / 2 : (size_t)N * K * 2, SB = (size_t)N * K / 16; const int NCOPY = (int)(((size_t)2 << 30) / (MB + (fmt == 1 ? SB : 0)));
    const double* yref = fmt == 1 ? yref4 : yref8; const double useful = fmt == 1 ? (double)(MB + SB) : (double)MB;
    id<MTLBuffer> wbuf = nil, sbuf = nil;
    if (!check) { wbuf = [d newBufferWithLength:MB * NCOPY options:MTLResourceStorageModeShared]; uint8_t* pk = wbuf.contents;
      if (fmt == 0) memcpy(pk, wq, MB); else if (fmt == 1) { memcpy(pk, nib, MB); sbuf = [d newBufferWithLength:SB * NCOPY options:MTLResourceStorageModeShared]; for (int c = 0; c < NCOPY; c++) memcpy((uint8_t*)sbuf.contents + (size_t)c * SB, bsc, SB); }
      else { uint16_t* ph = (uint16_t*)pk; for (size_t i = 0; i < (size_t)N * K; i++) ph[i] = f2h(fp8(wq[i])); }
      for (int c = 1; c < NCOPY; c++) memcpy(pk + (size_t)c * MB, pk, MB); }
    printf("\n=== %s: %.1f MB per matrix, %d copies per measurement (min-of-3); GB/s = weight (+scale) bytes streamed; TFLOP/s counts 2*TM*N*K ===\n",
      fmt == 0 ? "gemm_fp8: FP8 weights dequantized tile by tile into threadgroup memory, then matmul2d" : fmt == 1 ? "gemm_nvfp4: NVFP4 nibbles x E4M3 block scales dequantized into threadgroup memory, then matmul2d" : "gemm_half: half weights straight from device memory (accelerator upper bound)", useful / 1e6, NCOPY);
    printf("  %-3s %-3s %-3s %-2s | %-9s %-9s %-9s | %s\n", "TM", "TN", "TK", "S", "ms/matrix", "GB/s", "TFLOP/s", "err (8 real rows)");
    for (int ci = 0; ci < NCFG; ci++) { int TM = cfg[ci].TM, TN = cfg[ci].TN, TK = cfg[ci].TK, S = cfg[ci].S;
      MTLCompileOptions* co = [MTLCompileOptions new]; co.languageVersion = (MTLLanguageVersion)((4 << 16) | 0); co.preprocessorMacros = @{ @"TM": @(TM), @"TN": @(TN), @"TK": @(TK), @"S": @(S) };
      id<MTLLibrary> lib = [d newLibraryWithSource:src options:co error:&e]; if (!lib) { printf("  %-3d %-3d %-3d %-2d | compile: %s\n", TM, TN, TK, S, [[e.localizedDescription componentsSeparatedByString:@"\n"] firstObject].UTF8String); continue; }
      id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:fmt == 0 ? @"gemm_fp8" : fmt == 1 ? @"gemm_nvfp4" : @"gemm_half"] error:&e]; if (!ps) { printf("  %-3d %-3d %-3d %-2d | pso: %s\n", TM, TN, TK, S, [[e.localizedDescription componentsSeparatedByString:@"\n"] firstObject].UTF8String); continue; }
      if (check) { printf("  %-3d %-3d %-3d %-2d | ok (maxTotalThreads %lu, tg mem %lu)\n", TM, TN, TK, S, (unsigned long)ps.maxTotalThreadsPerThreadgroup, (unsigned long)ps.staticThreadgroupMemoryLength); continue; }
      struct { uint32_t n_rows, p0, p1, p2; float scale, f1, f2, f3; } P = { N, 0, 0, 0, 1.0f, 0, 0, 0 }; double best = 1e9; bool err_cb = false;
      for (int rep = 0; rep < 3; rep++) { memset(yb.contents, 0, (size_t)32 * N * 4); id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:ps];
        [en setThreadgroupMemoryLength:(fmt < 2 ? (NSUInteger)TN * TK * 2 : 0) atIndex:0];
        for (int c = 0; c < NCOPY; c++) { [en setBuffer:wbuf offset:(size_t)c * MB atIndex:0]; [en setBuffer:xb offset:0 atIndex:1]; [en setBuffer:yb offset:0 atIndex:2]; [en setBytes:&P length:sizeof(P) atIndex:3]; if (sbuf) [en setBuffer:sbuf offset:(size_t)c * SB atIndex:4];
          [en dispatchThreadgroups:MTLSizeMake(N / TN,1,1) threadsPerThreadgroup:MTLSizeMake(32 * S,1,1)]; }
        [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; if (cb.error) { err_cb = true; printf("  %-3d %-3d %-3d %-2d | GPU error: %s\n", TM, TN, TK, S, cb.error.localizedDescription.UTF8String); break; }
        double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / NCOPY; if (t < best) best = t; }
      if (err_cb) continue;
      float* y = yb.contents; double mx = 0, md = 0; for (size_t i = 0; i < (size_t)TREAL * N; i++) { double a = fabs(yref[i]); if (a > mx) mx = a; double dd = fabs(y[i] - yref[i]); if (dd > md) md = dd; }
      printf("  %-3d %-3d %-3d %-2d | %9.3f %9.1f %9.2f | %.1e\n", TM, TN, TK, S, best, useful / 1e9 / (best / 1e3), 2.0 * TM * N * K / (best / 1e3) / 1e12, md / mx); }
    wbuf = nil; sbuf = nil; }
}}
