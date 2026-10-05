#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "probe_common.h"
#include <math.h>
// Real decode-GEMV kernels (FP8-E4M3 and NVFP4, block-lane-major packs, 16-byte loads, R-row blocks, T tokens) at the crew geometry and at
// conventional occupancy, validated against a CPU reference. ./build/p13_decode_gemv check   compiles every variant without running.
static const uint32_t N = 17408, K = 5120;   // gate/up projection shape of Qwen3.8-27B
static float fp8(uint32_t q) { uint32_t e = (q >> 3) & 15, m = q & 7; float v = e ? ldexpf(1.0f + m / 8.0f, (int)e - 7) : ldexpf((float)m, -9); return (q & 0x80) ? -v : v; }
static float fp4(uint32_t q) { uint32_t e = (q >> 1) & 3, m = q & 1; float v = e ? ldexpf(1.0f + m / 2.0f, (int)e - 1) : m * 0.5f; return (q & 8) ? -v : v; }
static float h2f(uint16_t h) { uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v = e ? ldexpf(1.0f + m / 1024.0f, (int)e - 15) : ldexpf((float)m, -24); return s ? -v : v; }
static uint16_t f2h(float f) { /* f in [-1,1]: simple round-to-nearest */ uint32_t u; memcpy(&u, &f, 4); uint32_t s = u >> 31, e = (u >> 23) & 255, m = u & 0x7FFFFF; if (e == 0) return (uint16_t)(s << 15); int he = (int)e - 127 + 15; if (he <= 0) return (uint16_t)(s << 15); return (uint16_t)((s << 15) | (he << 10) | (m >> 13)); }
static uint32_t widx(uint32_t layout, uint32_t R, uint32_t C4, uint32_t lane, uint32_t r, uint32_t j) { return layout == 0 ? (lane * R + r) * C4 + j : (r * C4 + j) * 32 + lane; }
int main(int argc, char** argv) { @autoreleasepool {
  bool check = argc > 1 && !strcmp(argv[1], "check");
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError* e = nil; const int NC = gpu_cores();
  NSString* src = [NSString stringWithContentsOfFile:@"p13_decode_gemv.metal" encoding:NSUTF8StringEncoding error:&e];
  id<MTLCommandQueue> q = [d newCommandQueue];
  // random weights (bit patterns) and activations, shared by every variant; the CPU reference is computed once per (fmt, T)
  srand(12345);
  uint8_t* wq = malloc((size_t)N * K);            for (size_t i = 0; i < (size_t)N * K; i++) wq[i] = (uint8_t)(rand() & 0xBF);                 // FP8 codes with e <= 7 (|w| < 2, no NaN)
  uint8_t* nib = malloc((size_t)N * K);           for (size_t i = 0; i < (size_t)N * K; i++) nib[i] = (uint8_t)(rand() & 15);                   // FP4 codes
  uint8_t* bsc = malloc((size_t)N * K / 16);      for (size_t i = 0; i < (size_t)N * K / 16; i++) bsc[i] = (uint8_t)(0x28 + (rand() % 32));       // E4M3 block scales in [0.25, 2)
  uint16_t* xh = malloc((size_t)8 * K * 2);       for (size_t i = 0; i < (size_t)8 * K; i++) xh[i] = f2h((float)(rand() % 2001 - 1000) / 1000.0f);
  id<MTLBuffer> xb = [d newBufferWithBytes:xh length:(size_t)8 * K * 2 options:MTLResourceStorageModeShared];
  id<MTLBuffer> yb = [d newBufferWithLength:(size_t)8 * N * 4 options:MTLResourceStorageModeShared];
  double* yref[2][9] = {{0}}; // [fmt][T]
  struct { int R, T; } RT[] = { {32,1}, {16,1}, {8,1}, {4,1}, {16,2}, {8,4}, {8,8} };
  printf("Decode GEMV on %s (%d cores, crew = %d SIMD-groups): y[T][%u] = x[T][%u] . W^T. Each measurement streams ~2.1 GB of packed weights\n"
         "(NCOPY identical matrices, one dispatch each, one command buffer, min-of-3). GB/s counts useful weight (+scale) bytes. err = max |y - y_ref| / max |y_ref|.\n", d.name.UTF8String, NC, 12*NC, N, K);
  for (int fmt = 0; fmt < 2; fmt++) {
    const uint32_t C4 = fmt == 0 ? 10 : 6; const size_t MB = (size_t)N * 32 * C4 * 16;   // packed bytes per matrix (N/R blocks x R rows x 32 lanes x C4 uint4)
    const double useful = fmt == 0 ? (double)N * K : (double)N * K / 2 + (double)N * K / 16;
    const int NCOPY = (int)(((size_t)2 << 30) / MB); id<MTLBuffer> wbuf = check ? nil : [d newBufferWithLength:MB * NCOPY options:MTLResourceStorageModeShared];
    printf("\n=== %s: %.1f MB packed per matrix (%.1f useful), %d copies ===\n", fmt == 0 ? "FP8 E4M3, per-tensor scale" : "NVFP4 (E2M1 + E4M3 scale per 16), 96 B per lane-row incl. 6 B pad", MB / 1e6, useful / 1e6, NCOPY);
    printf("%-3s %-2s %-18s | %-34s | %-22s | %-22s | %-22s | %s\n", "R", "T", "layout", "crew: 1 TG(384)/core [blocks/SG, tail eff]", "2 TG(384)/core", "1 block/SG, TG=384", "1 block/SG, TG=64", "err");
    for (int li = 0; li < 2; li++) for (int vi = 0; vi < 7; vi++) { const int R = RT[vi].R, T = RT[vi].T, RG = T >= 8 ? 2 : 4;
      MTLCompileOptions* co = [MTLCompileOptions new]; co.preprocessorMacros = @{ @"FMT": @(fmt), @"R": @(R), @"T": @(T), @"LAYOUT": @(li), @"RG": @(RG) };
      id<MTLLibrary> lib = [d newLibraryWithSource:src options:co error:&e]; if (!lib) { printf("  compile FMT=%d R=%d T=%d LAYOUT=%d: %s\n", fmt, R, T, li, [[e.localizedDescription componentsSeparatedByString:@"\n"] firstObject].UTF8String); continue; }
      id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv"] error:&e]; if (!ps) { printf("  pso: %s\n", e.localizedDescription.UTF8String); continue; }
      if (check) { printf("  ok FMT=%d R=%d T=%d LAYOUT=%d  maxTotalThreads=%lu\n", fmt, R, T, li, (unsigned long)ps.maxTotalThreadsPerThreadgroup); continue; }
      // pack matrix 0 in this (R, layout) and replicate
      uint8_t* pk = (uint8_t*)wbuf.contents; const uint32_t blk4 = R * 32 * C4;
      for (uint32_t n = 0; n < N; n++) { uint32_t b = n / R, r = n % R; uint8_t* base = pk + (size_t)b * blk4 * 16;
        if (fmt == 0) { for (uint32_t k = 0; k < K; k++) { uint32_t lane = k / 160, c = k % 160; base[widx(li, R, C4, lane, r, c / 16) * 16 + (c % 16)] = wq[(size_t)n * K + k]; } }
        else { for (uint32_t lane = 0; lane < 32; lane++) { uint8_t* sc = base + widx(li, R, C4, lane, r, 5) * 16; memset(sc, 0, 16);
            for (uint32_t c = 0; c < 160; c++) { uint32_t k = lane * 160 + c; uint8_t* wp = base + widx(li, R, C4, lane, r, c / 32) * 16 + (c % 32) / 2;
              if (c & 1) *wp |= (uint8_t)(nib[(size_t)n * K + k] << 4); else *wp = nib[(size_t)n * K + k]; if ((c % 16) == 0) sc[c / 16] = bsc[((size_t)n * K + k) / 16]; } } } }
      for (int cpy = 1; cpy < NCOPY; cpy++) memcpy(pk + (size_t)cpy * MB, pk, MB);
      if (!yref[fmt][T]) { double* yr = malloc((size_t)T * N * 8); yref[fmt][T] = yr;
        for (uint32_t n = 0; n < N; n++) for (int t = 0; t < T; t++) { double s = 0; const uint8_t* wr = (fmt == 0 ? wq : nib) + (size_t)n * K;
          if (fmt == 0) for (uint32_t k = 0; k < K; k++) s += (double)fp8(wr[k]) * h2f(xh[t * K + k]);
          else for (uint32_t k = 0; k < K; k++) s += (double)fp4(wr[k]) * fp8(bsc[((size_t)n * K + k) / 16]) * h2f(xh[t * K + k]);
          yr[(size_t)t * N + n] = s; } }
      const uint32_t nblocks = N / R; struct { const char* name; uint32_t n_sg; int tpg; } geos[] = { {"crew", (uint32_t)(12*NC), 384}, {"2tg", (uint32_t)(24*NC), 384}, {"1blk384", nblocks, 384}, {"1blk64", nblocks, 64} };
      printf("%-3d %-2d %-18s |", R, T, li == 0 ? "lane-contiguous" : "lane-interleaved");
      double err = 0;
      for (int gi = 0; gi < 4; gi++) { struct { uint32_t n_sg, n_blocks, n_rows, pad; float scale, p1, p2, p3; } P = { geos[gi].n_sg, nblocks, N, 0, 1.0f, 0, 0, 0 }; double best = 1e9;
        for (int rep = 0; rep < 3; rep++) { id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder]; [en setComputePipelineState:ps];
          for (int cpy = 0; cpy < NCOPY; cpy++) { [en setBuffer:wbuf offset:(size_t)cpy * MB atIndex:0]; [en setBuffer:xb offset:0 atIndex:1]; [en setBuffer:yb offset:0 atIndex:2]; [en setBytes:&P length:sizeof(P) atIndex:3];
            [en dispatchThreadgroups:MTLSizeMake(((size_t)P.n_sg * 32 + geos[gi].tpg - 1) / geos[gi].tpg,1,1) threadsPerThreadgroup:MTLSizeMake(geos[gi].tpg,1,1)]; }
          [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / NCOPY; if (t < best) best = t; }
        float* y = yb.contents; double mx = 0, md = 0; for (size_t i = 0; i < (size_t)T * N; i++) { double a = fabs(yref[fmt][T][i]); if (a > mx) mx = a; double dd = fabs(y[i] - yref[fmt][T][i]); if (dd > md) md = dd; } if (md / mx > err) err = md / mx;
        if (gi == 0) printf(" %6.3f ms %6.1f GB/s [%4.1f, %3.0f%%] |", best, useful / 1e9 / (best / 1e3), (double)nblocks / P.n_sg, 100.0 * ((double)nblocks / P.n_sg) / ceil((double)nblocks / P.n_sg));
        else printf(" %6.3f ms %6.1f GB/s |", best, useful / 1e9 / (best / 1e3)); }
      printf(" %.1e\n", err); }
    wbuf = nil; }
}}
