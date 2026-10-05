// Shared by all probes: the number of GPU cores of this machine. Every probe derives its launch geometry from it
// (one 384-thread threadgroup per core = 12 SIMD-groups per core), so the suite means the same thing on a 10-core M4,
// an 18-core M3 Pro or a 40-core M5 Max. Override with GPU_CORES=<n>.
#pragma once
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
static int gpu_cores(void) {
  const char* e = getenv("GPU_CORES"); if (e && atoi(e) > 0) return atoi(e);
  int n = 0; io_iterator_t it;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS) { io_object_t o;
    while ((o = IOIteratorNext(it))) { CFTypeRef p = IORegistryEntryCreateCFProperty(o, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (p) { if (CFGetTypeID(p) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &n); CFRelease(p); }
      IOObjectRelease(o); if (n > 0) break; }
    IOObjectRelease(it); }
  if (n <= 0) { fprintf(stderr, "warning: could not read gpu-core-count; assuming 18 (set GPU_CORES)\n"); n = 18; }
  return n;
}
static int cmp_int(const void* a, const void* b) { return *(const int*)a - *(const int*)b; }
// sort + dedup + drop non-positive entries; returns the new length
static int tidy(int* v, int n) { qsort(v, n, sizeof(int), cmp_int); int m = 0; for (int i = 0; i < n; i++) if (v[i] > 0 && (m == 0 || v[i] != v[m-1])) v[m++] = v[i]; return m; }
