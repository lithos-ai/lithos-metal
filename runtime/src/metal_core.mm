#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include "metal_core.h"

namespace monolith {

struct DeviceImpl { id<MTLDevice> dev; };
struct BufferImpl { id<MTLBuffer> buf; void* mapped = nullptr; size_t mapped_len = 0; };
struct LibraryImpl { id<MTLLibrary> lib; id<MTLDevice> dev; };
struct PipelineImpl { id<MTLComputePipelineState> pso; };
struct QueueImpl { id<MTLCommandQueue> q; };

static int ioreg_gpu_cores() {
  int n = 0; io_iterator_t it;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS) {
    io_object_t o;
    while ((o = IOIteratorNext(it))) {
      CFTypeRef p = IORegistryEntryCreateCFProperty(o, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (p) { if (CFGetTypeID(p) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &n); CFRelease(p); }
      IOObjectRelease(o); if (n > 0) break;
    }
    IOObjectRelease(it);
  }
  return n;
}

Device::Device() : impl(std::make_shared<DeviceImpl>()) {
  impl->dev = MTLCreateSystemDefaultDevice();
  if (!impl->dev) throw std::runtime_error("no Metal device");
}
Device::~Device() = default;

DeviceInfo Device::info() const {
  DeviceInfo i;
  i.name = [impl->dev.name UTF8String];
  i.gpu_cores = ioreg_gpu_cores();
  for (int f = 1; f <= 12; f++) if ([impl->dev supportsFamily:(MTLGPUFamily)(1000 + f)]) i.apple_family = f;
  i.max_buffer_length = impl->dev.maxBufferLength;
  i.recommended_working_set = impl->dev.recommendedMaxWorkingSetSize;
  i.has_unified_memory = impl->dev.hasUnifiedMemory;
  return i;
}

Buffer::Buffer(const Device& d, size_t nbytes, const void* data) : impl(std::make_shared<BufferImpl>()) {
  impl->buf = data ? [d.impl->dev newBufferWithBytes:data length:nbytes options:MTLResourceStorageModeShared]
                   : [d.impl->dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  if (!impl->buf) throw std::runtime_error("newBuffer failed (" + std::to_string(nbytes) + " bytes)");
}

Buffer::Buffer(const Device& d, const std::string& path, uint64_t offset, size_t nbytes) : impl(std::make_shared<BufferImpl>()) {
  long page = sysconf(_SC_PAGESIZE);
  if (offset % page || nbytes % page) throw std::runtime_error("mmap buffers need page-aligned offset and length");
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open failed: " + path);
  // Immutable weights need no private copy-on-write mapping. MAP_PRIVATE
  // makes Metal's first residency preparation expensive even for cached file
  // pages; shared read-only mappings avoid that work across layout switches.
  void* p = mmap(nullptr, nbytes, PROT_READ, MAP_SHARED, fd, (off_t)offset);
  close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  impl->mapped = p; impl->mapped_len = nbytes;
  impl->buf = [d.impl->dev newBufferWithBytesNoCopy:p length:nbytes options:MTLResourceStorageModeShared deallocator:nil];
  if (!impl->buf) { munmap(p, nbytes); throw std::runtime_error("newBufferWithBytesNoCopy failed"); }
}

Buffer::~Buffer() { if (impl && impl.use_count() == 1 && impl->mapped) { impl->buf = nil; munmap(impl->mapped, impl->mapped_len); } }
size_t Buffer::nbytes() const { return impl->buf.length; }
void* Buffer::contents() const { return impl->buf.contents; }
uint64_t Buffer::gpu_address() const { return impl->buf.gpuAddress; }

Library::Library(const Device& d, const std::string& source, const std::map<std::string, std::string>& macros,
                 uint32_t language_version, bool fast_math) : impl(std::make_shared<LibraryImpl>()) {
  MTLCompileOptions* o = [MTLCompileOptions new];
  if (language_version) o.languageVersion = (MTLLanguageVersion)language_version;
  o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
  NSMutableDictionary* m = [NSMutableDictionary new];
  for (auto& kv : macros) m[[NSString stringWithUTF8String:kv.first.c_str()]] = [NSString stringWithUTF8String:kv.second.c_str()];
  o.preprocessorMacros = m;
  NSError* err = nil;
  impl->dev = d.impl->dev;
  impl->lib = [d.impl->dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:o error:&err];
  if (!impl->lib) throw std::runtime_error(std::string("MSL compile failed: ") + (err ? [err.localizedDescription UTF8String] : "?"));
}
Library::~Library() = default;

Pipeline::Pipeline(const Library& lib, const std::string& function, bool support_icb) : impl(std::make_shared<PipelineImpl>()) {
  id<MTLFunction> fn = [lib.impl->lib newFunctionWithName:[NSString stringWithUTF8String:function.c_str()]];
  if (!fn) throw std::runtime_error("no kernel named " + function);
  NSError* err = nil;
  MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = fn; pd.supportIndirectCommandBuffers = support_icb;
  impl->pso = [lib.impl->dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&err];
  if (!impl->pso) throw std::runtime_error(std::string("pipeline failed: ") + (err ? [err.localizedDescription UTF8String] : "?"));
}
Pipeline::~Pipeline() = default;
uint32_t Pipeline::max_threads_per_threadgroup() const { return (uint32_t)impl->pso.maxTotalThreadsPerThreadgroup; }
uint32_t Pipeline::thread_execution_width() const { return (uint32_t)impl->pso.threadExecutionWidth; }

Queue::Queue(const Device& d) : impl(std::make_shared<QueueImpl>()) { impl->q = [d.impl->dev newCommandQueue]; }
Queue::~Queue() = default;

static double now_ms() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }

RunResult Queue::run(const std::vector<Dispatch>& dispatches, bool concurrent) {
  @autoreleasepool {
    double t0 = now_ms();
    id<MTLCommandBuffer> cb = [impl->q commandBuffer];
    id<MTLComputeCommandEncoder> en = concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
    for (auto& d : dispatches) {
      [en setComputePipelineState:d.pipeline->impl->pso];
      for (auto& b : d.buffers) [en setBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
      for (auto& b : d.bytes) [en setBytes:b.bytes.data() length:b.bytes.size() atIndex:b.index];
      for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
      if (concurrent && d.barrier_before) [en memoryBarrierWithScope:MTLBarrierScopeBuffers];   // the same semantics as the ICB flag
      [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
    }
    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    RunResult r;
    r.gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    r.wall_ms = now_ms() - t0;
    if (cb.error) r.error = [cb.error.localizedDescription UTF8String];
    return r;
  }
}

bool Queue::supports_profiling() const {
  id<MTLDevice> dev = impl->q.device;
  return [dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] && dev.counterSets.count > 0;
}

std::vector<std::pair<double, double>> Queue::profile(const std::vector<Dispatch>& dispatches) {
  @autoreleasepool {
    std::vector<std::pair<double, double>> out;
    if (dispatches.empty()) return out;
    id<MTLDevice> dev = impl->q.device;
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> cs in dev.counterSets)
      if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = cs;
    if (!ts || ![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
      throw std::runtime_error("timestamp counter sampling at stage boundaries is not supported on this device");
    MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
    desc.counterSet = ts;
    desc.storageMode = MTLStorageModeShared;
    desc.sampleCount = 2 * dispatches.size();
    NSError* err = nil;
    id<MTLCounterSampleBuffer> samples = [dev newCounterSampleBufferWithDescriptor:desc error:&err];
    if (!samples) throw std::runtime_error(std::string("counter sample buffer: ") + (err ? [err.localizedDescription UTF8String] : "?"));
    MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
    [dev sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    id<MTLCommandBuffer> cb = [impl->q commandBuffer];
    for (size_t i = 0; i < dispatches.size(); i++) {
      const Dispatch& d = dispatches[i];
      MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
      MTLComputePassSampleBufferAttachmentDescriptor* att = pass.sampleBufferAttachments[0];
      att.sampleBuffer = samples;
      att.startOfEncoderSampleIndex = 2 * i;
      att.endOfEncoderSampleIndex = 2 * i + 1;
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoderWithDescriptor:pass];
      [en setComputePipelineState:d.pipeline->impl->pso];
      for (auto& b : d.buffers) [en setBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
      for (auto& b : d.bytes) [en setBytes:b.bytes.data() length:b.bytes.size() atIndex:b.index];
      for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
      [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
      [en endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    [dev sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    if (cb.error) throw std::runtime_error([cb.error.localizedDescription UTF8String]);
    NSData* data = [samples resolveCounterRange:NSMakeRange(0, 2 * dispatches.size())];
    if (!data || data.length < 2 * dispatches.size() * sizeof(MTLCounterResultTimestamp))
      throw std::runtime_error("counter samples could not be resolved");
    const MTLCounterResultTimestamp* r = (const MTLCounterResultTimestamp*)data.bytes;
    // GPU ticks -> ms through the CPU/GPU timestamp correlation (both sampled around the run)
    const double scale = (gpu1 > gpu0) ? double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1.0;   // ns per GPU tick
    const uint64_t base = r[0].timestamp;
    for (size_t i = 0; i < dispatches.size(); i++) {
      const uint64_t a = r[2 * i].timestamp, b = r[2 * i + 1].timestamp;
      if (a == MTLCounterErrorValue || b == MTLCounterErrorValue) { out.push_back({-1.0, -1.0}); continue; }
      out.push_back({double(a - base) * scale * 1e-6, double(b - base) * scale * 1e-6});
    }
    return out;
  }
}

}  // namespace monolith

// ---------------------------------------------------------------------------------------------------------------
// ICB replay and the host pump
#include <mach/mach.h>
#include <deque>
#include <mutex>

namespace monolith {

struct IcbImpl { id<MTLIndirectCommandBuffer> icb; size_t count = 0; };

static void encode_icb_command(id<MTLIndirectComputeCommand> c, const Dispatch& d) {
  [c setComputePipelineState:d.pipeline->impl->pso];
  for (auto& b : d.buffers) {
    if (b.offset > 0xFFFFFFFFull) throw std::runtime_error("ICB buffer offsets must fit 32 bits");
    [c setKernelBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
  }
  for (auto& t : d.threadgroup_memory) [c setThreadgroupMemoryLength:t.second atIndex:t.first];
  [c concurrentDispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
              threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
  if (d.barrier_before) [c setBarrier];                                  // the command waits for all commands before it
}

Icb::Icb(const Device& d, const std::vector<Dispatch>& ops) : impl(std::make_shared<IcbImpl>()) {
  if (ops.empty()) throw std::runtime_error("Icb: no ops");
  uint32_t max_bind = 0;
  for (auto& o : ops) {
    if (!o.bytes.empty()) throw std::runtime_error("Icb: setBytes is not available in an indirect command buffer; use a parameter buffer");
    for (auto& b : o.buffers) max_bind = std::max(max_bind, b.index + 1);
  }
  MTLIndirectCommandBufferDescriptor* desc = [MTLIndirectCommandBufferDescriptor new];
  desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  desc.inheritBuffers = NO;
  desc.inheritPipelineState = NO;
  desc.maxKernelBufferBindCount = max_bind;
  impl->icb = [d.impl->dev newIndirectCommandBufferWithDescriptor:desc maxCommandCount:ops.size() options:0];
  if (!impl->icb) throw std::runtime_error("newIndirectCommandBuffer failed");
  for (size_t i = 0; i < ops.size(); i++) encode_icb_command([impl->icb indirectComputeCommandAtIndex:i], ops[i]);
  impl->count = ops.size();
}
Icb::~Icb() = default;
size_t Icb::count() const { return impl->count; }

struct RunnerImpl {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  std::shared_ptr<IcbImpl> icb; std::vector<Dispatch> ops;
  std::vector<id<MTLBuffer>> resources;
  std::vector<MTLResourceUsage> resource_usage;
  id<MTLBuffer> state; uint32_t done_off, head_off, tail_off;
  id<MTLBuffer> ring; uint32_t cap;
  uint32_t tail = 0;                       // next ring slot the host reads
  std::vector<int32_t> tokens; std::mutex mu;
};

Runner::Runner(const Device& d, const Icb& icb, const std::vector<Dispatch>& ops, std::vector<const Buffer*> resources,
               const Buffer& step_state, uint32_t done_offset, uint32_t ring_head_offset, uint32_t ring_tail_offset,
               const Buffer& ring, uint32_t ring_capacity, std::vector<const Buffer*> read_only_resources)
    : impl(std::make_shared<RunnerImpl>()) {
  impl->dev = d.impl->dev; impl->q = [d.impl->dev newCommandQueue];
  impl->icb = icb.impl; impl->ops = ops;
  for (auto* r : resources) {
    impl->resources.push_back(r->impl->buf);
    bool read_only = std::find(read_only_resources.begin(), read_only_resources.end(), r) != read_only_resources.end();
    impl->resource_usage.push_back(read_only ? MTLResourceUsageRead : (MTLResourceUsageRead | MTLResourceUsageWrite));
  }
  impl->state = step_state.impl->buf; impl->done_off = done_offset; impl->head_off = ring_head_offset; impl->tail_off = ring_tail_offset;
  impl->tail = *(volatile uint32_t*)((char*)impl->state.contents + impl->tail_off);   // resume where a previous runner over the same StepState/ring stopped
  impl->ring = ring.impl->buf; impl->cap = ring_capacity;
  if (ring.nbytes() < (size_t)ring_capacity * 8) throw std::runtime_error("ring buffer needs 8 bytes per slot (token + sequence)");
}
Runner::~Runner() = default;

// CPU time of the CALLING thread only (getrusage would count Metal's driver threads too)
static double cpu_ms_now() {
  thread_basic_info_data_t info; mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
  if (thread_info(mach_thread_self(), THREAD_BASIC_INFO, (thread_info_t)&info, &count) != KERN_SUCCESS) return 0;
  return (info.user_time.seconds + info.system_time.seconds) * 1e3 + (info.user_time.microseconds + info.system_time.microseconds) / 1e3;
}

// The ring is drained while later command buffers may still be running, so the host must not trust a head counter
// it reads from the state buffer: writes of an in-flight buffer become visible in no particular order. Every slot
// therefore carries its own 1-based sequence number in the high 32 bits (one aligned 8-byte store on the GPU); the
// host takes slots as long as the next expected sequence is present and publishes its tail for the GPU's overflow
// check. `ring_head` in StepState is the GPU's own counter and is read by the host only after everything completed.
static void drain_ring(RunnerImpl& r) {
  const volatile uint64_t* slots = (const volatile uint64_t*)r.ring.contents;
  std::lock_guard<std::mutex> lk(r.mu);
  for (;;) {
    uint64_t v = slots[r.tail % r.cap];
    if ((uint32_t)(v >> 32) != r.tail + 1) break;
    r.tokens.push_back((int32_t)(uint32_t)v);
    r.tail++;
  }
  *(volatile uint32_t*)((char*)r.state.contents + r.tail_off) = r.tail;
}

RunnerStats Runner::run(uint32_t max_steps, uint32_t steps_per_cb, uint32_t in_flight, bool reencode, uint64_t max_tokens) {
  impl->tail = *(volatile uint32_t*)((char*)impl->state.contents + impl->tail_off);   // the ring tail lives in StepState: resume (or restart after a reset) from it
  RunnerImpl& r = *impl;
  RunnerStats st;
  if (steps_per_cb == 0 || in_flight == 0) throw std::runtime_error("steps_per_cb and in_flight must be >= 1");
  double t0 = now_ms(), c0 = cpu_ms_now();
  std::deque<id<MTLCommandBuffer>> pending;
  size_t tokens_seen = 0;
  auto observe = [&](id<MTLCommandBuffer> cb) {
    double waiting = now_ms();
    [cb waitUntilCompleted];
    st.wait_ms += now_ms() - waiting;
    st.gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    st.command_buffers++;
    if (cb.error && st.error.empty()) st.error = [cb.error.localizedDescription UTF8String];
    drain_ring(r);
    st.done = *(volatile uint32_t*)((char*)r.state.contents + r.done_off) != 0;
    { std::lock_guard<std::mutex> lk(r.mu); tokens_seen = r.tokens.size(); }   // drained since the last drain(): this call's tokens
  };
  while (st.steps_submitted < max_steps && !st.done && st.error.empty() && (max_tokens == 0 || tokens_seen < max_tokens)) {
    @autoreleasepool {
      double encoding = now_ms();
      uint32_t n = std::min<uint64_t>(steps_per_cb, max_steps - st.steps_submitted);
      id<MTLCommandBuffer> cb = [r.q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];   // serial: step s+1 starts after step s
      if (reencode) {
        for (uint32_t s = 0; s < n; s++)
          for (auto& d : r.ops) {
            [en setComputePipelineState:d.pipeline->impl->pso];
            for (auto& b : d.buffers) [en setBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
            for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
            [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
          }
      } else {
        for (size_t i = 0; i < r.resources.size(); i++) [en useResource:r.resources[i] usage:r.resource_usage[i]];
        for (uint32_t s = 0; s < n; s++) [en executeCommandsInBuffer:r.icb->icb withRange:NSMakeRange(0, r.icb->count)];
      }
      [en endEncoding];
      st.encode_ms += now_ms() - encoding;
      double committing = now_ms();
      [cb commit];
      st.commit_ms += now_ms() - committing;
      pending.push_back(cb);
      st.steps_submitted += n;
    }
    while (pending.size() >= in_flight) { observe(pending.front()); pending.pop_front(); }
  }
  while (!pending.empty()) { observe(pending.front()); pending.pop_front(); }
  st.wall_ms = now_ms() - t0;
  st.host_busy_ms = cpu_ms_now() - c0;
  return st;
}

std::vector<int32_t> Runner::drain() {
  drain_ring(*impl);
  std::lock_guard<std::mutex> lk(impl->mu);
  std::vector<int32_t> out; out.swap(impl->tokens);
  return out;
}

}  // namespace monolith
