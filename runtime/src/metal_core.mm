#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include "metal_core.h"

namespace monolith {

struct DeviceImpl { id<MTLDevice> dev; };
struct BufferImpl { id<MTLBuffer> buf; void* mapped = nullptr; size_t mapped_len = 0; bool resident = false; };

// One residency set for every buffer a Runner binds, kept while the buffer lives: buffers that several programs
// bind (the prompt graph and the decoder share weights) stay resident when the programs change, and no command
// buffer declares them again. LITHOS_RESIDENCY_SET=0 declares each program's buffers per command buffer instead.
static std::mutex residency_mu;
static id<MTLResidencySet> residency_set = nil;
static bool residency_enabled() { const char* v = getenv("LITHOS_RESIDENCY_SET"); return !v || v[0] != '0'; }
static void residency_release(BufferImpl& b) {
  if (!b.resident) return;
  std::lock_guard<std::mutex> lk(residency_mu);
  [residency_set removeAllocation:b.buf];
  [residency_set commit];
  b.resident = false;
}
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

Buffer::~Buffer() {
  if (!impl || impl.use_count() != 1) return;
  residency_release(*impl);
  if (impl->mapped) { impl->buf = nil; munmap(impl->mapped, impl->mapped_len); }
}
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

// A completed command buffer's failure: its error, then every report Metal shader validation (MTL_SHADER_VALIDATION=1)
// logged on it (`reported`: there was one). Validation drops an out-of-bounds access and records it only in `logs`,
// which are complete when waitUntilCompleted returns: the status stays completed and the error nil (measured), so this
// string is how the caller sees it, with or without MTL_SHADER_VALIDATION_REPORT_TO_STDERR. A report ends with the host
// stack that allocated the buffer — the same nanobind frames for every buffer — which is cut. Each report is counted
// (validation_reports()).
static std::atomic<uint64_t> reports{0};
uint64_t validation_reports() { return reports.load(); }

static std::string failure(id<MTLCommandBuffer> cb, bool* reported = nullptr) {
  std::string e = cb.error ? [cb.error.localizedDescription UTF8String] : "";
  for (id<MTLFunctionLog> log in cb.logs) {
    NSString* d = log.description;
    NSRange cut = [d rangeOfString:@"\nMTLBuffer allocated here"];
    if (cut.location != NSNotFound) d = [d substringToIndex:cut.location];
    e += (e.empty() ? "" : "\n") + std::string("shader validation: ") + [d UTF8String];
    reports++;
    if (reported) *reported = true;
  }
  return e;
}

static MTLResourceUsage use_usage(const BufferUse& u) { return u.write ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead; }

RunResult Queue::run(const std::vector<Dispatch>& dispatches, bool concurrent) {
  RunResult r;
  bool reported = false;
  @autoreleasepool {
    double t0 = now_ms();
    id<MTLCommandBuffer> cb = [impl->q commandBuffer];
    id<MTLComputeCommandEncoder> en = concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
    for (auto& d : dispatches) {
      [en setComputePipelineState:d.pipeline->impl->pso];
      for (auto& b : d.buffers) [en setBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
      for (auto& b : d.bytes) [en setBytes:b.bytes.data() length:b.bytes.size() atIndex:b.index];
      for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
      for (auto& u : d.uses) [en useResource:u.buffer->impl->buf usage:use_usage(u)];
      if (concurrent && d.barrier_before) [en memoryBarrierWithScope:MTLBarrierScopeBuffers];   // the same semantics as the ICB flag
      [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
    }
    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    r.gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    r.wall_ms = now_ms() - t0;
    r.error = failure(cb, &reported);
  }
  if (reported) throw std::runtime_error(r.error);   // a caller that drops the RunResult still fails on a report
  return r;
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
      for (auto& u : d.uses) [en useResource:u.buffer->impl->buf usage:use_usage(u)];
      [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
      [en endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    [dev sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    if (std::string e = failure(cb); !e.empty()) throw std::runtime_error(e);
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

static constexpr uint32_t kDoneLog = 64;   // steps of done snapshots kept; the GPU runs at most two steps ahead of the host

// Gives a task back, unless another runner has claimed it since (a schedule may hold one task at several points, so
// this runs more than once for it).
static void release_task(HostTask& task, const void* owner) {
  const void* held = owner;
  task.owner.compare_exchange_strong(held, nullptr);
}

struct RunnerImpl {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  std::shared_ptr<IcbImpl> icb; std::vector<Dispatch> ops;
  std::vector<id<MTLBuffer>> resources;
  std::vector<MTLResourceUsage> resource_usage;
  id<MTLBuffer> state; uint32_t done_off, head_off, tail_off;
  id<MTLBuffer> ring; uint32_t cap;
  uint32_t tail = 0;                       // next ring slot the host reads
  std::vector<int32_t> tokens; std::mutex mu;
  uint64_t taken = 0;                      // tokens drained from the ring since the current run began (max_tokens)
  std::atomic<std::thread::id> owner{};    // the thread inside run() or set_host_points(): one pump per runner
  bool resident = false;                   // the queue carries residency_set (replaces useResource)
  // host steps (set_host_points)
  std::vector<HostPoint> points;
  uint32_t error_off = 0, timeout_ms = 20000, spin_us = 20000;
  std::vector<uint32_t> pass_cuts;         // boundaries that end a pass: 0, n and every interior publish or wait
  std::vector<uint32_t> cb_cuts;           // boundaries that end a command buffer: 0, n and every interior wait
  std::vector<std::vector<uint32_t>> pub_at, wait_at;   // per boundary 0..n: the points that publish / wait there
  std::vector<uint32_t> pub_seg;           // per boundary 1..n: the command buffer (index into cb_cuts) whose passes end there
  id<MTLSharedEvent> gpu_ev = nil;         // the GPU signals V(step, point) after the pass ending at the publish boundary
  id<MTLFence> fence = nil;                // orders every pass behind the one before it, across command buffers
  bool fence_live = false;                 // a pass has updated the fence
  id<MTLBuffer> done_log = nil;            // StepState.done copied at the end of each step, indexed by step % kDoneLog
  uint64_t step_base = 0;                  // absolute index of the next step the runner begins
  uint64_t event_base = 0;                 // step index at which gpu_ev was created (its values count from there)
  bool drop = false;                       // test support: encode_segment leaves out the signal of (drop_step, drop_point)
  uint64_t drop_step = 0;
  uint32_t drop_point = 0;
  ~RunnerImpl() {
    for (auto& p : points) if (p.task) release_task(*p.task, this);    // its tasks serve another runner again
    if (resident) [q removeResidencySet:residency_set];
  }
};

// Holds the runner for one call of run() or set_host_points(). A second call fails at once instead of waiting: run()
// releases the GIL, and a host task runs on the thread that holds the runner.
struct Hold {
  std::atomic<std::thread::id>& owner;
  Hold(std::atomic<std::thread::id>& o, const char* what) : owner(o) {
    std::thread::id held{};
    if (!owner.compare_exchange_strong(held, std::this_thread::get_id()))
      throw std::runtime_error(std::string(what) + (held == std::this_thread::get_id() ? ": called from one of the runner's own host tasks"
                                                                                     : ": already running on another thread"));
  }
  ~Hold() { owner.store(std::thread::id{}); }
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
  if (residency_enabled()) {
    std::lock_guard<std::mutex> lk(residency_mu);
    if (!residency_set) {
      MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
      rd.initialCapacity = 1024;
      NSError* err = nil;
      residency_set = [impl->dev newResidencySetWithDescriptor:rd error:&err];
      if (!residency_set) throw std::runtime_error(std::string("newResidencySet failed: ") + (err ? [err.localizedDescription UTF8String] : "?"));
    }
    bool added = false;
    for (auto* r : resources)
      if (!r->impl->resident) { [residency_set addAllocation:r->impl->buf]; r->impl->resident = true; added = true; }
    if (added) { [residency_set commit]; [residency_set requestResidency]; }
    [impl->q addResidencySet:residency_set];
    impl->resident = true;
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
    r.taken++;
  }
  *(volatile uint32_t*)((char*)r.state.contents + r.tail_off) = r.tail;
}

// ---------------------------------------------------------------------------------------------------------------
// Host steps. Op i of step s; boundary b of step s sits before op b, and boundary n of step s is boundary 0 of step
// s + 1. A command buffer covers the ops of one step between two consecutive cb_cuts; its passes end at pass_cuts, and
// the pass that ends at a publish boundary is followed by the signal of every point publishing there. Command buffers
// are committed in order, the one starting at boundary b only after every point waiting at b has been served, so a
// committed command buffer never waits on the host and every point's signal is committed before its turn to be served.

void Runner::set_host_points(std::vector<HostPoint> points, uint32_t error_offset, uint32_t timeout_ms, uint32_t spin_us) {
  RunnerImpl& r = *impl;
  std::vector<HostPoint> old;   // the replaced points die last, after the hold: a task's last reference may run Python,
                                // which may walk this runner's points (the collector) or call it
  Hold hold(r.owner, "set_host_points");
  const uint32_t n = (uint32_t)r.ops.size();
  uint32_t prev = 0;
  for (auto& p : points) {
    if (!p.task) throw std::runtime_error("host point without a task");
    if (p.publish > p.wait || p.wait > n) throw std::runtime_error("host point: need 0 <= publish <= wait <= the op count");
    if (p.publish < prev) throw std::runtime_error("host points: publish boundaries must not decrease in program order");
    prev = p.publish;
  }
  if ((size_t)error_offset + 4 > r.state.length) throw std::runtime_error("host points: the error offset lies outside StepState");
  if (timeout_ms == 0) throw std::runtime_error("host points: timeout_ms must be at least 1");
  // Test support: LITHOS_HOST_STEPS_DROP_SIGNAL=<step>:<point>, two decimal numbers, leaves out that one signal (the
  // step counts every step the runner begins), so a command buffer finishes without it. An empty value is unset.
  bool drop = false; uint64_t drop_step = 0; uint32_t drop_point = 0;
  if (const char* v = getenv("LITHOS_HOST_STEPS_DROP_SIGNAL"); v && *v) {
    const std::string s(v);
    const size_t colon = s.find(':');
    auto digits = [](const std::string& t) {       // at most 19 digits: std::stoull cannot overflow
      return !t.empty() && t.size() <= 19 && t.find_first_not_of("0123456789") == std::string::npos;
    };
    if (colon == std::string::npos || !digits(s.substr(0, colon)) || !digits(s.substr(colon + 1)))
      throw std::runtime_error("LITHOS_HOST_STEPS_DROP_SIGNAL: expected <step>:<point> in decimal digits, got '" + s + "'");
    const uint64_t k = std::stoull(s.substr(colon + 1));
    if (!points.empty() && k >= points.size())
      throw std::runtime_error("LITHOS_HOST_STEPS_DROP_SIGNAL: point " + std::to_string(k) + ", but the program has " +
                               std::to_string(points.size()) + " host points");
    drop = true; drop_step = std::stoull(s.substr(0, colon)); drop_point = (uint32_t)k;
  }
  // The new schedule is built beside the old one: a throw (an allocation failure) leaves the runner as it was
  std::vector<std::vector<uint32_t>> pub_at(n + 1), wait_at(n + 1);
  std::vector<bool> pass_cut(n + 1, false), cb_cut(n + 1, false);
  pass_cut[0] = pass_cut[n] = cb_cut[0] = cb_cut[n] = true;
  for (uint32_t k = 0; k < points.size(); k++) {
    const HostPoint& p = points[k];
    pub_at[p.publish].push_back(k);
    wait_at[p.wait].push_back(k);
    pass_cut[p.publish] = pass_cut[p.wait] = cb_cut[p.wait] = true;
  }
  std::vector<uint32_t> pass_cuts, cb_cuts;
  for (uint32_t b = 0; b <= n; b++) {
    if (pass_cut[b]) pass_cuts.push_back(b);
    if (cb_cut[b]) cb_cuts.push_back(b);
  }
  std::vector<uint32_t> pub_seg(n + 1, 0);
  for (uint32_t j = 0, b = 1; b <= n; b++) {
    while (cb_cuts[j + 1] < b) j++;
    pub_seg[b] = j;
  }
  id<MTLSharedEvent> gpu_ev = [r.dev newSharedEvent];   // a fresh event: its values count from the current step
  if (!r.fence) r.fence = [r.dev newFence];
  if (!r.done_log) r.done_log = [r.dev newBufferWithLength:kDoneLog * 4 options:MTLResourceStorageModeShared];
  if (!gpu_ev || !r.fence || !r.done_log) throw std::runtime_error("host points: newSharedEvent, newFence or newBuffer failed");
  std::vector<HostTask*> claimed;                        // taken for this runner; given back if a later point refuses
  try {
    for (auto& p : points) {
      const void* free_task = nullptr;
      if (!p.task->owner.compare_exchange_strong(free_task, &r) && free_task != &r)
        throw std::runtime_error("host points: this task is installed on another runner (a task serves one runner)");
      if (free_task == nullptr) claimed.push_back(p.task.get());
    }
  } catch (...) {
    for (auto* t : claimed) t->owner.store(nullptr);
    throw;
  }
  old.swap(r.points);                                    // from here on nothing allocates or throws
  r.points = std::move(points);
  r.pub_at.swap(pub_at); r.wait_at.swap(wait_at); r.pass_cuts.swap(pass_cuts); r.cb_cuts.swap(cb_cuts); r.pub_seg.swap(pub_seg);
  r.gpu_ev = gpu_ev; r.event_base = r.step_base;
  r.error_off = error_offset; r.timeout_ms = timeout_ms; r.spin_us = spin_us;
  r.drop = drop; r.drop_step = drop_step; r.drop_point = drop_point;
  for (auto& p : old) {                                  // a task this runner no longer holds serves another again
    if (!p.task) continue;
    bool kept = false;
    for (auto& q : r.points) kept = kept || q.task == p.task;
    if (!kept) release_task(*p.task, &r);
  }
}

std::vector<std::shared_ptr<HostTask>> Runner::host_tasks() const {
  std::vector<std::shared_ptr<HostTask>> out;
  for (auto& p : impl->points) if (p.task) out.push_back(p.task);
  return out;
}

void Runner::release_host_tasks() {
  std::vector<std::shared_ptr<HostTask>> dying;   // the points hold null tasks before any task's destructor runs Python
  for (auto& p : impl->points) {
    if (p.task) release_task(*p.task, impl.get());   // released: the task serves another runner again
    dying.push_back(std::move(p.task));
  }
}

static uint64_t event_value(const RunnerImpl& r, uint64_t step, uint32_t point) {
  return (step - r.event_base) * r.points.size() + point + 1;
}

// One pass: ops [a, b) of a step, behind the fence of the pass before it (on any earlier command buffer).
static void encode_pass(RunnerImpl& r, id<MTLCommandBuffer> cb, uint32_t a, uint32_t b, bool reencode) {
  id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
  if (r.fence_live) [en waitForFence:r.fence];
  if (!r.resident)   // buffers a kernel reaches only through a GPU address are declared here too
    for (size_t i = 0; i < r.resources.size(); i++) [en useResource:r.resources[i] usage:r.resource_usage[i]];
  if (reencode) {
    for (uint32_t i = a; i < b; i++) {
      const Dispatch& d = r.ops[i];
      [en setComputePipelineState:d.pipeline->impl->pso];
      for (auto& bb : d.buffers) [en setBuffer:bb.buffer->impl->buf offset:bb.offset atIndex:bb.index];
      for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
      [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
    }
  } else {
    [en executeCommandsInBuffer:r.icb->icb withRange:NSMakeRange(a, b - a)];
  }
  [en updateFence:r.fence];
  [en endEncoding];
  r.fence_live = true;
}

// Encodes command buffer `seg` of `step`, which the caller commits: its passes, each followed by the signals of the
// points publishing at its end. After the step's last pass, StepState.done is copied into done_log (whether the next
// step begins live), and the next step's points that publish at boundary 0 are signalled.
static id<MTLCommandBuffer> encode_segment(RunnerImpl& r, uint64_t step, uint32_t seg, bool reencode) {
  const uint32_t n = (uint32_t)r.ops.size(), a = r.cb_cuts[seg], b = r.cb_cuts[seg + 1];
  id<MTLCommandBuffer> cb = [r.q commandBuffer];
  auto signal = [&](uint64_t s, uint32_t k) {
    if (!r.drop || s != r.drop_step || k != r.drop_point) [cb encodeSignalEvent:r.gpu_ev value:event_value(r, s, k)];
  };
  uint32_t start = a;
  for (uint32_t c : r.pass_cuts) {
    if (c <= a) continue;
    if (c > b) break;
    encode_pass(r, cb, start, c, reencode);
    if (c == n) {
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      [bl waitForFence:r.fence];
      [bl copyFromBuffer:r.state sourceOffset:r.done_off toBuffer:r.done_log destinationOffset:(step % kDoneLog) * 4 size:4];
      [bl updateFence:r.fence];
      [bl endEncoding];
    }
    for (uint32_t k : r.pub_at[c]) signal(step, k);
    if (c == n) for (uint32_t k : r.pub_at[0]) signal(step + 1, k);
    start = c;
  }
  return cb;
}

static std::string cb_error(id<MTLCommandBuffer> cb) {
  return std::string("command buffer failed: ") + (cb.error ? [cb.error.localizedDescription UTF8String] : "unknown error");
}

// Waits until the GPU signals `v`: the passes before the point's publish boundary completed and their writes are
// visible. `carrier` is the command buffer holding the signal (nil once it has been retired): if it finishes without
// the signal, the wait fails at once. Spins for spin_us, then blocks in 1 ms slices; fails after timeout_ms.
static bool wait_publish(const RunnerImpl& r, uint64_t v, id<MTLCommandBuffer> carrier, std::string& err) {
  const double t0 = now_ms();
  for (uint32_t spins = 0; r.gpu_ev.signaledValue < v; spins++) {
    if ((spins & 63) != 63) continue;
    const bool finished = carrier == nil || carrier.status >= MTLCommandBufferStatusCompleted;
    if (finished && ![r.gpu_ev waitUntilSignaledValue:v timeoutMS:10]) {   // a finished buffer's signal lands at once
      err = carrier != nil && carrier.status == MTLCommandBufferStatusError ? cb_error(carrier)
                                                                            : "its command buffer finished without the signal";
      return false;
    }
    const double elapsed = now_ms() - t0;
    if (elapsed > r.timeout_ms) { err = "the GPU did not reach the point within " + std::to_string(r.timeout_ms) + " ms"; return false; }
    if (elapsed * 1e3 > r.spin_us) [r.gpu_ev waitUntilSignaledValue:v timeoutMS:1];
  }
  // The event's value is read with a plain load: the host's later loads of what the GPU wrote must not run before it
  std::atomic_thread_fence(std::memory_order_acquire);
  return true;
}

// Waits for a committed command buffer; it never waits on the host, so only its GPU work decides how long.
static bool wait_cb(const RunnerImpl& r, id<MTLCommandBuffer> cb, std::string& err) {
  const double t0 = now_ms();
  while (cb.status < MTLCommandBufferStatusCompleted) {
    const double elapsed = now_ms() - t0;
    if (elapsed > r.timeout_ms) { err = "a command buffer did not complete within " + std::to_string(r.timeout_ms) + " ms"; return false; }
    if (elapsed * 1e3 > r.spin_us) usleep(100);
  }
  std::atomic_thread_fence(std::memory_order_acquire);   // as in wait_publish: the status is read with a plain load
  return true;
}

struct HostCb { id<MTLCommandBuffer> cb; uint64_t step; uint32_t seg; bool after_point; };   // after_point: committed once
                                                                                            // a host point's task returned

static RunnerStats run_host_steps(RunnerImpl& r, uint32_t max_steps, bool reencode, uint64_t max_tokens) {
  RunnerStats st;
  std::string& err = st.error;
  for (auto& p : r.points)
    if (!p.task) { err = "host steps: a task has been released"; return st; }
  const uint32_t n = (uint32_t)r.ops.size(), nseg = (uint32_t)r.cb_cuts.size() - 1, K = (uint32_t)r.points.size();
  const uint64_t S0 = r.step_base;
  uint64_t S1 = S0 + max_steps;                    // steps [S0, S1) may begin; a stop lowers S1 to the steps begun
  uint64_t begun = S0;                             // steps [S0, begun) have begun, and each runs to its end
  uint64_t enc_step = S0; uint32_t enc_seg = 0;    // the next command buffer to commit
  uint64_t srv_step = S0; uint32_t srv_k = 0;      // the next point to serve
  bool stop = false;
  std::deque<HostCb> pending;
  const double t0 = now_ms(), c0 = cpu_ms_now();
  const bool done_at_start = *(volatile uint32_t*)((char*)r.state.contents + r.done_off) != 0;   // nothing is in flight
  // A step's tasks run when the program was live as the step began: done at the end of the previous step, from
  // done_log. Reading StepState at the point itself would race an op that runs beside the task and sets done.
  auto live = [&](uint64_t s) {
    return s == S0 ? !done_at_start : ((volatile uint32_t*)r.done_log.contents)[(s - 1) % kDoneLog] == 0;
  };
  auto served = [&](uint64_t s, uint32_t k) { return s < srv_step || (s == srv_step && k < srv_k); };
  auto may_begin = [&](uint64_t s) {               // a step begins unless a stop came first
    if (s < begun) return true;
    if (stop || s >= S1) { S1 = begun; return false; }
    begun = s + 1;
    return true;
  };
  auto observe_state = [&]() {                     // the GPU's writes so far are visible: drain, check the stops
    drain_ring(r);
    st.done = *(volatile uint32_t*)((char*)r.state.contents + r.done_off) != 0;
    uint64_t seen; { std::lock_guard<std::mutex> lk(r.mu); seen = r.taken; }   // a task may drain() the tokens
    if (st.done || (max_tokens && seen >= max_tokens)) stop = true;
  };
  double prev_end = 0;                             // GPU end of the command buffer retired last (0: none in this run)
  auto retire = [&](bool block) {                  // retires completed command buffers, waiting for one if `block`
    while (!pending.empty() && err.empty()) {
      id<MTLCommandBuffer> cb = pending.front().cb;
      if (cb.status < MTLCommandBufferStatusCompleted) {
        if (!block) return;
        const double w0 = now_ms();
        const bool ok = wait_cb(r, cb, err);
        st.wait_ms += now_ms() - w0;
        if (!ok) return;
      }
      [cb waitUntilCompleted];                     // returns at once; the validation logs are complete only after it
      st.gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      st.command_buffers++;
      if (pending.front().after_point && prev_end > 0) st.handoff_gaps_ms.push_back((cb.GPUStartTime - prev_end) * 1e3);
      prev_end = cb.GPUEndTime;
      pending.pop_front();
      if (std::string e = failure(cb); !e.empty()) { err = e; return; }   // its error or validation reports stop the run
      block = false;
      observe_state();
    }
  };
  // Whatever stops the loop, a failed task, a timeout or an exception the pump itself meets, reaches the cleanup below
  try {
    for (;;) {
      // commit every command buffer whose waits have all been served
      while (err.empty() && enc_step < S1) {
        const uint32_t a = r.cb_cuts[enc_seg];
        bool ready = true;
        for (uint32_t k : r.wait_at[a]) ready = ready && served(enc_step, k);
        if (a == 0 && enc_step > S0) for (uint32_t k : r.wait_at[n]) ready = ready && served(enc_step - 1, k);
        if (!ready || (enc_seg == 0 && !may_begin(enc_step))) break;
        const double e0 = now_ms();
        const bool after_point = !r.wait_at[a].empty() || (a == 0 && enc_step > S0 && !r.wait_at[n].empty());
        @autoreleasepool {
          id<MTLCommandBuffer> cb = encode_segment(r, enc_step, enc_seg, reencode);
          pending.push_back({cb, enc_step, enc_seg, after_point});   // tracked before it is committed: the failure
          const double c0 = now_ms();                                // cleanup waits for every committed buffer
          st.encode_ms += c0 - e0;
          [cb commit];
          st.commit_ms += now_ms() - c0;
        }
        if (++enc_seg == nseg) { enc_seg = 0; enc_step++; }
      }
      if (!err.empty()) break;
      // serve the next point once its signal is committed; a point publishing at the run's first boundary needs none
      if (srv_step < S1) {
        const HostPoint& p = r.points[srv_k];
        const bool first = p.publish == 0 && srv_step == S0;
        const bool committed = first || (p.publish == 0 ? enc_step >= srv_step
                                                        : srv_step < enc_step || (srv_step == enc_step && r.pub_seg[p.publish] < enc_seg));
        if (committed && may_begin(srv_step)) {
          if (!first) {
            const uint64_t cs = p.publish == 0 ? srv_step - 1 : srv_step;
            const uint32_t cg = p.publish == 0 ? nseg - 1 : r.pub_seg[p.publish];
            id<MTLCommandBuffer> carrier = nil;
            for (auto& h : pending) if (h.step == cs && h.seg == cg) carrier = h.cb;
            const double w0 = now_ms();
            std::string werr;
            const bool ok = wait_publish(r, event_value(r, srv_step, srv_k), carrier, werr);
            st.wait_ms += now_ms() - w0;
            if (!ok) { err = "host step " + std::to_string(srv_step) + ", point " + std::to_string(srv_k) + ": " + werr; break; }
          }
          observe_state();
          if (live(srv_step)) {                    // a step that began after the program finished does no host work
            const HostCall call{srv_step, srv_k, r.state.contents};
            std::string terr;
            bool ok = false;
            try { ok = p.task->run(call, terr); }
            catch (const std::exception& e) { terr = e.what(); }
            catch (...) { terr = "the task threw an exception that is not a std::exception"; }   // an int, an NSException
            if (!ok) {
              err = "host step " + std::to_string(srv_step) + ", point " + std::to_string(srv_k) + ": " + (terr.empty() ? "the task failed" : terr);
              st.failed_task = (int32_t)srv_k;
              st.failed = p.task;
              break;
            }
            observe_state();                       // a task may set done itself: then no later step begins
          }
          if (++srv_k == K) { srv_k = 0; srv_step++; }
          retire(false);
          continue;
        }
      }
      if (pending.empty()) {                       // nothing committed is outstanding and nothing more can start
        if (enc_step < S1 || srv_step < S1) err = "host steps: no command buffer in flight and no point to serve";
        break;
      }
      retire(true);
      if (!err.empty()) break;
    }
  } catch (const std::exception& e) {
    if (err.empty()) err = std::string("host steps: ") + e.what();
  } catch (...) {
    if (err.empty()) err = "host steps: an exception that is not a std::exception";
  }
  if (!err.empty()) {
    // Committed command buffers never wait on the host and every kernel loop is bounded; the GPU watchdog ends a hung
    // one. They finish before StepState is touched, so the failure leaves no GPU work behind.
    for (auto& h : pending) {
      [h.cb waitUntilCompleted];
      st.gpu_ms += (h.cb.GPUEndTime - h.cb.GPUStartTime) * 1e3;
      st.command_buffers++;
      failure(h.cb);   // counts its validation reports; the run's error is the first failure's
    }
    pending.clear();
    volatile uint32_t* state = (volatile uint32_t*)r.state.contents;
    state[r.done_off / 4] = 1;
    if (state[r.error_off / 4] == 0) state[r.error_off / 4] = 4;
  }
  r.step_base = begun;
  drain_ring(r);
  { std::lock_guard<std::mutex> lk(r.mu); st.tokens = r.tokens; }       // this run's tokens, read under its own hold
  st.done = *(volatile uint32_t*)((char*)r.state.contents + r.done_off) != 0;
  st.steps_submitted = begun - S0;
  st.wall_ms = now_ms() - t0;
  st.host_busy_ms = cpu_ms_now() - c0;
  return st;
}

RunnerStats Runner::run(uint32_t max_steps, uint32_t steps_per_cb, uint32_t in_flight, bool reencode, uint64_t max_tokens) {
  RunnerImpl& r = *impl;
  Hold hold(r.owner, "Runner.run");
  {
    std::lock_guard<std::mutex> lk(r.mu);   // drain() may run on another thread meanwhile
    r.tokens.clear();                        // an earlier run's undrained tokens are not this run's
    r.taken = 0;
    r.tail = *(volatile uint32_t*)((char*)r.state.contents + r.tail_off);   // the ring tail lives in StepState: resume (or restart after a reset) from it
  }
  if (!r.points.empty()) return run_host_steps(r, max_steps, reencode, max_tokens);
  RunnerStats st;
  if (steps_per_cb == 0 || in_flight == 0) throw std::runtime_error("steps_per_cb and in_flight must be >= 1");
  double t0 = now_ms(), c0 = cpu_ms_now();
  std::deque<id<MTLCommandBuffer>> pending;
  uint64_t tokens_seen = 0;
  auto observe = [&](id<MTLCommandBuffer> cb) {
    double waiting = now_ms();
    [cb waitUntilCompleted];
    st.wait_ms += now_ms() - waiting;
    st.gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    st.command_buffers++;
    std::string e = failure(cb);   // every buffer's reports are counted; the first failure stops the pump
    if (st.error.empty()) st.error = e;
    drain_ring(r);
    st.done = *(volatile uint32_t*)((char*)r.state.contents + r.done_off) != 0;
    { std::lock_guard<std::mutex> lk(r.mu); tokens_seen = r.taken; }   // this call's tokens, whoever drained them
  };
  while (st.steps_submitted < max_steps && !st.done && st.error.empty() && (max_tokens == 0 || tokens_seen < max_tokens)) {
    @autoreleasepool {
      double encoding = now_ms();
      uint32_t n = std::min<uint64_t>(steps_per_cb, max_steps - st.steps_submitted);
      id<MTLCommandBuffer> cb = [r.q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];   // serial: step s+1 starts after step s
      if (!r.resident)   // buffers a kernel reaches only through a GPU address are declared here too
        for (size_t i = 0; i < r.resources.size(); i++) [en useResource:r.resources[i] usage:r.resource_usage[i]];
      if (reencode) {
        for (uint32_t s = 0; s < n; s++)
          for (auto& d : r.ops) {
            [en setComputePipelineState:d.pipeline->impl->pso];
            for (auto& b : d.buffers) [en setBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
            for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
            [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
          }
      } else {
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
  r.step_base += st.steps_submitted;   // HostCall.step counts the steps of either pump
  { std::lock_guard<std::mutex> lk(r.mu); st.tokens = r.tokens; }       // this run's tokens, read under its own hold
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
