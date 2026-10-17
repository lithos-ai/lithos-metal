// The runtime core, v0: device, buffers, runtime-compiled libraries, pipelines and timed command buffers of
// dispatches. This is what the bench harness and the kernel tests drive today and what the ICB builder, the
// host pump and the token ring build on (plan M2). Objective-C++ behind a C++ interface so the nanobind layer
// stays free of Objective-C.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace monolith {

struct DeviceImpl;
struct BufferImpl;
struct LibraryImpl;
struct PipelineImpl;
struct QueueImpl;

struct DeviceInfo {
  std::string name;
  int gpu_cores = 0;          // IORegistry gpu-core-count
  int apple_family = 0;       // highest supported MTLGPUFamilyApple<N>
  uint64_t max_buffer_length = 0;
  uint64_t recommended_working_set = 0;
  bool has_unified_memory = false;
};

class Device {
 public:
  Device();
  ~Device();
  DeviceInfo info() const;
  std::shared_ptr<DeviceImpl> impl;
};

class Buffer {
 public:
  // shared-storage buffer of `nbytes`, optionally initialized from `data`
  Buffer(const Device& d, size_t nbytes, const void* data = nullptr);
  // wraps an immutable, shared read-only file mapping with newBufferWithBytesNoCopy
  Buffer(const Device& d, const std::string& path, uint64_t offset, size_t nbytes);
  ~Buffer();
  size_t nbytes() const;
  void* contents() const;       // host pointer (shared storage / mmap)
  uint64_t gpu_address() const;
  std::shared_ptr<BufferImpl> impl;
};

class Library {
 public:
  // compiles MSL source at run time; `macros` become -D definitions; language_version e.g. 0x30002 (3.2), 0x40000
  Library(const Device& d, const std::string& source, const std::map<std::string, std::string>& macros,
          uint32_t language_version, bool fast_math);
  ~Library();
  std::shared_ptr<LibraryImpl> impl;
};

class Pipeline {
 public:
  Pipeline(const Library& lib, const std::string& function, bool support_icb);
  ~Pipeline();
  uint32_t max_threads_per_threadgroup() const;
  uint32_t thread_execution_width() const;
  std::shared_ptr<PipelineImpl> impl;
};

struct BufferBinding { uint32_t index; const Buffer* buffer; uint64_t offset; };
struct BytesBinding { uint32_t index; std::vector<uint8_t> bytes; };
struct BufferUse { const Buffer* buffer; bool write; };

struct Dispatch {
  const Pipeline* pipeline = nullptr;
  std::vector<BufferBinding> buffers;
  std::vector<BytesBinding> bytes;
  std::vector<std::pair<uint32_t, uint32_t>> threadgroup_memory;   // (index, length)
  uint32_t grid[3] = {1, 1, 1};        // threadgroups
  uint32_t threadgroup[3] = {1, 1, 1}; // threads per threadgroup
  bool barrier_before = false;         // this dispatch waits for every dispatch before it (ICBs, concurrent encoders)
  std::vector<BufferUse> uses;         // buffers the kernel reaches only through GPU addresses: Queue.run and
                                       // Queue.profile make them resident for its dispatch (useResource)
};

// `error`: the command buffer's error (empty = clean). A Metal shader-validation report leaves the command buffer
// completed without one, so run() raises the reports instead, one "shader validation: …" line each: a test fails
// on a report whether or not it checks `error`.
struct RunResult { double gpu_ms; double wall_ms; std::string error; };

// The shader-validation reports read so far from this process's completed command buffers, on every path that commits
// one: the test suite fails a test phase in which this grows, where a harness caught the raised error or a thread
// raised it.
uint64_t validation_reports();

class Queue {
 public:
  explicit Queue(const Device& d);
  ~Queue();
  // one command buffer holding `dispatches` in order (serial encoder, or concurrent with explicit barriers),
  // committed and waited for; returns GPU time from the command buffer timestamps
  RunResult run(const std::vector<Dispatch>& dispatches, bool concurrent);
  // per-dispatch GPU time: one compute encoder per dispatch with timestamp counter samples at its start and end
  // (MTLCounterSamplingPointAtStageBoundary, the granularity Apple GPUs support); returns (start_ms, end_ms) per
  // dispatch relative to the first start. The encoder boundaries add ~µs gaps that do not exist in the ICB replay.
  std::vector<std::pair<double, double>> profile(const std::vector<Dispatch>& dispatches);
  bool supports_profiling() const;
  std::shared_ptr<QueueImpl> impl;
};

// ---------------------------------------------------------------------------------------------------------------
// The step program (design §5.1, §5.4): the ops of one step encoded ONCE into an indirect command buffer and
// replayed by the host pump, which keeps a few bounded command buffers in flight, drains the token ring and stops
// when StepState.done is set. Pipelines used in an ICB must be created with support_icb = true; ICBs cannot carry
// setBytes, so every op reads its parameters from buffers.
struct IcbImpl;
struct RunnerImpl;

class Icb {
 public:
  // `ops` in program order; an op's barrier makes it wait for every command before it in the buffer — measured
  // (tools/bench, 2026-09-24): `setBarrier` on an ICB command orders that command behind all preceding ones, and
  // commands without it may start while their predecessors run. Buffer offsets must fit 32 bits (Metal's ICB limit)
  // — slabs are mapped so that they do.
  Icb(const Device& d, const std::vector<Dispatch>& ops);
  ~Icb();
  size_t count() const;
  std::shared_ptr<IcbImpl> impl;
};

// Host steps: a step pauses for CPU work between two of its GPU passes. A running dispatch has no documented view of
// CPU writes, and the GPU watchdog kills a command buffer parked at an event wait for 5 s (probes/p16_host_handoff:
// 4.5 s survives), so the GPU never waits for the host: the pass that ends at a point's publish boundary signals an
// MTLSharedEvent, the host waits for the signal and runs the point's task, and only then commits the command buffer
// that starts at the point's wait boundary. A slow task idles the GPU; it cannot trip the watchdog.
struct HostCall {
  uint64_t step;        // the runner's absolute step index (it counts every step the runner has begun)
  uint32_t point;       // the host point's index in the program
  void* step_state;     // StepState. A task that reads it declares reads=["step_state"] on its point, so that no op
                        // running beside the task writes it (Program.check_host_points)
};

class HostTask {
 public:
  virtual ~HostTask() = default;
  // Runs on the pump thread once the GPU completed the ops before the point's publish boundary; the ops from its wait
  // boundary on are committed after it returns. Returns false with `error` set to stop the run. A task bounds its own
  // duration (its I/O carries timeouts): the runner cannot interrupt it.
  virtual bool run(const HostCall& call, std::string& error) = 0;
};

struct HostPoint {
  uint32_t publish = 0;   // boundary 0..n: the task starts once ops [0, publish) of the step completed (0: the previous step)
  uint32_t wait = 0;      // boundary publish..n: ops [wait, n) of the step need the task (n: the next step needs it)
  std::shared_ptr<HostTask> task;
};

struct RunnerStats {
  uint64_t steps_submitted = 0;
  uint64_t command_buffers = 0;
  double gpu_ms = 0;          // sum of the command buffers' GPU times
  double wall_ms = 0;         // wall time of run()
  double host_busy_ms = 0;    // CPU time spent by the pump thread (encode + drain), from getrusage
  double encode_ms = 0;       // host wall time preparing command buffers
  double commit_ms = 0;       // host wall time in Metal commit()
  double wait_ms = 0;         // host wall time waiting for completion (overlaps GPU time)
  bool done = false;          // StepState.done was observed
  std::string error;          // the first failed command buffer's error and validation reports; stops the pump (Engine.run raises it)
  int32_t failed_task = -1;   // the host point whose task failed the run (-1: none did)
  std::shared_ptr<HostTask> failed;      // that point's task itself: the caller reads its exception without resolving
                                         // the index against a list another thread may have replaced meanwhile. A
                                         // Python task's exception is the runner's to keep, so read it while the
                                         // runner lives (Engine.run reads it inside the call that returns these stats)
  std::vector<double> handoff_gaps_ms;   // per command buffer committed once a host point's task returned: its GPU start
                                         // minus the GPU end of the one before it, the GPU's idle time at that handoff
  std::vector<int32_t> tokens;           // the tokens the ring held when the run ended, read while the run still owned
                                         // the runner: a run on another thread cannot clear them first. drain() still
                                         // hands the same tokens to a caller that takes them that way
};

class Runner {
 public:
  // `resources`: every buffer the program touches (ICB execution needs them made resident explicitly).
  // `read_only_resources`: the immutable subset (weights/parameters); other buffers permit writes.
  // `step_state` holds the StepState struct; `done_offset` / `ring_head_offset` / `ring_tail_offset` are its field
  // offsets. `ring` is the token ring: `ring_capacity` 8-byte slots, each `(sequence << 32) | token` written by the
  // GPU as one aligned store; the host drains by sequence and publishes `ring_tail` for the GPU's overflow check.
  Runner(const Device& d, const Icb& icb, const std::vector<Dispatch>& ops, std::vector<const Buffer*> resources,
         const Buffer& step_state, uint32_t done_offset, uint32_t ring_head_offset, uint32_t ring_tail_offset,
         const Buffer& ring, uint32_t ring_capacity, std::vector<const Buffer*> read_only_resources = {});
  ~Runner();
  // Replays the step program up to `max_steps` times: `steps_per_cb` steps per command buffer (the max_cb_ms
  // control), `in_flight` command buffers queued ahead. `reencode` = the fallback path (fresh encoder per step,
  // same ops) instead of ICB replay. Blocks until done / max_steps; tokens are collected as buffers complete.
  // `max_tokens` > 0 stops submitting once that many tokens have been drained during this call (a speculative
  // program commits several tokens per step, so a step count over-runs); the buffers already queued still complete.
  // With host points the command buffers are cut at the steps' boundaries and at the points' wait boundaries instead,
  // so steps_per_cb and in_flight do not apply; a run always submits whole steps. A run drops the tokens an earlier run
  // left undrained. One run() or set_host_points() at a time: a call while another runs, on another thread or from
  // one of the runner's own host tasks, throws.
  RunnerStats run(uint32_t max_steps, uint32_t steps_per_cb, uint32_t in_flight, bool reencode, uint64_t max_tokens = 0);
  // Switches run() to host steps. Points are in program order: 0 <= publish <= wait <= op count, publish boundaries
  // never decrease. `error_offset`: StepState.error, set to 4 when a host step fails. `timeout_ms`: the longest run()
  // waits for the GPU to reach a publish boundary or finish a command buffer; `spin_us`: how long such a wait spins
  // before it blocks.
  void set_host_points(std::vector<HostPoint> points, uint32_t error_offset, uint32_t timeout_ms, uint32_t spin_us = 20000);
  // The installed tasks, and their release: the bindings' garbage-collector support (a Python task's callable usually
  // reaches the object that owns this runner).
  std::vector<std::shared_ptr<HostTask>> host_tasks() const;
  void release_host_tasks();
  // Tokens drained so far (in ring order); cleared by the call.
  std::vector<int32_t> drain();
  std::shared_ptr<RunnerImpl> impl;
};

}  // namespace monolith
