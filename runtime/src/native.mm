// nanobind layer over metal_core: keeps Python free of Objective-C and the core free of Python.
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <tuple>
#include "metal_core.h"

namespace nb = nanobind;
using namespace monolith;

// A host task written in Python: fn(step, point). run() is called with the GIL released, so the call takes it; an
// exception fails the run with its message and stays on the task (take_exception) for the caller to raise. The slot is
// the task's own, so a task serves one runner: Engine wraps each callable in a CallableTask of its own.
struct CallableTask : HostTask {
  nb::callable fn;
  nb::object exception;
  explicit CallableTask(nb::callable f) : fn(std::move(f)) {}
  bool run(const HostCall& call, std::string& error) override {
    nb::gil_scoped_acquire gil;
    if (!fn.is_valid()) { error = "the task's callable has been released"; return false; }
    try {
      fn(call.step, call.point);
      return true;
    } catch (nb::python_error& e) {
      error = e.what();
      exception = nb::borrow(e.value());
      return false;
    }
  }
};

// Test support: a C++ task that throws, at `step`, what a C++ task may throw besides a std::exception: an int, or an
// NSException (`objc`).
struct ThrowingTask : HostTask {
  uint64_t step;
  bool objc;
  ThrowingTask(uint64_t s, bool o) : step(s), objc(o) {}
  bool run(const HostCall& call, std::string&) override {
    if (call.step == step) {
      if (objc) [NSException raise:NSGenericException format:@"thrown at step %llu", (unsigned long long)step];
      throw 42;
    }
    return true;
  }
};

// Garbage-collector support. A task's callable usually reaches the Engine that owns the Runner holding the task, a cycle
// through C++ references the collector cannot see unless these types report and break them.
static int callable_task_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  CallableTask* task = nb::inst_ptr<CallableTask>(self);
  Py_VISIT(task->fn.ptr());
  Py_VISIT(task->exception.ptr());                    // its traceback's frames usually reach the Engine too
  return 0;
}
static int callable_task_clear(PyObject* self) {
  if (!nb::inst_ready(self)) return 0;
  CallableTask* task = nb::inst_ptr<CallableTask>(self);
  nb::callable fn = std::move(task->fn);              // the task holds neither before they die
  nb::object exception = std::move(task->exception);
  return 0;
}
static int runner_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  // A task handed in from Python is a shared_ptr whose deleter holds one reference to the Python object, per control
  // block; a task made in C++ holds none. Visit exactly those references, without making a wrapper.
  std::vector<const void*> blocks;
  for (auto& task : nb::inst_ptr<Runner>(self)->host_tasks()) {
    auto* owner = std::get_deleter<nb::detail::py_deleter>(task);
    if (!owner || std::find(blocks.begin(), blocks.end(), owner) != blocks.end()) continue;
    blocks.push_back(owner);
    Py_VISIT(owner->o);
  }
  return 0;
}
static int runner_clear(PyObject* self) {
  if (!nb::inst_ready(self)) return 0;
  nb::inst_ptr<Runner>(self)->release_host_tasks();
  return 0;
}
static PyType_Slot callable_task_slots[] = {{Py_tp_traverse, (void*)callable_task_traverse}, {Py_tp_clear, (void*)callable_task_clear}, {0, nullptr}};
static PyType_Slot runner_slots[] = {{Py_tp_traverse, (void*)runner_traverse}, {Py_tp_clear, (void*)runner_clear}, {0, nullptr}};

NB_MODULE(_native, m) {
  m.doc() = "Metal runtime core (v0): device, buffers, runtime-compiled libraries, pipelines, timed dispatches";

  m.def("validation_reports", &validation_reports, "shader-validation reports read from completed command buffers so far");

  nb::class_<DeviceInfo>(m, "DeviceInfo")
      .def_ro("name", &DeviceInfo::name).def_ro("gpu_cores", &DeviceInfo::gpu_cores)
      .def_ro("apple_family", &DeviceInfo::apple_family).def_ro("max_buffer_length", &DeviceInfo::max_buffer_length)
      .def_ro("recommended_working_set", &DeviceInfo::recommended_working_set)
      .def_ro("has_unified_memory", &DeviceInfo::has_unified_memory);

  nb::class_<Device>(m, "Device").def(nb::init<>()).def("info", &Device::info);

  nb::class_<Buffer>(m, "Buffer")
      .def("__init__", [](Buffer* self, const Device& d, size_t nbytes) { new (self) Buffer(d, nbytes); }, nb::arg("device"), nb::arg("nbytes"))
      .def("__init__", [](Buffer* self, const Device& d, nb::bytes data) { new (self) Buffer(d, data.size(), data.c_str()); }, nb::arg("device"), nb::arg("data"))
      .def_static("from_file", [](const Device& d, const std::string& path, uint64_t offset, size_t nbytes) { return new Buffer(d, path, offset, nbytes); },
                  nb::arg("device"), nb::arg("path"), nb::arg("offset"), nb::arg("nbytes"))
      .def_prop_ro("nbytes", &Buffer::nbytes)
      .def_prop_ro("gpu_address", &Buffer::gpu_address)
      .def("read", [](const Buffer& b, size_t offset, size_t n) {
        if (offset + n > b.nbytes()) throw std::out_of_range("Buffer.read out of range");
        return nb::bytes((const char*)b.contents() + offset, n); }, nb::arg("offset") = 0, nb::arg("nbytes") = 0)
      .def("write", [](Buffer& b, nb::bytes data, size_t offset) {
        if (offset + data.size() > b.nbytes()) throw std::out_of_range("Buffer.write out of range");
        std::memcpy((char*)b.contents() + offset, data.c_str(), data.size()); }, nb::arg("data"), nb::arg("offset") = 0)
      .def("fill", [](Buffer& b, uint8_t v) { std::memset(b.contents(), v, b.nbytes()); });

  nb::class_<Library>(m, "Library")
      .def(nb::init<const Device&, const std::string&, const std::map<std::string, std::string>&, uint32_t, bool>(),
           nb::arg("device"), nb::arg("source"), nb::arg("macros") = std::map<std::string, std::string>{},
           nb::arg("language_version") = 0, nb::arg("fast_math") = false);

  nb::class_<Pipeline>(m, "Pipeline")
      .def(nb::init<const Library&, const std::string&, bool>(), nb::arg("library"), nb::arg("function"), nb::arg("support_icb") = false)
      .def_prop_ro("max_threads_per_threadgroup", &Pipeline::max_threads_per_threadgroup)
      .def_prop_ro("thread_execution_width", &Pipeline::thread_execution_width);

  nb::class_<Dispatch>(m, "Dispatch")
      .def(nb::init<>())
      .def("pipeline", [](Dispatch& d, const Pipeline& p) -> Dispatch& { d.pipeline = &p; return d; }, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
      .def("buffer", [](Dispatch& d, uint32_t index, const Buffer& b, uint64_t offset) -> Dispatch& { d.buffers.push_back({index, &b, offset}); return d; },
           nb::arg("index"), nb::arg("buffer"), nb::arg("offset") = 0, nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
      .def("bytes", [](Dispatch& d, uint32_t index, nb::bytes data) -> Dispatch& {
        d.bytes.push_back({index, std::vector<uint8_t>((const uint8_t*)data.c_str(), (const uint8_t*)data.c_str() + data.size())}); return d; },
           nb::arg("index"), nb::arg("data"), nb::rv_policy::reference_internal)
      .def("threadgroup_memory", [](Dispatch& d, uint32_t index, uint32_t length) -> Dispatch& { d.threadgroup_memory.push_back({index, length}); return d; }, nb::rv_policy::reference_internal)
      .def("grid", [](Dispatch& d, uint32_t x, uint32_t y, uint32_t z) -> Dispatch& { d.grid[0] = x; d.grid[1] = y; d.grid[2] = z; return d; },
           nb::arg("x"), nb::arg("y") = 1, nb::arg("z") = 1, nb::rv_policy::reference_internal)
      .def("threadgroup", [](Dispatch& d, uint32_t x, uint32_t y, uint32_t z) -> Dispatch& { d.threadgroup[0] = x; d.threadgroup[1] = y; d.threadgroup[2] = z; return d; },
           nb::arg("x"), nb::arg("y") = 1, nb::arg("z") = 1, nb::rv_policy::reference_internal)
      .def("barrier", [](Dispatch& d, bool on) -> Dispatch& { d.barrier_before = on; return d; }, nb::arg("on") = true, nb::rv_policy::reference_internal)
      .def("use", [](Dispatch& d, const Buffer& b, bool write) -> Dispatch& { d.uses.push_back({&b, write}); return d; },
           nb::arg("buffer"), nb::arg("write") = false, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>());

  nb::class_<RunResult>(m, "RunResult")
      .def_ro("gpu_ms", &RunResult::gpu_ms).def_ro("wall_ms", &RunResult::wall_ms).def_ro("error", &RunResult::error);

  nb::class_<Queue>(m, "Queue")
      .def(nb::init<const Device&>(), nb::arg("device"))
      .def("run", &Queue::run, nb::arg("dispatches"), nb::arg("concurrent") = false)
      .def("profile", &Queue::profile, nb::arg("dispatches"))
      .def("supports_profiling", &Queue::supports_profiling);

  nb::class_<Icb>(m, "Icb")
      .def(nb::init<const Device&, const std::vector<Dispatch>&>(), nb::arg("device"), nb::arg("ops"), nb::keep_alive<1, 3>())
      .def_prop_ro("count", &Icb::count);

  nb::class_<HostTask>(m, "HostTask");    // implemented in C++ (later tiers) or by CallableTask
  nb::class_<CallableTask, HostTask>(m, "CallableTask", nb::type_slots(callable_task_slots))
      .def(nb::init<nb::callable>(), nb::arg("fn"))
      .def("take_exception", [](CallableTask& t) -> nb::object {
             nb::object e = std::move(t.exception);
             if (!e.is_valid()) return nb::none();
             return e; },
           "the exception the task's last failing call raised (None if none); the task no longer holds it");
  nb::class_<ThrowingTask, HostTask>(m, "_ThrowingTask").def(nb::init<uint64_t, bool>(), nb::arg("step"), nb::arg("objc") = false);

  nb::class_<RunnerStats>(m, "RunnerStats")
      .def_ro("steps_submitted", &RunnerStats::steps_submitted).def_ro("command_buffers", &RunnerStats::command_buffers)
      .def_ro("gpu_ms", &RunnerStats::gpu_ms).def_ro("wall_ms", &RunnerStats::wall_ms).def_ro("host_busy_ms", &RunnerStats::host_busy_ms)
      .def_ro("encode_ms", &RunnerStats::encode_ms).def_ro("commit_ms", &RunnerStats::commit_ms).def_ro("wait_ms", &RunnerStats::wait_ms)
      .def_ro("done", &RunnerStats::done).def_ro("error", &RunnerStats::error)
      .def_ro("failed_task", &RunnerStats::failed_task).def_ro("failed", &RunnerStats::failed).def_ro("handoff_gaps_ms", &RunnerStats::handoff_gaps_ms)
      .def_ro("tokens", &RunnerStats::tokens);

  nb::class_<Runner>(m, "Runner", nb::type_slots(runner_slots))
      .def("__init__", [](Runner* self, const Device& d, const Icb& icb, const std::vector<Dispatch>& ops, std::vector<const Buffer*> resources,
                          const Buffer& state, uint32_t done_offset, uint32_t ring_head_offset, uint32_t ring_tail_offset, const Buffer& ring, uint32_t ring_capacity,
                          std::vector<const Buffer*> read_only_resources) {
             new (self) Runner(d, icb, ops, resources, state, done_offset, ring_head_offset, ring_tail_offset, ring, ring_capacity, read_only_resources); },
           nb::arg("device"), nb::arg("icb"), nb::arg("ops"), nb::arg("resources"), nb::arg("step_state"), nb::arg("done_offset"),
           nb::arg("ring_head_offset"), nb::arg("ring_tail_offset"), nb::arg("ring"), nb::arg("ring_capacity"),
           nb::arg("read_only_resources") = std::vector<const Buffer*>{},
           nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 10>())
      .def("run", &Runner::run, nb::arg("max_steps"), nb::arg("steps_per_cb") = 8, nb::arg("in_flight") = 3, nb::arg("reencode") = false, nb::arg("max_tokens") = 0,
           nb::call_guard<nb::gil_scoped_release>())
      .def("set_host_points", [](Runner& r, const std::vector<std::tuple<uint32_t, uint32_t, std::shared_ptr<HostTask>>>& points,
                                 uint32_t error_offset, uint32_t timeout_ms, uint32_t spin_us) {
             std::vector<HostPoint> pts;
             for (auto& [publish, wait, task] : points) pts.push_back({publish, wait, task});
             r.set_host_points(std::move(pts), error_offset, timeout_ms, spin_us); },
           nb::arg("points"), nb::arg("error_offset"), nb::arg("timeout_ms") = 20000, nb::arg("spin_us") = 20000)
      .def("host_tasks", &Runner::host_tasks, "the installed tasks, one per host point in program order")
      .def("drain", &Runner::drain);
}
