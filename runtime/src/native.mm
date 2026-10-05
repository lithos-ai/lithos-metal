// nanobind layer over metal_core: keeps Python free of Objective-C and the core free of Python.
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <cstring>
#include "metal_core.h"

namespace nb = nanobind;
using namespace monolith;

NB_MODULE(_native, m) {
  m.doc() = "Metal runtime core (v0): device, buffers, runtime-compiled libraries, pipelines, timed dispatches";

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
      .def("barrier", [](Dispatch& d, bool on) -> Dispatch& { d.barrier_before = on; return d; }, nb::arg("on") = true, nb::rv_policy::reference_internal);

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

  nb::class_<RunnerStats>(m, "RunnerStats")
      .def_ro("steps_submitted", &RunnerStats::steps_submitted).def_ro("command_buffers", &RunnerStats::command_buffers)
      .def_ro("gpu_ms", &RunnerStats::gpu_ms).def_ro("wall_ms", &RunnerStats::wall_ms).def_ro("host_busy_ms", &RunnerStats::host_busy_ms)
      .def_ro("encode_ms", &RunnerStats::encode_ms).def_ro("commit_ms", &RunnerStats::commit_ms).def_ro("wait_ms", &RunnerStats::wait_ms)
      .def_ro("done", &RunnerStats::done).def_ro("error", &RunnerStats::error);

  nb::class_<Runner>(m, "Runner")
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
      .def("drain", &Runner::drain);
}
