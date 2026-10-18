// nanobind layer over metal_core: keeps Python free of Objective-C and the core free of Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <array>
#include <cstring>
#include <stdexcept>
#include "basalt_cpu.h"
#include "metal_core.h"

namespace nb = nanobind;
using namespace monolith;

// ---- basalt's two CPU Q2_0 kernels (basalt_cpu.h) ------------------------------------------------------------
// Caller-owned numpy arrays, ActQ-shaped and C-contiguous: nanobind refuses a wrong dtype, rank or layout (the
// arguments are .noconvert(), so it cannot quietly write into a copy), and these checks refuse a short buffer - a
// kernel here reads and writes raw pointers, so the binding is where "every store is inside its binding" is kept.
namespace bc = monolith::basalt_cpu;
using F32In1 = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U8In1 = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I8Out1 = nb::ndarray<int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32Out1 = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F32Out1 = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I8In2 = nb::ndarray<const int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F32In2 = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F32Out2 = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

static void need(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

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
      .def("barrier", [](Dispatch& d, bool on) -> Dispatch& { d.barrier_before = on; return d; }, nb::arg("on") = true, nb::rv_policy::reference_internal)
      .def("use", [](Dispatch& d, const Buffer& b) -> Dispatch& { d.uses.push_back(&b); return d; },
           nb::arg("buffer"), nb::rv_policy::reference_internal, nb::keep_alive<1, 2>());

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

  m.def("basalt_act_quant_q8_1", [](F32In1 x, I8Out1 q, F32Out1 scale, I32Out1 sum, F32Out1 hx) {
    const size_t len = x.shape(0);             // checked at full width: a narrowing cast first would accept a
    need(len > 0 && len % bc::QKA == 0 && len <= (size_t)bc::H,   // length whose low 32 bits look valid
         "basalt_act_quant_q8_1: len(x) must be a positive multiple of 32, at most 2560");
    const int n = (int)len;
    need(q.shape(0) == (size_t)bc::H, "basalt_act_quant_q8_1: q must hold H = 2560 int8 codes");
    need(scale.shape(0) == (size_t)bc::MAXC && sum.shape(0) == (size_t)bc::MAXC && hx.shape(0) == (size_t)bc::MAXC,
         "basalt_act_quant_q8_1: scale, sum and hx must each hold MAXC = 80 elements");
    bc::ActQ a;
    nb::gil_scoped_release unlock;
    bc::act_quant_q8_1(x.data(), n, a);
    std::memcpy(q.data(), a.q, (size_t)n);                                      // only what the kernel wrote
    std::memcpy(scale.data(), a.scale, (size_t)a.nchunks * sizeof(float));
    std::memcpy(sum.data(), a.sum, (size_t)a.nchunks * sizeof(int32_t));
    std::memcpy(hx.data(), a.hx, (size_t)a.nchunks * sizeof(float));
    return a.nchunks;
  }, nb::arg("x").noconvert(), nb::arg("q").noconvert(), nb::arg("scale").noconvert(), nb::arg("sum").noconvert(),
     nb::arg("hx").noconvert(),
     "basalt's act_quant_q8_1 on ARM NEON: writes q[:n] and scale/sum/hx[:n // 32], and returns nchunks");

  m.def("basalt_q2_0_gguf_rows_multi", [](U8In1 w, size_t row_bytes, int nblocks, I8In2 q, F32In2 scale, F32In2 hx,
                                          F32Out2 out, int r0, int r1) {
    const int nt = (int)q.shape(0);
    const size_t nrows = out.shape(1);        // full width: the row stride below, and a narrowing cast here would
                                              // accept a shape whose low 32 bits look small and stride past it
    need(nt >= 1 && nt <= bc::MAXT, "basalt_q2_0_gguf_rows_multi: nt must be 1..8");
    need(scale.shape(0) == (size_t)nt && hx.shape(0) == (size_t)nt && out.shape(0) == (size_t)nt,
         "basalt_q2_0_gguf_rows_multi: q, scale, hx and out must agree on nt");
    need(q.shape(1) == (size_t)bc::H, "basalt_q2_0_gguf_rows_multi: q rows must hold H = 2560 int8 codes");
    need(scale.shape(1) == (size_t)bc::MAXC && hx.shape(1) == (size_t)bc::MAXC,
         "basalt_q2_0_gguf_rows_multi: scale and hx rows must hold MAXC = 80 elements");
    // Every bound divides rather than multiplies: the products overflow, and a wrapped product passes the check it
    // is meant to fail (nblocks = 2^25 makes QK * nblocks reach INT_MIN, row_bytes * nrows reaches 2^64 and wraps
    // to 0), which the kernel then reads past the ActQ arrays and past w.
    need(nblocks >= 1 && nblocks <= bc::H / bc::QK, "basalt_q2_0_gguf_rows_multi: nblocks must be 1..40");
    need(row_bytes >= (size_t)bc::BB * (size_t)nblocks, "basalt_q2_0_gguf_rows_multi: row_bytes is below 18 * nblocks");
    need(r0 >= 0 && r1 >= r0 && (size_t)r1 <= nrows, "basalt_q2_0_gguf_rows_multi: need 0 <= r0 <= r1 <= out.shape[1]");
    need(nrows == 0 || row_bytes <= w.size() / nrows, "basalt_q2_0_gguf_rows_multi: w is shorter than row_bytes * nrows");
    std::array<bc::ActQ, bc::MAXT> act{};
    const bc::ActQ* ap[bc::MAXT];
    float* op[bc::MAXT];
    for (int t = 0; t < nt; ++t) {
      std::memcpy(act[t].q, q.data() + (size_t)t * bc::H, bc::H);
      std::memcpy(act[t].scale, scale.data() + (size_t)t * bc::MAXC, bc::MAXC * sizeof(float));
      std::memcpy(act[t].hx, hx.data() + (size_t)t * bc::MAXC, bc::MAXC * sizeof(float));
      ap[t] = &act[t];                                                          // sum and nchunks stay zero: unread
      op[t] = out.data() + (size_t)t * nrows;   // nrows is size_t: the stride is the row's real width
    }
    nb::gil_scoped_release unlock;
    bc::q2_0_gguf_rows_multi(w.data(), row_bytes, nblocks, ap, nt, op, r0, r1);
  }, nb::arg("w").noconvert(), nb::arg("row_bytes"), nb::arg("nblocks"), nb::arg("q").noconvert(),
     nb::arg("scale").noconvert(), nb::arg("hx").noconvert(), nb::arg("out").noconvert(), nb::arg("r0"), nb::arg("r1"),
     "basalt's q2_0_gguf_rows_multi on ARM NEON: writes out[t][r] for r0 <= r < r1 and nothing else");
}
