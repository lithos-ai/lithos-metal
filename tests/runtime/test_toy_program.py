"""M2's exit gate: a two-op self-advancing program replayed from ONE encode for 1,000 steps, tokens drained from the
ring with no waitUntilCompleted on the hot path; ICB replay ≡ re-encode; early exit after `done`."""
import struct

import numpy as np
import pytest

from monolith.core.step_state import StepStateLayout
from monolith.runtime import BufferSpec, Engine, KernelSpec, OpSpec, Program

LAYOUT = StepStateLayout(t_max=8, gamma_max=7)
H = 4096

SRC = """
#include <metal_stdlib>
using namespace metal;
""" + LAYOUT.to_msl() + """
struct Params { uint n_stop; uint ring_cap; uint width; uint pad; };
// op A (MAP): every element of h advances with the step; returns at once when the program is done
kernel void toy_map(device float* h [[buffer(0)]], device const StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  if (i < p.width) h[i] = h[i] * 0.5f + float(st->step + 1);
}
// op B (SERIAL, one thread): consume h[0], emit a token, advance the state, stop at n_stop
kernel void toy_serial(device const float* h [[buffer(0)]], device StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                       device ulong* ring [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  uint s = st->step + 1;
  uint head = st->ring_head;
  if (head - st->ring_tail >= p.ring_cap) { st->error = 1; st->done = 1; return; }   // ring overflow: the host fell behind
  int token = int(s * 7u) + int(h[0] > 0.0f ? 0 : 1000000);
  ring[head % p.ring_cap] = (ulong(head + 1u) << 32) | ulong(uint(token));        // one aligned 8-byte store: token + sequence
  st->ring_head = head + 1u;
  st->step = s;
  st->position = st->position + 1;
  if (s >= p.n_stop) st->done = 1;
}
"""


def _program(n_stop: int, ring_cap: int = 256) -> Program:
    params = struct.pack("<IIII", n_stop, ring_cap, H, 0)
    return Program(
        kernels={"toy": KernelSpec(SRC, "toy_map"), "toy_serial": KernelSpec(SRC, "toy_serial")},
        buffers={"h": BufferSpec(H * 4, np.ones(H, np.float32).tobytes(), "arena"),
                 "step_state": BufferSpec(LAYOUT.size, LAYOUT.pack({}), "step_state"),
                 "params": BufferSpec(16, params, "params"),
                 "ring": BufferSpec(ring_cap * 8, None, "ring")},
        ops=[OpSpec("toy", [(0, "h", 0), (1, "step_state", 0), (2, "params", 0)], (H // 384 + 1, 1, 1), (384, 1, 1), True, name="map"),
             OpSpec("toy_serial", [(0, "h", 0), (1, "step_state", 0), (2, "params", 0), (3, "ring", 0)], (1, 1, 1), (32, 1, 1), True, name="advance")],
        ring_capacity=ring_cap, layout=LAYOUT)


def _expected_h(steps: int) -> float:
    h = 1.0
    for s in range(steps):
        h = h * 0.5 + (s + 1)
    return h


def test_icb_replay_1000_steps_from_one_encode():
    eng = Engine(_program(n_stop=1000))
    rep = eng.run(1000, steps_per_cb=25, in_flight=3)          # 40 command buffers, 25 steps each
    assert rep.steps == 1000 and rep.done and rep.command_buffers == 40
    assert rep.tokens == [7 * s for s in range(1, 1001)]       # every step's token, in order, none lost
    st = eng.state()
    assert st["step"] == 1000 and st["position"] == 1000 and st["ring_head"] == 1000
    h = np.frombuffer(eng.read("h"), dtype=np.float32)
    assert np.allclose(h, _expected_h(1000), rtol=1e-6)
    # the pump thread's own CPU time; each toy step is ~13 us of GPU time so the per-command-buffer host cost is a
    # large share here — the < 5 % gate applies to ~60 ms tokens (design §5.4) and is measured in M4
    assert rep.host_fraction < 0.9
    print(f"\n1000 steps: gpu {rep.gpu_ms:.2f} ms, wall {rep.wall_ms:.2f} ms, host busy {rep.host_busy_ms:.2f} ms ({100 * rep.host_fraction:.1f} %)")


def test_reencode_fallback_matches_icb():
    a = Engine(_program(n_stop=300)); ra = a.run(300, steps_per_cb=10, in_flight=2, reencode=False)
    b = Engine(_program(n_stop=300)); rb = b.run(300, steps_per_cb=10, in_flight=2, reencode=True)
    assert ra.tokens == rb.tokens and a.state() == b.state()
    assert a.read("h") == b.read("h")                          # bit-identical


def test_early_exit_after_done_and_ring_wrap():
    eng = Engine(_program(n_stop=100, ring_cap=64))            # the ring wraps; the pump drains after every buffer
    rep = eng.run(1000, steps_per_cb=4, in_flight=2)           # asks for 1000 but the program stops itself at 100
    assert rep.done and rep.steps <= 100 + 4 * 2 and eng.state()["step"] == 100 and eng.state()["error"] == 0
    assert rep.tokens == [7 * s for s in range(1, 101)]
    h = np.frombuffer(eng.read("h"), dtype=np.float32)
    assert np.allclose(h, _expected_h(100), rtol=1e-6)        # the queued steps after done returned at their first instruction


def test_program_json_roundtrip(tmp_path):
    p = _program(5)
    p.save(tmp_path / "program.json")
    q = Program.load(tmp_path / "program.json")
    assert q.to_json() == p.to_json() and q.ops[1].name == "advance" and q.buffers["params"].init == p.buffers["params"].init


def test_pipeline_cache_survives_engine_replacement_without_reusing_state():
    from monolith.runtime import _native as nt
    device, cache = nt.Device(), {}
    first = Engine(_program(2), device, pipeline_cache=cache)
    pipelines = dict(first.pipelines)
    assert first.run(2).tokens == [7, 14]
    del first
    second = Engine(_program(3), device, pipeline_cache=cache)
    assert all(second.pipelines[name] is pipeline for name, pipeline in pipelines.items())
    assert second.run(3).tokens == [7, 14, 21]
    assert len(cache) == 2
    changed = Engine(_program(1), device, fast_math=True, pipeline_cache=cache)
    assert all(changed.pipelines[name] is not pipeline for name, pipeline in pipelines.items())
    assert changed.run(1).tokens == [7]
    assert len(cache) == 4


def test_ring_overflow_is_an_error_not_corruption():
    eng = Engine(_program(n_stop=1000, ring_cap=16))           # 16 slots, 25 steps per buffer: the GPU outruns the host
    rep = eng.run(1000, steps_per_cb=25, in_flight=3)
    st = eng.state()
    assert st["error"] == 1 and st["done"] == 1 and rep.tokens == [7 * s for s in range(1, len(rep.tokens) + 1)]


def test_profile_gives_per_op_gpu_times():
    """Per-dispatch timestamps (one encoder per op, counter samples at stage boundaries): every op has a positive
    duration, ops are ordered, and the trace file is valid Chrome-trace JSON."""
    import json

    from monolith.trace import format_budget, op_timings, write_chrome_trace

    eng = Engine(_program(n_stop=50))
    eng.run(5, steps_per_cb=5, in_flight=1)
    runs = eng.profile(3)
    assert len(runs) == 3 and all(len(r) == 2 for r in runs)
    for run in runs:
        (s0, e0), (s1, e1) = run
        assert 0 <= s0 <= e0 <= s1 <= e1 and e1 - s0 < 5.0 and e0 > s0
    t = op_timings(eng.program, runs)
    assert [x.name for x in t] == ["map", "advance"] and all(x.ms_min > 0 for x in t)
    assert "total" in format_budget(t, 300.0)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        write_chrome_trace(f.name, eng.program, runs)
        ev = json.load(open(f.name))["traceEvents"]
        assert len(ev) == 6 and ev[0]["ph"] == "X" and ev[0]["name"] == "map"


def test_runner_reports_host_phases_without_counting_wait_as_cpu():
    eng = Engine(_program(n_stop=20))
    report = eng.run(20, steps_per_cb=5, in_flight=2)
    assert report.tokens == [7 * s for s in range(1, 21)]
    assert report.encode_ms >= 0 and report.commit_ms >= 0 and report.wait_ms >= 0
    assert report.encode_ms + report.commit_ms + report.wait_ms <= report.wall_ms + 0.1


def test_read_only_file_weights_survive_engine_recreation(tmp_path):
    import os
    path = tmp_path / 'weights.pack'
    page = os.sysconf('SC_PAGESIZE')
    values = np.arange(page // 4, dtype=np.uint32)
    path.write_bytes(bytes(page) + values.tobytes())
    source = """#include <metal_stdlib>
using namespace metal;
kernel void copy_weights(device const uint* w [[buffer(0)]], device uint* out [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) { out[i] = w[i] + 7u; }
"""
    program = _program(1)
    program.buffers['weights'] = BufferSpec(page, role='weights', file=str(path), file_offset=page)
    program.buffers['out'] = BufferSpec(page, role='arena')
    program.kernels['copy'] = KernelSpec(source, 'copy_weights')
    program.ops = [OpSpec('copy', [(0, 'weights', 0), (1, 'out', 0)],
                         (page // 4 // 128, 1, 1), (128, 1, 1), True)]
    for _ in range(2):
        engine = Engine(program)
        engine.run(1, steps_per_cb=1, in_flight=1)
        np.testing.assert_array_equal(np.frombuffer(engine.read('out'), dtype=np.uint32), values + 7)
        del engine
    assert path.read_bytes() == bytes(page) + values.tobytes()
