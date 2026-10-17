"""Host steps: a step pauses for CPU work between two of its GPU passes (HostPoint). The GPU's writes before a point's
publish boundary reach the task, the task's writes reach the ops from the point's wait boundary on, a split step is
bitwise the unsplit step, failures and timeouts end the run with every command buffer finished, and a run submits whole
steps. Every test here runs clean under MTL_SHADER_VALIDATION=1."""
import struct
import time

import numpy as np
import pytest

from monolith.core.step_state import StepStateLayout
from monolith.runtime import BufferSpec, Engine, HostPoint, KernelSpec, OpSpec, Program
from monolith.runtime import _native as nt

LAYOUT = StepStateLayout(t_max=8, gamma_max=7)
M = 0xFFFFFFFF
W = 256                                   # words a step hands across a host point

SRC = """
#include <metal_stdlib>
using namespace metal;
""" + LAYOUT.to_msl() + """
struct Params { uint n_stop; uint ring_cap; uint width; uint pad; };

static uint mix(uint x, uint s) { return (x ^ 0x5bd1e995u) * 0x27d4eb2du + s; }
static bool emit(device StepState* st, constant Params& p, device ulong* ring, uint token) {
  uint head = st->ring_head;
  if (head - st->ring_tail >= p.ring_cap) { st->error = 1u; st->done = 1u; return false; }
  ring[head % p.ring_cap] = (ulong(head + 1u) << 32) | ulong(token & 0x7fffffffu);
  st->ring_head = head + 1u;
  return true;
}
static void advance(device StepState* st, constant Params& p) {
  uint s = st->step + 1u;
  st->step = s;
  st->position += 1u;
  if (s >= p.n_stop || (st->stop_at != 0u && st->ring_head >= st->stop_at)) st->done = 1u;
}

// the exchange program: produce -> (host or transform) -> consume
kernel void produce(device uint* x [[buffer(0)]], device const StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  if (i < p.width) x[i] = (st->step + 1u) * 2654435761u + i * 40503u;
}
kernel void transform(device const uint* x [[buffer(0)]], device uint* staging [[buffer(1)]], device const StepState* st [[buffer(2)]],
                      constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  if (i < p.width) staging[i] = mix(x[i], st->step);
}
kernel void consume(device const uint* staging [[buffer(0)]], device uint* acc [[buffer(1)]], device StepState* st [[buffer(2)]],
                    constant Params& p [[buffer(3)]], device ulong* ring [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  uint a = acc[0];
  for (uint k = 0; k < p.width; k++) a = a * 31u + staging[k];
  acc[0] = a;
  if (emit(st, p, ring, a)) advance(st, p);
}

// a float chain whose middle op is independent of the first (barrier_before = false)
kernel void chain_a(device float* y [[buffer(0)]], device const StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  if (i < p.width) y[i] = y[i] * 0.9993f + float(st->step + i) * 1.0e-3f;
}
kernel void chain_b(device float* z [[buffer(0)]], device const StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  if (i < p.width) z[i] = z[i] * 1.0007f - float(i) * 1.0e-4f + 0.25f;
}
kernel void chain_c(device const float* y [[buffer(0)]], device const float* z [[buffer(1)]], device StepState* st [[buffer(2)]],
                    constant Params& p [[buffer(3)]], device ulong* ring [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  float t = 0.0f;
  for (uint k = 0; k < p.width; k++) t = t + y[k] * z[k];
  if (emit(st, p, ring, as_type<uint>(t))) advance(st, p);
}

// speculative-style window: picks for T rows, then a commit of the kept rows the host chose
static uint next_token(uint v) { return (v * 1103515245u + 12345u) % 1000u; }
kernel void verify(device uint* picks [[buffer(0)]], device const StepState* st [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  uint t_n = min(st->t_this_step, 4u);
  for (uint t = 0; t < t_n; t++) picks[t] = next_token(t == 0 ? uint(st->anchor) : uint(st->pending_tokens[t]));
}
kernel void commit(device const uint* picks [[buffer(0)]], device const uint* keep [[buffer(1)]], device StepState* st [[buffer(2)]],
                   constant Params& p [[buffer(3)]], device ulong* ring [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  uint n_keep = clamp(keep[0], 1u, min(st->t_this_step, 4u));
  for (uint t = 0; t < n_keep; t++) if (!emit(st, p, ring, picks[t])) return;
  st->anchor = int(picks[n_keep - 1]);
  st->accepted = n_keep - 1;
  uint s = st->step + 1u;
  st->step = s;
  if (st->ring_head >= p.n_stop) st->done = 1u;
}

// several points per step
kernel void s_a(device uint* a [[buffer(0)]], device const uint* h3 [[buffer(1)]], device const StepState* st [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
  if (i == 0 && !st->done) a[0] = h3[0] + (st->step + 1u) * 7u;
}
kernel void s_b(device uint* b [[buffer(0)]], device const StepState* st [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i == 0 && !st->done) b[0] = (st->step + 1u) * 13u;
}
kernel void s_c(device uint* c [[buffer(0)]], device const uint* h1 [[buffer(1)]], device const uint* h2 [[buffer(2)]],
                device const StepState* st [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i == 0 && !st->done) c[0] = h1[0] * 3u + h2[0];
}
kernel void s_adv(device const uint* c [[buffer(0)]], device StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                  device ulong* ring [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  if (emit(st, p, ring, c[0])) advance(st, p);
}

// a point that publishes at the step's start and is needed only by the second op (the n-gram rows' shape)
kernel void ng_first(device uint* l0 [[buffer(0)]], device const StepState* st [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i == 0 && !st->done) l0[0] = (st->step + 1u) * 11u;
}
kernel void ng_second(device const uint* l0 [[buffer(0)]], device const uint* rows [[buffer(1)]], device uint* prev [[buffer(2)]],
                      device StepState* st [[buffer(3)]], constant Params& p [[buffer(4)]], device ulong* ring [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  uint out = l0[0] + rows[0];
  prev[0] = out;
  if (emit(st, p, ring, out)) advance(st, p);
}

// a bounded loop of a few hundred milliseconds before a publish boundary
kernel void slow(device float* sink [[buffer(0)]], device const StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  float v = sink[0];
  for (uint k = 0; k < p.width; k++) v = fma(v, 1.0000001f, 1.0e-7f);
  sink[0] = v;
}
kernel void tick(device StepState* st [[buffer(0)]], constant Params& p [[buffer(1)]], device ulong* ring [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  if (emit(st, p, ring, st->step)) advance(st, p);
}

// a kernel that reaches two unbound buffers through GPU addresses the host wrote into a table
struct Ptr { device const uint* p; };
kernel void gather(device const Ptr* table [[buffer(0)]], device uint* out [[buffer(1)]], device const StepState* st [[buffer(2)]],
                   constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  if (i < p.width) out[i] = table[i & 1u].p[i >> 1] + st->step;
}
"""

KERNELS = ("produce", "transform", "consume", "chain_a", "chain_b", "chain_c", "verify", "commit", "s_a", "s_b", "s_c",
           "s_adv", "ng_first", "ng_second", "slow", "tick", "gather")


def _params(n_stop, ring_cap, width):
    return BufferSpec(16, struct.pack("<IIII", n_stop, ring_cap, width, 0), "params")


def _base_buffers(n_stop, ring_cap=1024, width=W, step_state=None):
    return {"step_state": BufferSpec(LAYOUT.size, LAYOUT.pack(step_state or {}), "step_state"),
            "params": _params(n_stop, ring_cap, width), "ring": BufferSpec(ring_cap * 8, None, "ring")}


def _op(kernel, bindings, writes, threads=32, barrier=True):
    """One threadgroup; ``writes``: the binding slots the op writes (meta["writes"])."""
    return OpSpec(kernel, bindings, (1, 1, 1), (threads, 1, 1), barrier, name=kernel, meta={"writes": list(writes)})


def _tick():
    """Emits StepState.step and advances it."""
    return _op("tick", [(0, "step_state", 0), (1, "params", 0), (2, "ring", 0)], [0, 2])


def _program(buffers, ops, points=(), ring_cap=1024):
    return Program(kernels={k: KernelSpec(SRC, k) for k in KERNELS}, buffers=buffers, ops=list(ops),
                   ring_capacity=ring_cap, layout=LAYOUT, host_points=list(points))


def _mix(x, s):
    return (((x.astype(np.uint64) ^ 0x5BD1E995) * 0x27D4EB2D + s) & M).astype(np.uint32)


def _x(step):
    return (((step + 1) * 2654435761 + np.arange(W, dtype=np.uint64) * 40503) & M).astype(np.uint32)


def _exchange_model(steps):
    a, tokens = 0, []
    for s in range(steps):
        for v in _mix(_x(s), s).tolist():
            a = (a * 31 + v) & M
        tokens.append(a & 0x7FFFFFFF)
    return tokens, a


def _exchange_program(n_stop, split=True):
    buffers = {**_base_buffers(n_stop), "x": BufferSpec(W * 4), "acc": BufferSpec(16),
               "staging": BufferSpec(W * 4, role="host" if split else "arena")}
    produce = _op("produce", [(0, "x", 0), (1, "step_state", 0), (2, "params", 0)], [0], threads=W)
    transform = _op("transform", [(0, "x", 0), (1, "staging", 0), (2, "step_state", 0), (3, "params", 0)], [1], threads=W)
    consume = _op("consume", [(0, "staging", 0), (1, "acc", 0), (2, "step_state", 0), (3, "params", 0), (4, "ring", 0)], [1, 2, 4])
    if not split:
        return _program(buffers, [produce, transform, consume])
    return _program(buffers, [produce, consume], [HostPoint("exchange", 1, 1, writes=["staging"], reads=["x", "step_state"])])


class _Exchange:
    """The exchange program's host task: checks the GPU's produce output, writes what transform would."""

    def __init__(self, fail_at=None):
        self.dev = nt.Device()
        self.staging = nt.Buffer(self.dev, W * 4)
        self.calls, self.fail_at, self.eng = [], fail_at, None

    def __call__(self, step, point):
        st = self.eng.state()
        x = np.frombuffer(self.eng.read("x"), dtype=np.uint32)[:W]
        self.calls.append((step, point, st["step"], bool(np.array_equal(x, _x(st["step"])))))
        if step == self.fail_at:
            raise ValueError(f"task failed at step {step}")
        self.staging.write(_mix(x, st["step"]).tobytes(), 0)

    def engine(self, n_stop, **kw):
        self.eng = Engine(_exchange_program(n_stop), self.dev, buffers={"staging": self.staging},
                          host_tasks={"exchange": self}, **kw)
        return self.eng


@pytest.mark.parametrize("reencode", [False, True])
def test_gpu_writes_reach_the_task_and_task_writes_reach_the_next_pass(reencode):
    ex = _Exchange()
    rep = ex.engine(300).run(300, reencode=reencode)
    tokens, acc = _exchange_model(300)
    assert rep.steps == 300 and rep.done and rep.tokens == tokens
    assert [c[0] for c in ex.calls] == list(range(300)) and all(c[1] == 0 for c in ex.calls)
    assert [c[2] for c in ex.calls] == list(range(300))     # StepState.step the previous step wrote was visible
    assert all(c[3] for c in ex.calls)                      # every word produce wrote was visible to the task
    assert np.frombuffer(ex.eng.read("acc"), dtype=np.uint32)[0] == acc


@pytest.mark.parametrize("reencode", [False, True])
def test_split_step_is_bitwise_the_unsplit_step(reencode):
    unsplit = Engine(_exchange_program(200, split=False))
    ru = unsplit.run(200, steps_per_cb=7, in_flight=3, reencode=reencode)
    ex = _Exchange()
    rs = ex.engine(200).run(200, reencode=reencode)
    assert rs.tokens == ru.tokens == _exchange_model(200)[0]
    for name in ("x", "acc", "staging", "ring"):
        assert ex.eng.read(name) == unsplit.read(name)
    assert ex.eng.state() == unsplit.state()


def _chain_program(with_points):
    buffers = {**_base_buffers(150), "y": BufferSpec(W * 4, np.ones(W, np.float32).tobytes()),
               "z": BufferSpec(W * 4, np.full(W, 0.5, np.float32).tobytes())}
    ops = [_op("chain_a", [(0, "y", 0), (1, "step_state", 0), (2, "params", 0)], [0], threads=W),
           _op("chain_b", [(0, "z", 0), (1, "step_state", 0), (2, "params", 0)], [0], threads=W, barrier=False),
           _op("chain_c", [(0, "y", 0), (1, "z", 0), (2, "step_state", 0), (3, "params", 0), (4, "ring", 0)], [2, 4])]
    points = [HostPoint("noop", 1, 1), HostPoint("noop", 2, 3, reads=["y"])] if with_points else []
    return _program(buffers, ops, points)


def test_cuts_through_an_unbarriered_pair_keep_every_bit():
    calls = []
    plain = Engine(_chain_program(False))
    rp = plain.run(150, steps_per_cb=10, in_flight=2)
    cut = Engine(_chain_program(True), host_tasks={"noop": lambda step, point: calls.append((step, point))})
    rc = cut.run(150)
    assert rc.tokens == rp.tokens and len(rc.tokens) == 150
    for name in ("y", "z", "ring"):
        assert cut.read(name) == plain.read(name)
    assert cut.state() == plain.state()
    assert calls == [(s, k) for s in range(150) for k in (0, 1)]


def _window_program(n_stop, start):
    buffers = {**_base_buffers(n_stop, step_state={"anchor": start, "t_this_step": 1}),
               "picks": BufferSpec(16), "keep": BufferSpec(16, role="host")}
    ops = [_op("verify", [(0, "picks", 0), (1, "step_state", 0)], [0]),
           _op("commit", [(0, "picks", 0), (1, "keep", 0), (2, "step_state", 0), (3, "params", 0), (4, "ring", 0)], [2, 4])]
    points = [HostPoint("window", 0, 0, writes=["step_state"], reads=["step_state"]),
              HostPoint("accept", 1, 1, writes=["keep"], reads=["picks", "step_state"])]
    return _program(buffers, ops, points)


def test_host_chooses_each_window_and_accepts_before_the_commit():
    """The host picks the window (T and drafts, written as data at a sync point at the step's start) and accepts after
    the verify op (a sync point before the commit). Drafts are right or wrong by a fixed pattern; the tokens must be
    the plain greedy chain whatever the windows were."""
    def nxt(v):
        return ((v * 1103515245 + 12345) & M) % 1000             # the kernel's 32-bit arithmetic

    start, n_stop = 7, 120
    dev = nt.Device()
    keep = nt.Buffer(dev, 16)
    windows, eng = [], None

    def window(step, point):
        st = eng.state()
        T = 1 + step % 4
        drafts, v = [st["anchor"]], st["anchor"]
        for t in range(1, T):
            v = nxt(v)
            drafts.append(v if (step + t) % 3 else (v + 1) % 1000)   # every third draft is wrong
            v = drafts[-1]
        windows.append(drafts)
        sbuf = eng.buffers["step_state"]
        sbuf.write(struct.pack(f"<{LAYOUT.t_max}i", *(drafts + [0] * (LAYOUT.t_max - T))), LAYOUT.offset("pending_tokens"))
        sbuf.write(struct.pack("<I", T), LAYOUT.offset("t_this_step"))

    def accept(step, point):
        drafts = windows[step]
        picks = list(struct.unpack("<4I", eng.read("picks", 16)))
        n_keep = 1
        while n_keep < len(drafts) and drafts[n_keep] == picks[n_keep - 1]:
            n_keep += 1
        keep.write(struct.pack("<I", n_keep), 0)

    eng = Engine(_window_program(n_stop, start), dev, buffers={"keep": keep}, host_tasks={"window": window, "accept": accept})
    rep = eng.run(1000)
    chain, v = [], start
    for _ in range(len(rep.tokens)):
        v = nxt(v)
        chain.append(v)
    assert rep.done and len(rep.tokens) >= n_stop and rep.tokens == chain
    assert len(windows) == eng.state()["step"] < len(rep.tokens)   # one window per committed step; several kept more than one row
    assert {len(w) for w in windows} == {1, 2, 3, 4}


def _multi_program(n_stop):
    host = {name: BufferSpec(16, role="host") for name in ("h1", "h2", "h3")}
    buffers = {**_base_buffers(n_stop), **host, "a": BufferSpec(16), "b": BufferSpec(16), "c": BufferSpec(16)}
    ops = [_op("s_a", [(0, "a", 0), (1, "h3", 0), (2, "step_state", 0)], [0]),
           _op("s_b", [(0, "b", 0), (1, "step_state", 0)], [0]),
           _op("s_c", [(0, "c", 0), (1, "h1", 0), (2, "h2", 0), (3, "step_state", 0)], [0]),
           _op("s_adv", [(0, "c", 0), (1, "step_state", 0), (2, "params", 0), (3, "ring", 0)], [1, 3])]
    points = [HostPoint("p1", 1, 2, writes=["h1"], reads=["a"]),        # s_b runs beside the task
              HostPoint("p2", 2, 2, writes=["h2"], reads=["b"]),        # a sync point
              HostPoint("p3", 3, 4, writes=["h3"], reads=["c"])]        # s_adv runs beside it; the next step waits
    return _program(buffers, ops, points)


def test_several_points_per_step_see_their_boundaries():
    dev = nt.Device()
    h = {name: nt.Buffer(dev, 16) for name in ("h1", "h2", "h3")}
    seen = {"p1": [], "p2": [], "p3": []}
    eng = None

    def word(name):
        return struct.unpack("<I", eng.read(name, 4))[0]

    def put(name, v):
        h[name].write(struct.pack("<I", v & M), 0)

    def p1(step, point):
        seen["p1"].append((step, point, word("a")))
        put("h1", word("a") * 5 + 1)

    def p2(step, point):
        seen["p2"].append((step, point, word("b")))
        put("h2", word("b") + 100)

    def p3(step, point):
        seen["p3"].append((step, point, word("c")))
        put("h3", word("c") ^ 0x1234)

    n = 120
    eng = Engine(_multi_program(n), dev, buffers=h, host_tasks={"p1": p1, "p2": p2, "p3": p3})
    rep = eng.run(n)
    h3, want = 0, {"p1": [], "p2": [], "p3": []}
    tokens = []
    for s in range(n):
        a = (h3 + (s + 1) * 7) & M
        h1 = (a * 5 + 1) & M
        b = ((s + 1) * 13) & M
        c = (h1 * 3 + b + 100) & M
        want["p1"].append((s, 0, a))
        want["p2"].append((s, 1, b))
        want["p3"].append((s, 2, c))
        tokens.append(c & 0x7FFFFFFF)
        h3 = c ^ 0x1234
    assert seen == want and rep.tokens == tokens and rep.steps == n


def _start_point_engine():
    """publish 0, wait 1: the task reads what the previous step left and writes rows only the step's second op reads,
    while the first op may already run. Returns the engine and the task's (step, StepState.step, prev) calls."""
    dev = nt.Device()
    rows = nt.Buffer(dev, 16)
    buffers = {**_base_buffers(1000), "l0": BufferSpec(16), "prev": BufferSpec(16), "rows": BufferSpec(16, role="host")}
    ops = [_op("ng_first", [(0, "l0", 0), (1, "step_state", 0)], [0]),
           _op("ng_second", [(0, "l0", 0), (1, "rows", 0), (2, "prev", 0), (3, "step_state", 0), (4, "params", 0), (5, "ring", 0)],
               [2, 3, 5])]
    calls, eng = [], None

    def task(step, point):
        prev = struct.unpack("<I", eng.read("prev", 4))[0]
        calls.append((step, eng.state()["step"], prev))
        rows.write(struct.pack("<I", (prev * 7 + step) & M), 0)

    eng = Engine(_program(buffers, ops, [HostPoint("rows", 0, 1, writes=["rows"], reads=["prev", "step_state"])]), dev,
                 buffers={"rows": rows}, host_tasks={"rows": task})
    return eng, calls


def _start_point_model(steps):
    """The task's calls and the tokens of ``steps`` steps of the start-point program."""
    prev, want_calls, want = 0, [], []
    for s in range(steps):
        want_calls.append((s, s, prev))
        out = ((s + 1) * 11 + ((prev * 7 + s) & M)) & M
        want.append(out & 0x7FFFFFFF)
        prev = out
    return want_calls, want


@pytest.mark.parametrize("reencode", [False, True])
def test_a_point_at_the_step_start_feeds_a_later_op_across_runs(reencode):
    """The first step of each run is served without a signal."""
    eng, calls = _start_point_engine()
    tokens = eng.run(40, reencode=reencode).tokens + eng.run(25, reencode=reencode).tokens
    assert (calls, tokens) == _start_point_model(65)


@pytest.mark.parametrize("reencode", [False, True])
def test_max_tokens_with_a_point_at_the_step_start_across_runs(reencode):
    """A max_tokens stop comes after the signal of the next step's point was committed: that step does not begin in
    the stopped run, and the next run serves it without a signal, with its own StepState.step."""
    eng, calls = _start_point_engine()
    r1 = eng.run(100, max_tokens=10, reencode=reencode)
    r2 = eng.run(7, reencode=reencode)
    assert 10 <= r1.steps <= 12 and not r1.done and r2.steps == 7
    assert (calls, r1.tokens + r2.tokens) == _start_point_model(r1.steps + 7)


def test_a_failing_task_ends_the_run_with_every_command_buffer_finished():
    ex = _Exchange(fail_at=7)
    eng = ex.engine(100)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="task failed at step 7"):
        eng.run(100)
    assert time.monotonic() - t0 < 5.0
    st = eng.state()
    assert st["done"] == 1 and st["error"] == 4 and st["step"] == 7       # step 7's consume never ran
    assert [c[0] for c in ex.calls] == list(range(8))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="host step failed"):
        eng.run(10)                                                       # a finished program's steps do no host work
    assert time.monotonic() - t0 < 5.0 and len(ex.calls) == 8


def _slow_tick(n_stop, width):
    """A bounded slow loop (``width`` iterations) into ``sink``, then the step's advance."""
    buffers = {**_base_buffers(n_stop, width=width), "sink": BufferSpec(16)}
    return buffers, [_op("slow", [(0, "sink", 0), (1, "step_state", 0), (2, "params", 0)], [0]), _tick()]


def _sink(eng):
    return struct.unpack("<f", eng.read("sink", 4))[0]


def test_a_failure_returns_after_the_gpu_work_beside_the_task():
    def fail(step, point):
        raise ValueError("stop here")

    buffers, ops = _slow_tick(10, 60_000_000)
    eng = Engine(_program(buffers, ops, [HostPoint("fail", 0, 1)]), host_tasks={"fail": fail})
    with pytest.raises(RuntimeError, match="stop here"):
        eng.run(5)
    assert _sink(eng) > 0                                                 # slow ran beside the task and finished first
    st = eng.state()
    assert st["done"] == 1 and st["error"] == 4 and st["step"] == 0


@pytest.mark.parametrize("objc", [False, True], ids=["cpp", "objc"])
def test_a_task_throwing_a_non_standard_exception_fails_the_run_cleanly(objc):
    """A C++ task may throw what is not a std::exception (an int, an NSException): the run still waits for the GPU,
    marks the program failed and counts the step, so the next run numbers its steps on."""
    steps = []
    buffers, ops = _slow_tick(100, 20_000_000)
    eng = Engine(_program(buffers, ops, [HostPoint("rec", 0, 1), HostPoint("throw", 0, 1)]),
                 host_tasks={"rec": lambda s, k: steps.append(s), "throw": nt._ThrowingTask(3, objc)})
    with pytest.raises(RuntimeError, match="host step 3, point 1: the task threw"):
        eng.run(10)
    sink = _sink(eng)
    time.sleep(0.5)
    st = eng.state()
    assert _sink(eng) == sink                                             # step 3's slow op finished before run() returned
    assert st["done"] == 1 and st["error"] == 4 and st["step"] == 3 and steps == [0, 1, 2, 3]
    for field in ("done", "error"):
        eng.buffers["step_state"].write(struct.pack("<I", 0), LAYOUT.offset(field))
    assert eng.run(2).steps == 2 and steps == [0, 1, 2, 3, 4, 5]


def test_a_host_timeout_waits_for_the_gpu_and_fails():
    buffers = {**_base_buffers(10, width=60_000_000), "sink": BufferSpec(16)}
    ops = [_op("slow", [(0, "sink", 0), (1, "step_state", 0), (2, "params", 0)], [0]),
           _op("tick", [(0, "step_state", 0), (1, "params", 0), (2, "ring", 0)], [0, 2])]
    calls = []
    eng = Engine(_program(buffers, ops, [HostPoint("late", 1, 1)]), host_tasks={"late": lambda s, k: calls.append(s)},
                 host_timeout_ms=20)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="did not reach the point within 20 ms"):
        eng.run(10)
    elapsed = time.monotonic() - t0
    st = eng.state()
    assert elapsed < 5.0 and calls == [] and st["done"] == 1 and st["error"] == 4 and st["step"] == 0
    assert _sink(eng) > 0                                                 # the slow op before the point completed first


def test_a_command_buffer_that_finishes_without_its_signal_fails_at_once(monkeypatch):
    """LITHOS_HOST_STEPS_DROP_SIGNAL leaves out one point's signal; the exchange program commits nothing later before
    that point is served, so no higher signal masks it."""
    monkeypatch.setenv("LITHOS_HOST_STEPS_DROP_SIGNAL", "3:0")
    ex = _Exchange()
    eng = ex.engine(100, host_timeout_ms=10_000)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"host step 3, point 0: its command buffer finished without the signal"):
        eng.run(10)
    assert time.monotonic() - t0 < 1.0 and [c[0] for c in ex.calls] == [0, 1, 2]
    st = eng.state()
    assert st["done"] == 1 and st["error"] == 4


@pytest.mark.parametrize("value", ["-1:0", "+3:0", " 3:0", "3:", ":0", "3:0x", "3:9", "123456789012345678901:0"])
def test_a_malformed_drop_signal_is_refused(monkeypatch, value):
    """The hook takes two decimal numbers and a point the program has; anything else fails instead of dropping nothing."""
    monkeypatch.setenv("LITHOS_HOST_STEPS_DROP_SIGNAL", value)
    with pytest.raises(RuntimeError, match="LITHOS_HOST_STEPS_DROP_SIGNAL"):
        _Exchange().engine(10)


def test_an_empty_drop_signal_is_unset(monkeypatch):
    monkeypatch.setenv("LITHOS_HOST_STEPS_DROP_SIGNAL", "")
    assert _Exchange().engine(10).run(3).steps == 3


def test_a_failed_runs_tokens_leave_with_its_error():
    """The failed run's report rides on its error, so none of its tokens reach a later run (a session resets StepState
    and the ring, then runs the next request on the same Engine)."""
    calls = []

    def watch(step, point):
        calls.append(step)
        if step == 3:
            raise ValueError("stop at step 3")

    eng = Engine(_program(_base_buffers(100), [_tick()], [HostPoint("watch", 1, 1)]), host_tasks={"watch": watch})
    with pytest.raises(RuntimeError, match="stop at step 3") as failed:
        eng.run(10)
    assert failed.value.report.steps == 4 and failed.value.report.tokens == [0, 1, 2, 3]
    for name, spec in eng.program.buffers.items():                       # the session's reset
        if spec.role in ("state", "step_state", "ring"):
            eng.buffers[name].fill(0)
    rep = eng.run(5, max_tokens=5)
    assert rep.steps == 5 and rep.tokens == [0, 1, 2, 3, 4] and calls == list(range(9))


@pytest.mark.parametrize("host_point", [True, False], ids=["host-steps", "plain"])
def test_a_run_drops_tokens_an_earlier_run_left_undrained(host_point):
    """A run interrupted before Engine.run drained it (a KeyboardInterrupt right after the pump returns), or a direct
    Runner.run, leaves its tokens in the runner; the next run's report holds only its own."""
    points = [HostPoint("t", 1, 1)] if host_point else []
    eng = Engine(_program(_base_buffers(100), [_tick()], points), host_tasks={"t": lambda s, k: None})
    st = eng.runner.run(3, 1, 1, False, 0)
    assert st.steps_submitted == 3 and not st.error
    assert eng.run(2).tokens == [3, 4]


def test_a_task_exception_no_run_took_stays_out_of_later_errors():
    """A task raised in a run nothing took the exception from (a direct Runner.run here, or an Engine.run interrupted
    before it did); a later run that fails for another reason carries no cause."""
    def watch(step, point):
        if step == 2:
            raise ValueError("stale, from the direct run")

    eng = Engine(_program(_base_buffers(100), [_tick()], [HostPoint("watch", 1, 1)]), host_tasks={"watch": watch})
    assert "stale, from the direct run" in eng.runner.run(5, 1, 1, False, 0).error
    for field in ("done", "error"):
        eng.buffers["step_state"].write(struct.pack("<I", 0), LAYOUT.offset(field))
    assert eng.run(3).steps == 3
    eng.buffers["step_state"].write(struct.pack("<I", 1), LAYOUT.offset("done"))
    eng.buffers["step_state"].write(struct.pack("<I", 3), LAYOUT.offset("error"))
    with pytest.raises(RuntimeError, match="bounded worker barrier timed out") as failed:
        eng.run(1)
    assert failed.value.__cause__ is None


@pytest.mark.parametrize("raised", [KeyboardInterrupt, SystemExit, LookupError])
def test_a_tasks_exception_reaches_the_caller(raised):
    """KeyboardInterrupt and SystemExit leave run() as the task raised them; any other exception is the cause of the
    run's RuntimeError. The step stopped at the point either way, so the program stays failed until a reset."""
    def watch(step, point):
        if step == 3:
            raise raised("from the task")

    eng = Engine(_program(_base_buffers(100), [_tick()], [HostPoint("watch", 1, 1)]), host_tasks={"watch": watch})
    if issubclass(raised, Exception):
        with pytest.raises(RuntimeError, match="from the task") as failed:
            eng.run(10)
        assert isinstance(failed.value.__cause__, raised)
    else:
        with pytest.raises(raised, match="from the task"):
            eng.run(10)
    st = eng.state()
    assert st["done"] == 1 and st["error"] == 4 and st["step"] == 4


def test_max_tokens_and_stop_at_end_runs_on_whole_steps():
    ex = _Exchange()
    eng = ex.engine(1000)
    tokens = _exchange_model(40)[0]
    rep = eng.run(100, max_tokens=10)
    assert 10 <= rep.steps <= 12 and rep.tokens == tokens[:rep.steps] and not rep.done
    assert [c[0] for c in ex.calls] == list(range(rep.steps))             # every begun step's task ran once
    first = rep.steps
    rep2 = eng.run(5)
    assert rep2.tokens == tokens[first:first + 5] and [c[0] for c in ex.calls] == list(range(first + 5))
    eng.buffers["step_state"].write(struct.pack("<I", first + 5 + 15), LAYOUT.offset("stop_at"))
    rep3 = eng.run(100)
    assert rep3.done and rep3.tokens == tokens[first + 5:first + 20]
    steps_with_work = first + 20
    assert [c[0] for c in ex.calls] == list(range(steps_with_work))       # the steps after done did no host work
    assert eng.state()["step"] == steps_with_work


@pytest.mark.parametrize("stop_point", [0, 1])
def test_a_task_that_sets_done_ends_the_run_after_its_step(stop_point):
    """A task may stop the program by writing StepState.done (host acceptance at EOS): the step whose task wrote it runs
    to its end, and no later step begins or runs a task."""
    calls, eng = [], None

    def watch(step, point):
        calls.append((step, point))
        if step == 5 and point == stop_point:
            eng.buffers["step_state"].write(struct.pack("<I", 1), LAYOUT.offset("done"))

    points = [HostPoint("watch", 1, 1, writes=["step_state"]), HostPoint("watch", 1, 1, writes=["step_state"])]
    eng = Engine(_program(_base_buffers(1000), [_tick()], points), host_tasks={"watch": watch})
    rep = eng.run(100)
    assert rep.done and rep.steps == 6 and rep.tokens == list(range(6))
    assert calls == [(s, k) for s in range(6) for k in (0, 1)]


def test_done_set_by_an_op_beside_the_tasks_keeps_the_steps_tasks():
    """tick runs beside both tasks and sets done in the last step; that step began live, so its second task still runs.
    q waits (bounded) until tick has advanced the step; it reads StepState undeclared on purpose."""
    calls, eng = {"q": [], "p": []}, None

    def q(step, point):
        calls["q"].append(step)
        for _ in range(1000):
            if eng.state()["step"] >= step + 1:
                break
            time.sleep(0.001)
        assert eng.state()["step"] >= step + 1

    def p(step, point):
        calls["p"].append(step)

    ops = [_op("s_b", [(0, "b", 0), (1, "step_state", 0)], [0]), _tick()]
    eng = Engine(_program({**_base_buffers(5), "b": BufferSpec(16)}, ops, [HostPoint("q", 1, 2), HostPoint("p", 1, 2)]),
                 host_tasks={"q": q, "p": p})
    rep = eng.run(100)
    assert rep.done and rep.steps == 5 and rep.tokens == list(range(5))
    assert calls == {"q": list(range(5)), "p": list(range(5))}


def test_a_second_concurrent_run_fails_at_once():
    import threading
    entered, release = threading.Event(), threading.Event()

    def hold(step, point):
        if step == 0:
            entered.set()
            assert release.wait(5.0)

    eng = Engine(_program(_base_buffers(10), [_tick()], [HostPoint("hold", 1, 1)]), host_tasks={"hold": hold})
    report = []
    first = threading.Thread(target=lambda: report.append(eng.run(3)))
    first.start()
    try:
        assert entered.wait(5.0)                                          # the first run holds the runner
        for call in (lambda: eng.run(1), lambda: eng.runner.run(1, 1, 1, False, 0)):
            with pytest.raises(RuntimeError, match="Runner.run: already running on another thread"):
                call()
        with pytest.raises(RuntimeError, match="set_host_points: already running on another thread"):
            eng.runner.set_host_points([], LAYOUT.offset("error"), 1000)
    finally:
        release.set()
        first.join(10.0)
    assert not first.is_alive() and eng.state()["step"] == 3
    assert report[0].tokens == list(range(3))                             # the refused run took none of them


def test_a_task_that_calls_its_own_runner_is_refused():
    errors, eng = [], None

    def reenter(step, point):
        if step == 1:
            for call in (lambda: eng.run(1), lambda: eng.runner.run(1, 1, 1, False, 0),
                         lambda: eng.runner.set_host_points([], LAYOUT.offset("error"), 1000)):
                with pytest.raises(RuntimeError) as refused:
                    call()
                errors.append(str(refused.value))


    eng = Engine(_program(_base_buffers(5), [_tick()], [HostPoint("reenter", 1, 1)]), host_tasks={"reenter": reenter})
    rep = eng.run(5)
    assert rep.steps == 5 and rep.tokens == list(range(5))
    assert errors == ["Runner.run: called from one of the runner's own host tasks",
                      "Runner.run: called from one of the runner's own host tasks",
                      "set_host_points: called from one of the runner's own host tasks"]


def test_a_run_reports_the_tokens_it_took():
    """The run takes the ring's tokens while it still owns the runner, so a drain() afterwards finds none left and no
    other caller can take them first."""
    eng = Engine(_program(_base_buffers(1000), [_tick()]))
    st = eng.runner.run(4, 8, 3, False, 0)                                # the pump's own stats, not Engine's report
    assert list(st.tokens) == list(range(4))
    assert eng.runner.drain() == []                                       # the run left none behind to be taken later
    assert eng.run(2).tokens == [4, 5]


def test_plain_runs_between_host_step_runs_keep_the_step_count():
    """HostCall.step counts every step the runner began, on either pump."""
    steps = []
    eng = Engine(_program(_base_buffers(1000), [_tick()], [HostPoint("watch", 1, 1)]),
                 host_tasks={"watch": lambda s, k: steps.append(s)})
    error = LAYOUT.offset("error")
    eng.run(3)
    eng.runner.set_host_points([], error, 1000)                           # the plain pump
    eng.run(4, steps_per_cb=1, in_flight=1)
    eng.runner.set_host_points(eng.host_tasks, error, 1000)
    eng.run(2)
    assert steps == [0, 1, 2, 7, 8] and eng.state()["step"] == 9


def test_set_host_points_again_with_fewer_points_waits_for_the_gpu():
    """A new point list gets a new event: the old one already holds values the new numbering reaches early."""
    seen, eng = [], None

    def look(step, point):
        seen.append((step, point, eng.state()["step"], _sink(eng)))

    buffers, ops = _slow_tick(1000, 1_000_000)
    eng = Engine(_program(buffers, ops, [HostPoint("look", 1, 1, reads=["sink"]), HostPoint("look", 1, 1, reads=["sink"])]),
                 host_tasks={"look": look})
    assert eng.run(10).steps == 10
    eng.runner.set_host_points([(1, 1, nt.CallableTask(look))], LAYOUT.offset("error"), 20000)
    seen.clear()
    rep = eng.run(10)
    assert rep.steps == 10 and rep.tokens == list(range(10, 20))
    assert [(s, k, st) for s, k, st, _ in seen] == [(s, 0, s) for s in range(10, 20)]
    sinks = [v for *_, v in seen]
    assert all(b > a for a, b in zip(sinks, sinks[1:]))                  # each task ran after its step's slow op


def test_handoff_gaps_measure_each_handoff():
    """One gap per handoff: the GPU idle time from the command buffer that ends at the point to the one after the task."""
    buffers, ops = _slow_tick(100, 1000)
    eng = Engine(_program(buffers, ops, [HostPoint("t", 1, 1)]), host_tasks={"t": lambda s, k: None})
    gaps = eng.run(10).handoff_gaps_ms
    assert len(gaps) == 10 and min(gaps) > 0 and max(gaps) < 1000
    assert Engine(_program(*_slow_tick(100, 1000))).run(10).handoff_gaps_ms == []


def test_profile_refuses_a_program_with_host_points():
    with pytest.raises(RuntimeError, match="host points"):
        _Exchange().engine(10).profile(1)                                 # its ops would run without their tasks


def test_profile_refuses_tasks_installed_on_the_runner():
    eng = Engine(_program(_base_buffers(10), [_tick()]))
    eng.runner.set_host_points([(1, 1, nt.CallableTask(lambda s, k: None))], LAYOUT.offset("error"), 1000)
    with pytest.raises(RuntimeError, match="host points"):
        eng.profile(1)


def test_a_task_that_drains_does_not_defeat_max_tokens():
    """max_tokens counts the tokens taken from the ring during the run, also those a task drained itself."""
    drained, eng = [], None

    def take(step, point):
        drained.extend(eng.runner.drain())

    eng = Engine(_program(_base_buffers(100), [_tick()], [HostPoint("take", 1, 1)]), host_tasks={"take": take})
    rep = eng.run(50, max_tokens=2)
    assert rep.steps <= 3 and drained + rep.tokens == list(range(rep.steps))


def test_a_task_cannot_write_weights_through_an_alias():
    dev = nt.Device()
    shared = nt.Buffer(dev, 16384)
    program = _validation_program([HostPoint("t", 1, 1, writes=["staging"])])
    with pytest.raises(ValueError, match=r"writes 'staging', which reaches the read-only weights buffer 'w'"):
        Engine(program, dev, buffers={"staging": shared, "w": shared}, host_tasks={"t": lambda s, k: None})


def test_a_zero_step_run_reports_an_earlier_host_failure():
    ex = _Exchange(fail_at=2)
    eng = ex.engine(100)
    with pytest.raises(RuntimeError, match="task failed at step 2"):
        eng.run(10)
    eng.runner.set_host_points([], LAYOUT.offset("error"), 1000)          # the plain pump
    with pytest.raises(RuntimeError, match="a host step failed in an earlier run"):
        eng.run(0)


def test_host_steps_report_commit_time():
    rep = _Exchange().engine(100).run(10)
    assert rep.commit_ms > 0 and rep.encode_ms > 0


def _validation_program(points, **overrides):
    buffers = {**_base_buffers(10), "x": BufferSpec(W * 4), "acc": BufferSpec(16), "staging": BufferSpec(W * 4, role="host"),
               "w": BufferSpec(16384, role="weights")}
    buffers.update(overrides)
    ops = [_op("produce", [(0, "x", 0), (1, "step_state", 0), (2, "params", 0)], [0], threads=W),
           _op("consume", [(0, "staging", 0), (1, "acc", 0), (2, "step_state", 0), (3, "params", 0), (4, "ring", 0)], [1, 2, 4])]
    return _program(buffers, ops, points)


@pytest.mark.parametrize("point, message", [
    (HostPoint("t", 2, 1), r"need 0 <= publish \(2\) <= wait \(1\)"),
    (HostPoint("t", 0, 3), r"<= 2, the op count"),
    (HostPoint("t", 0, 1, reads=["nope"]), "unknown buffer 'nope'"),
    (HostPoint("t", 1, 1, writes=["w"]), "read-only weights buffer"),
    (HostPoint("t", 0, 1, reads=["x"]), r"op 0 \(produce\) writes 'x' while the task reads 'x'"),
    (HostPoint("t", 1, 2, writes=["staging"]), r"op 1 \(consume\) accesses 'staging' while the task writes 'staging'"),
    (HostPoint("t", 0, 1, writes=["step_state"]), r"op 0 \(produce\) accesses 'step_state'"),
])
def test_check_host_points_rejects(point, message):
    with pytest.raises(ValueError, match=message):
        _validation_program([point]).check_host_points()


@pytest.mark.parametrize("point, message", [
    (HostPoint("t", 0, 1, writes=["staging"]), r"op 0 \(produce\) accesses 'staging' while the task writes 'staging'"),
    (HostPoint("t", 0, 1, reads=["acc"]), r"op 0 \(produce\) writes 'acc' while the task reads 'acc'"),
])
def test_check_host_points_sees_what_ops_reach_through_gpu_addresses(point, message):
    program = _validation_program([point])
    program.ops[0].meta.update(indirect_reads=["staging"], indirect_writes=["acc"])
    with pytest.raises(ValueError, match=message):
        program.check_host_points()


def test_check_host_points_accepts_reads_of_what_the_window_only_reads_and_decreasing_publishes_fail():
    _validation_program([HostPoint("t", 0, 1, reads=["step_state"])]).check_host_points()
    with pytest.raises(ValueError, match="must not decrease"):
        _validation_program([HostPoint("t", 1, 1), HostPoint("u", 0, 2)]).check_host_points()


def test_engine_rejects_missing_tasks_host_buffers_and_aliases():
    dev = nt.Device()
    staging = nt.Buffer(dev, W * 4)
    program = _validation_program([HostPoint("t", 1, 1, writes=["staging"])])
    with pytest.raises(ValueError, match="no task supplied"):
        Engine(program, dev, buffers={"staging": staging})
    with pytest.raises(ValueError, match="the session allocates it"):
        Engine(program, dev, host_tasks={"t": lambda s, k: None})
    with pytest.raises(ValueError, match=f"{W * 4 - 4} bytes, program needs {W * 4}"):
        Engine(program, dev, buffers={"staging": nt.Buffer(dev, W * 4 - 4)}, host_tasks={"t": lambda s, k: None})
    # the task writes staging while produce (in the window) writes x: one buffer passed under both names
    aliased = _validation_program([HostPoint("t", 0, 1, writes=["staging"])])
    _validation_program([HostPoint("t", 0, 1, writes=["staging"])]).check_host_points()   # distinct by name
    with pytest.raises(ValueError, match=r"op 0 \(produce\) accesses 'x' while the task writes 'staging'"):
        Engine(aliased, dev, buffers={"staging": staging, "x": staging}, host_tasks={"t": lambda s, k: None})


@pytest.mark.parametrize("points", [[], [HostPoint("t", 0, 1, writes=["b"])]], ids=["plain", "host-steps"])
def test_engine_rejects_an_unknown_indirect_buffer(points):
    buffers = {**_base_buffers(3), "x": BufferSpec(W * 4), "b": BufferSpec(16)}
    produce = OpSpec("produce", [(0, "x", 0), (1, "step_state", 0), (2, "params", 0)], (1, 1, 1), (W, 1, 1), True,
                     name="produce", meta={"writes": [0], "indirect_reads": ["nope"]})
    with pytest.raises(ValueError, match="reaches 'nope' through a GPU address, which is not a program buffer"):
        Engine(_program(buffers, [produce, _tick()], points), host_tasks={"t": lambda s, k: None})


@pytest.mark.parametrize("host_point", [True, False], ids=["host-steps", "plain"])
@pytest.mark.parametrize("reencode", [False, True])
@pytest.mark.parametrize("residency_set", ["1", "0"])
def test_unbound_host_buffers_reached_through_gpu_addresses(monkeypatch, residency_set, reencode, host_point):
    monkeypatch.setenv("LITHOS_RESIDENCY_SET", residency_set)
    dev = nt.Device()
    n = 512
    left = np.arange(n // 2, dtype=np.uint32) * 3 + 1
    right = np.arange(n // 2, dtype=np.uint32) * 5 + 2
    ha, hb = nt.Buffer(dev, left.tobytes()), nt.Buffer(dev, right.tobytes())
    table = nt.Buffer(dev, struct.pack("<QQ", ha.gpu_address, hb.gpu_address))
    out = nt.Buffer(dev, n * 4)
    buffers = {**_base_buffers(3, width=n), "table": BufferSpec(16, role="host"), "out": BufferSpec(n * 4, role="host"),
               "ha": BufferSpec(n * 2, role="host"), "hb": BufferSpec(n * 2, role="host")}
    gather = OpSpec("gather", [(0, "table", 0), (1, "out", 0), (2, "step_state", 0), (3, "params", 0)], (1, 1, 1), (n, 1, 1),
                    True, name="gather", meta={"writes": [1], "indirect_reads": ["ha", "hb"]})
    calls = []
    points = [HostPoint("t", 1, 1, reads=["out"])] if host_point else []
    eng = Engine(_program(buffers, [gather, _tick()], points), dev,
                 buffers={"table": table, "out": out, "ha": ha, "hb": hb}, host_tasks={"t": lambda s, k: calls.append(s)})
    rep = eng.run(3, reencode=reencode)
    want = np.empty(n, np.uint32)
    want[0::2], want[1::2] = left, right
    assert rep.steps == 3 and calls == ([0, 1, 2] if host_point else [])
    assert np.array_equal(np.frombuffer(out.read(0, n * 4), dtype=np.uint32), want + 2)   # the last step ran at step 2, in place
    if not host_point:                                                    # Engine.profile declares them on its own queue
        eng.buffers["step_state"].write(struct.pack("<I", 0), LAYOUT.offset("done"))
        eng.profile(1)
        assert np.array_equal(np.frombuffer(out.read(0, n * 4), dtype=np.uint32), want + 3)


def test_residency_declarations_hold_under_shader_validation():
    """Only Metal shader validation sees a kernel reach a buffer nothing declared resident: the cases above run again
    with it on, whatever this run's environment."""
    import os
    import re
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_") and k != "PYTEST_ADDOPTS"}   # its summary line
    env["MTL_SHADER_VALIDATION"] = "1"
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", str(here), "-k", "unbound_host_buffers"],
                       cwd=here.parents[2], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and re.search(r"\b8 passed\b", p.stdout), (p.stdout[-3000:], p.stderr[-2000:])


def test_host_points_round_trip_through_json(tmp_path):
    p = _multi_program(5)
    p.save(tmp_path / "program.json")
    q = Program.load(tmp_path / "program.json")
    assert q.to_json() == p.to_json() and q.host_points == p.host_points
    import json
    d = json.loads(p.to_json())
    del d["host_points"]
    assert Program.from_json(json.dumps(d)).host_points == []


_REENTRANT = """
import gc, sys
sys.path[:0] = [{root!r}, {here!r}]
import test_host_steps as t
from monolith.runtime import _native as nt

class Collects:
    def __call__(self, step, point):
        pass
    def __del__(self):
        gc.collect()                         # a collector pass while the runner drops this task's last reference

for i in range(50):
    ex = t._Exchange()
    eng = ex.engine(5)
    err = eng.program.layout.offset("error")
    eng.runner.set_host_points([(1, 1, nt.CallableTask(Collects()))], err, 1000)
    eng.runner.set_host_points([(1, 1, nt.CallableTask(Collects()))], err, 1000)
    del eng, ex
    gc.collect()
print("done")
"""


def test_a_task_dropped_while_python_runs_leaves_the_runner_consistent():
    """Dropping a task's last reference runs Python (its finalizer), which may walk the runner's points through the
    collector: set_host_points and the collector's clear must leave the points consistent first. Runs under malloc
    scribbling, which turns a read of freed task state into a crash."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    env = {**os.environ, "MallocScribble": "1", "MallocGuardEdges": "1"}
    code = _REENTRANT.format(root=str(here.parents[1]), here=str(here))
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "done" in p.stdout, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])


def test_an_engine_its_own_task_reaches_is_collected():
    import gc
    import weakref
    ex = _Exchange()
    eng = ex.engine(5)
    assert eng.run(5).steps == 5
    ref = weakref.ref(eng)
    del eng, ex
    gc.collect()
    assert ref() is None
