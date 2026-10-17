"""Instantiate a :class:`Program` on the device and replay it: the Python face of the host pump (design §5.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from . import _native as nt
from .program import Program


@dataclass
class StepReport:
    steps: int
    command_buffers: int
    gpu_ms: float
    wall_ms: float
    host_busy_ms: float
    done: bool
    tokens: List[int]
    encode_ms: float = 0.0
    commit_ms: float = 0.0
    wait_ms: float = 0.0
    handoff_gaps_ms: List[float] = field(default_factory=list)   # GPU idle time at each host-point handoff

    @property
    def host_fraction(self) -> float:
        """CPU time of the pump over wall time — the '< 5 % of a core' metric."""
        return self.host_busy_ms / self.wall_ms if self.wall_ms else 0.0


def compile_pipelines(program, device, *, fast_math=False, cache=None):
    """Prepare executable kernels without allocating or retaining model buffers."""
    pipelines = {}
    for key, k in program.kernels.items():
        identity = (device, k.source, tuple(sorted(k.macros.items())),
                    k.language_version, fast_math, k.function)
        pipeline = cache.get(identity) if cache is not None else None
        if pipeline is None:
            lib = nt.Library(device, k.source, k.macros, k.language_version, fast_math)
            pipeline = nt.Pipeline(lib, k.function, True)
            if cache is not None:
                cache[identity] = pipeline
        pipelines[key] = pipeline
    return pipelines


class Engine:
    """``buffers`` lets several programs share device buffers by name (a prefill program at T = P and a decode
    program at T = 1 over the same weights, states and StepState), and supplies every buffer of role "host"."""

    def __init__(self, program: Program, device: Optional[nt.Device] = None, buffers: Optional[Dict[str, nt.Buffer]] = None,
                 fast_math: bool = False, pipeline_cache: Optional[dict] = None,
                 host_tasks: Optional[Dict[str, Union["nt.HostTask", Callable[[int, int], None]]]] = None,
                 host_timeout_ms: int = 20000) -> None:
        """``fast_math``: compile the kernels with Metal's fast math mode (the default is the safe mode).
        ``host_tasks``: the task of each host point, by its task name: an ``nt.HostTask`` or a Python callable
        ``fn(step, point)``. ``host_timeout_ms``: the longest a run waits for the GPU to reach a host point or finish a
        command buffer."""
        self.program = program
        self.fast_math = fast_math
        self.dev = device or nt.Device()
        self.buffers: Dict[str, nt.Buffer] = {}
        for name, spec in program.buffers.items():
            if spec.role == "host":                          # the session allocates it; the program binds it by name
                if buffers is None or name not in buffers:
                    raise ValueError(f"host buffer {name}: the session allocates it and passes it in Engine(buffers=...)")
                if buffers[name].nbytes < spec.nbytes:
                    raise ValueError(f"host buffer {name}: {buffers[name].nbytes} bytes, program needs {spec.nbytes}")
                self.buffers[name] = buffers[name]
                continue
            if buffers is not None and name in buffers and spec.role != "params" and buffers[name].nbytes >= spec.nbytes:
                self.buffers[name] = buffers[name]           # shared (weights, states, StepState, ring, arena values)
                continue                                     # never a params record: its bytes are this program's own
            if buffers is not None and name in buffers and spec.role in ("state", "step_state", "ring", "weights"):
                raise ValueError(f"shared buffer {name}: {buffers[name].nbytes} bytes, program needs {spec.nbytes}")
            if spec.file is not None:
                self.buffers[name] = nt.Buffer.from_file(self.dev, spec.file, spec.file_offset, spec.nbytes)
                continue
            if spec.init is not None:
                if len(spec.init) > spec.nbytes:
                    raise ValueError(f"buffer {name}: init larger than nbytes")
                buf = nt.Buffer(self.dev, spec.nbytes)
                buf.fill(0)
                buf.write(spec.init, 0)
            else:
                buf = nt.Buffer(self.dev, spec.nbytes)
                buf.fill(0)
            self.buffers[name] = buf
        for op in program.ops:
            for name in [*op.meta.get("indirect_reads", ()), *op.meta.get("indirect_writes", ())]:
                if name not in self.buffers:
                    raise ValueError(f"op {op.name or op.kernel}: reaches {name!r} through a GPU address, which is not a program buffer")
        if program.host_points:
            program.check_host_points(self._same_memory)
        self.pipelines = compile_pipelines(program, self.dev, fast_math=fast_math, cache=pipeline_cache)
        self.ops = []
        for o in program.ops:
            d = nt.Dispatch().pipeline(self.pipelines[o.kernel]).grid(*o.grid).threadgroup(*o.threadgroup).barrier(o.barrier_before)
            for index, bname, off in o.bindings:
                d.buffer(index, self.buffers[bname], off)
            for index, length in o.threadgroup_memory:
                d.threadgroup_memory(index, length)
            for name in o.meta.get("indirect_reads", ()):     # Queue.run and Queue.profile declare these resident
                d.use(self.buffers[name])
            for name in o.meta.get("indirect_writes", ()):
                d.use(self.buffers[name], True)
            self.ops.append(d)
        self.icb = nt.Icb(self.dev, self.ops)
        lay = program.layout
        ring = self.buffers[program.ring]
        if ring.nbytes < program.ring_capacity * 8:
            raise ValueError("the token ring needs 8 bytes per slot: (sequence << 32) | token")
        # Programs can retain unreferenced state buffers for session reuse. Only buffers the ICB binds, buffers its
        # kernels reach through GPU addresses, and the session's host buffers need a Metal residency declaration.
        resources = {name: self.buffers[name] for op in program.ops for _, name, _ in op.bindings}
        for op in program.ops:
            for name in [*op.meta.get("indirect_reads", ()), *op.meta.get("indirect_writes", ())]:
                resources[name] = self.buffers[name]
        resources.update({name: self.buffers[name] for name, spec in program.buffers.items() if spec.role == "host"})
        self.runner = nt.Runner(self.dev, self.icb, self.ops, list(resources.values()), self.buffers[program.step_state],
                                lay.offset("done"), lay.offset("ring_head"), lay.offset("ring_tail"), ring, program.ring_capacity,
                                [buffer for name, buffer in resources.items() if program.buffers[name].role in ('weights', 'params')])
        self.host_tasks = []
        if program.host_points:
            for p in program.host_points:
                task = (host_tasks or {}).get(p.task)
                if task is None:
                    raise ValueError(f"host point {p.task!r}: no task supplied (Engine(host_tasks={{{p.task!r}: ...}}))")
                self.host_tasks.append((p.publish, p.wait, task if isinstance(task, nt.HostTask) else nt.CallableTask(task)))
            self.runner.set_host_points(self.host_tasks, lay.offset("error"), host_timeout_ms)

    def _same_memory(self, a: str, b: str) -> bool:
        """Whether two buffer names reach overlapping memory (a session can pass one buffer under two names)."""
        x, y = self.buffers[a], self.buffers[b]
        return a == b or (x.gpu_address < y.gpu_address + y.nbytes and y.gpu_address < x.gpu_address + x.nbytes)

    def run(self, max_steps: int, *, steps_per_cb: int = 8, in_flight: int = 3, reencode: bool = False, max_tokens: int = 0) -> StepReport:
        """Replay up to ``max_steps`` steps (``max_tokens`` > 0: stop submitting once that many tokens arrived; the
        queued buffers still complete, so a few more steps may run). A failed run raises RuntimeError with the run's
        StepReport, its tokens included, as ``report``; the exception a Python task raised is its cause, except that a
        task's KeyboardInterrupt or SystemExit is raised as it was."""
        tasks = self.runner.host_tasks()                     # read before the run: a reconfiguration cannot shift the
        st = self.runner.run(max_steps, steps_per_cb, in_flight, reencode, max_tokens)   # index of the failing task
        report = StepReport(st.steps_submitted, st.command_buffers, st.gpu_ms, st.wall_ms, st.host_busy_ms, st.done, list(st.tokens),
                            st.encode_ms, st.commit_ms, st.wait_ms, list(st.handoff_gaps_ms))
        code = int(self.state()["error"]) if st.done or st.steps_submitted == 0 else 0
        message = st.error or {3: "Megakernel: bounded worker barrier timed out",
                               4: "a host step failed in an earlier run; the program's state is unusable"}.get(code)
        if not message:
            return report
        cause = None
        if 0 <= st.failed_task < len(tasks):                  # the failing task's own: an older one left elsewhere stays out
            task = tasks[st.failed_task]
            cause = task.take_exception() if isinstance(task, nt.CallableTask) else None
        if cause is not None and not isinstance(cause, Exception):
            raise cause                                       # KeyboardInterrupt, SystemExit
        error = RuntimeError(message)
        error.report = report
        raise error from cause

    def profile(self, steps: int = 3) -> List[List[Tuple[float, float]]]:
        """Per-dispatch GPU (start, end) ms for ``steps`` re-encoded steps (one encoder per op with timestamp counter
        samples, so the numbers carry encoder-boundary gaps the ICB replay does not have — use them for the shares
        and the per-op durations, not for the step total)."""
        if self.program.host_points or self.runner.host_tasks():
            raise RuntimeError("profile() dispatches the ops without their tasks; a program with host points runs only through run()")
        q = nt.Queue(self.dev)
        return [q.profile(self.ops) for _ in range(steps)]

    def state(self) -> Dict[str, object]:
        buf = self.buffers[self.program.step_state]
        return self.program.layout.unpack(buf.read(0, self.program.layout.size))

    def read(self, name: str, nbytes: Optional[int] = None) -> bytes:
        b = self.buffers[name]
        return b.read(0, nbytes if nbytes is not None else b.nbytes)
