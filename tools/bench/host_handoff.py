"""GPU idle time per host handoff, as the host-step runtime makes it (RunnerStats.handoff_gaps_ms): for each command
buffer the pump commits once a host point's task returned, its GPU start minus the GPU end of the command buffer before
it. The program is one GPU-wide pass of about half a millisecond, one host point with a no-op Python task, and a tick.
Run without shader validation:

    python tools/bench/host_handoff.py --rounds 4 --steps 300
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.core.step_state import StepStateLayout  # noqa: E402
from monolith.runtime import BufferSpec, Engine, HostPoint, KernelSpec, OpSpec, Program  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402

LAYOUT = StepStateLayout(t_max=8, gamma_max=7)
SRC = """
#include <metal_stdlib>
using namespace metal;
""" + LAYOUT.to_msl() + """
struct Params { uint n_stop; uint ring_cap; uint iters; uint pad; };
kernel void busy(device float* sink [[buffer(0)]], device const StepState* st [[buffer(1)]], constant Params& p [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  if (st->done) return;
  float v = float(i);
  for (uint k = 0; k < p.iters; k++) v = fma(v, 1.0000001f, 1.0e-7f);
  sink[i] = v;
}
kernel void tick(device StepState* st [[buffer(0)]], constant Params& p [[buffer(1)]], device ulong* ring [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  uint head = st->ring_head;
  if (head - st->ring_tail >= p.ring_cap) { st->error = 1u; st->done = 1u; return; }
  ring[head % p.ring_cap] = (ulong(head + 1u) << 32) | ulong(st->step & 0x7fffffffu);
  st->ring_head = head + 1u;
  st->step += 1u;
  if (st->step >= p.n_stop) st->done = 1u;
}
"""


def engine(cores: int, iters: int) -> Engine:
    threads = cores * 256
    buffers = {"step_state": BufferSpec(LAYOUT.size, LAYOUT.pack({}), "step_state"),
               "params": BufferSpec(16, struct.pack("<IIII", 1 << 30, 1024, iters, 0), "params"),
               "ring": BufferSpec(1024 * 8, None, "ring"), "sink": BufferSpec(threads * 4)}
    ops = [OpSpec("busy", [(0, "sink", 0), (1, "step_state", 0), (2, "params", 0)], (cores, 1, 1), (256, 1, 1), True,
                  name="busy", meta={"writes": [0]}),
           OpSpec("tick", [(0, "step_state", 0), (1, "params", 0), (2, "ring", 0)], (1, 1, 1), (32, 1, 1), True,
                  name="tick", meta={"writes": [0, 2]})]
    program = Program(kernels={k: KernelSpec(SRC, k) for k in ("busy", "tick")}, buffers=buffers, ops=ops,
                      ring_capacity=1024, layout=LAYOUT, host_points=[HostPoint("noop", 1, 1)])
    return Engine(program, host_tasks={"noop": lambda step, point: None})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300, help="steps per round, after --warmup steps")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20000, help="loop iterations per thread of the pass before the point")
    args = ap.parse_args()
    cores = nt.Device().info().gpu_cores
    eng = engine(cores, args.iters)
    eng.run(args.warmup)
    print(f"{cores}-core GPU; one host point after a {cores}x256-thread pass of {args.iters} iterations per thread")
    for r in range(args.rounds):
        rep = eng.run(args.steps)
        gaps = np.array(rep.handoff_gaps_ms) * 1e3
        p10, p50, p90 = np.percentile(gaps, [10, 50, 90])
        print(f"round {r + 1}: {len(gaps)} handoffs, GPU idle per handoff: min {gaps.min():.1f}  p10 {p10:.1f}  "
              f"median {p50:.1f}  p90 {p90:.1f}  max {gaps.max():.1f} us")


if __name__ == "__main__":
    main()
