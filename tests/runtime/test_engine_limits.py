"""The Engine's pipeline checks: an op whose threadgroup exceeds its pipeline's ``max_threads_per_threadgroup`` is
refused when the program is instantiated, and so is a pipeline whose SIMD-group is not 32 lanes (every kernel's crews
and butterflies assume 32; basalt's warp is a SIMD-group)."""

import numpy as np
import pytest

from monolith.core.step_state import StepStateLayout
from monolith.runtime import BufferSpec, Engine, KernelSpec, OpSpec, Program
from monolith.runtime import _native as nt
from monolith.runtime import engine as engine_module

LAYOUT = StepStateLayout()
CAPPED = """
#include <metal_stdlib>
using namespace metal;
[[max_total_threads_per_threadgroup(64)]]
kernel void capped(device uint* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = i + 1u; }
"""


def capped_program(threads):
    """One dispatch of ``capped``, a kernel built to take at most 64 threads a threadgroup, with ``threads``."""
    return Program(kernels={"capped": KernelSpec(CAPPED, "capped")},
                   buffers={"y": BufferSpec(4 * threads, None, "arena"),
                            "step_state": BufferSpec(LAYOUT.size, LAYOUT.pack({}), "step_state"),
                            "ring": BufferSpec(16 * 8, None, "ring")},
                   ops=[OpSpec("capped", [(0, "y", 0)], (1, 1, 1), (threads, 1, 1), True, name="capped",
                               meta={"writes": [0]})],
                   ring_capacity=16, layout=LAYOUT)


def test_refuses_a_threadgroup_past_its_pipelines_limit():
    pso = nt.Pipeline(nt.Library(nt.Device(), CAPPED, {}), "capped")
    assert (pso.max_threads_per_threadgroup, pso.thread_execution_width) == (64, 32)
    eng = Engine(capped_program(64))
    assert eng.run(1).steps == 1
    assert np.array_equal(np.frombuffer(eng.read("y"), np.uint32), np.arange(1, 65))
    with pytest.raises(ValueError, match="op capped: a threadgroup of 128 threads; its pipeline takes at most 64"):
        Engine(capped_program(128))


def test_refuses_a_simd_group_of_other_than_32_lanes(monkeypatch):
    """No Apple GPU reports another width, so the check meets a stand-in pipeline that reports 16 lanes."""
    real = engine_module.compile_pipelines

    class Narrow:
        thread_execution_width = 16

        def __init__(self, pso):
            self.max_threads_per_threadgroup = pso.max_threads_per_threadgroup

    assert all(p.thread_execution_width == 32 for p in Engine(capped_program(64)).pipelines.values())
    monkeypatch.setattr(engine_module, "compile_pipelines",
                        lambda *a, **k: {n: Narrow(p) for n, p in real(*a, **k).items()})
    with pytest.raises(ValueError, match="kernel capped: SIMD-groups of 16 lanes; the kernels assume 32"):
        Engine(capped_program(64))
