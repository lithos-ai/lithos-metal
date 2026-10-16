"""The cases ``test_shader_validation.py`` runs in a pytest subprocess under Metal shader validation; a tier run does not
collect this file (no ``test_`` prefix). One kernel stores 64 floats into a buffer of 64 (in bounds) or 32 (out of
bounds) through each command-buffer path of the runtime: ``Queue.run``, ``Engine.run`` (ICB replay by the host pump)
and ``Engine.profile`` (``Queue.profile``). Each case checks only what an ordinary GPU test checks."""
import pytest

from monolith.core.step_state import StepStateLayout
from monolith.runtime import BufferSpec, Engine, KernelSpec, OpSpec, Program, _native as nt

SRC = """#include <metal_stdlib>
using namespace metal;
kernel void store64(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i] = float(i); }
"""
LAYOUT = StepStateLayout()


def _program(floats: int) -> Program:
    return Program(kernels={"store": KernelSpec(SRC, "store64")},
                   buffers={"out": BufferSpec(floats * 4),
                            "step_state": BufferSpec(LAYOUT.size, LAYOUT.pack({}), "step_state"),
                            "ring": BufferSpec(64, role="ring")},
                   ops=[OpSpec("store", [(0, "out", 0)], (2, 1, 1), (32, 1, 1))], ring_capacity=8, layout=LAYOUT)


@pytest.mark.parametrize("floats", [64, 32], ids=["in_bounds", "out_of_bounds"])
@pytest.mark.parametrize("path", ["queue", "engine", "profile"])
def test_store(path, floats):
    if path == "queue":
        dev = nt.Device()
        out = nt.Buffer(dev, floats * 4)
        pso = nt.Pipeline(nt.Library(dev, SRC), "store64")
        r = nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0, out).grid(2).threadgroup(32)])
        assert not r.error, r.error
    elif path == "engine":
        Engine(_program(floats)).run(1, steps_per_cb=1, in_flight=1)
    else:
        Engine(_program(floats)).profile(1)
