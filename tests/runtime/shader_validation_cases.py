"""The cases ``test_shader_validation.py`` runs in a pytest subprocess under Metal shader validation; a tier run does not
collect this file (no ``test_`` prefix). One kernel stores 64 floats into a buffer of 64 (in bounds) or 32 (out of
bounds) through each command-buffer path of the runtime: ``Queue.run``, ``Engine.run`` (ICB replay by the host pump)
and ``Engine.profile`` (``Queue.profile``). No case checks a result (the queue case drops its ``RunResult``), so only
the runtime can fail one; the module fixture's teardown stores out of bounds after every test body has ended. Three
cases never let the raised error reach pytest as a failure, so only ``tests/conftest.py`` can fail them: ``caught``
swallows it (``except Exception`` in a harness), ``skipped`` turns it into a skip and ``thread`` raises it on another
thread (pytest only warns)."""
import threading

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
@pytest.mark.parametrize("path", ["queue", "engine", "profile", "caught", "skipped", "thread"])
def test_store(path, floats):
    if path in ("queue", "caught", "skipped"):
        dev = nt.Device()
        out = nt.Buffer(dev, floats * 4)
        pso = nt.Pipeline(nt.Library(dev, SRC), "store64")
        try:
            nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0, out).grid(2).threadgroup(32)])
        except RuntimeError as e:
            if path == "queue":
                raise
            if path == "skipped":
                pytest.skip(f"refused: {e}")
    elif path == "engine":
        Engine(_program(floats)).run(1, steps_per_cb=1, in_flight=1)
    elif path == "thread":
        t = threading.Thread(target=Engine(_program(floats)).run, args=(1,), kwargs={"steps_per_cb": 1, "in_flight": 1})
        t.start()
        t.join()
    else:
        Engine(_program(floats)).profile(1)


@pytest.fixture(scope="module")
def store_after_the_tests():
    yield
    Engine(_program(32)).run(1, steps_per_cb=1, in_flight=1)


def test_outside_the_test_body(store_after_the_tests):
    pass
