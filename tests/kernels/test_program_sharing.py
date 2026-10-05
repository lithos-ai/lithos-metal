"""Programs of one session share device buffers by name (weights, states, StepState, the ring, arena values); a
params record is a program's own bytes and is never taken from the shared set — a static T = 8 program's o_proj
tiles once ran with another GEMV's record (24576 rows into a 4096-row output: out-of-bounds loads and stores) because
the dynamic program at t_max = 8 had a params buffer of the same name (#113)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.bench import profile_for_device  # noqa: E402
from monolith.compiler import compile_program  # noqa: E402
from monolith.core import StepStateLayout  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime import Engine  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402


def test_params_records_are_never_shared(tmp_path):
    import pytest

    dev = nt.Device()
    info = dev.info()
    prof = profile_for_device(info.gpu_cores, info.apple_family)
    if prof is None:
        pytest.skip("no chip profile for this device")
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16, lane_order=prof.lane_order))
    pf = PackFile(tmp_path / "pack")
    layout = StepStateLayout()
    dyn = compile_program(m, pf, prof, dynamic_t=True, layout=layout)
    static = compile_program(m, pf, prof, t=layout.t_max, layout=layout)
    first = Engine(dyn, dev)
    # a shared set that holds every params name of the second program with other bytes of the same size: the engine
    # must allocate its own records and take the states, StepState and ring from the set
    shared = dict(first.buffers)
    for name, spec in static.buffers.items():
        if spec.role == "params":
            b = nt.Buffer(dev, spec.nbytes)
            b.write(bytes([0xFF]) * spec.nbytes, 0)
            shared[name] = b
    second = Engine(static, dev, buffers=shared)
    for eng in (first, second):
        for op in eng.program.ops:
            for _index, name, off in op.bindings:
                spec = eng.program.buffers[name]
                if spec.role == "params":
                    assert eng.read(name)[off: off + spec.nbytes] == spec.init, (op.name, name)
    for name, spec in static.buffers.items():
        if spec.role in ("state", "step_state", "ring", "weights") and name in first.buffers:
            assert second.buffers[name] is first.buffers[name], name
