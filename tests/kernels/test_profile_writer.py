"""tools/profile_writer.py on the device: a short measurement at a small shape produces a profile the loader accepts
and the decisions' inputs are all present (the real run takes minutes and writes the selected backend configuration)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith.core.profile import Profile, load_profile  # noqa: E402
from monolith.core.profile_writer import decide, merge_profile  # noqa: E402
from monolith.runtime import is_available  # noqa: E402

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


def test_writer_measures_and_the_profile_loads(tmp_path):
    from tools.profile_writer import device_facts, measure
    from monolith.runtime import _native as nt

    lines = []
    m = measure(shape=(2048, 1024), formats=["fp8_e4m3", "nvfp4"], ts=[1, 2], tms=[8], copies=4, reps=1, attention=(8, 2, 64), ctxs=[256],
                attn_ts=[1], log=lines.append)
    assert set(m["lane_order_gbps"]) == {"interleaved16", "contiguous"} and set(m["threadgroups_ms"]) == {1, 2}
    assert set(m["shader_ms"]["fp8_e4m3"]) == {1, 2} and set(m["shader_ms"]["nvfp4"]) == {1, 2} and all(v > 0 for v in m["shader_ms"]["nvfp4"].values())
    assert not any("ORACLE FAIL" in l for l in lines), [l for l in lines if "FAIL" in l]
    engine, notes = decide(m, None)
    info = nt.Device().info()
    doc = merge_profile(None, device=device_facts(info, target_gb=21.0), engine=engine, measurements=m, notes=notes, written="2026-09-25", command="test")
    p = Profile.from_dict("x", doc)
    assert p.family == f"Apple{info.apple_family}" and p.gpu_cores == info.gpu_cores and p.cost("fp8", 1) == 1.0 and 2 in p.cost_t["fp8"]
    assert p.lane_order in ("contiguous", "interleaved16") and p.threadgroups_per_core in (1, 2) and p.attention in ("v1", "v2", "auto")
    if m["tile_ms"]:                                                       # Apple10: the tile ran and its rows are in the same unit
        assert p.cost_t["accelerator_fp8"][8] > 0 and p.accelerator in ("on", "off")
    out = tmp_path / "x.json"
    out.write_text(json.dumps(doc, indent=2))
    assert load_profile(out).nominal_gbps == p.nominal_gbps and doc["writer"]["decisions"]["lane_order"]
