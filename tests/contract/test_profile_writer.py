"""The profile writer's decisions (monolith.core.profile_writer, #49): pure functions over harness numbers, tested
here without a GPU; tools/profile_writer.py runs the harnesses and hands them these numbers."""

import json

import pytest

from monolith.core.profile import Profile, load_profiles
from monolith.core.profile_writer import (NEVER, accelerator_plan, attention_choice, choose_lane_order, choose_threadgroups, cost_table, decide,
                                          merge_profile, profile_name, tile_rows)


def test_profile_name_and_tile_rows():
    assert profile_name("Apple M5 Pro", 20) == "apple-m5-pro-20c" and profile_name("Apple M3 Pro", 18) == "apple-m3-pro-18c"
    assert [tile_rows(t) for t in (2, 8, 9, 16, 17, 32)] == [8, 8, 16, 16, 32, 32]
    with pytest.raises(ValueError):
        tile_rows(33)


def test_lane_order_and_threadgroups_follow_the_noise_rule():
    assert choose_lane_order({"interleaved16": 275, "contiguous": 238})[0] == "interleaved16"      # the M5 Pro (p13)
    assert choose_lane_order({"contiguous": 134, "interleaved16": 132}, "contiguous")[0] == "contiguous"   # the M3 Pro's tie keeps its value
    assert choose_lane_order({"contiguous": 134, "interleaved16": 132})[0] == "interleaved16"      # a tie on a new chip: the default
    assert choose_lane_order({"contiguous": 150, "interleaved16": 132})[0] == "contiguous"
    assert choose_threadgroups({1: 1.0, 2: 0.99})[0] == 1 and choose_threadgroups({1: 1.0, 2: 0.9})[0] == 2 and choose_threadgroups({1: 1.0, 2: 1.2})[0] == 1
    assert choose_threadgroups({1: 1.0, 2: 0.99}, current=2)[0] == 2
    with pytest.raises(ValueError):
        choose_lane_order({})


def test_cost_table_is_relative_to_t1():
    assert cost_table({1: 2.0, 2: 2.2, 4: 2.4, 8: 7.2}) == {1: 1.0, 2: 1.1, 4: 1.2, 8: 3.6}
    with pytest.raises(ValueError):
        cost_table({2: 1.0})


def test_accelerator_plan_reproduces_the_m5_pro_decision():
    m5 = load_profiles()["apple-m5-pro-20c"]
    shader = {"fp8": m5.cost_t["fp8"], "nvfp4": m5.cost_t["nvfp4"]}
    tile = {"fp8": m5.cost_t["accelerator_fp8"], "nvfp4": m5.cost_t["accelerator_nvfp4"]}
    on, min_t, why = accelerator_plan(shader, tile)
    assert on == "on" and min_t == {"fp8": 2, "nvfp4": 2} and "fp8: the tile from T = 2" in why      # FP8's 1.09 vs 1.08 is a tie: the tile
    # a format the tile never wins keeps the shader everywhere; a chip without the tile is off
    assert accelerator_plan({"bf16": {1: 1.0, 2: 1.1, 4: 1.3, 8: 3.0}}, {"bf16": {8: 4.0, 16: 4.1, 32: 5.0}})[:2] == ("off", {"bf16": NEVER})
    assert accelerator_plan({"bf16": {1: 1.0, 2: 1.1, 4: 1.3, 8: 3.0}}, {"bf16": {8: 1.3, 16: 1.5, 32: 2.5}})[:2] == ("on", {"bf16": 4})   # 1.3 > 1.1 × 1.03, ≤ 1.3 × 1.03
    assert accelerator_plan(shader, {})[:2] == ("off", {})


def test_attention_choice():
    assert attention_choice({(1024, 1): 0.10, (4096, 4): 0.30}, {(1024, 1): 0.08, (4096, 4): 0.25})[0] == "v2"
    assert attention_choice({(1024, 1): 0.10, (4096, 4): 0.30}, {(1024, 1): 0.08, (4096, 4): 0.30})[0] == "v1"    # not everywhere
    assert attention_choice({(1024, 1): 0.10}, {})[0] == "v1"
    # v3: auto (= v3) when it beats v1 and v2 at every point measured; otherwise the v1 / v2 rule
    v1m, v2m = {(1024, 1): 0.10, (4096, 1): 0.30, (4096, 4): 0.60}, {(1024, 1): 0.08, (4096, 1): 0.25, (4096, 4): 0.50}
    assert attention_choice(v1m, v2m, v3={(1024, 1): 0.05, (4096, 1): 0.20, (4096, 4): 0.40})[0] == "auto"
    assert attention_choice(v1m, v2m, v3={(1024, 1): 0.05, (4096, 1): 0.29})[0] == "v2"                     # not faster than v2 at (4096, 1)
    assert attention_choice(v1m, {**v2m, (4096, 4): 0.61}, v3={(1024, 1): 0.05, (4096, 1): 0.20, (4096, 4): 0.40})[0] == "auto"  # v3 wins everywhere; v2's loss is moot
    assert attention_choice(v1m, {**v2m, (4096, 4): 0.61}, v3={(1024, 1): 0.05, (4096, 1): 0.31, (4096, 4): 0.40})[0] == "v1"    # v3 loses a point, and so does v2: v1
    assert attention_choice(v1m, v2m, v3={(8192, 8): 0.1})[0] == "v2"                                       # no common point
    assert attention_choice(v1m, v2m, v3={(1024, 1): 0.05, (4096, 1): 0.20})[0] == "v2"                     # a v1 point v3 could not run: not auto


def _measurements():
    return {"family": "Apple10", "chip": "Apple M5 Pro", "shape": "17408x5120",
            "lane_order_gbps": {"interleaved16": 275.0, "contiguous": 238.0}, "threadgroups_ms": {1: 0.324, 2: 0.330},
            "shader_ms": {"fp8_e4m3": {1: 0.324, 2: 0.350, 4: 0.360, 8: 1.166}, "nvfp4": {1: 0.196, 2: 0.251, 4: 0.351, 8: 1.039}},
            "tile_ms": {"fp8_e4m3": {8: 0.353, 16: 0.376, 32: 0.680}, "nvfp4": {8: 0.202, 16: 0.204, 32: 0.455}},
            "attention_ms": {"v1": {(1024, 1): 0.05, (4096, 4): 0.2}, "v2": {(1024, 1): 0.06, (4096, 4): 0.21}}}


def test_leaf_writer_preserves_layer_fusion_selection():
    cfg = load_profiles()["apple-m5-max-40c"].gdn_mixer_fusion
    engine, notes = decide(_measurements(), {"gdn_mixer_fusion": cfg})
    assert engine["gdn_mixer_fusion"] == cfg
    assert "layer study" in notes["gdn_mixer_fusion"]


def test_decide_and_merge_produce_a_loadable_profile(tmp_path):
    engine, notes = decide(_measurements(), {"sibling_order": "alu_first", "max_cb_ms": 16, "lane_order": "contiguous", "attention_rows": 8})
    assert engine["attention_rows"] == 8                                                  # not measured: the file's value carries over
    assert decide(_measurements(), {"sibling_order": "alu_first"})[0]["attention_rows"] == 4
    assert engine["family"] == "Apple10" and engine["lane_order"] == "interleaved16" and engine["threadgroups_per_core"] == 1
    assert engine["sibling_order"] == "alu_first" and engine["max_cb_ms"] == 16 and engine["attention"] == "v1"
    assert engine["accelerator"] == "on" and engine["accelerator_min_t"] == {"fp8": 2, "nvfp4": 2}
    assert engine["cost_T"]["fp8"] == {"1": 1.0, "2": 1.08, "4": 1.111, "8": 3.599} and engine["cost_T"]["nvfp4"]["4"] == pytest.approx(1.791)
    assert engine["cost_T"]["accelerator_fp8"] == {"8": 1.09, "16": 1.16, "32": 2.099} and set(notes) == {"lane_order", "scale_placement", "threadgroups_per_core", "accelerator", "attention"}
    assert engine["scale_placement"] == "inline"                                          # no placement measurement: the default stays
    existing = {"chip": "Apple M5 Pro", "gpu_cores": 20, "nominal_gbps": 307.0, "measured": "2026-09-22", "streaming": {"probe": "p5b"},
                "engine": {"family": "Apple10", "lane_order": "contiguous", "sibling_order": "alu_first"}}
    device = {"chip": "Apple M5 Pro", "gpu_family": "Apple10", "gpu_cores": 20, "memory_gb": 24, "os": "macOS 26.5.1", "gpu_working_set_gb": 19.07,
              "max_buffer_gb": 14.3, "hosts_target_model": False, "hosts_target_model_note": "21 GB against 19.07"}
    doc = merge_profile(existing, device=device, engine=engine, measurements=_measurements(), notes=notes, written="2026-09-25", command="python tools/profile_writer.py")
    assert doc["streaming"] == {"probe": "p5b"} and doc["measured"] == "2026-09-22" and doc["nominal_gbps"] == 307.0 and "nominal_note" not in doc
    assert doc["writer"]["previous_engine"]["lane_order"] == "contiguous" and doc["writer"]["measurements"]["attention_ms"]["v1"]["(1024, 1)"] == 0.05
    p = Profile.from_dict("apple-m5-pro-20c", json.loads(json.dumps(doc)))
    assert p.lane_order == "interleaved16" and p.accelerator == "on" and p.cost("accelerator_nvfp4", 16) == pytest.approx(1.041) and p.cost("fp8", 3) == pytest.approx((1.08 + 1.111) / 2)
    # a new chip without a spec figure: the measured stand-in, flagged
    fresh = merge_profile(None, device=device, engine=engine, measurements=_measurements(), notes=notes, written="2026-09-25", command="x")
    assert fresh["nominal_gbps"] == 280.0 and "stand-in" in fresh["nominal_note"] and fresh["measured"] == "2026-09-25"
    assert Profile.from_dict("new", fresh).threadgroups_per_core == 1
    # no tile measurement (Apple9): the shader path, an empty min_t
    no_tile = dict(_measurements(), tile_ms={}, attention_ms={})
    eng9, notes9 = decide(dict(no_tile, family="Apple9"), None)
    assert eng9["accelerator"] == "off" and eng9["accelerator_min_t"] == {} and eng9["attention"] == "v1" and eng9["sibling_order"] == "either"
    assert "no tile measurement" in notes9["accelerator"] and "accelerator_fp8" not in eng9["cost_T"]
