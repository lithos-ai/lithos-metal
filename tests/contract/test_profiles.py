import pytest

from monolith.core import load_profiles, profiles_dir
from monolith.backends.metal import config_path
from monolith.core.profile import Profile


def test_repo_profiles_load():
    ps = load_profiles()
    assert {"apple-m3-pro-18c", "apple-m5-pro-20c"} <= set(ps)
    m5 = ps["apple-m5-pro-20c"]
    assert m5.family == "Apple10" and m5.key == "apple10" and m5.gpu_cores == 20 and m5.lane_order == "interleaved16"
    assert m5.sibling_order == "alu_first" and m5.max_cb_ms == 16
    m3 = ps["apple-m3-pro-18c"]
    assert m3.key == "apple9" and m3.lane_order == "contiguous" and (profiles_dir() / "README.md").exists()


def test_cost_tables_measured_points_and_interpolation():
    """The measured points come back as written (the profile writer rewrites them, so the expectations are read from
    the file's engine block, not hard-coded), the points in between are linear, nothing extrapolates."""
    import json

    m5 = load_profiles()["apple-m5-pro-20c"]
    with open(config_path("apple-m5-pro-20c")) as f:
        table = json.load(f)["engine"]["cost_T"]
    assert m5.cost("fp8", 1) == 1.0 and m5.cost("fp8", 2) == pytest.approx(table["fp8"]["2"]) and m5.cost("nvfp4", 4) == pytest.approx(table["nvfp4"]["4"])
    assert 1.0 < table["fp8"]["2"] < table["fp8"]["4"] < table["fp8"]["8"]                                      # a pass costs more with T
    assert m5.cost("fp8", 3) == pytest.approx((table["fp8"]["2"] + table["fp8"]["4"]) / 2)                        # linear between T = 2 and T = 4
    assert m5.cost("accelerator_fp8", 8) == pytest.approx(table["accelerator_fp8"]["8"])                          # gemm_tile rows (the writer, #49)
    assert m5.cost("accelerator_nvfp4", 16) == pytest.approx(table["accelerator_nvfp4"]["16"]) and table["accelerator_nvfp4"]["8"] < 1.2
    with pytest.raises(ValueError):
        m5.cost("fp8", 9)                                                    # never extrapolate
    with pytest.raises(KeyError):
        load_profiles()["apple-m3-pro-18c"].cost("fp8", 2)                  # unmeasured there


def test_profile_validation():
    with pytest.raises(ValueError):
        Profile.from_dict("x", {"gpu_cores": 1, "nominal_gbps": 1.0})                      # no engine block
    with pytest.raises(ValueError):
        Profile.from_dict("x", {"gpu_cores": 1, "nominal_gbps": 1.0, "engine": {"family": "Apple9", "lane_order": "zigzag"}})
    with pytest.raises(ValueError):
        Profile.from_dict("x", {"gpu_cores": 1, "nominal_gbps": 1.0,
                                "engine": {"family": "Apple9", "lane_order": "contiguous", "cost_T": {"fp8": {"1": 1.2}}}})


def test_accelerator_fields():
    from monolith.core.profile import Profile, load_profiles

    base = {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}}
    assert Profile.from_dict("p", base).accelerator == "off" and Profile.from_dict("p", base).accelerator_min_t == {}
    on = Profile.from_dict("p", dict(base, engine=dict(base["engine"], accelerator="on", accelerator_min_t={"nvfp4": 2, "fp8": 4})))
    assert on.accelerator == "on" and on.accelerator_min_t == {"nvfp4": 2, "fp8": 4}
    with pytest.raises(ValueError):
        Profile.from_dict("p", dict(base, engine=dict(base["engine"], accelerator="maybe")))
    with pytest.raises(ValueError):
        Profile.from_dict("p", dict(base, engine=dict(base["engine"], accelerator_min_t={"nvfp4": 0})))
    m5 = load_profiles()["apple-m5-pro-20c"]
    assert m5.accelerator == "on" and m5.accelerator_min_t["nvfp4"] == 2 and m5.accelerator_min_t["bf16"] == 4 and 0.9 < m5.cost("accelerator_nvfp4", 8) < 1.2   # the writer's row (1.04 with the V3 decode)


def test_gdn_mixer_default_is_limited_to_measured_profile():
    ps = load_profiles()
    assert {name for name, p in ps.items() if p.gdn_mixer_fusion} == {"apple-m5-max-40c"}
    p = ps["apple-m5-max-40c"]
    assert p.gdn_mixer_fusion["shape"] == [5120, 16, 48, 128, 128, 4]
    assert (p.gdn_mixer_fusion["workers"], p.gdn_mixer_fusion["sgs"]) == (80, 8)
    for change in ({"workers": 0}, {"workers": 161}, {"barrier": "leader"}, {"sgs": 32}, {"shape": [5120]},
                   {"q_outer": True}, {"fp8_decode": "half"}, {"fp8_tile_block": 512}, {"direct_norm": False},
                   {"dual_permute": False}, {"perm_sgs": 32}, {"gemm_overrides": {}}, {"unknown_knob": 1}):
        doc = dict(p.raw, engine=dict(p.raw["engine"], gdn_mixer_fusion=dict(p.gdn_mixer_fusion, **change)))
        with pytest.raises(ValueError, match="gdn_mixer_fusion"):
            Profile.from_dict("invalid", doc)
    for change in ({"groups": 0}, {"groups": 241}, {"ksplit": 3}, {"ragged_teams": False}, {"tn": 16}):
        overrides = dict(p.gdn_mixer_fusion["gemm_overrides"])
        overrides["0"] = dict(overrides["0"], **change)
        doc = dict(p.raw, engine=dict(p.raw["engine"], gdn_mixer_fusion=dict(p.gdn_mixer_fusion, gemm_overrides=overrides)))
        with pytest.raises(ValueError, match="gdn_mixer_fusion"):
            Profile.from_dict("invalid", doc)
