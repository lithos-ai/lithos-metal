"""The self-hosted Mac lane: the skip audit of tools/ci/mac_gpu.py and the commit statuses of tools/ci/ci_status.py."""

import types

from tools.ci import ci_status, mac_gpu


JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" failures="0"
skipped="4" tests="6" time="1.5"><testcase classname="tests.kernels.test_a" name="test_x" time="0.1"/>
<testcase classname="tests.kernels.test_a" name="test_y" time="0.1"/>
<testcase classname="tests.kernels.test_b" name="test_z"><skipped type="pytest.skip"
message="coherent task fusion requires Apple10">skip</skipped></testcase>
<testcase classname="tests.models.test_c" name="test_w"><skipped type="pytest.skip"
message="checkpoint Qwen3.5-0.8B not found under $MONOLITH_MODELS or ~/models">skip</skipped></testcase>
<testcase classname="tests.kernels.test_d" name="test_v"><skipped type="pytest.skip"
message="monolith.runtime._native is not built">skip</skipped></testcase>
<testcase classname="" name="tests.contract.test_serve"><skipped type="pytest.skip" message="collection skipped">
tests/contract/test_serve.py:7: could not import 'httpx': No module named 'httpx'</skipped></testcase>
</testsuite></testsuites>"""


def test_skip_audit_fails_a_tier_that_skipped_for_missing_infrastructure(tmp_path):
    path = tmp_path / "kernels.xml"
    path.write_text(JUNIT)
    counts = mac_gpu.parse_junit(path)
    assert (counts["tests"], counts["skipped"]) == (6, 4)
    bad = mac_gpu.disallowed_skips(counts["skips"])
    assert [s["test"] for s in bad] == ["tests.models.test_c::test_w", "tests.kernels.test_d::test_v",
                                        "::tests.contract.test_serve"]
    assert "could not import 'httpx'" in bad[2]["reason"]
    problems = mac_gpu.judge("kernels", counts, 0)
    assert len(problems) == 1 and problems[0].startswith("3 skipped for reasons CI does not allow")


def test_judge_fails_empty_failed_and_timed_out_tiers():
    empty = {"tests": 2, "failures": 0, "errors": 0, "skipped": 2, "skips": []}
    assert "no test ran" in mac_gpu.judge("runtime", empty, 0)
    failed = {"tests": 3, "failures": 1, "errors": 0, "skipped": 0, "skips": []}
    assert mac_gpu.judge("runtime", failed, 1) == ["1 failed, 0 errors"]
    ok = {"tests": 3, "failures": 0, "errors": 0, "skipped": 0, "skips": []}
    assert mac_gpu.judge("runtime", ok, 0) == []
    assert mac_gpu.judge("runtime", ok, None) == ["timed out"]


def test_commit_status_of_each_result():
    tier = {"tier": "kernels", "tests": 10, "skipped": 1, "problems": []}
    assert ci_status.status_for({"status": "passed", "tiers": [tier]}) == ("success", "9 tests passed in 1 tiers")
    bad = {**tier, "problems": ["1 failed, 0 errors"]}
    assert ci_status.status_for({"status": "failed", "tiers": [bad, bad]}) == (
        "failure", "kernels: 1 failed, 0 errors (+1 more)")
    assert ci_status.status_for({"status": "could not run", "tiers": [], "error": "disk"})[0] == "error"
    assert ci_status.status_for(None)[0] == "error"
    assert len(ci_status.status_for({"status": "failed", "tiers": [{**tier, "problems": ["x" * 300]}]})[1]) == 140


class Statuses:
    def __init__(self, statuses):
        self.statuses = statuses

    def get(self, path):
        return self.statuses


def test_commit_status_is_left_to_a_newer_run():
    gh = Statuses([{"context": "mac-gpu (m4_pro)", "created_at": "2026-10-12T04:31:22Z"},
                   {"context": "mac-gpu (m5_max_40c)", "created_at": "2026-10-12T04:00:00Z"}])
    started = "2026-10-12T04:30:00Z"
    assert ci_status.superseded(gh, "f" * 40, "mac-gpu (m4_pro)", started)
    assert not ci_status.superseded(gh, "f" * 40, "mac-gpu (m5_max_40c)", started)
    assert not ci_status.superseded(gh, "f" * 40, "mac-gpu (m3_ultra_60c)", started)


def test_fresh_dir_never_follows_or_keeps_what_a_test_left(tmp_path):
    outside = tmp_path / "home"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"precious")
    root = tmp_path / "ci"
    root.mkdir()
    (root / "references").symlink_to(outside)            # a symlink on the way
    path = mac_gpu.fresh_dir(root, "references/deepspec")
    assert path.is_dir() and not (root / "references").is_symlink() and list(path.iterdir()) == []
    (path / "stale").write_bytes(b"x")
    (root / "venv").write_bytes(b"a file where a directory belongs")
    assert list(mac_gpu.fresh_dir(root, "references/deepspec").iterdir()) == []
    assert mac_gpu.fresh_dir(root, "venv").is_dir()
    assert (outside / "keep.txt").read_bytes() == b"precious"


def test_lane_environment_names_the_ple_fixture(tmp_path, monkeypatch):
    py, log = tmp_path / "venv" / "bin" / "python", []
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(mac_gpu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mac_gpu.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mac_gpu.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10**12))
    monkeypatch.setattr(mac_gpu, "chip_name", lambda: "test")
    monkeypatch.setattr(mac_gpu, "prepare_venv", lambda root, repo, python: py)
    monkeypatch.setattr(mac_gpu, "run", lambda cmd, **kw: log.append(cmd))
    monkeypatch.setattr(mac_gpu, "run_tier", lambda tier, py, repo, env, report, timeout: log.append(env) or
                        {"tier": tier, "tests": 1, "failures": 0, "errors": 0, "skipped": 0, "wall_s": 0.0,
                         "problems": []})
    # a contract-only run: the contract tier reads the fixture too, and provisions nothing else
    assert mac_gpu.main(["--root", str(tmp_path), "--tiers", "contract", "--report-dir", str(tmp_path / "report")]) == 0
    fixture = tmp_path / "fixtures" / "ple"
    writer = str(mac_gpu.ROOT / "tests" / "contract" / "fixtures" / "make_basalt_ple_fixture.py")
    assert log[0] == [str(py), writer, "--full", str(fixture)] and fixture.is_dir()     # fetched before the tier
    assert len(log) == 2 and log[1]["BASALT_PLE_FIXTURE"] == str(fixture)
