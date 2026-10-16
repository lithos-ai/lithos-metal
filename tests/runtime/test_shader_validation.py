"""Metal shader validation fails the test that triggers it. Validation drops an out-of-bounds access and logs it on the
command buffer without an error; the runtime raises those logs where the command buffer completes (``Queue.run``,
``Engine.run``, ``Engine.profile``). ``shader_validation_cases.py`` runs in a pytest subprocess with the variables set
before Metal initializes: its out-of-bounds cases must fail, its in-bounds ones pass and its store after the test
bodies error, in every capture mode, with the report text on stderr or only in the unified log, and the CI lane's
judge must fail the tier."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import monolith
from tools.ci import mac_gpu

ROOT = Path(monolith.__file__).resolve().parents[1]     # the checkout whose runtime this process imported
CASES = "tests/runtime/shader_validation_cases.py"


def _pytest(args, env):
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args, CASES], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("args,to_stderr", [((), True), (("-s",), True), ((), False)],
                         ids=["captured", "no-capture", "report-in-unified-log"])
def test_validation_report_fails_the_test(args, to_stderr):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_")}
    env["MTL_SHADER_VALIDATION"] = "1"
    if to_stderr:
        env["MTL_SHADER_VALIDATION_REPORT_TO_STDERR"] = "1"
    p = _pytest(args, env)
    out = p.stdout + p.stderr
    assert p.returncode == 1 and re.search(r"\b3 failed, 4 passed, 1 error\b", out), out
    failed = re.findall(r"^FAILED \S+::test_store\[(\w+)-(\w+)\]", out, re.M)
    assert sorted(failed) == [(path, "out_of_bounds") for path in ("engine", "profile", "queue")], out
    assert re.findall(r"^ERROR \S+::(\w+)", out, re.M) == ["test_outside_the_test_body"], out
    assert out.count("shader validation: Invalid store") >= 4, out


def test_ci_lane_fails_the_tier(tmp_path):
    """The lane's own environment (``--shader-validation``), command line and judge."""
    junit = tmp_path / "runtime.xml"
    p = _pytest(["-rs", f"--junitxml={junit}"], mac_gpu.tier_env(tmp_path, {}, shader_validation=True))
    counts = mac_gpu.parse_junit(junit)
    assert counts["tests"] == 7 and mac_gpu.judge("runtime", counts, p.returncode) == ["3 failed, 1 errors"], p.stdout
    assert junit.read_text().count("shader validation: Invalid store") >= 4
