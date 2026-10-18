"""Metal shader validation fails the test that triggers it. Validation drops an out-of-bounds access and logs it on the
command buffer without an error; the runtime raises those logs where the command buffer completes (``Queue.run``,
``Engine.run``, ``Engine.profile``), and ``tests/conftest.py`` fails a test phase whose reports were caught or raised on
another thread. ``shader_validation_cases.py`` runs in a pytest subprocess with the variables set before Metal
initializes: its out-of-bounds cases must fail, its in-bounds ones pass and its store after the test bodies error, in
every capture mode, with the report text on stderr or only in the unified log, and the CI lane's judge must fail the
tier."""
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import monolith
from tools.ci import mac_gpu

ROOT = Path(monolith.__file__).resolve().parents[1]     # the checkout whose runtime this process imported
CASES = "tests/runtime/shader_validation_cases.py"
# How each out-of-bounds case fails: the runtime raises the report where the command buffer completes, or
# tests/conftest.py fails the phase whose report nothing raised (caught, skipped, raised on a thread).
REASONS = {"queue": "RuntimeError", "engine": "RuntimeError", "host_steps": "RuntimeError", "profile": "RuntimeError",
           "caught": "Failed", "skipped": "Failed", "thread": "Failed"}


def _pytest(args, env):
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args, CASES], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("args,to_stderr", [((), True), (("-s",), True), ((), False)],
                         ids=["captured", "no-capture", "report-in-unified-log"])
def test_validation_report_fails_the_test(args, to_stderr):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_") and k != "PYTEST_ADDOPTS"}   # its summary line
    env["MTL_SHADER_VALIDATION"] = "1"
    env["COLUMNS"] = "1000"                                 # the short summary keeps each failure's exception type
    if to_stderr:
        env["MTL_SHADER_VALIDATION_REPORT_TO_STDERR"] = "1"
    p = _pytest(args, env)
    out = p.stdout + p.stderr
    assert p.returncode == 1 and re.search(r"\b7 failed, 8 passed, (\d+ warnings?, )?1 error\b", out), out   # the thread's warning
    reasons = dict(re.findall(r"^FAILED \S+::test_store\[(\w+)-out_of_bounds\] - (\w+)", out, re.M))
    assert reasons == REASONS, out
    assert re.findall(r"^ERROR \S+::(\w+)", out, re.M) == ["test_outside_the_test_body"], out
    assert out.count("shader validation: Invalid store") >= 4, out
    assert out.count("in this call phase that nothing raised") >= 3, out     # caught, skipped, thread: tests/conftest.py


def test_ci_lane_fails_the_tier(tmp_path):
    """The lane's own environment (``--shader-validation``), command line and judge."""
    junit = tmp_path / "runtime.xml"
    p = _pytest(["-rs", f"--junitxml={junit}"], mac_gpu.tier_env(tmp_path, {}, shader_validation=True))
    counts = mac_gpu.parse_junit(junit)
    assert counts["tests"] == 15 and mac_gpu.judge("runtime", counts, p.returncode) == ["7 failed, 1 errors"], p.stdout
    assert junit.read_text().count("shader validation: Invalid store") >= 4
    reasons = {re.search(r"\[(\w+)-out_of_bounds\]", case.get("name")).group(1): failure.get("message").split(":")[0]
               for case in ET.parse(junit).iter("testcase") if (failure := case.find("failure")) is not None}
    assert reasons == REASONS, junit.read_text()
