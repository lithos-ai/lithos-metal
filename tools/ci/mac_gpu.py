#!/usr/bin/env python3
"""Run the test tiers on a bare-metal Apple silicon Mac: the self-hosted CI lane (docs/ci.md).

    python3 tools/ci/mac_gpu.py --root <dir> [--tiers contract,kernels,runtime,spec,layers,models]
                                [--shader-validation] [--report-dir <dir>]

1. A new virtual environment under ``<root>/venv`` with the checkout installed editable (this builds the Metal
   module), the oracle and serving extras and mlx-lm. The module must import (``monolith.runtime.is_available()``).
2. The pinned checkpoints under ``<root>/models`` (``tools/ci/provision.py``, every file re-hashed), exported as
   ``$MONOLITH_MODELS``, and the pinned reference implementations checked out afresh under ``<root>/references``
   (``REFERENCES``); for every tier, the n-gram block's two real-weight byte ranges fetched afresh under
   ``<root>/fixtures/ple`` (``tests/contract/fixtures/make_basalt_ple_fixture.py --full``), exported as
   ``$BASALT_PLE_FIXTURE``. The tests run as the runner's user and can write all of these, so a run trusts nothing an
   earlier run left behind except files whose hashes it checks.
3. One pytest run per tier, each writing JUnit XML. Every tier but the contract tier runs inside
   ``$LITHOS_CI_GPU_WRAPPER`` when the runner sets one: a command prefix that serializes GPU work on a shared machine.
4. The skip audit. A test may skip only for a reason ``ALLOWED_SKIPS`` lists (the chip lacks a feature, a case the
   test itself rules out). A module that did not build, a missing
   checkpoint or package, a chip without a profile: these skip the tests silently on a developer's machine and must
   fail here, or the lane passes with nothing tested.

Writes ``<report-dir>/result.json`` and, under GitHub Actions, a summary table. Exit status: 0 every tier passed;
1 a test failed, timed out or skipped for a reason not allowed; 3 the lane could not run (platform, disk, install,
download).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("contract", "kernels", "runtime", "spec", "layers", "models")
EXTRAS = "dev,oracle,serve"
# mlx-lm: tests/models/qwen3_5/test_mlx_int4.py, tests/kernels/test_modelopt_mlx.py; httpx: the serving contract tests.
EXTRA_PACKAGES = ("mlx-lm", "httpx>=0.28")
MIN_FREE_GB = 30
# Skip reasons that do not mean "nothing was tested". Each entry is a regular expression searched in the reason.
ALLOWED_SKIPS = (
    r"requires? Apple10",                                              # a feature of the chip's GPU family
    r"the scale cache needs whole lane groups",                        # a parametrized case the test rules out
    r"Resident verification prefill requires a speculative decoder recipe",
    r"Shared dense slab range; cooperative layout has separate packing tests",
    r"a tile shape the Apple9 cooperative layout does not take",       # the chip's tensor-ops register layout
    r"the tile operand recipe repacks for the Apple10 cooperative layout",
)
# Reference implementations an oracle tier imports, pinned: (name, environment variable, repository, commit).
REFERENCES = (("deepspec", "MONOLITH_DEEPSPEC", "https://github.com/deepseek-ai/DeepSpec",
               "005e03b81cec38b7da6399833d609ee89a2587f2"),)               # tests/spec/test_dspark_oracle.py
EXIT_FAIL, EXIT_COULD_NOT_RUN = 1, 3


class CouldNotRun(Exception):
    pass


def run(cmd: List[str], **kw) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode:
        raise CouldNotRun(f"{shlex.join(cmd[:4])} … exited {r.returncode}")


def fresh_dir(root: Path, rel: str) -> Path:
    """An empty directory at ``root/rel`` made of real directories only. The tests run as this user and could leave a
    symlink, a file or a directory tree on the way; a symlink is removed, never followed."""
    path = root
    for part in Path(rel).parts:
        path = path / part
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            path.unlink()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def prepare_venv(root: Path, repo: Path, python: str) -> Path:
    """A new environment every run, the checkout installed editable (which builds the Metal module in the tree): the
    last run's tests could write to the last one."""
    venv = fresh_dir(root, "venv")
    py = venv / "bin" / "python"
    run([python, "-m", "venv", str(venv)])
    run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-q", "-e", f"{repo}[{EXTRAS}]", *EXTRA_PACKAGES], cwd=repo)
    check = subprocess.run([str(py), "-c", "from monolith.runtime import is_available; assert is_available()"],
                           cwd=repo)
    if check.returncode:
        raise CouldNotRun("the Metal module did not build or does not import (monolith.runtime.is_available())")
    return py


def fetch_references(root: Path) -> Dict[str, str]:
    """Check out each pinned reference afresh under ``<root>/references`` (the last run's tests could write to the
    last checkout); returns the environment pointing at them."""
    env = {}
    for name, var, url, commit in REFERENCES:
        path = fresh_dir(root, f"references/{name}-{commit[:12]}")
        run(["git", "init", "-q", str(path)])
        run(["git", "-C", str(path), "fetch", "-q", "--depth", "1", url, commit])
        run(["git", "-C", str(path), "checkout", "-q", "FETCH_HEAD"])
        env[var] = str(path)
    return env


def fetch_ple_fixture(root: Path, py: Path, repo: Path) -> Dict[str, str]:
    """Fetch the n-gram block's two real-weight byte ranges afresh under ``<root>/fixtures/ple`` with the checkout's
    writer (the last run's tests could write to the last copy); returns the environment pointing at them."""
    path = fresh_dir(root, "fixtures/ple")
    run([str(py), str(repo / "tests/contract/fixtures/make_basalt_ple_fixture.py"), "--full", str(path)])
    return {"BASALT_PLE_FIXTURE": str(path)}


def tier_env(models: Path, references: Dict[str, str], shader_validation: bool) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env.update(MONOLITH_MODELS=str(models), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", **references)
    if shader_validation:
        env.update(MTL_SHADER_VALIDATION="1", MTL_SHADER_VALIDATION_REPORT_TO_STDERR="1")
    return env


def parse_junit(path: Path) -> dict:
    """Counts and skip reasons of one pytest JUnit file."""
    out = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0, "skips": []}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            out[key] += int(suite.get(key, 0))
        out["time"] += float(suite.get("time", 0))
        for case in suite.iter("testcase"):
            skip = case.find("skipped")
            if skip is not None:       # a module skipped at collection says why in the text, not the message
                test = f"{case.get('classname', '')}::{case.get('name', '')}"
                reason = " ".join(x.strip() for x in (skip.get("message"), skip.text) if x and x.strip())
                out["skips"].append({"test": test, "reason": reason})
    return out


def disallowed_skips(skips: List[dict], allowed=ALLOWED_SKIPS) -> List[dict]:
    return [s for s in skips if not any(re.search(p, s["reason"]) for p in allowed)]


def judge(tier: str, counts: dict, returncode: Optional[int]) -> List[str]:
    """The reasons a tier fails (empty = it passed)."""
    problems = []
    if returncode is None:
        problems.append("timed out")
    if counts["tests"] - counts["skipped"] <= 0:
        problems.append("no test ran")
    if counts["failures"] or counts["errors"]:
        problems.append(f"{counts['failures']} failed, {counts['errors']} errors")
    elif returncode not in (0, None):
        problems.append(f"pytest exited {returncode}")
    bad = disallowed_skips(counts["skips"])
    if bad:
        reasons = sorted({s["reason"][:120] for s in bad})
        problems.append(f"{len(bad)} skipped for reasons CI does not allow: " + "; ".join(reasons))
    return problems


def run_tier(tier: str, py: Path, repo: Path, env: Dict[str, str], report: Path, timeout: int) -> dict:
    junit = report / f"{tier}.xml"
    junit.unlink(missing_ok=True)
    cmd = [str(py), "-m", "pytest", "-p", "no:cacheprovider", "-rs", f"--junitxml={junit}", f"tests/{tier}"]
    wrapper = os.environ.get("LITHOS_CI_GPU_WRAPPER", "")
    if tier != "contract" and wrapper:
        cmd = shlex.split(wrapper) + cmd
    print(f"\n=== {tier}\n+ {shlex.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=repo, env=env, start_new_session=True)
    try:
        rc: Optional[int] = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        proc.wait()
        rc = None
    counts = parse_junit(junit) if junit.exists() else {"tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                                                         "time": 0.0, "skips": []}
    problems = judge(tier, counts, rc)
    return {"tier": tier, "returncode": rc, "wall_s": round(time.time() - t0, 1), **counts, "problems": problems}


def summary_markdown(result: dict) -> str:
    lines = [f"### mac-gpu on {result.get('chip', '?')}: {result['status']}", "",
             "| tier | passed | failed | skipped | time | problems |", "|---|---|---|---|---|---|"]
    for t in result["tiers"]:
        passed = t["tests"] - t["failures"] - t["errors"] - t["skipped"]
        lines.append(f"| {t['tier']} | {passed} | {t['failures'] + t['errors']} | {t['skipped']} | "
                     f"{t['wall_s']:.0f} s | {'; '.join(t['problems']) or 'ok'} |")
    if result.get("error"):
        lines += ["", f"Could not run: {result['error']}"]
    return "\n".join(lines) + "\n"


def chip_name() -> str:
    r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    return r.stdout.strip() or platform.machine()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="persistent CI directory on this machine")
    ap.add_argument("--repo", type=Path, default=ROOT, help="the checkout under test")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--shader-validation", action="store_true", help="run the Metal tiers under shader validation")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--python", default=os.environ.get("LITHOS_CI_PYTHON", sys.executable))
    ap.add_argument("--tier-timeout", type=int, default=5400, help="seconds per tier, waiting for the GPU included")
    args = ap.parse_args(argv)
    tiers = [t for t in args.tiers.split(",") if t]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        ap.error(f"unknown tiers: {', '.join(unknown)}")
    report = (args.report_dir or args.root / "reports" / time.strftime("%Y%m%dT%H%M%S")).resolve()
    report.mkdir(parents=True, exist_ok=True)
    repo = args.repo.resolve()
    result: dict = {"status": "could not run", "tiers": [], "commit": None, "chip": None}
    try:
        result["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                          text=True).stdout.strip()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise CouldNotRun("this lane runs on Apple silicon macOS only")
        result["chip"] = chip_name()
        if args.root.is_symlink():      # what the lane deletes and rebuilds under it must not land elsewhere
            raise CouldNotRun(f"{args.root} is a symlink; set LITHOS_CI_ROOT to the real directory")
        args.root.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(args.root).free / 1e9
        if free_gb < MIN_FREE_GB:
            raise CouldNotRun(f"only {free_gb:.0f} GB free under {args.root}; the lane needs {MIN_FREE_GB} GB")
        py = prepare_venv(args.root, repo, args.python)
        models, references = args.root / "models", {}
        if any(t != "contract" for t in tiers):
            # --rehash: the last run's tests could rewrite a file and its verification record alike
            run([str(py), str(repo / "tools/ci/provision.py"), "--dest", str(models), "--rehash"])
            references = fetch_references(args.root)
        references.update(fetch_ple_fixture(args.root, py, repo))     # every tier reads it, the contract tier too
        env = tier_env(models, references, args.shader_validation)
        for tier in tiers:
            result["tiers"].append(run_tier(tier, py, repo, env, report, args.tier_timeout))
        result["status"] = "failed" if any(t["problems"] for t in result["tiers"]) else "passed"
    except CouldNotRun as e:
        result["error"] = str(e)
    (report / "result.json").write_text(json.dumps(result, indent=1))
    text = summary_markdown(result)
    print("\n" + text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    return {"passed": 0, "failed": EXIT_FAIL}.get(result["status"], EXIT_COULD_NOT_RUN)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
