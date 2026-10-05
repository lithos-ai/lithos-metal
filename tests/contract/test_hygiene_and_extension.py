import subprocess
import sys
from pathlib import Path

from tools.ci import extension_check, hygiene

ROOT = Path(__file__).resolve().parents[2]


def test_repo_passes_hygiene():
    imports, idents, models = hygiene.run(ROOT)
    assert imports == [] and idents == [] and models == []


def test_hygiene_catches_violations(tmp_path):
    (tmp_path / "monolith").mkdir()
    (tmp_path / "monolith" / "bad.py").write_text(
        "import mirage\nfrom mirage import x\nmpk_thing = 1\nNAME = 'Qwen3_5ForConditionalGeneration'\n")
    imports, idents, models = hygiene.run(tmp_path)
    assert len(imports) == 2 and any("mpk_thing" in s for s in idents) and any("qwen" in s for s in models)


def test_extension_check_allows_only_model_dirs():
    assert extension_check.check(["monolith/models/qwen3_5/model.py", "tests/models/test_x.py", "docs/porting.md",
                                  "third_party/NOTICE"]) == []
    assert extension_check.check(["monolith/compiler/passes/fuse.py", "kernels/gemv.metal"]) == [
        "monolith/compiler/passes/fuse.py", "kernels/gemv.metal"]


def test_hygiene_cli_runs():
    r = subprocess.run([sys.executable, str(ROOT / "tools/ci/hygiene.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
