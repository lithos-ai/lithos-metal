#!/usr/bin/env python3
"""The extension test (design D16): a model PR may change only ``monolith/models/``, ``tests/`` and ``docs/``.

    python tools/ci/extension_check.py --base origin/main [--allow prefix ...]

Lists the files changed between ``base`` and ``HEAD`` and fails if any lies outside the allowed prefixes. A few
housekeeping files are always allowed (``third_party/NOTICE``, ``CHANGELOG.md``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable, List, Sequence

DEFAULT_ALLOW = ("monolith/models/", "tests/", "docs/")
ALWAYS_ALLOW = ("third_party/NOTICE", "CHANGELOG.md")


def check(files: Iterable[str], allow: Sequence[str] = DEFAULT_ALLOW) -> List[str]:
    """Return the changed paths that are not allowed (empty = pass)."""
    bad = []
    for f in files:
        f = f.strip()
        if not f or f in ALWAYS_ALLOW:
            continue
        if not any(f.startswith(a) for a in allow):
            bad.append(f)
    return bad


def changed_files(base: str) -> List[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l.strip()]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base ref, e.g. origin/main")
    ap.add_argument("--allow", nargs="*", default=list(DEFAULT_ALLOW))
    a = ap.parse_args(argv)
    bad = check(changed_files(a.base), a.allow)
    if bad:
        print("extension check: FAIL — a model PR may only touch " + ", ".join(a.allow))
        for f in bad:
            print("  " + f)
        return 1
    print("extension check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
