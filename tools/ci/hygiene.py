#!/usr/bin/env python3
"""Standalone-repo and model-agnostic hygiene (design D15, D16). Exit 1 with a report on any violation.

1. No code file imports ``mirage`` (``import mirage`` / ``from mirage``).
2. No identifier in ``monolith/`` contains ``mpk`` or ``mirage`` (comments, docstrings and provenance lines may name
   them; identifiers may not).
3. No model name (a token list below) appears in ``monolith/`` outside ``monolith/models/`` — in identifiers *or*
   string literals, so weight-name maps cannot leak into the engine either.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tokenize
from pathlib import Path
from typing import Iterable, List, Tuple

CODE_SUFFIXES = {".py", ".mm", ".m", ".cpp", ".cc", ".h", ".hpp", ".metal", ".txt", ".toml", ".cmake"}
IMPORT_RE = re.compile(r"^\s*(import\s+mirage\b|from\s+mirage\b)", re.MULTILINE)
FORBIDDEN_IDENT = re.compile(r"(mpk|mirage)", re.IGNORECASE)
MODEL_TOKENS = ("qwen", "llama", "deepseek", "gemma", "kimi", "mistral", "gpt", "phi", "yi_", "glm", "minimax")
SKIP_DIRS = {".git", ".venv", "build", "node_modules", "__pycache__", ".pytest_cache", "third_party"}


def iter_files(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in suffixes or fn == "CMakeLists.txt":
                yield p


def check_no_mirage_import(root: Path) -> List[str]:
    out = []
    for p in iter_files(root, CODE_SUFFIXES):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for m in IMPORT_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append(f"{p.relative_to(root)}:{line}: imports mirage")
    return out


def _py_tokens(p: Path) -> Iterable[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(p.read_text()).readline))
    except (tokenize.TokenError, SyntaxError, OSError):
        return []


def check_identifiers(root: Path) -> List[str]:
    out = []
    for p in iter_files(root / "monolith", {".py"}):
        for tok in _py_tokens(p):
            if tok.type == tokenize.NAME and FORBIDDEN_IDENT.search(tok.string):
                out.append(f"{p.relative_to(root)}:{tok.start[0]}: identifier {tok.string!r} names MPK/mirage")
    return out


def check_model_names(root: Path) -> List[str]:
    out = []
    for p in iter_files(root / "monolith", {".py"}):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("monolith/models/"):
            continue
        for tok in _py_tokens(p):
            if tok.type not in (tokenize.NAME, tokenize.STRING):
                continue
            if tok.type == tokenize.STRING and (tok.string.endswith('"""') or tok.string.endswith("'''")):
                continue                      # docstrings may cite checkpoints and other engines; code may not
            low = tok.string.lower()
            hit = next((t for t in MODEL_TOKENS if t in low), None)
            if hit:
                out.append(f"{rel}:{tok.start[0]}: {tok.string[:40]!r} names a model ({hit}) outside monolith/models/")
    return out


def run(root: Path) -> Tuple[List[str], List[str], List[str]]:
    return check_no_mirage_import(root), check_identifiers(root), check_model_names(root)


def main(argv: List[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[2]
    problems = [x for group in run(root) for x in group]
    if problems:
        print("hygiene: FAIL")
        for line in problems:
            print("  " + line)
        return 1
    print("hygiene: ok (no mirage imports, no MPK/mirage identifiers, no model names outside monolith/models/)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
