#!/usr/bin/env python3
"""STS calibration of the confidence chain (design §5.8, #40): per-position temperatures τ_k such that
σ(logit_k / τ_k) matches the measured acceptance of draft k. Runs the speculative session with the whole block
verified over a prompt set, collects (position, confidence, accepted?) from the program's logs, fits τ_k by a
log-spaced search minimizing the binary cross-entropy, reports the calibration error before and after, and writes
``{"temperatures": [...]}`` for ``--sts`` of the generate CLI and the bench.

    python tools/bench/sts_calibrate.py --model <ckpt> --pack <pack> --drafter <ckpt> --drafter-pack <pack> [-n 128] [--out sts.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def fit_temperature(conf: np.ndarray, y: np.ndarray) -> float:
    """τ minimizing BCE(σ(logit(c) / τ), y) over a log-spaced grid (a 1-D convex problem; the grid is enough)."""
    c = np.clip(conf.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.log(c / (1 - c))
    best, best_t = None, 1.0
    for t in np.exp(np.linspace(math.log(0.2), math.log(5.0), 121)):
        p = 1 / (1 + np.exp(-z / t))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if best is None or bce < best:
            best, best_t = bce, float(t)
    return best_t


def ece(conf: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - y[m].mean())
    return float(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--drafter-pack", required=True)
    ap.add_argument("--drafter-kind", default="dspark")
    ap.add_argument("--prompts", default="code,math,chat,text")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--out", default="sts.json")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    from monolith.generate import load_session
    from spec_bench import PROMPTS

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1, drafter_dir=a.drafter, drafter_pack=a.drafter_pack,
                        drafter_kind=a.drafter_kind, verify="threshold", verify_threshold=0.0)       # the whole block: every position observed
    gamma = sess.drafter.gamma
    samples = [([], []) for _ in range(gamma)]
    for cat in a.prompts.split(","):
        for prompt in PROMPTS[cat]:
            ids = tok.encode(prompt, add_special_tokens=False).ids
            gen = sess.generate(ids, a.max_new_tokens)
            for acc, L, conf in zip(gen.accepted, gen.verify_len, gen.confidences):
                for k in range(min(L, acc + 1)):                        # positions < acc accepted, position acc rejected (if acc < L)
                    samples[k][0].append(conf[k])
                    samples[k][1].append(1.0 if k < acc else 0.0)
    temps, report = [], []
    for k in range(gamma):
        c, y = np.array(samples[k][0]), np.array(samples[k][1])
        if len(c) < 20:
            temps.append(1.0)
            report.append(f"  position {k}: {len(c)} samples (too few; τ = 1)")
            continue
        t = fit_temperature(c, y)
        cal = 1 / (1 + np.exp(-np.log(np.clip(c, 1e-6, 1 - 1e-6) / (1 - np.clip(c, 1e-6, 1 - 1e-6))) / t))
        temps.append(round(t, 4))
        report.append(f"  position {k}: {len(c):5d} samples, acceptance {y.mean():.3f}, mean confidence {c.mean():.3f} -> {cal.mean():.3f}, "
                      f"τ = {t:.3f}, ECE {ece(c, y):.3f} -> {ece(cal, y):.3f}")
    print("\n".join(report))
    with open(a.out, "w") as f:
        json.dump({"temperatures": temps, "gamma": gamma, "prompts": a.prompts, "new_tokens": a.max_new_tokens}, f, indent=1)
    print(f"wrote {a.out}: {temps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
