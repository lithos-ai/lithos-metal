#!/usr/bin/env python3
"""Speculative decoding on a prompt set (plan M6, #40): plain decode vs the round under each verify rule — the
cost-aware rule, the confident-prefix threshold and fixed verify lengths — with the accepted-length histogram per
prompt, tokens per step and ms per token. Appends one JSON row per (prompt, mode) to a results file and prints the
gate table. The prompts use the chat template of the Qwen-family checkpoints (what the public drafters were trained
on) plus two plain-text prompts; the model is whatever the checkpoint is.

    python tools/bench/spec_bench.py --model <ckpt> --pack <pack> --drafter <ckpt> --drafter-pack <pack>
        [--modes plain,cost,threshold:0.5,fixed:1,fixed:2,fixed:3,fixed:7] [-n 128] [--prompts code,math,chat,text]
        [--sts sts.json] [--out tools/bench/results/<chip>_spec.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHAT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPTS = {
    "code": [CHAT.format(q="Write a Python function that checks whether a number is prime, with a docstring."),
             CHAT.format(q="Implement binary search over a sorted list in Python and explain its complexity."),
             CHAT.format(q="Write a bash one-liner that counts the lines of every .py file under the current directory.")],
    "math": [CHAT.format(q="What is 17 × 24? Show the steps."),
             CHAT.format(q="Solve for x: 3x + 7 = 31. Explain each step."),
             CHAT.format(q="A train travels 180 km in 2.5 hours. What is its average speed in km/h and in m/s?")],
    "chat": [CHAT.format(q="Give me three tips for sleeping better."),
             CHAT.format(q="Explain what a hash map is to a beginner."),
             CHAT.format(q="Summarize the plot of Romeo and Juliet in five sentences.")],
    "text": ["Write a short story about a robot who learns to paint.",
             "The history of the Roman Empire begins"],
}


def parse_mode(mode: str):
    if mode == "plain":
        return {}
    if mode == "cost":
        return {"verify": "cost"}
    kind, _, arg = mode.partition(":")
    if kind == "threshold":
        return {"verify": "threshold", "verify_threshold": float(arg or 0.5)}
    if kind == "fixed":
        return {"verify": "fixed", "verify_length": int(arg)}
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--drafter-pack", required=True)
    ap.add_argument("--drafter-kind", default="dspark")
    ap.add_argument("--draft-gamma", type=int, default=None, help="an LM drafter's chain length outside the fixed modes (default 5); fixed:N chains N")
    ap.add_argument("--modes", default="plain,cost,threshold:0.5,fixed:1,fixed:2,fixed:3,fixed:7")
    ap.add_argument("--prompts", default="code,math,chat,text")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--sts", default=None)
    ap.add_argument("--accelerator", default=None, choices=["on", "off"], help="T > 1 GEMVs on the tensor-ops tile (default: the chip profile's)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    from monolith.generate import load_session

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    prompts = [(cat, i, p) for cat in a.prompts.split(",") for i, p in enumerate(PROMPTS[cat])]
    rows = []
    sessions = {}
    for mode in a.modes.split(","):
        opts = parse_mode(mode)
        if opts:
            dopts = None
            if a.drafter_kind == "lm":
                dopts = {"gamma": opts["verify_length"] if opts.get("verify") == "fixed" else (a.draft_gamma or 5)}
            sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1, drafter_dir=a.drafter, drafter_pack=a.drafter_pack,
                                drafter_kind=a.drafter_kind, sts_path=a.sts, accelerator=a.accelerator, drafter_options=dopts, **opts)
        else:
            sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1, accelerator=a.accelerator)
        sessions[mode] = sess
        info = sess.dev.info()
        for cat, i, prompt in prompts:
            ids = tok.encode(prompt, add_special_tokens=False).ids
            gen = sess.generate(ids, a.max_new_tokens)
            hist = {}
            for c in (gen.accepted or []):
                hist[c] = hist.get(c, 0) + 1
            row = {"chip": info.name, "date": time.strftime("%Y-%m-%d %H:%M"), "mode": mode, "accelerator": a.accelerator or "profile",
                   "prompt": f"{cat}{i}", "prompt_tokens": len(ids),
                   "new_tokens": len(gen.tokens), "steps": gen.steps, "ms_per_token": round(gen.ms_per_token, 3),
                   "tokens_per_step": round(gen.tokens_per_step, 3), "mean_accepted": round(gen.mean_accepted, 3),
                   "mean_verify_len": round(sum(gen.verify_len) / len(gen.verify_len), 3) if gen.verify_len else 0.0,
                   "hist": {str(k): v for k, v in sorted(hist.items())}, "sts": bool(a.sts)}
            rows.append(row)
            print(f"{mode:14s} {cat}{i}: {row['ms_per_token']:6.2f} ms/token  {row['tokens_per_step']:.2f} tok/step  accepted {row['mean_accepted']:.2f}  "
                  f"L {row['mean_verify_len']:.2f}  hist {row['hist']}", flush=True)
        # one session per mode keeps the packs mapped once; drop the engines before the next mode's compile
        sess.engines.clear()
    # the gate table: per mode, the token-weighted mean over the prompt set and per category
    print("\n# gate table (ms per token, token-weighted; tokens per step; mean accepted)")
    cats = sorted({r["prompt"].rstrip("0123456789") for r in rows})
    print(f"{'mode':14s} " + " ".join(f"{c:>22s}" for c in cats) + f" {'all':>22s}")
    for mode in a.modes.split(","):
        cells = []
        for c in cats + [None]:
            sel = [r for r in rows if r["mode"] == mode and (c is None or r["prompt"].rstrip("0123456789") == c)]
            toks = sum(r["new_tokens"] for r in sel)
            ms = sum(r["ms_per_token"] * r["new_tokens"] for r in sel) / toks
            tps = sum(r["tokens_per_step"] * r["new_tokens"] for r in sel) / toks
            acc = sum(r["mean_accepted"] * r["new_tokens"] for r in sel) / toks
            cells.append(f"{ms:7.2f} ms {tps:4.2f} t/s {acc:4.2f}")
        print(f"{mode:14s} " + " ".join(f"{x:>22s}" for x in cells))
    if a.out or True:
        info = next(iter(sessions.values())).dev.info()
        chip = info.name.lower().replace(" ", "-")
        out = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / f"{chip}-{info.gpu_cores}c_spec.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
