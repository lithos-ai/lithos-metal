#!/usr/bin/env python3
"""The speculative gate against mlx-lm (#103): per-token latency of our DSpark round vs mlx-lm's speculative decoding
(a draft model, ``num_draft_tokens`` N) on the same target checkpoint bytes (the MLX NVFP4 conversion, read by both),
the same prompt set (spec_bench's), greedy, N new tokens, paired alternating runs, the best of ``--reps`` per prompt.
"Same configuration" = the same number of drafted tokens per step: ours at fixed L = N (and the cost-aware rule), mlx-lm
at N; both engines' plain decode in the same run as the floor. Appends JSON rows; prints the gate table.

    python tools/bench/spec_vs_mlx.py --model <mlx nvfp4 ckpt> --pack <pack> --drafter <dspark ckpt> --drafter-pack <pack>
        --mlx-draft <mlx draft ckpt> [--ns 1,2,3,4,5,7] [-n 128] [--reps 2] [--prompts code,math,chat,text] [--sts sts.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_bench import PROMPTS  # noqa: E402


def mlx_run(model, tokenizer, ids, n, draft=None, num_draft=0):
    """(tokens, wall tok/s, tokens per step): mlx-lm's stream; with a draft model a step yields the accepted drafts plus
    one target token, ``from_draft`` marking the drafted ones."""
    from mlx_lm import stream_generate

    toks, last, steps = [], None, 0
    kw = {"draft_model": draft, "num_draft_tokens": num_draft} if draft is not None else {}
    for r in stream_generate(model, tokenizer, prompt=list(ids), max_tokens=n, **kw):
        toks.append(int(r.token))
        steps += 0 if getattr(r, "from_draft", False) else 1
        last = r
    return toks, float(last.generation_tps), (len(toks) / max(1, steps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--drafter-pack", required=True)
    ap.add_argument("--drafter-kind", default="dspark")
    ap.add_argument("--draft-gamma", type=int, default=None, help="an LM drafter's chain length in the cost mode (default 5); a fixed:N mode chains N")
    ap.add_argument("--mlx-draft", default=None, help="mlx-lm's draft model (a small same-tokenizer LM); without it only plain mlx-lm runs")
    ap.add_argument("--ns", default="1,2,3,4,5,7")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompts", default="code,math,chat,text")
    ap.add_argument("--max-context", type=int, default=1024, help="the sessions' context (one per mode stays resident beside mlx-lm's models: 1024 keeps the KV caches small)")
    ap.add_argument("--sts", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    import mlx.core as mx
    from mlx_lm import load as mlx_load

    from monolith.generate import load_session

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    prompts = [(cat, i, p) for cat in a.prompts.split(",") for i, p in enumerate(PROMPTS[cat])]
    ns = [int(x) for x in a.ns.split(",")]
    mlx_model, mlx_tok = mlx_load(a.model)
    mlx_draft = mlx_load(a.mlx_draft)[0] if a.mlx_draft else None
    # one of our sessions is alive at a time (its packs and caches beside mlx-lm's models: nine at once exhausted the
    # GPU working set on a 24 GB machine and mlx-lm's command buffers timed out); the modes alternate, a rep runs them
    # in reverse, the best of the reps per (mode, prompt) is kept
    our_modes = ["plain"] + [f"fixed:{n_}" for n_ in ns] + ["cost"]
    mlx_modes = ["plain"] + ([f"draft:{n_}" for n_ in ns] if mlx_draft is not None else [])
    modes = [("ours", m) for m in our_modes] + [("mlx", m) for m in mlx_modes]
    plain = load_session(a.model, a.pack, max_context=a.max_context, eos=-1)
    info = plain.dev.info()
    plain.engines.clear()

    def our_session(mode):
        if mode == "plain":
            return load_session(a.model, a.pack, max_context=a.max_context, eos=-1)
        opts = {"verify": "cost"} if mode == "cost" else {"verify": "fixed", "verify_length": int(mode.split(":")[1])}
        dopts = None
        if a.drafter_kind == "lm":                                                 # an LM drafter chains exactly what the mode verifies
            dopts = {"gamma": int(mode.split(":")[1]) if mode != "cost" else (a.draft_gamma or 5)}
        return load_session(a.model, a.pack, max_context=a.max_context, eos=-1, drafter_dir=a.drafter, drafter_pack=a.drafter_pack,
                            drafter_kind=a.drafter_kind, sts_path=a.sts, drafter_options=dopts, **opts)

    best = {}
    ids0 = tok.encode(prompts[0][2], add_special_tokens=False).ids
    out = a.out or str(Path(__file__).parent / "results" / f"{info.name.lower().replace(' ', '-')}-{info.gpu_cores}c_spec_vs_mlx.jsonl")
    print(f"{info.name}: ours ({a.drafter_kind} drafter) vs mlx-lm ({'draft ' + a.mlx_draft if mlx_draft is not None else 'plain only'}), {len(prompts)} prompts × {a.reps} reps, {a.max_new_tokens} tokens")
    for rep in range(a.reps):
        for eng, mode in (modes if rep % 2 == 0 else modes[::-1]):
            sess = our_session(mode) if eng == "ours" else None
            if sess is not None:
                sess.generate(ids0, 8)                                             # compile, tune, warm
            elif mode == "plain":
                mlx_run(mlx_model, mlx_tok, ids0, 8)
            else:
                mlx_run(mlx_model, mlx_tok, ids0, 8, mlx_draft, 2)
            for cat, i, prompt in prompts:
                ids = tok.encode(prompt, add_special_tokens=False).ids
                if sess is not None:
                    g = sess.generate(ids, a.max_new_tokens)
                    rec = {"engine": "ours", "mode": mode, "ms_per_token": g.decode_wall_ms / max(1, g.decode_tokens), "gpu_ms_per_token": g.ms_per_token,
                           "tokens_per_step": g.tokens_per_step, "accepted": g.mean_accepted, "tokens": len(g.tokens)}
                else:
                    n_ = int(mode.split(":")[1]) if ":" in mode else 0
                    t, tps, tps_step = mlx_run(mlx_model, mlx_tok, ids, a.max_new_tokens, mlx_draft if n_ else None, n_)
                    mx.clear_cache()
                    rec = {"engine": "mlx", "mode": mode, "ms_per_token": 1e3 / tps, "gpu_ms_per_token": 0.0, "tokens_per_step": tps_step,
                           "accepted": max(0.0, tps_step - 1), "tokens": len(t)}
                key = (eng, mode, f"{cat}{i}")
                if key not in best or rec["ms_per_token"] < best[key]["ms_per_token"]:
                    best[key] = dict(rec, prompt_tokens=len(ids))
                print(f"  rep {rep} {eng}/{mode:8s} {cat}{i}: {rec['ms_per_token']:6.2f} ms/token  {rec['tokens_per_step']:.2f} tok/step", flush=True)
            if sess is not None:
                sess.engines.clear()
                del sess
    rows = []
    for (eng, mode, prompt), r in best.items():
        rows.append({"chip": info.name, "date": time.strftime("%Y-%m-%d %H:%M"), "prompt": prompt, "new_tokens": a.max_new_tokens, "reps": a.reps,
                     "mlx_draft": a.mlx_draft, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # the gate table: token-weighted ms per token per (engine, mode), overall and per category
    def mean_ms(sel):
        return sum(r["ms_per_token"] * r["new_tokens"] for r in sel) / max(1, sum(r["new_tokens"] for r in sel))
    cats = sorted({r["prompt"].rstrip("0123456789") for r in rows})
    modes = sorted({(r["engine"], r["mode"]) for r in rows}, key=lambda m: (m[0] != "ours", m[1]))
    print("\n| engine / mode | " + " | ".join(["all"] + cats) + " |\n|---|" + "---|" * (len(cats) + 1))
    for eng, mode in modes:
        cells = []
        for c in [None] + cats:
            sel = [r for r in rows if r["engine"] == eng and r["mode"] == mode and (c is None or r["prompt"].rstrip("0123456789") == c)]
            cells.append(f"{mean_ms(sel):.2f} ms ({sum(r['tokens_per_step'] * r['new_tokens'] for r in sel) / max(1, sum(r['new_tokens'] for r in sel)):.2f} t/s)")
        print(f"| {eng}/{mode} | " + " | ".join(cells) + " |")
    best_ours = min(mean_ms([r for r in rows if r["engine"] == "ours" and r["mode"] == m]) for _, m in modes if _ == "ours")
    best_mlx = min(mean_ms([r for r in rows if r["engine"] == "mlx" and r["mode"] == m]) for e, m in modes if e == "mlx")
    print(f"\nbest ours {best_ours:.2f} ms per token vs best mlx-lm {best_mlx:.2f}: {'GATE MET' if best_ours <= best_mlx else 'not yet'} (ratio {best_ours / best_mlx:.3f}); rows appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
