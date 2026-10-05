#!/usr/bin/env python3
"""Plain-decode baseline on this machine (plan M5's go/no-go #2, #36): our engine against mlx-lm on the same model
(the NVFP4 checkpoint and its MLX conversion), the same prompt set (spec_bench's), greedy, N new tokens, paired
alternating runs with min-of-N per prompt. Both engines are timed by wall clock over the decode phase (mlx-lm's
``generation_tps``; ours ``decode_wall_ms``), ours also by GPU time; GB/s counts the pack's slab bytes per token
(the same 4.5 bits per weight in both engines) against the profile's nominal bandwidth. Appends JSON rows and prints
the gate table: the token-weighted mean over the prompt set per engine and the ratio ours / mlx-lm.

    python tools/bench/plain_baseline.py --model <ckpt> --pack <pack> --mlx-model <mlx ckpt> [-n 128] [--reps 3]
        [--prompts code,math,chat,text] [--out tools/bench/results/<chip>_plain_baseline.jsonl]
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


def mlx_generate(model, tokenizer, ids, n):
    """(tokens, wall tok/s) of one greedy mlx-lm generation over the prompt ids (the stream's last response carries the
    generation rate over the whole decode)."""
    from mlx_lm import stream_generate

    toks, last = [], None
    for r in stream_generate(model, tokenizer, prompt=list(ids), max_tokens=n):
        toks.append(int(r.token))
        last = r
    return toks, float(last.generation_tps), int(last.generation_tokens)


def mlx_bytes_per_token(path: str) -> int:
    """The MLX checkpoint's weight bytes a decode step streams (every tensor but the embedding table)."""
    import json
    from pathlib import Path as _P

    total = 0
    idx = _P(path) / "model.safetensors.index.json"
    files = sorted({v for v in json.load(open(idx))["weight_map"].values()}) if idx.exists() else ["model.safetensors"]
    for fn in files:
        from safetensors import safe_open

        with safe_open(str(_P(path) / fn), "np") as f:
            for k in f.keys():
                if "embed_tokens" in k:
                    continue
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                total += n * {"U32": 4, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1}.get(sl.get_dtype(), 2)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--mlx-model", required=True)
    ap.add_argument("-n", "--max-new-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompts", default="code,math,chat,text")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    import mlx.core as mx
    from mlx_lm import load as mlx_load

    from monolith.generate import load_session
    from monolith.packs import PackFile

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1)
    info = sess.dev.info()
    # bytes a decode step streams: every slab but the embedding table (one row of it per token)
    bytes_per_token = sum(int(s["nbytes"]) for s in PackFile(a.pack).manifest["slabs"] if "embed_tokens" not in s["name"])
    nominal = sess.profile.nominal_gbps
    mlx_bytes = mlx_bytes_per_token(a.mlx_model)
    mlx_model, mlx_tok = mlx_load(a.mlx_model)
    prompts = [(cat, i, p) for cat in a.prompts.split(",") for i, p in enumerate(PROMPTS[cat])]
    # warm both engines once (pipeline compilation, MLX's graph)
    ids0 = tok.encode(prompts[0][2], add_special_tokens=False).ids
    sess.generate(ids0, 8)
    mlx_generate(mlx_model, mlx_tok, ids0, 8)
    mx.clear_cache()
    rows = []
    out = a.out or str(Path(__file__).parent / "results" / f"{info.name.lower().replace(' ', '-')}-{info.gpu_cores}c_plain_baseline.jsonl")
    print(f"{info.name}: ours vs mlx-lm, {len(prompts)} prompts × {a.reps} paired reps, {a.max_new_tokens} new tokens; weights streamed per token: "
          f"ours {bytes_per_token / 1e9:.2f} GB, mlx-lm {mlx_bytes / 1e9:.2f} GB")
    for cat, i, prompt in prompts:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        best = {"ours": None, "mlx": None}
        toks = {}
        for rep in range(a.reps):
            order = ("ours", "mlx") if rep % 2 == 0 else ("mlx", "ours")
            for who in order:
                if who == "ours":
                    gen = sess.generate(ids, a.max_new_tokens)
                    tps = gen.decode_tokens / (gen.decode_wall_ms / 1e3)
                    rec = {"tps": tps, "gpu_ms_per_token": gen.ms_per_token, "wall_ms_per_token": gen.decode_wall_ms / max(1, gen.decode_tokens),
                           "host_pct": 100 * gen.host_busy_ms / max(gen.decode_wall_ms, 1e-9)}
                    toks["ours"] = gen.tokens
                else:
                    t, tps, n_gen = mlx_generate(mlx_model, mlx_tok, ids, a.max_new_tokens)
                    rec = {"tps": tps, "wall_ms_per_token": 1e3 / tps, "tokens": n_gen}
                    toks["mlx"] = t
                if best[who] is None or rec["tps"] > best[who]["tps"]:
                    best[who] = rec
        agree = 0
        for x, y in zip(toks["ours"], toks["mlx"]):
            if x != y:
                break
            agree += 1
        for who in ("ours", "mlx"):
            r = best[who]
            gbps = (bytes_per_token if who == "ours" else mlx_bytes) * r["tps"] / 1e9
            row = {"chip": info.name, "date": time.strftime("%Y-%m-%d %H:%M"), "engine": who, "model": a.model if who == "ours" else a.mlx_model, "prompt": f"{cat}{i}", "prompt_tokens": len(ids),
                   "bytes_per_token": bytes_per_token if who == "ours" else mlx_bytes,
                   "new_tokens": a.max_new_tokens, "tps": round(r["tps"], 2), "wall_ms_per_token": round(r["wall_ms_per_token"], 3),
                   "gpu_ms_per_token": round(r.get("gpu_ms_per_token", 0.0), 3), "gbps": round(gbps, 1), "pct_nominal": round(100 * gbps / nominal, 1),
                   "host_pct": round(r.get("host_pct", 0.0), 2), "agree_prefix": agree, "reps": a.reps}
            rows.append(row)
        o, m = best["ours"], best["mlx"]
        print(f"{cat}{i:<2} ours {o['tps']:6.2f} tok/s ({o['gpu_ms_per_token']:.2f} ms GPU, {o['wall_ms_per_token']:.2f} ms wall)   "
              f"mlx-lm {m['tps']:6.2f} tok/s ({m['wall_ms_per_token']:.2f} ms)   ratio {o['tps'] / m['tps']:.3f}   agree {agree}/{a.max_new_tokens}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # the gate table: token-weighted mean tok/s per engine, overall and per category
    def mean_tps(sel):
        return sum(r["tps"] * r["new_tokens"] for r in sel) / max(1, sum(r["new_tokens"] for r in sel))
    cats = sorted({r["prompt"].rstrip("0123456789") for r in rows})
    print("\n| set | ours tok/s | mlx-lm tok/s | ratio | ours GB/s (% of nominal) | mlx-lm GB/s (% of nominal) |\n|---|---|---|---|---|---|")
    for c in [None] + cats:
        ours = [r for r in rows if r["engine"] == "ours" and (c is None or r["prompt"].rstrip("0123456789") == c)]
        mlx = [r for r in rows if r["engine"] == "mlx" and (c is None or r["prompt"].rstrip("0123456789") == c)]
        to, tm = mean_tps(ours), mean_tps(mlx)
        gb, gm = bytes_per_token * to / 1e9, mlx_bytes * tm / 1e9
        print(f"| {c or 'all'} | {to:.2f} | {tm:.2f} | {to / tm:.3f} | {gb:.0f} ({100 * gb / nominal:.0f} %) | {gm:.0f} ({100 * gm / nominal:.0f} %) |")
    ratio = mean_tps([r for r in rows if r["engine"] == "ours"]) / mean_tps([r for r in rows if r["engine"] == "mlx"])
    verdict = "go (>= 1.10x)" if ratio >= 1.10 else ("parity (within +-5 %)" if abs(ratio - 1) <= 0.05 else ("ahead but short of 1.10x" if ratio > 1 else "behind"))
    print(f"\nours / mlx-lm = {ratio:.3f}: {verdict}; rows appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
