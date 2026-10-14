"""Alternating lithos-metal / mlx-lm / ollama plain-decode bench. usage: bench.py --ckpt --pack [--mlx PATH] [--ollama TAG] [-n 128] [--reps 5]"""
import argparse, json, re, subprocess, sys, time, urllib.request, random
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--pack", required=True)
ap.add_argument("--mlx"); ap.add_argument("--ollama"); ap.add_argument("-n", type=int, default=128)
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--lens", default="5,200,1000")
ap.add_argument("--out"); a = ap.parse_args()
BASE = "The history of computing begins with simple tools for counting and calculation, and moves through mechanical engines, vacuum tubes, transistors and integrated circuits. "
def prompt(n_tok):
    if n_tok <= 8: return "The capital of France is"
    return (BASE * (n_tok // 30 + 1))[: n_tok * 5]  # ~5 chars/token; exact count taken from the engines
lens = [int(x) for x in a.lens.split(",")]
if a.mlx:
    from mlx_lm import load, stream_generate
    mm, mt = load(a.mlx)
def run_lithos(p):
    r = subprocess.run([sys.executable, "-m", "monolith.generate", "--model", a.ckpt, "--pack", a.pack, "--prompt", p,
        "-n", str(a.n), "--no-eos", "--max-context", "2048"], capture_output=True, text=True, cwd="/Users/gr/_/Temp/6/lithos-metal")
    o = r.stdout + r.stderr
    pm = re.search(r"prefill ([\d.]+) ms \((\d+) prompt tokens\)", o); wm = re.search(r"wall ([\d.]+) ms/token", o)
    if not (pm and wm): raise RuntimeError(o[-500:] + r.stderr[-500:])
    pt = int(pm[2]); return dict(ptoks=pt, prefill_tps=pt / float(pm[1]) * 1000, decode_tps=1000 / float(wm[1]))
def run_mlx(p):
    last = None
    for last in stream_generate(mm, mt, prompt=p, max_tokens=a.n): pass
    return dict(ptoks=last.prompt_tokens, prefill_tps=last.prompt_tps, decode_tps=last.generation_tps)
def run_ollama(p):
    body = json.dumps(dict(model=a.ollama, prompt=f"[{random.random()}] " + p, stream=False, think=False, keep_alive="10m",
        options=dict(num_predict=a.n, temperature=0, num_ctx=4096))).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", body), timeout=600))
    return dict(ptoks=d["prompt_eval_count"], prefill_tps=d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9,
                decode_tps=d["eval_count"] / d["eval_duration"] * 1e9)
engines = {"lithos": run_lithos}
if a.mlx: engines["mlx-lm"] = run_mlx
if a.ollama: engines["ollama"] = run_ollama
res = {}
for L in lens:
    p = prompt(L)
    for rep in range(a.reps + 1):               # rep 0 = warm-up, discarded
        for name, fn in engines.items():
            r = fn(p)
            if rep: res.setdefault((L, name), []).append(r)
            time.sleep(3)                       # let the clocks settle between engines
rows = []
print(f"{'len':>5} {'engine':8} {'ptoks':>6} {'prefill tok/s (min-max)':>26} {'decode tok/s (min-max)':>26}")
for (L, name), rs in res.items():
    pf = [r["prefill_tps"] for r in rs]; dc = [r["decode_tps"] for r in rs]
    rows.append(dict(len=L, engine=name, ptoks=rs[0]["ptoks"], prefill=pf, decode=dc))
    print(f"{L:>5} {name:8} {rs[0]['ptoks']:>6} {max(pf):>12.0f} ({min(pf):.0f}-{max(pf):.0f}) {max(dc):>12.1f} ({min(dc):.1f}-{max(dc):.1f})")
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
