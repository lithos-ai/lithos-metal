"""Per-op GPU timings of a program (plan M5, #33): the budget table a token is made of, and a Chrome-trace file
(open in Perfetto or chrome://tracing) — no viewer of our own is needed.

    python -m monolith.trace --model ~/models/<ckpt> --pack <pack dir> [--steps 5] [--trace step.trace.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .runtime.program import Program


@dataclass
class OpTiming:
    index: int
    name: str
    kernel: str
    kind: str
    ms_min: float
    ms_median: float
    bytes: int

    @property
    def gbps(self) -> Optional[float]:
        return self.bytes / 1e9 / (self.ms_min / 1e3) if self.bytes and self.ms_min > 0 else None


def op_timings(program: Program, runs: Sequence[Sequence[Tuple[float, float]]]) -> List[OpTiming]:
    """One timing per op; the predicated per-T variants of a GEMV (consecutive ops of one name with ``t_range``)
    count as one op whose per-step time is the sum over its variants (one runs, the others return at once) and whose
    bytes are counted once."""
    out = []
    i = 0
    n = len(program.ops)
    while i < n:
        op = program.ops[i]
        j = i + 1
        if op.meta.get("t_range") is not None:
            while j < n and program.ops[j].name == op.name and program.ops[j].meta.get("t_range") is not None:
                j += 1
        durs = []
        for run in runs:
            d = [max(0.0, e - s) for (s, e) in run[i:j] if s >= 0]
            if d:
                durs.append(sum(d))
        if durs:
            durs.sort()
            out.append(OpTiming(i, op.name, op.kernel.split("|")[0], str(op.meta.get("kind", op.name.split(":")[0])),
                                durs[0], durs[len(durs) // 2], int(op.meta.get("bytes", 0))))
        i = j
    return out


def budget(timings: Sequence[OpTiming]) -> List[Tuple[str, int, float, int]]:
    """Per kind: (kind, count, ms_min summed, bytes summed), largest first."""
    acc: Dict[str, List] = defaultdict(lambda: [0, 0.0, 0])
    for t in timings:
        a = acc[t.kind]
        a[0] += 1
        a[1] += t.ms_min
        a[2] += t.bytes
    rows = [(k, v[0], v[1], v[2]) for k, v in acc.items()]
    return sorted(rows, key=lambda r: -r[2])


def format_budget(timings: Sequence[OpTiming], nominal_gbps: Optional[float] = None) -> str:
    total = sum(t.ms_min for t in timings)
    lines = [f"{'op kind':18s} {'n':>4s} {'ms (min)':>9s} {'share':>6s} {'GB streamed':>12s} {'GB/s':>7s}"]
    for kind, n, ms, nbytes in budget(timings):
        gbps = f"{nbytes / 1e9 / (ms / 1e3):7.1f}" if nbytes and ms > 0 else "      -"
        lines.append(f"{kind:18s} {n:4d} {ms:9.3f} {100 * ms / total:5.1f}% {nbytes / 1e9:12.3f} {gbps}")
    tb = sum(t.bytes for t in timings)
    bound = f"; bandwidth bound of the streamed bytes at {nominal_gbps:.0f} GB/s: {tb / 1e9 / nominal_gbps * 1e3:.2f} ms" if nominal_gbps and tb else ""
    lines.append(f"{'total':18s} {len(timings):4d} {total:9.3f} (sum of per-op minima; encoder gaps excluded){bound}")
    return "\n".join(lines)


def write_chrome_trace(path: str, program: Program, runs: Sequence[Sequence[Tuple[float, float]]]) -> None:
    events = []
    t_off = 0.0
    for r, run in enumerate(runs):
        for i, (s, e) in enumerate(run):
            if s < 0:
                continue
            op = program.ops[i]
            events.append({"name": op.name, "cat": str(op.meta.get("kind", "")), "ph": "X", "ts": (t_off + s) * 1e3,
                           "dur": max(0.0, e - s) * 1e3, "pid": 1, "tid": 1, "args": {"kernel": op.kernel, "step": r, **op.meta}})
        t_off += max((e for _, e in run), default=0.0) + 0.05
    with open(path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--no-autotune", action="store_true")
    ap.add_argument("--drafter", default=None, help="profile the speculative round with this drafter checkpoint")
    ap.add_argument("--drafter-pack", default=None)
    ap.add_argument("--drafter-kind", default="dspark")
    ap.add_argument("--draft-gamma", type=int, default=None, help="an LM drafter's drafts per round (--drafter-kind lm)")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer

    from .generate import load_session

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1, autotune=not a.no_autotune, drafter_dir=a.drafter,
                        drafter_pack=a.drafter_pack, drafter_kind=a.drafter_kind,
                        drafter_options={"gamma": a.draft_gamma} if a.draft_gamma is not None else None)
    sess.generate(ids, 4)                                  # prefill + a few decode steps so the states are real
    dec = sess.engine(0 if sess.drafter is not None else 1)
    layout = dec.program.layout                            # the request is served (stop_at → done): re-arm the program so the
    stb = dec.buffers[dec.program.step_state]              # profiled steps do real work (they continue the generation)
    state = layout.unpack(stb.read(0, layout.size))
    state.update(done=0, stop_at=0)
    stb.write(layout.pack(state), 0)
    runs = dec.profile(a.steps)
    timings = op_timings(dec.program, runs)
    print(f"# decode step of {Path(a.model).name} on {sess.dev.info().name}: {len(dec.program.ops)} dispatches, {a.steps} profiled steps")
    print(format_budget(timings, sess.profile.nominal_gbps))
    top = sorted(timings, key=lambda t: -t.ms_min)[:12]
    print("\nlargest ops (min ms): " + ", ".join(f"{t.name} {t.ms_min:.3f}" for t in top))
    if a.trace:
        write_chrome_trace(str(a.trace), dec.program, runs)
        print(f"trace written: {a.trace} (open in https://ui.perfetto.dev)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
