"""Draw the full-attention mixer computation-to-task figure (no GPU execution).

Matches plot_gdn_task_graph.py. The selected 8K configuration has 120 persistent
workers, one task per queue claim, local partitions of 1024 keys, and no separate
Q/K preparation stage. The phase structure follows static_fusion.py and the
attention module: input layout, QKV, attention/gate siblings, merge, output.
Orange counter-managed events depict the existing all-worker phase barriers.
Unnumbered task boxes illustrate queue use, not measured assignments or latency.

Sources:
  docs/research/m5max-27b-attention-optimization.md
  monolith/backends/metal/m5_max_40c/recipes/attention-optimization/selected-contexts.json
  monolith/compiler/static_fusion.py
  monolith/compiler/attention_fusion.py
  monolith/nn/attention.py

Outputs PNG and editable SVG. No runtime or configuration is modified.
"""

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from plot_gdn_task_graph import CORE, EDGE, EVENT, INK, MUTED, NEUTRAL, PROJECTION


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "monolith/backends/metal/m5_max_40c/recipes/attention-optimization"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/research/figures/attention-lithos-metal-task-graph")
    args = parser.parse_args()
    recipes = json.loads((EVIDENCE / "selected-contexts.json").read_text())
    config = json.loads((EVIDENCE / recipes["8192"]).read_text())["attention"]
    if config["attention_prepare"] or config["schedule"] != "queue":
        raise ValueError("Figure requires the queued recipe with inline Q/K preparation")
    workers_count = config["workers"]
    partition_keys = config["attention_chunk_tiles"] * 32

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(14.5, 8.2), facecolor="white")
    ax = fig.add_axes((0.025, 0.025, 0.95, 0.95))
    ax.set(xlim=(0, 14.15), ylim=(1.06, 8.85))
    ax.set_axis_off()
    box_labels = []

    def label(x, y, value, size=11, weight="normal", color=INK, ha="center"):
        return ax.text(x, y, value, ha=ha, va="center", fontsize=size,
                       weight=weight, color=color, linespacing=1.3, zorder=5)

    def box(x, y, w, h, value, color=PROJECTION, size=11):
        node = Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(node)
        text = label(x + w / 2, y + h / 2, value, size)
        box_labels.append((node, text))

    def path(points, arrow=True, color=EDGE, width=1.05):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=color, linewidth=width, zorder=1)
        a, b = points[-2:]
        if arrow:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                        shrinkA=0, shrinkB=1, linewidth=width,
                                        color=color, zorder=2))
        else:
            ax.plot((a[0], b[0]), (a[1], b[1]), color=color, linewidth=width, zorder=1)

    label(7.075, 8.56, "lithos-metal converts full attention in Qwen3.8 27B into a megakernel", 17, "medium")
    label(1.99, 8.10, "(a) Computation graph", 12, "medium")
    label(9.60, 8.10, "(b) Example worker task graph", 12, "medium")
    label(9.60, 7.78,
          f"8K prefix · {workers_count} persistent GPU worker groups · one megakernel dispatch\n"
          "Counter-managed events synchronize task phases",
          9.5, color=MUTED)

    # Gate and residual paths bypass the attention core. The core appends K/V
    # and computes partial softmax results over the prefix plus new tokens.
    label(1.99, 7.52, "Hidden input · 8 tokens × 5120 channels", 10)
    box(.69, 6.88, 2.60, .40, "RMSNorm + input layouts", CORE)
    path([(1.99, 7.39), (1.99, 7.28)])
    for x, w, name in ((.39, 1.91, "QKV projection\nFP8"),
                       (2.55, 1.26, "Gate\nFP8")):
        path([(1.99, 6.88), (1.99, 6.57), (x + w / 2, 6.57), (x + w / 2, 6.24)])
        box(x, 5.65, w, .59, name)
    box(.39, 3.73, 2.30, 1.40,
        "Attention core\nQ/K RMSNorm + RoPE\nCausal QKᵀ + softmax\nWeighted sum of V\nKV append", CORE, 10)
    path([(1.345, 5.65), (1.345, 5.39), (1.54, 5.39), (1.54, 5.13)])
    box(3.00, 4.08, .82, .64, "KV\ncache", NEUTRAL, 10)
    path([(3.00, 4.54), (2.69, 4.54)])
    path([(2.69, 4.23), (3.00, 4.23)])
    box(.69, 2.60, 2.60, .60, "Merge softmax partials\n× sigmoid(gate)", CORE, 10.5)
    path([(1.54, 3.73), (1.54, 3.44), (1.99, 3.44), (1.99, 3.20)])
    path([(3.18, 5.65), (3.94, 5.65), (3.94, 2.90), (3.29, 2.90)])
    box(.69, 1.50, 2.60, .68, "Output projection (FP8)\n+ residual", size=10.5)
    path([(1.99, 2.60), (1.99, 2.18)])
    path([(1.99, 7.385), (.14, 7.385), (.14, 1.84), (.69, 1.84)])
    label(.24, 3.46, "Residual", 9, ha="left")

    path([(4.07, 4.54), (4.34, 4.54)], width=1.5)
    box(4.34, 4.25, .90, .58, "lithos-\nmetal", NEUTRAL, 11)
    path([(5.24, 4.54), (5.70, 4.54)], width=1.5)
    label(4.79, 3.96, "Tile +\nschedule", 10, color=MUTED)

    frame = Rectangle((5.71, 1.26), 7.78, 6.32, fill=False,
                      edgecolor="#ccd4dd", linewidth=1)
    ax.add_patch(frame)
    workers = (6.75, 9.60, 12.45)
    worker_ids = (0, workers_count // 2, workers_count - 1)
    for x, worker_id in zip(workers, worker_ids):
        label(x, 7.47, f"worker {worker_id}", 11, "medium")
    label(8.20, 7.47, "…", 13, color=MUTED)
    label(11.03, 7.47, "…", 13, color=MUTED)
    box(6.07, 6.88, 7.06, .40, "Input normalization + layout tasks", CORE)

    def event(y, before, after, number, input_band=False):
        if input_band:
            path([(9.60, before), (9.60, y + .125)], arrow=False)
        else:
            for x in workers:
                path([(x, before), (x, y + .18), (9.60, y + .18),
                      (9.60, y + .125)], arrow=False)
        node = FancyBboxPatch((9.09, y - .125), 1.02, .25,
                              boxstyle="round,pad=0,rounding_size=0.125",
                              facecolor=EVENT, edgecolor="none", zorder=4)
        ax.add_patch(node)
        text = label(9.60, y, f"event {number}", 9)
        box_labels.append((node, text))
        path([(9.60, y - .125), (9.60, after)])

    def pool(y, title, first_top, size=10):
        box(6.07, y, 7.06, .25, title, NEUTRAL, size)
        for x in workers:
            path([(x, y), (x, first_top)])

    event(6.65, 6.88, 6.50, 0, input_band=True)
    pool(6.25, "Shared task queue: QKV projection", 6.12)
    for x in workers:
        for i, y in enumerate((5.85, 5.49, 5.13)):
            box(x - .90, y, 1.80, .27, "QKV tile", PROJECTION, 10)
            if i < 2:
                path([(x, y), (x, y - .09)])

    event(4.90, 5.13, 4.76, 1)
    pool(4.51, f"Shared task queue: attention partitions (≤ {partition_keys} keys) + gate", 4.37, 9.5)
    core_order = (("Attention partition", "Gate tile", "Attention partition"),
                  ("Attention partition", "Attention partition", "Gate tile"),
                  ("Gate tile", "Attention partition", "Attention partition"))
    for x, tasks in zip(workers, core_order):
        for i, (y, task) in enumerate(zip((4.10, 3.74, 3.38), tasks)):
            box(x - .90, y, 1.80, .27, task,
                CORE if task == "Attention partition" else PROJECTION, 9.5)
            if i < 2:
                path([(x, y), (x, y - .09)])

    event(3.16, 3.38, 2.94, 2)
    box(6.07, 2.60, 7.06, .34,
        "Merge softmax partials · × sigmoid(gate) · output layout", CORE, 10.5)
    event(2.36, 2.60, 2.22, 3, input_band=True)
    pool(1.97, "Shared task queue: output projection", 1.84)
    for x in workers:
        box(x - .90, 1.50, 1.80, .34, "Output tiles + residual", PROJECTION, 9.5)

    # Verify the rendered labels, including the expanded event and worker names.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for node, text in box_labels:
        rb, tb = node.get_window_extent(renderer), text.get_window_extent(renderer)
        if not (rb.x0 <= tb.x0 and tb.x1 <= rb.x1 and rb.y0 <= tb.y0 and tb.y1 <= rb.y1):
            raise RuntimeError(f"Label does not fit its node: {text.get_text()!r}")
    text_items = list(ax.texts)
    bounds = ax.get_window_extent(renderer)
    for i, a in enumerate(text_items):
        ab = a.get_window_extent(renderer)
        if not (bounds.x0 <= ab.x0 and ab.x1 <= bounds.x1
                and bounds.y0 <= ab.y0 and ab.y1 <= bounds.y1):
            raise RuntimeError(f"Text outside figure: {a.get_text()!r}")
        for b in text_items[i + 1:]:
            bb = b.get_window_extent(renderer)
            if min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > 1 and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > 1:
                raise RuntimeError(f"Text overlaps: {a.get_text()!r} / {b.get_text()!r}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".svg"):
        fig.savefig(args.out.with_suffix(extension), dpi=240, facecolor="white")
    plt.close(fig)
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
