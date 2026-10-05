"""Draw the requested lithos-metal computation-to-task figure (no GPU execution).

The right panel is a PROPOSED balanced schedule, not the measured static-stage
default. Representative workers pull finer ready tiles from shared pools;
independent QKV/A-B and core/Z work can be interleaved within readiness phases.
Tile labels and box sizes are schematic, not a measured assignment or timeline.
No kernel, compiler, or profile configuration is modified by this script.

Outputs PNG plus editable SVG in docs/research/figures by default.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
INK = "#1d2938"
EDGE = "#788797"
MUTED = "#59697b"
PROJECTION = "#dceaf9"
CORE = "#e0efdf"
EVENT = "#f6d6b9"
NEUTRAL = "#edf0f3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/research/figures/gdn-lithos-metal-balanced-schedule")
    args = parser.parse_args()
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
        r = Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(r)
        t = label(x + w / 2, y + h / 2, value, size)
        box_labels.append((r, t))

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

    label(7.075, 8.56, "lithos-metal converts Gated DeltaNet in Qwen3.8 27B into a megakernel", 17, "medium")
    label(1.99, 8.10, "(a) Computation graph", 12, "medium")
    label(9.60, 8.10, "(b) Proposed worker task graph", 12, "medium")
    label(9.60, 7.78,
          "One megakernel dispatch · 80 persistent GPU worker groups\n"
          "Counter-managed events release tasks when dependencies finish",
          9.5, color=MUTED)

    # Computation dependencies. Z bypasses recurrence and feeds gated norm.
    label(1.99, 7.52, "Hidden input · 8 tokens × 5120 channels", 10)
    box(.69, 6.88, 2.60, .40, "RMSNorm + input layouts", CORE)
    path([(1.99, 7.39), (1.99, 7.28)])
    projections = [(.39, 1.03, "QKV\nFP8"), (1.67, .85, "A/B\nBF16"), (2.77, 1.04, "Z\nFP8")]
    for x, w, name in projections:
        path([(1.99, 6.88), (1.99, 6.57), (x + w / 2, 6.57), (x + w / 2, 6.24)])
        box(x, 5.65, w, .59, name)
    box(.39, 3.73, 2.30, 1.00, "GDN core\nConv1D + SiLU\nQ/K norm + gates\nDelta recurrence", CORE, 10.5)
    for x in (.905, 2.095):
        path([(x, 5.65), (x, 5.20), (1.54, 5.20), (1.54, 4.73)])
    box(3.00, 3.91, .82, .64, "Conv. /\nrecurrent\nstate", NEUTRAL, 8.5)
    path([(3.00, 4.37), (2.69, 4.37)])
    path([(2.69, 4.05), (3.00, 4.05)])
    box(.69, 2.60, 2.60, .60, "Gated RMSNorm\n× SiLU(Z)", CORE)
    path([(1.54, 3.73), (1.54, 3.44), (1.99, 3.44), (1.99, 3.20)])
    path([(3.29, 5.65), (3.94, 5.65), (3.94, 2.90), (3.29, 2.90)])
    box(.69, 1.50, 2.60, .68, "Output projection (FP8)\n+ residual", size=10.5)
    path([(1.99, 2.60), (1.99, 2.18)])
    path([(1.99, 7.385), (.14, 7.385), (.14, 1.84), (.69, 1.84)])
    label(.24, 4.92, "Residual", 9, ha="left")

    # Requested graph-to-task transformation label. Not a benchmarked claim.
    path([(4.07, 4.54), (4.34, 4.54)], width=1.5)
    box(4.34, 4.25, .90, .58, "lithos-\nmetal", NEUTRAL, 11)
    path([(5.24, 4.54), (5.70, 4.54)], width=1.5)
    label(4.79, 3.96, "Tile +\nschedule", 10, color=MUTED)

    # Representative, deliberately unnumbered tiles: equal-looking boxes
    # illustrate the balancing objective and are not latency measurements.
    frame = Rectangle((5.71, 1.26), 7.78, 6.32, fill=False, edgecolor="#ccd4dd", linewidth=1)
    ax.add_patch(frame)
    workers = (6.75, 9.60, 12.45)
    for x, name in zip(workers, ("worker 0", "worker 40", "worker 79")):
        label(x, 7.47, name, 11, "medium")
    label(8.20, 7.47, "…", 13, color=MUTED)
    label(11.03, 7.47, "…", 13, color=MUTED)
    box(6.07, 6.88, 7.06, .40, "Input normalization + layout tasks", CORE)

    def event(y, before, after, number, input_band=False, output_band=False):
        if input_band:
            path([(9.60, before), (9.60, y + .125)], arrow=False)
        else:
            for x in workers:
                path([(x, before), (x, y + .18), (9.60, y + .18), (9.60, y + .125)], arrow=False)
        node = FancyBboxPatch((9.09, y - .125), 1.02, .25,
                              boxstyle="round,pad=0,rounding_size=0.125",
                              facecolor=EVENT, edgecolor="none", zorder=4)
        ax.add_patch(node)
        text = label(9.60, y, f"event {number}", 9)
        box_labels.append((node, text))
        if output_band:
            path([(9.60, y - .125), (9.60, after)])
        else:
            for x in workers:
                path([(9.60, y - .125), (9.60, y - .17), (x, y - .17), (x, after)])

    def pool(y, title, first_top):
        box(6.07, y, 7.06, .25, title, NEUTRAL, 10)
        for x in workers:
            path([(x, y), (x, first_top)])

    event(6.65, 6.88, 6.50, 0, input_band=True, output_band=True)
    pool(6.25, "Shared ready-task pool: QKV + A/B", 6.12)
    projection_order = (("QKV tile", "A/B tile", "QKV tile"),
                        ("QKV tile", "QKV tile", "A/B tile"),
                        ("A/B tile", "QKV tile", "QKV tile"))
    for x, tasks in zip(workers, projection_order):
        for i, (y, task) in enumerate(zip((5.85, 5.49, 5.13), tasks)):
            box(x - .90, y, 1.80, .27, task, PROJECTION, 10)
            if i < 2:
                path([(x, y), (x, y - .09)])

    event(4.90, 5.13, 4.76, 1, output_band=True)
    pool(4.51, "Shared ready-task pool: Core + Z", 4.37)
    core_order = (("Core slice", "Z tile", "Z tile"),
                  ("Z tile", "Core slice", "Z tile"),
                  ("Z tile", "Z tile", "Core slice"))
    for x, tasks in zip(workers, core_order):
        for i, (y, task) in enumerate(zip((4.10, 3.74, 3.38), tasks)):
            box(x - .90, y, 1.80, .27, task, CORE if task == "Core slice" else PROJECTION, 10)
            if i < 2:
                path([(x, y), (x, y - .09)])

    event(3.16, 3.38, 2.94, 2, output_band=True)
    box(6.07, 2.60, 7.06, .34, "Gated normalization + output layout tasks", CORE, 10.5)
    event(2.36, 2.60, 2.22, 3, input_band=True, output_band=True)
    pool(1.97, "Shared ready-task pool: output projection", 1.84)
    for x in workers:
        box(x - .90, 1.50, 1.80, .34, "Output tiles + residual", PROJECTION, 9.5)

    # Check the actual rendered label bounds before saving the deliverables.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for r, t in box_labels:
        rb, tb = r.get_window_extent(renderer), t.get_window_extent(renderer)
        if not (rb.x0 <= tb.x0 and tb.x1 <= rb.x1 and rb.y0 <= tb.y0 and tb.y1 <= rb.y1):
            raise RuntimeError(f"Label does not fit its node: {t.get_text()!r}")
    text_items = list(ax.texts)
    for i, a in enumerate(text_items):
        ab = a.get_window_extent(renderer)
        for b in text_items[i + 1:]:
            bb = b.get_window_extent(renderer)
            if min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > 1 and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > 1:
                raise RuntimeError(f"Text overlaps: {a.get_text()!r} / {b.get_text()!r}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".svg"):
        fig.savefig(args.out.with_suffix(extension), dpi=240, facecolor="white")
    plt.close(fig)
    print(args.out.with_suffix('.png'))


if __name__ == "__main__":
    main()
