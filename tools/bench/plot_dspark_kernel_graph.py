"""Render the current DSpark draft topology from the saved dispatch inventory.

The overview collapses repeated layers; the right panel expands the actual
mixer fusion boundary. This is the measured implementation, not a proposed
schedule. No GPU execution or changes to kernel recipes are needed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
INK = '#203043'
EDGE = '#718397'
BLUE = '#e3edf9'
GREEN = '#e6f2eb'
ORANGE = '#fff0de'
GRAY = '#eff2f6'
PURPLE = '#ede7f6'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT/'tools/bench/results/m5max-27b-dspark/refinement-20261003/full-bf16-128.json')
    parser.add_argument('--out', type=Path, default=ROOT/'docs/research/figures/dspark-current-kernel-graph')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())['contexts'][0]['draft_profile']
    kernels = profile['kernels']
    counts = {group: sum(k['category'] == group for k in kernels)
              for group in ('mixer_megakernels', 'mlp_gate_up', 'mlp_down', 'markov_projection')}
    assert counts == dict(mixer_megakernels=5, mlp_gate_up=5, mlp_down=5, markov_projection=7)
    assert len(kernels) == 68
    for k in kernels:
        if k['category'] == 'mixer_megakernels':
            assert k['meta']['task_functions'] == ['gemm_tile', 'gqa_decode_mma', 'gqa_merge', 'gemm_tile']

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(17, 11.5), facecolor='white')
    ax = fig.add_axes((.02, .025, .96, .95))
    ax.set(xlim=(0, 17), ylim=(0, 11.5))
    ax.set_axis_off()
    labels = []

    def text(x, y, value, size=11, weight='normal', color=INK, ha='center'):
        return ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
                       ha=ha, va='center', linespacing=1.28, zorder=5)

    def box(x, y, w, h, value, color=BLUE, size=11, edge='none', lw=1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.07',
                              facecolor=color, edgecolor=edge, linewidth=lw, zorder=3)
        ax.add_patch(patch)
        label = text(x+w/2, y+h/2, value, size)
        labels.append((patch, label))

    def path(points, color=EDGE, arrow=True, dashed=False, lw=1.1):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=color, linewidth=lw,
                    linestyle='--' if dashed else '-', zorder=1)
        a, b = points[-2:]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>' if arrow else '-',
                                    mutation_scale=10, linewidth=lw, color=color,
                                    linestyle='--' if dashed else '-', shrinkA=0,
                                    shrinkB=1, zorder=2))

    text(8.5, 11.2, 'Lithos Metal · current DSpark draft kernel graph', 20, 'medium')
    text(8.5, 10.85, 'Qwen3.8-27B · BF16 draft · 7-token block · 128 / 32K context configuration', 11, color=EDGE)
    text(1.95, 10.35, 'Block pipeline', 13, 'medium')
    text(10.75, 10.35, 'Inside each draft decoder layer', 13, 'medium')

    # Program overview. Norm statistics and output layouts are fused into
    # projection epilogues where available; only the first input needs setup.
    x, w, cx = .35, 3.2, 1.95
    overview = [
        (9.43, .63, 'Shared embedding\n[anchor, mask × 6]', BLUE),
        (8.63, .52, 'Initial norm + input layout', GRAY),
        (7.94, .47, 'Draft layer 0', GREEN),
        (7.30, .47, 'Draft layer 1', GREEN),
        (6.66, .47, 'Draft layer 2', GREEN),
        (6.02, .47, 'Draft layer 3', GREEN),
        (5.38, .47, 'Draft layer 4', GREEN),
        (4.48, .66, 'Final norm + shared LM head\n7 × 248320 base logits', BLUE),
    ]
    for i, (y, h, value, color) in enumerate(overview):
        box(x, y, w, h, value, color, 10.6)
        if i:
            path([(cx, overview[i-1][0]), (cx, y+h)])

    # The context feature branch is shared by all layers, but each layer owns
    # its K/V projection and cache. QKV remains inside the mixer megakernel.
    box(4.25, 8.93, 3.85, 1.12,
        'Committed target features\ntaps 5, 19, 33, 47, 61\ntap_concat → FC → RMSNorm', BLUE, 10.8)
    text(6.175, 8.69, 'Shared features for all five layers', 9.5, color=EDGE)
    box(4.65, 7.70, 3.05, .62, 'Context K/V projection\nseparate native kernel', BLUE, 10.6)
    path([(6.175, 8.93), (6.175, 8.82)], arrow=False)
    path([(6.175, 8.53), (6.175, 8.32)])
    box(4.65, 6.58, 3.05, .68, 'Layer context K/V cache\nappend committed rows only', GRAY, 10.2)

    # Four sequential stages inside one real megakernel dispatch.
    frame = FancyBboxPatch((8.65, 5.95), 7.15, 3.78,
                           boxstyle='round,pad=0,rounding_size=.12',
                           facecolor=ORANGE, edgecolor='#dc9e52', linewidth=1.3, zorder=0)
    ax.add_patch(frame)
    text(12.225, 9.49, 'Mixer megakernel · one dispatch', 12, 'medium')
    bx, bw, mid = 9.05, 5.38, 11.74
    stages = [
        (8.69, .54, 'Block QKV projection'),
        (7.80, .64, 'Matrix attention + Q/K norm + YaRN\ncached prefix + injected KV + proposal KV'),
        (7.07, .48, 'Attention partial reduction'),
        (6.23, .59, 'Output projection + residual\npost-norm statistics + output layout'),
    ]
    for i, (y, h, value) in enumerate(stages):
        box(bx, y, bw, h, value, '#fffaf2', 10.3)
        if i:
            path([(mid, stages[i-1][0]), (mid, y+h)])
    text(12.225, 10.04, 'Layer hidden input', 11)
    path([(mid, 9.88), (mid, 9.73)], arrow=False)
    path([(mid, 9.30), (mid, 9.23)])
    # Residual hidden input bypasses attention within the fused region.
    path([(mid, 9.88), (15.15, 9.88), (15.15, 6.525), (14.43, 6.525)])
    text(15.18, 7.10, 'residual', 9, ha='left', color=EDGE)
    path([(7.70, 8.01), (8.30, 8.01), (8.30, 8.19), (9.05, 8.19)])
    path([(7.70, 7.09), (8.07, 7.09), (8.07, 7.93), (9.05, 7.93)])
    path([(9.05, 7.84), (8.40, 7.84), (8.40, 6.80), (7.70, 6.80)], dashed=True)

    box(bx, 5.09, bw, .60, 'MLP gate/up + SwiGLU\nnative kernel 1', GREEN, 10.8)
    path([(mid, 6.23), (mid, 5.69)])
    box(bx, 4.20, bw, .64, 'MLP down + residual\nnext-layer norm statistics / layout · native kernel 2', GREEN, 10.0)
    path([(mid, 5.09), (mid, 4.84)])
    path([(mid, 5.83), (15.15, 5.83), (15.15, 4.52), (14.43, 4.52)])
    text(15.18, 5.06, 'residual', 9, ha='left', color=EDGE)
    text(6.175, 5.45, 'Repeated for layers 0–4', 11, 'medium')
    text(6.175, 4.91, 'Context KV stays outside fusion.\nMLP remains two native kernels.', 10.1, color=EDGE)

    # Markov position k consumes its own base-logit row and the selected token
    # from k-1. Each position emits gather, W2 GEMV, argmax partial and final.
    text(8.5, 3.75, 'Sequential Markov chain · 7 positions × 4 dispatches', 13, 'medium')
    centers = [1.80 + i*2.17 for i in range(7)]
    path([(cx, 4.48), (cx, 4.00), (.52, 4.00), (.52, 3.24), (centers[-1], 3.24)], arrow=False)
    text(8.25, 3.40, 'base_logits[k] supplied to each position', 9.5, color=EDGE)
    for k, center in enumerate(centers):
        box(center-.91, 2.10, 1.82, .91, f'd{k}\nW1 gather → W2\n+ base → argmax', PURPLE, 10.0)
        path([(center, 3.24), (center, 3.01)])
        if k:
            path([(centers[k-1]+.91, 2.55), (center-.91, 2.55)])
    text(.40, 2.76, 'anchor', 9)
    path([(.08, 2.55), (centers[0]-.91, 2.55)])

    text(5.56, 1.63, 'Draft hidden + saved W1 rows', 10, color=EDGE)
    box(3.55, .70, 4.02, .67, 'Confidence head\n1 kernel', BLUE, 10.8)
    path([(5.56, 1.46), (5.56, 1.37)])
    box(9.00, .70, 4.1, .67, 'verify_select\nfixed 7 drafts → next T = 8', BLUE, 10.8)
    path([(7.57, 1.035), (9.00, 1.035)])
    path([(centers[-1], 2.10), (centers[-1], 1.67), (11.05, 1.67), (11.05, 1.37)])
    text(13.28, 1.84, 'draft_ids[0:7]', 9.5, color=EDGE)
    path([(13.10, 1.035), (14.05, 1.035)])
    text(15.30, 1.035, 'Anchor + 7 drafts\nto target verification', 10.3)
    text(8.5, .19, '128 / 32K: 5 mixer megakernels + 10 native MLP kernels · 68 draft dispatches\n'
         '4K–16K: each mixer uses 5 native dispatches · 88 draft dispatches', 9.5, color=EDGE)

    # Reject clipped labels rather than silently producing an unreadable graph.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, label in labels:
        pb, tb = patch.get_window_extent(renderer), label.get_window_extent(renderer)
        if not (pb.x0 <= tb.x0 and tb.x1 <= pb.x1 and pb.y0 <= tb.y0 and tb.y1 <= pb.y1):
            raise RuntimeError('Label exceeds box: '+label.get_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.svg'):
        fig.savefig(args.out.with_suffix(extension), dpi=200, facecolor='white')
    plt.close(fig)
    print(args.out.with_suffix('.png'))


if __name__ == '__main__':
    main()
