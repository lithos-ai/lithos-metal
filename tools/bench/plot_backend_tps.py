"""Plot projected TPS from saved eight-row latency evidence (no GPU run).

Requires matplotlib. Run from any directory with a Python environment containing
matplotlib; outputs PNG, SVG, PDF, and the exact plotted CSV alongside the report.
"""

import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tools/bench/results/m5max-27b-n7"
OUTPUT = ROOT / "docs/research/figures"
CONTEXTS = (128, 4096, 8192, 16384, 32768)
ACCEPTED_TOKENS = 6


def load_series():
    with (RESULTS / "mlp-optimization/final.csv").open() as stream:
        layers = [row for row in csv.DictReader(stream) if row["part"] == "layer"]
    gdn = [row for row in layers if row["kind"] == "gdn" and int(row["ctx"]) == 128]
    assert len(gdn) == 48
    series = {name: [] for name in ("Lithos Metal", "MLX", "Ollama", "vLLM-Metal")}
    for ctx in CONTEXTS:
        attention = [
            row for row in layers if row["kind"] == "attention" and int(row["ctx"]) == ctx
        ]
        assert len(attention) == 16
        # GDN has fixed recurrent state: reuse its measured 128-token layer
        # estimates, exactly as in the prior decoder-only latency estimates.
        for name, field in (
            ("Lithos Metal", "optimized_mlp_native_us"),
            ("MLX", "fastest_mlx_us"),
        ):
            selected = gdn + attention
            assert len({row["layer"] for row in selected}) == 64
            assert all(row["oracle_pass"] == "True" for row in selected)
            latency = math.fsum(float(row[field]) for row in selected) / 1000
            series[name].append(latency)
    for name, filename in (("Ollama", "ollama.jsonl"), ("vLLM-Metal", "vllm-bf16.jsonl")):
        rows = [json.loads(line) for line in (RESULTS / "backend-verification" / filename).read_text().splitlines()]
        if name == "Ollama":
            rows = [row for row in rows if not row["capture_per_token_states"]]
        assert len(rows) == len(CONTEXTS)
        by_context = {row["context"]: row for row in rows}
        assert set(by_context) == set(CONTEXTS)
        for ctx in CONTEXTS:
            row = by_context[ctx]
            assert row["target_rows"] == 8 and len(row["wall_ms"]) == 20
            series[name].append(median(row["wall_ms"]))
    return series


def main():
    series = load_series()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "m5max-27b-projected-tps"
    with stem.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "backend", "context_tokens", "target_rows", "assumed_accepted_tokens",
            "latency_ms", "projected_tps", "latency_basis",
        ])
        writer.writeheader()
        for name, latencies in series.items():
            for ctx, latency in zip(CONTEXTS, latencies):
                assert math.isfinite(latency) and latency > 0
                writer.writerow({
                    "backend": name, "context_tokens": ctx, "target_rows": 8,
                    "assumed_accepted_tokens": ACCEPTED_TOKENS,
                    "latency_ms": latency,
                    "projected_tps": 1000 / latency * ACCEPTED_TOKENS,
                    "latency_basis": (
                        "sum of isolated decoder-layer minima; 128-token GDN reused at all contexts"
                        if name in ("Lithos Metal", "MLX") else
                        "median full forward; final state; FP8 projections materialized to BF16"
                        if name == "Ollama" else
                        "median paged-prefill full-forward proxy; FP8 projections materialized to BF16"
                    ),
                })

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": "#243247", "text.color": "#17263C",
        "xtick.color": "#41536B", "ytick.color": "#41536B",
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(12, 6.8), facecolor="white")
    ax = fig.add_axes((0.10, 0.135, 0.865, 0.665))
    fig.text(0.10, 0.94, "Qwen3.8-27B", fontsize=22, weight="bold")

    styles = {
        "Lithos Metal": ("#007C78", "o", 0),
        "MLX": ("#2563C9", "s", 4),
        "Ollama": ("#CC7015", "D", 0),
        "vLLM-Metal": ("#8B4BB3", "^", -4),
    }
    handles = []
    for name, latencies in series.items():
        color, marker, label_offset = styles[name]
        values = [1000 / latency * ACCEPTED_TOKENS for latency in latencies]
        line, = ax.plot(range(5), values, label=name, color=color, marker=marker,
                        linestyle="-", linewidth=2.6, markersize=7,
                        markeredgecolor="white", markeredgewidth=1.1)
        handles.append(line)
        # Annotate each series' highest throughput across the five contexts.
        peak_index = max(range(len(values)), key=values.__getitem__)
        ax.annotate(f"{values[peak_index]:.1f}", (peak_index, values[peak_index]),
                    xytext=(-12, label_offset), textcoords="offset points",
                    ha="right", va="center", fontsize=11,
                    weight="bold", color=color)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.093, 0.885),
               frameon=False, ncol=4, fontsize=11, columnspacing=2.5, handlelength=3)
    ax.set_xticks(range(5), ["128", "4K", "8K", "16K", "32K"])
    ax.set_xlim(-0.5, 4.15)
    ax.set_ylim(0, 180)
    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.set_xlabel("Context length (tokens)", labelpad=12)
    ax.set_ylabel("TPS (tokens/s) · higher is better", labelpad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=200, facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
