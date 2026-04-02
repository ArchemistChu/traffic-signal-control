#!/usr/bin/env python3
"""Regenerate the Experiment 1 bar chart with zoomed y-axes."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

OUTPUT_DIR = Path("Report/figures")

FILES = {
    "FIXED_TIME": "output/eval_cologne_FIXED_TIME_scale07_ep120_seed42.json",
    "SOTL": "output/eval_cologne_SOTL_scale07_ep120_seed342.json",
    "ADAPTIVE": "output/eval_cologne_ADAPTIVE_scale07_ep120_seed242.json",
    "MAX_PRESSURE": "output/eval_cologne_MAX_PRESSURE_scale07_ep120_seed142.json",
    "MARL_DQN": "output/eval_cologne_MARL_DQN_v8_scale07_ep300_seed42.json",
}

STRATEGY_ORDER = ["FIXED_TIME", "SOTL", "ADAPTIVE", "MAX_PRESSURE", "MARL_DQN"]

NICE_NAMES = {
    "FIXED_TIME": "Fixed-Time",
    "SOTL": "SOTL",
    "ADAPTIVE": "Adaptive",
    "MAX_PRESSURE": "Max Pressure",
    "MARL_DQN": "MARL DQN (Ours)",
}

COLORS = {
    "FIXED_TIME": "#264653",
    "SOTL": "#E76F51",
    "ADAPTIVE": "#2A9D8F",
    "MAX_PRESSURE": "#E9C46A",
    "MARL_DQN": "#E63946",
}

BAR_METRICS = [
    ("avg_waiting_time", "Avg Waiting Time (s)", False),
    ("throughput_per_hour", "Throughput (veh/hr)", True),
    ("avg_queue_length", "Avg Queue Length (veh)", False),
    ("congestion_index", "Congestion Index", False),
]


def mean_std_ci(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t_val = sp_stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    lo = mu - t_val * se
    hi = mu + t_val * se
    return mu, sd, lo, hi


def load_metrics(path, metric_key):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [r["metrics"][metric_key] for r in data["results"] if metric_key in r["metrics"]]


def main():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    axes = axes.flatten()

    for idx, (ax, (metric, ylabel, higher_better)) in enumerate(zip(axes, BAR_METRICS)):
        means, cis, colors, labels_list = [], [], [], []
        for strat in STRATEGY_ORDER:
            path = FILES[strat]
            vals = load_metrics(path, metric)
            mu, sd, lo, hi = mean_std_ci(vals)
            if math.isnan(mu):
                continue
            means.append(mu)
            cis.append(sd)
            colors.append(COLORS[strat])
            labels_list.append(NICE_NAMES[strat])

        x = np.arange(len(labels_list))
        bars = ax.bar(
            x, means, yerr=cis, capsize=4, color=colors,
            edgecolor="black", linewidth=0.4, alpha=0.85,
            error_kw=dict(linewidth=0.8),
        )

        best_idx = int(np.argmax(means) if higher_better else np.argmin(means))
        bars[best_idx].set_edgecolor("#E63946")
        bars[best_idx].set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_labels[idx]} {ylabel}", loc="left", fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

        data_min = min(m - c for m, c in zip(means, cis))
        data_max = max(m + c for m, c in zip(means, cis))
        data_range = data_max - data_min
        pad = max(data_range * 1.5, abs(data_max) * 0.005)
        ax.set_ylim(data_min - pad, data_max + pad)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "nb_bar_chart_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "nb_bar_chart_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    report_pdf_dir = OUTPUT_DIR / "report_figure_pdfs"
    if report_pdf_dir.exists():
        fig2, axes2 = plt.subplots(2, 2, figsize=(7.5, 6))
        axes2 = axes2.flatten()
        for idx, (ax, (metric, ylabel, higher_better)) in enumerate(zip(axes2, BAR_METRICS)):
            means, cis, colors, labels_list = [], [], [], []
            for strat in STRATEGY_ORDER:
                vals = load_metrics(FILES[strat], metric)
                mu, sd, lo, hi = mean_std_ci(vals)
                if math.isnan(mu):
                    continue
                means.append(mu)
                cis.append(sd)
                colors.append(COLORS[strat])
                labels_list.append(NICE_NAMES[strat])
            x = np.arange(len(labels_list))
            bars = ax.bar(x, means, yerr=cis, capsize=4, color=colors,
                          edgecolor="black", linewidth=0.4, alpha=0.85,
                          error_kw=dict(linewidth=0.8))
            best_idx = int(np.argmax(means) if higher_better else np.argmin(means))
            bars[best_idx].set_edgecolor("#E63946")
            bars[best_idx].set_linewidth(2.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_list, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{panel_labels[idx]} {ylabel}", loc="left", fontsize=10)
            ax.grid(True, alpha=0.2, axis="y")
            data_min = min(m - c for m, c in zip(means, cis))
            data_max = max(m + c for m, c in zip(means, cis))
            data_range = data_max - data_min
            pad = max(data_range * 1.5, abs(data_max) * 0.005)
            ax.set_ylim(data_min - pad, data_max + pad)
        fig2.tight_layout()
        fig2.savefig(report_pdf_dir / "nb_bar_chart_comparison.pdf", bbox_inches="tight")
        fig2.savefig(report_pdf_dir / "nb_bar_chart_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Also saved to {report_pdf_dir}")

    print("Done. Saved zoomed bar chart to:")
    print(f"  {OUTPUT_DIR / 'nb_bar_chart_comparison.pdf'}")
    print(f"  {OUTPUT_DIR / 'nb_bar_chart_comparison.png'}")


if __name__ == "__main__":
    main()
