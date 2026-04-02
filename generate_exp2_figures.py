"""Generate Experiment 2 (v16, scale 0.85) figures for the FYP report."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("Report", "figures", "report_figure_pdfs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRATEGIES = {
    "Adaptive":     "output/eval_cologne_ADAPTIVE_scale085_ep30_seed500_emissions.json",
    "MARL DQN":     "output/eval_cologne_MARL_v16_scale085_r02_ep30_seed500_emissions.json",
    "MaxPressure":  "output/eval_cologne_MAX_PRESSURE_scale085_ep30_seed500_emissions.json",
    "GA":           "output/eval_cologne_GA_scale085_ep30_seed500_emissions.json",
    "Fixed-Time":   "output/eval_cologne_FIXED_TIME_scale085_ep30_seed500_emissions.json",
    "SOTL":         "output/eval_cologne_SOTL_scale085_ep30_seed500_emissions.json",
}

COLORS = {
    "Adaptive":    "#2196F3",
    "MARL DQN":    "#E53935",
    "MaxPressure": "#43A047",
    "GA":          "#FF9800",
    "Fixed-Time":  "#7E57C2",
    "SOTL":        "#00ACC1",
}


def load_all():
    data = {}
    for name, path in STRATEGIES.items():
        with open(path) as f:
            raw = json.load(f)
        episodes = []
        for r in raw["results"]:
            m = r["metrics"]
            episodes.append({
                "episode":    r["episode"],
                "wait":       m["avg_waiting_time"],
                "speed":      m["avg_speed"],
                "throughput": m["throughput"],
                "queue":      m["avg_queue_length"],
                "co2":        m["total_co2"],
                "fuel":       m["total_fuel"],
                "nox":        m["total_nox"],
                "pmx":        m["total_pmx"],
            })
        data[name] = episodes
    return data


def fig_bar_chart(data):
    """Grouped bar chart: 4 traffic metrics x 6 strategies."""
    metrics = [
        ("wait",       "Avg Waiting Time (s)"),
        ("speed",      "Avg Speed (m/s)"),
        ("throughput", "Throughput (veh/ep)"),
        ("queue",      "Avg Queue Length"),
    ]
    names = list(STRATEGIES.keys())
    n_groups = len(metrics)
    n_bars = len(names)
    x = np.arange(n_groups)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, name in enumerate(names):
        vals = [np.mean([e[m] for e in data[name]]) for m, _ in metrics]
        errs = [np.std([e[m] for e in data[name]])  for m, _ in metrics]
        # Normalize each metric to the Fixed-Time mean for visual comparison
        ft_means = [np.mean([e[m] for e in data["Fixed-Time"]]) for m, _ in metrics]
        norm_vals = [v / ft for v, ft in zip(vals, ft_means)]
        norm_errs = [e / ft for e, ft in zip(errs, ft_means)]
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, norm_vals, width, yerr=norm_errs,
                       label=name, color=COLORS[name], capsize=2, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Relative to Fixed-Time (1.0 = FT baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_title("Experiment 2: Traffic Metrics Comparison (demand 0.85, ratio 0.2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_bar_chart_comparison.pdf"), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_bar_chart_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Created exp2_bar_chart_comparison.pdf")


def fig_emissions(data):
    """Grouped bar chart: 4 emission metrics x 6 strategies."""
    metrics = [
        ("co2",  "Total CO₂ (×10⁹ mg)"),
        ("fuel", "Total Fuel (×10⁹ ml)"),
        ("nox",  "Total NOₓ (×10⁶ mg)"),
        ("pmx",  "Total PMₓ (×10⁶ mg)"),
    ]
    divisors = {"co2": 1e9, "fuel": 1e9, "nox": 1e6, "pmx": 1e6}
    names = list(STRATEGIES.keys())
    n_groups = len(metrics)
    n_bars = len(names)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax_idx, (mkey, mlabel) in enumerate(metrics):
        ax = axes[ax_idx]
        div = divisors[mkey]
        means = []
        stds = []
        for name in names:
            vals = [e[mkey] / div for e in data[name]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(n_bars)
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                       color=[COLORS[n] for n in names], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("Fixed-Time", "FT").replace("MaxPressure", "MaxP")
                            for n in names], fontsize=7, rotation=30, ha="right")
        ax.set_title(mlabel, fontsize=9)

        ymin = min(means) - 2 * max(stds)
        ymax = max(means) + 2 * max(stds)
        margin = (ymax - ymin) * 0.15
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Experiment 2: Environmental Metrics (demand 0.85, ratio 0.2)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_emissions_comparison.pdf"), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_emissions_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Created exp2_emissions_comparison.pdf")


def fig_waiting_time_vs_episode(data):
    """Line plot: avg waiting time per episode for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in STRATEGIES:
        eps = sorted(data[name], key=lambda e: e["episode"])
        x = [e["episode"] for e in eps]
        y = [e["wait"] for e in eps]
        ax.plot(x, y, label=name, color=COLORS[name], linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Waiting Time (s)")
    ax.set_title("Experiment 2: Average Waiting Time per Episode (demand 0.85, ratio 0.2)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_waiting_time_vs_episode.pdf"), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_waiting_time_vs_episode.png"), dpi=150)
    plt.close(fig)
    print("  Created exp2_waiting_time_vs_episode.pdf")


def fig_throughput_vs_episode(data):
    """Line plot: throughput per episode for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in STRATEGIES:
        eps = sorted(data[name], key=lambda e: e["episode"])
        x = [e["episode"] for e in eps]
        y = [e["throughput"] for e in eps]
        ax.plot(x, y, label=name, color=COLORS[name], linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (vehicles arrived)")
    ax.set_title("Experiment 2: Throughput per Episode (demand 0.85, ratio 0.2)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_throughput_vs_episode.pdf"), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, "exp2_throughput_vs_episode.png"), dpi=150)
    plt.close(fig)
    print("  Created exp2_throughput_vs_episode.pdf")


if __name__ == "__main__":
    print("Loading evaluation data...")
    data = load_all()
    print(f"Loaded {len(data)} strategies, {len(next(iter(data.values())))} episodes each.\n")

    print("Generating figures:")
    fig_bar_chart(data)
    fig_emissions(data)
    fig_waiting_time_vs_episode(data)
    fig_throughput_vs_episode(data)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
