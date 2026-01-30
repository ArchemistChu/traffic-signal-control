#!/usr/bin/env python3
"""
Summarize evaluation JSON files produced by evaluate_marl_osm.py.

Why:
- Papers typically report mean ± std (and often 95% CI) across *independent runs* (different seeds),
  not across repeated identical episodes.

What this does:
- Loads the eval JSON
- Extracts selected metrics from each episode
- Optionally groups episodes into "runs" (e.g., 10 episodes per run)
- Computes mean/std/95% CI over runs and prints:
  - JSON summary
  - LaTeX table row snippet

Usage:
  python summarize_eval_json.py --in output/eval_cologne_from_vancouver_merged_ep120.json --group-size 10
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_METRICS = [
    "avg_waiting_time",
    "avg_queue_length",
    "throughput_per_hour",
    "congestion_index",
    "avg_speed",
    "vehicles_arrived",
    "avg_pressure",
    "total_co2",
    "total_fuel",
]


@dataclass
class SummaryStats:
    n: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    min: float
    max: float


def _t_crit_975(df: int) -> float:
    """Approximate t_{0.975,df} without SciPy (good enough for report)."""
    # Normal approximation for df>=30
    if df >= 30:
        return 1.96
    # Small lookup table for common dfs
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
    }
    return table.get(df, 1.96)


def summarize(values: List[float]) -> Optional[SummaryStats]:
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(max(0.0, var))
    else:
        std = 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    t = _t_crit_975(n - 1) if n >= 2 else 0.0
    ci_low = mean - t * sem
    ci_high = mean + t * sem
    return SummaryStats(
        n=n,
        mean=mean,
        std=std,
        ci95_low=ci_low,
        ci95_high=ci_high,
        min=min(values),
        max=max(values),
    )


def group(values: List[float], group_size: int) -> List[List[float]]:
    if group_size <= 1:
        return [[v] for v in values]
    out: List[List[float]] = []
    for i in range(0, len(values), group_size):
        out.append(values[i : i + group_size])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Eval JSON input path")
    ap.add_argument("--out", dest="out_path", default="", help="Optional JSON output summary path")
    ap.add_argument("--group-size", type=int, default=10, help="Episodes per independent run (default: 10)")
    ap.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    ap.add_argument("--label", type=str, default="", help="Label used in LaTeX row (e.g., MARL_DQN (VAN→COL))")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    results = data.get("results") or []
    metrics = [m.strip() for m in (args.metrics or "").split(",") if m.strip()]

    # collect episode-level
    episode_vals: Dict[str, List[float]] = {m: [] for m in metrics}
    for ep in results:
        mobj = ep.get("metrics") or {}
        for m in metrics:
            if m in mobj and mobj[m] is not None:
                try:
                    episode_vals[m].append(float(mobj[m]))
                except Exception:
                    pass

    # group into runs
    run_vals: Dict[str, List[float]] = {}
    for m, vals in episode_vals.items():
        chunks = group(vals, int(args.group_size))
        # use per-run mean
        per_run = [(sum(c) / len(c)) for c in chunks if c]
        run_vals[m] = per_run

    # summarize over runs
    run_stats: Dict[str, Dict[str, float]] = {}
    warnings: List[str] = []
    for m, per_run in run_vals.items():
        if not per_run:
            continue
        uniq = len(set(per_run))
        if uniq <= 1:
            warnings.append(
                f"Metric '{m}' has no variation across runs (unique={uniq}). "
                f"This usually means SUMO was deterministic (missing --seed) or demand is fixed."
            )
        st = summarize(per_run)
        if st is None:
            continue
        run_stats[m] = {
            "n_runs": st.n,
            "mean": st.mean,
            "std": st.std,
            "ci95_low": st.ci95_low,
            "ci95_high": st.ci95_high,
            "min": st.min,
            "max": st.max,
        }

    summary = {
        "input": args.in_path,
        "dataset": data.get("dataset"),
        "model": data.get("model"),
        "episodes": data.get("episodes"),
        "group_size": int(args.group_size),
        "n_runs": max((v.get("n_runs", 0) for v in run_stats.values()), default=0),
        "run_stats": run_stats,
        "warnings": warnings,
    }

    print(json.dumps(summary, indent=2))

    # LaTeX snippet (single row, pick key metrics if present)
    label = args.label.strip() or "Model"
    def _fmt(m: str) -> str:
        if m not in run_stats:
            return "n/a"
        mu = run_stats[m]["mean"]
        sd = run_stats[m]["std"]
        return f"{mu:.2f} $\\pm$ {sd:.2f}"

    print("\n% LaTeX table row snippet:")
    print(
        f"% {label} & {_fmt('avg_waiting_time')} & {_fmt('avg_queue_length')} & {_fmt('throughput_per_hour')} & {_fmt('congestion_index')} \\\\"
    )

    if args.out_path:
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

