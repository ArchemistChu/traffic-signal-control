#!/usr/bin/env python3
"""
Compare evaluation JSON summaries produced by evaluate_marl_osm.py.
Prints averaged metrics across episodes for easy side-by-side comparison.
"""

import argparse
import json
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def avg_metric(results, key: str) -> float | None:
    vals = []
    for ep in results:
        metrics = ep.get("metrics") or {}
        if key in metrics and metrics[key] is not None:
            try:
                vals.append(float(metrics[key]))
            except Exception:
                pass
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def print_summary(label: str, data: Dict[str, Any]) -> Dict[str, float | None]:
    results = data.get("results") or []
    metrics = {
        "avg_waiting_time": avg_metric(results, "avg_waiting_time"),
        "avg_queue_length": avg_metric(results, "avg_queue_length"),
        "throughput_per_hour": avg_metric(results, "throughput_per_hour"),
        "congestion_index": avg_metric(results, "congestion_index"),
    }
    print(f"\n{label}")
    print("-" * len(label))
    print(f"dataset: {data.get('dataset')}")
    print(f"episodes: {data.get('episodes')} duration: {data.get('duration')}")
    for k, v in metrics.items():
        if v is None:
            print(f"{k}: n/a")
        else:
            print(f"{k}: {v:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Path to first eval JSON (e.g., Vancouver)")
    parser.add_argument("--b", required=True, help="Path to second eval JSON (e.g., Los Angeles)")
    args = parser.parse_args()

    data_a = load_json(args.a)
    data_b = load_json(args.b)

    print_summary("A", data_a)
    print_summary("B", data_b)


if __name__ == "__main__":
    main()
