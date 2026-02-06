#!/usr/bin/env python3
"""
Evaluate baseline controllers on OSM maps with seeded episodes and optional emissions.

This produces an eval JSON with the same shape as evaluate_marl_osm.py, so you can:
- run summarize_eval_json.py for mean/std/CI
- use visualize_statistics.ipynb to plot ATT/AIP/ASIP + emissions vs episode

Example:
  python evaluate_baselines_osm.py --dataset cologne --strategy FIXED_TIME --episodes 120 --duration 1200 --seed 42 --run-id 1 ^
    --controlled-lights-ratio 0.2 --sumo-emissions-output --out output/eval_cologne_FIXED_TIME_ep120.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cologne", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["FIXED_TIME", "ADAPTIVE", "MAX_PRESSURE", "SOTL", "GA", "PRESSLIGHT"],
        help="Baseline strategy to evaluate",
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=5, help="Kept for metadata parity (controller checks internally).")
    parser.add_argument("--controlled-lights-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--sumo-emissions-output", action="store_true", help="Enable SUMO native emission-output and parse totals.")
    parser.add_argument("--keep-emissions-xml", action="store_true", help="Keep emission XML files (default: delete after parsing).")
    parser.add_argument("--out", type=str, default="", help="Output eval JSON path")
    args = parser.parse_args()

    if args.controlled_lights_ratio <= 0.0 or args.controlled_lights_ratio > 1.0:
        raise ValueError("--controlled-lights-ratio must be in (0,1].")

    np.random.seed(int(args.seed))

    SimulationConfig.set_dataset(args.dataset)
    SimulationConfig.create_output_dirs()

    out_path = args.out.strip() or os.path.join(
        SimulationConfig.OUTPUT_DIR,
        f"eval_{args.dataset}_{args.strategy}_ep{int(args.episodes)}.json",
    )

    results: List[Dict[str, Any]] = []

    print("=" * 78)
    print(f"Baseline evaluation: dataset={args.dataset} strategy={args.strategy}")
    print(f"Episodes={args.episodes} duration={args.duration}s ratio={args.controlled_lights_ratio:.2f} seed={args.seed} run_id={args.run_id}")
    if args.sumo_emissions_output:
        print("Emissions: SUMO native emission-output ENABLED")
    print(f"Output: {out_path}")
    print("=" * 78)

    for ep in range(1, int(args.episodes) + 1):
        sumo_seed = int(args.seed) + int(args.run_id) * 10000 + int(ep)
        sim = TrafficSimulator(
            use_gui=False,
            dataset=args.dataset,
            enable_sumo_emissions_output=bool(args.sumo_emissions_output),
            sumo_seed=sumo_seed,
            controlled_lights_ratio=float(args.controlled_lights_ratio),
        )

        start = time.time()
        metrics: Dict[str, Any] = {}
        try:
            metrics = sim.run_simulation(duration=int(args.duration), strategy=args.strategy) or {}
        finally:
            # run_simulation() closes SUMO in its finally; ensure it's closed even if exceptions occur.
            try:
                sim.close_simulation()
            except Exception:
                pass

        # Delete large emissions XML after parsing (unless user wants to keep it)
        if args.sumo_emissions_output and (not args.keep_emissions_xml):
            try:
                ef = getattr(sim, "emissions_output_file", None)
                if ef and os.path.exists(ef):
                    os.remove(ef)
            except Exception:
                pass

        wall = time.time() - start
        print(f"Episode {ep:04d}/{args.episodes} | wall={wall:.2f}s")
        results.append(
            {
                "episode": int(ep),
                "sumo_seed": int(sumo_seed),
                "wall_seconds": float(wall),
                "reward_sum": None,
                "avg_reward": None,
                "decisions": None,
                "metrics": metrics,
            }
        )

    payload = {
        "dataset": args.dataset,
        "strategy": args.strategy,
        "episodes": int(args.episodes),
        "duration": int(args.duration),
        "decision_interval": int(args.decision_interval),
        "controlled_lights_ratio": float(args.controlled_lights_ratio),
        "seed": int(args.seed),
        "run_id": int(args.run_id),
        "sumo_emissions_output": bool(args.sumo_emissions_output),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved baseline eval summary: {out_path}")


if __name__ == "__main__":
    main()

