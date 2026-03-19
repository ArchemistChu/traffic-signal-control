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


def _to_json_safe(obj):
    """Recursively convert objects (including dict keys) to JSON-safe types."""
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            safe[str(k)] = _to_json_safe(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


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
    parser.add_argument("--episode-start", type=int, default=1, help="Starting episode number (for parallel evaluation)")
    parser.add_argument("--episode-end", type=int, help="Ending episode number (for parallel evaluation, defaults to episodes if not specified)")
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=5, help="Kept for metadata parity (controller checks internally).")
    parser.add_argument("--controlled-lights-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--port", type=int, default=8813, help="TraCI port for SUMO connection (for parallel evaluation)")
    parser.add_argument("--sumo-emissions-output", action="store_true", help="Enable SUMO native emission-output and parse totals.")
    parser.add_argument("--keep-emissions-xml", action="store_true", help="Keep emission XML files (default: delete after parsing).")
    parser.add_argument(
        "--data-collection-interval",
        type=int,
        default=10,
        help="Collect vehicle data every N steps. Use 1 for reliable metrics; 50+ for faster runs (may miss data if simulation ends early).",
    )
    parser.add_argument("--demand-scale", type=float, default=0.0, help="SUMO --scale for traffic demand (0=default, 0.7=70%%)")
    parser.add_argument("--out", type=str, default="", help="Output eval JSON path")
    args = parser.parse_args()

    if args.controlled_lights_ratio <= 0.0 or args.controlled_lights_ratio > 1.0:
        raise ValueError("--controlled-lights-ratio must be in (0,1].")

    # Determine episode range
    episode_start = max(1, int(args.episode_start))
    episode_end = args.episode_end if args.episode_end is not None else int(args.episodes)
    total_episodes = episode_end - episode_start + 1

    if total_episodes <= 0:
        raise ValueError("--episode-end must be >= --episode-start")

    np.random.seed(int(args.seed))

    SimulationConfig.set_dataset(args.dataset)
    SimulationConfig.create_output_dirs()

    out_path = args.out.strip() or os.path.join(
        SimulationConfig.OUTPUT_DIR,
        f"eval_{args.dataset}_{args.strategy}_ep{episode_start}-{episode_end}.json",
    )

    results: List[Dict[str, Any]] = []

    print("=" * 78)
    print(f"Baseline evaluation: dataset={args.dataset} strategy={args.strategy}")
    print(f"Episodes={episode_start}-{episode_end} ({total_episodes} total) duration={args.duration}s ratio={args.controlled_lights_ratio:.2f} seed={args.seed} run_id={args.run_id}")
    if args.sumo_emissions_output:
        print("Emissions: SUMO native emission-output ENABLED")
    print(f"Output: {out_path}")
    print("=" * 78)

    for ep in range(episode_start, episode_end + 1):
        sumo_seed = int(args.seed) + int(args.run_id) * 10000 + int(ep)
        sim = TrafficSimulator(
            use_gui=False,
            port=int(args.port),
            dataset=args.dataset,
            enable_sumo_emissions_output=bool(args.sumo_emissions_output),
            sumo_seed=sumo_seed,
            controlled_lights_ratio=float(args.controlled_lights_ratio),
            demand_scale=float(args.demand_scale) if args.demand_scale > 0 else None,
        )
        sim.data_collection_interval = max(1, int(args.data_collection_interval))

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
        if not metrics:
            print(f"Episode {ep:04d}/{args.episodes} | wall={wall:.2f}s | WARNING: empty metrics (sim may have ended early)")
        else:
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
        "episodes": total_episodes,
        "episode_range": f"{episode_start}-{episode_end}",
        "duration": int(args.duration),
        "decision_interval": int(args.decision_interval),
        "controlled_lights_ratio": float(args.controlled_lights_ratio),
        "seed": int(args.seed),
        "run_id": int(args.run_id),
        "sumo_emissions_output": bool(args.sumo_emissions_output),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(payload), f, indent=2)

    print(f"\nSaved baseline eval summary: {out_path}")


if __name__ == "__main__":
    main()

