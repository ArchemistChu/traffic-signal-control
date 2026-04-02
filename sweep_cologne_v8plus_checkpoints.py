#!/usr/bin/env python3
"""
Sweep numbered MARL checkpoints (e.g. *_ep0250.pt) via evaluate_marl_osm.py, write
per-checkpoint JSON summaries, aggregate ranks, and print a suggested full eval command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _stem_from_checkpoint_prefix(prefix: str) -> str:
    p = prefix.strip()
    if p.lower().endswith(".pt"):
        return p[:-3]
    return p


def _checkpoint_path(stem: str, episode: int) -> str:
    return f"{stem}_ep{int(episode):04d}.pt"


def _episode_throughput_efficiency(m: Dict[str, Any]) -> Optional[float]:
    if not m:
        return None
    raw = m.get("throughput_efficiency")
    if raw is not None and _finite(raw):
        return float(raw)
    tp = m.get("throughput")
    vd = m.get("vehicles_departed")
    if tp is not None and vd is not None and _finite(tp) and _finite(vd) and float(vd) != 0.0:
        return float(tp) / float(vd)
    tph = m.get("throughput_per_hour")
    dph = m.get("departed_per_hour")
    if tph is not None and dph is not None and _finite(tph) and _finite(dph) and float(dph) != 0.0:
        return float(tph) / float(dph)
    return None


def _metric_series(results: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for r in results:
        m = r.get("metrics")
        if not isinstance(m, dict):
            continue
        v = m.get(key)
        if _finite(v):
            out.append(float(v))
    return out


def _te_series(results: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for r in results:
        m = r.get("metrics")
        if not isinstance(m, dict):
            continue
        te = _episode_throughput_efficiency(m)
        if te is not None:
            out.append(te)
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _load_eval_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_eval_json(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    results = data.get("results")
    if not isinstance(results, list):
        results = []

    keys = (
        "throughput",
        "avg_waiting_time",
        "avg_queue_length",
        "congestion_index",
        "avg_speed",
        "total_co2",
        "avg_co2_per_vehicle",
        "normalized_waiting_burden",
    )
    summary: Dict[str, Optional[float]] = {k: _mean(_metric_series(results, k)) for k in keys}
    summary["throughput_efficiency"] = _mean(_te_series(results))
    return summary


def _rank_sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Ascending sort = best first: high te, low wait, low ci, high spd."""
    te = row.get("throughput_efficiency")
    awt = row.get("avg_waiting_time")
    ci = row.get("congestion_index")
    spd = row.get("avg_speed")

    te_v = float(te) if te is not None and math.isfinite(float(te)) else float("-inf")
    awt_v = float(awt) if awt is not None and math.isfinite(float(awt)) else float("inf")
    ci_v = float(ci) if ci is not None and math.isfinite(float(ci)) else float("inf")
    spd_v = float(spd) if spd is not None and math.isfinite(float(spd)) else float("-inf")

    return (-te_v, awt_v, ci_v, -spd_v)


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def _build_eval_cmd(
    python_exe: str,
    repo_root: str,
    dataset: str,
    model_path: str,
    eval_episodes: int,
    duration: int,
    decision_interval: int,
    controlled_lights_ratio: float,
    lanes_per_tl: int,
    demand_scale: float,
    regional_reward_weight: float,
    region_grid_size: float,
    seed: int,
    out_json: str,
) -> str:
    parts = [
        python_exe,
        os.path.join(repo_root, "evaluate_marl_osm.py"),
        "--dataset",
        dataset,
        "--model",
        model_path,
        "--episodes",
        str(eval_episodes),
        "--duration",
        str(duration),
        "--decision-interval",
        str(decision_interval),
        "--controlled-lights-ratio",
        str(controlled_lights_ratio),
        "--lanes-per-tl",
        str(lanes_per_tl),
        "--demand-scale",
        str(demand_scale),
        "--regional-reward-weight",
        str(regional_reward_weight),
        "--region-grid-size",
        str(region_grid_size),
        "--seed",
        str(seed),
        "--out",
        out_json,
    ]
    return subprocess.list2cmdline(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cologne", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument(
        "--checkpoint-prefix",
        default="models/marl_cologne_shared_dqn_regionaware_v8_plus_scale07.pt",
        help="Base .pt path or stem; numbered checkpoints use _epNNNN.pt",
    )
    parser.add_argument(
        "--candidate-episodes",
        type=int,
        nargs="+",
        default=[250, 300, 350, 400, 450, 500],
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--controlled-lights-ratio", type=float, default=0.5)
    parser.add_argument("--lanes-per-tl", type=int, default=20)
    parser.add_argument("--demand-scale", type=float, default=0.7)
    parser.add_argument("--regional-reward-weight", type=float, default=0.01)
    parser.add_argument("--region-grid-size", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/v8_plus_scale07_checkpoint_sweep")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for evaluate_marl_osm.py")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    repo = _repo_root()
    stem = _stem_from_checkpoint_prefix(args.checkpoint_prefix)
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(repo, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    eval_script = os.path.join(repo, "evaluate_marl_osm.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"Missing {eval_script}")

    rows_out: List[Dict[str, Any]] = []

    for ep in args.candidate_episodes:
        ck = _checkpoint_path(stem, ep)
        ck_abs = ck if os.path.isabs(ck) else os.path.join(repo, ck)
        if not os.path.isfile(ck_abs):
            print(f"[skip] missing checkpoint: {ck}", file=sys.stderr)
            continue

        out_json = os.path.join(out_dir, f"eval_ep{ep:04d}.json")

        if args.reuse_existing and os.path.isfile(out_json):
            print(f"[reuse] {out_json}")
        else:
            cmd = [
                args.python,
                eval_script,
                "--dataset",
                args.dataset,
                "--model",
                ck_abs,
                "--episodes",
                str(args.eval_episodes),
                "--duration",
                str(args.duration),
                "--decision-interval",
                str(args.decision_interval),
                "--controlled-lights-ratio",
                str(args.controlled_lights_ratio),
                "--lanes-per-tl",
                str(args.lanes_per_tl),
                "--demand-scale",
                str(args.demand_scale),
                "--regional-reward-weight",
                str(args.regional_reward_weight),
                "--region-grid-size",
                str(args.region_grid_size),
                "--seed",
                str(args.seed),
                "--out",
                out_json,
            ]
            print(f"[eval] ep={ep} -> {out_json}")
            subprocess.run(cmd, cwd=repo, check=True)

        data = _load_eval_json(out_json)
        metrics_mean = _summarize_eval_json(data)
        row = {
            "checkpoint_episode": ep,
            "checkpoint_path": os.path.normpath(ck_abs),
            "eval_json": os.path.normpath(os.path.abspath(out_json)),
            **metrics_mean,
        }
        rows_out.append(row)

    if not rows_out:
        print("No checkpoints evaluated.", file=sys.stderr)
        sys.exit(1)

    ranked = sorted(rows_out, key=_rank_sort_key)

    csv_path = os.path.join(out_dir, "checkpoint_summary.csv")
    json_path = os.path.join(out_dir, "checkpoint_summary.json")

    fieldnames = [
        "rank",
        "checkpoint_episode",
        "checkpoint_path",
        "eval_json",
        "throughput_efficiency",
        "throughput",
        "avg_waiting_time",
        "avg_queue_length",
        "congestion_index",
        "avg_speed",
        "total_co2",
        "avg_co2_per_vehicle",
        "normalized_waiting_burden",
    ]

    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in ranked:
            w.writerow(r)

    payload = {
        "ranking": "throughput_efficiency (desc), avg_waiting_time (asc), congestion_index (asc), avg_speed (desc)",
        "checkpoints": ranked,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    best = ranked[0]
    print()
    print("Checkpoint sweep (best -> worst)")
    print(
        "Rank  Ep    te_eff   wait    ci      spd     throughput  eval_json"
    )
    print("-" * 88)
    for r in ranked:
        print(
            f"{r['rank']:4d}  {r['checkpoint_episode']:4d}  "
            f"{_fmt(r.get('throughput_efficiency'), 6):>8}  "
            f"{_fmt(r.get('avg_waiting_time'), 2):>6}  "
            f"{_fmt(r.get('congestion_index'), 4):>6}  "
            f"{_fmt(r.get('avg_speed'), 4):>6}  "
            f"{_fmt(r.get('throughput'), 2):>10}  "
            f"{os.path.basename(r['eval_json'])}"
        )
    print()

    final_out = os.path.join(out_dir, f"eval_best_ep{best['checkpoint_episode']:04d}_full20.json")
    rec = _build_eval_cmd(
        args.python,
        repo,
        args.dataset,
        best["checkpoint_path"],
        20,
        args.duration,
        args.decision_interval,
        args.controlled_lights_ratio,
        args.lanes_per_tl,
        args.demand_scale,
        args.regional_reward_weight,
        args.region_grid_size,
        args.seed,
        final_out,
    )
    print("Recommended full evaluation (20 episodes):")
    print(rec)
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
