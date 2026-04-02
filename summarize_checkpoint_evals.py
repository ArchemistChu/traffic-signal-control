#!/usr/bin/env python3
"""
Aggregate MARL checkpoint evaluation JSON files, rank checkpoints by a fixed
heuristic, and optionally write JSON / Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EP_PATTERN = re.compile(r"_ep(\d{3,})", re.IGNORECASE)

RANK_HEURISTIC_DESC = (
    "throughput_efficiency (higher), congestion_index (lower), throughput (higher), "
    "avg_waiting_time (lower), avg_queue_length (lower), total_co2 (lower), avg_speed (higher)"
)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


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


def _metric_series(results: List[Dict[str, Any]], key: str, nested: str = "metrics") -> List[float]:
    out: List[float] = []
    for r in results:
        if nested:
            m = r.get(nested) or {}
            if not isinstance(m, dict):
                continue
            v = m.get(key)
        else:
            v = r.get(key)
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


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def parse_checkpoint_episode(model_path: str, input_path: str) -> Optional[int]:
    for s in (model_path or "", Path(input_path).name):
        m = EP_PATTERN.search(s.replace("\\", "/"))
        if m:
            return int(m.group(1))
    return None


def load_eval_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_file(path: Path) -> Dict[str, Any]:
    data = load_eval_json(path)
    results = data.get("results")
    if not isinstance(results, list):
        results = []

    model = str(data.get("model", "") or "")
    ck_ep = parse_checkpoint_episode(model, str(path))

    te_vals = _te_series(results)
    tp_vals = _metric_series(results, "throughput")
    awt_vals = _metric_series(results, "avg_waiting_time")
    ci_vals = _metric_series(results, "congestion_index")
    aql_vals = _metric_series(results, "avg_queue_length")
    spd_vals = _metric_series(results, "avg_speed")
    co2_vals = _metric_series(results, "total_co2")
    aco2_vals = _metric_series(results, "avg_co2_per_vehicle")
    rs_vals = _metric_series(results, "reward_sum", nested="")
    ar_vals = _metric_series(results, "avg_reward", nested="")

    def pack(vals: List[float]) -> Dict[str, Optional[float]]:
        mu, sd = _mean_std(vals)
        return {"mean": mu, "stddev": sd}

    metrics_mean: Dict[str, Optional[float]] = {}
    metrics_std: Dict[str, Optional[float]] = {}
    for name, vals in (
        ("throughput_efficiency", te_vals),
        ("throughput", tp_vals),
        ("avg_waiting_time", awt_vals),
        ("congestion_index", ci_vals),
        ("avg_queue_length", aql_vals),
        ("avg_speed", spd_vals),
        ("total_co2", co2_vals),
        ("avg_co2_per_vehicle", aco2_vals),
        ("reward_sum", rs_vals),
        ("avg_reward", ar_vals),
    ):
        mu, sd = _mean_std(vals)
        metrics_mean[name] = mu
        metrics_std[name] = sd

    return {
        "input_path": str(path.resolve()),
        "dataset": data.get("dataset"),
        "model": model,
        "eval_episodes": data.get("episodes"),
        "checkpoint_episode": ck_ep,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
    }


def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
    mm = row["metrics_mean"]
    te = mm.get("throughput_efficiency")
    ci = mm.get("congestion_index")
    tp = mm.get("throughput")
    awt = mm.get("avg_waiting_time")
    aql = mm.get("avg_queue_length")
    co2 = mm.get("total_co2")
    spd = mm.get("avg_speed")
    return (
        float(te) if te is not None else float("-inf"),
        -float(ci) if ci is not None else float("-inf"),
        float(tp) if tp is not None else float("-inf"),
        -float(awt) if awt is not None else float("-inf"),
        -float(aql) if aql is not None else float("-inf"),
        -float(co2) if co2 is not None else float("-inf"),
        float(spd) if spd is not None else float("-inf"),
    )


def format_optional(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def print_console(candidates: List[Dict[str, Any]], best: Dict[str, Any]) -> None:
    print("Checkpoint evaluation summary (ranked best -> worst)")
    print(RANK_HEURISTIC_DESC)
    print()
    hdr = (
        f"{'Rank':>4}  {'Ep':>6}  {'te_mean':>10}  {'ci_mean':>10}  {'tp_mean':>10}  "
        f"{'wait_m':>10}  {'co2_mean':>12}  {'spd_m':>8}  path"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in candidates:
        mm = c["metrics_mean"]
        ep = c.get("checkpoint_episode")
        ep_s = str(ep) if ep is not None else "?"
        print(
            f"{c['rank']:4d}  {ep_s:>6}  "
            f"{format_optional(mm.get('throughput_efficiency'), 6):>10}  "
            f"{format_optional(mm.get('congestion_index')):>10}  "
            f"{format_optional(mm.get('throughput'), 2):>10}  "
            f"{format_optional(mm.get('avg_waiting_time'), 2):>10}  "
            f"{format_optional(mm.get('total_co2'), 2):>12}  "
            f"{format_optional(mm.get('avg_speed'), 4):>8}  "
            f"{Path(c['input_path']).name}"
        )
    print()
    bmm = best["metrics_mean"]
    print("Recommended best checkpoint:")
    print(f"  checkpoint_episode: {best.get('checkpoint_episode')}")
    print(f"  model_path:         {best.get('model')}")
    print(f"  eval_json:          {best.get('input_path')}")
    print(
        f"  metrics (means): te={format_optional(bmm.get('throughput_efficiency'), 6)} "
        f"ci={format_optional(bmm.get('congestion_index'))} "
        f"tp={format_optional(bmm.get('throughput'), 2)} "
        f"wait={format_optional(bmm.get('avg_waiting_time'), 2)} "
        f"co2={format_optional(bmm.get('total_co2'), 2)} "
        f"spd={format_optional(bmm.get('avg_speed'), 4)}"
    )


def write_markdown(path: Path, candidates: List[Dict[str, Any]], best: Dict[str, Any]) -> None:
    lines = [
        "# Checkpoint evaluation summary",
        "",
        f"Heuristic: {RANK_HEURISTIC_DESC}",
        "",
        "## Ranked checkpoints",
        "",
        "| Rank | Ep | te_mean | ci_mean | tp_mean | wait_mean | q_mean | co2_mean | spd_mean | eval JSON |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in candidates:
        mm = c["metrics_mean"]
        ep = c.get("checkpoint_episode")
        lines.append(
            "| {rank} | {ep} | {te} | {ci} | {tp} | {wt} | {ql} | {co2} | {spd} | `{name}` |".format(
                rank=c["rank"],
                ep="?" if ep is None else ep,
                te=format_optional(mm.get("throughput_efficiency"), 6),
                ci=format_optional(mm.get("congestion_index")),
                tp=format_optional(mm.get("throughput"), 2),
                wt=format_optional(mm.get("avg_waiting_time"), 2),
                ql=format_optional(mm.get("avg_queue_length"), 2),
                co2=format_optional(mm.get("total_co2"), 2),
                spd=format_optional(mm.get("avg_speed"), 4),
                name=Path(c["input_path"]).name,
            )
        )
    lines.extend(
        [
            "",
            "## Recommended",
            "",
            f"- **checkpoint_episode**: {best.get('checkpoint_episode')}",
            f"- **model_path**: `{best.get('model')}`",
            f"- **eval_json**: `{best.get('input_path')}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MARL checkpoint eval JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more eval JSON files.")
    parser.add_argument("--out", type=str, default="", help="Optional summary JSON output path.")
    parser.add_argument("--markdown", type=str, default="", help="Optional Markdown report path.")
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.inputs]
    for p in paths:
        if not p.is_file():
            print(f"Error: input not found: {p}", file=sys.stderr)
            return 1

    rows = [summarize_file(p) for p in paths]
    ranked = sorted(rows, key=sort_key, reverse=True)
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i

    best = ranked[0]
    best_out = {
        "rank": best["rank"],
        "checkpoint_episode": best.get("checkpoint_episode"),
        "checkpoint_path": best.get("model"),
        "source_eval": best.get("input_path"),
        "metrics_mean": best.get("metrics_mean"),
        "metrics_std": best.get("metrics_std"),
    }

    print_console(ranked, best)

    payload = {
        "rank_heuristic": RANK_HEURISTIC_DESC,
        "generated_from": [str(p) for p in paths],
        "best": best_out,
        "candidates": ranked,
    }

    if args.out:
        outp = Path(args.out).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON summary: {outp}")

    if args.markdown:
        md_path = Path(args.markdown).expanduser().resolve()
        write_markdown(md_path, ranked, best)
        print(f"Wrote Markdown report: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
