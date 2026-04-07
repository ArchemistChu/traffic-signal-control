#!/usr/bin/env python3
"""
Runs comparative evaluation (MARL + baselines) as a separate OS process.
Started by app_flask so work continues when Flask debug mode reloads the server
(daemon threads do not survive a reload).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict


def _safe_json_load(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _mean_metrics_from_eval_payload(payload: dict) -> dict:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    keys = [
        "avg_waiting_time",
        "avg_speed",
        "throughput_per_hour",
        "total_co2",
        "total_fuel",
        "total_nox",
        "total_pmx",
        "avg_queue_length",
        "avg_pressure",
        "congestion_index",
    ]
    sums = {k: 0.0 for k in keys}
    cnts = {k: 0 for k in keys}
    for ep in results:
        m = (ep or {}).get("metrics", {}) or {}
        for k in keys:
            if k in m and m[k] is not None:
                try:
                    sums[k] += float(m[k])
                    cnts[k] += 1
                except Exception:
                    pass
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = (sums[k] / cnts[k]) if cnts[k] > 0 else None
    out["episodes_counted"] = len(results)
    return out


def _write_status(status_path: str, obj: dict) -> None:
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparative eval worker (Flask-spawned).")
    parser.add_argument("run_dir", help="Directory containing job.json and status.json")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    job_path = os.path.join(run_dir, "job.json")
    status_path = os.path.join(run_dir, "status.json")
    summary_path = os.path.join(run_dir, "summary.json")
    log_path = os.path.join(run_dir, "worker.log")

    proj_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(proj_root)

    def _log(line: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(line + "\n")
        except Exception:
            pass

    job = _safe_json_load(job_path)
    if not isinstance(job, dict):
        _write_status(status_path, {"status": "error", "message": "Missing or invalid job.json", "run_dir": run_dir})
        return 1

    dataset = str(job["dataset"])
    model_path = str(job["model_path"])
    episodes = int(job["episodes"])
    duration = int(job["duration"])
    decision_interval = int(job["decision_interval"])
    ratio = float(job["controlled_lights_ratio"])
    lanes_per_tl = int(job["lanes_per_tl"])
    demand_scale = float(job.get("demand_scale", 0.0))
    seed = int(job.get("seed", 42))
    run_id = int(job.get("run_id", 1))
    regional_reward_weight = float(job.get("regional_reward_weight", 0.0))
    region_grid_size = float(job.get("region_grid_size", 500.0))
    sumo_emissions_output = bool(job.get("sumo_emissions_output", True))
    baselines = job.get("baselines") or ["FIXED_TIME", "MAX_PRESSURE", "SOTL", "ADAPTIVE"]
    if not isinstance(baselines, list):
        baselines = []
    baselines = [str(x).upper() for x in baselines if str(x).upper() in {"FIXED_TIME", "ADAPTIVE", "MAX_PRESSURE", "SOTL"}]

    if not os.path.isfile(model_path):
        msg = f"Model file not found: {model_path}"
        _log(msg)
        _write_status(
            status_path,
            {"status": "error", "message": msg, "run_dir": run_dir, "worker": "subprocess"},
        )
        return 1

    py = sys.executable

    try:
        outputs: Dict[str, str] = {}
        cmds: list = []

        marl_out = os.path.join(run_dir, "eval_model.json")
        marl_cmd = [
            py, "evaluate_marl_osm.py",
            "--dataset", dataset,
            "--model", model_path,
            "--episodes", str(episodes),
            "--duration", str(duration),
            "--decision-interval", str(decision_interval),
            "--controlled-lights-ratio", str(ratio),
            "--lanes-per-tl", str(lanes_per_tl),
            "--seed", str(seed),
            "--run-id", str(run_id),
            "--demand-scale", str(demand_scale),
            "--regional-reward-weight", str(regional_reward_weight),
            "--region-grid-size", str(region_grid_size),
            "--out", marl_out,
        ]
        if sumo_emissions_output:
            marl_cmd.append("--sumo-emissions-output")
        cmds.append(("MARL_DQN", marl_cmd, marl_out))

        for strat in baselines:
            out_path = os.path.join(run_dir, f"eval_{strat}.json")
            cmd = [
                py, "evaluate_baselines_osm.py",
                "--dataset", dataset,
                "--strategy", strat,
                "--episodes", str(episodes),
                "--duration", str(duration),
                "--decision-interval", str(decision_interval),
                "--controlled-lights-ratio", str(ratio),
                "--seed", str(seed),
                "--run-id", str(run_id),
                "--demand-scale", str(demand_scale),
                "--out", out_path,
            ]
            if sumo_emissions_output:
                cmd.append("--sumo-emissions-output")
            cmds.append((strat, cmd, out_path))

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _log(f"\n===== comparative worker {ts} =====")
        _log(f"cwd={proj_root}")
        _log(f"SUMO/Python steps: {len(cmds)}")

        _write_status(status_path, {
            "status": "running",
            "progress": 0,
            "total_jobs": len(cmds),
            "completed_jobs": 0,
            "current": None,
            "run_dir": run_dir,
            "worker": "subprocess",
        })

        for idx, (name, cmd, out_path) in enumerate(cmds, start=1):
            _write_status(status_path, {
                "status": "running",
                "progress": int((idx - 1) * 100 / max(1, len(cmds))),
                "total_jobs": len(cmds),
                "completed_jobs": idx - 1,
                "current": name,
                "cmd": " ".join(str(c) for c in cmd),
                "run_dir": run_dir,
                "worker": "subprocess",
            })
            _log(f"\n--- START {name} ---\n" + " ".join(str(c) for c in cmd))
            with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=proj_root,
                    stdin=subprocess.DEVNULL,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            _log(f"--- END {name} (ok) ---")
            outputs[name] = out_path

        summary = {
            "dataset": dataset,
            "episodes": episodes,
            "duration": duration,
            "decision_interval": decision_interval,
            "controlled_lights_ratio": ratio,
            "demand_scale": demand_scale,
            "seed": seed,
            "run_id": run_id,
            "model_path": model_path,
            "sumo_emissions_output": sumo_emissions_output,
            "files": outputs,
            "mean_metrics": {},
        }
        for name, path in outputs.items():
            payload = _safe_json_load(path) or {}
            summary["mean_metrics"][name] = _mean_metrics_from_eval_payload(payload)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        _write_status(status_path, {
            "status": "completed",
            "progress": 100,
            "run_dir": run_dir,
            "summary_path": summary_path,
            "files": outputs,
            "worker": "subprocess",
        })
        return 0
    except subprocess.CalledProcessError as e:
        cmd_s = " ".join(str(x) for x in (e.cmd or []))
        _log(f"CalledProcessError returncode={e.returncode}\n{cmd_s}")
        _write_status(status_path, {
            "status": "error",
            "message": (
                f"Step failed with exit code {e.returncode}. "
                f"Open worker.log in the run folder for full output. Command (truncated): {cmd_s[:400]}"
            ),
            "run_dir": run_dir,
            "worker": "subprocess",
            "failed_command": cmd_s[:800],
        })
        return 1
    except Exception as e:
        _log(traceback.format_exc())
        _write_status(status_path, {
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "run_dir": run_dir,
            "worker": "subprocess",
        })
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as e:
        try:
            if len(sys.argv) > 1:
                rd = os.path.abspath(sys.argv[1])
                sp = os.path.join(rd, "status.json")
                _write_status(
                    sp,
                    {
                        "status": "error",
                        "message": f"Worker crashed before setup: {type(e).__name__}: {e}",
                        "run_dir": rd,
                        "worker": "subprocess",
                    },
                )
        except Exception:
            pass
        raise
