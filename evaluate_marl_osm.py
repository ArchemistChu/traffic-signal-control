#!/usr/bin/env python3
"""
Evaluate a trained MARL shared DQN on OSM maps (Los Angeles / Vancouver / Cologne).

This script reuses the same local observation and 2-action policy used in
train_marl_los_angeles.py, but runs in evaluation mode (no training).
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator
from src.rl_agent import RLAgent, TrafficState as RLTState

try:
    import traci
except ImportError:
    traci = None


def get_phase_count(tl_id: str) -> int:
    """SUMO-version-safe phase count for a TLS."""
    if hasattr(traci.trafficlight, "getPhaseNumber"):
        try:
            return int(traci.trafficlight.getPhaseNumber(tl_id))
        except Exception:
            pass
    try:
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if logics:
            return int(len(logics[0].phases))
    except Exception:
        pass
    return 1


def get_controlled_lanes_for_tl(tl_id: str) -> List[str]:
    """Get lanes controlled by a TLS (safe across SUMO versions)."""
    if hasattr(traci.trafficlight, "getControlledLanes"):
        try:
            lanes = list(traci.trafficlight.getControlledLanes(tl_id))
            lanes = [l for l in lanes if l]
            if lanes:
                return sorted(set(lanes))
        except Exception:
            pass

    lanes = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        for link_list in controlled_links:
            for link in link_list:
                if link and len(link) > 0:
                    lanes.add(link[0])
    except Exception:
        pass
    return sorted(lanes)


def build_local_state(tl_id: str, lane_ids_fixed: List[str]) -> Dict:
    """Build a SUMO state dict compatible with src.rl_agent.TrafficState."""
    sim_time = float(traci.simulation.getTime())

    lane_vehicle_counts: Dict[str, int] = {}
    lane_queue_lengths: Dict[str, int] = {}
    lane_mean_speeds: Dict[str, float] = {}

    for lane_id in lane_ids_fixed:
        if not lane_id:
            lane_vehicle_counts[lane_id] = 0
            lane_queue_lengths[lane_id] = 0
            lane_mean_speeds[lane_id] = 0.0
            continue
        try:
            lane_vehicle_counts[lane_id] = int(traci.lane.getLastStepVehicleNumber(lane_id))
            lane_queue_lengths[lane_id] = int(traci.lane.getLastStepHaltingNumber(lane_id))
            lane_mean_speeds[lane_id] = float(traci.lane.getLastStepMeanSpeed(lane_id))
        except Exception:
            lane_vehicle_counts[lane_id] = 0
            lane_queue_lengths[lane_id] = 0
            lane_mean_speeds[lane_id] = 0.0

    try:
        phase = int(traci.trafficlight.getPhase(tl_id)) % 8
    except Exception:
        phase = 0

    try:
        remaining = float(traci.trafficlight.getNextSwitch(tl_id) - sim_time)
        remaining = float(max(0.0, remaining))
    except Exception:
        remaining = 0.0

    return {
        "time": sim_time,
        "lane_vehicle_counts": lane_vehicle_counts,
        "lane_queue_lengths": lane_queue_lengths,
        "lane_mean_speeds": lane_mean_speeds,
        "detector_occupancy": {},
        "traffic_light_phase": phase,
        "traffic_light_remaining_time": remaining,
    }


def local_reward(prev_state: Dict, curr_state: Dict, action: int) -> float:
    """Local reward: reduce queue; small penalty for switching."""
    prev_q = sum(prev_state.get("lane_queue_lengths", {}).values())
    curr_q = sum(curr_state.get("lane_queue_lengths", {}).values())
    improvement = prev_q - curr_q
    r = -0.1 * curr_q + 0.05 * improvement
    if action == 1:
        r -= 0.02
    return float(r)


def pad_or_truncate(lanes: List[str], k: int) -> List[str]:
    lanes = [l for l in lanes if l]
    lanes = lanes[:k]
    if len(lanes) < k:
        lanes = lanes + [""] * (k - len(lanes))
    return lanes


def build_region_map(tls_ids: List[str], grid_size: float) -> Dict[str, Tuple[int, int]]:
    """Assign each TLS to a spatial grid cell for regional congestion shaping."""
    region_map: Dict[str, Tuple[int, int]] = {}
    if grid_size <= 0.0:
        grid_size = 500.0
    for tl_id in tls_ids:
        try:
            x, y = traci.trafficlight.getPosition(tl_id)
            rx = int(float(x) // grid_size)
            ry = int(float(y) // grid_size)
            region_map[tl_id] = (rx, ry)
        except Exception:
            region_map[tl_id] = (0, 0)
    return region_map


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
            safe_key = str(k)
            safe[safe_key] = _to_json_safe(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="los_angeles", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument("--model", type=str, default="", help="Path to trained model (.pt)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--max-controlled-lights", type=int, default=3)
    parser.add_argument(
        "--controlled-lights-ratio",
        type=float,
        default=0.0,
        help="If >0, use this ratio of total TLS (overrides --max-controlled-lights). Example: 0.2 for 20%",
    )
    parser.add_argument("--lanes-per-tl", type=int, default=8)
    parser.add_argument(
        "--regional-reward-weight",
        type=float,
        default=0.0,
        help="If >0, add a regional congestion penalty to the reward (weight on avg queue in region)",
    )
    parser.add_argument("--region-grid-size", type=float, default=500.0)
    parser.add_argument(
        "--collect-emissions",
        action="store_true",
        help="Enable TraCI emission collection (slower, higher memory).",
    )
    parser.add_argument(
        "--sumo-emissions-output",
        action="store_true",
        help=(
            "Enable SUMO native --emission-output during evaluation (recommended for OSM maps). "
            "This allows total CO2/fuel/NOx/PMx to be parsed efficiently after the run."
        ),
    )
    parser.add_argument(
        "--keep-emissions-xml",
        action="store_true",
        help="Keep generated emissions_*.xml files (default: delete after parsing to save disk space).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help=(
            "Optional run identifier to decorrelate SUMO seeds across repeated invocations. "
            "Effective SUMO seed used per episode is: seed + run_id*10000 + episode."
        ),
    )
    parser.add_argument("--out", type=str, default="", help="Optional JSON output for eval summary")
    parser.add_argument("--calc-metrics", action="store_true", help="Compute performance metrics (slower)")
    parser.add_argument(
        "--data-collection-interval",
        type=int,
        default=10,
        help="Collect detailed TraCI vehicle snapshots every N simulation steps (larger is faster).",
    )
    args = parser.parse_args()

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    if args.controlled_lights_ratio < 0.0 or args.controlled_lights_ratio > 1.0:
        raise ValueError("--controlled-lights-ratio must be in [0.0, 1.0]")

    np.random.seed(args.seed)

    SimulationConfig.set_dataset(args.dataset)
    if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = 10**9
    else:
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = int(args.max_controlled_lights)
    SimulationConfig.create_output_dirs()

    default_model = os.path.join(SimulationConfig.MODEL_DIR, f"marl_{args.dataset}_shared_dqn.pt")
    model_path = args.model.strip() or default_model
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path} (cwd={os.getcwd()})")

    results = []
    agent: RLAgent | None = None

    print("=" * 78)
    print(f"MARL evaluation: {args.dataset} | model={model_path}")
    print(f"Episodes={args.episodes} duration={args.duration}s interval={args.decision_interval}s")
    if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
        print(f"Agents(TLS)=ratio {args.controlled_lights_ratio:.2f} lanes_per_tl={args.lanes_per_tl}")
    else:
        print(f"Agents(TLS)={args.max_controlled_lights} lanes_per_tl={args.lanes_per_tl}")
    if args.regional_reward_weight and args.regional_reward_weight > 0.0:
        print(f"Regional reward: weight={args.regional_reward_weight} grid={args.region_grid_size}m")
    print("=" * 78)

    for ep in range(1, args.episodes + 1):
        # Make evaluation statistically meaningful:
        # vary SUMO random seed across episodes and across repeated invocations (run-id).
        sumo_seed = int(args.seed) + int(args.run_id) * 10000 + int(ep)
        simulator = TrafficSimulator(
            use_gui=False,
            dataset=args.dataset,
            enable_sumo_emissions_output=bool(args.sumo_emissions_output),
            sumo_seed=sumo_seed,
        )
        simulator.data_collection_interval = max(1, int(args.data_collection_interval))
        if args.collect_emissions:
            simulator.collect_emissions = True
            simulator.data_collection_interval = 1
        simulator.requested_duration = int(args.duration)
        ok = simulator.start_simulation()
        if not ok or not simulator.controlled_traffic_lights:
            simulator.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected.")

        if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
            total_tls = len(simulator.traffic_lights) if simulator.traffic_lights else len(simulator.controlled_traffic_lights)
            desired = int(round(total_tls * float(args.controlled_lights_ratio)))
            desired = max(1, min(total_tls, desired))
            tls_ids = (simulator.traffic_lights or simulator.controlled_traffic_lights)[:desired]
            if ep == 1:
                print(f"Computed TLS count: {desired}/{total_tls} ({args.controlled_lights_ratio:.2f})")
        else:
            tls_ids = simulator.controlled_traffic_lights[: int(args.max_controlled_lights)]

        tl_lanes: Dict[str, List[str]] = {}
        tl_phase_counts: Dict[str, int] = {}
        for tl_id in tls_ids:
            lanes = get_controlled_lanes_for_tl(tl_id)
            tl_lanes[tl_id] = pad_or_truncate(lanes, int(args.lanes_per_tl))
            tl_phase_counts[tl_id] = max(1, get_phase_count(tl_id))

        tl_regions: Dict[str, Tuple[int, int]] = {}
        if args.regional_reward_weight and args.regional_reward_weight > 0.0:
            tl_regions = build_region_map(tls_ids, float(args.region_grid_size))

        # Init agent once we know state_dim (fixed by lanes-per-tl)
        if agent is None:
            s0 = build_local_state(tls_ids[0], tl_lanes[tls_ids[0]])
            state_dim = len(RLTState(s0, lane_list=tl_lanes[tls_ids[0]]).to_vector())
            agent = RLAgent(config={
                "state_dim": state_dim,
                "action_dim": 2,
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 64,
                "memory_size": 100000,
                "target_update_freq": 500,
                "dueling": True,
                "double_dqn": True,
                "n_step": 3,
            })
            agent.load_model(model_path)

        start_wall = time.time()
        last_decision_t = 0.0
        ep_reward = 0.0
        decisions = 0

        prev_state_dict: Dict[str, Dict] = {
            tl_id: build_local_state(tl_id, tl_lanes[tl_id]) for tl_id in tls_ids
        }
        prev_state_obj: Dict[str, RLTState] = {
            tl_id: RLTState(prev_state_dict[tl_id], lane_list=tl_lanes[tl_id]) for tl_id in tls_ids
        }

        while True:
            if not simulator.simulation_step():
                break
            sim_t = float(simulator.simulation_time)
            done = sim_t >= float(args.duration)

            if done or (sim_t - last_decision_t) >= float(args.decision_interval):
                last_decision_t = sim_t

                actions: Dict[str, int] = {}
                for tl_id in tls_ids:
                    actions[tl_id] = agent.select_action(prev_state_obj[tl_id], training=False)

                for tl_id, action in actions.items():
                    if action != 1:
                        continue
                    try:
                        cur = int(traci.trafficlight.getPhase(tl_id))
                        nph = int(tl_phase_counts.get(tl_id, 1))
                        traci.trafficlight.setPhase(tl_id, (cur + 1) % max(1, nph))
                    except Exception:
                        pass

                curr_state_dict: Dict[str, Dict] = {}
                curr_state_obj: Dict[str, RLTState] = {}
                for tl_id in tls_ids:
                    curr_dict = build_local_state(tl_id, tl_lanes[tl_id])
                    curr_state_dict[tl_id] = curr_dict
                    curr_state_obj[tl_id] = RLTState(curr_dict, lane_list=tl_lanes[tl_id])

                region_sum: Dict[Tuple[int, int], float] = {}
                region_cnt: Dict[Tuple[int, int], int] = {}
                if tl_regions:
                    for tl_id in tls_ids:
                        region = tl_regions.get(tl_id, (0, 0))
                        q = float(sum(curr_state_dict[tl_id].get("lane_queue_lengths", {}).values()))
                        region_sum[region] = region_sum.get(region, 0.0) + q
                        region_cnt[region] = region_cnt.get(region, 0) + 1

                for tl_id in tls_ids:
                    r = local_reward(prev_state_dict[tl_id], curr_state_dict[tl_id], actions[tl_id])
                    if tl_regions:
                        region = tl_regions.get(tl_id, (0, 0))
                        denom = max(1, region_cnt.get(region, 1))
                        region_avg_q = region_sum.get(region, 0.0) / float(denom)
                        r += -float(args.regional_reward_weight) * region_avg_q
                    ep_reward += float(r)
                    decisions += 1

                    prev_state_dict[tl_id] = curr_state_dict[tl_id]
                    prev_state_obj[tl_id] = curr_state_obj[tl_id]

            if done:
                break

        # Important for emissions correctness:
        # SUMO writes emission-output XML incrementally; ensure SUMO is closed before parsing,
        # otherwise the XML can be incomplete and iterparse may stop early.
        simulator.close_simulation()

        metrics = {}
        if args.calc_metrics:
            try:
                # Give the OS a brief moment to flush file buffers after SUMO exits.
                if args.sumo_emissions_output:
                    try:
                        time.sleep(0.25)
                    except Exception:
                        pass

                metrics = simulator._calculate_performance_metrics()

                # If we enabled SUMO native emissions output, delete the huge XML after parsing
                # (unless user asked to keep it).
                try:
                    if args.sumo_emissions_output and (not args.keep_emissions_xml):
                        ef = getattr(simulator, "emissions_output_file", None)
                        if ef and os.path.exists(ef):
                            os.remove(ef)
                except Exception:
                    pass
            except Exception:
                metrics = {}
        wall = time.time() - start_wall
        avg_r = ep_reward / float(max(1, decisions))
        print(f"Episode {ep:04d}/{args.episodes} | wall={wall:.2f}s | reward(sum)={ep_reward:.2f} | avg_r={avg_r:.4f} | decisions={decisions}")
        results.append({
            "episode": ep,
            "sumo_seed": int(sumo_seed),
            "wall_seconds": float(wall),
            "reward_sum": float(ep_reward),
            "avg_reward": float(avg_r),
            "decisions": int(decisions),
            "metrics": metrics,
        })

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe({
                "dataset": args.dataset,
                "model": model_path,
                "episodes": args.episodes,
                "duration": args.duration,
                "decision_interval": args.decision_interval,
                "lanes_per_tl": args.lanes_per_tl,
                "controlled_lights_ratio": args.controlled_lights_ratio,
                "max_controlled_lights": args.max_controlled_lights,
                "regional_reward_weight": args.regional_reward_weight,
                "region_grid_size": args.region_grid_size,
                "seed": int(args.seed),
                "run_id": int(args.run_id),
                "results": results,
            }), f, indent=2)
        print(f"Saved eval summary: {args.out}")


if __name__ == "__main__":
    main()
