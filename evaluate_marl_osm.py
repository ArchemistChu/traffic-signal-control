#!/usr/bin/env python3
"""
Evaluate a trained MARL shared DQN on OSM maps (Los Angeles / Vancouver / Cologne).

Supports both the legacy 2-action models (keep/next) and the newer N-action
models (keep + jump-to-green-phase). The action dimension is read from the
loaded checkpoint so the correct mapping is applied automatically.
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


MAX_GREEN_PHASES = 6
PRESSURE_FEATURE_COUNT = 4
PRESSURE_FEATURE_SCALE = 20.0
PRESSURE_TOTAL_QUEUE_SCALE = 40.0
PRESSURE_REWARD_WEIGHT = 0.20
PRESSURE_REWARD_SCALE = 15.0

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


def get_phase_lane_map(tl_id: str) -> Dict:
    """Return green phases and the lanes they serve for one TLS."""
    def _is_ped(lane_id: str) -> bool:
        if not lane_id:
            return False
        low = lane_id.lower()
        return any(k in low for k in [":w", "ped", "walk", "sidewalk", "foot", "crossing"])

    result = {"green_phases": [], "phase_to_lanes": {}}
    try:
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if not logics:
            phase_count = max(1, get_phase_count(tl_id))
            result["green_phases"] = list(range(phase_count))
            result["phase_to_lanes"] = {p: [] for p in result["green_phases"]}
            return result
        links = traci.trafficlight.getControlledLinks(tl_id)
        phases = logics[0].phases
        for p_idx, phase in enumerate(phases):
            state_str = phase.state
            has_vehicle_green = False
            phase_lanes = set()
            for li, ch in enumerate(state_str):
                if ch in ("G", "g"):
                    if li < len(links):
                        for link in links[li]:
                            if link and len(link) > 0 and not _is_ped(link[0]):
                                has_vehicle_green = True
                                phase_lanes.add(link[0])
                                break
                    else:
                        has_vehicle_green = True
            if has_vehicle_green:
                result["green_phases"].append(int(p_idx))
            result["phase_to_lanes"][int(p_idx)] = sorted(phase_lanes)
        if not result["green_phases"]:
            result["green_phases"] = list(range(max(1, len(phases))))
            for p_idx in result["green_phases"]:
                result["phase_to_lanes"].setdefault(int(p_idx), [])
        return result
    except Exception:
        phase_count = max(1, get_phase_count(tl_id))
        result["green_phases"] = list(range(phase_count))
        result["phase_to_lanes"] = {p: [] for p in result["green_phases"]}
        return result


def get_green_phases(tl_id: str) -> List[int]:
    """Return indices of phases that give green to at least one vehicle lane."""
    return list(get_phase_lane_map(tl_id).get("green_phases", []))


def get_reference_green_phase(current_phase: int, green_phases: List[int]) -> int:
    """Map a raw SUMO phase to the most recent vehicle-green phase."""
    if not green_phases:
        return int(current_phase)
    ordered = sorted({int(p) for p in green_phases})
    if current_phase in ordered:
        return int(current_phase)
    prior = [p for p in ordered if p <= int(current_phase)]
    return int(prior[-1] if prior else ordered[-1])


def get_next_green_phase(current_phase: int, green_phases: List[int], phase_count: int) -> int:
    """Cycle to the next vehicle-green phase while skipping yellow/all-red phases."""
    if not green_phases:
        return (int(current_phase) + 1) % max(1, int(phase_count))
    ordered = sorted({int(p) for p in green_phases})
    if current_phase in ordered:
        idx = ordered.index(int(current_phase))
        return int(ordered[(idx + 1) % len(ordered)])
    for phase in ordered:
        if phase > int(current_phase):
            return int(phase)
    return int(ordered[0])


def compute_phase_pressure(state_dict: Dict, phase_lanes: List[str]) -> float:
    """Simple local pressure proxy based on queued vehicles on served lanes."""
    lane_queues = state_dict.get("lane_queue_lengths", {})
    return float(sum(float(lane_queues.get(lane_id, 0.0)) for lane_id in phase_lanes if lane_id))


def add_pressure_features(
    state_dict: Dict,
    phase_to_lanes: Dict[int, List[str]],
    green_phases: List[int],
    phase_count: int,
    feature_scale: float = PRESSURE_FEATURE_SCALE,
    total_queue_scale: float = PRESSURE_TOTAL_QUEUE_SCALE,
) -> Dict:
    """Append pressure-aware features without changing legacy state fields."""
    out = dict(state_dict)
    current_phase = int(state_dict.get("traffic_light_phase", 0))
    current_green = get_reference_green_phase(current_phase, green_phases)
    next_green = get_next_green_phase(current_green, green_phases, phase_count)

    current_pressure = compute_phase_pressure(state_dict, phase_to_lanes.get(current_green, []))
    next_pressure = compute_phase_pressure(state_dict, phase_to_lanes.get(next_green, []))
    total_queue = float(sum(state_dict.get("lane_queue_lengths", {}).values()))

    out["extra_features"] = [
        float(np.clip(current_pressure / max(1e-6, float(feature_scale)), 0.0, 1.0)),
        float(np.clip(next_pressure / max(1e-6, float(feature_scale)), 0.0, 1.0)),
        float(np.clip((next_pressure - current_pressure) / max(1e-6, float(feature_scale)), -1.0, 1.0)),
        float(np.clip(total_queue / max(1e-6, float(total_queue_scale)), 0.0, 1.0)),
    ]
    return out


def get_max_phase_pressure(state_dict: Dict, phase_to_lanes: Dict[int, List[str]], green_phases: List[int]) -> float:
    """Maximum queue pressure over available vehicle-green phases."""
    if not green_phases:
        return float(sum(state_dict.get("lane_queue_lengths", {}).values()))
    return float(max(
        compute_phase_pressure(state_dict, phase_to_lanes.get(int(phase), []))
        for phase in green_phases
    ))


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


def get_lane_waiting_time(lane_id: str) -> float:
    """Sum of waiting times of all vehicles on a lane."""
    try:
        total = 0.0
        for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
            total += float(traci.vehicle.getWaitingTime(veh_id))
        return total
    except Exception:
        return 0.0


def build_local_state(tl_id: str, lane_ids_fixed: List[str]) -> Dict:
    """Build a SUMO state dict compatible with src.rl_agent.TrafficState."""
    sim_time = float(traci.simulation.getTime())

    lane_vehicle_counts: Dict[str, int] = {}
    lane_queue_lengths: Dict[str, int] = {}
    lane_mean_speeds: Dict[str, float] = {}
    lane_waiting_times: Dict[str, float] = {}

    for lane_id in lane_ids_fixed:
        if not lane_id:
            lane_vehicle_counts[lane_id] = 0
            lane_queue_lengths[lane_id] = 0
            lane_mean_speeds[lane_id] = 0.0
            lane_waiting_times[lane_id] = 0.0
            continue
        try:
            lane_vehicle_counts[lane_id] = int(traci.lane.getLastStepVehicleNumber(lane_id))
            lane_queue_lengths[lane_id] = int(traci.lane.getLastStepHaltingNumber(lane_id))
            lane_mean_speeds[lane_id] = float(traci.lane.getLastStepMeanSpeed(lane_id))
            lane_waiting_times[lane_id] = get_lane_waiting_time(lane_id)
        except Exception:
            lane_vehicle_counts[lane_id] = 0
            lane_queue_lengths[lane_id] = 0
            lane_mean_speeds[lane_id] = 0.0
            lane_waiting_times[lane_id] = 0.0

    try:
        phase = int(traci.trafficlight.getPhase(tl_id)) % 16
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
        "lane_waiting_times": lane_waiting_times,
        "detector_occupancy": {},
        "traffic_light_phase": phase,
        "traffic_light_remaining_time": remaining,
    }


def local_reward(
    prev_state: Dict,
    curr_state: Dict,
    action: int,
    prev_action: int = 0,
    n_green_phases: int = 2,
    phase_to_lanes: Dict[int, List[str]] | None = None,
    green_phases: List[int] | None = None,
    wait_penalty_weight: float = 0.50,
    wait_penalty_scale: float = 200.0,
    queue_reward_weight: float = 0.35,
    queue_reward_scale: float = 10.0,
    switch_penalty: float = 0.05,
    pressure_reward_weight: float = PRESSURE_REWARD_WEIGHT,
    pressure_reward_scale: float = PRESSURE_REWARD_SCALE,
) -> float:
    """Reward aligned with training — targets waiting time + queue reduction."""
    curr_wait = sum(curr_state.get("lane_waiting_times", {}).values())
    curr_q = sum(curr_state.get("lane_queue_lengths", {}).values())
    prev_q = sum(prev_state.get("lane_queue_lengths", {}).values())
    queue_change = prev_q - curr_q

    r_wait = -float(wait_penalty_weight) * np.clip(curr_wait / max(1e-6, float(wait_penalty_scale)), 0.0, 1.0)
    r_queue = float(queue_reward_weight) * np.clip(queue_change / max(1e-6, float(queue_reward_scale)), -1.0, 1.0)
    r_switch = -float(switch_penalty) if action == 1 else 0.0
    r_pressure = 0.0
    if phase_to_lanes and green_phases:
        prev_max_pressure = get_max_phase_pressure(prev_state, phase_to_lanes, green_phases)
        curr_max_pressure = get_max_phase_pressure(curr_state, phase_to_lanes, green_phases)
        pressure_delta = prev_max_pressure - curr_max_pressure
        r_pressure = float(pressure_reward_weight) * np.clip(
            pressure_delta / max(1e-6, float(pressure_reward_scale)), -1.0, 1.0
        )
    return float(r_wait + r_queue + r_switch + r_pressure)


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


def sync_phase_timer(
    tl_id: str,
    current_phase: int,
    sim_t: float,
    tl_last_phase: Dict[str, int],
    tl_phase_start_time: Dict[str, float],
) -> float:
    """Track how long the current raw SUMO phase has been active."""
    last_phase = tl_last_phase.get(tl_id)
    if last_phase is None or int(last_phase) != int(current_phase):
        tl_last_phase[tl_id] = int(current_phase)
        tl_phase_start_time[tl_id] = float(sim_t)
    phase_start = float(tl_phase_start_time.get(tl_id, sim_t))
    return float(max(0.0, float(sim_t) - phase_start))


def build_switch_action_mask(
    current_phase: int,
    phase_elapsed: float,
    green_phases: List[int],
    phase_count: int,
    *,
    mask_invalid_switches: bool,
    min_green_seconds: float,
) -> List[bool]:
    """Mask the switch action unless it is actually meaningful to take."""
    can_switch = int(phase_count) > 1
    if mask_invalid_switches:
        green_set = {int(phase) for phase in green_phases}
        can_switch = can_switch and (int(current_phase) in green_set) and (float(phase_elapsed) >= float(min_green_seconds))
    return [True, bool(can_switch)]


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
        help="If >0, use this ratio of total TLS (overrides --max-controlled-lights). Example: 0.2 for 20%%",
    )
    parser.add_argument("--lanes-per-tl", type=int, default=20)
    parser.add_argument("--demand-scale", type=float, default=0.0, help="SUMO --scale for traffic demand (0=default, 0.7=70%%)")
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
    parser.add_argument("--port", type=int, default=8813, help="TraCI port for SUMO connection")
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help=(
            "Optional run identifier to decorrelate SUMO seeds across repeated invocations. "
            "Effective SUMO seed used per episode is: seed + run_id*10000 + episode."
        ),
    )
    parser.add_argument("--out", type=str, default="", help="Optional JSON output for eval summary")
    parser.add_argument("--calc-metrics", action="store_true", default=True, help="Compute performance metrics (default: on)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if CUDA is available")
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
    is_n_action = False
    switch_mode = "legacy_cycle"
    use_pressure_state_features = False
    pressure_reward_weight_local = PRESSURE_REWARD_WEIGHT
    pressure_reward_scale_local = PRESSURE_REWARD_SCALE
    pressure_feature_scale_local = PRESSURE_FEATURE_SCALE
    pressure_total_queue_scale_local = PRESSURE_TOTAL_QUEUE_SCALE
    wait_penalty_weight_local = 0.50
    wait_penalty_scale_local = 200.0
    queue_reward_weight_local = 0.35
    queue_reward_scale_local = 10.0
    switch_penalty_local = 0.05
    regional_penalty_clip_local = 0.3
    mask_invalid_switches = False
    min_green_seconds = 0.0

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
            port=args.port,
            dataset=args.dataset,
            enable_sumo_emissions_output=bool(args.sumo_emissions_output),
            sumo_seed=sumo_seed,
            controlled_lights_ratio=float(args.controlled_lights_ratio) if args.controlled_lights_ratio > 0 else None,
            demand_scale=float(args.demand_scale) if args.demand_scale > 0 else None,
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
        tl_green_phases: Dict[str, List[int]] = {}
        tl_phase_lanes: Dict[str, Dict[int, List[str]]] = {}
        for tl_id in tls_ids:
            lanes = get_controlled_lanes_for_tl(tl_id)
            tl_lanes[tl_id] = pad_or_truncate(lanes, int(args.lanes_per_tl))
            tl_phase_counts[tl_id] = max(1, get_phase_count(tl_id))
            phase_map = get_phase_lane_map(tl_id)
            gp = list(phase_map.get("green_phases", [])) or list(range(tl_phase_counts[tl_id]))
            tl_green_phases[tl_id] = gp
            tl_phase_lanes[tl_id] = {
                int(phase): list(lanes_for_phase)
                for phase, lanes_for_phase in phase_map.get("phase_to_lanes", {}).items()
            }
            if ep == 1:
                print(f"  TL {tl_id}: {len(gp)} green phases {gp[:8]}")

        tl_regions: Dict[str, Tuple[int, int]] = {}
        if args.regional_reward_weight and args.regional_reward_weight > 0.0:
            tl_regions = build_region_map(tls_ids, float(args.region_grid_size))

        # Init agent once we know state_dim; read action_dim from checkpoint
        if agent is None:
            import torch as _torch
            ckpt = _torch.load(model_path, map_location="cpu")
            ckpt_cfg = ckpt.get("config", {})
            ckpt_action_dim = ckpt_cfg.get("action_dim", 2)
            is_n_action = ckpt_action_dim > 2
            switch_mode = str(ckpt_cfg.get("switch_mode", "legacy_cycle"))
            use_pressure_state_features = bool(ckpt_cfg.get("pressure_state_features", False))
            pressure_reward_weight_local = float(ckpt_cfg.get("pressure_reward_weight", PRESSURE_REWARD_WEIGHT))
            pressure_reward_scale_local = float(ckpt_cfg.get("pressure_reward_scale", PRESSURE_REWARD_SCALE))
            pressure_feature_scale_local = float(ckpt_cfg.get("pressure_feature_scale", PRESSURE_FEATURE_SCALE))
            pressure_total_queue_scale_local = float(ckpt_cfg.get("pressure_total_queue_scale", PRESSURE_TOTAL_QUEUE_SCALE))
            wait_penalty_weight_local = float(ckpt_cfg.get("wait_penalty_weight", 0.50))
            wait_penalty_scale_local = float(ckpt_cfg.get("wait_penalty_scale", 200.0))
            queue_reward_weight_local = float(ckpt_cfg.get("queue_reward_weight", 0.35))
            queue_reward_scale_local = float(ckpt_cfg.get("queue_reward_scale", 10.0))
            switch_penalty_local = float(ckpt_cfg.get("switch_penalty", 0.05))
            regional_penalty_clip_local = float(ckpt_cfg.get("regional_penalty_clip", 0.3))
            mask_invalid_switches = bool(ckpt_cfg.get("mask_invalid_switches", False))
            min_green_seconds = float(ckpt_cfg.get("min_green_seconds", 0.0))
            mode_label = "N-action" if is_n_action else ("2-action next-green" if switch_mode == "next_green" else "legacy 2-action")
            print(
                f"Model action_dim={ckpt_action_dim} ({mode_label}) | pressure_state={use_pressure_state_features} "
                f"| masked_switch={mask_invalid_switches} | min_green={min_green_seconds:.1f}s"
            )
            print(
                "Reward config: "
                f"wait=-{wait_penalty_weight_local:.2f}*clip(wait/{wait_penalty_scale_local:.1f}) "
                f"queue=+{queue_reward_weight_local:.2f}*clip(dq/{queue_reward_scale_local:.1f}) "
                f"switch=-{switch_penalty_local:.2f} "
                f"region_clip={regional_penalty_clip_local:.2f}"
            )

            first_tl = tls_ids[0]
            s0 = build_local_state(first_tl, tl_lanes[first_tl])
            if use_pressure_state_features:
                s0 = add_pressure_features(
                    s0,
                    tl_phase_lanes[first_tl],
                    tl_green_phases[first_tl],
                    tl_phase_counts[first_tl],
                    feature_scale=pressure_feature_scale_local,
                    total_queue_scale=pressure_total_queue_scale_local,
                )
            state_dim = len(RLTState(s0, lane_list=tl_lanes[tls_ids[0]]).to_vector())
            agent_config = dict(ckpt_cfg)
            agent_config["state_dim"] = state_dim
            agent_config["action_dim"] = ckpt_action_dim
            if args.force_cpu:
                agent_config["device"] = "cpu"
            agent = RLAgent(config=agent_config)
            agent.load_model(model_path)

        start_wall = time.time()
        last_decision_t = 0.0
        ep_reward = 0.0
        decisions = 0
        prev_actions: Dict[str, int] = {tl_id: 0 for tl_id in tls_ids}
        tl_last_phase: Dict[str, int] = {}
        tl_phase_start_time: Dict[str, float] = {}
        for tl_id in tls_ids:
            try:
                tl_last_phase[tl_id] = int(traci.trafficlight.getPhase(tl_id))
            except Exception:
                tl_last_phase[tl_id] = 0
            tl_phase_start_time[tl_id] = float(simulator.simulation_time)

        prev_state_dict: Dict[str, Dict] = {}
        for tl_id in tls_ids:
            state_dict = build_local_state(tl_id, tl_lanes[tl_id])
            if use_pressure_state_features:
                state_dict = add_pressure_features(
                    state_dict,
                    tl_phase_lanes[tl_id],
                    tl_green_phases[tl_id],
                    tl_phase_counts[tl_id],
                    feature_scale=pressure_feature_scale_local,
                    total_queue_scale=pressure_total_queue_scale_local,
                )
            prev_state_dict[tl_id] = state_dict
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
                action_masks: Dict[str, List[bool]] = {}
                for tl_id in tls_ids:
                    try:
                        current_phase = int(traci.trafficlight.getPhase(tl_id))
                    except Exception:
                        current_phase = 0
                    phase_elapsed = sync_phase_timer(
                        tl_id,
                        current_phase,
                        sim_t,
                        tl_last_phase,
                        tl_phase_start_time,
                    )
                    action_mask = None
                    if not is_n_action:
                        action_mask = build_switch_action_mask(
                            current_phase,
                            phase_elapsed,
                            tl_green_phases.get(tl_id, []),
                            tl_phase_counts.get(tl_id, 1),
                            mask_invalid_switches=mask_invalid_switches,
                            min_green_seconds=min_green_seconds,
                        )
                    action_masks[tl_id] = action_mask or [True] * int(agent.config.get("action_dim", 2))
                    actions[tl_id] = agent.select_action(
                        prev_state_obj[tl_id],
                        training=False,
                        action_mask=action_mask,
                    )

                # Apply actions — support legacy 2-action, next-green 2-action, and N-action.
                for tl_id, action in actions.items():
                    if is_n_action:
                        if action == 0:
                            continue
                        gp = tl_green_phases.get(tl_id, [])
                        if not gp:
                            continue
                        phase_idx = (action - 1) % len(gp)
                        target_phase = gp[phase_idx]
                        try:
                            cur = int(traci.trafficlight.getPhase(tl_id))
                            if target_phase != cur:
                                traci.trafficlight.setPhase(tl_id, target_phase)
                        except Exception:
                            pass
                    elif switch_mode == "next_green":
                        try:
                            cur = int(traci.trafficlight.getPhase(tl_id))
                            gp = tl_green_phases.get(tl_id, [])
                            nph = int(tl_phase_counts.get(tl_id, 1))
                            if action == 0 or not action_masks.get(tl_id, [True, True])[1]:
                                if cur in gp:
                                    traci.trafficlight.setPhaseDuration(tl_id, float(args.decision_interval))
                                continue
                            target_phase = get_next_green_phase(cur, gp, nph)
                            if target_phase != cur:
                                traci.trafficlight.setPhase(tl_id, target_phase)
                            traci.trafficlight.setPhaseDuration(tl_id, float(args.decision_interval))
                        except Exception:
                            pass
                    else:
                        if action == 0 or not action_masks.get(tl_id, [True, True])[1]:
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
                    if use_pressure_state_features:
                        curr_dict = add_pressure_features(
                            curr_dict,
                            tl_phase_lanes[tl_id],
                            tl_green_phases[tl_id],
                            tl_phase_counts[tl_id],
                            feature_scale=pressure_feature_scale_local,
                            total_queue_scale=pressure_total_queue_scale_local,
                        )
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
                    gp = tl_green_phases.get(tl_id, [])
                    r = local_reward(
                        prev_state_dict[tl_id], curr_state_dict[tl_id],
                        actions[tl_id], prev_actions.get(tl_id, 0),
                        n_green_phases=len(gp),
                        phase_to_lanes=tl_phase_lanes.get(tl_id, {}) if use_pressure_state_features else None,
                        green_phases=gp if use_pressure_state_features else None,
                        wait_penalty_weight=wait_penalty_weight_local,
                        wait_penalty_scale=wait_penalty_scale_local,
                        queue_reward_weight=queue_reward_weight_local,
                        queue_reward_scale=queue_reward_scale_local,
                        switch_penalty=switch_penalty_local,
                        pressure_reward_weight=pressure_reward_weight_local,
                        pressure_reward_scale=pressure_reward_scale_local,
                    )
                    if tl_regions:
                        region = tl_regions.get(tl_id, (0, 0))
                        denom = max(1, region_cnt.get(region, 1))
                        region_avg_q = region_sum.get(region, 0.0) / float(denom)
                        regional_penalty = -float(args.regional_reward_weight) * region_avg_q
                        if regional_penalty_clip_local > 0.0:
                            r += float(np.clip(regional_penalty, -regional_penalty_clip_local, 0.0))
                        else:
                            r += float(regional_penalty)
                    ep_reward += float(r)
                    decisions += 1

                    prev_state_dict[tl_id] = curr_state_dict[tl_id]
                    prev_state_obj[tl_id] = curr_state_obj[tl_id]

                prev_actions = dict(actions)

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
