#!/usr/bin/env python3
"""
Multi-agent RL training on OSM Los Angeles (parameter-sharing DQN).

What this is:
- Multi-agent: control N traffic lights at the same time.
- Parameter sharing: all agents use ONE shared DQN (shared weights + shared replay).
  This is the most practical MARL baseline for a final-year project.

Key simplifications (intentional):
- Local observation per traffic light (only lanes controlled by that TLS).
- Fixed observation size via lane padding/truncation (lanes_per_tl).
- Action space = 2:
    0 = keep current phase
    1 = switch to next SUMO phase (cycle)
- Headless SUMO for speed; disable emission-output during training to avoid huge files.
  (You can evaluate emissions later with your normal simulator runs.)
"""

import argparse
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator
from src.rl_agent import RLAgent, TrafficState as RLTState

try:
    import traci
except ImportError:
    traci = None


MAX_GREEN_PHASES = 6  # kept for checkpoint metadata compatibility
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
                    if li >= len(links):
                        break
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

    # Fallback: parse controlled links
    lanes = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        for link_list in controlled_links:
            for link in link_list:
                if link and len(link) > 0:
                    lanes.add(link[0])  # incoming lane
    except Exception:
        pass
    return sorted(lanes)


def get_lane_waiting_time(lane_id: str) -> float:
    """Lane-level waiting time with a fast TraCI path first."""
    if not lane_id:
        return 0.0
    try:
        if hasattr(traci.lane, "getWaitingTime"):
            return float(traci.lane.getWaitingTime(lane_id))
    except Exception:
        pass
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
    wait_penalty_scale: float = 200.0,
    queue_reward_scale: float = 10.0,
    switch_penalty: float = 0.05,
) -> float:
    """Simple 3-term reward: waiting time penalty + queue delta + switch penalty."""
    curr_wait = sum(curr_state.get("lane_waiting_times", {}).values())
    curr_q = sum(curr_state.get("lane_queue_lengths", {}).values())
    prev_q = sum(prev_state.get("lane_queue_lengths", {}).values())
    queue_change = prev_q - curr_q

    r_wait = -0.50 * np.clip(curr_wait / wait_penalty_scale, 0.0, 1.0)
    r_queue = 0.35 * np.clip(queue_change / queue_reward_scale, -1.0, 1.0)
    r_switch = -switch_penalty if action == 1 else 0.0

    return float(r_wait + r_queue + r_switch)


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


def build_checkpoint_path(out_path: str, episode: int) -> str:
    """Create a numbered checkpoint path alongside the latest checkpoint."""
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".pt"
    return f"{root}_ep{int(episode):04d}{ext}"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="los_angeles", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--duration", type=int, default=1200, help="Seconds per episode")
    parser.add_argument("--decision-interval", type=int, default=5, help="Seconds between actions per TLS")
    parser.add_argument("--max-controlled-lights", type=int, default=3, help="How many TLS agents to train simultaneously")
    parser.add_argument(
        "--controlled-lights-ratio",
        type=float,
        default=0.0,
        help="If >0, use this ratio of total TLS (overrides --max-controlled-lights). Example: 0.2 for 20%%",
    )
    parser.add_argument(
        "--regional-reward-weight",
        type=float,
        default=0.01,
        help=(
            "Region-aware reward shaping. If >0, add a regional congestion penalty to each agent reward: "
            "reward += -weight * (avg queued vehicles in the agent's region). "
            "Typical values: 0.005–0.05 (start small). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--region-grid-size",
        type=float,
        default=500.0,
        help="Grid size in meters for regional congestion (default: 500)",
    )
    parser.add_argument("--lanes-per-tl", type=int, default=20, help="Fixed lanes per TLS in observation (pad/truncate)")
    parser.add_argument("--demand-scale", type=float, default=0.0, help="SUMO --scale for traffic demand (0=default, 1.0=100%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--traci-port",
        type=int,
        default=8813,
        help="Base TraCI TCP port. Each episode uses (base + episode_index - 1) so SUMO can restart cleanly.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="", help="Model output path")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--aggressive-upgrade",
        action="store_true",
        help="Enable the higher-upside preset without auto-enabling prioritized replay",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Minimum learning rate for cosine scheduling")
    parser.add_argument("--constant-lr", action="store_true", help="Disable LR scheduling and keep a fixed learning rate")
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=200000)
    parser.add_argument("--target-update-freq", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=0,
        help=(
            "If >0, epsilon reaches epsilon_end after this many episodes "
            "(instead of decaying over the full run).  Recommended: 40-60%% of total episodes."
        ),
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--target-q-clip", type=float, default=2.0)
    parser.add_argument("--soft-target-updates", action="store_true")
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-start", type=float, default=0.4)
    parser.add_argument("--priority-beta-end", type=float, default=1.0)
    parser.add_argument("--priority-beta-steps", type=int, default=100000)
    parser.add_argument("--use-pressure-state-features", action="store_true")
    parser.add_argument("--pressure-feature-scale", type=float, default=PRESSURE_FEATURE_SCALE)
    parser.add_argument("--pressure-total-queue-scale", type=float, default=PRESSURE_TOTAL_QUEUE_SCALE)
    parser.add_argument("--mask-invalid-switches", action="store_true")
    parser.add_argument("--min-green-seconds", type=float, default=0.0)
    parser.add_argument("--wait-penalty-scale", type=float, default=200.0)
    parser.add_argument("--queue-reward-scale", type=float, default=10.0)
    parser.add_argument("--switch-penalty", type=float, default=0.05)
    parser.add_argument(
        "--regional-penalty-clip",
        type=float,
        default=0.3,
        help="Maximum magnitude of the negative regional penalty added each step (0 disables clipping).",
    )
    parser.add_argument(
        "--curriculum-ratio-start",
        type=float,
        default=0.0,
        help=(
            "If >0 and < controlled ratio, linearly ramp controlled_lights_ratio from this value "
            "to --controlled-lights-ratio over --curriculum-ramp-episodes."
        ),
    )
    parser.add_argument(
        "--curriculum-ramp-episodes",
        type=int,
        default=120,
        help="Episodes used to ramp ratio curriculum (when enabled).",
    )
    parser.add_argument("--plateau-window", type=int, default=20, help="MA window for plateau detection.")
    parser.add_argument("--plateau-patience", type=int, default=15, help="Episodes with no MA improvement before recovery.")
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3, help="Minimum MA improvement to count as progress.")
    parser.add_argument("--plateau-lr-decay", type=float, default=0.7, help="Multiply LR by this factor on plateau.")
    parser.add_argument("--plateau-min-lr", type=float, default=5e-5, help="LR floor for plateau recovery.")
    parser.add_argument("--plateau-epsilon-boost", type=float, default=0.03, help="Temporary epsilon boost on plateau.")
    parser.add_argument(
        "--auto-scale-for-ratio",
        action="store_true",
        help=(
            "Automatically adjust gamma, n_step, and enable pressure state features "
            "based on controlled-lights-ratio."
        ),
    )
    args = parser.parse_args()

    # --- Auto-scaling for high controlled-lights-ratio ---
    if args.auto_scale_for_ratio and args.controlled_lights_ratio >= 0.25:
        ratio = float(args.controlled_lights_ratio)
        scale_factor = ratio / 0.2

        if args.target_q_clip == 2.0:
            args.target_q_clip = min(10.0, 2.0 * scale_factor)

        if args.memory_size == 200000:
            args.memory_size = int(min(800000, 200000 * scale_factor))

        if args.learning_starts == 0:
            args.learning_starts = int(min(50000, 5000 * scale_factor))

        if args.regional_penalty_clip == 0.15 or args.regional_penalty_clip == 0.3:
            args.regional_penalty_clip = float(
                np.clip(args.regional_penalty_clip / scale_factor, 0.02, 0.15)
            )

        if args.regional_reward_weight > 0:
            args.regional_reward_weight = float(
                np.clip(args.regional_reward_weight / scale_factor, 1e-4, 0.01)
            )

        if args.curriculum_ratio_start == 0.0 and ratio > 0.3:
            args.curriculum_ratio_start = 0.2
            if args.curriculum_ramp_episodes == 120:
                args.curriculum_ramp_episodes = min(int(args.episodes // 5), 80)

        if args.epsilon_decay_episodes == 0:
            args.epsilon_decay_episodes = max(60, int(args.episodes * 0.45))

        if not args.use_pressure_state_features:
            args.use_pressure_state_features = True

        if args.gamma <= 0.90:
            args.gamma = 0.97

        if args.n_step == 1:
            args.n_step = 3

    if args.aggressive_upgrade:
        args.use_pressure_state_features = True
        args.mask_invalid_switches = True
        args.soft_target_updates = True
        if args.lr == 5e-5:
            args.lr = 2e-4
        if args.lr_min == 1e-5:
            args.lr_min = 5e-5
        if args.gamma == 0.90:
            args.gamma = 0.97
        if args.memory_size == 200000:
            args.memory_size = 300000
        if args.n_step == 1:
            args.n_step = 3
        if args.learning_starts == 0:
            args.learning_starts = 10000
        if args.epsilon_end == 0.05:
            args.epsilon_end = 0.02
        if args.gradient_clip_norm == 0.5:
            args.gradient_clip_norm = 1.0
        if args.target_q_clip == 2.0:
            args.target_q_clip = 4.0
        if args.tau == 0.005:
            args.tau = 0.003
        if args.priority_beta_steps == 100000:
            args.priority_beta_steps = 200000
        if args.min_green_seconds <= 0.0:
            args.min_green_seconds = max(10.0, float(args.decision_interval) * 2.0)
        if args.wait_penalty_scale == 200.0:
            args.wait_penalty_scale = max(1000.0, float(args.lanes_per_tl) * 100.0)

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.controlled_lights_ratio < 0.0 or args.controlled_lights_ratio > 1.0:
        raise ValueError("--controlled-lights-ratio must be in [0.0, 1.0]")

    SimulationConfig.set_dataset(args.dataset)
    if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = 10**9
    else:
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = int(args.max_controlled_lights)
    SimulationConfig.create_output_dirs()

    suffix = ""
    try:
        if float(args.regional_reward_weight or 0.0) > 0.0:
            suffix = "_regionaware"
    except Exception:
        suffix = ""
    default_out = os.path.join(SimulationConfig.MODEL_DIR, f"marl_{args.dataset}_shared_dqn{suffix}.pt")
    out_path = args.out.strip() or default_out

    agent: RLAgent | None = None
    lr_scheduler = None

    print("=" * 78)
    print(f"MARL training: {args.dataset} | parameter-sharing DQN")
    print(f"Episodes={args.episodes} duration={args.duration}s interval={args.decision_interval}s")
    if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
        print(f"Agents(TLS)=ratio {args.controlled_lights_ratio:.2f} lanes_per_tl={args.lanes_per_tl}")
    else:
        print(f"Agents(TLS)={args.max_controlled_lights} lanes_per_tl={args.lanes_per_tl}")
    if args.regional_reward_weight and args.regional_reward_weight > 0.0:
        print(f"Regional reward: weight={args.regional_reward_weight} grid={args.region_grid_size}m")
    if args.use_pressure_state_features:
        print("Pressure state features: ON")
    if args.mask_invalid_switches:
        print(f"Action masking: ON (min_green={args.min_green_seconds}s)")
    if args.auto_scale_for_ratio:
        print("Auto-scale for ratio: ON")
    if args.curriculum_ratio_start > 0.0:
        print(
            f"Curriculum: ratio {args.curriculum_ratio_start:.2f} -> "
            f"{args.controlled_lights_ratio:.2f} over {args.curriculum_ramp_episodes} episodes"
        )
    eps_decay_ep = int(args.epsilon_decay_episodes) if args.epsilon_decay_episodes > 0 else int(args.episodes)
    print(
        f"Epsilon schedule: 1.0 -> {args.epsilon_end:.2f} over {eps_decay_ep} episodes "
        f"({'accelerated' if args.epsilon_decay_episodes > 0 else 'full-run'})"
    )
    print(f"Hyperparams: gamma={args.gamma} n_step={args.n_step} lr={args.lr}")
    print(f"Reward: wait_scale={args.wait_penalty_scale} queue_scale={args.queue_reward_scale} switch_pen={args.switch_penalty}")
    if args.demand_scale > 0:
        print(f"Demand scale: {args.demand_scale}")
    print(f"Output: {out_path}")
    print("=" * 78)

    resume_episode_count = 0
    resume_epsilon = None
    if args.resume and os.path.isfile(args.resume):
        try:
            resume_ckpt = torch.load(args.resume, map_location="cpu")
            resume_episode_count = int(resume_ckpt.get("episode_count", 0))
            resume_epsilon = float(resume_ckpt.get("epsilon", 1.0))
        except Exception as exc:
            print(f"Warning: could not read resume metadata from {args.resume}: {exc}")

    start_ep = resume_episode_count + 1
    total_episodes = resume_episode_count + int(args.episodes)
    epsilon_run_start = float(resume_epsilon) if resume_epsilon is not None else 1.0
    target_ratio = float(args.controlled_lights_ratio) if args.controlled_lights_ratio > 0 else 0.0
    curriculum_enabled = (
        target_ratio > 0.0
        and args.curriculum_ratio_start > 0.0
        and args.curriculum_ratio_start < target_ratio
    )
    best_reward_ma = float("-inf")
    episodes_without_improve = 0

    ep = start_ep - 1
    while ep < total_episodes:
        ep += 1
        sumo_seed = int(args.seed) + int(ep)
        run_ratio = target_ratio
        if curriculum_enabled:
            ramp_progress = float(ep - start_ep) / float(max(1, int(args.curriculum_ramp_episodes) - 1))
            ramp_progress = float(np.clip(ramp_progress, 0.0, 1.0))
            run_ratio = float(args.curriculum_ratio_start) + (target_ratio - float(args.curriculum_ratio_start)) * ramp_progress
        traci_port = int(args.traci_port) + int(ep) - 1
        simulator = TrafficSimulator(
            use_gui=False,
            port=traci_port,
            dataset=args.dataset,
            enable_sumo_emissions_output=False,
            sumo_seed=sumo_seed,
            controlled_lights_ratio=run_ratio if run_ratio > 0 else None,
            demand_scale=float(args.demand_scale) if args.demand_scale > 0 else None,
        )
        ok = simulator.start_simulation()
        if not ok or not simulator.controlled_traffic_lights:
            simulator.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected.")

        if run_ratio > 0.0:
            total_tls = len(simulator.traffic_lights) if simulator.traffic_lights else len(simulator.controlled_traffic_lights)
            desired = int(round(total_tls * float(run_ratio)))
            desired = max(1, min(total_tls, desired))
            tls_ids = (simulator.traffic_lights or simulator.controlled_traffic_lights)[:desired]
            if agent is None:
                print(f"Computed TLS count: {desired}/{total_tls} ({run_ratio:.2f})")
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
            if agent is None:
                print(f"  TL {tl_id}: {len(gp)} green phases {gp[:8]}")

        tl_regions: Dict[str, Tuple[int, int]] = {}
        if args.regional_reward_weight and args.regional_reward_weight > 0.0:
            tl_regions = build_region_map(tls_ids, float(args.region_grid_size))

        tl_last_phase: Dict[str, int] = {}
        tl_phase_start_time: Dict[str, float] = {}
        current_sim_t = float(traci.simulation.getTime())
        for tl_id in tls_ids:
            try:
                tl_last_phase[tl_id] = int(traci.trafficlight.getPhase(tl_id))
            except Exception:
                tl_last_phase[tl_id] = 0
            tl_phase_start_time[tl_id] = float(current_sim_t)

        if agent is None:
            first_tl = tls_ids[0]
            s0 = build_local_state(first_tl, tl_lanes[first_tl])
            if args.use_pressure_state_features:
                s0 = add_pressure_features(
                    s0,
                    tl_phase_lanes[first_tl],
                    tl_green_phases[first_tl],
                    tl_phase_counts[first_tl],
                    feature_scale=float(args.pressure_feature_scale),
                    total_queue_scale=float(args.pressure_total_queue_scale),
                )
            s0_vec = RLTState(s0, lane_list=tl_lanes[first_tl]).to_vector()
            state_dim = len(s0_vec)
            action_dim = 2
            print(f"State dim: {state_dim} (lanes={args.lanes_per_tl}x3 + phase_onehot=16 + remaining=1"
                  f"{f' + pressure={PRESSURE_FEATURE_COUNT}' if args.use_pressure_state_features else ''})")
            print(f"Action space: {action_dim} (0=keep, 1=switch to next green phase)")
            agent = RLAgent(config={
                "state_dim": state_dim,
                "action_dim": action_dim,
                "lr": float(args.lr),
                "gamma": float(args.gamma),
                "epsilon_start": 1.0,
                "epsilon_end": float(args.epsilon_end),
                "epsilon_decay": 1.0,
                "batch_size": int(args.batch_size),
                "memory_size": int(args.memory_size),
                "target_update_freq": int(args.target_update_freq),
                "tau": float(args.tau),
                "dueling": True,
                "double_dqn": True,
                "n_step": int(args.n_step),
                "learning_starts": int(args.learning_starts),
                "gradient_clip_norm": float(args.gradient_clip_norm),
                "target_q_clip": float(args.target_q_clip),
                "soft_target_updates": bool(args.soft_target_updates),
                "prioritized_replay": bool(args.prioritized_replay),
                "priority_alpha": float(args.priority_alpha),
                "priority_beta_start": float(args.priority_beta_start),
                "priority_beta_end": float(args.priority_beta_end),
                "priority_beta_steps": int(args.priority_beta_steps),
                "max_green_phases": MAX_GREEN_PHASES,
                "dataset": args.dataset,
                "lanes_per_tl": int(args.lanes_per_tl),
                "decision_interval": int(args.decision_interval),
                "max_controlled_lights": int(args.max_controlled_lights),
                "controlled_lights_ratio": float(args.controlled_lights_ratio),
                "regional_reward_weight": float(args.regional_reward_weight),
                "region_grid_size": float(args.region_grid_size),
                "switch_mode": "legacy_cycle",
                "pressure_state_features": bool(args.use_pressure_state_features),
                "pressure_feature_count": PRESSURE_FEATURE_COUNT if args.use_pressure_state_features else 0,
                "pressure_feature_scale": float(args.pressure_feature_scale),
                "pressure_total_queue_scale": float(args.pressure_total_queue_scale),
                "wait_penalty_scale": float(args.wait_penalty_scale),
                "queue_reward_scale": float(args.queue_reward_scale),
                "switch_penalty": float(args.switch_penalty),
                "regional_penalty_clip": float(args.regional_penalty_clip),
                "mask_invalid_switches": bool(args.mask_invalid_switches),
                "min_green_seconds": float(args.min_green_seconds),
                "aggressive_upgrade": bool(args.aggressive_upgrade),
            })
            if args.resume and os.path.isfile(args.resume):
                try:
                    agent.load_model(args.resume)
                    epsilon_run_start = float(agent.epsilon)
                    print(f"Resumed from {args.resume} (ep={agent.episode_count}, eps={agent.epsilon:.4f})")
                except Exception as exc:
                    simulator.close_simulation()
                    raise RuntimeError(
                        "Failed to resume checkpoint. Resume requires the same state/action configuration "
                        "(dataset, lanes_per_tl, pressure-state setting, and switch semantics)."
                    ) from exc

            print(f"Training episodes {start_ep}..{total_episodes} ({args.episodes} new)")

            if not args.constant_lr:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    agent.optimizer,
                    T_max=max(1, args.episodes),
                    eta_min=float(args.lr_min),
                )
            else:
                lr_scheduler = None

        # Episode loop
        start_wall = time.time()
        last_decision_t = 0.0
        decision_interval_s = max(1.0, float(args.decision_interval))

        prev_state_dict: Dict[str, Dict] = {}
        for tl_id in tls_ids:
            init_state = build_local_state(tl_id, tl_lanes[tl_id])
            if args.use_pressure_state_features:
                init_state = add_pressure_features(
                    init_state,
                    tl_phase_lanes[tl_id],
                    tl_green_phases[tl_id],
                    tl_phase_counts[tl_id],
                    feature_scale=float(args.pressure_feature_scale),
                    total_queue_scale=float(args.pressure_total_queue_scale),
                )
            prev_state_dict[tl_id] = init_state
        prev_state_obj: Dict[str, RLTState] = {
            tl_id: RLTState(prev_state_dict[tl_id], lane_list=tl_lanes[tl_id]) for tl_id in tls_ids
        }

        losses: List[float] = []
        ep_rewards_per_step: List[float] = []
        ep_reward_sum = 0.0
        tl_decisions = 0
        phase_switches = 0

        while True:
            target_t = min(float(args.duration), float(last_decision_t + decision_interval_s))
            traci.simulationStep(target_t)
            sim_t = float(traci.simulation.getTime())
            done = sim_t >= float(args.duration)

            if done or (sim_t - last_decision_t) >= decision_interval_s:
                last_decision_t = sim_t

                actions: Dict[str, int] = {}
                action_masks: Dict[str, List[bool]] = {}
                current_phases: Dict[str, int] = {}
                for tl_id in tls_ids:
                    try:
                        current_phase = int(traci.trafficlight.getPhase(tl_id))
                    except Exception:
                        current_phase = 0
                    current_phases[tl_id] = current_phase
                    phase_elapsed = sync_phase_timer(
                        tl_id,
                        current_phase,
                        sim_t,
                        tl_last_phase,
                        tl_phase_start_time,
                    )
                    action_mask = build_switch_action_mask(
                        current_phase,
                        phase_elapsed,
                        tl_green_phases.get(tl_id, []),
                        tl_phase_counts.get(tl_id, 1),
                        mask_invalid_switches=bool(args.mask_invalid_switches),
                        min_green_seconds=float(args.min_green_seconds),
                    )
                    action_masks[tl_id] = action_mask

                    actions[tl_id] = agent.select_action(
                        prev_state_obj[tl_id],
                        training=True,
                        action_mask=action_mask,
                    )

                # Execute actions with setPhaseDuration to hold the decision
                for tl_id, action in actions.items():
                    try:
                        cur = int(current_phases.get(tl_id, 0))
                        gp = tl_green_phases.get(tl_id, [])
                        nph = max(1, tl_phase_counts.get(tl_id, 1))

                        if action == 0 or not action_masks.get(tl_id, [True, True])[1]:
                            traci.trafficlight.setPhaseDuration(tl_id, decision_interval_s)
                            continue

                        target_phase = get_next_green_phase(cur, gp, nph)
                        if target_phase != cur:
                            traci.trafficlight.setPhase(tl_id, target_phase)
                        traci.trafficlight.setPhaseDuration(tl_id, decision_interval_s)
                        phase_switches += 1
                    except Exception:
                        pass

                curr_state_dict: Dict[str, Dict] = {}
                curr_state_obj: Dict[str, RLTState] = {}
                for tl_id in tls_ids:
                    curr_dict = build_local_state(tl_id, tl_lanes[tl_id])
                    if args.use_pressure_state_features:
                        curr_dict = add_pressure_features(
                            curr_dict,
                            tl_phase_lanes[tl_id],
                            tl_green_phases[tl_id],
                            tl_phase_counts[tl_id],
                            feature_scale=float(args.pressure_feature_scale),
                            total_queue_scale=float(args.pressure_total_queue_scale),
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

                step_rewards: List[float] = []
                for tl_id in tls_ids:
                    r = local_reward(
                        prev_state_dict[tl_id], curr_state_dict[tl_id],
                        actions[tl_id],
                        wait_penalty_scale=float(args.wait_penalty_scale),
                        queue_reward_scale=float(args.queue_reward_scale),
                        switch_penalty=float(args.switch_penalty),
                    )
                    if tl_regions:
                        region = tl_regions.get(tl_id, (0, 0))
                        denom = max(1, region_cnt.get(region, 1))
                        region_avg_q = region_sum.get(region, 0.0) / float(denom)
                        regional_penalty = -float(args.regional_reward_weight) * region_avg_q
                        if float(args.regional_penalty_clip) > 0.0:
                            r += float(np.clip(regional_penalty, -float(args.regional_penalty_clip), 0.0))
                        else:
                            r += float(regional_penalty)

                    step_rewards.append(r)
                    agent.store_experience(
                        prev_state_obj[tl_id],
                        actions[tl_id],
                        r,
                        curr_state_obj[tl_id],
                        done,
                        trajectory_id=tl_id,
                    )
                    ep_reward_sum += float(r)
                    tl_decisions += 1

                    prev_state_dict[tl_id] = curr_state_dict[tl_id]
                    prev_state_obj[tl_id] = curr_state_obj[tl_id]

                loss = agent.train_step()
                if loss is not None:
                    losses.append(float(loss))

                ep_rewards_per_step.append(float(np.mean(step_rewards)))

            if done:
                break

        n_agents = max(1, len(tls_ids))
        avg_r_per_step = float(np.mean(ep_rewards_per_step)) if ep_rewards_per_step else 0.0
        avg_r_per_tl_step = float(ep_reward_sum / max(1, tl_decisions))
        switch_rate = float(phase_switches / max(1, tl_decisions))
        agent.end_episode(avg_reward_override=avg_r_per_tl_step)
        agent.training_history.setdefault("episode_reward_sums", []).append(float(ep_reward_sum))
        agent.training_history.setdefault("episode_reward_mean_per_tls_step", []).append(float(avg_r_per_tl_step))
        agent.training_history.setdefault("episode_switch_rates", []).append(float(switch_rate))

        eps_decay_over = int(args.epsilon_decay_episodes) if args.epsilon_decay_episodes > 0 else int(args.episodes)
        progress = float(ep - start_ep + 1) / float(max(1, eps_decay_over))
        progress = float(np.clip(progress, 0.0, 1.0))
        agent.epsilon = max(
            agent.config['epsilon_end'],
            epsilon_run_start - (epsilon_run_start - agent.config['epsilon_end']) * progress
        )
        if losses and lr_scheduler is not None:
            lr_scheduler.step()

        simulator.close_simulation()
        # Windows/Linux: prior SUMO may still hold the old port/socket briefly; stagger ports + small delay.
        time.sleep(0.35)

        wall = time.time() - start_wall
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        current_lr = float(agent.optimizer.param_groups[0]["lr"])
        recent_tl_rewards = agent.training_history.get("episode_reward_mean_per_tls_step", [])
        reward_tl_ma20 = float(np.mean(recent_tl_rewards[-20:])) if recent_tl_rewards else float("nan")
        plateau_window = max(5, int(args.plateau_window))
        reward_tl_ma = float(np.mean(recent_tl_rewards[-plateau_window:])) if recent_tl_rewards else float("nan")

        if np.isfinite(reward_tl_ma):
            min_delta = float(args.plateau_min_delta)
            if reward_tl_ma > (best_reward_ma + min_delta):
                best_reward_ma = reward_tl_ma
                episodes_without_improve = 0
                best_root, best_ext = os.path.splitext(out_path)
                best_model_path = f"{best_root}_best_reward{best_ext or '.pt'}"
                best_path = build_checkpoint_path(best_model_path, ep)
                agent.save_model(best_path)
            else:
                episodes_without_improve += 1

            if episodes_without_improve >= max(1, int(args.plateau_patience)):
                old_lr = float(agent.optimizer.param_groups[0]["lr"])
                new_lr = max(float(args.plateau_min_lr), old_lr * float(args.plateau_lr_decay))
                for group in agent.optimizer.param_groups:
                    group["lr"] = new_lr
                agent.epsilon = min(
                    max(0.20, float(agent.config["epsilon_end"])),
                    float(agent.epsilon) + float(args.plateau_epsilon_boost),
                )
                episodes_without_improve = 0
                print(f"Plateau recovery: lr {old_lr:.2e}->{new_lr:.2e}, epsilon boosted to {agent.epsilon:.3f}")

        print(
            f"Episode {ep:04d}/{total_episodes} | wall={wall:.2f}s | "
            f"reward(sum)={ep_reward_sum:.2f} | reward/tl_step={avg_r_per_tl_step:.4f} | "
            f"reward/tl_step_ma20={reward_tl_ma20:.4f} | "
            f"reward/decision={avg_r_per_step:.4f} | "
            f"switch_rate={switch_rate:.3f} | "
            f"avg_loss={avg_loss:.4f} | epsilon={agent.epsilon:.3f} | lr={current_lr:.2e} | "
            f"tls={n_agents} | ratio={run_ratio:.3f} | sumo_seed={sumo_seed}"
        )

        if ep % int(args.save_every) == 0:
            agent.save_model(out_path)
            snapshot_path = build_checkpoint_path(out_path, ep)
            agent.save_model(snapshot_path)
            print(f"Saved checkpoint snapshot: {snapshot_path}")

    agent.save_model(out_path)
    print(f"\nDone. MARL model saved to: {out_path}")


if __name__ == "__main__":
    main()
