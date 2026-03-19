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
        float(np.clip(current_pressure / PRESSURE_FEATURE_SCALE, 0.0, 1.0)),
        float(np.clip(next_pressure / PRESSURE_FEATURE_SCALE, 0.0, 1.0)),
        float(np.clip((next_pressure - current_pressure) / PRESSURE_FEATURE_SCALE, -1.0, 1.0)),
        float(np.clip(total_queue / PRESSURE_TOTAL_QUEUE_SCALE, 0.0, 1.0)),
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
) -> float:
    """Legacy v8-style reward targeting waiting time and queue reduction.

    Components:
      1. Waiting time penalty (absolute) — THE key metric baselines are measured on
      2. Queue reduction (delta)         — encourages clearing queues
      3. Phase-switch penalty            — discourages rapid oscillation
    """
    curr_wait = sum(curr_state.get("lane_waiting_times", {}).values())
    curr_q = sum(curr_state.get("lane_queue_lengths", {}).values())
    prev_q = sum(prev_state.get("lane_queue_lengths", {}).values())
    queue_change = prev_q - curr_q

    r_wait = -0.50 * np.clip(curr_wait / 200.0, 0.0, 1.0)
    r_queue = 0.35 * np.clip(queue_change / 10.0, -1.0, 1.0)
    r_switch = -0.05 if action == 1 else 0.0

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
        help="If >0, use this ratio of total TLS (overrides --max-controlled-lights). Example: 0.2 for 20%",
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
    parser.add_argument("--demand-scale", type=float, default=0.0, help="SUMO --scale for traffic demand (0=default, 0.7=70%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="", help="Model output path")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    args = parser.parse_args()

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
        # Use a large cap so we can compute ratio from full TLS list.
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = 10**9
    else:
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = int(args.max_controlled_lights)
    SimulationConfig.create_output_dirs()

    # Default output model path:
    # - keep backward-compatible name when regional_reward_weight==0
    # - otherwise, avoid overwriting the non-region-aware model by adding a suffix
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

    ep = start_ep - 1
    while ep < total_episodes:
        ep += 1
        sumo_seed = int(args.seed) + int(ep)
        simulator = TrafficSimulator(
            use_gui=False,
            dataset=args.dataset,
            enable_sumo_emissions_output=False,
            sumo_seed=sumo_seed,
            controlled_lights_ratio=float(args.controlled_lights_ratio) if args.controlled_lights_ratio > 0 else None,
            demand_scale=float(args.demand_scale) if args.demand_scale > 0 else None,
        )
        ok = simulator.start_simulation()
        if not ok or not simulator.controlled_traffic_lights:
            simulator.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected.")

        if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
            total_tls = len(simulator.traffic_lights) if simulator.traffic_lights else len(simulator.controlled_traffic_lights)
            desired = int(round(total_tls * float(args.controlled_lights_ratio)))
            desired = max(1, min(total_tls, desired))
            tls_ids = (simulator.traffic_lights or simulator.controlled_traffic_lights)[:desired]
            if agent is None:
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
            if agent is None:
                print(f"  TL {tl_id}: {len(gp)} green phases {gp[:8]}")

        tl_regions: Dict[str, Tuple[int, int]] = {}
        if args.regional_reward_weight and args.regional_reward_weight > 0.0:
            tl_regions = build_region_map(tls_ids, float(args.region_grid_size))

        if agent is None:
            first_tl = tls_ids[0]
            s0 = build_local_state(first_tl, tl_lanes[first_tl])
            state_dim = len(RLTState(s0, lane_list=tl_lanes[first_tl]).to_vector())
            action_dim = 2
            print(f"Action space: {action_dim} (0=keep, 1=switch to next phase)")
            agent = RLAgent(config={
                "state_dim": state_dim,
                "action_dim": action_dim,
                "lr": 5e-5,
                "gamma": 0.90,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay": 1.0,
                "batch_size": 128,
                "memory_size": 200000,
                "target_update_freq": 2000,
                "tau": 0.005,
                "dueling": True,
                "double_dqn": True,
                "n_step": 1,
                "max_green_phases": MAX_GREEN_PHASES,
                "dataset": args.dataset,
                "lanes_per_tl": int(args.lanes_per_tl),
                "decision_interval": int(args.decision_interval),
                "max_controlled_lights": int(args.max_controlled_lights),
                "controlled_lights_ratio": float(args.controlled_lights_ratio),
                "regional_reward_weight": float(args.regional_reward_weight),
                "region_grid_size": float(args.region_grid_size),
                "switch_mode": "legacy_cycle",
                "pressure_state_features": False,
                "pressure_feature_count": 0,
            })
            if args.resume and os.path.isfile(args.resume):
                try:
                    agent.load_model(args.resume)
                    epsilon_run_start = float(agent.epsilon)
                    print(f"Resumed from {args.resume} (ep={agent.episode_count}, eps={agent.epsilon:.4f})")
                except Exception as exc:
                    simulator.close_simulation()
                    raise RuntimeError(
                        "Failed to resume checkpoint. v9-safe expects the legacy 2-action raw-phase model "
                        "with the same state size as v8."
                    ) from exc

            print(f"Training episodes {start_ep}..{total_episodes} ({args.episodes} new)")

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                agent.optimizer, T_max=max(1, args.episodes), eta_min=1e-5
            )

        # Episode loop
        start_wall = time.time()
        last_decision_t = 0.0

        prev_state_dict: Dict[str, Dict] = {
            tl_id: build_local_state(tl_id, tl_lanes[tl_id]) for tl_id in tls_ids
        }
        prev_state_obj: Dict[str, RLTState] = {
            tl_id: RLTState(prev_state_dict[tl_id], lane_list=tl_lanes[tl_id]) for tl_id in tls_ids
        }

        losses: List[float] = []
        ep_rewards_per_step: List[float] = []
        prev_actions: Dict[str, int] = {tl_id: 0 for tl_id in tls_ids}
        ep_reward_sum = 0.0
        tl_decisions = 0
        phase_switches = 0

        while True:
            traci.simulationStep()
            sim_t = float(traci.simulation.getTime())
            done = sim_t >= float(args.duration)

            if done or (sim_t - last_decision_t) >= float(args.decision_interval):
                last_decision_t = sim_t

                actions: Dict[str, int] = {}
                for tl_id in tls_ids:
                    actions[tl_id] = agent.select_action(prev_state_obj[tl_id], training=True)

                for tl_id, action in actions.items():
                    if action == 0:
                        continue
                    try:
                        cur = int(traci.trafficlight.getPhase(tl_id))
                        nph = max(1, tl_phase_counts.get(tl_id, 1))
                        traci.trafficlight.setPhase(tl_id, (cur + 1) % nph)
                        phase_switches += 1
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

                step_rewards: List[float] = []
                for tl_id in tls_ids:
                    gp = tl_green_phases.get(tl_id, [])
                    r = local_reward(
                        prev_state_dict[tl_id], curr_state_dict[tl_id],
                        actions[tl_id], prev_actions.get(tl_id, 0),
                        n_green_phases=len(gp),
                    )
                    if tl_regions:
                        region = tl_regions.get(tl_id, (0, 0))
                        denom = max(1, region_cnt.get(region, 1))
                        region_avg_q = region_sum.get(region, 0.0) / float(denom)
                        regional_penalty = -float(args.regional_reward_weight) * region_avg_q
                        r += float(np.clip(regional_penalty, -0.3, 0.0))

                    step_rewards.append(r)
                    agent.store_experience(
                        prev_state_obj[tl_id],
                        actions[tl_id],
                        r,
                        curr_state_obj[tl_id],
                        done
                    )
                    ep_reward_sum += float(r)
                    tl_decisions += 1

                    prev_state_dict[tl_id] = curr_state_dict[tl_id]
                    prev_state_obj[tl_id] = curr_state_obj[tl_id]

                prev_actions = dict(actions)

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

        progress = float(ep - start_ep + 1) / float(max(1, args.episodes))
        progress = float(np.clip(progress, 0.0, 1.0))
        agent.epsilon = max(
            agent.config['epsilon_end'],
            epsilon_run_start - (epsilon_run_start - agent.config['epsilon_end']) * progress
        )
        if losses:
            lr_scheduler.step()

        simulator.close_simulation()

        wall = time.time() - start_wall
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        current_lr = lr_scheduler.get_last_lr()[0]
        print(
            f"Episode {ep:04d}/{total_episodes} | wall={wall:.2f}s | "
            f"reward(sum)={ep_reward_sum:.2f} | reward/tl_step={avg_r_per_tl_step:.4f} | "
            f"reward/decision={avg_r_per_step:.4f} | switch_rate={switch_rate:.3f} | "
            f"avg_loss={avg_loss:.4f} | epsilon={agent.epsilon:.3f} | lr={current_lr:.2e} | "
            f"tls={n_agents} | sumo_seed={sumo_seed}"
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

