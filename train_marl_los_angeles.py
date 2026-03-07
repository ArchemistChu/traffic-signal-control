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


MAX_GREEN_PHASES = 6  # fixed action dimension — pad/mask if fewer green phases

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


def get_green_phases(tl_id: str) -> List[int]:
    """Return indices of phases that give green to at least one vehicle lane."""
    def _is_ped(lane_id: str) -> bool:
        if not lane_id:
            return False
        low = lane_id.lower()
        return any(k in low for k in [":w", "ped", "walk", "sidewalk", "foot", "crossing"])

    try:
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if not logics:
            return list(range(max(1, get_phase_count(tl_id))))
        links = traci.trafficlight.getControlledLinks(tl_id)
        phases = logics[0].phases
        greens = []
        for p_idx, phase in enumerate(phases):
            state_str = phase.state
            has_vehicle_green = False
            for li, ch in enumerate(state_str):
                if ch in ("G", "g"):
                    if li < len(links):
                        for link in links[li]:
                            if link and len(link) > 0 and not _is_ped(link[0]):
                                has_vehicle_green = True
                                break
                    else:
                        has_vehicle_green = True
                if has_vehicle_green:
                    break
            if has_vehicle_green:
                greens.append(int(p_idx))
        return greens if greens else list(range(max(1, len(phases))))
    except Exception:
        return list(range(max(1, get_phase_count(tl_id))))


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


def build_local_state(tl_id: str, lane_ids_fixed: List[str]) -> Dict:
    """Build a SUMO state dict compatible with src.rl_agent.TrafficState."""
    sim_time = float(traci.simulation.getTime())

    lane_vehicle_counts: Dict[str, int] = {}
    lane_queue_lengths: Dict[str, int] = {}
    lane_mean_speeds: Dict[str, float] = {}

    for lane_id in lane_ids_fixed:
        if not lane_id:
            # padding lane
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


def local_reward(prev_state: Dict, curr_state: Dict, action: int,
                  prev_action: int = 0, n_green_phases: int = 2) -> float:
    """Reward targeting the metrics we measure: queue length, waiting time, speed.

    Components (all clipped to keep total in roughly [-1.5, +1]):
      1. Queue improvement (delta)       — encourages queue reduction
      2. Absolute queue penalty           — penalises large standing queues
      3. Speed improvement (delta)        — encourages flow
      4. Phase-switch penalty             — discourages oscillation
    """
    prev_q = sum(prev_state.get("lane_queue_lengths", {}).values())
    curr_q = sum(curr_state.get("lane_queue_lengths", {}).values())
    queue_change = prev_q - curr_q

    prev_speeds = list(prev_state.get("lane_mean_speeds", {}).values())
    curr_speeds = list(curr_state.get("lane_mean_speeds", {}).values())
    prev_avg_speed = np.mean(prev_speeds) if prev_speeds else 0.0
    curr_avg_speed = np.mean(curr_speeds) if curr_speeds else 0.0
    speed_change = curr_avg_speed - prev_avg_speed

    QUEUE_SCALE = 10.0
    SPEED_SCALE = 5.0
    ABS_QUEUE_SCALE = 30.0

    r_delta_q = 0.45 * np.clip(queue_change / QUEUE_SCALE, -1.0, 1.0)
    r_abs_q = -0.25 * np.clip(curr_q / ABS_QUEUE_SCALE, 0.0, 1.0)
    r_speed = 0.25 * np.clip(speed_change / SPEED_SCALE, -1.0, 1.0)

    r_switch = 0.0
    if action != 0 and action != prev_action:
        r_switch = -0.05

    return float(r_delta_q + r_abs_q + r_speed + r_switch)


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
    parser.add_argument("--lanes-per-tl", type=int, default=8, help="Fixed lanes per TLS in observation (pad/truncate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="", help="Model output path")
    args = parser.parse_args()

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    np.random.seed(args.seed)

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

    for ep in range(1, args.episodes + 1):
        simulator = TrafficSimulator(
            use_gui=False,
            dataset=args.dataset,
            enable_sumo_emissions_output=False,
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
            if ep == 1:
                print(f"Computed TLS count: {desired}/{total_tls} ({args.controlled_lights_ratio:.2f})")
        else:
            tls_ids = simulator.controlled_traffic_lights[: int(args.max_controlled_lights)]
        # per TLS: fixed lane list
        tl_lanes: Dict[str, List[str]] = {}
        tl_phase_counts: Dict[str, int] = {}
        for tl_id in tls_ids:
            lanes = get_controlled_lanes_for_tl(tl_id)
            tl_lanes[tl_id] = pad_or_truncate(lanes, int(args.lanes_per_tl))
            tl_phase_counts[tl_id] = max(1, get_phase_count(tl_id))

        # Per-TL green phase list (for N-action mapping)
        tl_green_phases: Dict[str, List[int]] = {}
        for tl_id in tls_ids:
            gp = get_green_phases(tl_id)
            tl_green_phases[tl_id] = gp
            if ep == 1:
                print(f"  TL {tl_id}: {len(gp)} green phases {gp[:8]}")

        tl_regions: Dict[str, Tuple[int, int]] = {}
        if args.regional_reward_weight and args.regional_reward_weight > 0.0:
            tl_regions = build_region_map(tls_ids, float(args.region_grid_size))

        # Init agent once we know state_dim (fixed by lanes-per-tl)
        # action_dim = MAX_GREEN_PHASES+1: action 0 = keep, actions 1..N = jump to green phase i
        if agent is None:
            s0 = build_local_state(tls_ids[0], tl_lanes[tls_ids[0]])
            state_dim = len(RLTState(s0, lane_list=tl_lanes[tls_ids[0]]).to_vector())
            action_dim = MAX_GREEN_PHASES + 1
            print(f"Action space: {action_dim} (0=keep, 1..{MAX_GREEN_PHASES}=jump to green phase)")
            agent = RLAgent(config={
                "state_dim": state_dim,
                "action_dim": action_dim,
                "lr": 1e-4,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay": 1.0,
                "batch_size": 64,
                "memory_size": 200000,
                "target_update_freq": 500,
                "tau": 0.01,
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
            })
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                agent.optimizer, T_max=args.episodes, eta_min=1e-5
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

        while True:
            traci.simulationStep()
            sim_t = float(traci.simulation.getTime())
            done = sim_t >= float(args.duration)

            if done or (sim_t - last_decision_t) >= float(args.decision_interval):
                last_decision_t = sim_t

                actions: Dict[str, int] = {}
                for tl_id in tls_ids:
                    actions[tl_id] = agent.select_action(prev_state_obj[tl_id], training=True)

                # Apply N-action: 0=keep, 1..N=jump to green phase index
                for tl_id, action in actions.items():
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

                    prev_state_dict[tl_id] = curr_state_dict[tl_id]
                    prev_state_obj[tl_id] = curr_state_obj[tl_id]

                prev_actions = dict(actions)

                NUM_TRAIN_STEPS = 4
                for _ in range(NUM_TRAIN_STEPS):
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(float(loss))

                ep_rewards_per_step.append(float(np.mean(step_rewards)))

            if done:
                break

        n_agents = max(1, len(tls_ids))
        avg_r_per_step = float(np.mean(ep_rewards_per_step)) if ep_rewards_per_step else 0.0
        agent.end_episode(avg_reward_override=avg_r_per_step)

        # Linear epsilon decay: 1.0 -> 0.05 over all episodes
        agent.epsilon = max(
            agent.config['epsilon_end'],
            1.0 - (1.0 - agent.config['epsilon_end']) * ep / args.episodes
        )
        lr_scheduler.step()

        simulator.close_simulation()

        wall = time.time() - start_wall
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Episode {ep:04d}/{args.episodes} | wall={wall:.2f}s | avg_reward/step={avg_r_per_step:.4f} | avg_loss={avg_loss:.4f} | epsilon={agent.epsilon:.3f} | lr={current_lr:.2e} | tls={n_agents}")

        if ep % int(args.save_every) == 0:
            agent.save_model(out_path)

    agent.save_model(out_path)
    print(f"\nDone. MARL model saved to: {out_path}")


if __name__ == "__main__":
    main()

