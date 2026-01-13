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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=int, default=300, help="Seconds per episode")
    parser.add_argument("--decision-interval", type=int, default=5, help="Seconds between actions per TLS")
    parser.add_argument("--max-controlled-lights", type=int, default=3, help="How many TLS agents to train simultaneously")
    parser.add_argument("--lanes-per-tl", type=int, default=8, help="Fixed lanes per TLS in observation (pad/truncate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="", help="Model output path")
    args = parser.parse_args()

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    np.random.seed(args.seed)

    SimulationConfig.set_dataset("los_angeles")
    SimulationConfig.CONFIGS["los_angeles"]["max_controlled_lights"] = int(args.max_controlled_lights)
    SimulationConfig.create_output_dirs()

    default_out = os.path.join(SimulationConfig.MODEL_DIR, "marl_los_angeles_shared_dqn.pt")
    out_path = args.out.strip() or default_out

    agent: RLAgent | None = None

    print("=" * 78)
    print("MARL training: Los Angeles | parameter-sharing DQN")
    print(f"Episodes={args.episodes} duration={args.duration}s interval={args.decision_interval}s")
    print(f"Agents(TLS)={args.max_controlled_lights} lanes_per_tl={args.lanes_per_tl}")
    print(f"Output: {out_path}")
    print("=" * 78)

    for ep in range(1, args.episodes + 1):
        simulator = TrafficSimulator(
            use_gui=False,
            dataset="los_angeles",
            enable_sumo_emissions_output=False,
        )
        ok = simulator.start_simulation()
        if not ok or not simulator.controlled_traffic_lights:
            simulator.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected.")

        tls_ids = simulator.controlled_traffic_lights[: int(args.max_controlled_lights)]
        # per TLS: fixed lane list
        tl_lanes: Dict[str, List[str]] = {}
        tl_phase_counts: Dict[str, int] = {}
        for tl_id in tls_ids:
            lanes = get_controlled_lanes_for_tl(tl_id)
            tl_lanes[tl_id] = pad_or_truncate(lanes, int(args.lanes_per_tl))
            tl_phase_counts[tl_id] = max(1, get_phase_count(tl_id))

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

        while True:
            traci.simulationStep()
            sim_t = float(traci.simulation.getTime())
            done = sim_t >= float(args.duration)

            if done or (sim_t - last_decision_t) >= float(args.decision_interval):
                last_decision_t = sim_t

                # Act for each TLS (multi-agent)
                actions: Dict[str, int] = {}
                for tl_id in tls_ids:
                    actions[tl_id] = agent.select_action(prev_state_obj[tl_id], training=True)

                # Apply actions
                for tl_id, action in actions.items():
                    if action != 1:
                        continue
                    try:
                        cur = int(traci.trafficlight.getPhase(tl_id))
                        nph = int(tl_phase_counts.get(tl_id, 1))
                        traci.trafficlight.setPhase(tl_id, (cur + 1) % max(1, nph))
                    except Exception:
                        pass

                # Observe next + store experience per agent
                for tl_id in tls_ids:
                    curr_dict = build_local_state(tl_id, tl_lanes[tl_id])
                    curr_obj = RLTState(curr_dict, lane_list=tl_lanes[tl_id])
                    r = local_reward(prev_state_dict[tl_id], curr_dict, actions[tl_id])

                    agent.store_experience(prev_state_obj[tl_id], actions[tl_id], r, curr_obj, done)
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(float(loss))

                    prev_state_dict[tl_id] = curr_dict
                    prev_state_obj[tl_id] = curr_obj

            if done:
                break

        agent.end_episode()
        simulator.close_simulation()

        wall = time.time() - start_wall
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        ep_reward = agent.training_history["episode_rewards"][-1] if agent.training_history["episode_rewards"] else 0.0
        print(f"Episode {ep:04d}/{args.episodes} | wall={wall:.2f}s | reward(sum)={ep_reward:.2f} | avg_loss={avg_loss:.4f} | epsilon={agent.epsilon:.3f} | tls={len(tls_ids)}")

        if ep % int(args.save_every) == 0:
            agent.save_model(out_path)

    agent.save_model(out_path)
    print(f"\nDone. MARL model saved to: {out_path}")


if __name__ == "__main__":
    main()

