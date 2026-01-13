#!/usr/bin/env python3
"""
Train a real DQN agent on the OSM Los Angeles map.

Design choices (practical + stable for FYP):
- Train on ONE controlled traffic light (max_controlled_lights=1)
- Action space = 2 actions:
    0 = keep current phase
    1 = switch to next phase (cycle)
- Headless SUMO (use_gui=False)
- Disable SUMO emission-output during training to avoid huge XML per episode.
  (You can still evaluate emissions later using the normal simulator runs.)

This trains a neural network (src/rl_agent.py) and saves checkpoints under models/.
"""

import argparse
import os
import time
from typing import Dict, List

import numpy as np

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator
from src.rl_agent import RLAgent, TrafficState as RLTState

try:
    import traci
except ImportError:
    traci = None


def get_phase_count(tl_id: str) -> int:
    """
    Return number of phases for a traffic light in a SUMO-version-safe way.

    Some traci wheels (even when SUMO itself is new) may not expose
    trafficlight.getPhaseNumber(), so we fall back to reading program logics.
    """
    if hasattr(traci.trafficlight, "getPhaseNumber"):
        try:
            return int(traci.trafficlight.getPhaseNumber(tl_id))
        except Exception:
            pass

    try:
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if logics:
            # take the first logic as active/default
            return int(len(logics[0].phases))
    except Exception:
        pass

    # Safe fallback: allow "next phase" action but keep modulo 1 => no-op.
    return 1


def build_state(tl_id: str, lane_ids: List[str]) -> Dict:
    """Build a sumo_state dict compatible with src.rl_agent.TrafficState."""
    sim_time = traci.simulation.getTime()

    lane_vehicle_counts = {}
    lane_queue_lengths = {}
    lane_mean_speeds = {}

    for lane_id in lane_ids:
        try:
            lane_vehicle_counts[lane_id] = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_mean_speeds[lane_id] = traci.lane.getLastStepMeanSpeed(lane_id)

            # Queue = vehicles with waiting time > 5s
            q = 0
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    if traci.vehicle.getWaitingTime(vid) > 5.0:
                        q += 1
                except Exception:
                    pass
            lane_queue_lengths[lane_id] = q
        except Exception:
            lane_vehicle_counts[lane_id] = 0
            lane_queue_lengths[lane_id] = 0
            lane_mean_speeds[lane_id] = 0.0

    try:
        phase = traci.trafficlight.getPhase(tl_id)
    except Exception:
        phase = 0

    # rl_agent.TrafficState uses one-hot with max_phases=8
    # Keep it stable even if the TLS has >8 SUMO phases.
    phase = int(phase) % 8

    try:
        remaining = traci.trafficlight.getNextSwitch(tl_id) - sim_time
    except Exception:
        remaining = 0.0
    remaining = float(max(0.0, remaining))

    return {
        "time": float(sim_time),
        "lane_vehicle_counts": lane_vehicle_counts,
        "lane_queue_lengths": lane_queue_lengths,
        "lane_mean_speeds": lane_mean_speeds,
        "detector_occupancy": {},  # OSM maps typically have no induction loops
        "traffic_light_phase": phase,
        "traffic_light_remaining_time": remaining,
    }


def reward_from_queues(prev_state: Dict, curr_state: Dict, action: int) -> float:
    """Simple, stable reward for large maps: reduce queues, penalize switching."""
    prev_q = sum(prev_state.get("lane_queue_lengths", {}).values())
    curr_q = sum(curr_state.get("lane_queue_lengths", {}).values())
    improvement = prev_q - curr_q

    # Main objective: small queues
    r = -0.1 * curr_q + 0.05 * improvement

    # Penalize switching a bit (stability)
    if action == 1:
        r -= 0.02
    return float(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--duration", type=int, default=300, help="Seconds per episode")
    parser.add_argument("--decision-interval", type=int, default=5, help="Control decision interval (seconds)")
    parser.add_argument("--max-controlled-lights", type=int, default=1, help="Control only first N traffic lights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N episodes")
    parser.add_argument("--out", type=str, default="", help="Output model path (default: models/rl_los_angeles_dqn.pt)")
    args = parser.parse_args()

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    np.random.seed(args.seed)

    # Set dataset
    SimulationConfig.set_dataset("los_angeles")
    # Limit controlled lights for training
    SimulationConfig.CONFIGS["los_angeles"]["max_controlled_lights"] = int(args.max_controlled_lights)

    # Prepare model output path
    SimulationConfig.create_output_dirs()
    default_out = os.path.join(SimulationConfig.MODEL_DIR, "rl_los_angeles_dqn.pt")
    out_path = args.out.strip() or default_out

    agent = None
    lane_ids = None
    tl_id = None
    n_phases = None

    print("=" * 70)
    print("Training DQN on Los Angeles (1 TLS, 2 actions keep/next)")
    print(f"Episodes: {args.episodes}, Duration: {args.duration}s, Decision interval: {args.decision_interval}s")
    print(f"Max controlled lights: {args.max_controlled_lights}")
    print(f"Model output: {out_path}")
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        simulator = TrafficSimulator(
            use_gui=False,
            dataset="los_angeles",
            enable_sumo_emissions_output=False,  # IMPORTANT: don't generate huge emission XML during training
        )

        ok = simulator.start_simulation()
        if not ok or not simulator.controlled_traffic_lights:
            simulator.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected in LA map.")

        # Select first controlled TLS
        tl_id = simulator.controlled_traffic_lights[0]
        lane_ids = simulator.lanes
        n_phases = get_phase_count(tl_id)

        # Initialize agent once we know state dimension
        if agent is None:
            # Build a sample state
            s0 = build_state(tl_id, lane_ids)
            state_dim = len(RLTState(s0, lane_list=lane_ids).to_vector())
            agent = RLAgent(config={
                "state_dim": state_dim,
                "action_dim": 2,         # keep / next-phase
                "lr": 1e-4,
                "gamma": 0.99,
                "batch_size": 64,
                "memory_size": 50000,
                "target_update_freq": 200,
                "dueling": True,
                "double_dqn": True,
                "n_step": 3,
            })

        # Episode loop
        start_wall = time.time()
        step = 0
        last_decision_t = 0

        prev_state_dict = build_state(tl_id, lane_ids)
        prev_state = RLTState(prev_state_dict, lane_list=lane_ids)

        done = False
        losses = []

        while True:
            traci.simulationStep()
            sim_t = traci.simulation.getTime()
            step += 1

            if sim_t >= args.duration:
                done = True

            # Make a decision every decision-interval seconds
            if done or (sim_t - last_decision_t) >= args.decision_interval:
                last_decision_t = sim_t

                # Choose action
                action = agent.select_action(prev_state, training=True)

                # Apply action
                if action == 1 and n_phases and n_phases > 0:
                    try:
                        cur_phase = traci.trafficlight.getPhase(tl_id)
                        next_phase = (int(cur_phase) + 1) % int(n_phases)
                        traci.trafficlight.setPhase(tl_id, next_phase)
                    except Exception:
                        pass

                # Observe next state
                curr_state_dict = build_state(tl_id, lane_ids)
                curr_state = RLTState(curr_state_dict, lane_list=lane_ids)

                # Reward
                r = reward_from_queues(prev_state_dict, curr_state_dict, action)

                # Store + train
                agent.store_experience(prev_state, action, r, curr_state, done)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                prev_state_dict = curr_state_dict
                prev_state = curr_state

            if done:
                break

        agent.end_episode()
        simulator.close_simulation()

        wall = time.time() - start_wall
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        ep_reward = agent.training_history["episode_rewards"][-1] if agent.training_history["episode_rewards"] else 0.0
        print(f"Episode {ep:04d}/{args.episodes} | wall={wall:.1f}s | reward={ep_reward:.2f} | avg_loss={avg_loss:.4f} | epsilon={agent.epsilon:.3f} | tl={tl_id} | phases={n_phases} | lanes={len(lane_ids)}")

        if ep % args.save_every == 0:
            agent.save_model(out_path)

    # Final save
    agent.save_model(out_path)
    print(f"\nDone. Model saved to: {out_path}")


if __name__ == "__main__":
    main()

