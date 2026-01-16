#!/usr/bin/env python3
"""
Train a PressLight-inspired baseline on SUMO OSM maps (parameter sharing DQN).

This is NOT the official PressLight implementation (which is typically in CityFlow).
It is a SUMO-adapted variant that uses the key PressLight idea:
  - state based on phase "pressure" (queue-based demand)
  - choose which green phase to serve next

We use one shared DQN policy across all controlled intersections (parameter sharing),
which is practical for large-scale networks.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="los_angeles", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--max-controlled-lights", type=int, default=5)
    parser.add_argument("--k", type=int, default=8, help="Pad/truncate number of phase-pressure features")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="", help="Output checkpoint path")
    args = parser.parse_args()

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    np.random.seed(args.seed)

    SimulationConfig.set_dataset(args.dataset)
    SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = int(args.max_controlled_lights)
    SimulationConfig.create_output_dirs()

    out_path = args.out.strip() or os.path.join(SimulationConfig.MODEL_DIR, "presslight_shared_dqn.pt")

    # PressLight-like state: K phase-pressures as "fake lanes"
    state_dim = 3 * int(args.k) + 8 + 1
    action_dim = int(args.k)

    agent = RLAgent(config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "lr": 3e-4,
        "gamma": 0.95,
        "batch_size": 64,
        "memory_size": 50000,
        "target_update_freq": 500,
        "dueling": True,
        "double_dqn": True,
        "n_step": 3,
    })

    print("=" * 78)
    print("PressLight-inspired training (SUMO-adapted)")
    print(f"dataset={args.dataset} episodes={args.episodes} duration={args.duration}s interval={args.decision_interval}s")
    print(f"tls={args.max_controlled_lights} K={args.k} out={out_path}")
    print("=" * 78)

    fake_lanes = [f"p{i}" for i in range(int(args.k))]

    for ep in range(1, args.episodes + 1):
        sim = TrafficSimulator(use_gui=False, dataset=args.dataset, enable_sumo_emissions_output=False)
        ok = sim.start_simulation()
        if not ok or not sim.controlled_traffic_lights:
            sim.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected.")

        tls_ids = sim.controlled_traffic_lights[: int(args.max_controlled_lights)]
        last_decision_t = 0.0

        # init previous states
        prev_state_obj: Dict[str, RLTState] = {}
        prev_pressures: Dict[str, List[float]] = {}

        def tl_green_phases(tl_id: str) -> List[int]:
            pm = sim._get_tl_phase_lane_map(tl_id)
            phases = pm.get("green_phases", []) or []
            nph = sim._get_tl_phase_count(tl_id)
            return phases if phases else list(range(max(1, nph)))

        def lane_weight(lane_id: str, state: Dict) -> float:
            q = float(state.get("lane_queue_lengths", {}).get(lane_id, 0))
            bus_c = float(state.get("lane_bus_counts", {}).get(lane_id, 0))
            bus_w = float(state.get("bus_waiting_time", {}).get(lane_id, 0.0))
            return q + (2.5 * bus_c) + (0.3 * bus_w)

        def compute_pressures(tl_id: str, state: Dict) -> List[float]:
            pm = sim._get_tl_phase_lane_map(tl_id)
            greens = tl_green_phases(tl_id)
            vals = []
            for p in greens[: int(args.k)]:
                lanes = pm.get("phase_to_lanes", {}).get(int(p), []) or []
                vals.append(float(sum(lane_weight(l, state) for l in lanes)))
            while len(vals) < int(args.k):
                vals.append(0.0)
            return vals

        # Build initial states
        init_state = sim.get_current_state()
        for tl_id in tls_ids:
            prs = compute_pressures(tl_id, init_state)
            prev_pressures[tl_id] = prs
            sumo_state = {
                "time": float(sim.simulation_time),
                "lane_vehicle_counts": {fake_lanes[i]: prs[i] for i in range(int(args.k))},
                "lane_queue_lengths": {fake_lanes[i]: prs[i] for i in range(int(args.k))},
                "lane_mean_speeds": {fake_lanes[i]: 0.0 for i in range(int(args.k))},
                "detector_occupancy": {},
                "traffic_light_phase": int(traci.trafficlight.getPhase(tl_id)) % 8,
                "traffic_light_remaining_time": float(max(0.0, traci.trafficlight.getNextSwitch(tl_id) - sim.simulation_time)),
            }
            prev_state_obj[tl_id] = RLTState(sumo_state, lane_list=fake_lanes)

        losses = []
        while True:
            traci.simulationStep()
            sim.simulation_time = traci.simulation.getTime()
            if sim.simulation_time >= float(args.duration):
                done = True
            else:
                done = False

            if done or (float(sim.simulation_time) - last_decision_t) >= float(args.decision_interval):
                last_decision_t = float(sim.simulation_time)
                state = sim.get_current_state()

                for tl_id in tls_ids:
                    # Build state
                    prs = compute_pressures(tl_id, state)
                    sumo_state = {
                        "time": float(sim.simulation_time),
                        "lane_vehicle_counts": {fake_lanes[i]: prs[i] for i in range(int(args.k))},
                        "lane_queue_lengths": {fake_lanes[i]: prs[i] for i in range(int(args.k))},
                        "lane_mean_speeds": {fake_lanes[i]: 0.0 for i in range(int(args.k))},
                        "detector_occupancy": {},
                        "traffic_light_phase": int(traci.trafficlight.getPhase(tl_id)) % 8,
                        "traffic_light_remaining_time": float(max(0.0, traci.trafficlight.getNextSwitch(tl_id) - sim.simulation_time)),
                    }
                    curr_state_obj = RLTState(sumo_state, lane_list=fake_lanes)

                    # Action = choose index in green phases list
                    action = int(agent.select_action(prev_state_obj[tl_id], training=True))

                    greens = tl_green_phases(tl_id)
                    if greens:
                        target_phase = int(greens[action % len(greens)])
                        try:
                            cur_p = int(traci.trafficlight.getPhase(tl_id))
                            if target_phase != cur_p and (traci.trafficlight.getNextSwitch(tl_id) - sim.simulation_time) <= 1.0:
                                traci.trafficlight.setPhase(tl_id, target_phase)
                                traci.trafficlight.setPhaseDuration(tl_id, 30.0)
                        except Exception:
                            pass

                    # Reward: reduce total pressure
                    prev_sum = float(sum(prev_pressures[tl_id]))
                    curr_sum = float(sum(prs))
                    reward = -(curr_sum) + 0.3 * (prev_sum - curr_sum)

                    agent.store_experience(prev_state_obj[tl_id], action, reward, curr_state_obj, done)
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(float(loss))

                    prev_state_obj[tl_id] = curr_state_obj
                    prev_pressures[tl_id] = prs

            if done:
                break

        agent.end_episode()
        sim.close_simulation()

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        ep_reward = agent.training_history["episode_rewards"][-1] if agent.training_history["episode_rewards"] else 0.0
        print(f"Episode {ep:04d}/{args.episodes} | reward(sum)={ep_reward:.2f} | avg_loss={avg_loss:.4f} | eps={agent.epsilon:.3f}")

        if ep % int(args.save_every) == 0:
            agent.save_model(out_path)

    agent.save_model(out_path)
    print(f"\nDone. Saved to: {out_path}")


if __name__ == "__main__":
    main()

