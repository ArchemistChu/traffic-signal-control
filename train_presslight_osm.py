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


def build_region_map(tls_ids: List[str], grid_size: float) -> Dict[str, tuple[int, int]]:
    """Assign each TLS to a spatial grid cell for regional congestion shaping."""
    region_map: Dict[str, tuple[int, int]] = {}
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
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--max-controlled-lights", type=int, default=5)
    parser.add_argument(
        "--controlled-lights-ratio",
        type=float,
        default=0.0,
        help="If >0, use this ratio of total TLS (overrides --max-controlled-lights). Example: 0.2 for 20%",
    )
    parser.add_argument(
        "--regional-reward-weight",
        type=float,
        default=0.0,
        help="If >0, add a regional congestion penalty to the reward (weight on avg pressure in region)",
    )
    parser.add_argument(
        "--region-grid-size",
        type=float,
        default=500.0,
        help="Grid size in meters for regional congestion (default: 500)",
    )
    parser.add_argument("--k", type=int, default=8, help="Pad/truncate number of phase-pressure features")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="", help="Output checkpoint path")
    args = parser.parse_args()

    if traci is None:
        raise RuntimeError("traci is not installed. Please install SUMO tools: pip install traci sumolib")

    if args.controlled_lights_ratio < 0.0 or args.controlled_lights_ratio > 1.0:
        raise ValueError("--controlled-lights-ratio must be in [0.0, 1.0]")

    np.random.seed(args.seed)

    SimulationConfig.set_dataset(args.dataset)
    if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
        # Use a large cap so we can compute ratio from full TLS list.
        SimulationConfig.CONFIGS[args.dataset]["max_controlled_lights"] = 10**9
    else:
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
    if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
        print(f"tls=ratio {args.controlled_lights_ratio:.2f} K={args.k} out={out_path}")
    else:
        print(f"tls={args.max_controlled_lights} K={args.k} out={out_path}")
    if args.regional_reward_weight and args.regional_reward_weight > 0.0:
        print(f"Regional reward: weight={args.regional_reward_weight} grid={args.region_grid_size}m")
    print("=" * 78)

    fake_lanes = [f"p{i}" for i in range(int(args.k))]

    for ep in range(1, args.episodes + 1):
        sim = TrafficSimulator(use_gui=False, dataset=args.dataset, enable_sumo_emissions_output=False)
        ok = sim.start_simulation()
        if not ok or not sim.controlled_traffic_lights:
            sim.close_simulation()
            raise RuntimeError("Failed to start SUMO or no traffic lights detected.")

        if args.controlled_lights_ratio and args.controlled_lights_ratio > 0.0:
            total_tls = len(sim.traffic_lights) if sim.traffic_lights else len(sim.controlled_traffic_lights)
            desired = int(round(total_tls * float(args.controlled_lights_ratio)))
            desired = max(1, min(total_tls, desired))
            tls_ids = (sim.traffic_lights or sim.controlled_traffic_lights)[:desired]
            if ep == 1:
                print(f"Computed TLS count: {desired}/{total_tls} ({args.controlled_lights_ratio:.2f})")
        else:
            tls_ids = sim.controlled_traffic_lights[: int(args.max_controlled_lights)]
        last_decision_t = 0.0

        tl_regions: Dict[str, tuple[int, int]] = {}
        if args.regional_reward_weight and args.regional_reward_weight > 0.0:
            tl_regions = build_region_map(tls_ids, float(args.region_grid_size))

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
        last_progress_t = -1.0
        progress_every = 100.0  # sim-seconds
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

                curr_state_obj: Dict[str, RLTState] = {}
                curr_pressures: Dict[str, List[float]] = {}
                curr_sums: Dict[str, float] = {}
                actions: Dict[str, int] = {}

                for tl_id in tls_ids:
                    # Build state
                    prs = compute_pressures(tl_id, state)
                    curr_pressures[tl_id] = prs
                    sumo_state = {
                        "time": float(sim.simulation_time),
                        "lane_vehicle_counts": {fake_lanes[i]: prs[i] for i in range(int(args.k))},
                        "lane_queue_lengths": {fake_lanes[i]: prs[i] for i in range(int(args.k))},
                        "lane_mean_speeds": {fake_lanes[i]: 0.0 for i in range(int(args.k))},
                        "detector_occupancy": {},
                        "traffic_light_phase": int(traci.trafficlight.getPhase(tl_id)) % 8,
                        "traffic_light_remaining_time": float(max(0.0, traci.trafficlight.getNextSwitch(tl_id) - sim.simulation_time)),
                    }
                    curr_state_obj[tl_id] = RLTState(sumo_state, lane_list=fake_lanes)
                    curr_sums[tl_id] = float(sum(prs))

                    # Action = choose index in green phases list
                    actions[tl_id] = int(agent.select_action(prev_state_obj[tl_id], training=True))

                # Apply actions
                for tl_id in tls_ids:
                    greens = tl_green_phases(tl_id)
                    if greens:
                        action = actions[tl_id]
                        target_phase = int(greens[action % len(greens)])
                        try:
                            cur_p = int(traci.trafficlight.getPhase(tl_id))
                            if target_phase != cur_p and (traci.trafficlight.getNextSwitch(tl_id) - sim.simulation_time) <= 1.0:
                                traci.trafficlight.setPhase(tl_id, target_phase)
                                traci.trafficlight.setPhaseDuration(tl_id, 30.0)
                        except Exception:
                            pass

                # Regional congestion (avg pressure per region)
                region_sum: Dict[tuple[int, int], float] = {}
                region_cnt: Dict[tuple[int, int], int] = {}
                if tl_regions:
                    for tl_id in tls_ids:
                        region = tl_regions.get(tl_id, (0, 0))
                        region_sum[region] = region_sum.get(region, 0.0) + float(curr_sums.get(tl_id, 0.0))
                        region_cnt[region] = region_cnt.get(region, 0) + 1

                # Reward + training
                for tl_id in tls_ids:
                    prev_sum = float(sum(prev_pressures[tl_id]))
                    curr_sum = float(curr_sums[tl_id])
                    reward = -(curr_sum) + 0.3 * (prev_sum - curr_sum)
                    if tl_regions:
                        region = tl_regions.get(tl_id, (0, 0))
                        denom = max(1, region_cnt.get(region, 1))
                        region_avg = region_sum.get(region, 0.0) / float(denom)
                        reward += -float(args.regional_reward_weight) * region_avg

                    agent.store_experience(prev_state_obj[tl_id], actions[tl_id], reward, curr_state_obj[tl_id], done)
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(float(loss))

                    prev_state_obj[tl_id] = curr_state_obj[tl_id]
                    prev_pressures[tl_id] = curr_pressures[tl_id]

            if done:
                break

            # Lightweight progress log (avoid noisy per-step prints)
            if float(sim.simulation_time) - last_progress_t >= progress_every:
                last_progress_t = float(sim.simulation_time)
                print(f"  sim_t={last_progress_t:.0f}s / {args.duration}s")

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

