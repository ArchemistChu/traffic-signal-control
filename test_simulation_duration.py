#!/usr/bin/env python3
"""Test if simulations run for full duration"""

import sys
sys.path.append(".")
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig
import time

# Set dataset
SimulationConfig.set_dataset('custom')

strategies = ['FIXED_TIME', 'ADAPTIVE', 'DQN']
requested_duration = 600  # 10 minutes

print("=" * 70)
print(f"Testing if simulations run for full duration ({requested_duration}s)")
print("=" * 70)

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"Testing {strategy}")
    print(f"{'='*70}")
    
    simulator = TrafficSimulator(dataset='custom', use_gui=False)
    
    start_real = time.time()
    metrics = simulator.run_simulation(duration=requested_duration, strategy=strategy)
    elapsed_real = time.time() - start_real
    
    print(f"\nResults:")
    print(f"  Requested duration: {requested_duration}s")
    print(f"  Actual simulation time: {simulator.simulation_time:.1f}s")
    print(f"  Real time elapsed: {elapsed_real:.2f}s")
    print(f"  Total steps: {simulator.step_count}")
    print(f"  Vehicle data records: {len(simulator.vehicle_data)}")
    print(f"  Unique vehicles: {metrics.get('throughput', 'N/A')}")
    print(f"  Throughput per hour: {metrics.get('throughput_per_hour', 'N/A'):.1f} veh/h")
    print(f"  Avg waiting time: {metrics.get('avg_waiting_time', 'N/A'):.4f}s")
    print(f"  Max waiting time: {metrics.get('max_waiting_time', 'N/A'):.4f}s")
    
    # Check if simulation ran for full duration
    if simulator.simulation_time < requested_duration * 0.9:  # Allow 10% tolerance
        print(f"  ⚠️  WARNING: Simulation ended early! Only {simulator.simulation_time/requested_duration*100:.1f}% of requested duration")
    else:
        print(f"  ✓ Simulation ran for full duration")
    
    simulator.close_simulation()

