#!/usr/bin/env python3
"""Debug script to check simulation results for different strategies"""

import sys
sys.path.append(".")
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig
import pandas as pd

# Set dataset
SimulationConfig.set_dataset('custom')

strategies = ['FIXED_TIME', 'ADAPTIVE', 'DQN']

print("=" * 60)
print("DEBUGGING SIMULATION RESULTS")
print("=" * 60)

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Testing {strategy}")
    print(f"{'='*60}")
    
    simulator = TrafficSimulator(dataset='custom', use_gui=False)
    
    # Run simulation for 60 seconds
    metrics = simulator.run_simulation(duration=60, strategy=strategy)
    
    print(f"\nSimulation completed:")
    print(f"  Simulation time: {simulator.simulation_time:.1f}s")
    print(f"  Total steps: {simulator.step_count}")
    print(f"  Vehicle data records: {len(simulator.vehicle_data)}")
    
    if simulator.vehicle_data:
        df = pd.DataFrame(simulator.vehicle_data)
        print(f"\nVehicle data analysis:")
        print(f"  Unique vehicles: {df['vehicle_id'].nunique()}")
        print(f"  Total records: {len(df)}")
        
        if 'waiting_time' in df.columns:
            print(f"\nWaiting time statistics:")
            print(f"  Mean: {df['waiting_time'].mean():.4f}s")
            print(f"  Max: {df['waiting_time'].max():.4f}s")
            print(f"  Min: {df['waiting_time'].min():.4f}s")
            print(f"  Std: {df['waiting_time'].std():.4f}s")
            print(f"  Records with waiting_time > 0: {(df['waiting_time'] > 0).sum()}")
            print(f"  Records with waiting_time > 1: {(df['waiting_time'] > 1).sum()}")
            print(f"  Records with waiting_time > 5: {(df['waiting_time'] > 5).sum()}")
            
            # Check per vehicle
            vehicle_waiting = df.groupby('vehicle_id')['waiting_time'].max()
            print(f"\nPer-vehicle max waiting time:")
            print(f"  Mean: {vehicle_waiting.mean():.4f}s")
            print(f"  Max: {vehicle_waiting.max():.4f}s")
            print(f"  Vehicles with waiting > 0: {(vehicle_waiting > 0).sum()}")
            print(f"  Vehicles with waiting > 1: {(vehicle_waiting > 1).sum()}")
    
    print(f"\nMetrics from _calculate_performance_metrics:")
    print(f"  avg_waiting_time: {metrics.get('avg_waiting_time', 'N/A')}")
    print(f"  max_waiting_time: {metrics.get('max_waiting_time', 'N/A')}")
    print(f"  throughput: {metrics.get('throughput', 'N/A')}")
    print(f"  throughput_per_hour: {metrics.get('throughput_per_hour', 'N/A')}")
    
    simulator.close_simulation()

