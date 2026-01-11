#!/usr/bin/env python3
"""Analyze waiting time calculation issue"""

import sys
sys.path.append(".")
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig
import pandas as pd
import numpy as np

# Set dataset
SimulationConfig.set_dataset('custom')

strategies = ['FIXED_TIME', 'ADAPTIVE', 'DQN']

print("=" * 70)
print("ANALYZING WAITING TIME CALCULATION")
print("=" * 70)

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"Testing {strategy}")
    print(f"{'='*70}")
    
    simulator = TrafficSimulator(dataset='custom', use_gui=False)
    
    # Run simulation for 60 seconds (shorter for debugging)
    metrics = simulator.run_simulation(duration=60, strategy=strategy)
    
    print(f"\nSimulation completed:")
    print(f"  Simulation time: {simulator.simulation_time:.1f}s")
    print(f"  Total vehicle data records: {len(simulator.vehicle_data)}")
    
    if simulator.vehicle_data:
        df = pd.DataFrame(simulator.vehicle_data)
        
        print(f"\nRaw data analysis:")
        print(f"  Total records: {len(df)}")
        print(f"  Unique vehicles: {df['vehicle_id'].nunique()}")
        
        if 'waiting_time' in df.columns:
            print(f"\nWaiting time column statistics:")
            print(f"  Mean: {df['waiting_time'].mean():.4f}s")
            print(f"  Max: {df['waiting_time'].max():.4f}s")
            print(f"  Min: {df['waiting_time'].min():.4f}s")
            print(f"  Std: {df['waiting_time'].std():.4f}s")
            print(f"  Non-zero count: {(df['waiting_time'] > 0).sum()}")
            print(f"  Zero count: {(df['waiting_time'] == 0).sum()}")
            print(f"  > 1s count: {(df['waiting_time'] > 1).sum()}")
            print(f"  > 5s count: {(df['waiting_time'] > 5).sum()}")
            
            # Check per-vehicle cumulative waiting time
            print(f"\nPer-vehicle analysis:")
            vehicle_max_wait = df.groupby('vehicle_id')['waiting_time'].max()
            vehicle_sum_wait = df.groupby('vehicle_id')['waiting_time'].sum()
            
            print(f"  Max waiting time per vehicle:")
            print(f"    Mean: {vehicle_max_wait.mean():.4f}s")
            print(f"    Max: {vehicle_max_wait.max():.4f}s")
            print(f"    Vehicles with max_wait > 0: {(vehicle_max_wait > 0).sum()}")
            print(f"    Vehicles with max_wait > 1s: {(vehicle_max_wait > 1).sum()}")
            
            print(f"  Total waiting time per vehicle:")
            print(f"    Mean: {vehicle_sum_wait.mean():.4f}s")
            print(f"    Max: {vehicle_sum_wait.max():.4f}s")
            print(f"    Vehicles with total_wait > 0: {(vehicle_sum_wait > 0).sum()}")
            
            # Show sample of vehicles with highest waiting time
            print(f"\nTop 5 vehicles by max waiting time:")
            top_vehicles = vehicle_max_wait.nlargest(5)
            for veh_id, max_wait in top_vehicles.items():
                veh_data = df[df['vehicle_id'] == veh_id]
                print(f"  {veh_id}: max_wait={max_wait:.4f}s, records={len(veh_data)}, "
                      f"avg_speed={veh_data['speed'].mean():.2f}m/s, "
                      f"min_speed={veh_data['speed'].min():.2f}m/s")
            
            # Check if there are vehicles with low speed but zero waiting time
            print(f"\nVehicles with low speed but zero waiting time:")
            low_speed = df[(df['speed'] < 0.5) & (df['waiting_time'] == 0)]
            if len(low_speed) > 0:
                print(f"  Found {len(low_speed)} records with speed < 0.5 m/s but waiting_time = 0")
                print(f"  Sample vehicle IDs: {low_speed['vehicle_id'].unique()[:5].tolist()}")
            else:
                print(f"  None found")
        
        # Check speed distribution
        if 'speed' in df.columns:
            print(f"\nSpeed statistics:")
            print(f"  Mean: {df['speed'].mean():.2f} m/s")
            print(f"  Min: {df['speed'].min():.2f} m/s")
            print(f"  Records with speed < 0.1 m/s: {(df['speed'] < 0.1).sum()}")
            print(f"  Records with speed < 0.5 m/s: {(df['speed'] < 0.5).sum()}")
            print(f"  Records with speed == 0 m/s: {(df['speed'] == 0).sum()}")
    
    print(f"\nMetrics from _calculate_performance_metrics:")
    print(f"  avg_waiting_time: {metrics.get('avg_waiting_time', 'N/A')}")
    print(f"  max_waiting_time: {metrics.get('max_waiting_time', 'N/A')}")
    print(f"  throughput: {metrics.get('throughput', 'N/A')}")
    print(f"  throughput_per_hour: {metrics.get('throughput_per_hour', 'N/A'):.1f} veh/h")
    
    simulator.close_simulation()

