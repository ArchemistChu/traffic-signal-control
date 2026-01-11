#!/usr/bin/env python3
"""
Example: Running simulation with TAPASCologne dataset
This script demonstrates how to switch to and use the TAPASCologne dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator

def example_tapas_cologne_simulation():
    """Example: Run a simple simulation with TAPASCologne dataset"""
    
    print("=" * 70)
    print("Example: TAPASCologne Dataset Simulation")
    print("=" * 70)
    
    # Step 1: Set the dataset to TAPASCologne
    print("\n[1] Setting dataset to TAPASCologne...")
    SimulationConfig.set_dataset('tapas_cologne')
    
    # Step 2: Create simulator with GUI
    print("\n[2] Creating simulator with GUI...")
    print("    Note: This will open SUMO GUI showing the real Cologne network")
    simulator = TrafficSimulator(use_gui=True, dataset='tapas_cologne')
    
    # Step 3: Run simulation with different strategies
    print("\n[3] Running simulation...")
    print("    Strategy: FIXED_TIME")
    print("    Duration: 600 seconds (10 minutes)")
    print("    This will take a while - watch the SUMO GUI!")
    
    metrics = simulator.run_simulation(
        duration=600,  # 10 minutes
        strategy='FIXED_TIME'
    )
    
    # Step 4: Display results
    print("\n[4] Simulation Results:")
    print("-" * 70)
    
    if metrics:
        print(f"  Total vehicles processed: {metrics.get('throughput', 'N/A')}")
        print(f"  Average waiting time: {metrics.get('avg_waiting_time', 0):.2f} seconds")
        print(f"  Average speed: {metrics.get('avg_speed', 0):.2f} m/s")
        print(f"  Throughput per hour: {metrics.get('throughput_per_hour', 0):.0f} vehicles/hour")
        
        if 'total_co2' in metrics:
            print(f"  Total CO2 emissions: {metrics['total_co2']:.2f} mg")
        if 'total_fuel' in metrics:
            print(f"  Total fuel consumption: {metrics['total_fuel']:.2f} ml")
        
        print("\n  Data exported to: output/")
        print("    - vehicle_data.csv")
        print("    - traffic_light_data.csv")
        print("    - detector_data.csv")
        print("    - summary_report.md")
    else:
        print("  No metrics available")
    
    print("\n" + "=" * 70)
    print("Simulation completed!")
    print("=" * 70)

def example_compare_datasets():
    """Example: Quick comparison of dataset configurations"""
    
    print("\n" + "=" * 70)
    print("Dataset Comparison")
    print("=" * 70)
    
    for dataset in ['custom', 'tapas_cologne']:
        SimulationConfig.set_dataset(dataset)
        config = SimulationConfig.get_current_config()
        
        print(f"\n{dataset.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Config file: {config['sumo_config']}")
        print(f"  Control mode: {config['control_mode']}")
        
        if config['traffic_lights']:
            print(f"  Traffic lights: {config['traffic_lights']}")
        else:
            print(f"  Traffic lights: Auto-detected (max {config.get('max_controlled_lights', 'N/A')} controlled)")
        
        if config['lanes']:
            print(f"  Lanes: {len(config['lanes'])} predefined")
        else:
            print(f"  Lanes: Auto-detected")

def example_get_network_info():
    """Example: Get information about the TAPASCologne network"""
    
    print("\n" + "=" * 70)
    print("TAPASCologne Network Information")
    print("=" * 70)
    
    SimulationConfig.set_dataset('tapas_cologne')
    simulator = TrafficSimulator(use_gui=False)
    
    if simulator.start_simulation():
        print(f"\nNetwork Statistics:")
        print(f"  Total traffic lights: {len(simulator.traffic_lights)}")
        print(f"  Controlled traffic lights: {len(simulator.controlled_traffic_lights)}")
        print(f"  Total lanes: {len(simulator.lanes)}")
        print(f"  Total detectors: {len(simulator.detectors)}")
        
        print(f"\nFirst 10 controlled traffic lights:")
        for i, tl_id in enumerate(simulator.controlled_traffic_lights[:10], 1):
            lanes = simulator._get_lanes_for_traffic_light(tl_id)
            print(f"  {i}. {tl_id} - {len(lanes)} lanes")
        
        # Get state for first traffic light
        if simulator.controlled_traffic_lights:
            tl_id = simulator.controlled_traffic_lights[0]
            state = simulator.get_current_state(tl_id)
            
            print(f"\nSample state for traffic light '{tl_id}':")
            print(f"  Current phase: {state.get('traffic_light_phase', 'N/A')}")
            print(f"  Lanes with vehicles: {len([v for v in state.get('lane_vehicle_counts', {}).values() if v > 0])}")
            print(f"  Total lanes monitored: {len(state.get('lane_vehicle_counts', {}))}")
        
        simulator.close_simulation()
        print("\n✓ Network information retrieved successfully")
    else:
        print("\n✗ Failed to start simulation")

def main():
    """Main function - choose which example to run"""
    
    print("\n🚦 TAPASCologne Dataset Examples\n")
    print("Available examples:")
    print("  1. Run full simulation with TAPASCologne (with GUI)")
    print("  2. Compare dataset configurations")
    print("  3. Get TAPASCologne network information (no GUI)")
    print("  4. Run all examples")
    
    choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        example_tapas_cologne_simulation()
    elif choice == '2':
        example_compare_datasets()
    elif choice == '3':
        example_get_network_info()
    elif choice == '4':
        example_compare_datasets()
        example_get_network_info()
        print("\n\n⚠ Skipping full simulation (option 1) to save time.")
        print("   Run option 1 separately to see the full simulation with GUI.")
    elif choice.lower() == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
