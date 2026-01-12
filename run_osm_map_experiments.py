#!/usr/bin/env python3
"""
OSM Map Experiment Runner
Run experiments on OSM-based maps (Cologne, Vancouver, Palo Alto)
"""

import sys
import os
sys.path.append(".")

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator


def run_single_experiment(map_name: str, strategy: str, duration: int = 600, use_gui: bool = False):
    """
    Run a single experiment on a specified map
          

           
    Args:
        map_name: 'cologne', 'vancouver', or 'palo_alto'
        strategy: 'FIXED_TIME', 'ADAPTIVE', 'MAX_PRESSURE', or 'DQN'
        duration: Simulation duration in seconds
        use_gui: Whether to use SUMO GUI
    """
    print("=" * 80)
    print(f"Running Experiment: {map_name.upper()} - {strategy}")
    print("=" * 80)
    
    # Set dataset
    print(f"\n[1] Setting dataset to {map_name}...")
    try:
        SimulationConfig.set_dataset(map_name)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available maps: cologne, vancouver, palo_alto")
        return None
    
    # Create simulator
    print(f"\n[2] Creating simulator...")
    print(f"    GUI: {'Yes' if use_gui else 'No'}")
    simulator = TrafficSimulator(use_gui=use_gui, dataset=map_name)
    
    # Run simulation
    print(f"\n[3] Running simulation...")
    print(f"    Strategy: {strategy}")
    print(f"    Duration: {duration} seconds ({duration/60:.1f} minutes)")
    
    try:
        metrics = simulator.run_simulation(duration=duration, strategy=strategy)
        
        # Display results
        print(f"\n[4] Simulation Results:")
        print("-" * 80)
        
        if metrics:
            print(f"  Total vehicles processed: {metrics.get('throughput', 'N/A')}")
            print(f"  Throughput per hour: {metrics.get('throughput_per_hour', 0):.0f} vehicles/hour")
            print(f"  Average waiting time: {metrics.get('avg_waiting_time', 0):.2f} seconds")
            print(f"  Average speed: {metrics.get('avg_speed', 0):.2f} m/s ({metrics.get('avg_speed', 0)*3.6:.2f} km/h)")
            
            if 'total_co2' in metrics:
                print(f"  Total CO2 emissions: {metrics['total_co2']:.2f} mg")
            if 'total_fuel' in metrics:
                print(f"  Total fuel consumption: {metrics['total_fuel']:.2f} ml")
            
            if 'congestion_index' in metrics:
                print(f"  Congestion index: {metrics['congestion_index']:.3f}")
            
            return metrics
        else:
            print("  No metrics available")
            return None
            
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        simulator.close_simulation()


def get_map_info(map_name: str):
    """
    Get information about a map (traffic lights, lanes, etc.)
    
    Args:
        map_name: 'cologne', 'vancouver', or 'palo_alto'
    """
    print("=" * 80)
    print(f"Map Information: {map_name.upper()}")
    print("=" * 80)
    
    # Set dataset
    try:
        SimulationConfig.set_dataset(map_name)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create simulator (no GUI for faster startup)
    simulator = TrafficSimulator(use_gui=False, dataset=map_name)
    
    if simulator.start_simulation():
        print(f"\nNetwork Statistics:")
        print(f"  Total traffic lights: {len(simulator.traffic_lights)}")
        print(f"  Controlled traffic lights: {len(simulator.controlled_traffic_lights)}")
        print(f"  Total lanes: {len(simulator.lanes)}")
        print(f"  Total detectors: {len(simulator.detectors)}")
        
        if simulator.controlled_traffic_lights:
            print(f"\nFirst 10 controlled traffic lights:")
            for i, tl_id in enumerate(simulator.controlled_traffic_lights[:10], 1):
                try:
                    lanes = simulator._get_lanes_for_traffic_light(tl_id)
                    print(f"  {i}. {tl_id} - {len(lanes)} lanes")
                except:
                    print(f"  {i}. {tl_id}")
        
        simulator.close_simulation()
        print("\n✓ Map information retrieved successfully")
    else:
        print("\n✗ Failed to start simulation")


def run_comparison_experiment(maps: list, strategies: list, duration: int = 600):
    """
    Run comparison experiments across multiple maps and strategies
    
    Args:
        maps: List of map names ['cologne', 'vancouver', 'palo_alto']
        strategies: List of strategies ['FIXED_TIME', 'ADAPTIVE', 'MAX_PRESSURE', 'DQN']
        duration: Simulation duration in seconds
    """
    print("=" * 80)
    print("COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Maps: {', '.join(maps)}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Duration: {duration} seconds per run")
    print(f"Total runs: {len(maps) * len(strategies)}")
    print("=" * 80)
    
    results = {}
    
    for map_name in maps:
        results[map_name] = {}
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Running: {map_name.upper()} - {strategy}")
            print(f"{'='*80}")
            
            metrics = run_single_experiment(map_name, strategy, duration, use_gui=False)
            results[map_name][strategy] = metrics
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for map_name in maps:
        print(f"\n{map_name.upper()}:")
        print("-" * 80)
        for strategy in strategies:
            metrics = results[map_name].get(strategy, {})
            if metrics:
                avg_wait = metrics.get('avg_waiting_time', 0)
                throughput = metrics.get('throughput_per_hour', 0)
                print(f"  {strategy:15s}: Avg Wait={avg_wait:6.2f}s, Throughput={throughput:8.0f} veh/h")
            else:
                print(f"  {strategy:15s}: Failed or No Data")
    
    return results


def main():
    """Main function - interactive menu"""
    
    print("\n" + "=" * 80)
    print("OSM Map Experiment Runner")
    print("=" * 80)
    print("\nAvailable maps:")
    print("  1. cologne")
    print("  2. vancouver")
    print("  3. palo_alto")
    print("\nAvailable strategies:")
    print("  - FIXED_TIME")
    print("  - ADAPTIVE")
    print("  - MAX_PRESSURE")
    print("  - DQN")
    print("\nOptions:")
    print("  1. Run single experiment (choose map and strategy)")
    print("  2. Get map information (no simulation)")
    print("  3. Run comparison (multiple maps and strategies)")
    print("  4. Quick test (cologne, FIXED_TIME, 60 seconds)")
    
    choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        map_name = input("Enter map name (cologne/vancouver/palo_alto): ").strip().lower()
        strategy = input("Enter strategy (FIXED_TIME/ADAPTIVE/MAX_PRESSURE/DQN): ").strip().upper()
        duration_str = input("Enter duration in seconds (default 600): ").strip()
        duration = int(duration_str) if duration_str else 600
        use_gui_str = input("Use GUI? (y/n, default n): ").strip().lower()
        use_gui = use_gui_str == 'y'
        
        run_single_experiment(map_name, strategy, duration, use_gui)
        
    elif choice == '2':
        map_name = input("Enter map name (cologne/vancouver/palo_alto): ").strip().lower()
        get_map_info(map_name)
        
    elif choice == '3':
        maps_str = input("Enter map names (comma-separated, e.g., cologne,vancouver): ").strip()
        maps = [m.strip().lower() for m in maps_str.split(',')]
        strategies_str = input("Enter strategies (comma-separated, e.g., FIXED_TIME,ADAPTIVE): ").strip()
        strategies = [s.strip().upper() for s in strategies_str.split(',')]
        duration_str = input("Enter duration in seconds (default 600): ").strip()
        duration = int(duration_str) if duration_str else 600
        
        run_comparison_experiment(maps, strategies, duration)
        
    elif choice == '4':
        print("\nRunning quick test: cologne, FIXED_TIME, 60 seconds...")
        run_single_experiment('cologne', 'FIXED_TIME', 60, use_gui=False)
        
    elif choice.lower() == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
