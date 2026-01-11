#!/usr/bin/env python3
"""
Quick test script for TAPASCologne dataset
Tests the simulation with the new configuration
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SimulationConfig
from src.traffic_simulator import TrafficSimulator

def test_tapas_cologne():
    """Test TAPASCologne dataset"""
    print("=" * 60)
    print("Testing TAPASCologne Dataset")
    print("=" * 60)
    
    # Set to TAPASCologne dataset
    SimulationConfig.set_dataset('tapas_cologne')
    
    print(f"\nConfiguration:")
    config = SimulationConfig.get_current_config()
    print(f"  Description: {config['description']}")
    print(f"  Config file: {config['sumo_config']}")
    print(f"  Control mode: {config['control_mode']}")
    print(f"  Max controlled lights: {config.get('max_controlled_lights', 'N/A')}")
    
    # Create simulator
    print("\nInitializing simulator...")
    simulator = TrafficSimulator(use_gui=False, dataset='tapas_cologne')
    
    # Start simulation to test initialization
    print("\nStarting simulation...")
    if simulator.start_simulation():
        print(f"\n✓ Successfully connected to SUMO")
        print(f"  Total traffic lights in network: {len(simulator.traffic_lights)}")
        print(f"  Controlled traffic lights: {len(simulator.controlled_traffic_lights)}")
        print(f"  First 5 controlled lights: {simulator.controlled_traffic_lights[:5]}")
        print(f"  Total lanes detected: {len(simulator.lanes)}")
        print(f"  Total detectors: {len(simulator.detectors)}")
        
        # Run for a few steps
        print("\nRunning 10 simulation steps...")
        for i in range(10):
            if not simulator.simulation_step():
                break
            if (i + 1) % 5 == 0:
                print(f"  Step {i + 1} - Time: {simulator.simulation_time}s")
        
        # Get a sample state
        print("\nGetting sample traffic state...")
        state = simulator.get_current_state()
        print(f"  State keys: {list(state.keys())}")
        print(f"  Traffic light: {state.get('traffic_light_id')}")
        print(f"  Current phase: {state.get('traffic_light_phase')}")
        print(f"  Lanes with vehicles: {sum(1 for v in state.get('lane_vehicle_counts', {}).values() if v > 0)}")
        
        simulator.close_simulation()
        print("\n✓ Test completed successfully!")
        return True
    else:
        print("\n✗ Failed to start simulation")
        return False

def test_custom():
    """Test custom dataset"""
    print("\n" + "=" * 60)
    print("Testing Custom Single Intersection Dataset")
    print("=" * 60)
    
    # Set to custom dataset
    SimulationConfig.set_dataset('custom')
    
    print(f"\nConfiguration:")
    config = SimulationConfig.get_current_config()
    print(f"  Description: {config['description']}")
    print(f"  Config file: {config['sumo_config']}")
    print(f"  Control mode: {config['control_mode']}")
    print(f"  Predefined traffic lights: {config.get('traffic_lights', [])}")
    
    # Check if config file exists
    if not os.path.exists(config['sumo_config']):
        print(f"\n⚠ Config file not found: {config['sumo_config']}")
        print("  Skipping custom dataset test")
        return False
    
    print("\n✓ Custom dataset configuration loaded")
    return True

def main():
    """Main test function"""
    print("\n🚦 Traffic Simulation Dataset Test\n")
    
    # Test TAPASCologne
    tapas_success = test_tapas_cologne()
    
    # Test custom
    custom_success = test_custom()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"TAPASCologne: {'✓ PASS' if tapas_success else '✗ FAIL'}")
    print(f"Custom:       {'✓ PASS' if custom_success else '✗ FAIL'}")
    print("\nTo use TAPASCologne dataset in your code:")
    print("  from src.config import SimulationConfig")
    print("  SimulationConfig.set_dataset('tapas_cologne')")
    print("  simulator = TrafficSimulator(dataset='tapas_cologne')")
    print()

if __name__ == "__main__":
    main()
