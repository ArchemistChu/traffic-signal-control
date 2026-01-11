#!/usr/bin/env python3
"""
SUMO simulation environment test script
Validates correctness of all configuration files and basic functionality of TrafficSimulator
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.traffic_simulator import TrafficSimulator
except ImportError:
    print("Unable to import TrafficSimulator class, please check file path")
    sys.exit(1)

def test_file_existence():
    """Test if required files exist"""
    print("=== Checking configuration files ===")
    
    required_files = [
        "intersection.net.xml",
        "routes.rou.xml", 
        "traffic_lights.add.xml",
        "simulation.sumocfg"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} - exists")
        else:
            print(f"✗ {file_path} - missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nError: Missing configuration files: {missing_files}")
        return False
    
    print("All configuration files checked successfully!")
    return True

def test_xml_validation():
    """Validate XML file format"""
    print("\n=== Validating XML file format ===")
    
    import xml.etree.ElementTree as ET
    
    xml_files = [
        "intersection.net.xml",
        "routes.rou.xml",
        "traffic_lights.add.xml", 
        "simulation.sumocfg"
    ]
    
    for xml_file in xml_files:
        try:
            ET.parse(xml_file)
            print(f"✓ {xml_file} - XML format correct")
        except ET.ParseError as e:
            print(f"✗ {xml_file} - XML format error: {e}")
            return False
        except FileNotFoundError:
            print(f"✗ {xml_file} - file does not exist")
            return False
    
    print("All XML file formats validated successfully!")
    return True

def test_sumo_installation():
    """Test SUMO installation"""
    print("\n=== Checking SUMO installation ===")
    
    try:
        import traci
        import sumolib
        print("✓ TraCI and sumolib imported successfully")
    except ImportError as e:
        print(f"✗ SUMO Python library import failed: {e}")
        print("Please install SUMO Python library: pip install traci sumolib")
        return False
    
    # Test SUMO executable
    import subprocess
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"✓ SUMO executable available: {version}")
        else:
            print("✗ SUMO executable not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"✗ SUMO executable check failed: {e}")
        print("Please ensure SUMO is installed and added to system PATH")
        return False
    
    return True

def test_basic_simulation():
    """Test basic simulation functionality"""
    print("\n=== Testing basic simulation functionality ===")
    
    try:
        # Create simulator instance (without GUI)
        simulator = TrafficSimulator(use_gui=False)
        print("✓ TrafficSimulator instance created successfully")
        
        # Start simulation
        if not simulator.start_simulation():
            print("✗ Simulation startup failed")
            return False
        print("✓ SUMO simulation started successfully")
        
        # Run several simulation steps
        step_count = 0
        max_steps = 50
        
        print("Starting simulation steps...")
        while simulator.simulation_step() and step_count < max_steps:
            step_count += 1
            if step_count % 10 == 0:
                print(f"  Completed {step_count} steps")
        
        print(f"✓ Completed {step_count} simulation steps")
        
        # Test state retrieval
        state = simulator.get_current_state()
        if state:
            print("✓ Simulation state retrieved successfully")
            print(f"  Current simulation time: {state.get('time', 0):.1f}s")
            print(f"  Lane vehicle count: {sum(state.get('lane_vehicle_counts', {}).values())}")
        else:
            print("✗ Simulation state retrieval failed")
        
        # Close simulation
        simulator.close_simulation()
        print("✓ Simulation closed normally")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic simulation test failed: {e}")
        return False

def test_traffic_light_control():
    """Test traffic light control functionality"""
    print("\n=== Testing traffic light control functionality ===")
    
    try:
        simulator = TrafficSimulator(use_gui=False)
        
        if not simulator.start_simulation():
            print("✗ Simulation startup failed")
            return False
        
        # Run a few steps to stabilize simulation
        for _ in range(10):
            if not simulator.simulation_step():
                break
        
        # Test traffic light phase switching
        original_phase = None
        state = simulator.get_current_state()
        if state:
            original_phase = state.get('traffic_light_phase', 0)
            print(f"Original traffic light phase: {original_phase}")
        
        # Switch to different phase
        test_phase = (original_phase + 1) % 4 if original_phase is not None else 1
        simulator.set_traffic_light_phase(test_phase)
        
        # Run a few steps to check if phase changed
        for _ in range(5):
            simulator.simulation_step()
        
        new_state = simulator.get_current_state()
        if new_state:
            new_phase = new_state.get('traffic_light_phase', 0)
            print(f"New traffic light phase: {new_phase}")
            
            if new_phase == test_phase:
                print("✓ Traffic light control functionality normal")
                result = True
            else:
                print("✗ Traffic light phase did not switch correctly")
                result = False
        else:
            print("✗ Unable to retrieve traffic light state")
            result = False
        
        simulator.close_simulation()
        return result
        
    except Exception as e:
        print(f"✗ Traffic light control test failed: {e}")
        return False

def test_data_collection():
    """Test data collection functionality"""
    print("\n=== Testing data collection functionality ===")
    
    try:
        simulator = TrafficSimulator(use_gui=False)
        
        if not simulator.start_simulation():
            print("✗ Simulation startup failed")
            return False
        
        # Run long enough to generate some data
        step_count = 0
        while simulator.simulation_step() and step_count < 100:
            step_count += 1
        
        # Check data collection
        vehicle_data_count = len(simulator.vehicle_data)
        tl_data_count = len(simulator.traffic_light_data)
        detector_data_count = len(simulator.detector_data)
        
        print(f"Collected data:")
        print(f"  Vehicle data records: {vehicle_data_count}")
        print(f"  Traffic light data records: {tl_data_count}")
        print(f"  Detector data records: {detector_data_count}")
        
        # Calculate performance metrics
        metrics = simulator._calculate_performance_metrics()
        if metrics:
            print("✓ Performance metrics calculated successfully")
            print(f"  Average waiting time: {metrics.get('avg_waiting_time', 0):.2f}s")
            print(f"  Vehicle throughput: {metrics.get('throughput', 0)}")
        else:
            print("✗ Performance metrics calculation failed")
        
        simulator.close_simulation()
        
        # Check if there's sufficient data
        if vehicle_data_count > 0 and tl_data_count > 0:
            print("✓ Data collection functionality normal")
            return True
        else:
            print("✗ Insufficient data collection")
            return False
            
    except Exception as e:
        print(f"✗ Data collection test failed: {e}")
        return False

def run_full_test():
    """Run full test suite"""
    print("SUMO Traffic Simulation Environment Test")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("File existence check", test_file_existence),
        ("XML format validation", test_xml_validation), 
        ("SUMO installation check", test_sumo_installation),
        ("Basic simulation functionality", test_basic_simulation),
        ("Traffic light control", test_traffic_light_control),
        ("Data collection functionality", test_data_collection)
    ]
    
    for test_name, test_func in tests:
        print(f"\nStarting test: {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"Test exception: {e}")
            test_results.append((test_name, False))
        
        duration = time.time() - start_time
        print(f"Test completed, duration: {duration:.2f}s")
    
    # Summarize test results
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\nTotal: {passed_tests}/{len(test_results)} tests passed")
    
    if passed_tests == len(test_results):
        print("\n🎉 All tests passed! SUMO simulation environment configured correctly.")
        print("You can now start developing signal control and reinforcement learning modules.")
        return True
    else:
        print(f"\n⚠️  {len(test_results) - passed_tests} tests failed.")
        print("Please fix issues according to the error messages above and rerun the tests.")
        return False

if __name__ == "__main__":
    # Switch to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Current working directory: {os.getcwd()}")
    
    success = run_full_test()
    sys.exit(0 if success else 1)
