#!/usr/bin/env python3
"""
Traffic Simulation Engine Module (TrafficSimulator)
Traffic simulation using SUMO and TraCI API
Complete implementation of all features from original requirements
Supports both custom single intersection and TAPASCologne dataset
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

# Import configuration
try:
    from src.config import SimulationConfig
except ImportError:
    from config import SimulationConfig

# SUMO TraCI imports
try:
    import traci
    import sumolib
except ImportError:
    print("Warning: SUMO TraCI library not installed. Please install: pip install traci sumolib")
    traci = None
    sumolib = None

class TrafficSimulator:
    """
    Traffic simulation engine class
    Manages SUMO simulation startup, control, and data collection
    Complete implementation of all features from original requirements
    """
    
    def __init__(self, config_file: str = None, 
                 use_gui: bool = False, port: int = 8813, dataset: str = None):
        """
        Initialize traffic simulator
        
        Args:
            config_file: SUMO configuration file path (None for auto from SimulationConfig)
            use_gui: whether to use graphical interface
            port: TraCI connection port
            dataset: 'custom', 'cologne', 'vancouver', or 'palo_alto' (None for current config)
        """
        # Set dataset if specified
        if dataset:
            SimulationConfig.set_dataset(dataset)
        
        # Get configuration
        self.sim_config = SimulationConfig.get_current_config()
        
        # Use config file from SimulationConfig if not provided
        self.config_file = config_file or SimulationConfig.get_sumo_config_path()
        self.use_gui = use_gui
        self.port = port
        self.is_connected = False
        self.simulation_time = 0
        self.step_count = 0
        self.requested_duration = None  # Store requested duration for fair comparison
        
        # Simulation data storage
        self.vehicle_data = []
        self.traffic_light_data = []
        self.detector_data = []
        self.emission_data = []
        self.performance_metrics = {}
        
        # Traffic lights, lanes, and detectors (will be detected or loaded from config)
        self.traffic_lights = []
        self.controlled_traffic_lights = []  # Subset to control
        self.lanes = []
        self.detectors = []
        
        # Legacy intersection ID (for backward compatibility)
        self.intersection_id = "intersection"
        
        # Track original working directory and config directory
        self.original_cwd = os.getcwd()
        self.config_dir = os.path.dirname(os.path.abspath(self.config_file))
        
        print(f"Initialized simulator with dataset: {SimulationConfig.DATASET}")
        print(f"Config file: {self.config_file}")
        print(f"Control mode: {self.sim_config['control_mode']}")
    
    def _find_sumo_binary(self) -> Optional[str]:
        """Find SUMO executable file path"""
        import shutil
        
        # Determine executable file name to search for
        binary_name = "sumo-gui" if self.use_gui else "sumo"
        
        # Method 1: Try using system PATH
        sumo_path = shutil.which(binary_name)
        if sumo_path:
            return sumo_path
        
        # Method 2: Check common Windows installation paths
        if os.name == 'nt':
            exe_name = f"{binary_name}.exe"
            common_paths = [
                r"C:\Program Files (x86)\Eclipse\Sumo\bin",
                r"C:\Program Files\Eclipse\Sumo\bin",
            ]
            
            for bin_path in common_paths:
                sumo_exe = os.path.join(bin_path, exe_name)
                if os.path.exists(sumo_exe):
                    # Set SUMO_HOME environment variable
                    sumo_home = os.path.dirname(bin_path)
                    os.environ['SUMO_HOME'] = sumo_home
                    print(f"Found SUMO: {sumo_exe}")
                    print(f"Set SUMO_HOME: {sumo_home}")
                    return sumo_exe
        
        print("Error: Could not find SUMO executable file")
        return None
    
    def _initialize_network_elements(self):
        """Initialize network elements (traffic lights, lanes, detectors) based on config"""
        if not self.is_connected:
            return
        
        # Load from config or auto-detect
        config = self.sim_config
        
        # Traffic lights
        if config['traffic_lights'] is not None:
            # Use predefined traffic lights (custom dataset)
            self.traffic_lights = config['traffic_lights']
            self.controlled_traffic_lights = self.traffic_lights.copy()
            print(f"Using predefined traffic lights: {self.traffic_lights}")
        else:
            # Auto-detect traffic lights from network
            self.traffic_lights = list(traci.trafficlight.getIDList())
            print(f"Detected {len(self.traffic_lights)} traffic lights in network")
            
            # For TAPASCologne, limit to selected or max number
            if config['control_mode'] == 'selected' and config.get('selected_traffic_lights'):
                self.controlled_traffic_lights = [tl for tl in config['selected_traffic_lights'] 
                                                  if tl in self.traffic_lights]
            else:
                # Limit to max_controlled_lights
                max_lights = config.get('max_controlled_lights', 10)
                self.controlled_traffic_lights = self.traffic_lights[:max_lights]
            
            print(f"Controlling {len(self.controlled_traffic_lights)} traffic lights: {self.controlled_traffic_lights[:5]}...")
        
        # Update intersection_id for backward compatibility (use first traffic light)
        if self.controlled_traffic_lights:
            self.intersection_id = self.controlled_traffic_lights[0]
        
        # Lanes
        if config['lanes'] is not None:
            # Use predefined lanes
            self.lanes = config['lanes']
            print(f"Using predefined lanes: {len(self.lanes)} lanes")
        else:
            # Auto-detect lanes (get incoming lanes for controlled traffic lights)
            self.lanes = self._get_controlled_lanes()
            print(f"Auto-detected {len(self.lanes)} lanes for controlled intersections")
        
        # Detectors
        if config['detectors'] is not None:
            # Use predefined detectors
            self.detectors = config['detectors']
        else:
            # Auto-detect available detectors
            try:
                self.detectors = list(traci.inductionloop.getIDList())
                print(f"Auto-detected {len(self.detectors)} induction loop detectors")
            except:
                self.detectors = []
                print("No induction loop detectors found")
    
    def _get_controlled_lanes(self) -> List[str]:
        """Get all incoming lanes for controlled traffic lights"""
        lanes = set()
        for tl_id in self.controlled_traffic_lights:
            try:
                # Get all links controlled by this traffic light
                controlled_links = traci.trafficlight.getControlledLinks(tl_id)
                for link_list in controlled_links:
                    for link in link_list:
                        # link is tuple: (incoming_lane, outgoing_lane, via_lane)
                        if link and len(link) > 0:
                            incoming_lane = link[0]
                            lanes.add(incoming_lane)
            except Exception as e:
                print(f"Warning: Could not get lanes for traffic light {tl_id}: {e}")
        return sorted(list(lanes))    
    def start_simulation(self) -> bool:
        """Start SUMO simulation"""
        if traci is None:
            print("Error: TraCI library not available")
            return False
            
        try:
            # Find SUMO executable file
            sumo_binary = self._find_sumo_binary()
            if not sumo_binary:
                return False
            
            # Get the directory containing the config file
            # SUMO resolves relative paths from the working directory, not from config file location
            config_filename = os.path.basename(self.config_file)
            
            # Change to config file directory so relative paths in config file work correctly
            # Keep this directory during the entire simulation for output files
            os.chdir(self.config_dir)
            
            sumo_cmd = [
                sumo_binary,
                "-c", config_filename,  # Use just filename since we're in the config directory
                "--start",
                "--no-step-log",  # Disable step logging to prevent early termination
                "--no-warnings",  # Suppress warnings
                "--end", "999999"  # Set a very large end time to prevent early termination
                # Note: We control the actual duration via TraCI, not SUMO's end time
            ]
            
            # Start SUMO and connect TraCI
            traci.start(sumo_cmd, port=self.port)
            self.is_connected = True
            self.simulation_time = 0
            self.step_count = 0
            
            print(f"SUMO simulation started (port: {self.port})")
            print(f"Configuration file: {self.config_file}")
            print(f"Working directory: {self.config_dir}")
            print(f"Graphical interface: {'Yes' if self.use_gui else 'No'}")
            
            # Auto-detect or load network elements
            self._initialize_network_elements()
            
            return True
            
        except Exception as e:
            print(f"Failed to start SUMO simulation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close_simulation(self):
        """Close SUMO simulation"""
        if self.is_connected:
            try:
                traci.close()
                self.is_connected = False
                print("SUMO simulation closed")
                
                # Force-close SUMO GUI to avoid dialog prompts
                # This prevents the "Do you want to close all open files and views?" dialog
                if self.use_gui:
                    time.sleep(0.5)  # Give TraCI time to close connection
                    try:
                        if sys.platform == 'win32':
                            # Windows: Kill all sumo-gui.exe processes
                            subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                                         capture_output=True, timeout=3)
                        else:
                            # Linux/Mac: Kill sumo-gui processes
                            subprocess.run(['pkill', '-f', 'sumo-gui'], 
                                         capture_output=True, timeout=3)
                    except Exception as e:
                        # Ignore errors - GUI might already be closed
                        pass
            except Exception as e:
                print(f"Error closing simulation: {e}")
            finally:
                # Restore original working directory
                if hasattr(self, 'original_cwd'):
                    try:
                        os.chdir(self.original_cwd)
                        print(f"Restored working directory: {self.original_cwd}")
                    except Exception as e:
                        print(f"Warning: Could not restore working directory: {e}")
    
    def simulation_step(self) -> bool:
        """Execute one simulation step"""
        if not self.is_connected:
            return False
            
        try:
            # Try to execute simulation step
            traci.simulationStep()
            self.simulation_time = traci.simulation.getTime()
            self.step_count += 1
            
            # Collect data
            self._collect_step_data()
            
            return True
            
        except traci.exceptions.FatalTraCIError as e:
            # SUMO connection lost - simulation terminated
            print(f"SUMO simulation terminated: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            # Check if it's a connection error
            error_str = str(e).lower()
            if 'connection' in error_str or 'closed' in error_str or 'terminated' in error_str:
                print(f"SUMO connection lost: {e}")
                self.is_connected = False
                return False
            else:
                # Other error - log and return False
                print(f"Simulation step error: {e}")
                return False
    
    def run_simulation(self, duration: Optional[int] = None, strategy: str = "FIXED_TIME") -> Dict:
        """
        Run complete simulation
        
        Args:
            duration: simulation duration (seconds), None means run until end
            strategy: control strategy ('FIXED_TIME', 'ADAPTIVE', 'DQN', 'MAX_PRESSURE')
            
        Returns:
            Dict: simulation statistics results
        """
        # Store requested duration for later use
        self.requested_duration = duration
        
        if not self.start_simulation():
            return {}
        
        # Initialize signal controller
        controller = None
        if strategy is not None:
            try:
                from src.signal_controller import SignalController, ControlStrategy, TrafficState
                strategy_map = {
                    'FIXED_TIME': ControlStrategy.FIXED_TIME,
                    'ADAPTIVE': ControlStrategy.ADAPTIVE,
                    'DQN': ControlStrategy.DQN,
                    'MAX_PRESSURE': ControlStrategy.MAX_PRESSURE
                }
                control_strategy = strategy_map.get(strategy, ControlStrategy.FIXED_TIME)
                controller = SignalController(control_strategy)
                print(f"Signal control strategy: {strategy} ({control_strategy.value})")
            except ImportError as e:
                print(f"Warning: Unable to import signal controller, using default SUMO control: {e}")
                controller = None
        else:
            print("Using default SUMO traffic light control (no custom controller)")
        
        print("Starting simulation...")
        start_time = time.time()
        
        try:
            # Manually control simulation steps to avoid TraCI early termination
            # Calculate max_steps based on duration (assuming 1 second per step)
            max_steps = int(duration * 1.1) if duration else 10000  # Add 10% buffer
            step = 0
            
            while step < max_steps:
                # Check duration limit - this is the primary stopping condition
                if duration and self.simulation_time >= duration:
                    print(f"Reached requested duration: {duration}s (simulation_time: {self.simulation_time:.1f}s)")
                    break

                # Execute simulation step
                step_success = self.simulation_step()
                step += 1
                
                # If step failed, check if SUMO terminated
                if not step_success:
                    # Check if we've lost connection (SUMO terminated)
                    if not self.is_connected:
                        print(f"SUMO terminated early at {self.simulation_time:.1f}s (requested: {duration}s)")
                        # If we haven't reached the duration, this is a problem
                        if duration and self.simulation_time < duration:
                            print(f"WARNING: Simulation ended early! Expected {duration}s but got {self.simulation_time:.1f}s")
                        break
                    # Otherwise, continue - might be a temporary error
                    continue
                
                # If in GUI mode, add extra delay so user can see the process
                if self.use_gui:
                    time.sleep(0.04)  # 40ms delay (5x faster: 200ms -> 40ms)
                
                # Apply signal control strategy
                # Check every step, but only make decisions when phase is about to end or controller requests change
                if controller:
                    try:
                        # Get current traffic state
                        state_dict = self.get_current_state()

                        # Convert to TrafficState object
                        # Use the first controlled traffic light (or intersection_id for backward compatibility)
                        tl_id = self.controlled_traffic_lights[0] if self.controlled_traffic_lights else self.intersection_id
                        sumo_phase = traci.trafficlight.getPhase(tl_id)
                        phase_remaining_time = traci.trafficlight.getNextSwitch(tl_id) - self.simulation_time
                        
                        # Map SUMO phase to logical phase (0-3)
                        # SUMO has 6 phases: 0=EW green, 1=EW yellow, 2=NS short, 3=NS yellow, 4=NS green, 5=NS yellow
                        # Our logical phases: 0=EW straight, 1=EW left, 2=NS straight, 3=NS left
                        # Map SUMO phases to logical: 0->0, 4->2, others are transitions
                        logical_phase = self._sumo_phase_to_logical(sumo_phase)
                        
                        traffic_state = TrafficState(
                            time=self.simulation_time,
                            lane_vehicle_counts=state_dict.get('lane_vehicle_counts', {}),
                            lane_queue_lengths=state_dict.get('lane_queue_lengths', {}),
                            lane_mean_speeds=state_dict.get('lane_mean_speeds', {}),
                            detector_occupancy=state_dict.get('detector_occupancy', {}),
                            current_phase=logical_phase,
                            phase_remaining_time=phase_remaining_time
                        )

                        # Debug: print controller state every 50 steps
                        if self.step_count % 50 == 0:
                            print(f"Controller debug - Step {self.step_count}: SUMO_phase={sumo_phase}, logical_phase={logical_phase}, remaining={phase_remaining_time:.1f}s, queues={sum(traffic_state.lane_queue_lengths.values())}")

                        # Check if we should make a control decision:
                        # 1. Phase is about to end (within 1 second) AND we're on a green phase
                        # 2. Or check every 5 steps for adaptive controllers (but only on green phases)
                        is_green_phase = (sumo_phase in [0, 4])  # Only green phases in SUMO
                        should_check = (is_green_phase and phase_remaining_time <= 1.0) or (is_green_phase and self.step_count % 5 == 0)
                        
                        if should_check:
                            # Get control decision
                            # Note: Use phase_duration to avoid shadowing the function parameter 'duration'
                            next_logical_phase, phase_duration = controller.control_step(traffic_state)

                            # Map logical phase back to SUMO phase
                            next_sumo_phase = self._logical_phase_to_sumo(next_logical_phase, sumo_phase)

                            # Apply control decision to all controlled traffic lights
                            if next_sumo_phase != sumo_phase:
                                print(f"Controller changing phase: SUMO {sumo_phase} (logical {logical_phase}) -> SUMO {next_sumo_phase} (logical {next_logical_phase}) (duration: {phase_duration}s)")
                                for controlled_tl_id in self.controlled_traffic_lights:
                                    try:
                                        traci.trafficlight.setPhase(controlled_tl_id, next_sumo_phase)
                                        traci.trafficlight.setPhaseDuration(controlled_tl_id, phase_duration)
                                    except Exception as e:
                                        print(f"Failed to set phase for {controlled_tl_id}: {e}")
                            else:
                                # Same phase - this might be an extension request
                                # Only extend if phase_duration > 0 (otherwise it's just "keep current")
                                if phase_duration > 0:
                                    # Extend the current phase duration (add to remaining time)
                                    if phase_remaining_time > 0:
                                        new_duration = phase_remaining_time + phase_duration
                                    else:
                                        new_duration = phase_duration
                                    print(f"Controller extending phase: SUMO {sumo_phase} (logical {logical_phase}) (extend by {phase_duration}s, new duration: {new_duration}s)")
                                    for controlled_tl_id in self.controlled_traffic_lights:
                                        try:
                                            traci.trafficlight.setPhaseDuration(controlled_tl_id, new_duration)
                                        except Exception as e:
                                            print(f"Failed to extend phase for {controlled_tl_id}: {e}")
                                # If phase_duration is 0, do nothing (keep current phase as-is)

                    except Exception as e:
                        print(f"Signal control error at step {self.step_count}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                # Output progress every 100 steps
                if self.step_count % 100 == 0:
                    print(f"Simulation time: {self.simulation_time:.1f}s, "
                          f"Steps: {self.step_count}, Strategy: {strategy}")
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            
            real_time = time.time() - start_time
            print(f"\nSimulation completed!")
            print(f"Simulation time: {self.simulation_time:.1f}s")
            print(f"Real time elapsed: {real_time:.2f}s")
            print(f"Total steps: {self.step_count}")
            
            return metrics
            
        except KeyboardInterrupt:
            print("\nUser interrupted simulation")
            return {}
        finally:
            self.close_simulation()
    
    def _collect_step_data(self):
        """Collect simulation data for current step - complete version"""
        try:
            # Collect vehicle data (including emission data)
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                try:
                    veh_data = {
                        'time': self.simulation_time,
                        'vehicle_id': veh_id,
                        'position': traci.vehicle.getPosition(veh_id),
                        'speed': traci.vehicle.getSpeed(veh_id),
                        'lane_id': traci.vehicle.getLaneID(veh_id),
                        'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                        'acceleration': traci.vehicle.getAcceleration(veh_id),
                        'distance': traci.vehicle.getDistance(veh_id),
                        'route_id': traci.vehicle.getRouteID(veh_id),
                        'type_id': traci.vehicle.getTypeID(veh_id)
                    }
                    
                    # Try to collect emission data
                    try:
                        veh_data.update({
                            'co2_emission': traci.vehicle.getCO2Emission(veh_id),
                            'fuel_consumption': traci.vehicle.getFuelConsumption(veh_id),
                            'noise_emission': traci.vehicle.getNoiseEmission(veh_id),
                            'hc_emission': traci.vehicle.getHCEmission(veh_id),
                            'pmx_emission': traci.vehicle.getPMxEmission(veh_id),
                            'nox_emission': traci.vehicle.getNOxEmission(veh_id)
                        })
                    except Exception:
                        # If emission data unavailable, use default values
                        veh_data.update({
                            'co2_emission': 0.0,
                            'fuel_consumption': 0.0,
                            'noise_emission': 0.0,
                            'hc_emission': 0.0,
                            'pmx_emission': 0.0,
                            'nox_emission': 0.0
                        })
                    
                    self.vehicle_data.append(veh_data)
                except Exception as e:
                    print(f"Failed to collect data for vehicle {veh_id}: {e}")
            
            # Collect traffic light data
            for tl_id in self.controlled_traffic_lights:
                try:
                    tl_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                    tl_phase = traci.trafficlight.getPhase(tl_id)
                    tl_data = {
                        'time': self.simulation_time,
                        'traffic_light_id': tl_id,
                        'state': tl_state,
                        'phase': tl_phase,
                        'phase_duration': traci.trafficlight.getPhaseDuration(tl_id),
                        'next_switch': traci.trafficlight.getNextSwitch(tl_id)
                    }
                    self.traffic_light_data.append(tl_data)
                except Exception as e:
                    print(f"Failed to collect traffic light data for {tl_id}: {e}")
            
            # Collect detector data
            for det_id in self.detectors:
                try:
                    if det_id in traci.inductionloop.getIDList():
                        det_data = {
                            'time': self.simulation_time,
                            'detector_id': det_id,
                            'vehicle_count': traci.inductionloop.getLastStepVehicleNumber(det_id),
                            'occupancy': traci.inductionloop.getLastStepOccupancy(det_id),
                            'mean_speed': traci.inductionloop.getLastStepMeanSpeed(det_id),
                            'vehicle_ids': list(traci.inductionloop.getLastStepVehicleIDs(det_id))
                        }
                        self.detector_data.append(det_data)
                except Exception as e:
                    print(f"Failed to collect detector {det_id} data: {e}")
                    
        except Exception as e:
            print(f"Overall data collection failed: {e}")
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate complete performance metrics"""
        metrics = {}
        
        if not self.vehicle_data:
            return metrics
            
        # Convert to DataFrame for easier analysis
        df_vehicles = pd.DataFrame(self.vehicle_data)
        df_detectors = pd.DataFrame(self.detector_data) if self.detector_data else pd.DataFrame()
        
        # 1. Average waiting time
        if 'waiting_time' in df_vehicles.columns:
            metrics['avg_waiting_time'] = df_vehicles['waiting_time'].mean()
            metrics['max_waiting_time'] = df_vehicles['waiting_time'].max()
            metrics['total_waiting_time'] = df_vehicles['waiting_time'].sum()
            
            # Additional metrics: per-vehicle analysis for better understanding
            vehicle_max_wait = df_vehicles.groupby('vehicle_id')['waiting_time'].max()
            metrics['avg_max_waiting_time_per_vehicle'] = vehicle_max_wait.mean()
            metrics['vehicles_with_waiting'] = (vehicle_max_wait > 0).sum()
            metrics['vehicles_total'] = len(vehicle_max_wait)
            metrics['percentage_vehicles_waited'] = (metrics['vehicles_with_waiting'] / metrics['vehicles_total'] * 100) if metrics['vehicles_total'] > 0 else 0
        
        # 2. Average speed
        if 'speed' in df_vehicles.columns:
            metrics['avg_speed'] = df_vehicles['speed'].mean()
            metrics['max_speed'] = df_vehicles['speed'].max()
        
        # 3. System throughput
        unique_vehicles = df_vehicles['vehicle_id'].nunique()
        metrics['throughput'] = unique_vehicles
        
        # Calculate throughput per hour using ACTUAL simulation time
        # This gives realistic throughput values based on what actually happened
        # If simulation ended early, this will reflect that
        if self.simulation_time > 0:
            metrics['throughput_per_hour'] = unique_vehicles * 3600 / self.simulation_time
        else:
            metrics['throughput_per_hour'] = 0
        
        # 4. Congestion index (based on queue length)
        queue_lengths = []
        for lane_id in self.lanes:
            lane_vehicles = df_vehicles[df_vehicles['lane_id'] == lane_id]
            if not lane_vehicles.empty:
                # Calculate average queue length for this lane (vehicles with waiting time > 5 seconds)
                waiting_vehicles = lane_vehicles[lane_vehicles['waiting_time'] > 5.0]
                avg_queue = len(waiting_vehicles) / len(lane_vehicles) if len(lane_vehicles) > 0 else 0
                queue_lengths.append(avg_queue)
        
        if queue_lengths:
            metrics['congestion_index'] = np.mean(queue_lengths)
            metrics['max_queue_length'] = max(queue_lengths)
        
        # 5. Emission metrics
        if 'co2_emission' in df_vehicles.columns:
            metrics['total_co2'] = df_vehicles['co2_emission'].sum()
            metrics['avg_co2_per_vehicle'] = df_vehicles.groupby('vehicle_id')['co2_emission'].sum().mean()
            metrics['co2_per_km'] = metrics['total_co2'] / (df_vehicles['distance'].sum() / 1000) if df_vehicles['distance'].sum() > 0 else 0
        
        if 'fuel_consumption' in df_vehicles.columns:
            metrics['total_fuel'] = df_vehicles['fuel_consumption'].sum()
            metrics['avg_fuel_per_vehicle'] = df_vehicles.groupby('vehicle_id')['fuel_consumption'].sum().mean()
        
        if 'nox_emission' in df_vehicles.columns:
            metrics['total_nox'] = df_vehicles['nox_emission'].sum()
        
        if 'pmx_emission' in df_vehicles.columns:
            metrics['total_pmx'] = df_vehicles['pmx_emission'].sum()
        
        # 6. Lane utilization
        if not df_detectors.empty and 'occupancy' in df_detectors.columns:
            lane_occupancy = df_detectors.groupby('detector_id')['occupancy'].mean()
            metrics['avg_lane_occupancy'] = lane_occupancy.mean()
            metrics['lane_occupancy_std'] = lane_occupancy.std()
            metrics['max_lane_occupancy'] = lane_occupancy.max()
        
        # 7. Acceleration statistics (comfort index)
        if 'acceleration' in df_vehicles.columns:
            metrics['avg_acceleration'] = df_vehicles['acceleration'].mean()
            metrics['acceleration_std'] = df_vehicles['acceleration'].std()
            harsh_braking = (df_vehicles['acceleration'] < -3.0).sum()
            metrics['harsh_braking_events'] = harsh_braking
        
        # 8. Statistics by vehicle type
        if 'type_id' in df_vehicles.columns:
            type_stats = df_vehicles.groupby('type_id').agg({
                'waiting_time': ['mean', 'sum'],
                'speed': 'mean',
                'co2_emission': 'sum' if 'co2_emission' in df_vehicles.columns else lambda x: 0
            }).round(2)
            metrics['vehicle_type_stats'] = type_stats.to_dict()
        
        return metrics
    
    def _sumo_phase_to_logical(self, sumo_phase: int) -> int:
        """Map SUMO phase number to logical phase (0-3)
        
        SUMO has 6 phases: 0=EW green, 1=EW yellow, 2=NS short, 3=NS yellow, 4=NS green, 5=NS yellow
        Our logical phases: 0=EW straight, 1=EW left, 2=NS straight, 3=NS left
        """
        if sumo_phase == 0:
            return 0  # East-West straight
        elif sumo_phase == 4:
            return 2  # North-South straight
        else:
            # For yellow/transition phases, return the previous green phase
            if sumo_phase in [1]:
                return 0  # EW yellow -> EW
            elif sumo_phase in [2, 3, 5]:
                return 2  # NS yellow -> NS
            else:
                return 0  # Default
    
    def _logical_phase_to_sumo(self, logical_phase: int, current_sumo_phase: int) -> int:
        """Map logical phase (0-3) to SUMO phase number
        
        Logical phases: 0=EW straight, 1=EW left, 2=NS straight, 3=NS left
        SUMO phases: 0=EW green, 4=NS green
        """
        if logical_phase in [0, 1]:
            return 0  # East-West (straight or left)
        elif logical_phase in [2, 3]:
            return 4  # North-South (straight or left)
        else:
            return current_sumo_phase  # Keep current if unknown
    
    def get_current_state(self) -> Dict:
        """
        Get current simulation state (for reinforcement learning)
        
        Returns:
            Dict: current state information
        """
        if not self.is_connected:
            return {}
            
        state = {
            'time': self.simulation_time,
            'lane_vehicle_counts': {},
            'lane_queue_lengths': {},
            'traffic_light_phase': 0,
            'traffic_light_remaining_time': 0,
            'detector_occupancy': {},
            'lane_mean_speeds': {}
        }
        
        try:
            # Vehicle counts and queue lengths per lane
            for lane_id in self.lanes:
                try:
                    state['lane_vehicle_counts'][lane_id] = traci.lane.getLastStepVehicleNumber(lane_id)
                    state['lane_mean_speeds'][lane_id] = traci.lane.getLastStepMeanSpeed(lane_id)
                    
                    # Queue length (number of waiting vehicles)
                    waiting_vehicles = [veh for veh in traci.lane.getLastStepVehicleIDs(lane_id) 
                                      if traci.vehicle.getWaitingTime(veh) > 5.0]
                    state['lane_queue_lengths'][lane_id] = len(waiting_vehicles)
                except:
                    state['lane_vehicle_counts'][lane_id] = 0
                    state['lane_queue_lengths'][lane_id] = 0
                    state['lane_mean_speeds'][lane_id] = 0
            
            # Detector occupancy
            for det_id in self.detectors:
                try:
                    if det_id in traci.inductionloop.getIDList():
                        state['detector_occupancy'][det_id] = traci.inductionloop.getLastStepOccupancy(det_id)
                except:
                    state['detector_occupancy'][det_id] = 0
            
            # Traffic light state
            if traci.trafficlight.getIDList():
                try:
                    sumo_phase = traci.trafficlight.getPhase(self.intersection_id)
                    state['traffic_light_phase'] = self._sumo_phase_to_logical(sumo_phase)
                    state['traffic_light_remaining_time'] = traci.trafficlight.getNextSwitch(self.intersection_id) - self.simulation_time
                except:
                    pass
            
        except Exception as e:
            print(f"Failed to get state: {e}")
            
        return state
    
    def _get_lanes_for_traffic_light(self, tl_id: str) -> List[str]:
        """Get lanes controlled by a specific traffic light"""
        lanes = set()
        try:
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)
            for link_list in controlled_links:
                for link in link_list:
                    if link and len(link) > 0:
                        lanes.add(link[0])
        except Exception as e:
            print(f"Warning: Could not get lanes for {tl_id}: {e}")
        return sorted(list(lanes))
    
    def set_traffic_light_phase(self, phase: int, tl_id: str = None):
        """
        Set traffic light phase (for reinforcement learning control)
        
        Args:
            phase: phase number
            tl_id: traffic light ID (None for intersection_id)
        """
        if tl_id is None:
            tl_id = self.intersection_id
            
        if self.is_connected and traci.trafficlight.getIDList():
            try:
                traci.trafficlight.setPhase(tl_id, phase)
            except Exception as e:
                print(f"Failed to set traffic light phase for {tl_id}: {e}")
    
    def extend_green_phase(self, duration: float, tl_id: str = None):
        """
        Extend current green phase
        
        Args:
            duration: extension time (seconds)
            tl_id: traffic light ID (None for intersection_id)
        """
        if tl_id is None:
            tl_id = self.intersection_id
            
        if self.is_connected and traci.trafficlight.getIDList():
            try:
                current_duration = traci.trafficlight.getPhaseDuration(tl_id)
                new_duration = current_duration + duration
                traci.trafficlight.setPhaseDuration(tl_id, new_duration)
            except Exception as e:
                print(f"Failed to extend green phase for {tl_id}: {e}")
    
    def export_data(self, output_dir: str = "output"):
        """
        Export simulation data
        
        Args:
            output_dir: output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export vehicle data
        if self.vehicle_data:
            df_vehicles = pd.DataFrame(self.vehicle_data)
            df_vehicles.to_csv(f"{output_dir}/vehicle_data.csv", index=False)
            print(f"Vehicle data exported: {len(self.vehicle_data)} records")
        
        # Export traffic light data  
        if self.traffic_light_data:
            df_tl = pd.DataFrame(self.traffic_light_data)
            df_tl.to_csv(f"{output_dir}/traffic_light_data.csv", index=False)
            print(f"Traffic light data exported: {len(self.traffic_light_data)} records")
        
        # Export detector data
        if self.detector_data:
            df_det = pd.DataFrame(self.detector_data)
            df_det.to_csv(f"{output_dir}/detector_data.csv", index=False)
            print(f"Detector data exported: {len(self.detector_data)} records")
        
        # Export performance metrics
        if self.performance_metrics:
            import json
            with open(f"{output_dir}/performance_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)
            print("Performance metrics exported")
        
        # Create summary report
        self._generate_summary_report(output_dir)
        
        print(f"All data exported to: {output_dir}/")
    
    def _generate_summary_report(self, output_dir: str):
        """Generate summary report"""
        report = []
        report.append("# SUMO Traffic Simulation Summary Report\n")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Simulation duration: {self.simulation_time:.1f} seconds\n")
        report.append(f"Simulation steps: {self.step_count}\n\n")
        
        if self.performance_metrics:
            report.append("## Main Performance Metrics\n")
            for key, value in self.performance_metrics.items():
                if isinstance(value, (int, float)):
                    report.append(f"- {key}: {value:.2f}\n")
        
        report.append(f"\n## Data Collection Statistics\n")
        report.append(f"- Vehicle data records: {len(self.vehicle_data)} entries\n")
        report.append(f"- Traffic light data records: {len(self.traffic_light_data)} entries\n")
        report.append(f"- Detector data records: {len(self.detector_data)} entries\n")
        
        with open(f"{output_dir}/summary_report.md", 'w', encoding='utf-8') as f:
            f.writelines(report)


def main():
    """Main function - for testing complete functionality"""
    print("Traffic Simulator Complete Test")
    
    # Ensure running in project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Parent directory of src
    original_cwd = os.getcwd()
    
    try:
        os.chdir(project_root)
        print(f"Switched working directory to: {project_root}")
        
        # Create simulator instance
        simulator = TrafficSimulator(use_gui=False)
        
        # Run simulation
        metrics = simulator.run_simulation(duration=300)  # Run for 5 minutes
        
        # Output results
        print("\n=== Complete Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
        
        # Export data
        simulator.export_data()
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main() 