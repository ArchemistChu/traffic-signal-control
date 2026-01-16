#!/usr/bin/env python3
"""
Traffic Simulation Engine Module (TrafficSimulator)
Traffic simulation using SUMO and TraCI API
Complete implementation of all features from original requirements
Supports custom single intersection and OSM-based maps (Cologne, Vancouver, Los Angeles)
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime

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
                 use_gui: bool = False, port: int = 8813, dataset: str = None,
                 gui_delay: float = 0.0,
                 enable_sumo_emissions_output: bool = True):
        """
        Initialize traffic simulator
        
        Args:
            config_file: SUMO configuration file path (None for auto from SimulationConfig)
            use_gui: whether to use graphical interface
            port: TraCI connection port
            dataset: 'custom', 'cologne', 'vancouver', or 'los_angeles' (None for current config)
            gui_delay: delay per step in GUI mode in seconds (0.0 = no delay, fastest)
            enable_sumo_emissions_output: for OSM maps, enable SUMO --emission-output (Option A).
                Set False for RL training loops to avoid generating huge XML each episode.
        """
        # Set dataset if specified
        if dataset:
            SimulationConfig.set_dataset(dataset)
        
        # Get configuration
        self.sim_config = SimulationConfig.get_current_config()
        
        # Use config file from SimulationConfig if not provided
        self.config_file = config_file or SimulationConfig.get_sumo_config_path()
        self.use_gui = use_gui
        self.gui_delay = gui_delay  # Delay per step in GUI mode (0 = no delay, fastest)
        self.enable_sumo_emissions_output = enable_sumo_emissions_output
        self.port = port
        self.is_connected = False
        self.simulation_time = 0
        self.step_count = 0
        self.requested_duration = None  # Store requested duration for fair comparison

        # System-level counters (used for correct throughput)
        self.total_departed = 0
        self.total_arrived = 0

        # GA tuning parameters (used only for GA baseline runs)
        self._ga_params: Optional[Dict[str, float]] = None
        
        # Simulation data storage
        self.vehicle_data = []
        self.traffic_light_data = []
        self.detector_data = []
        self.emission_data = []
        self.performance_metrics = {}

        # Performance knobs (especially important for large OSM maps)
        # - data_collection_interval: collect/export detailed per-step data every N steps
        # - collect_emissions: emission TraCI calls are expensive; disable by default for OSM maps
        if SimulationConfig.is_custom_dataset():
            self.data_collection_interval = 1
            self.collect_emissions = True
        else:
            self.data_collection_interval = 5
            self.collect_emissions = False

        # SUMO native emissions output file (Option A: compute in SUMO, parse after run)
        self.emissions_output_file: Optional[str] = None
        
        # Traffic lights, lanes, and detectors (will be detected or loaded from config)
        self.traffic_lights = []
        self.controlled_traffic_lights = []  # Subset to control
        self.lanes = []
        self.detectors = []
        
        # Legacy intersection ID (for backward compatibility)
        self.intersection_id = "intersection"

        # Phase mapping (auto-detected for custom single-intersection)
        self._ew_green_phase: Optional[int] = None
        self._ns_green_phase: Optional[int] = None
        self._major_green_phases: List[int] = []  # typically [ew, ns]
        # Track how long a SUMO phase has been active (per traffic light)
        self._tl_last_phase: Dict[str, Optional[int]] = {}
        self._tl_phase_start_time: Dict[str, float] = {}
        # Track last time pedestrians received green (cheap fairness constraint)
        self._tl_last_ped_green_time: Dict[str, float] = {}
        # Cache: for each TL, which lanes get green in each phase
        # {tl_id: {"green_phases": [int], "phase_to_lanes": {phase_idx: [lane_id,...]},
        #          "ped_green_phases": [int], "phase_has_ped_green": {phase_idx: bool}}}
        self._tl_phase_lane_cache: Dict[str, Dict] = {}
        
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

        # Reset per-TL caches/timers (important when switching datasets or rerunning)
        for tl in self.controlled_traffic_lights:
            self._tl_last_phase.setdefault(tl, None)
            self._tl_phase_start_time.setdefault(tl, float(self.simulation_time))
            self._tl_last_ped_green_time.setdefault(tl, float(self.simulation_time))
        # Phase/lane cache depends on the TL program; clear on init to be safe
        self._tl_phase_lane_cache = {}
        
        # Update intersection_id for backward compatibility (use first traffic light)
        if self.controlled_traffic_lights:
            self.intersection_id = self.controlled_traffic_lights[0]

        # For the custom single-intersection dataset, auto-detect which SUMO phase
        # corresponds to EW-green vs NS-green. The earlier hard-coded mapping (0 and 4)
        # does not always match the generated tlLogic, causing controllers to never act.
        if SimulationConfig.is_custom_dataset() and self.controlled_traffic_lights:
            try:
                self._detect_major_green_phases(self.controlled_traffic_lights[0])
                if self._major_green_phases:
                    print(
                        f"Detected major green phases for {self.controlled_traffic_lights[0]}: "
                        f"EW={self._ew_green_phase}, NS={self._ns_green_phase}"
                    )
                # Initialize phase tracking
                tl0 = self.controlled_traffic_lights[0]
                try:
                    ph = traci.trafficlight.getPhase(tl0)
                    self._tl_last_phase[tl0] = ph
                    self._tl_phase_start_time[tl0] = float(self.simulation_time)
                except Exception:
                    pass
            except Exception as e:
                print(f"Warning: could not auto-detect major green phases: {e}")
        
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

    def _get_tl_phase_count(self, tl_id: str) -> int:
        """
        SUMO-version-safe number of phases for a traffic light.

        Some Python traci builds do not expose trafficlight.getPhaseNumber(), so we
        fall back to reading program logics.
        """
        if traci is None:
            return 1
        if hasattr(traci.trafficlight, "getPhaseNumber"):
            try:
                return int(traci.trafficlight.getPhaseNumber(tl_id))
            except Exception:
                pass
        try:
            logics = traci.trafficlight.getAllProgramLogics(tl_id)
            if logics:
                return int(len(logics[0].phases))
        except Exception:
            pass
        return 1

    def _get_tl_phase_lane_map(self, tl_id: str) -> Dict:
        """
        Build/cache mapping from phase index -> incoming lanes that have green.

        This uses SUMO's signal state strings (phase.state) and controlled links,
        so it works for OSM maps with arbitrary phase counts.
        """
        if tl_id in self._tl_phase_lane_cache:
            return self._tl_phase_lane_cache[tl_id]

        result = {
            "green_phases": [],
            "phase_to_lanes": {},
            "ped_green_phases": [],
            "phase_has_ped_green": {}
        }
        if traci is None:
            self._tl_phase_lane_cache[tl_id] = result
            return result

        try:
            def _is_ped_lane_id(lane_id: Optional[str]) -> bool:
                if not lane_id:
                    return False
                s = str(lane_id).lower()
                # Common SUMO naming patterns for pedestrian infrastructure
                if "walkingarea" in s or "crossing" in s or "ped" in s or "foot" in s or "sidewalk" in s:
                    return True
                # Internal lanes often start with ":"; pedestrian internal lanes frequently include "_w"
                if s.startswith(":") and ("_w" in s or "walk" in s or "cross" in s):
                    return True
                return False

            program_id = None
            if hasattr(traci.trafficlight, "getProgram"):
                try:
                    program_id = traci.trafficlight.getProgram(tl_id)
                except Exception:
                    program_id = None

            logics = traci.trafficlight.getAllProgramLogics(tl_id)
            logic = None
            if logics:
                if program_id is not None:
                    for lg in logics:
                        if getattr(lg, "programID", None) == program_id:
                            logic = lg
                            break
                logic = logic or logics[0]

            controlled_links = traci.trafficlight.getControlledLinks(tl_id)  # len == number of signal indices
            if not logic or not getattr(logic, "phases", None) or not controlled_links:
                self._tl_phase_lane_cache[tl_id] = result
                return result

            phases = list(logic.phases)
            for p_idx, ph in enumerate(phases):
                state = getattr(ph, "state", "")
                if not state:
                    continue

                lanes = set()
                any_vehicle_green = False
                any_ped_green = False
                # Each character corresponds to one controlled link index
                for i, ch in enumerate(state):
                    if ch not in ("G", "g"):
                        continue
                    if i >= len(controlled_links):
                        continue
                    for link in controlled_links[i]:
                        # link: (incoming, outgoing, via)
                        if link and len(link) > 0 and link[0]:
                            in_lane = link[0]
                            if _is_ped_lane_id(in_lane):
                                any_ped_green = True
                            else:
                                any_vehicle_green = True
                                lanes.add(in_lane)

                if any_vehicle_green:
                    result["green_phases"].append(int(p_idx))
                if any_ped_green:
                    result["ped_green_phases"].append(int(p_idx))
                result["phase_to_lanes"][int(p_idx)] = sorted(lanes)
                result["phase_has_ped_green"][int(p_idx)] = bool(any_ped_green)

        except Exception:
            # If mapping fails, keep empty; controller will fall back to cycling.
            pass

        self._tl_phase_lane_cache[tl_id] = result
        return result

    def _get_controlled_lanes_for_tl(self, tl_id: str) -> List[str]:
        """Get lanes controlled by a specific traffic light (safe across SUMO versions)."""
        lanes = set()
        if traci is None:
            return []
        if hasattr(traci.trafficlight, "getControlledLanes"):
            try:
                for l in traci.trafficlight.getControlledLanes(tl_id):
                    if l:
                        lanes.add(l)
            except Exception:
                pass
        if lanes:
            return sorted(lanes)

        # Fallback: parse controlled links
        try:
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)
            for link_list in controlled_links:
                for link in link_list:
                    if link and len(link) > 0 and link[0]:
                        lanes.add(link[0])
        except Exception:
            pass
        return sorted(lanes)

    @staticmethod
    def _pad_or_truncate(items: List[str], k: int) -> List[str]:
        items = [x for x in items if x]
        items = items[:k]
        if len(items) < k:
            items = items + [""] * (k - len(items))
        return items
    
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
                "--end", "999999",  # Set a very large end time to prevent early termination
            ]

            # Option A: For OSM maps, enable SUMO-native emissions output (faster than per-vehicle TraCI calls)
            # Keep file in output/ so it doesn't clutter map folders.
            if self.enable_sumo_emissions_output and not SimulationConfig.is_custom_dataset():
                try:
                    SimulationConfig.create_output_dirs()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.emissions_output_file = os.path.join(SimulationConfig.OUTPUT_DIR, f"emissions_{SimulationConfig.DATASET}_{ts}.xml")
                    sumo_cmd.extend([
                        "--device.emissions.probability", "1.0",
                        "--device.emissions.explicit", "CO,CO2,HC,NOx,PMx,fuel",
                        "--emission-output", self.emissions_output_file,
                        "--emission-output.precision", "3"
                    ])
                except Exception:
                    # If anything goes wrong, just skip native emissions output
                    self.emissions_output_file = None
            
            # For GUI mode, add quit-on-end to auto-close without dialog
            if self.use_gui:
                sumo_cmd.append("--quit-on-end")  # Auto-close GUI when simulation ends
            
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

            # Track departures/arrivals (cheap + correct throughput)
            try:
                self.total_departed += int(traci.simulation.getDepartedNumber())
            except Exception:
                pass
            try:
                self.total_arrived += int(traci.simulation.getArrivedNumber())
            except Exception:
                pass
            
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

        # GA baseline: run a small genetic search to tune timing parameters, then run final simulation
        if strategy == "GA":
            return self._run_ga_optimized(duration=duration)
        
        if not self.start_simulation():
            return {}

        # --- Multi-Agent DQN (trained) ---
        marl_enabled = (strategy == "MARL_DQN")
        marl_agent = None
        marl_tls_ids: List[str] = []
        marl_tl_lanes: Dict[str, List[str]] = {}
        marl_tl_phase_counts: Dict[str, int] = {}
        marl_last_decision_t: float = 0.0
        marl_decision_interval: float = 5.0  # seconds; keep consistent with training default
        # Cheap pedestrian support: force a pedestrian-green phase at least every N seconds
        ped_max_red: float = 90.0  # seconds (OSM maps)
        ped_min_green: float = 12.0  # seconds

        # --- PressLight-inspired (pressure DQN) ---
        presslight_enabled = (strategy == "PRESSLIGHT")
        presslight_agent = None
        presslight_tls_ids: List[str] = []
        presslight_last_decision_t: float = 0.0
        presslight_decision_interval: float = 10.0  # align with paper-style timestep
        presslight_k: int = 8  # number of phase-pressure features (pad/truncate)

        if marl_enabled:
            try:
                import torch  # local import to avoid making it a hard dependency for non-RL runs
                from src.rl_agent import RLAgent, TrafficState as RLTState
            except Exception as e:
                print(f"Warning: MARL_DQN requested but RL imports failed: {e}")
                marl_enabled = False

        if presslight_enabled:
            try:
                import torch
                from src.rl_agent import RLAgent, TrafficState as RLTState
            except Exception as e:
                print(f"Warning: PRESSLIGHT requested but RL imports failed: {e}")
                presslight_enabled = False

        if marl_enabled:
            # Load trained checkpoint
            model_path = os.path.join(SimulationConfig.MODEL_DIR, "marl_los_angeles_shared_dqn.pt")
            if not os.path.exists(model_path):
                print(f"Warning: MARL model not found at {model_path}. Falling back to FIXED_TIME.")
                marl_enabled = False
                strategy = "FIXED_TIME"
            else:
                # Load checkpoint config first so state_dim matches training
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    ckpt_cfg = checkpoint.get("config", {})
                    state_dim = int(ckpt_cfg.get("state_dim", 33))
                    action_dim = int(ckpt_cfg.get("action_dim", 2))
                    marl_agent = RLAgent(config={"state_dim": state_dim, "action_dim": action_dim})
                    marl_agent.load_model(model_path)
                    marl_agent.epsilon = 0.0  # inference
                except Exception as e:
                    print(f"Warning: failed to load MARL model ({model_path}): {e}. Falling back to FIXED_TIME.")
                    marl_enabled = False
                    strategy = "FIXED_TIME"

        if marl_enabled and marl_agent is not None:
            # Use all controlled TLS (respects max_controlled_lights from dataset config)
            marl_tls_ids = list(self.controlled_traffic_lights)

            # Infer lanes_per_tl from state_dim if possible (TrafficState vector is: 3*lanes + detectors + 8 + 1)
            lanes_per_tl = 8
            try:
                state_dim = int(marl_agent.config.get("state_dim", 33))
                lanes_per_tl = max(1, int((state_dim - 9) // 3))
            except Exception:
                lanes_per_tl = 8

            for tl_id in marl_tls_ids:
                lanes = self._get_controlled_lanes_for_tl(tl_id)
                marl_tl_lanes[tl_id] = self._pad_or_truncate(lanes, lanes_per_tl)
                marl_tl_phase_counts[tl_id] = max(1, self._get_tl_phase_count(tl_id))
                self._tl_last_ped_green_time.setdefault(tl_id, float(self.simulation_time))

            print(f"MARL_DQN enabled: loaded {os.path.basename(model_path)}")
            print(f"MARL agents (traffic lights): {len(marl_tls_ids)} | lanes_per_tl={lanes_per_tl} | interval={marl_decision_interval}s")

        if presslight_enabled:
            model_path = os.path.join(SimulationConfig.MODEL_DIR, "presslight_shared_dqn.pt")
            if not os.path.exists(model_path):
                print(f"Warning: PressLight model not found at {model_path}. Falling back to MAX_PRESSURE.")
                presslight_enabled = False
                strategy = "MAX_PRESSURE"
            else:
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    ckpt_cfg = checkpoint.get("config", {})
                    state_dim = int(ckpt_cfg.get("state_dim", 33))
                    action_dim = int(ckpt_cfg.get("action_dim", presslight_k))
                    presslight_agent = RLAgent(config={"state_dim": state_dim, "action_dim": action_dim})
                    presslight_agent.load_model(model_path)
                    presslight_agent.epsilon = 0.0
                    # infer K from state_dim: TrafficState vector = 3*K + 8 + 1 (no detectors)
                    try:
                        presslight_k = max(1, int((state_dim - 9) // 3))
                    except Exception:
                        presslight_k = 8
                    presslight_tls_ids = list(self.controlled_traffic_lights)
                    print(f"PRESSLIGHT enabled: loaded {os.path.basename(model_path)} | K={presslight_k} | tls={len(presslight_tls_ids)}")
                except Exception as e:
                    print(f"Warning: failed to load PressLight model ({model_path}): {e}. Falling back to MAX_PRESSURE.")
                    presslight_enabled = False
                    strategy = "MAX_PRESSURE"
        
        # Initialize signal controller
        controller = None
        if strategy is not None and not marl_enabled and not presslight_enabled:
            try:
                from src.signal_controller import SignalController, ControlStrategy, TrafficState
                strategy_map = {
                    'FIXED_TIME': ControlStrategy.FIXED_TIME,
                    'ADAPTIVE': ControlStrategy.ADAPTIVE,
                    'DQN': ControlStrategy.DQN,
                    'MAX_PRESSURE': ControlStrategy.MAX_PRESSURE,
                    'SOTL': ControlStrategy.SOTL
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
                
                # Apply GUI delay if specified (0 = no delay for maximum speed)
                if self.use_gui and self.gui_delay > 0:
                    time.sleep(self.gui_delay)
                
                # Apply signal control strategy
                # Check every step, but only make decisions when phase is about to end or controller requests change
                if presslight_enabled and presslight_agent is not None:
                    # Parameter-sharing DQN on pressure-state (PressLight-inspired)
                    if (self.simulation_time - presslight_last_decision_t) >= presslight_decision_interval:
                        presslight_last_decision_t = float(self.simulation_time)
                        try:
                            # Use current queues to compute pressure per phase (cached phase->lanes)
                            state_dict = self.get_current_state()

                            def _lane_weighted_queue(lane_id: str) -> float:
                                q = float(state_dict.get("lane_queue_lengths", {}).get(lane_id, 0))
                                bus_c = float(state_dict.get("lane_bus_counts", {}).get(lane_id, 0))
                                bus_w = float(state_dict.get("bus_waiting_time", {}).get(lane_id, 0.0))
                                return q + (2.5 * bus_c) + (0.3 * bus_w)

                            for tl_id in presslight_tls_ids:
                                try:
                                    cur_phase = int(traci.trafficlight.getPhase(tl_id))
                                    remaining = float(traci.trafficlight.getNextSwitch(tl_id) - self.simulation_time)
                                except Exception:
                                    continue

                                phase_map = self._get_tl_phase_lane_map(tl_id)
                                green_phases = phase_map.get("green_phases", []) or []
                                nph = self._get_tl_phase_count(tl_id)
                                if not green_phases:
                                    green_phases = list(range(max(1, nph)))

                                # Pedestrian fairness override
                                try:
                                    if phase_map.get("phase_has_ped_green", {}).get(cur_phase, False):
                                        self._tl_last_ped_green_time[tl_id] = float(self.simulation_time)
                                    ped_phases = phase_map.get("ped_green_phases", []) or []
                                    last_ped = float(self._tl_last_ped_green_time.get(tl_id, 0.0))
                                    if ped_phases and (float(self.simulation_time) - last_ped) >= ped_max_red:
                                        traci.trafficlight.setPhase(tl_id, int(ped_phases[0]))
                                        traci.trafficlight.setPhaseDuration(tl_id, ped_min_green)
                                        self._tl_last_ped_green_time[tl_id] = float(self.simulation_time)
                                        continue
                                except Exception:
                                    pass

                                # Compute pressure per green phase and pad/truncate to K
                                pressures = []
                                for p in green_phases[:presslight_k]:
                                    lanes = phase_map.get("phase_to_lanes", {}).get(int(p), []) or []
                                    pressures.append(float(sum(_lane_weighted_queue(l) for l in lanes)))
                                while len(pressures) < presslight_k:
                                    pressures.append(0.0)

                                fake_lanes = [f"p{i}" for i in range(presslight_k)]
                                sumo_state = {
                                    "time": float(self.simulation_time),
                                    "lane_vehicle_counts": {fake_lanes[i]: pressures[i] for i in range(presslight_k)},
                                    "lane_queue_lengths": {fake_lanes[i]: pressures[i] for i in range(presslight_k)},
                                    "lane_mean_speeds": {fake_lanes[i]: 0.0 for i in range(presslight_k)},
                                    "detector_occupancy": {},
                                    "traffic_light_phase": int(cur_phase) % 8,
                                    "traffic_light_remaining_time": max(0.0, remaining),
                                }
                                state_obj = RLTState(sumo_state, lane_list=fake_lanes)
                                action = int(presslight_agent.select_action(state_obj, training=False))

                                # Map action to a target green phase
                                if green_phases:
                                    target = int(green_phases[action % len(green_phases)])
                                    if target != cur_phase and remaining <= 1.0:
                                        traci.trafficlight.setPhase(tl_id, target)
                                        traci.trafficlight.setPhaseDuration(tl_id, 30.0)
                        except Exception as e:
                            print(f"PRESSLIGHT control error: {e}")

                elif marl_enabled and marl_agent is not None:
                    # Make one decision per TLS every marl_decision_interval seconds
                    if (self.simulation_time - marl_last_decision_t) >= marl_decision_interval:
                        marl_last_decision_t = float(self.simulation_time)
                        try:
                            # Update "last ped green" timestamps based on current phases
                            for tl_id in marl_tls_ids:
                                try:
                                    cur_p = int(traci.trafficlight.getPhase(tl_id))
                                    phase_map = self._get_tl_phase_lane_map(tl_id)
                                    if phase_map.get("phase_has_ped_green", {}).get(cur_p, False):
                                        self._tl_last_ped_green_time[tl_id] = float(self.simulation_time)
                                except Exception:
                                    pass

                            # Build local state per TLS and select actions (inference)
                            actions: Dict[str, int] = {}
                            for tl_id in marl_tls_ids:
                                lane_ids_fixed = marl_tl_lanes.get(tl_id, [])
                                sim_time = float(self.simulation_time)

                                lane_vehicle_counts: Dict[str, int] = {}
                                lane_queue_lengths: Dict[str, int] = {}
                                lane_mean_speeds: Dict[str, float] = {}

                                for lane_id in lane_ids_fixed:
                                    if not lane_id:
                                        lane_vehicle_counts[lane_id] = 0
                                        lane_queue_lengths[lane_id] = 0
                                        lane_mean_speeds[lane_id] = 0.0
                                        continue
                                    try:
                                        lane_vehicle_counts[lane_id] = int(traci.lane.getLastStepVehicleNumber(lane_id))
                                        lane_queue_lengths[lane_id] = int(traci.lane.getLastStepHaltingNumber(lane_id))
                                        lane_mean_speeds[lane_id] = float(traci.lane.getLastStepMeanSpeed(lane_id))
                                    except Exception:
                                        lane_vehicle_counts[lane_id] = 0
                                        lane_queue_lengths[lane_id] = 0
                                        lane_mean_speeds[lane_id] = 0.0

                                try:
                                    phase = int(traci.trafficlight.getPhase(tl_id)) % 8
                                except Exception:
                                    phase = 0

                                try:
                                    remaining = float(traci.trafficlight.getNextSwitch(tl_id) - sim_time)
                                    remaining = float(max(0.0, remaining))
                                except Exception:
                                    remaining = 0.0

                                sumo_state = {
                                    "time": sim_time,
                                    "lane_vehicle_counts": lane_vehicle_counts,
                                    "lane_queue_lengths": lane_queue_lengths,
                                    "lane_mean_speeds": lane_mean_speeds,
                                    "detector_occupancy": {},
                                    "traffic_light_phase": phase,
                                    "traffic_light_remaining_time": remaining,
                                }
                                state_obj = RLTState(sumo_state, lane_list=lane_ids_fixed)
                                actions[tl_id] = int(marl_agent.select_action(state_obj, training=False))

                            # Apply actions
                            for tl_id, act in actions.items():
                                # Cheap pedestrian fairness override
                                try:
                                    phase_map = self._get_tl_phase_lane_map(tl_id)
                                    ped_phases = phase_map.get("ped_green_phases", []) or []
                                    last_ped = float(self._tl_last_ped_green_time.get(tl_id, 0.0))
                                    overdue = (float(self.simulation_time) - last_ped) >= ped_max_red
                                    if overdue and ped_phases:
                                        target_ped_phase = int(ped_phases[0])
                                        traci.trafficlight.setPhase(tl_id, target_ped_phase)
                                        traci.trafficlight.setPhaseDuration(tl_id, ped_min_green)
                                        self._tl_last_ped_green_time[tl_id] = float(self.simulation_time)
                                        continue
                                except Exception:
                                    pass

                                if act != 1:
                                    continue
                                try:
                                    cur_phase = int(traci.trafficlight.getPhase(tl_id))
                                    nph = int(marl_tl_phase_counts.get(tl_id, 1))
                                    traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % max(1, nph))
                                except Exception:
                                    pass
                        except Exception as e:
                            # Don't crash the simulation because of model inference
                            print(f"MARL_DQN control error: {e}")

                elif controller:
                    try:
                        # --- OSM maps: use real SUMO phases per traffic light ---
                        # The original controller logic assumes the custom 4-phase intersection and maps SUMO phases
                        # via _sumo_phase_to_logical(). On OSM maps (Los Angeles/Cologne/Vancouver), TLS can have
                        # arbitrary phase programs; restricting decisions to [0,4] often means the controller NEVER runs.
                        # Here we implement per-TL control using each TLS's own green phases and served lanes.
                        if not SimulationConfig.is_custom_dataset():
                            state_dict = self.get_current_state()

                            def _lane_weighted_queue(lane_id: str) -> float:
                                q = float(state_dict.get("lane_queue_lengths", {}).get(lane_id, 0))
                                bus_c = float(state_dict.get("lane_bus_counts", {}).get(lane_id, 0))
                                bus_w = float(state_dict.get("bus_waiting_time", {}).get(lane_id, 0.0))
                                # buses count more (transit priority)
                                return q + (2.5 * bus_c) + (0.3 * bus_w)

                            def _phase_pressure(tl_id: str, phase_idx: int) -> float:
                                phase_map = self._get_tl_phase_lane_map(tl_id)
                                lanes = phase_map.get("phase_to_lanes", {}).get(int(phase_idx), []) or []
                                return float(sum(_lane_weighted_queue(l) for l in lanes))

                            for tl_id in self.controlled_traffic_lights:
                                try:
                                    cur_phase = int(traci.trafficlight.getPhase(tl_id))
                                    remaining = float(traci.trafficlight.getNextSwitch(tl_id) - self.simulation_time)
                                except Exception:
                                    continue

                                # Track phase active time for max-green cap
                                last = self._tl_last_phase.get(tl_id)
                                if last is None or last != cur_phase:
                                    self._tl_last_phase[tl_id] = cur_phase
                                    self._tl_phase_start_time[tl_id] = float(self.simulation_time)

                                phase_start = float(self._tl_phase_start_time.get(tl_id, self.simulation_time))
                                active_time = float(self.simulation_time) - phase_start
                                max_green = 120.0

                                # Decide only near phase end (cheap + stable)
                                should_check = (remaining <= 1.0) or (self.step_count % 5 == 0 and remaining <= 5.0)
                                if not should_check:
                                    continue

                                phase_map = self._get_tl_phase_lane_map(tl_id)
                                green_phases = phase_map.get("green_phases", []) or []
                                nph = self._get_tl_phase_count(tl_id)
                                if not green_phases:
                                    green_phases = list(range(max(1, nph)))

                                # Cheap pedestrian fairness override (max-red), per TL
                                try:
                                    if phase_map.get("phase_has_ped_green", {}).get(cur_phase, False):
                                        self._tl_last_ped_green_time[tl_id] = float(self.simulation_time)
                                    ped_phases = phase_map.get("ped_green_phases", []) or []
                                    last_ped = float(self._tl_last_ped_green_time.get(tl_id, 0.0))
                                    if ped_phases and (float(self.simulation_time) - last_ped) >= 90.0:
                                        traci.trafficlight.setPhase(tl_id, int(ped_phases[0]))
                                        traci.trafficlight.setPhaseDuration(tl_id, 12.0)
                                        self._tl_last_ped_green_time[tl_id] = float(self.simulation_time)
                                        continue
                                except Exception:
                                    pass

                                # Select next phase based on strategy using real green phases
                                next_phase = cur_phase
                                set_duration = 30.0

                                if strategy == "GA_INTERNAL":
                                    # GA-tuned pressure-duration controller (baseline).
                                    # Cycle through green phases, but set the next green duration as:
                                    #   dur = clamp(10, 60, base_green + pressure_scale * pressure(next_phase))
                                    base_green = float((self._ga_params or {}).get("base_green", 25.0))
                                    pressure_scale = float((self._ga_params or {}).get("pressure_scale", 1.0))
                                    try:
                                        idx = green_phases.index(cur_phase)
                                        next_phase = int(green_phases[(idx + 1) % len(green_phases)])
                                    except ValueError:
                                        next_phase = int(green_phases[0]) if green_phases else ((cur_phase + 1) % max(1, nph))
                                    ph_p = float(_phase_pressure(tl_id, next_phase))
                                    set_duration = float(min(60.0, max(10.0, base_green + pressure_scale * ph_p)))

                                elif strategy == "FIXED_TIME":
                                    # Cycle through green phases (not all phases incl yellow/all-red)
                                    try:
                                        idx = green_phases.index(cur_phase)
                                        next_phase = green_phases[(idx + 1) % len(green_phases)]
                                    except ValueError:
                                        next_phase = green_phases[0]
                                    set_duration = 30.0

                                elif strategy == "MAX_PRESSURE":
                                    # MaxPressure: always serve the phase with maximum pressure.
                                    pressures = {p: _phase_pressure(tl_id, p) for p in green_phases}
                                    best_phase = max(pressures.keys(), key=lambda p: pressures[p]) if pressures else cur_phase
                                    best_pressure = float(pressures.get(best_phase, 0.0))

                                    if int(best_phase) == int(cur_phase):
                                        # Extend proportional to pressure (bounded)
                                        cap_remaining = max(0.0, max_green - active_time)
                                        extension = min(20.0, 5.0 + best_pressure * 1.5)
                                        new_dur = min(cap_remaining, max(0.0, remaining) + extension)
                                        if new_dur > 0.5:
                                            traci.trafficlight.setPhaseDuration(tl_id, float(new_dur))
                                        continue

                                    next_phase = int(best_phase)
                                    set_duration = float(min(60.0, max(15.0, 20.0 + best_pressure * 2.0)))

                                elif strategy == "ADAPTIVE":
                                    # Adaptive (OSM): keep cyclic order, but extend/skips based on demand.
                                    pressures = {p: _phase_pressure(tl_id, p) for p in green_phases}
                                    cur_pressure = float(pressures.get(cur_phase, _phase_pressure(tl_id, cur_phase)))
                                    best_phase = max(pressures.keys(), key=lambda p: pressures[p]) if pressures else cur_phase
                                    best_pressure = float(pressures.get(best_phase, 0.0))

                                    # If there's meaningful demand on current phase, extend a bit (hysteresis).
                                    if cur_pressure >= 1.0 and remaining > 3.0:
                                        cap_remaining = max(0.0, max_green - active_time)
                                        extension = min(15.0, 3.0 + cur_pressure * 1.0)
                                        new_dur = min(cap_remaining, max(0.0, remaining) + extension)
                                        if new_dur > 0.5:
                                            traci.trafficlight.setPhaseDuration(tl_id, float(new_dur))
                                        continue

                                    # Otherwise, move to the next phase in cyclic order.
                                    try:
                                        idx = green_phases.index(cur_phase)
                                        next_phase = int(green_phases[(idx + 1) % len(green_phases)])
                                    except ValueError:
                                        next_phase = int(green_phases[0])

                                    # If the next phase has almost no demand, optionally jump to best.
                                    next_pressure = float(pressures.get(next_phase, _phase_pressure(tl_id, next_phase)))
                                    if best_pressure >= 2.0 and next_pressure < 0.2 * best_pressure:
                                        next_phase = int(best_phase)
                                        next_pressure = best_pressure

                                    set_duration = float(min(60.0, max(15.0, 15.0 + next_pressure * 2.0)))

                                elif strategy == "DQN":
                                    # NOTE: On OSM maps, the "DQN" option in the UI is not a trained model.
                                    # Treat it as a MaxPressure-like baseline with slightly more switching penalty.
                                    pressures = {p: _phase_pressure(tl_id, p) for p in green_phases}
                                    best_phase = max(pressures.keys(), key=lambda p: pressures[p]) if pressures else cur_phase
                                    best_pressure = float(pressures.get(best_phase, 0.0))
                                    cur_pressure = float(pressures.get(cur_phase, _phase_pressure(tl_id, cur_phase)))

                                    # If current is close to best, prefer holding to reduce oscillation.
                                    if cur_pressure >= 0.85 * max(1e-6, best_pressure):
                                        cap_remaining = max(0.0, max_green - active_time)
                                        extension = min(12.0, 3.0 + cur_pressure * 0.8)
                                        new_dur = min(cap_remaining, max(0.0, remaining) + extension)
                                        if new_dur > 0.5:
                                            traci.trafficlight.setPhaseDuration(tl_id, float(new_dur))
                                        continue

                                    next_phase = int(best_phase)
                                    set_duration = float(min(60.0, max(15.0, 18.0 + best_pressure * 1.8)))

                                elif strategy == "SOTL":
                                    # SOTL (OSM): threshold-based switching to the best-demand green phase.
                                    pressures = {p: _phase_pressure(tl_id, p) for p in green_phases}
                                    best_phase = max(pressures.keys(), key=lambda p: pressures[p]) if pressures else cur_phase
                                    best_pressure = float(pressures.get(best_phase, 0.0))
                                    cur_pressure = float(pressures.get(cur_phase, _phase_pressure(tl_id, cur_phase)))

                                    # If current phase still has demand, keep it (small extension).
                                    if cur_pressure >= 1.0 and remaining > 3.0:
                                        cap_remaining = max(0.0, max_green - active_time)
                                        extension = min(12.0, 3.0 + cur_pressure * 0.8)
                                        new_dur = min(cap_remaining, max(0.0, remaining) + extension)
                                        if new_dur > 0.5:
                                            traci.trafficlight.setPhaseDuration(tl_id, float(new_dur))
                                        continue

                                    # Otherwise, if best demand exceeds threshold, switch to it; else cycle.
                                    if best_pressure >= 6.0:
                                        next_phase = int(best_phase)
                                        set_duration = float(min(60.0, max(15.0, 18.0 + best_pressure * 1.5)))
                                    else:
                                        try:
                                            idx = green_phases.index(cur_phase)
                                            next_phase = int(green_phases[(idx + 1) % len(green_phases)])
                                        except ValueError:
                                            next_phase = int(green_phases[0]) if green_phases else ((cur_phase + 1) % max(1, nph))
                                        set_duration = 30.0

                                # Max-green cap: force switch even if controller would keep
                                if active_time >= max_green:
                                    try:
                                        idx = green_phases.index(cur_phase)
                                        next_phase = green_phases[(idx + 1) % len(green_phases)]
                                    except ValueError:
                                        next_phase = green_phases[0] if green_phases else ((cur_phase + 1) % max(1, nph))
                                    set_duration = 30.0

                                if int(next_phase) != int(cur_phase):
                                    try:
                                        traci.trafficlight.setPhase(tl_id, int(next_phase))
                                        traci.trafficlight.setPhaseDuration(tl_id, float(set_duration))
                                    except Exception:
                                        # Fallback: cycle to next valid phase
                                        try:
                                            traci.trafficlight.setPhase(tl_id, (int(cur_phase) + 1) % max(1, nph))
                                        except Exception:
                                            pass

                            # OSM strategy handled for this step; skip legacy (custom-only) mapping logic
                            continue

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
                            phase_remaining_time=phase_remaining_time,
                            lane_bus_counts=state_dict.get('lane_bus_counts', {}),
                            lane_pedestrian_counts=state_dict.get('lane_pedestrian_counts', {}),
                            bus_waiting_time=state_dict.get('bus_waiting_time', {})
                        )

                        # Debug: print controller state every 50 steps
                        if self.step_count % 50 == 0:
                            print(f"Controller debug - Step {self.step_count}: SUMO_phase={sumo_phase}, logical_phase={logical_phase}, remaining={phase_remaining_time:.1f}s, queues={sum(traffic_state.lane_queue_lengths.values())}")

                        # Track phase start time (for fairness / max-green cap)
                        last = self._tl_last_phase.get(tl_id)
                        if last is None:
                            self._tl_last_phase[tl_id] = sumo_phase
                            self._tl_phase_start_time[tl_id] = float(self.simulation_time)
                        elif last != sumo_phase:
                            self._tl_last_phase[tl_id] = sumo_phase
                            self._tl_phase_start_time[tl_id] = float(self.simulation_time)

                        # Check if we should make a control decision:
                        # 1. Phase is about to end (within 1 second) AND we're on a major green phase
                        # 2. Or check every 5 steps, but only near the end (<=5s remaining)
                        green_phases = self._major_green_phases if self._major_green_phases else [0, 4]
                        is_green_phase = (sumo_phase in green_phases)
                        should_check = (
                            is_green_phase and (
                                phase_remaining_time <= 1.0 or
                                (self.step_count % 5 == 0 and phase_remaining_time <= 5.0)
                            )
                        )
                        
                        if should_check:
                            # Get control decision
                            # Note: Use phase_duration to avoid shadowing the function parameter 'duration'
                            next_logical_phase, phase_duration = controller.control_step(traffic_state)

                            # Map logical phase back to SUMO phase
                            next_sumo_phase = self._logical_phase_to_sumo(next_logical_phase, sumo_phase)

                            # --- Fairness / max-green cap (prevents "NS always green") ---
                            # If we're on a major green phase for too long, force a switch to the other major green.
                            # This is especially important for adaptive controllers that may keep extending forever.
                            max_green = 60.0 if SimulationConfig.is_custom_dataset() else 120.0
                            phase_start = self._tl_phase_start_time.get(tl_id, float(self.simulation_time))
                            active_time = float(self.simulation_time) - float(phase_start)
                            if is_green_phase and active_time >= max_green:
                                if self._ew_green_phase is not None and self._ns_green_phase is not None:
                                    forced = self._ew_green_phase if sumo_phase == self._ns_green_phase else self._ns_green_phase
                                    if forced != sumo_phase:
                                        print(f"Max-green cap hit ({active_time:.1f}s). Forcing phase switch {sumo_phase} -> {forced}")
                                        next_sumo_phase = forced
                                        # Use a reasonable green time after forcing
                                        phase_duration = 30.0

                            # Apply control decision to all controlled traffic lights
                            if next_sumo_phase != sumo_phase:
                                print(f"Controller changing phase: SUMO {sumo_phase} (logical {logical_phase}) -> SUMO {next_sumo_phase} (logical {next_logical_phase}) (duration: {phase_duration}s)")
                                for controlled_tl_id in self.controlled_traffic_lights:
                                    try:
                                        # Cheap pedestrian support (max-red): if pedestrians haven't seen green for a while,
                                        # force a pedestrian-green phase for this TL (OSM maps only).
                                        if not SimulationConfig.is_custom_dataset():
                                            try:
                                                cur_p = int(traci.trafficlight.getPhase(controlled_tl_id))
                                                phase_map = self._get_tl_phase_lane_map(controlled_tl_id)
                                                if phase_map.get("phase_has_ped_green", {}).get(cur_p, False):
                                                    self._tl_last_ped_green_time[controlled_tl_id] = float(self.simulation_time)
                                                ped_phases = phase_map.get("ped_green_phases", []) or []
                                                last_ped = float(self._tl_last_ped_green_time.get(controlled_tl_id, 0.0))
                                                overdue = (float(self.simulation_time) - last_ped) >= 90.0
                                                if overdue and ped_phases:
                                                    traci.trafficlight.setPhase(controlled_tl_id, int(ped_phases[0]))
                                                    traci.trafficlight.setPhaseDuration(controlled_tl_id, 12.0)
                                                    self._tl_last_ped_green_time[controlled_tl_id] = float(self.simulation_time)
                                                    continue
                                            except Exception:
                                                pass

                                        # IMPORTANT: Different traffic lights can have different numbers of phases.
                                        # On OSM maps many TLS have only 2–4 phases, so a phase like "4" may be invalid.
                                        n_phases = self._get_tl_phase_count(controlled_tl_id)
                                        current_phase_i = traci.trafficlight.getPhase(controlled_tl_id)

                                        phase_to_set = int(next_sumo_phase)
                                        if phase_to_set < 0 or phase_to_set >= n_phases:
                                            # For custom single-intersection, our detected mapping should be valid,
                                            # but clamp just in case. For OSM maps, fall back to cycling to next phase.
                                            if SimulationConfig.is_custom_dataset():
                                                phase_to_set = max(0, min(n_phases - 1, phase_to_set))
                                            else:
                                                phase_to_set = (current_phase_i + 1) % max(1, n_phases)

                                        traci.trafficlight.setPhase(controlled_tl_id, phase_to_set)
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
                                    # Cap extensions to avoid runaway greens
                                    cap_remaining = max(0.0, max_green - active_time) if is_green_phase else None
                                    if cap_remaining is not None:
                                        new_duration = min(new_duration, cap_remaining)
                                    print(f"Controller extending phase: SUMO {sumo_phase} (logical {logical_phase}) (extend by {phase_duration}s, new duration: {new_duration}s)")
                                    for controlled_tl_id in self.controlled_traffic_lights:
                                        try:
                                            traci.trafficlight.setPhaseDuration(controlled_tl_id, new_duration)
                                        except Exception as e:
                                            print(f"Failed to extend phase for {controlled_tl_id}: {e}")
                                # If phase_duration is 0, do nothing (keep current phase as-is)

                    except Exception as e:
                        # Log error but continue simulation - don't let controller errors stop the simulation
                        print(f"Signal control error at step {self.step_count}: {e}")
                        # Only print full traceback in debug mode or for critical errors
                        if self.step_count % 50 == 0:  # Only print traceback occasionally to avoid spam
                            import traceback
                            traceback.print_exc()
                        # Continue simulation with default SUMO control
                        continue
                    
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

    def _run_ga_optimized(self, duration: Optional[int]) -> Dict:
        """
        Genetic Algorithm (GA) baseline.

        We optimize two global parameters used to set green durations from pressure:
        - base_green: base green time in seconds
        - pressure_scale: how strongly pressure increases green time

        This is a practical GA baseline for SUMO city networks that keeps the search space small
        and computationally manageable, while still being a true optimization loop.
        """
        import random

        eval_duration = int(min(600, duration or 600))  # short evaluation horizon per candidate
        pop_size = 8
        generations = 4
        elite_k = 2
        mutation_prob = 0.25

        # Parameter bounds
        base_range = (10.0, 40.0)
        scale_range = (0.0, 4.0)

        def clamp(x: float, lo: float, hi: float) -> float:
            return float(max(lo, min(hi, x)))

        def make_individual():
            return {
                "base_green": random.uniform(*base_range),
                "pressure_scale": random.uniform(*scale_range),
            }

        def crossover(a, b):
            return {
                "base_green": a["base_green"] if random.random() < 0.5 else b["base_green"],
                "pressure_scale": a["pressure_scale"] if random.random() < 0.5 else b["pressure_scale"],
            }

        def mutate(ind):
            if random.random() < mutation_prob:
                ind["base_green"] = clamp(ind["base_green"] + random.uniform(-5, 5), *base_range)
            if random.random() < mutation_prob:
                ind["pressure_scale"] = clamp(ind["pressure_scale"] + random.uniform(-0.8, 0.8), *scale_range)
            return ind

        def fitness(ind) -> float:
            # Run a short headless simulation for this candidate using GA_INTERNAL strategy
            sim = TrafficSimulator(
                use_gui=False,
                dataset=SimulationConfig.DATASET,
                enable_sumo_emissions_output=False
            )
            sim._ga_params = dict(ind)
            metrics = sim.run_simulation(duration=eval_duration, strategy="GA_INTERNAL")
            # Fitness: lower waiting/queue is better; higher throughput is better
            wt = float(metrics.get("avg_waiting_time", 1e9) or 1e9)
            aql = float(metrics.get("avg_queue_length", 1e9) or 1e9)
            thr = float(metrics.get("throughput_per_hour", 0) or 0)
            # Combine (tuned weights)
            return (-wt) - (0.2 * aql) + (0.001 * thr)

        # Initialize population
        population = [make_individual() for _ in range(pop_size)]
        scored = []

        print(f"GA baseline: optimizing on {SimulationConfig.DATASET} (eval_duration={eval_duration}s, pop={pop_size}, gens={generations})")

        for gen in range(generations):
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_fit, best_ind = scored[0]
            print(f"GA gen {gen+1}/{generations}: best_fitness={best_fit:.3f} params={best_ind}")

            # Select elites
            elites = [dict(scored[i][1]) for i in range(min(elite_k, len(scored)))]
            # Breed next generation
            next_pop = elites[:]
            while len(next_pop) < pop_size:
                p1 = random.choice(elites)
                p2 = random.choice(population)
                child = mutate(crossover(p1, p2))
                next_pop.append(child)
            population = next_pop

        # Final best
        if not scored:
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(key=lambda x: x[0], reverse=True)

        best_params = dict(scored[0][1])
        print(f"GA best params: {best_params}")

        # Run final simulation using tuned params (keep current GUI preference)
        self._ga_params = best_params
        return self.run_simulation(duration=duration, strategy="GA_INTERNAL")
    
    def _collect_step_data(self):
        """Collect simulation data for current step - complete version"""
        try:
            # Sampling: avoid expensive per-step collection on large maps
            if self.data_collection_interval > 1 and (self.step_count % self.data_collection_interval != 0):
                return

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
                    
                    # Emissions are expensive; collect only if enabled
                    if self.collect_emissions:
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
                            veh_data.update({
                                'co2_emission': 0.0,
                                'fuel_consumption': 0.0,
                                'noise_emission': 0.0,
                                'hc_emission': 0.0,
                                'pmx_emission': 0.0,
                                'nox_emission': 0.0
                            })
                    else:
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
        # NOTE: "Throughput" should mean vehicles that completed their trips (arrived),
        # not just vehicles that existed at any point in the simulation.
        vehicles_seen = int(df_vehicles['vehicle_id'].nunique())
        vehicles_arrived = int(getattr(self, "total_arrived", 0) or 0)
        vehicles_departed = int(getattr(self, "total_departed", 0) or 0)

        metrics['vehicles_seen'] = vehicles_seen
        metrics['vehicles_departed'] = vehicles_departed
        metrics['vehicles_arrived'] = vehicles_arrived

        # Keep a legacy key, but make it "arrived vehicles" (true throughput)
        metrics['throughput'] = vehicles_arrived
        
        # Calculate throughput per hour using ACTUAL simulation time
        # This gives realistic throughput values based on what actually happened
        # If simulation ended early, this will reflect that
        if self.simulation_time > 0:
            metrics['throughput_per_hour'] = vehicles_arrived * 3600 / self.simulation_time
            metrics['departed_per_hour'] = vehicles_departed * 3600 / self.simulation_time
        else:
            metrics['throughput_per_hour'] = 0
            metrics['departed_per_hour'] = 0
        
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

        # --- DiffusionLight-style metrics ---
        # AQL: Average Queue Length (vehicles) over time (approx from sampled vehicle_data)
        # Pressure: weighted queued vehicles (buses weighted higher), averaged over time
        try:
            if {'time', 'waiting_time', 'lane_id'}.issubset(df_vehicles.columns):
                dfq = df_vehicles.copy()
                dfq['is_queued'] = dfq['waiting_time'] > 5.0
                queued = dfq[dfq['is_queued']]

                # Total queued vehicles per sampled time step
                q_per_t = queued.groupby('time').size()
                all_times = pd.Index(sorted(dfq['time'].unique()))
                q_per_t = q_per_t.reindex(all_times, fill_value=0)
                metrics['avg_queue_length'] = float(q_per_t.mean())
                metrics['max_queue_length_vehicles'] = float(q_per_t.max())

                # Pressure proxy: weight queued buses more heavily
                if 'type_id' in queued.columns:
                    def _w(t: str) -> float:
                        s = str(t).lower()
                        return 2.5 if ('bus' in s or 'pt_' in s) else 1.0
                    queued['q_weight'] = queued['type_id'].map(_w)
                    p_per_t = queued.groupby('time')['q_weight'].sum().reindex(all_times, fill_value=0.0)
                else:
                    p_per_t = q_per_t.astype(float)

                metrics['avg_pressure'] = float(p_per_t.mean())
                metrics['max_pressure'] = float(p_per_t.max())
        except Exception:
            pass
        
        # 5. Emission metrics
        # Option A: Prefer SUMO-native emissions output for OSM maps (faster, no per-vehicle TraCI calls)
        if not SimulationConfig.is_custom_dataset() and self.emissions_output_file and os.path.exists(self.emissions_output_file):
            emissions_totals = self._parse_emissions_output(self.emissions_output_file)
            metrics.update(emissions_totals)
            # Also compute emissions per km if distance is available
            if df_vehicles is not None and 'distance' in df_vehicles.columns and df_vehicles['distance'].sum() > 0:
                metrics['co2_per_km'] = metrics.get('total_co2', 0.0) / (df_vehicles['distance'].sum() / 1000)
            else:
                metrics['co2_per_km'] = metrics.get('co2_per_km', 0.0)
        else:
            # Fallback: use per-step collected emissions (custom dataset or when enabled)
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

    def _parse_emissions_output(self, file_path: str) -> Dict:
        """
        Parse SUMO emission-output XML in a streaming manner and return totals.
        This avoids keeping per-vehicle per-step emissions in Python memory.
        """
        totals = {
            'total_co2': 0.0,
            'total_fuel': 0.0,
            'total_nox': 0.0,
            'total_pmx': 0.0,
            'total_hc': 0.0,
            'total_co': 0.0,
        }
        per_vehicle_co2 = {}
        per_vehicle_fuel = {}

        try:
            # Stream parse <vehicle ...> entries under <timestep>
            for event, elem in ET.iterparse(file_path, events=("end",)):
                if elem.tag != "vehicle":
                    continue

                vid = elem.get("id")
                def _f(attr: str) -> float:
                    v = elem.get(attr)
                    if v is None:
                        return 0.0
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                co2 = _f("CO2")
                fuel = _f("fuel")
                nox = _f("NOx")
                pmx = _f("PMx")
                hc = _f("HC")
                co = _f("CO")

                totals['total_co2'] += co2
                totals['total_fuel'] += fuel
                totals['total_nox'] += nox
                totals['total_pmx'] += pmx
                totals['total_hc'] += hc
                totals['total_co'] += co

                if vid:
                    per_vehicle_co2[vid] = per_vehicle_co2.get(vid, 0.0) + co2
                    per_vehicle_fuel[vid] = per_vehicle_fuel.get(vid, 0.0) + fuel

                # Free memory
                elem.clear()

            # Add per-vehicle averages if we have any vehicles
            if per_vehicle_co2:
                totals['avg_co2_per_vehicle'] = sum(per_vehicle_co2.values()) / len(per_vehicle_co2)
            else:
                totals['avg_co2_per_vehicle'] = 0.0
            if per_vehicle_fuel:
                totals['avg_fuel_per_vehicle'] = sum(per_vehicle_fuel.values()) / len(per_vehicle_fuel)
            else:
                totals['avg_fuel_per_vehicle'] = 0.0

            return totals
        except Exception as e:
            print(f"Warning: failed to parse emissions output {file_path}: {e}")
            return totals
    
    def _sumo_phase_to_logical(self, sumo_phase: int) -> int:
        """Map SUMO phase number to logical phase (0-3)
        
        SUMO has 6 phases: 0=EW green, 1=EW yellow, 2=NS short, 3=NS yellow, 4=NS green, 5=NS yellow
        Our logical phases: 0=EW straight, 1=EW left, 2=NS straight, 3=NS left
        """
        # Prefer auto-detected major green phases for custom single-intersection
        if self._ew_green_phase is not None and sumo_phase == self._ew_green_phase:
            return 0  # East-West group
        if self._ns_green_phase is not None and sumo_phase == self._ns_green_phase:
            return 2  # North-South group

        # Fallback to legacy mapping used by some networks
        if sumo_phase == 0:
            return 0
        elif sumo_phase == 4:
            return 2
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
        # Prefer auto-detected major green phases for custom single-intersection
        if self._ew_green_phase is not None and self._ns_green_phase is not None:
            if logical_phase in [0, 1]:
                return self._ew_green_phase
            elif logical_phase in [2, 3]:
                return self._ns_green_phase

        # Fallback to legacy mapping
        if logical_phase in [0, 1]:
            return 0
        elif logical_phase in [2, 3]:
            return 4
        else:
            return current_sumo_phase  # Keep current if unknown

    def _detect_major_green_phases(self, tl_id: str) -> None:
        """
        Detect which SUMO phases correspond to EW vs NS green for the custom single intersection.

        Many generated tlLogics do NOT use the hard-coded phase indices (0 and 4).
        If we don't detect this, controllers may never run (because we only check "green phases")
        and phase mapping becomes wrong.
        """
        self._ew_green_phase = None
        self._ns_green_phase = None
        self._major_green_phases = []

        # Get controlled links, map signal index -> incoming lane/edge prefix
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        # controlled_links is a list where index i corresponds to state[i]
        signal_in_lanes: List[Optional[str]] = []
        for link_list in controlled_links:
            in_lane = None
            if link_list and len(link_list) > 0 and link_list[0] and len(link_list[0]) > 0:
                in_lane = link_list[0][0]  # (inLane, outLane, viaLane)
            signal_in_lanes.append(in_lane)

        # Load current program phases
        programs = traci.trafficlight.getAllProgramLogics(tl_id)
        if not programs:
            return
        phases = programs[0].phases
        if not phases:
            return

        best_ew = (-1, None)  # (score, phase_index)
        best_ns = (-1, None)

        for idx, ph in enumerate(phases):
            state = getattr(ph, "state", "")
            if not state:
                continue

            ew_score = 0
            ns_score = 0
            for si, ch in enumerate(state):
                if ch not in ("g", "G"):
                    continue
                in_lane = signal_in_lanes[si] if si < len(signal_in_lanes) else None
                if not in_lane:
                    continue
                lane_lower = in_lane.lower()
                if lane_lower.startswith("east") or lane_lower.startswith("west"):
                    ew_score += 1
                elif lane_lower.startswith("north") or lane_lower.startswith("south"):
                    ns_score += 1

            # We only consider phases that actually grant some green
            if ew_score > best_ew[0]:
                best_ew = (ew_score, idx)
            if ns_score > best_ns[0]:
                best_ns = (ns_score, idx)

        ew_idx = best_ew[1]
        ns_idx = best_ns[1]
        if ew_idx is None or ns_idx is None:
            return
        if ew_idx == ns_idx:
            # If the same phase wins both (degenerate), bail out
            return

        self._ew_green_phase = int(ew_idx)
        self._ns_green_phase = int(ns_idx)
        self._major_green_phases = [self._ew_green_phase, self._ns_green_phase]
    
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
            'lane_mean_speeds': {},
            'lane_bus_counts': {},
            'lane_pedestrian_counts': {},
            'bus_waiting_time': {}
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
                    
                    # Count buses and their waiting times (transit priority)
                    bus_count = 0
                    bus_wait_times = []
                    for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                        try:
                            veh_type = traci.vehicle.getTypeID(veh_id)
                            # Check if vehicle is a bus (vClass bus or type contains 'bus')
                            if 'bus' in veh_type.lower() or 'pt_' in veh_type.lower():
                                bus_count += 1
                                wait_time = traci.vehicle.getWaitingTime(veh_id)
                                if wait_time > 0:
                                    bus_wait_times.append(wait_time)
                        except:
                            pass
                    state['lane_bus_counts'][lane_id] = bus_count
                    state['bus_waiting_time'][lane_id] = sum(bus_wait_times) / len(bus_wait_times) if bus_wait_times else 0.0
                    
                    # Count pedestrians waiting at crossings (if available)
                    # Note: Pedestrians in SUMO are separate from vehicles
                    # They typically have separate signals, but we can check for pedestrian crossings
                    state['lane_pedestrian_counts'][lane_id] = 0  # Will be enhanced if pedestrian detection is needed
                    
                except:
                    state['lane_vehicle_counts'][lane_id] = 0
                    state['lane_queue_lengths'][lane_id] = 0
                    state['lane_mean_speeds'][lane_id] = 0
                    state['lane_bus_counts'][lane_id] = 0
                    state['lane_pedestrian_counts'][lane_id] = 0
                    state['bus_waiting_time'][lane_id] = 0.0
            
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