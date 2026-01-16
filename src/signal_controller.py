#!/usr/bin/env python3
"""
Signal Control Module (SignalController)
Implements three control strategies:
1. Traditional fixed-time control (fixed cycle)
2. Rule-based adaptive control (based on vehicle counts per lane from SUMO)
3. DQN reinforcement learning control (based on state-action-reward mechanism)
Supports both custom single intersection and TAPASCologne dataset
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

# Import configuration
try:
    from src.config import SimulationConfig
except ImportError:
    from config import SimulationConfig

class ControlStrategy(Enum):
    """Control strategy enumeration"""
    FIXED_TIME = "fixed_time"      # Traditional fixed-time control
    ADAPTIVE = "adaptive"          # Rule-based adaptive control
    DQN = "dqn"                    # DQN reinforcement learning control
    MAX_PRESSURE = "max_pressure"  # MaxPressure queue-based control
    SOTL = "sotl"                  # Self-Organizing Traffic Lights (threshold-based)

class PhaseType(Enum):
    """Signal phase type"""
    EAST_WEST_THROUGH = 0     # East-west straight
    EAST_WEST_LEFT = 1        # East-west left turn
    NORTH_SOUTH_THROUGH = 2   # North-south straight
    NORTH_SOUTH_LEFT = 3      # North-south left turn

@dataclass
class PhaseConfig:
    """Phase configuration"""
    phase_id: int
    name: str
    duration: float  # Base duration (seconds)
    min_duration: float  # Minimum green time
    max_duration: float  # Maximum green time
    yellow_time: float = 3.0   # Yellow time
    red_clear_time: float = 2.0  # All-red clearance time  

@dataclass
class TrafficState:
    """Traffic state information with support for buses and pedestrians"""
    time: float
    lane_vehicle_counts: Dict[str, int]
    lane_queue_lengths: Dict[str, int]
    lane_mean_speeds: Dict[str, float]
    detector_occupancy: Dict[str, float]
    current_phase: int
    phase_remaining_time: float
    # Vehicle type breakdowns (for buses, pedestrians, etc.) - Optional fields with defaults
    lane_bus_counts: Optional[Dict[str, int]] = None  # Number of buses per lane
    lane_pedestrian_counts: Optional[Dict[str, int]] = None  # Number of pedestrians waiting per lane
    bus_waiting_time: Optional[Dict[str, float]] = None  # Average waiting time for buses per lane

class BaseController(ABC):
    """Signal controller base class"""
    
    def __init__(self, intersection_id: str = "intersection", lane_list: List[str] = None):
        self.intersection_id = intersection_id
        self.current_phase = 0
        self.phase_start_time = 0.0
        self.decision_history = []
        
        # Standard four-phase configuration (for custom dataset)
        self.phases = {
            0: PhaseConfig(0, "East-West Straight", 30.0, 10.0, 60.0),
            1: PhaseConfig(1, "East-West Left Turn", 15.0, 8.0, 30.0),
            2: PhaseConfig(2, "North-South Straight", 30.0, 10.0, 60.0),
            3: PhaseConfig(3, "North-South Left Turn", 15.0, 8.0, 30.0)
        }
        
        # Lane list (from config or provided)
        if lane_list is None:
            config = SimulationConfig.get_current_config()
            if config['lanes'] is not None:
                self.lane_list = config['lanes']
            else:
                self.lane_list = []  # Will be set dynamically
        else:
            self.lane_list = lane_list
        
        # Lane grouping (for custom dataset)
        self.lane_groups = {
            'east_west': ['east_in_0', 'east_in_1', 'west_in_0', 'west_in_1'],
            'north_south': ['north_in_0', 'north_in_1', 'south_in_0', 'south_in_1']
        }
    
    @abstractmethod
    def decide_next_phase(self, traffic_state: TrafficState) -> Tuple[int, float]:
        """
        Decide next phase and duration based on traffic state
        
        Args:
            traffic_state: current traffic state
            
        Returns:
            Tuple[int, float]: (next phase ID, duration)
        """
        pass
    
    def get_phase_name(self, phase_id: int) -> str:
        """Get phase name"""
        return self.phases.get(phase_id, PhaseConfig(phase_id, f"Phase {phase_id}", 30.0, 10.0, 60.0)).name
    
    def log_decision(self, traffic_state: TrafficState, decision: Tuple[int, float], reason: str = ""):
        """Log control decision"""
        log_entry = {
            'time': traffic_state.time,
            'current_phase': traffic_state.current_phase,
            'next_phase': decision[0],
            'duration': decision[1],
            'reason': reason,
            'vehicle_counts': traffic_state.lane_vehicle_counts,
            'queue_lengths': traffic_state.lane_queue_lengths
        }
        self.decision_history.append(log_entry)

class FixedTimeController(BaseController):
    """Traditional fixed-time controller"""
    
    def __init__(self, intersection_id: str = "intersection", cycle_time: float = 120.0):
        super().__init__(intersection_id)
        self.cycle_time = cycle_time
        self.strategy_name = "Fixed-time Control"
        
        # Standard timing plan: East-West Straight 30s -> East-West Left Turn 15s -> North-South Straight 30s -> North-South Left Turn 15s -> interval
        self.timing_plan = [
            (0, 30.0),  # East-West Straight
            (1, 15.0),  # East-West Left Turn
            (2, 30.0),  # North-South Straight
            (3, 15.0)   # North-South Left Turn
        ]
        self.plan_index = 0
        self.initialized = False  # Track if we've initialized based on current phase
    
    def decide_next_phase(self, traffic_state: TrafficState) -> Tuple[int, float]:
        """Fixed-time control logic"""
        # Initialize plan_index based on current phase if first time
        if not self.initialized:
            # Find which phase in our plan matches the current phase
            for idx, (phase_id, _) in enumerate(self.timing_plan):
                if phase_id == traffic_state.current_phase:
                    self.plan_index = idx
                    break
            self.initialized = True
        
        # Only switch phase if current phase is about to end (within 1 second)
        # This prevents switching too early
        if traffic_state.phase_remaining_time > 1.0:
            # Keep current phase, return extension of 0 (no change)
            return traffic_state.current_phase, 0.0
        
        # Phase is ending, switch to next phase in sequence
        # Advance to next phase in plan
        self.plan_index = (self.plan_index + 1) % len(self.timing_plan)
        next_phase, duration = self.timing_plan[self.plan_index]
        
        reason = f"Fixed sequence control - {self.get_phase_name(next_phase)}"
        self.log_decision(traffic_state, (next_phase, duration), reason)
        
        return next_phase, duration

class AdaptiveController(BaseController):
    """Rule-based adaptive controller"""
    
    def __init__(self, intersection_id: str = "intersection"):
        super().__init__(intersection_id)
        self.strategy_name = "Adaptive Control"
        
        # Adaptive control parameters
        self.min_green_extension = 5.0    # Minimum green extension time
        self.max_green_extension = 20.0   # Maximum green extension time
        self.queue_threshold = 5          # Queue length threshold
        self.occupancy_threshold = 0.3    # Occupancy threshold
        self.speed_threshold = 5.0        # Speed threshold (m/s)
    
    def decide_next_phase(self, traffic_state: TrafficState) -> Tuple[int, float]:
        """Adaptive control logic with bus priority and pedestrian awareness"""
        current_phase = traffic_state.current_phase

        # Simple logic: check if current phase has waiting vehicles
        current_lanes = self._get_phase_lanes(current_phase)
        # Handle None or empty lanes - use all available lanes from traffic_state for OSM maps
        if not current_lanes and hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
            current_lanes = traffic_state.lane_list
        
        # Calculate weighted queue (buses count more due to transit priority)
        current_queue = 0
        current_bus_count = 0
        current_bus_wait = 0.0
        for lane in (current_lanes or []):
            queue = traffic_state.lane_queue_lengths.get(lane, 0)
            bus_count = (traffic_state.lane_bus_counts or {}).get(lane, 0)
            bus_wait = (traffic_state.bus_waiting_time or {}).get(lane, 0.0)
            # Buses count as 2x regular vehicles (transit priority)
            current_queue += queue + bus_count  # Add bus count as extra weight
            current_bus_count += bus_count
            current_bus_wait += bus_wait

        # Transit priority: If buses are waiting, give them priority
        bus_priority_bonus = 0
        if current_bus_count > 0:
            # Add extra time for buses (they're longer and need more time to clear)
            bus_priority_bonus = min(10.0, current_bus_count * 3.0 + current_bus_wait * 0.5)

        # If current phase has significant queue (including bus priority) and time remaining, extend it
        if (current_queue > 2 or current_bus_count > 0) and traffic_state.phase_remaining_time > 5:
            extension = min(self.max_green_extension, current_queue * 2.0 + bus_priority_bonus)
            reason = f"Current phase has {current_queue} vehicles ({current_bus_count} buses), extend {extension:.1f}s"
            if current_bus_count > 0:
                reason += f" [Transit Priority: +{bus_priority_bonus:.1f}s]"
            self.log_decision(traffic_state, (current_phase, extension), reason)
            return current_phase, extension

        # Otherwise, switch to next phase in sequence (like fixed time but adaptive duration)
        next_phase = (current_phase + 1) % len(self.phases)

        # Adjust duration based on queue length and bus priority for next phase
        next_lanes = self._get_phase_lanes(next_phase)
        # Handle None or empty lanes
        if not next_lanes and hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
            next_lanes = traffic_state.lane_list
        
        next_queue = 0
        next_bus_count = 0
        next_bus_wait = 0.0
        for lane in (next_lanes or []):
            queue = traffic_state.lane_queue_lengths.get(lane, 0)
            bus_count = (traffic_state.lane_bus_counts or {}).get(lane, 0)
            bus_wait = (traffic_state.bus_waiting_time or {}).get(lane, 0.0)
            next_queue += queue + bus_count  # Buses count extra
            next_bus_count += bus_count
            next_bus_wait += bus_wait
        
        base_duration = self.phases[next_phase].duration
        next_bus_bonus = min(10.0, next_bus_count * 3.0 + next_bus_wait * 0.5) if next_bus_count > 0 else 0

        if next_queue > 5 or next_bus_count > 0:
            duration = min(self.phases[next_phase].max_duration, 
                          base_duration * 1.5 + next_bus_bonus)
        elif next_queue < 1:
            duration = max(self.phases[next_phase].min_duration, base_duration * 0.7)
        else:
            duration = base_duration + next_bus_bonus

        reason = f"Switch to {self.get_phase_name(next_phase)}, queue: {next_queue}"
        if next_bus_count > 0:
            reason += f" ({next_bus_count} buses, Transit Priority)"
        self.log_decision(traffic_state, (next_phase, duration), reason)

        return next_phase, duration
    
    def _calculate_demand_score(self, traffic_state: TrafficState, phase_id: int) -> float:
        """Calculate phase demand score [0, 1]"""
        # Determine relevant lanes based on phase
        relevant_lanes = self._get_phase_lanes(phase_id)
        
        if not relevant_lanes:
            return 0.0
        
        # Calculate demand using multiple metrics
        queue_score = 0.0
        occupancy_score = 0.0
        speed_score = 0.0
        
        for lane in relevant_lanes:
            # Queue length score
            queue_length = traffic_state.lane_queue_lengths.get(lane, 0)
            queue_score += min(1.0, queue_length / 10.0)  # 10 vehicles = full score

            # Occupancy score - handle detector naming carefully
            detector_key = f"det_{lane.replace('_in_', '_')}"
            occupancy = traffic_state.detector_occupancy.get(detector_key, 0)
            occupancy_score += min(1.0, occupancy / 0.8)  # 80% occupancy = full score
            
            # Speed score (lower speed = higher demand)
            mean_speed = traffic_state.lane_mean_speeds.get(lane, 10.0)
            speed_score += max(0.0, 1.0 - mean_speed / 15.0)  # Above 15m/s = 0 score
        
        # Normalize scores
        num_lanes = len(relevant_lanes)
        if num_lanes > 0:
            queue_score /= num_lanes
            occupancy_score /= num_lanes
            speed_score /= num_lanes
        
        # Weighted composite score
        demand_score = 0.4 * queue_score + 0.3 * occupancy_score + 0.3 * speed_score
        return min(1.0, demand_score)
    
    def _get_phase_lanes(self, phase_id: int) -> List[str]:
        """Get lanes corresponding to phase"""
        # For custom single-intersection dataset with predefined mapping.
        # Lane indices follow the convention used in `routes.rou.xml`:
        #   0 = right turn, 1 = straight, 2 = left turn
        if SimulationConfig.is_custom_dataset():
            phase_lane_mapping = {
                # Phase 0: East–West straight movement (use straight lanes,
                # optionally including right‑turn lanes in the same group)
                0: ['east_in_1', 'west_in_1', 'east_in_0', 'west_in_0'],
                # Phase 1: East–West left turns
                1: ['east_in_2', 'west_in_2'],
                # Phase 2: North–South straight movement
                2: ['north_in_1', 'south_in_1', 'north_in_0', 'south_in_0'],
                # Phase 3: North–South left turns
                3: ['north_in_2', 'south_in_2']
            }
            return phase_lane_mapping.get(phase_id, [])
        else:
            # For TAPASCologne, use all lanes (simplified approach)
            # In a real implementation, you'd map phases to specific lanes
            return self.lane_list if hasattr(self, 'lane_list') else []

class DQNController(BaseController):
    """DQN reinforcement learning controller (simplified version)"""
    
    def __init__(self, intersection_id: str = "intersection", model_path: str = None):
        super().__init__(intersection_id)
        # Important: this is a simplified heuristic/Q-table placeholder, not the trained MARL model.
        self.strategy_name = "DQN-like Heuristic Baseline"
        self.model_path = model_path
        
        # DQN parameters
        self.state_dim = 12  # 8 lane vehicle counts + 4 phase time states
        self.action_dim = 2  # 0: keep current phase, 1: switch to next phase
        self.epsilon = 0.1   # Exploration rate
        
        # Q-value table (simplified version, should actually be a neural network)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        
        # Experience storage
        self.experience_buffer = []
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        
        # Performance statistics
        self.episode_rewards = []
        self.episode_steps = 0
        
        if model_path:
            self.load_model(model_path)

    def _get_phase_lanes(self, phase_id: int) -> List[str]:
        """Get lanes corresponding to phase"""
        # For custom single-intersection dataset with predefined mapping.
        # Lane indices follow the convention used in `routes.rou.xml`:
        #   0 = right turn, 1 = straight, 2 = left turn
        if SimulationConfig.is_custom_dataset():
            phase_lane_mapping = {
                # Phase 0: East–West straight movement (use straight lanes,
                # optionally including right‑turn lanes in the same group)
                0: ['east_in_1', 'west_in_1', 'east_in_0', 'west_in_0'],
                # Phase 1: East–West left turns
                1: ['east_in_2', 'west_in_2'],
                # Phase 2: North–South straight movement
                2: ['north_in_1', 'south_in_1', 'north_in_0', 'south_in_0'],
                # Phase 3: North–South left turns
                3: ['north_in_2', 'south_in_2']
            }
            return phase_lane_mapping.get(phase_id, [])
        else:
            # For OSM maps, lanes are auto-detected and stored in traffic_state
            # Use lanes from traffic_state if available, otherwise return empty list
            # This prevents TypeError when lanes is None in config
            config = SimulationConfig.get_current_config()
            lanes = config.get('lanes')
            if lanes is None:
                # For OSM maps, return empty list and use traffic_state lanes instead
                # The calling code should handle this by using traffic_state.lane_list
                return []
            return lanes if isinstance(lanes, list) else []

    def decide_next_phase(self, traffic_state: TrafficState) -> Tuple[int, float]:
        """DQN control logic with bus priority - simplified for stability"""
        current_phase = traffic_state.current_phase

        # Simple rule-based decision (can be enhanced with actual RL later)
        current_lanes = self._get_phase_lanes(current_phase)
        # Handle None or empty lanes - use all available lanes from traffic_state for OSM maps
        if not current_lanes and hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
            current_lanes = traffic_state.lane_list
        
        # Calculate weighted queue with bus priority
        current_queue = 0
        current_bus_count = 0
        for lane in (current_lanes or []):
            queue = traffic_state.lane_queue_lengths.get(lane, 0)
            bus_count = (traffic_state.lane_bus_counts or {}).get(lane, 0)
            current_queue += queue + (bus_count * 2.0)  # Buses count as 2x
            current_bus_count += bus_count
        
        bus_bonus = min(8.0, current_bus_count * 2.5) if current_bus_count > 0 else 0

        # If current phase has vehicles waiting (including bus priority), keep it green longer
        if (current_queue > 1 or current_bus_count > 0) and traffic_state.phase_remaining_time > 3:
            next_phase = current_phase
            duration = min(self.phases[current_phase].max_duration,
                          self.phases[current_phase].duration + current_queue * 1.5 + bus_bonus)
            reason = f"DQN: Keep current phase (queue: {current_queue}"
            if current_bus_count > 0:
                reason += f", {current_bus_count} buses [Transit Priority]"
            reason += ")"
        else:
            # Switch to next phase
            next_phase = (current_phase + 1) % len(self.phases)
            next_lanes = self._get_phase_lanes(next_phase)
            # Handle None or empty lanes - use all available lanes from traffic_state for OSM maps
            if not next_lanes and hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
                next_lanes = traffic_state.lane_list
            
            next_queue = 0
            next_bus_count = 0
            for lane in (next_lanes or []):
                queue = traffic_state.lane_queue_lengths.get(lane, 0)
                bus_count = (traffic_state.lane_bus_counts or {}).get(lane, 0)
                next_queue += queue + (bus_count * 2.0)
                next_bus_count += bus_count
            
            next_bus_bonus = min(8.0, next_bus_count * 2.5) if next_bus_count > 0 else 0
            base_duration = self.phases[next_phase].duration
            duration = min(self.phases[next_phase].max_duration,
                          max(self.phases[next_phase].min_duration, base_duration + next_queue * 1.0 + next_bus_bonus))
            reason = f"DQN: Switch to next phase (next_queue: {next_queue}"
            if next_bus_count > 0:
                reason += f", {next_bus_count} buses [Transit Priority]"
            reason += ")"

        # Log decision
        self.log_decision(traffic_state, (next_phase, duration), reason)

        return next_phase, duration
    
    def _state_to_vector(self, traffic_state: TrafficState) -> np.ndarray:
        """Convert traffic state to vector"""
        state_vector = []
        
        # Vehicle counts per lane (normalized)
        for lane in ['east_in_0', 'east_in_1', 'west_in_0', 'west_in_1',
                     'north_in_0', 'north_in_1', 'south_in_0', 'south_in_1']:
            count = traffic_state.lane_vehicle_counts.get(lane, 0)
            state_vector.append(min(1.0, count / 20.0))  # Normalize to [0,1]
        
        # Current phase information (one-hot encoding)
        for i in range(4):
            state_vector.append(1.0 if traffic_state.current_phase == i else 0.0)
        
        return np.array(state_vector)
    
    def _calculate_reward(self, traffic_state: TrafficState) -> float:
        """Calculate reward function with bus priority"""
        # Reward function: negative waiting time - negative queue length
        total_queue_length = sum(traffic_state.lane_queue_lengths.values())
        
        # Base reward: reduce queue length
        queue_penalty = -0.1 * total_queue_length
        
        # Bus priority reward: strongly penalize bus waiting (transit priority)
        total_bus_count = sum((traffic_state.lane_bus_counts or {}).values())
        total_bus_wait = sum((traffic_state.bus_waiting_time or {}).values())
        bus_penalty = -0.3 * total_bus_count - 0.2 * total_bus_wait  # Higher penalty for buses
        
        # Speed reward: encourage high speed
        avg_speed = np.mean(list(traffic_state.lane_mean_speeds.values())) if traffic_state.lane_mean_speeds else 0
        speed_reward = 0.05 * avg_speed
        
        # Balance reward: encourage balance across directions
        # For OSM maps, use all lanes if custom lanes not available
        if hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
            all_lanes = traffic_state.lane_list
            # Split lanes roughly in half for balance calculation
            mid = len(all_lanes) // 2
            ew_vehicles = sum(traffic_state.lane_vehicle_counts.get(lane, 0) for lane in all_lanes[:mid])
            ns_vehicles = sum(traffic_state.lane_vehicle_counts.get(lane, 0) for lane in all_lanes[mid:])
        else:
            # Custom dataset lanes
            ew_vehicles = sum(traffic_state.lane_vehicle_counts.get(lane, 0) 
                             for lane in ['east_in_0', 'east_in_1', 'west_in_0', 'west_in_1'])
            ns_vehicles = sum(traffic_state.lane_vehicle_counts.get(lane, 0)
                             for lane in ['north_in_0', 'north_in_1', 'south_in_0', 'south_in_1'])
        
        balance_penalty = -0.05 * abs(ew_vehicles - ns_vehicles)
        
        total_reward = queue_penalty + bus_penalty + speed_reward + balance_penalty
        
        return total_reward
    
    def _get_best_action(self, state_key: Tuple) -> int:
        """Get optimal action"""
        if state_key not in self.q_table:
            return np.random.randint(0, self.action_dim)
        
        q_values = self.q_table[state_key]
        return max(q_values.keys(), key=lambda a: q_values.get(a, 0.0))
    
    def _update_q_values(self, state: Tuple, action: int, reward: float, next_state: np.ndarray):
        """Update Q values"""
        next_state_key = tuple(np.round(next_state, 2))
        
        # Initialize Q table
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.action_dim)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in range(self.action_dim)}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state_key].values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def _store_experience(self, state: Tuple, action: int, reward: float, next_state: np.ndarray):
        """Store experience"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': tuple(np.round(next_state, 2)),
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)
        
        # Limit experience buffer size
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
    
    def save_model(self, path: str):
        """Save model"""
        model_data = {
            'q_table': {str(k): v for k, v in self.q_table.items()},
            'episode_rewards': self.episode_rewards,
            'epsilon': self.epsilon
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
    
    def load_model(self, path: str):
        """Load model"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # Restore Q table
            self.q_table = {}
            for k, v in model_data.get('q_table', {}).items():
                self.q_table[eval(k)] = v  # Convert string key back to tuple
            
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.epsilon = model_data.get('epsilon', 0.1)
            
            print(f"Successfully loaded DQN model: {path}")
        except Exception as e:
            print(f"Failed to load DQN model: {e}")


class SOTLController(BaseController):
    """
    Self-Organizing Traffic Lights (SOTL).

    Simple, strong baseline used in many TSC papers:
    - Maintain current green for at least min_green.
    - Compute demand on other phases; if demand exceeds threshold, switch.
    - Also enforce a max_green cap to prevent starvation.

    Note: For OSM maps, lane-phase mapping is handled in TrafficSimulator (per-TL),
    so this controller is mainly meaningful for the custom single-intersection dataset.
    """

    def __init__(self, intersection_id: str = "intersection"):
        super().__init__(intersection_id)
        self.strategy_name = "SOTL (Self-Organizing Traffic Lights)"

        # Typical SOTL parameters
        self.min_green = 10.0
        self.max_green = 60.0
        self.switch_threshold = 6.0  # vehicles (weighted later by buses)

    def decide_next_phase(self, traffic_state: TrafficState) -> Tuple[int, float]:
        current_phase = traffic_state.current_phase

        # If still plenty of time, keep current (do not preempt)
        if traffic_state.phase_remaining_time > 1.0:
            return current_phase, 0.0

        # Determine lanes for each phase (custom mapping)
        def phase_demand(phase_id: int) -> float:
            lanes = self._get_phase_lanes(phase_id)
            if not lanes and hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
                lanes = traffic_state.lane_list
            demand = 0.0
            for lane in (lanes or []):
                q = float(traffic_state.lane_queue_lengths.get(lane, 0))
                bus_c = float((traffic_state.lane_bus_counts or {}).get(lane, 0))
                bus_w = float((traffic_state.bus_waiting_time or {}).get(lane, 0.0))
                demand += q + (2.0 * bus_c) + (0.2 * bus_w)
            return demand

        # Phase active time (approx): if we've been green too long, force switch
        active_time = 0.0
        try:
            active_time = float(traffic_state.time) - float(self.phase_start_time)
        except Exception:
            active_time = 0.0

        # Compute demand on other phases
        other_phases = [p for p in self.phases.keys() if p != current_phase]
        other_demands = {p: phase_demand(p) for p in other_phases}
        best_phase = max(other_demands.keys(), key=lambda p: other_demands[p]) if other_demands else current_phase
        best_demand = float(other_demands.get(best_phase, 0.0))

        # Decide: if other demand exceeds threshold OR max green hit => switch to best
        if best_demand >= self.switch_threshold or active_time >= self.max_green:
            duration = float(self.phases[best_phase].duration)
            reason = f"SOTL switch: other_demand={best_demand:.1f} threshold={self.switch_threshold:.1f}"
            self.log_decision(traffic_state, (best_phase, duration), reason)
            # update phase start time for this controller (logical time)
            self.phase_start_time = float(traffic_state.time)
            return best_phase, duration

        # Otherwise keep current (small extension) to avoid oscillation
        extension = min(15.0, max(0.0, self.min_green))
        reason = f"SOTL hold: other_demand={best_demand:.1f} < threshold"
        self.log_decision(traffic_state, (current_phase, extension), reason)
        return current_phase, extension


class MaxPressureController(BaseController):
    """
    MaxPressure controller (simplified queue-based implementation).

    For each phase, compute a "pressure" score as the sum of queue lengths on
    the lanes that receive green in that phase, and serve the phase with the
    highest pressure. If the current phase already has the highest pressure and
    still has remaining time, extend it.
    """

    def __init__(self, intersection_id: str = "intersection"):
        super().__init__(intersection_id)
        self.strategy_name = "MaxPressure Control"

        # Minimum / maximum extension for current phase when it remains best
        self.min_extension = 5.0
        self.max_extension = 20.0

    def decide_next_phase(self, traffic_state: TrafficState) -> Tuple[int, float]:
        """MaxPressure control logic with bus priority"""
        current_phase = traffic_state.current_phase

        # Compute pressure for all phases (with bus priority weighting)
        phase_pressures: Dict[int, float] = {}
        # Get all available lanes (for OSM maps, use traffic_state lanes)
        all_lanes = []
        if hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
            all_lanes = traffic_state.lane_list
        elif hasattr(self, 'lane_list') and self.lane_list:
            all_lanes = self.lane_list
        
        for phase_id in self.phases.keys():
            lanes = self._get_phase_lanes(phase_id)
            # Handle None or empty lanes - use all available lanes for OSM maps
            if not lanes:
                lanes = all_lanes
            pressure = 0.0
            for lane in (lanes or []):
                q = traffic_state.lane_queue_lengths.get(lane, 0)
                # Add bus priority: buses count as 2.5x regular vehicles
                bus_count = (traffic_state.lane_bus_counts or {}).get(lane, 0)
                bus_wait = (traffic_state.bus_waiting_time or {}).get(lane, 0.0)
                # Transit priority: buses add significant pressure
                pressure += q + (bus_count * 2.5) + (bus_wait * 0.3)
            phase_pressures[phase_id] = pressure

        # Choose phase with maximum pressure
        best_phase = max(phase_pressures.keys(), key=lambda p: phase_pressures[p])
        best_pressure = phase_pressures[best_phase]
        current_pressure = phase_pressures.get(current_phase, 0.0)

        # If current phase is already best and has remaining time, extend it
        if best_phase == current_phase and traffic_state.phase_remaining_time > 0:
            # Calculate bus bonus for current phase
            current_lanes = self._get_phase_lanes(current_phase)
            if not current_lanes and hasattr(traffic_state, 'lane_list') and traffic_state.lane_list:
                current_lanes = traffic_state.lane_list
            
            current_bus_count = sum((traffic_state.lane_bus_counts or {}).get(lane, 0) 
                                   for lane in (current_lanes or []))
            bus_bonus = min(8.0, current_bus_count * 2.0) if current_bus_count > 0 else 0
            
            pressure_factor = min(1.0, best_pressure / 10.0)  # 10 queued vehicles → full factor
            extension = max(self.min_extension,
                            min(self.max_extension, pressure_factor * self.max_extension + bus_bonus))
            reason = (f"Keep current phase (MaxPressure) - pressure={current_pressure:.1f}, "
                      f"extend {extension:.1f}s")
            if current_bus_count > 0:
                reason += f" [Transit Priority: {current_bus_count} buses]"
            self.log_decision(traffic_state, (current_phase, extension), reason)
            return current_phase, extension

        # Otherwise, switch to the phase with highest pressure
        base_cfg = self.phases[best_phase]
        # Normalize pressure into [0,1] based on a reference queue size
        norm_p = min(1.0, best_pressure / 15.0)  # 15 queued vehicles → max duration
        duration = base_cfg.min_duration + norm_p * (base_cfg.max_duration - base_cfg.min_duration)

        reason = (f"Switch to {self.get_phase_name(best_phase)} (MaxPressure) - "
                  f"best_pressure={best_pressure:.1f}, current_pressure={current_pressure:.1f}")
        self.log_decision(traffic_state, (best_phase, duration), reason)

        return best_phase, duration

    def _get_phase_lanes(self, phase_id: int) -> List[str]:
        """
        Get the lanes that are served by a given phase.

        This mirrors the mapping in AdaptiveController so that both baselines
        use consistent lane–phase relationships on the custom single intersection.
        """
        if SimulationConfig.is_custom_dataset():
            phase_lane_mapping = {
                # Phase 0: East–West straight (include right + straight lanes)
                0: ['east_in_1', 'west_in_1', 'east_in_0', 'west_in_0'],
                # Phase 1: East–West left
                1: ['east_in_2', 'west_in_2'],
                # Phase 2: North–South straight (include right + straight lanes)
                2: ['north_in_1', 'south_in_1', 'north_in_0', 'south_in_0'],
                # Phase 3: North–South left
                3: ['north_in_2', 'south_in_2']
            }
            return phase_lane_mapping.get(phase_id, [])
        else:
            # For multi-intersection datasets, fall back to all controlled lanes
            return self.lane_list if hasattr(self, 'lane_list') else []

class SignalController:
    """Signal controller main class"""
    
    def __init__(self, strategy: ControlStrategy = ControlStrategy.FIXED_TIME, **kwargs):
        self.strategy = strategy
        
        # Create specific controller based on strategy
        if strategy == ControlStrategy.FIXED_TIME:
            self.controller = FixedTimeController(**kwargs)
        elif strategy == ControlStrategy.ADAPTIVE:
            self.controller = AdaptiveController(**kwargs)
        elif strategy == ControlStrategy.DQN:
            self.controller = DQNController(**kwargs)
        elif strategy == ControlStrategy.MAX_PRESSURE:
            self.controller = MaxPressureController(**kwargs)
        elif strategy == ControlStrategy.SOTL:
            self.controller = SOTLController(**kwargs)
        else:
            raise ValueError(f"Unsupported control strategy: {strategy}")
        
        print(f"Signal controller initialized: {self.controller.strategy_name}")
    
    def control_step(self, traffic_state: TrafficState) -> Tuple[int, float]:
        """Execute one control step"""
        return self.controller.decide_next_phase(traffic_state)
    
    def get_decision_history(self) -> List[Dict]:
        """Get decision history"""
        return self.controller.decision_history
    
    def export_performance_data(self, output_path: str):
        """Export performance data"""
        data = {
            'strategy': self.strategy.value,
            'controller_name': self.controller.strategy_name,
            'decision_history': self.controller.decision_history,
            'phases_config': {k: {
                'phase_id': v.phase_id,
                'name': v.name,
                'duration': v.duration,
                'min_duration': v.min_duration,
                'max_duration': v.max_duration
            } for k, v in self.controller.phases.items()}
        }
        
        # If it's a DQN controller, include additional information
        if isinstance(self.controller, DQNController):
            data['dqn_info'] = {
                'q_table_size': len(self.controller.q_table),
                'episode_rewards': self.controller.episode_rewards,
                'epsilon': self.controller.epsilon,
                'experience_buffer_size': len(self.controller.experience_buffer)
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Controller performance data exported: {output_path}")

def main():
    """Test signal controller"""
    print("Signal Controller Test")
    
    # Create test traffic state
    test_state = TrafficState(
        time=100.0,
        lane_vehicle_counts={
            'east_in_0': 8, 'east_in_1': 3,
            'west_in_0': 6, 'west_in_1': 2,
            'north_in_0': 4, 'north_in_1': 1,
            'south_in_0': 5, 'south_in_1': 2
        },
        lane_queue_lengths={
            'east_in_0': 3, 'east_in_1': 1,
            'west_in_0': 2, 'west_in_1': 0,
            'north_in_0': 1, 'north_in_1': 0,
            'south_in_0': 2, 'south_in_1': 1
        },
        lane_mean_speeds={
            'east_in_0': 8.5, 'east_in_1': 10.2,
            'west_in_0': 9.1, 'west_in_1': 12.0,
            'north_in_0': 11.5, 'north_in_1': 13.0,
            'south_in_0': 10.0, 'south_in_1': 11.8
        },
        detector_occupancy={
            'det_east_0': 0.4, 'det_east_1': 0.2,
            'det_west_0': 0.3, 'det_west_1': 0.1,
            'det_north_0': 0.2, 'det_north_1': 0.1,
            'det_south_0': 0.25, 'det_south_1': 0.15
        },
        current_phase=0,
        phase_remaining_time=15.0
    )
    
    # Test three control strategies
    strategies = [
        ControlStrategy.FIXED_TIME,
        ControlStrategy.ADAPTIVE,
        ControlStrategy.DQN
    ]
    
    for strategy in strategies:
        print(f"\n=== Testing {strategy.value.upper()} Control Strategy ===")
        
        controller = SignalController(strategy)
        
        # Execute several control decisions
        for step in range(3):
            test_state.time += 30.0
            next_phase, duration = controller.control_step(test_state)
            
            print(f"Step {step + 1}:")
            print(f"  Decision: Phase {next_phase} ({controller.controller.get_phase_name(next_phase)})")
            print(f"  Duration: {duration:.1f}s")
            
            # Update test state
            test_state.current_phase = next_phase
            test_state.phase_remaining_time = duration
        
        # Export decision history
        output_file = f"output/{strategy.value}_decisions.json"
        controller.export_performance_data(output_file)

if __name__ == "__main__":
    main() 