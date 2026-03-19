#!/usr/bin/env python3
"""
Reinforcement Learning Module (RLAgent)
DQN algorithm implementation using PyTorch
State space: vehicle counts per lane (from SUMO), current signal state, remaining time
Action space: extend current green light or switch to next phase
Reward function: based on average waiting time, queue length, and (extended) fuel consumption/emission metrics
Supports both custom single intersection and TAPASCologne dataset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import json
import os
from typing import Dict, List, Tuple, Optional
import time

# Import configuration
try:
    from src.config import SimulationConfig
except ImportError:
    from config import SimulationConfig

# Define data structure for experience replay (supports n-step returns)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Standard Deep Q-Network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128, 64]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Weight initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward propagation"""
        return self.network(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling Deep Q-Network
    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128, 64]):
        super(DuelingDQNNetwork, self).__init__()

        feature_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            feature_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        self.feature = nn.Sequential(*feature_layers)

        self.value_head = nn.Linear(input_dim, 1)
        self.advantage_head = nn.Linear(input_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TrafficState:
    """Traffic state"""
    
    def __init__(self, sumo_state: Dict, lane_list: List[str] = None):
        """
        Initialize from SUMO state
        
        Args:
            sumo_state: SUMO simulation state dictionary
            lane_list: list of lane IDs (None for auto-detect from state)
        """
        self.time = sumo_state.get('time', 0.0)
        self.lane_vehicle_counts = sumo_state.get('lane_vehicle_counts', {})
        self.lane_queue_lengths = sumo_state.get('lane_queue_lengths', {})
        self.lane_mean_speeds = sumo_state.get('lane_mean_speeds', {})
        self.detector_occupancy = sumo_state.get('detector_occupancy', {})
        self.current_phase = sumo_state.get('traffic_light_phase', 0)
        self.phase_remaining_time = sumo_state.get('traffic_light_remaining_time', 0.0)
        self.extra_features = [
            float(x) for x in sumo_state.get('extra_features', [])
        ]
        
        # Get lane list from config or state
        if lane_list is None:
            lane_list = sorted(list(self.lane_vehicle_counts.keys()))
        self.lane_list = lane_list
    
    def to_vector(self) -> np.ndarray:
        """Convert to state vector with dynamic dimensionality"""
        state_vector = []
        
        # 1. Vehicle counts per lane (normalized to [0,1])
        for lane in self.lane_list:
            count = self.lane_vehicle_counts.get(lane, 0)
            normalized_count = min(1.0, count / 20.0)  # Assume max 20 vehicles
            state_vector.append(normalized_count)
        
        # 2. Queue lengths per lane (normalized to [0,1])
        for lane in self.lane_list:
            queue = self.lane_queue_lengths.get(lane, 0)
            normalized_queue = min(1.0, queue / 15.0)  # Assume max 15 queued vehicles
            state_vector.append(normalized_queue)
        
        # 3. Mean speeds per lane (normalized to [0,1])
        for lane in self.lane_list:
            speed = self.lane_mean_speeds.get(lane, 0.0)
            normalized_speed = min(1.0, speed / 20.0)  # Assume max speed 20m/s
            state_vector.append(normalized_speed)
        
        # 4. Detector occupancy (if available)
        detector_list = sorted(list(self.detector_occupancy.keys()))
        for det in detector_list:
            occupancy = self.detector_occupancy.get(det, 0.0)
            state_vector.append(min(1.0, occupancy))  # Already in [0,1] range
        
        max_phases = 16
        phase_encoding = [0.0] * max_phases
        if 0 <= self.current_phase < max_phases:
            phase_encoding[self.current_phase] = 1.0
        state_vector.extend(phase_encoding)
        
        # 6. Phase remaining time (normalized)
        normalized_time = min(1.0, max(0.0, self.phase_remaining_time / 60.0))  # Assume max 60 seconds
        state_vector.append(normalized_time)

        # 7. Optional extra features for backward-compatible state extensions.
        state_vector.extend(self.extra_features)
        
        return np.array(state_vector, dtype=np.float32)

class RLAgent:
    """DQN reinforcement learning agent"""
    
    def __init__(self, config: Dict = None, simulator=None):
        """
        Initialize RL agent
        
        Args:
            config: configuration parameter dictionary
            simulator: TrafficSimulator instance (for auto-calculating dimensions)
        """
        # Calculate state dimension based on simulator if provided
        if simulator:
            # Get a sample state to determine dimensions
            sample_state = simulator.get_current_state()
            lane_count = len(sample_state.get('lane_vehicle_counts', {}))
            detector_count = len(sample_state.get('detector_occupancy', {}))
            # state_dim = lane_count * 3 (counts, queues, speeds) + detector_count + 8 (phase one-hot) + 1 (time)
            calculated_state_dim = lane_count * 3 + detector_count + 8 + 1
            print(f"Auto-calculated state_dim: {calculated_state_dim} (lanes: {lane_count}, detectors: {detector_count})")
        else:
            calculated_state_dim = 37  # Default for custom dataset
        
        # Default configuration
        default_config = {
            'state_dim': calculated_state_dim,
            'action_dim': 8,  # 8 actions: keep current phase, switch to phase 0,1,2,3, extend 5s,10s,15s
            'lr': 1e-3,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.97,
            'batch_size': 64,
            'memory_size': 10000,
            'target_update_freq': 100,
            'hidden_dims': [256, 128, 64],
            'device': 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0 and torch.version.cuda is not None) else 'cpu',
            # Extensions
            'double_dqn': True,   # Use Double DQN update rule
            'dueling': True,      # Use dueling network architecture
            'n_step': 1           # n-step return (1 = standard DQN). Can be increased for multi-step learning.
        }
        
        # Merge with RL config from SimulationConfig
        rl_config = SimulationConfig.RL_CONFIG.copy()
        default_config.update(rl_config)
        
        self.config = {**default_config, **(config or {})}
        
        # Device setup
        self.device = torch.device(self.config['device'])
        print(f"Using device: {self.device}")
        print(f"State dimension: {self.config['state_dim']}, Action dimension: {self.config['action_dim']}")
        
        # Choose backbone: standard DQN or dueling DQN
        NetClass = DuelingDQNNetwork if self.config.get('dueling', False) else DQNNetwork

        # Network initialization
        self.q_network = NetClass(
            self.config['state_dim'],
            self.config['action_dim'],
            self.config['hidden_dims']
        ).to(self.device)
        
        self.target_network = NetClass(
            self.config['state_dim'],
            self.config['action_dim'],
            self.config['hidden_dims']
        ).to(self.device)
        
        # Synchronize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['lr'])
        
        # Experience replay
        self.memory = ReplayBuffer(self.config['memory_size'])

        # Buffer for building n-step returns
        self.n_step = max(1, int(self.config.get('n_step', 1)))
        self.n_step_buffer: deque = deque(maxlen=self.n_step)
        
        # RL parameters
        self.epsilon = self.config['epsilon_start']
        self.step_count = 0
        self.episode_count = 0
        
        # Statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': []
        }
        
        # Current episode information
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def select_action(self, state: TrafficState, training: bool = True) -> int:
        """
        Select action
        
        Args:
            state: current state
            training: whether in training mode
            
        Returns:
            int: selected action
        """
        state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)
        
        # ε-greedy policy
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.config['action_dim'] - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_vector)
                action = q_values.argmax().item()
        
        self.step_count += 1
        return action
    
    def store_experience(self, state: TrafficState, action: int, reward: float, 
                        next_state: TrafficState, done: bool):
        """
        Store experience
        
        Args:
            state: current state
            action: executed action
            reward: obtained reward
            next_state: next state
            done: whether episode ended
        """
        # Convert to vectors once
        state_vec = state.to_vector()
        next_state_vec = next_state.to_vector() if next_state else None

        # Push into n-step buffer
        self.n_step_buffer.append((state_vec, action, reward, next_state_vec, done))

        # If we have not yet collected n steps and episode not done, wait
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Build n-step transition from buffer
        # state_0, a_0, r_0,..., r_{n-1}, next_state_n
        R = 0.0
        discount = 1.0
        for (_, _, r, _, d) in self.n_step_buffer:
            R += discount * r
            if d:
                break
            discount *= self.config['gamma']

        first_state_vec, first_action, _, _, _ = self.n_step_buffer[0]
        last_next_state_vec = self.n_step_buffer[-1][3]
        last_done = self.n_step_buffer[-1][4]

        experience = Experience(
            first_state_vec,
            first_action,
            R,
            last_next_state_vec,
            last_done
        )
        self.memory.push(experience)

        # If episode ended, clear buffer so next episode starts fresh
        if done:
            self.n_step_buffer.clear()
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
    
    def train_step(self) -> Optional[float]:
        """
        Execute one training step

        Returns:
            float: loss value, returns None if training is not possible
        """
        if len(self.memory) < self.config['batch_size']:
            return None

        batch = self.memory.sample(self.config['batch_size'])

        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)

        has_next = torch.BoolTensor([e.next_state is not None for e in batch]).to(self.device)
        state_dim = states.shape[1]
        next_state_np = np.array([
            (e.next_state if e.next_state is not None else np.zeros(state_dim, dtype=np.float32))
            for e in batch
        ], dtype=np.float32)
        next_states = torch.FloatTensor(next_state_np).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            effective_nonterminal = (~dones) & has_next

            if self.config.get('double_dqn', False):
                online_next_q = self.q_network(next_states)
                best_actions = online_next_q.argmax(1)
                target_next_q = self.target_network(next_states).gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                target_next_q = self.target_network(next_states).max(1)[0]

            next_q_values = torch.zeros(self.config['batch_size']).to(self.device)
            next_q_values[effective_nonterminal] = target_next_q[effective_nonterminal]
            gamma_n = self.config['gamma'] ** self.n_step
            target_q_values = rewards + gamma_n * next_q_values
            target_q_values = target_q_values.clamp(-2.0, 2.0)

        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        self.optimizer.step()

        # Hard target update: copy online -> target every target_update_freq steps
        if self.step_count % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_history['losses'].append(loss.item())
        self.training_history['epsilon_values'].append(self.epsilon)

        return loss.item()
    
    def end_episode(self, avg_reward_override: float = None):
        """End current episode and decay epsilon.

        Args:
            avg_reward_override: if provided, store this as the episode reward
                instead of the raw accumulated sum. Useful in multi-agent settings
                where the raw sum scales with agent count and is hard to interpret.
        """
        reward_to_log = avg_reward_override if avg_reward_override is not None else self.current_episode_reward
        self.training_history['episode_rewards'].append(reward_to_log)
        self.training_history['episode_lengths'].append(self.current_episode_length)

        self.episode_count += 1
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )
    
    def action_to_control_command(self, action: int, current_state: TrafficState) -> Tuple[str, Dict]:
        """
        Convert action to control command
        
        Args:
            action: selected action
            current_state: current state
            
        Returns:
            Tuple[str, Dict]: (command type, command parameters)
        """
        if action == 0:
            # Keep current phase
            return "keep_phase", {"duration": 10.0}
        elif action in [1, 2, 3, 4]:
            # Switch to specified phase
            target_phase = action - 1
            return "switch_phase", {"phase": target_phase, "duration": 30.0}
        elif action in [5, 6, 7]:
            # Extend current phase
            extension_times = [5.0, 10.0, 15.0]
            extension = extension_times[action - 5]
            return "extend_phase", {"extension": extension}
        else:
            # Default keep
            return "keep_phase", {"duration": 10.0}
    
    def calculate_reward(self, prev_state: TrafficState, current_state: TrafficState, 
                        action: int) -> float:
        """
        Calculate reward function
        
        Args:
            prev_state: previous state
            current_state: current state
            action: executed action
            
        Returns:
            float: reward value
        """
        # 1. Waiting time / queue penalty (dataset‑agnostic)
        total_queue_prev = sum(prev_state.lane_queue_lengths.values())
        total_queue_curr = sum(current_state.lane_queue_lengths.values())
        # Longer queues are bad, reduction is good
        queue_reward = -0.1 * total_queue_curr
        queue_improvement = 0.05 * (total_queue_prev - total_queue_curr)
        
        # 2. Speed reward (prefer higher average speeds)
        avg_speed_prev = np.mean(list(prev_state.lane_mean_speeds.values() or [0.0]))
        avg_speed_curr = np.mean(list(current_state.lane_mean_speeds.values() or [0.0]))
        speed_reward = 0.02 * avg_speed_curr
        speed_improvement = 0.03 * (avg_speed_curr - avg_speed_prev)
        
        # 3. Directional balance reward
        #    Instead of hard‑coding lane IDs, group by name patterns so it works
        #    for both single‑intersection and 4‑intersection datasets.
        def _sum_dir(prefixes):
            total = 0
            for lane_id, count in current_state.lane_vehicle_counts.items():
                lane_lower = lane_id.lower()
                if any(p in lane_lower for p in prefixes):
                    total += count
            return total

        # East/West vs North/South demand
        ew_vehicles = _sum_dir(["east", "west"])
        ns_vehicles = _sum_dir(["north", "south"])
        balance_penalty = -0.02 * abs(ew_vehicles - ns_vehicles)
        
        # 4. Efficiency reward (reduce unnecessary phase switches)
        efficiency_reward = 0.0
        if action == 0:  # Keep phase
            efficiency_reward = 0.01  # Small positive reward
        
        # 5. Occupancy reward (if detectors exist)
        if current_state.detector_occupancy:
            avg_occupancy = np.mean(list(current_state.detector_occupancy.values()))
            if avg_occupancy < 0.3:  # Low occupancy gives reward
                occupancy_reward = 0.02 * (0.3 - avg_occupancy)
            else:
                occupancy_reward = -0.01 * (avg_occupancy - 0.3)
        else:
            occupancy_reward = 0.0
        
        # Total reward
        total_reward = (queue_reward + queue_improvement + 
                       speed_reward + speed_improvement + 
                       balance_penalty + efficiency_reward + 
                       occupancy_reward)
        
        return total_reward
    
    def save_model(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }, filepath)
        print(f"model is saved to : {filepath}")
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"model file doesn't exists: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.epsilon = checkpoint.get('epsilon', self.config['epsilon_start'])
        self.training_history = checkpoint.get('training_history', {
            'episode_rewards': [], 'episode_lengths': [], 'losses': [], 
            'epsilon_values': [], 'q_values': []
        })
        
        print(f"Model loaded: {filepath}")
        print(f"Training steps: {self.step_count}, Episode: {self.episode_count}, Epsilon: {self.epsilon:.3f}")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        if not self.training_history['episode_rewards']:
            return {}
        
        recent_rewards = self.training_history['episode_rewards'][-100:]  # Last 100 episodes
        
        stats = {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'current_epsilon': self.epsilon,
            'avg_reward_recent': np.mean(recent_rewards) if recent_rewards else 0,
            'max_reward': max(self.training_history['episode_rewards']),
            'min_reward': min(self.training_history['episode_rewards']),
            'avg_episode_length': np.mean(self.training_history['episode_lengths']) if self.training_history['episode_lengths'] else 0,
            'avg_loss_recent': np.mean(self.training_history['losses'][-1000:]) if self.training_history['losses'] else 0
        }
        
        return stats

def main():
    """Test RL agent"""
    print("DQN Reinforcement Learning Agent Test")
    
    # Create agent
    config = {
        'lr': 1e-3,
        'batch_size': 32,
        'memory_size': 5000
    }
    agent = RLAgent(config)
    
    # Simulate traffic state
    sumo_state = {
        'time': 100.0,
        'lane_vehicle_counts': {'east_in_0': 8, 'east_in_1': 3, 'west_in_0': 6, 'west_in_1': 2,
                               'north_in_0': 4, 'north_in_1': 1, 'south_in_0': 5, 'south_in_1': 2},
        'lane_queue_lengths': {'east_in_0': 3, 'east_in_1': 1, 'west_in_0': 2, 'west_in_1': 0,
                              'north_in_0': 1, 'north_in_1': 0, 'south_in_0': 2, 'south_in_1': 1},
        'lane_mean_speeds': {'east_in_0': 8.5, 'east_in_1': 10.2, 'west_in_0': 9.1, 'west_in_1': 12.0,
                            'north_in_0': 11.5, 'north_in_1': 13.0, 'south_in_0': 10.0, 'south_in_1': 11.8},
        'detector_occupancy': {'det_east_0': 0.4, 'det_east_1': 0.2, 'det_west_0': 0.3, 'det_west_1': 0.1,
                              'det_north_0': 0.2, 'det_north_1': 0.1, 'det_south_0': 0.25, 'det_south_1': 0.15},
        'traffic_light_phase': 0,
        'traffic_light_remaining_time': 15.0
    }
    
    state = TrafficState(sumo_state)
    
    # Test state vector conversion
    state_vector = state.to_vector()
    print(f"State vector dimension: {len(state_vector)}")
    print(f"State vector: {state_vector[:10]}...")  # Show only first 10 elements
    
    # Test action selection
    for i in range(5):
        action = agent.select_action(state, training=True)
        command_type, command_params = agent.action_to_control_command(action, state)
        print(f"Step {i+1}: action={action}, command={command_type}, params={command_params}")
    
    # Test experience storage and training
    next_state = TrafficState(sumo_state)  # Simplified, using same state
    reward = agent.calculate_reward(state, next_state, 0)
    print(f"Reward calculation: {reward:.3f}")
    
    # Store some experiences
    for _ in range(100):
        action = agent.select_action(state, training=True)
        reward = agent.calculate_reward(state, next_state, action)
        agent.store_experience(state, action, reward, next_state, False)
    
    # Execute training
    loss = agent.train_step()
    if loss:
        print(f"Training loss: {loss:.4f}")
    
    # Display statistics
    stats = agent.get_training_stats()
    print("\nTraining statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
       from torch.utils.tensorboard import SummaryWriter
       agent = RLAgent()
       sample_state = torch.zeros(1, agent.config["state_dim"])  # shape (1, 37)
       writer = SummaryWriter(log_dir="runs/dqn_graph")
       writer.add_graph(agent.q_network, sample_state)
       writer.close()
