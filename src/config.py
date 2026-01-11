#!/usr/bin/env python3
"""
Configuration Module
Manages settings for switching between custom intersection and TAPASCologne dataset
"""

import os

class SimulationConfig:
    """Simulation configuration class"""
    
    # Dataset selection: 'custom', '4intersection', or 'tapas_cologne'
    DATASET = '4intersection'  # Options: 'custom' (single), '4intersection' (4-intersection grid), 'tapas_cologne'
    
    # Base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuration files for different datasets
    CONFIGS = {
        'custom': {
            'sumo_config': os.path.join(BASE_DIR, 'Dataset', 'Single Intersection', 'simulation.sumocfg'),
            'network_file': os.path.join(BASE_DIR, 'Dataset', 'Single Intersection', 'intersection.net.xml'),
            'route_file': os.path.join(BASE_DIR, 'Dataset', 'Single Intersection', 'routes.rou.xml'),
            'description': 'Single intersection simulation',
            # Predefined traffic lights (for custom dataset)
            'traffic_lights': ['intersection'],
            # Predefined lanes (for custom dataset) - 3 lanes per direction = 12 lanes total
            'lanes': [
                'east_in_0', 'east_in_1', 'east_in_2',
                'west_in_0', 'west_in_1', 'west_in_2',
                'north_in_0', 'north_in_1', 'north_in_2',
                'south_in_0', 'south_in_1', 'south_in_2'
            ],
            # Predefined detectors (for custom dataset) - 3 detectors per direction = 12 detectors total
            'detectors': [
                'det_east_0', 'det_east_1', 'det_east_2',
                'det_west_0', 'det_west_1', 'det_west_2',
                'det_north_0', 'det_north_1', 'det_north_2',
                'det_south_0', 'det_south_1', 'det_south_2'
            ],
            # Control mode: 'single' for single intersection
            'control_mode': 'single',
            # State dimension for RL (fixed for single intersection with 3 lanes per direction)
            # 12 lanes * 3 (count, queue, speed) + 12 detectors + 8 phase + 1 time = 57
            'state_dim': 57,
            'action_dim': 8
        },
        '4intersection': {
            'sumo_config': os.path.join(BASE_DIR, 'Dataset', '4 intersection', '4intersection.sumocfg'),
            'network_file': os.path.join(BASE_DIR, 'Dataset', '4 intersection', '4intersection.net.xml'),
            'route_file': os.path.join(BASE_DIR, 'Dataset', '4 intersection', '4intersection_routes.rou.xml'),
            'description': '4-intersection grid network simulation',
            # Predefined traffic lights (4 intersections)
            'traffic_lights': ['TL1', 'TL2', 'TL3', 'TL4'],
            # Auto-detect lanes and detectors for redesigned multi-lane network
            'lanes': None,
            'detectors': None,
            # Control mode: 'multi' for multiple intersections
            'control_mode': 'multi',
            # State dimension for RL (auto-calculated from simulator)
            'state_dim': None,
            'action_dim': 8
        },
        'tapas_cologne': {
            'sumo_config': os.path.join(BASE_DIR, 'TAPASCologne-0.32.0', 'cologne.sumocfg'),
            'network_file': os.path.join(BASE_DIR, 'TAPASCologne-0.32.0', 'cologne.net.xml'),
            'route_file': os.path.join(BASE_DIR, 'TAPASCologne-0.32.0', 'cologne.trips.xml'),
            'description': 'TAPASCologne - Real-world Cologne traffic network',
            # Traffic lights will be detected dynamically
            'traffic_lights': None,  # Auto-detect
            'lanes': None,  # Auto-detect
            'detectors': None,  # Auto-detect
            # Control mode: 'multi' for multiple intersections, 'selected' for specific intersections
            'control_mode': 'selected',
            # For 'selected' mode, specify which traffic lights to control
            'selected_traffic_lights': None,  # None means control all, or specify list like ['104255852', '1069892408']
            # Max number of traffic lights to control (to limit computational load)
            'max_controlled_lights': 10,
            # State dimension for RL (will be calculated dynamically)
            'state_dim': None,  # Auto-calculate based on selected intersections
            'action_dim': None  # Auto-calculate based on phases
        }
    }
    
    # Simulation parameters
    SIMULATION_DURATION = 3600  # Default simulation duration in seconds (1 hour)
    SIMULATION_STEP_LENGTH = 1.0  # Step length in seconds
    
    # RL training parameters
    RL_CONFIG = {
        'lr': 1e-4,  # Lower learning rate for complex network
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'memory_size': 50000,  # Larger memory for complex scenarios
        'target_update_freq': 200,
        'hidden_dims': [256, 256, 128],  # Larger network for complex state space
        'device': 'cuda'  # Will auto-fallback to cpu if cuda not available
    }
    
    # Output directories
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    @classmethod
    def get_current_config(cls):
        """Get current dataset configuration"""
        return cls.CONFIGS.get(cls.DATASET, cls.CONFIGS['custom'])
    
    @classmethod
    def set_dataset(cls, dataset_name: str):
        """
        Set active dataset
        
        Args:
            dataset_name: 'custom', '4intersection', or 'tapas_cologne'
        """
        if dataset_name not in cls.CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(cls.CONFIGS.keys())}")
        cls.DATASET = dataset_name
        print(f"Dataset switched to: {dataset_name}")
        print(f"Description: {cls.CONFIGS[dataset_name]['description']}")
    
    @classmethod
    def get_sumo_config_path(cls):
        """Get SUMO configuration file path"""
        config = cls.get_current_config()
        return config['sumo_config']
    
    @classmethod
    def get_network_file_path(cls):
        """Get network file path"""
        config = cls.get_current_config()
        return config['network_file']
    
    @classmethod
    def is_custom_dataset(cls):
        """Check if using custom dataset"""
        return cls.DATASET == 'custom'
    
    @classmethod
    def is_tapas_cologne(cls):
        """Check if using TAPASCologne dataset"""
        return cls.DATASET == 'tapas_cologne'
    
    @classmethod
    def create_output_dirs(cls):
        """Create output directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)


# Create output directories on import
SimulationConfig.create_output_dirs()
