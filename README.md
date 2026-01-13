# Project Overview

This project builds a complete intelligent traffic signal control simulation platform, aimed at providing scientific basis for the actual deployment of intelligent transportation systems through comparative research of different signal control strategies.

## Core Features
- **Realistic Simulation Environment** - Standard four-lane intersection modeling based on SUMO
- **Multiple Control Strategies** - Fixed-time control, adaptive control, DQN reinforcement learning, and MaxPressure control fully implemented
- **Vehicle Detection Simulation** - Simulates YOLO detection system with noise and delay
- **Comprehensive Performance Evaluation** - Multi-dimensional metrics for traffic efficiency and environmental impact
- **Intuitive Web Interface** - One-click startup, real-time observation, result comparison

## Core Architecture Layers
```
Application Layer (User Interface)
├── app_flask.py - Flask Web Interface
├── templates/ - HTML templates
└── start.py - One-click startup script

Control Decision Layer (Control Layer)  
├── SignalController - Traffic signal control strategy manager
│   ├── FixedTimeController - Fixed-time control
│   ├── AdaptiveController - Adaptive control  
│   ├── MaxPressureController - Max-pressure control
│   └── DQNController - DQN reinforcement learning control
└── RLAgent - Deep reinforcement learning agent (PyTorch)

Perception Layer (Perception Layer)
├── VehicleDetector - Vehicle detection simulation (YOLO simulation)
├── InductionLoops - Induction loop detectors
└── EmissionSensors - Emission monitoring sensors

Simulation Layer (Simulation Layer)
├── TrafficSimulator - Traffic simulation engine core
├── SUMO Backend - Microscopic traffic simulation platform
└── TraCI Interface - Python-SUMO real-time communication interface
```

## Project File Structure

```
traffic-signal-control/
├──  Core Configuration Files
│   ├── requirements.txt            # Python dependency package list
│   ├── README.md                   # Project documentation (this file)
│   └── LICENSE                     # License file
│
├──  Startup Scripts
│   ├── app_flask.py                # Flask main application interface
│   ├── start.py                    # Application startup script
│   └── templates/                  # HTML templates for Flask
│       ├── base.html               # Base template
│       ├── index.html              # Main page template
│       └── academic_experiments.html  # Academic experiments page
│
├──  Core Source Code (src/)
│   ├── traffic_simulator.py        # Traffic simulation engine core module
│   │                              # - SUMO connection management and TraCI communication
│   │                              # - Real-time data collection (vehicles, signals, emissions)
│   │                              # - Performance metrics calculation and export
│   │                              # - GUI/non-GUI mode simulation control
│   │
│   ├── signal_controller.py        # Signal control strategy implementation module  
│   │                              # - BaseController abstract base class
│   │                              # - FixedTimeController (fixed-time)
│   │                              # - AdaptiveController (adaptive control) 
│   │                              # - MaxPressureController (max-pressure control)
│   │                              # - DQNController (reinforcement learning control)
│   │                              # - Four-phase traffic light logic management
│   │
│   ├── rl_agent.py                 # DQN reinforcement learning agent module
│   │                              # - PyTorch deep neural network (256-128-64)
│   │                              # - Experience replay and target network mechanisms
│   │                              # - 37-dimensional state space, 8 discrete actions
│   │                              # - Comprehensive reward function (waiting time + efficiency + balance)
│   │
│   ├── academic_evaluator.py       # Academic evaluation module
│   │                              # - Statistical analysis and significance testing
│   │                              # - Confidence intervals and comprehensive metrics
│   │                              # - Academic report generation
│   │
│   ├── vehicle_detector.py         # Vehicle detection simulation module
│   │                              # - YOLO detection algorithm simulation
│   │                              # - Detection noise, delay, false detection simulation  
│   │                              # - Non-maximum suppression (NMS) algorithm
│   │                              # - Bounding box and confidence calculation
│   │
│   ├── config.py                   # Configuration module
│   │                              # - Dataset management (custom, OSM maps)
│   │                              # - Simulation configuration
│   │
│   └── test_simulation.py          # System environment test script
│                                  # - SUMO installation verification
│                                  # - Network file format checking
│                                  # - TraCI connection testing
│                                  # - Module functionality verification
│
├──  Dataset/
│   ├── Single Intersection/        # Custom single intersection simulation
│   │   ├── intersection.net.xml    # Network file
│   │   ├── routes.rou.xml          # Route file
│   │   └── simulation.sumocfg      # SUMO configuration file
│   │
│   ├── 4 intersection/             # 4-intersection grid network
│   │   └── 4intersection.sumocfg
│   │
│   └── LA Network/                 # LA network data
│
├──  OSM Maps/
│   ├── Cologne/                    # OSM Cologne map
│   │   └── osm.sumocfg
│   ├── Vancouver/                  # OSM Vancouver map
│   │   └── osm.sumocfg
│   └── Los Angeles/                # OSM Los Angeles map
│       └── osm.sumocfg
│
├──  Output Data Directory (output/) 
│   ├── vehicle_data_*.csv          # Vehicle trajectory data (position, speed, emissions)
│   ├── traffic_light_data_*.csv    # Traffic light state data (phase, time)
│   ├── performance_metrics_*.json  # Performance metrics report (JSON format)
│   └── simulation_summary_*.txt    # Simulation summary report (readable text)
│
└──  Runtime Generated Files
    ├── detectors.xml               # Detector data output
    ├── emissions.xml               # Vehicle emission data  
    ├── fcd_output.xml              # Floating car data (trajectory)
    ├── queue.xml                   # Queue length statistics
    └── summary.xml                 # SUMO simulation statistical summary
```

## Technology Stack

### Core Simulation Technology
- **SUMO v1.24.0** - Microscopic traffic simulation platform
  - Vehicle dynamics modeling (car-following, lane-changing, acceleration/deceleration)
  - Emission models (CO₂, NOx, PMx, fuel consumption)
  - Traffic light control and detector simulation
- **TraCI API** - Real-time communication interface between Python and SUMO
  - Real-time data acquisition (vehicle position, speed, waiting time)
  - Dynamic traffic light control (phase switching, duration adjustment)
  - Simulation step control and state management

### Artificial Intelligence Algorithms
- **PyTorch 2.0+** - Deep learning framework
  - DQN (Deep Q-Network) reinforcement learning algorithm
  - Experience replay mechanism
  - Target network for stable training
- **Deep Neural Network Architecture**
  - Input layer: 37-dimensional state space
  - Hidden layers: [256, 128, 64] neurons
  - Output layer: 8 discrete actions
  - Activation function: ReLU, Optimizer: Adam

### Data Processing and Analysis
- **NumPy 1.24+** - Numerical computation and matrix operations
- **Pandas 2.0+** - Data cleaning, aggregation, time series analysis
- **SciPy 1.11+** - Statistical analysis and numerical optimization
- **Scikit-learn 1.3+** - Data preprocessing and evaluation metrics

### Visualization and Interface
- **Flask 2.0+** - Web application framework
  - Server-side rendering with HTML templates
  - AJAX-based real-time updates
  - RESTful API endpoints
- **Plotly 5.15+** - Interactive chart library
  - Real-time data visualization (line charts, bar charts, scatter plots)
  - Responsive charts and zoom functionality
- **Bootstrap 5.3+** - Frontend CSS framework for responsive design

## Core Algorithm Implementation

### 1. Fixed-Time Control (FIXED_TIME)
```python
# Algorithm characteristics: Predefined timing, stable and reliable
Phase Sequence: [
    East-West Straight → 30 seconds,
    East-West Left Turn → 15 seconds, 
    North-South Straight → 30 seconds,
    North-South Left Turn → 15 seconds
]
Total Cycle: 120 seconds (including interval time)
```

### 2. Adaptive Control (ADAPTIVE) 
```python
# Algorithm logic: Dynamic adjustment based on real-time traffic flow
def adaptive_control_logic():
    1. Detect vehicle count on each lane (induction loops)
    2. Calculate lane weight = vehicle count × waiting time
    3. Select direction group with maximum weight
    4. Dynamically adjust green light duration:
       - Base time: 20-45 seconds
       - Extension condition: queue length > 5 vehicles 
       - Maximum limit: 60 seconds (ensures fairness)
    5. Low-traffic phase skip optimization
```

### 3. Max-Pressure Control (MAX_PRESSURE)
```python
# Algorithm logic: Serve phase with highest pressure (queue length)
def max_pressure_control_logic():
    1. Calculate pressure for each phase (incoming queue - outgoing queue)
    2. Select phase with maximum pressure
    3. Serve selected phase until pressure drops or time limit
```

### 4. DQN Reinforcement Learning Control (DQN)
```python
# State Space (37 dimensions)
state_space = [
    Lane vehicle counts[8],        # Current vehicle count on each lane
    Lane queue lengths[8],         # Waiting vehicle queue length
    Lane mean speeds[8],           # Average speed of vehicles in lane
    Detector occupancy[8],         # Induction loop occupancy
    Traffic light state[4],        # Current phase and remaining time
    System metrics[1]              # Global congestion index
]

# Action Space (8 discrete actions)
action_space = [
    0: Maintain current phase,
    1-4: Switch to specific phase (EW straight/left, NS straight/left),
    5-7: Extend current green light (5s/10s/15s)
]

# Reward Function Design
def calculate_reward(state, action, next_state):
    reward = (
        -0.1 * average_waiting_time +      # Reduce waiting time  
        +0.05 * average_speed +            # Increase travel speed
        -0.02 * speed_std_dev +            # Balance traffic flow in all directions
        +0.01 * traffic_efficiency +       # Improve system throughput
        -0.005 * detector_occupancy_variance  # Balance road usage
    )
    return reward
```

## Core Functionality Implementation

### 1. Real-Time Data Collection System
- **Vehicle Trajectory Tracking**: Real-time recording of position, speed, acceleration for all vehicles
- **Emission Data Monitoring**: Statistics of CO₂, NOx, PMx and other pollutant emissions  
- **Traffic Flow Analysis**: Lane occupancy, flow statistics, density calculation
- **Traffic Light State**: Phase switching history, green light utilization analysis

### 2. Vehicle Detection Simulation System  
```python
# YOLO Detection Performance Parameters
detection_params = {
    Detection Success Rate: 95%,        # Real vehicle detection probability
    False Detection Rate: 2%,          # False detection probability  
    Detection Delay: 50-200ms,         # Network transmission and processing delay
    Position Noise: ±0.5m,             # GPS error simulation
    Size Noise: ±10%,                  # Bounding box error
}

# NMS Non-Maximum Suppression
def non_maximum_suppression(detections, iou_threshold=0.5):
    # Remove duplicate detections, improve detection quality
```

### 3. Performance Evaluation System
```python
# Traffic Efficiency Metrics  
traffic_metrics = {
    'avg_waiting_time': 'Average waiting time (seconds)',
    'avg_speed': 'Average travel speed (m/s)', 
    'throughput_per_hour': 'System throughput (vehicles/hour)',
    'congestion_index': 'Congestion index (0-1)',
    'queue_length_variance': 'Queue length variance'
}

# Environmental Impact Metrics
environmental_metrics = {
    'total_co2': 'Total CO₂ emissions (mg)',
    'total_nox': 'Total NOx emissions (mg)', 
    'total_pmx': 'Total PMx emissions (mg)',
    'total_fuel': 'Total fuel consumption (ml)',
    'emission_per_km': 'Emission per unit distance (mg/km)'
}
```

### 4. Visualization and Analysis Interface
- **One-Click Startup**: Strategy selection → Click start → Watch simulation → View results
- **Real-Time Monitoring**: SUMO GUI window displays vehicle movement and signal changes in real-time  
- **Performance Comparison**: Automatic comparative analysis of multi-strategy run results
- **History Data Management**: Save multiple simulation results, support clear and reset
- **Academic Experiments**: Statistical experiments with multiple independent runs for research purposes

## Getting Started

### Prerequisites
- Python 3.8 or higher
- SUMO (Simulation of Urban MObility) installed and configured
- All dependencies from `requirements.txt`

### Installation
1. Install SUMO following the official installation guide
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify SUMO installation:
   ```bash
   python src/test_simulation.py
   ```

### Running the Application
1. Start the Flask web interface:
   ```bash
   python start.py
   ```
2. Open your browser and navigate to: `http://localhost:5000`
3. Select a dataset (Custom, Cologne, Vancouver, or Los Angeles)
4. Choose a control strategy
5. Click "Start Simulation" to begin

### Features
- **Multiple Datasets**: Custom single intersection, OSM-based real-world maps (Cologne, Vancouver, Los Angeles)
- **Four Control Strategies**: Fixed-time, Adaptive, Max-Pressure, and DQN reinforcement learning
- **Interactive Web Interface**: Easy-to-use Flask-based interface with real-time updates
- **Academic Experiments**: Statistical analysis with multiple runs, confidence intervals, and significance testing
- **Comprehensive Results**: Detailed performance metrics, comparison charts, and exported data files

## Documentation
- See `FLASK_README.md` for Flask interface details
- See `OSM_MAPS_GUIDE.md` for OSM map usage guide
- See `ACADEMIC_FEATURES.md` for academic experiment features

## License
See LICENSE file for details.
