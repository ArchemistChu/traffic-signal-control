import sys
sys.path.append('src')
from traffic_simulator import TrafficSimulator
from config import SimulationConfig

# Create simulator with custom dataset
simulator = TrafficSimulator(dataset='custom')

# Run simulation to check waiting times
print('Testing waiting time collection...')
print('Running simulation for 120 seconds...')
metrics = simulator.run_simulation(duration=120, strategy=None)  # Skip signal controller

print(f'Average waiting time: {metrics.get("avg_waiting_time", "N/A")}')
print(f'Max waiting time: {metrics.get("max_waiting_time", "N/A")}')
print(f'Total vehicles: {metrics.get("throughput", "N/A")}')

# Check raw vehicle data
if simulator.vehicle_data:
    import pandas as pd
    df = pd.DataFrame(simulator.vehicle_data)
    print(f'Raw data shape: {df.shape}')
    if 'waiting_time' in df.columns:
        print('Waiting time stats:')
        print(df['waiting_time'].describe())

simulator.close_simulation()
