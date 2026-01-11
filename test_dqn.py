import sys
sys.path.append('src')
from traffic_simulator import TrafficSimulator

print('Testing DQN controller...')
simulator = TrafficSimulator(dataset='custom')
metrics = simulator.run_simulation(duration=60, strategy='DQN')
print(f'Average waiting time: {metrics.get("avg_waiting_time", "N/A")}')
print(f'Max waiting time: {metrics.get("max_waiting_time", "N/A")}')
print(f'Total vehicles: {metrics.get("throughput", "N/A")}')
simulator.close_simulation()
