import sys
sys.path.append('src')
from traffic_simulator import TrafficSimulator

def test_strategy(strategy_name, duration=60):
    print(f'\n=== Testing {strategy_name} ===')
    try:
        simulator = TrafficSimulator(dataset='custom')
        print(f'Starting simulation with {strategy_name}...')

        metrics = simulator.run_simulation(duration=duration, strategy=strategy_name)

        print('Results:')
        print(f'  Simulation time: {simulator.simulation_time:.1f}s')
        print(f'  Total steps: {simulator.step_count}')
        print(f'  Average waiting time: {metrics.get("avg_waiting_time", "N/A")}')
        print(f'  Max waiting time: {metrics.get("max_waiting_time", "N/A")}')
        print(f'  Total vehicles: {metrics.get("throughput", "N/A")}')
        print(f'  Data records: {len(simulator.vehicle_data)}')

        # Check if there are any vehicles with waiting time > 0
        if hasattr(simulator, 'vehicle_data') and simulator.vehicle_data:
            import pandas as pd
            df = pd.DataFrame(simulator.vehicle_data)
            if 'waiting_time' in df.columns:
                waiting_vehicles = df[df['waiting_time'] > 0]
                print(f'  Vehicles with waiting time > 0: {len(waiting_vehicles)}')
                if len(waiting_vehicles) > 0:
                    print(f'  Max waiting time in data: {waiting_vehicles["waiting_time"].max()}')

        simulator.close_simulation()
        return True

    except Exception as e:
        print(f'Error with {strategy_name}: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    strategies = ['FIXED_TIME', 'ADAPTIVE', 'DQN']

    for strategy in strategies:
        success = test_strategy(strategy, duration=60)
        if not success:
            print(f'Skipping remaining strategies due to error with {strategy}')
            break
