import json
from pathlib import Path

def check_seeds_across_strategies():
    """Check if different strategies are using different SUMO seeds"""
    print("Checking SUMO seeds across strategies:")
    print("=" * 50)

    files = [
        ('FIXED_TIME', 'output/eval_cologne_FIXED_TIME_ep120.json'),
        ('MAX_PRESSURE', 'output/eval_cologne_MAX_PRESSURE_ep120.json'),
        ('ADAPTIVE', 'output/eval_cologne_ADAPTIVE_ep120.json'),
        ('MARL', 'output/eval_cologne_from_vancouver_regionaware_ep120_seeded_emissions.json')
    ]

    all_seeds = {}

    for strategy, filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            seeds = []
            for result in data['results'][:10]:  # Check first 10 episodes
                if 'sumo_seed' in result:
                    seeds.append(result['sumo_seed'])

            all_seeds[strategy] = seeds
            print(f"{strategy}: first 10 seeds = {seeds}")

        except Exception as e:
            print(f"Error reading {strategy}: {e}")

    # Check if seeds are identical across strategies
    print("\nChecking if strategies use identical seeds:")
    if len(all_seeds) >= 2:
        strategies = list(all_seeds.keys())
        for i in range(len(strategies)):
            for j in range(i+1, len(strategies)):
                s1, s2 = strategies[i], strategies[j]
                seeds1 = all_seeds[s1][:10]  # Compare first 10
                seeds2 = all_seeds[s2][:10]

                if seeds1 == seeds2:
                    print(f"WARNING: {s1} and {s2} use IDENTICAL seeds! This explains similar performance.")
                else:
                    print(f"GOOD: {s1} and {s2} use different seeds.")

def check_controlled_lights_ratio():
    """Check what percentage of lights are controlled"""
    print("\nChecking controlled lights ratio:")
    print("=" * 40)

    files = [
        'output/eval_cologne_FIXED_TIME_ep120.json',
        'output/eval_cologne_MAX_PRESSURE_ep120.json',
        'output/eval_cologne_ADAPTIVE_ep120.json'
    ]

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            ratio = data.get('controlled_lights_ratio', 'Not specified')
            strategy = data.get('strategy', Path(filepath).stem)
            print(f"{strategy}: controlled_lights_ratio = {ratio}")

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

def check_simulation_parameters():
    """Check key simulation parameters that could affect results"""
    print("\nChecking simulation parameters:")
    print("=" * 35)

    filepath = 'output/eval_cologne_FIXED_TIME_ep120.json'
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        params = {
            'duration': data.get('duration'),
            'decision_interval': data.get('decision_interval'),
            'episodes': data.get('episodes'),
            'dataset': data.get('dataset'),
            'controlled_lights_ratio': data.get('controlled_lights_ratio')
        }

        print("Key simulation parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error reading parameters: {e}")

def check_traffic_demand():
    """Check if traffic demand is consistent and realistic"""
    print("\nChecking traffic demand patterns:")
    print("=" * 35)

    filepath = 'output/eval_cologne_FIXED_TIME_ep120.json'
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = data['results'][:20]  # Check first 20 episodes

        vehicle_counts = []
        throughput_values = []
        waiting_times = []

        for result in results:
            metrics = result.get('metrics', {})
            vehicle_counts.append(metrics.get('vehicles_total', 0))
            throughput_values.append(metrics.get('throughput_per_hour', 0))
            waiting_times.append(metrics.get('avg_waiting_time', 0))

        print(f"Average vehicles per episode: {sum(vehicle_counts)/len(vehicle_counts):.0f}")
        print(f"Vehicle count range: {min(vehicle_counts)} - {max(vehicle_counts)}")
        print(f"Average throughput per hour: {sum(throughput_values)/len(throughput_values):.0f}")
        print(f"Throughput range: {min(throughput_values)} - {max(throughput_values)}")
        print(f"Average waiting time: {sum(waiting_times)/len(waiting_times):.1f}s")
        print(f"Waiting time range: {min(waiting_times):.1f} - {max(waiting_times):.1f}s")

        # Check if traffic is too light (which would make strategies perform similarly)
        avg_vehicles = sum(vehicle_counts)/len(vehicle_counts)
        if avg_vehicles < 1000:
            print("WARNING: Very low vehicle counts - traffic may be too light for meaningful differentiation")
        elif avg_vehicles < 3000:
            print("NOTE: Moderate vehicle counts - some differentiation possible")
        else:
            print("GOOD: High vehicle counts - should allow clear performance differences")

    except Exception as e:
        print(f"Error analyzing traffic demand: {e}")

def main():
    check_seeds_across_strategies()
    check_controlled_lights_ratio()
    check_simulation_parameters()
    check_traffic_demand()

if __name__ == "__main__":
    main()