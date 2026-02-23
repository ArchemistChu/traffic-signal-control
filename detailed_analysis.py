import json
import numpy as np
from pathlib import Path

def analyze_decisions_and_behavior(filepath):
    """Analyze the decision-making behavior of different strategies"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        strategy = data.get('strategy', 'Unknown')
        print(f"\nAnalyzing {strategy} strategy:")

        # Sample a few episodes to see decision patterns
        results = data['results'][:5]  # First 5 episodes

        total_decisions = 0
        phase_switches = 0
        total_extensions = 0
        decisions_per_episode = []

        for i, result in enumerate(results):
            decisions = result.get('decisions', [])
            if decisions:
                decisions_per_episode.append(len(decisions))
                total_decisions += len(decisions)

                # Analyze decision patterns
                phase_sequence = []
                for decision in decisions[:20]:  # First 20 decisions
                    if 'phase' in decision:
                        phase_sequence.append(decision['phase'])

                # Count phase switches
                switches = sum(1 for j in range(1, len(phase_sequence)) if phase_sequence[j] != phase_sequence[j-1])
                phase_switches += switches

                # Count extensions (decisions that keep the same phase)
                extensions = sum(1 for j in range(1, len(phase_sequence)) if phase_sequence[j] == phase_sequence[j-1])
                total_extensions += extensions

                print(f"  Episode {i+1}: {len(decisions)} decisions, {switches} switches, {extensions} extensions")

        if decisions_per_episode:
            avg_decisions = np.mean(decisions_per_episode)
            avg_switches = phase_switches / len(results)
            avg_extensions = total_extensions / len(results)

            print(f"  Average decisions per episode: {avg_decisions:.1f}")
            print(f"  Average phase switches per episode: {avg_switches:.1f}")
            print(f"  Average extensions per episode: {avg_extensions:.1f}")

        # Look at timing patterns
        if results and results[0].get('decisions'):
            first_decisions = results[0]['decisions'][:10]
            print(f"  First 10 decisions timing for episode 1:")
            for j, d in enumerate(first_decisions):
                if 'time' in d and 'phase' in d:
                    print(f"    {j+1}: t={d['time']:.1f}s, phase={d['phase']}")

    except Exception as e:
        print(f"Error analyzing decisions in {filepath}: {e}")

def compare_traffic_conditions():
    """Compare the traffic conditions across different evaluation runs"""
    print("\n" + "="*80)
    print("TRAFFIC CONDITION ANALYSIS")
    print("="*80)

    files_to_check = [
        'output/eval_cologne_FIXED_TIME_ep120.json',
        'output/eval_cologne_MAX_PRESSURE_ep120.json',
        'output/eval_cologne_from_vancouver_regionaware_ep120_seeded_emissions.json'
    ]

    traffic_stats = {}

    for filepath in files_to_check:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            strategy = data.get('strategy', Path(filepath).stem)
            results = data['results'][:10]  # First 10 episodes

            avg_vehicles_total = []
            avg_departed = []
            avg_arrived = []

            for result in results:
                metrics = result.get('metrics', {})
                avg_vehicles_total.append(metrics.get('vehicles_total', 0))
                avg_departed.append(metrics.get('vehicles_departed', 0))
                avg_arrived.append(metrics.get('vehicles_arrived', 0))

            traffic_stats[strategy] = {
                'avg_vehicles_total': np.mean(avg_vehicles_total),
                'avg_departed': np.mean(avg_departed),
                'avg_arrived': np.mean(avg_arrived),
                'episodes': len(results)
            }

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    print("Traffic conditions across strategies:")
    for strategy, stats in traffic_stats.items():
        print(f"  {strategy}:")
        print(f"    Avg total vehicles: {stats['avg_vehicles_total']:.0f}")
        print(f"    Avg departed: {stats['avg_departed']:.0f}")
        print(f"    Avg arrived: {stats['avg_arrived']:.0f}")

def check_for_identical_results():
    """Check if results are suspiciously identical across strategies"""
    print("\n" + "="*80)
    print("IDENTICAL RESULTS CHECK")
    print("="*80)

    # Compare key metrics across strategies
    files = [
        ('FIXED_TIME', 'output/eval_cologne_FIXED_TIME_ep120.json'),
        ('MAX_PRESSURE', 'output/eval_cologne_MAX_PRESSURE_ep120.json'),
        ('ADAPTIVE', 'output/eval_cologne_ADAPTIVE_ep120.json'),
        ('SOTL', 'output/eval_cologne_SOTL_ep120.json')
    ]

    episode_data = {}

    for strategy, filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Get first episode metrics
            if data['results']:
                metrics = data['results'][0].get('metrics', {})
                key_metrics = {
                    'avg_waiting_time': metrics.get('avg_waiting_time', 0),
                    'avg_speed': metrics.get('avg_speed', 0),
                    'throughput_per_hour': metrics.get('throughput_per_hour', 0),
                    'congestion_index': metrics.get('congestion_index', 0)
                }
                episode_data[strategy] = key_metrics

        except Exception as e:
            print(f"Error reading {strategy}: {e}")

    # Check for identical values
    print("First episode comparison (should show differences if strategies work):")
    for strategy, metrics in episode_data.items():
        print(f"  {strategy}: wait={metrics['avg_waiting_time']:.2f}, speed={metrics['avg_speed']:.2f}, throughput={metrics['throughput_per_hour']:.0f}")

    # Check if any strategies have identical results
    strategies = list(episode_data.keys())
    identical_pairs = []
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            m1, m2 = episode_data[s1], episode_data[s2]
            if (abs(m1['avg_waiting_time'] - m2['avg_waiting_time']) < 0.01 and
                abs(m1['avg_speed'] - m2['avg_speed']) < 0.001):
                identical_pairs.append((s1, s2))

    if identical_pairs:
        print(f"\nWARNING: Found {len(identical_pairs)} strategy pairs with nearly identical results:")
        for pair in identical_pairs:
            print(f"  {pair[0]} vs {pair[1]}")
    else:
        print("\nAll strategies show different results (good).")

def main():
    # Analyze decision behavior
    analyze_decisions_and_behavior('output/eval_cologne_FIXED_TIME_ep120.json')
    analyze_decisions_and_behavior('output/eval_cologne_MAX_PRESSURE_ep120.json')
    analyze_decisions_and_behavior('output/eval_cologne_ADAPTIVE_ep120.json')

    # Compare traffic conditions
    compare_traffic_conditions()

    # Check for identical results
    check_for_identical_results()

if __name__ == "__main__":
    main()