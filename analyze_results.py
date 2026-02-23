import json
import os
from pathlib import Path
import numpy as np

def analyze_evaluation_file(filepath):
    """Analyze a single evaluation file and return summary statistics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = data['results']
        metrics = []

        for result in results:
            if 'metrics' in result:
                m = result['metrics']
                metrics.append({
                    'avg_waiting_time': m.get('avg_waiting_time', 0),
                    'avg_speed': m.get('avg_speed', 0),
                    'throughput_per_hour': m.get('throughput_per_hour', 0),
                    'avg_queue_length': m.get('avg_queue_length', 0),
                    'congestion_index': m.get('congestion_index', 0),
                    'total_co2': m.get('total_co2', 0),
                })

        if not metrics:
            return None

        # Calculate averages
        avg_metrics = {}
        for key in metrics[0].keys():
            values = [m[key] for m in metrics if m[key] > 0]  # Filter out zero values
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'min_{key}'] = np.min(values)
                avg_metrics[f'max_{key}'] = np.max(values)
                avg_metrics[f'std_{key}'] = np.std(values)
            else:
                avg_metrics[f'avg_{key}'] = 0
                avg_metrics[f'min_{key}'] = 0
                avg_metrics[f'max_{key}'] = 0
                avg_metrics[f'std_{key}'] = 0

        avg_metrics['num_episodes'] = len(metrics)
        avg_metrics['strategy'] = data.get('strategy', 'Unknown')

        return avg_metrics

    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def main():
    output_dir = Path('output')
    evaluation_files = [
        'eval_cologne_FIXED_TIME_ep120.json',
        'eval_cologne_MAX_PRESSURE_ep120.json',
        'eval_cologne_SOTL_ep120.json',
        'eval_cologne_ADAPTIVE_ep120.json',
        'eval_cologne_from_vancouver_regionaware_ep120_seeded_emissions.json'
    ]

    results = {}

    for filename in evaluation_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"Analyzing {filename}...")
            analysis = analyze_evaluation_file(filepath)
            if analysis:
                strategy_name = filename.replace('eval_cologne_', '').replace('_ep120.json', '').replace('_seeded_emissions', '')
                results[strategy_name] = analysis

    # Print comparison
    print("\n" + "="*80)
    print("TRAFFIC SIGNAL CONTROL EVALUATION COMPARISON")
    print("="*80)

    metrics_to_compare = [
        'avg_avg_waiting_time',
        'avg_avg_speed',
        'avg_throughput_per_hour',
        'avg_avg_queue_length',
        'avg_congestion_index'
    ]

    print(f"{'Strategy':<25} {'Avg Wait (s)':<12} {'Avg Speed':<10} {'Throughput':<12} {'Avg Queue':<10} {'Congestion':<10}")
    print("-" * 90)

    for strategy, data in results.items():
        print(f"{strategy:<25} {data['avg_avg_waiting_time']:<12.1f} {data['avg_avg_speed']:<10.2f} {data['avg_throughput_per_hour']:<12.0f} {data['avg_avg_queue_length']:<10.0f} {data['avg_congestion_index']:<10.3f}")

    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    # Find best and worst performers for each metric
    for metric in ['avg_waiting_time', 'avg_speed', 'throughput_per_hour', 'avg_queue_length']:
        print(f"\n{metric.upper()} ANALYSIS:")
        sorted_strategies = sorted(results.items(), key=lambda x: x[1][f'avg_{metric}'], reverse=(metric == 'avg_speed' or metric == 'throughput_per_hour'))

        for strategy, data in sorted_strategies:
            avg_val = data[f'avg_{metric}']
            std_val = data[f'std_{metric}']
            min_val = data[f'min_{metric}']
            max_val = data[f'max_{metric}']
            print(f"  {strategy:<30}: {avg_val:.2f} ± {std_val:.2f} (range: {min_val:.2f} - {max_val:.2f})")

    # Check if FIXED_TIME is performing suspiciously well
    fixed_time_data = results.get('FIXED_TIME', {})
    max_pressure_data = results.get('MAX_PRESSURE', {})

    if fixed_time_data and max_pressure_data:
        print("\n" + "="*80)
        print("FIXED_TIME vs MAX_PRESSURE COMPARISON")
        print("="*80)

        for metric in ['avg_waiting_time', 'avg_speed', 'throughput_per_hour', 'avg_queue_length']:
            ft_avg = fixed_time_data[f'avg_{metric}']
            mp_avg = max_pressure_data[f'avg_{metric}']
            diff = ((mp_avg - ft_avg) / ft_avg) * 100 if ft_avg != 0 else 0

            print(f"{metric}: FIXED_TIME={ft_avg:.2f}, MAX_PRESSURE={mp_avg:.2f} ({diff:+.1f}%)")

if __name__ == "__main__":
    main()