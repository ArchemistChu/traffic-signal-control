#!/usr/bin/env python3
"""
Parallel MARL Evaluation Runner

Runs multiple MARL evaluations in parallel using multiprocessing.
Each process runs a subset of episodes on different TraCI ports to avoid conflicts.

Example:
    python run_parallel_marl_eval.py --dataset cologne --model models/marl_vancouver_shared_dqn_regionaware.pt --episodes 120 --duration 1200 --seed 442 --controlled-lights-ratio 0.5 --calc-metrics --force-cpu --out output/eval_cologne_MARL_ep120_seed442.json --processes 8
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from typing import List, Dict, Any


def run_single_marl_eval(args_dict: Dict[str, Any], process_id: int) -> Dict[str, Any]:
    """
    Run a single MARL evaluation process with a subset of episodes.

    Args:
        args_dict: Dictionary of command-line arguments
        process_id: Unique ID for this process (used for port offset)

    Returns:
        Dict containing the evaluation results
    """
    # Calculate episodes for this process
    total_episodes = args_dict['episodes']
    num_processes = args_dict['processes']
    episodes_per_process = total_episodes // num_processes

    # Distribute remaining episodes to first few processes
    extra_episodes = total_episodes % num_processes
    if process_id < extra_episodes:
        episodes_this_process = episodes_per_process + 1
        start_episode = process_id * (episodes_per_process + 1) + 1
    else:
        episodes_this_process = episodes_per_process
        start_episode = extra_episodes * (episodes_per_process + 1) + (process_id - extra_episodes) * episodes_per_process + 1

    end_episode = start_episode + episodes_this_process - 1

    # Create temporary output file for this process
    temp_output = f"{args_dict['out']}.part_{process_id}.json"

    # Use different port for each process to avoid conflicts
    port = 8813 + process_id * 100  # 8813, 8913, 9013, etc. (larger offset)

    # Build command arguments
    cmd_args = [
        sys.executable,  # python executable
        'evaluate_marl_osm.py',
        '--dataset', args_dict['dataset'],
        '--model', args_dict['model'],
        '--episodes', str(episodes_this_process),  # Episodes for this process only
        '--duration', str(args_dict['duration']),
        '--decision-interval', str(args_dict['decision_interval']),
        '--controlled-lights-ratio', str(args_dict['controlled_lights_ratio']),
        '--lanes-per-tl', str(args_dict['lanes_per_tl']),
        '--regional-reward-weight', str(args_dict.get('regional_reward_weight', 0.0)),
        '--region-grid-size', str(args_dict.get('region_grid_size', 500.0)),
        '--data-collection-interval', str(args_dict['data_collection_interval']),
        '--seed', str(args_dict['seed'] + start_episode - 1),  # Different seed per process
        '--run-id', str(process_id + 1),
        '--port', str(port),  # Add port argument
        '--out', temp_output
    ]

    # Add optional flags
    if args_dict.get('sumo_emissions_output'):
        cmd_args.append('--sumo-emissions-output')
    if args_dict.get('keep_emissions_xml'):
        cmd_args.append('--keep-emissions-xml')
    if args_dict.get('collect_emissions'):
        cmd_args.append('--collect-emissions')
    if args_dict.get('calc_metrics'):
        cmd_args.append('--calc-metrics')
    # Always force CPU since CUDA has compatibility issues
    cmd_args.append('--force-cpu')

    print(f"Process {process_id}: Running episodes {start_episode}-{end_episode} "
          f"(seed={args_dict['seed'] + start_episode - 1})")

    # Run the evaluation
    try:
        result = subprocess.run(
            cmd_args,
            cwd=os.path.dirname(__file__),
            capture_output=False,  # Let output show in real-time
            check=True
        )

        # Load the results from the temporary file
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                data = json.load(f)
            return data
        else:
            print(f"Warning: Output file {temp_output} not found for process {process_id}")
            return {}

    except subprocess.CalledProcessError as e:
        print(f"Error in process {process_id}: {e}")
        return {}
    finally:
        # Clean up temporary file
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass


def merge_marl_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple parallel MARL evaluations into a single JSON structure.

    Args:
        results_list: List of result dictionaries from each process

    Returns:
        Merged results dictionary
    """
    if not results_list:
        return {}

    # Use the first result as the base template
    merged = results_list[0].copy()
    merged_results = []

    # Collect all episode results
    for result in results_list:
        if 'results' in result:
            merged_results.extend(result['results'])

    # Sort by episode number
    merged_results.sort(key=lambda x: x.get('episode', 0))

    # Update the merged results
    merged['results'] = merged_results
    merged['episodes'] = len(merged_results)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Run parallel MARL evaluations")
    parser.add_argument("--dataset", type=str, default="cologne", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument("--model", type=str, required=True, help="Path to trained MARL model (.pt)")
    parser.add_argument("--episodes", type=int, default=120, help="Total episodes across all processes")
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--controlled-lights-ratio", type=float, default=0.5)
    parser.add_argument("--lanes-per-tl", type=int, default=8)
    parser.add_argument("--regional-reward-weight", type=float, default=0.0)
    parser.add_argument("--region-grid-size", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--sumo-emissions-output", action="store_true", help="Enable SUMO native emission-output")
    parser.add_argument("--keep-emissions-xml", action="store_true", help="Keep emission XML files")
    parser.add_argument("--collect-emissions", action="store_true", help="Enable TraCI emission collection")
    parser.add_argument("--calc-metrics", action="store_true", help="Compute performance metrics")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--data-collection-interval",
        type=int,
        default=10,
        help="Collect detailed TraCI vehicle snapshots every N simulation steps",
    )
    parser.add_argument("--out", type=str, required=True, help="Output eval JSON path")
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel processes to use")
    args = parser.parse_args()

    # Validate arguments
    if args.processes < 1:
        raise ValueError("--processes must be >= 1")
    if args.episodes < args.processes:
        raise ValueError("--episodes must be >= --processes for meaningful parallelization")

    if not os.path.isfile(args.model):
        raise ValueError(f"Model file not found: {args.model}")

    print("=" * 85)
    print(f"Parallel MARL Evaluation")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} | Total Episodes: {args.episodes} | Processes: {args.processes}")
    print(f"Duration: {args.duration}s | Controlled Lights: {args.controlled_lights_ratio:.2f}")
    print(f"Output: {args.out}")
    if args.force_cpu:
        print("Device: CPU (forced)")
    print("=" * 85)

    # Convert args to dict for multiprocessing
    args_dict = vars(args)

    # Run parallel evaluations
    with multiprocessing.Pool(processes=args.processes) as pool:
        # Create list of process IDs
        process_ids = list(range(args.processes))

        # Run evaluations in parallel
        results = pool.starmap(run_single_marl_eval, [(args_dict, pid) for pid in process_ids])

    # Filter out empty results
    valid_results = [r for r in results if r]

    if not valid_results:
        print("Error: No valid results from any process")
        sys.exit(1)

    # Merge results
    print(f"Merging results from {len(valid_results)} processes...")
    merged_results = merge_marl_results(valid_results)

    # Save merged results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2)

    print("✅ Parallel MARL evaluation completed!")
    print(f"📊 Total episodes: {merged_results.get('episodes', 0)}")
    print(f"💾 Results saved to: {args.out}")


if __name__ == "__main__":
    main()