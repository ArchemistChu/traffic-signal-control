#!/usr/bin/env python3
"""
Parallel Baseline Evaluation Runner

Runs multiple SUMO baseline evaluations in parallel using multiprocessing.
Each process runs a subset of episodes on different TraCI ports to avoid conflicts.

Example:
    python run_parallel_eval.py --dataset cologne --strategy SOTL --episodes 120 --duration 1200 --seed 42 --controlled-lights-ratio 0.2 --sumo-emissions-output --data-collection-interval 10 --out output/eval_cologne_SOTL_ep120.json --processes 4
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from typing import List, Dict, Any


def run_single_eval(args_dict: Dict[str, Any], process_id: int) -> Dict[str, Any]:
    """
    Run a single evaluation process with a subset of episodes.

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
    port = 8813 + process_id * 10  # 8813, 8823, 8833, etc.

    # Build command arguments
    cmd_args = [
        sys.executable,  # python executable
        'evaluate_baselines_osm.py',
        '--dataset', args_dict['dataset'],
        '--strategy', args_dict['strategy'],
        '--episodes', str(total_episodes),  # Keep total for metadata
        '--episode-start', str(start_episode),
        '--episode-end', str(end_episode),
        '--duration', str(args_dict['duration']),
        '--seed', str(args_dict['seed'] + start_episode - 1),  # Different seed per process
        '--run-id', str(process_id + 1),
        '--controlled-lights-ratio', str(args_dict['controlled_lights_ratio']),
        '--data-collection-interval', str(args_dict['data_collection_interval']),
        '--port', str(port),  # Add port argument
        '--out', temp_output
    ]

    # Add optional flags
    if args_dict['sumo_emissions_output']:
        cmd_args.append('--sumo-emissions-output')
    if args_dict['keep_emissions_xml']:
        cmd_args.append('--keep-emissions-xml')

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


def merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple parallel evaluations into a single JSON structure.

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
    parser = argparse.ArgumentParser(description="Run parallel SUMO baseline evaluations")
    parser.add_argument("--dataset", type=str, default="cologne", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["FIXED_TIME", "ADAPTIVE", "MAX_PRESSURE", "SOTL", "GA", "PRESSLIGHT"],
        help="Baseline strategy to evaluate",
    )
    parser.add_argument("--episodes", type=int, default=120, help="Total episodes across all processes")
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--controlled-lights-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sumo-emissions-output", action="store_true", help="Enable SUMO native emission-output and parse totals.")
    parser.add_argument("--keep-emissions-xml", action="store_true", help="Keep emission XML files (default: delete after parsing).")
    parser.add_argument(
        "--data-collection-interval",
        type=int,
        default=10,
        help="Collect vehicle data every N steps.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output eval JSON path")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes to use")
    args = parser.parse_args()

    # Validate arguments
    if args.processes < 1:
        raise ValueError("--processes must be >= 1")
    if args.episodes < args.processes:
        raise ValueError("--episodes must be >= --processes for meaningful parallelization")

    print("=" * 80)
    print(f"Parallel Baseline Evaluation")
    print(f"Strategy: {args.strategy} | Dataset: {args.dataset}")
    print(f"Total Episodes: {args.episodes} | Processes: {args.processes}")
    print(f"Duration: {args.duration}s | Controlled Lights: {args.controlled_lights_ratio:.2f}")
    print(f"Output: {args.out}")
    if args.sumo_emissions_output:
        print("Emissions: ENABLED")
    print("=" * 80)

    # Convert args to dict for multiprocessing
    args_dict = vars(args)

    # Run parallel evaluations
    with multiprocessing.Pool(processes=args.processes) as pool:
        # Create list of process IDs
        process_ids = list(range(args.processes))

        # Run evaluations in parallel
        results = pool.starmap(run_single_eval, [(args_dict, pid) for pid in process_ids])

    # Filter out empty results
    valid_results = [r for r in results if r]

    if not valid_results:
        print("Error: No valid results from any process")
        sys.exit(1)

    # Merge results
    print(f"Merging results from {len(valid_results)} processes...")
    merged_results = merge_results(valid_results)

    # Save merged results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2)

    print(f"✅ Parallel evaluation completed!")
    print(f"📊 Total episodes: {merged_results.get('episodes', 0)}")
    print(f"💾 Results saved to: {args.out}")


if __name__ == "__main__":
    main()