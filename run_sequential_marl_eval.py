#!/usr/bin/env python3
"""
Sequential MARL Evaluation Runner

Runs multiple MARL evaluations sequentially with different seeds.
This avoids multiprocessing issues on Windows.
"""

import argparse
import os
import subprocess
import sys
import json
from typing import List, Dict, Any


def run_single_evaluation(dataset: str, model: str, episodes: int, duration: int,
                         seed: int, controlled_lights_ratio: float, port: int,
                         output_file: str) -> bool:
    """Run a single MARL evaluation"""

    cmd_args = [
        sys.executable,
        'evaluate_marl_osm.py',
        '--dataset', dataset,
        '--model', model,
        '--episodes', str(episodes),
        '--duration', str(duration),
        '--seed', str(seed),
        '--controlled-lights-ratio', str(controlled_lights_ratio),
        '--calc-metrics',
        '--force-cpu',
        '--port', str(port),
        '--out', output_file
    ]

    print(f"Running evaluation with seed {seed}, port {port}")
    print(f"Command: {' '.join(cmd_args)}")

    try:
        result = subprocess.run(
            cmd_args,
            cwd=os.path.dirname(__file__),
            capture_output=False,  # Let output show in real-time
            check=True
        )
        print(f"✅ Evaluation with seed {seed} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in evaluation with seed {seed}: {e}")
        return False


def merge_results(output_files: List[str], final_output: str) -> bool:
    """Merge multiple evaluation results into one file"""

    all_results = []

    for file_path in output_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'results' in data:
                    all_results.extend(data['results'])
                    # Use the first file's metadata as template
                    if not all_results and 'results' in locals():
                        template_data = data.copy()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False

    if not all_results:
        print("No results to merge")
        return False

    # Sort by episode number
    all_results.sort(key=lambda x: x.get('episode', 0))

    # Use template data but update results and episode count
    if 'template_data' in locals():
        template_data['results'] = all_results
        template_data['episodes'] = len(all_results)
        merged_data = template_data
    else:
        merged_data = {
            'dataset': 'cologne',
            'model': 'merged',
            'episodes': len(all_results),
            'results': all_results
        }

    # Save merged results
    try:
        with open(final_output, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"✅ Merged results saved to {final_output}")
        return True
    except Exception as e:
        print(f"Error saving merged results: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run sequential MARL evaluations")
    parser.add_argument("--dataset", type=str, default="cologne", choices=["cologne", "vancouver", "los_angeles"])
    parser.add_argument("--model", type=str, required=True, help="Path to trained MARL model (.pt)")
    parser.add_argument("--episodes", type=int, default=30, help="Total episodes across all evaluations")
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--controlled-lights-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=442, help="Base seed")
    parser.add_argument("--evaluations", type=int, default=4, help="Number of sequential evaluations to run")
    parser.add_argument("--out", type=str, required=True, help="Final output JSON path")

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise ValueError(f"Model file not found: {args.model}")

    print("=" * 80)
    print("Sequential MARL Evaluation")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} | Episodes per run: {args.episodes} | Total runs: {args.evaluations}")
    print(f"Duration: {args.duration}s | Controlled Lights: {args.controlled_lights_ratio:.2f}")
    print(f"Output: {args.out}")
    print("=" * 80)

    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Run evaluations sequentially
    base_port = 8813
    output_files = []

    for i in range(args.evaluations):
        seed = args.seed + (i * 100)  # Different seed for each run
        port = base_port + (i * 200)  # Different port for each run
        temp_output = f"{args.out}.run_{i+1}.json"

        success = run_single_evaluation(
            dataset=args.dataset,
            model=args.model,
            episodes=args.episodes,
            duration=args.duration,
            seed=seed,
            controlled_lights_ratio=args.controlled_lights_ratio,
            port=port,
            output_file=temp_output
        )

        if success:
            output_files.append(temp_output)
        else:
            print(f"Skipping merge due to failed evaluation {i+1}")
            return

    # Merge results
    print(f"\nMerging {len(output_files)} evaluation results...")
    if merge_results(output_files, args.out):
        # Clean up temporary files
        for temp_file in output_files:
            try:
                os.remove(temp_file)
                print(f"Cleaned up {temp_file}")
            except:
                pass

        print("✅ Sequential MARL evaluation completed!")
    else:
        print("❌ Failed to merge results")


if __name__ == "__main__":
    main()