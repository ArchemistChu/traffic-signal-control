#!/usr/bin/env python3
"""
Simple status check for MARL evaluation.
"""

import psutil
from pathlib import Path

def main():
    print("Checking MARL evaluation status...")

    # Check running processes
    marl_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline'] or []
                if any('run_parallel_marl_eval.py' in str(arg) for arg in cmdline):
                    marl_procs.append(proc)
                elif any('evaluate_marl_osm.py' in str(arg) for arg in cmdline):
                    marl_procs.append(proc)
        except:
            continue

    print(f"Running MARL processes: {len(marl_procs)}")
    for proc in marl_procs:
        print(f"  PID {proc.pid}")

    # Check for result files
    output_dir = Path('output')
    marl_files = list(output_dir.glob('*MARL*.json'))
    print(f"MARL result files: {len(marl_files)}")
    for f in marl_files:
        print(f"  {f.name}")

    if marl_procs:
        print("\nEvaluation is still running - consider stopping it.")
        print("Use Task Manager to end Python processes, then try sequential evaluation.")
    else:
        print("\nNo running processes - safe to start new evaluation.")

if __name__ == "__main__":
    main()