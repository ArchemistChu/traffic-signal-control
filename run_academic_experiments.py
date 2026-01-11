#!/usr/bin/env python3
"""
Academic Experiment Runner
Runs multiple independent simulations for statistical analysis
"""

import sys
import os
sys.path.append(".")

from src.academic_evaluator import (
    AcademicEvaluator, ExperimentConfig, run_batch_experiments
)
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig


def create_simulator_and_run(strategy: str, duration: int):
    """
    Factory function to create and run a simulator
    
    Args:
        strategy: Control strategy name
        duration: Simulation duration in seconds
        
    Returns:
        Dictionary with simulation results
    """
    # Get current dataset
    dataset = SimulationConfig.DATASET
    
    # Create simulator (without GUI for batch runs)
    simulator = TrafficSimulator(use_gui=False, dataset=dataset)
    
    # Run simulation
    metrics = simulator.run_simulation(duration=duration, strategy=strategy)
    
    # Close simulation
    simulator.close_simulation()
    
    return metrics


def main():
    """Main function to run academic experiments"""
    
    # Configure experiment
    experiment_config = ExperimentConfig(
        experiment_name="traffic_signal_control_comparison",
        dataset="custom",  # or "4intersection", "tapas_cologne"
        strategies=["FIXED_TIME", "ADAPTIVE", "MAX_PRESSURE", "DQN"],
        num_runs=10,  # Number of independent runs for statistical significance
        duration=600,  # 10 minutes per run
        random_seed=42,  # For reproducibility
        traffic_flow_level="moderate",
        notes="Comparison of traffic signal control strategies with statistical analysis"
    )
    
    # Set dataset
    SimulationConfig.set_dataset(experiment_config.dataset)
    
    print("=" * 80)
    print("ACADEMIC EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Experiment: {experiment_config.experiment_name}")
    print(f"Dataset: {experiment_config.dataset}")
    print(f"Strategies: {experiment_config.strategies}")
    print(f"Number of runs: {experiment_config.num_runs}")
    print(f"Duration per run: {experiment_config.duration} seconds")
    print(f"Total runs: {len(experiment_config.strategies) * experiment_config.num_runs}")
    print("=" * 80)
    
    # Run batch experiments
    all_results = run_batch_experiments(experiment_config, create_simulator_and_run)
    
    # Analyze results
    evaluator = AcademicEvaluator()
    
    # Generate academic report
    print("\nGenerating academic report...")
    report = evaluator.generate_academic_report(experiment_config, all_results)
    print("\n" + "=" * 80)
    print("ACADEMIC REPORT GENERATED")
    print("=" * 80)
    print(report[:1000] + "...")  # Print first 1000 chars
    
    # Export results
    print("\nExporting results to JSON...")
    json_path = evaluator.export_results_json(experiment_config, all_results)
    print(f"Results exported to: {json_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

