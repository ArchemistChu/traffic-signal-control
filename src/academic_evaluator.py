#!/usr/bin/env python3
"""
Academic Evaluation Module
Provides statistical analysis, reproducibility, and comprehensive evaluation metrics
for academic research and publication
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments"""
    experiment_name: str
    dataset: str
    strategies: List[str]
    num_runs: int = 10  # Number of independent runs for statistical significance
    duration: int = 600  # Simulation duration in seconds
    random_seed: Optional[int] = None  # For reproducibility
    traffic_flow_level: str = "moderate"  # low, moderate, high
    notes: str = ""


@dataclass
class StatisticalResults:
    """Statistical analysis results"""
    mean: float
    std: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    ci_95_lower: float  # 95% confidence interval lower bound
    ci_95_upper: float  # 95% confidence interval upper bound
    min_value: float
    max_value: float
    n_samples: int


class AcademicEvaluator:
    """
    Academic evaluation module for traffic signal control research
    Provides statistical analysis, significance testing, and comprehensive metrics
    """
    
    def __init__(self, output_dir: str = "output/academic"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.experiment_results = []
    
    def calculate_statistics(self, data: List[float]) -> StatisticalResults:
        """
        Calculate comprehensive statistical measures
        
        Args:
            data: List of numerical values
            
        Returns:
            StatisticalResults object with all statistics
        """
        if not data or len(data) == 0:
            return StatisticalResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        data_array = np.array(data)
        n = len(data_array)
        
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)  # Sample standard deviation
        median = np.median(data_array)
        q25 = np.percentile(data_array, 25)
        q75 = np.percentile(data_array, 75)
        
        # 95% confidence interval using t-distribution
        if n > 1:
            sem = stats.sem(data_array)  # Standard error of the mean
            ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            ci_95_lower, ci_95_upper = ci_95
        else:
            ci_95_lower = ci_95_upper = mean
        
        return StatisticalResults(
            mean=float(mean),
            std=float(std),
            median=float(median),
            q25=float(q25),
            q75=float(q75),
            ci_95_lower=float(ci_95_lower),
            ci_95_upper=float(ci_95_upper),
            min_value=float(np.min(data_array)),
            max_value=float(np.max(data_array)),
            n_samples=n
        )
    
    def t_test(self, data1: List[float], data2: List[float]) -> Dict:
        """
        Perform independent samples t-test
        
        Args:
            data1: First group of data
            data2: Second group of data
            
        Returns:
            Dictionary with t-statistic, p-value, and interpretation
        """
        if len(data1) < 2 or len(data2) < 2:
            return {
                't_statistic': None,
                'p_value': None,
                'significant': False,
                'interpretation': 'Insufficient data for t-test'
            }
        
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        
        significant = p_value < 0.05
        
        interpretation = f"{'Significant' if significant else 'Not significant'} difference "
        interpretation += f"(p={p_value:.4f}, Cohen's d={cohens_d:.3f})"
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': significant,
            'cohens_d': float(cohens_d),
            'interpretation': interpretation
        }
    
    def mann_whitney_test(self, data1: List[float], data2: List[float]) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test)
        
        Args:
            data1: First group of data
            data2: Second group of data
            
        Returns:
            Dictionary with U-statistic, p-value, and interpretation
        """
        if len(data1) < 2 or len(data2) < 2:
            return {
                'u_statistic': None,
                'p_value': None,
                'significant': False,
                'interpretation': 'Insufficient data for Mann-Whitney test'
            }
        
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        significant = p_value < 0.05
        
        interpretation = f"{'Significant' if significant else 'Not significant'} difference "
        interpretation += f"(p={p_value:.4f})"
        
        return {
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': significant,
            'interpretation': interpretation
        }
    
    def analyze_multiple_runs(self, results: List[Dict], metric_name: str) -> Dict:
        """
        Analyze results from multiple independent runs
        
        Args:
            results: List of result dictionaries from multiple runs
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary with statistical analysis
        """
        values = [r.get(metric_name, 0) for r in results if metric_name in r]
        
        if not values:
            return {'error': f'Metric {metric_name} not found in results'}
        
        stats_result = self.calculate_statistics(values)
        
        return {
            'metric': metric_name,
            'statistics': asdict(stats_result),
            'raw_values': values
        }
    
    def compare_strategies(self, strategy_results: Dict[str, List[Dict]], 
                          metric_name: str) -> Dict:
        """
        Compare multiple strategies on a given metric
        
        Args:
            strategy_results: Dictionary mapping strategy names to lists of results
            metric_name: Name of the metric to compare
            
        Returns:
            Dictionary with comparison results including statistical tests
        """
        comparison = {
            'metric': metric_name,
            'strategies': {},
            'pairwise_comparisons': {}
        }
        
        # Extract values for each strategy
        strategy_values = {}
        for strategy, results in strategy_results.items():
            values = [r.get(metric_name, 0) for r in results if metric_name in r]
            if values:
                strategy_values[strategy] = values
                stats_result = self.calculate_statistics(values)
                comparison['strategies'][strategy] = asdict(stats_result)
        
        # Pairwise comparisons
        strategy_names = list(strategy_values.keys())
        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i+1:]:
                comparison_key = f"{strategy1}_vs_{strategy2}"
                
                # T-test
                t_test_result = self.t_test(
                    strategy_values[strategy1],
                    strategy_values[strategy2]
                )
                
                # Mann-Whitney test (non-parametric)
                mw_test_result = self.mann_whitney_test(
                    strategy_values[strategy1],
                    strategy_values[strategy2]
                )
                
                comparison['pairwise_comparisons'][comparison_key] = {
                    't_test': t_test_result,
                    'mann_whitney': mw_test_result
                }
        
        return comparison
    
    def calculate_comprehensive_metrics(self, single_result: Dict) -> Dict:
        """
        Calculate additional academic metrics from simulation results
        
        Args:
            single_result: Single simulation result dictionary
            
        Returns:
            Dictionary with additional academic metrics
        """
        metrics = {}
        
        # Efficiency metrics
        avg_wait = single_result.get('avg_waiting_time', 0)
        throughput = single_result.get('throughput_per_hour', 0)
        
        # Travel Time Index (TTI) - ratio of actual travel time to free-flow travel time
        # Higher TTI indicates more congestion
        avg_speed = single_result.get('avg_speed', 0)
        free_flow_speed = 50.0  # Assume 50 m/s as free-flow speed
        if avg_speed > 0:
            metrics['travel_time_index'] = free_flow_speed / avg_speed
        else:
            metrics['travel_time_index'] = float('inf')
        
        # Delay per vehicle (seconds)
        metrics['delay_per_vehicle'] = avg_wait
        
        # System efficiency score (combining waiting time and throughput)
        # Lower waiting time and higher throughput = better score
        if throughput > 0:
            metrics['efficiency_score'] = throughput / (1 + avg_wait)
        else:
            metrics['efficiency_score'] = 0
        
        # Environmental impact per vehicle
        total_co2 = single_result.get('total_co2', 0)
        total_vehicles = single_result.get('throughput', 0)
        if total_vehicles > 0:
            metrics['co2_per_vehicle'] = total_co2 / total_vehicles
        else:
            metrics['co2_per_vehicle'] = 0
        
        # Fairness metric - coefficient of variation of waiting times
        # Lower CV = more fair (less variation in waiting times)
        # This would require per-vehicle data, approximated here
        
        return metrics
    
    def generate_academic_report(self, experiment_config: ExperimentConfig,
                                 all_results: Dict[str, List[Dict]]) -> str:
        """
        Generate an academic-style report with statistical analysis
        
        Args:
            experiment_config: Experiment configuration
            all_results: Dictionary mapping strategy names to lists of results
            
        Returns:
            Markdown formatted report string
        """
        report = []
        report.append("# Academic Evaluation Report")
        report.append(f"\n**Experiment:** {experiment_config.experiment_name}")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Dataset:** {experiment_config.dataset}")
        report.append(f"**Number of Runs:** {experiment_config.num_runs}")
        report.append(f"**Simulation Duration:** {experiment_config.duration} seconds")
        if experiment_config.random_seed:
            report.append(f"**Random Seed:** {experiment_config.random_seed}")
        report.append(f"\n{experiment_config.notes}\n")
        
        # Key metrics to analyze
        key_metrics = [
            'avg_waiting_time',
            'throughput_per_hour',
            'avg_speed',
            'total_co2',
            'congestion_index'
        ]
        
        report.append("## Statistical Summary\n")
        
        for metric in key_metrics:
            if any(metric in r for results in all_results.values() for r in results):
                comparison = self.compare_strategies(all_results, metric)
                
                report.append(f"### {metric.replace('_', ' ').title()}\n")
                report.append("| Strategy | Mean | Std | Median | 95% CI Lower | 95% CI Upper |")
                report.append("|----------|------|-----|--------|--------------|--------------|")
                
                for strategy, stats_dict in comparison['strategies'].items():
                    stats = stats_dict
                    report.append(
                        f"| {strategy} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                        f"{stats['median']:.2f} | {stats['ci_95_lower']:.2f} | "
                        f"{stats['ci_95_upper']:.2f} |"
                    )
                
                # Pairwise comparisons
                report.append("\n#### Statistical Significance Tests\n")
                for comparison_key, tests in comparison['pairwise_comparisons'].items():
                    report.append(f"**{comparison_key}:**")
                    report.append(f"- T-test: {tests['t_test']['interpretation']}")
                    report.append(f"- Mann-Whitney: {tests['mann_whitney']['interpretation']}\n")
        
        # Save report
        report_path = os.path.join(
            self.output_dir,
            f"{experiment_config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    def export_results_json(self, experiment_config: ExperimentConfig,
                           all_results: Dict[str, List[Dict]], filename: str = None):
        """
        Export results in JSON format for further analysis
        
        Args:
            experiment_config: Experiment configuration
            all_results: Dictionary mapping strategy names to lists of results
            filename: Optional filename, auto-generated if None
        """
        if filename is None:
            filename = f"{experiment_config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'experiment_config': asdict(experiment_config),
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath


def run_batch_experiments(experiment_config: ExperimentConfig,
                         simulator_factory) -> Dict[str, List[Dict]]:
    """
    Run multiple independent experiments for statistical analysis
    
    Args:
        experiment_config: Experiment configuration
        simulator_factory: Function that creates and runs a simulator
        
    Returns:
        Dictionary mapping strategy names to lists of results
    """
    all_results = {strategy: [] for strategy in experiment_config.strategies}
    
    for run_num in range(experiment_config.num_runs):
        print(f"\n=== Run {run_num + 1}/{experiment_config.num_runs} ===")
        
        # Set random seed for reproducibility
        if experiment_config.random_seed is not None:
            seed = experiment_config.random_seed + run_num
            np.random.seed(seed)
            import random
            random.seed(seed)
        
        for strategy in experiment_config.strategies:
            print(f"Running {strategy}...")
            result = simulator_factory(strategy, experiment_config.duration)
            all_results[strategy].append(result)
    
    return all_results


if __name__ == "__main__":
    # Example usage
    evaluator = AcademicEvaluator()
    
    # Example data
    data1 = [10.5, 11.2, 9.8, 10.9, 11.5]
    data2 = [12.3, 13.1, 11.9, 12.7, 13.5]
    
    stats1 = evaluator.calculate_statistics(data1)
    stats2 = evaluator.calculate_statistics(data2)
    
    print("Statistics for Group 1:")
    print(f"Mean: {stats1.mean:.2f}, CI: [{stats1.ci_95_lower:.2f}, {stats1.ci_95_upper:.2f}]")
    
    print("\nStatistics for Group 2:")
    print(f"Mean: {stats2.mean:.2f}, CI: [{stats2.ci_95_lower:.2f}, {stats2.ci_95_upper:.2f}]")
    
    comparison = evaluator.t_test(data1, data2)
    print(f"\nT-test: {comparison['interpretation']}")

