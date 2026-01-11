# Academic Features Guide

This document describes the academic-focused features added to enhance the research value of the project.

## Overview

The project now includes comprehensive academic evaluation tools for:
- Statistical analysis and significance testing
- Reproducible experiments
- Multiple independent runs with confidence intervals
- Comparative analysis between strategies
- Academic-style report generation

## Key Features

### 1. Statistical Analysis (`src/academic_evaluator.py`)

#### Statistical Measures
- **Mean and Standard Deviation**: Basic descriptive statistics
- **Median and Quartiles**: Robust statistics (Q25, Q75)
- **95% Confidence Intervals**: Using t-distribution for small samples
- **Min/Max Values**: Range of results

#### Statistical Tests
- **Independent Samples T-test**: Compare means between two strategies
- **Mann-Whitney U Test**: Non-parametric alternative (robust to outliers)
- **Effect Size (Cohen's d)**: Measure of practical significance

### 2. Reproducibility

#### Experiment Configuration
```python
from src.academic_evaluator import ExperimentConfig

config = ExperimentConfig(
    experiment_name="my_experiment",
    dataset="custom",
    strategies=["FIXED_TIME", "ADAPTIVE", "DQN"],
    num_runs=10,  # Multiple runs for statistical significance
    duration=600,
    random_seed=42,  # For reproducibility
    notes="My experiment description"
)
```

#### Random Seed Control
- Set random seeds for SUMO and Python random number generators
- Ensures reproducible results across runs
- Each run uses `seed + run_number` for variation

### 3. Multiple Independent Runs

Run multiple simulations to get statistically significant results:

```python
from src.academic_evaluator import run_batch_experiments

all_results = run_batch_experiments(config, simulator_factory)
```

This runs each strategy multiple times and collects results for statistical analysis.

### 4. Comparative Analysis

Compare strategies with statistical significance testing:

```python
evaluator = AcademicEvaluator()

comparison = evaluator.compare_strategies(
    strategy_results={
        "FIXED_TIME": [result1, result2, ...],
        "ADAPTIVE": [result1, result2, ...],
        "DQN": [result1, result2, ...]
    },
    metric_name="avg_waiting_time"
)
```

Returns:
- Statistical summary for each strategy
- Pairwise comparisons with t-tests and Mann-Whitney tests
- P-values and effect sizes

### 5. Academic Report Generation

Generate publication-ready reports:

```python
report = evaluator.generate_academic_report(experiment_config, all_results)
```

The report includes:
- Experiment metadata
- Statistical summaries (mean, std, CI) for each strategy
- Pairwise statistical tests
- Markdown format for easy conversion to LaTeX/PDF

### 6. Additional Academic Metrics

Enhanced metrics beyond basic performance:

- **Travel Time Index (TTI)**: Ratio of actual to free-flow travel time
- **Delay per Vehicle**: Average waiting time
- **Efficiency Score**: Combined metric (throughput / (1 + waiting_time))
- **CO2 per Vehicle**: Environmental impact normalized by vehicle count
- **Fairness Metrics**: Coefficient of variation (requires per-vehicle data)

## Usage Examples

### Example 1: Quick Statistical Analysis

```python
from src.academic_evaluator import AcademicEvaluator

evaluator = AcademicEvaluator()

# Analyze multiple runs
results = [run1, run2, run3, run4, run5]  # List of result dicts
analysis = evaluator.analyze_multiple_runs(results, "avg_waiting_time")

print(f"Mean: {analysis['statistics']['mean']:.2f}")
print(f"95% CI: [{analysis['statistics']['ci_95_lower']:.2f}, "
      f"{analysis['statistics']['ci_95_upper']:.2f}]")
```

### Example 2: Compare Two Strategies

```python
# Run experiments
fixed_time_results = [run_simulation("FIXED_TIME") for _ in range(10)]
adaptive_results = [run_simulation("ADAPTIVE") for _ in range(10)]

# Statistical test
t_test = evaluator.t_test(
    [r['avg_waiting_time'] for r in fixed_time_results],
    [r['avg_waiting_time'] for r in adaptive_results]
)

print(t_test['interpretation'])
# Output: "Significant difference (p=0.0234, Cohen's d=0.456)"
```

### Example 3: Full Academic Experiment

```bash
python run_academic_experiments.py
```

This will:
1. Run each strategy 10 times (configurable)
2. Collect all results
3. Perform statistical analysis
4. Generate academic report
5. Export results to JSON

## Output Files

### Academic Reports
- Location: `output/academic/`
- Format: Markdown (`.md`)
- Contains: Statistical summaries, significance tests, comparisons

### JSON Results
- Location: `output/academic/`
- Format: JSON
- Contains: Raw results, statistics, experiment config
- Use for: Further analysis in R, Python, or other tools

## Best Practices for Academic Research

### 1. Number of Runs
- **Minimum**: 5 runs per strategy
- **Recommended**: 10-30 runs for robust statistics
- **For publication**: 20+ runs for high confidence

### 2. Statistical Significance
- Use **p < 0.05** as threshold for significance
- Report **effect size** (Cohen's d) for practical significance
- Use **non-parametric tests** (Mann-Whitney) if data is not normally distributed

### 3. Reproducibility
- Always set and report random seeds
- Document all parameters (traffic flow, duration, etc.)
- Save raw results for verification

### 4. Reporting
- Report **mean ± standard deviation**
- Include **95% confidence intervals**
- Show **statistical test results** (p-values, effect sizes)
- Use **box plots** or **violin plots** for visualization

## Integration with Existing Code

The academic evaluator works with existing simulation results:

```python
from src.traffic_simulator import TrafficSimulator
from src.academic_evaluator import AcademicEvaluator

# Run simulation (existing code)
simulator = TrafficSimulator(use_gui=False, dataset="custom")
metrics = simulator.run_simulation(duration=600, strategy="FIXED_TIME")

# Analyze with academic tools
evaluator = AcademicEvaluator()
stats = evaluator.calculate_statistics([metrics['avg_waiting_time']])
print(f"Mean waiting time: {stats.mean:.2f} ± {stats.std:.2f}")
```

## Future Enhancements

Potential additions for even more academic rigor:

1. **ANOVA**: Compare multiple strategies simultaneously
2. **Post-hoc tests**: Bonferroni, Tukey HSD
3. **Time-series analysis**: Convergence analysis for RL
4. **Sensitivity analysis**: Parameter sensitivity studies
5. **Fairness metrics**: Gini coefficient, Jain's index
6. **Visualization**: Academic-style plots (box plots, violin plots)
7. **LaTeX export**: Direct export to LaTeX tables

## References

- **Statistical Tests**: Scipy documentation
- **Effect Sizes**: Cohen, J. (1988). Statistical Power Analysis
- **Reproducibility**: Peng, R. D. (2011). Reproducible Research in Computational Science

