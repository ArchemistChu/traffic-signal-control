# OSM Web Wizard Parameter Guide

This guide explains how to configure SUMO's OSM Web Wizard parameters for generating realistic traffic networks for academic experiments.

## Understanding OSM Web Wizard Parameters

### Vehicle Type Parameters

#### **Cars (Passenger Vehicles)**
- **Count**: Number of passenger vehicles per hour
- **Recommendation**: 
  - **Low traffic**: 200-500 veh/h
  - **Moderate traffic**: 500-1000 veh/h (matches your current ~1,150 veh/h)
  - **High traffic**: 1000-2000 veh/h
  - **Congested**: 2000+ veh/h

#### **Trucks**
- **Count**: Number of trucks per hour
- **Recommendation**: 
  - **Urban**: 5-10% of car count (e.g., 50-100 trucks/h for 1000 cars/h)
  - **Industrial/Highway**: 15-25% of car count
  - **Your current setup**: ~100 trucks/h (10% of total)

#### **Buses**
- **Count**: Number of buses per hour
- **Recommendation**: 
  - **Low service**: 5-10 buses/h
  - **Moderate service**: 10-30 buses/h
  - **High service**: 30-60 buses/h
  - **Your current setup**: Not included (can add 10-20 buses/h)

#### **Motorcycles**
- **Count**: Number of motorcycles per hour
- **Recommendation**: 
  - **Low**: 20-50 motorcycles/h
  - **Moderate**: 50-100 motorcycles/h
  - **High**: 100-200 motorcycles/h
  - **Your current setup**: Not included

#### **Pedestrians**
- **Count**: Number of pedestrians per hour
- **Recommendation**: 
  - **Low activity**: 50-100 pedestrians/h
  - **Moderate activity**: 100-300 pedestrians/h
  - **High activity**: 300-500 pedestrians/h
  - **Note**: Requires pedestrian infrastructure (sidewalks, crossings)
  - **Your current setup**: Not included

### Traffic Flow Parameters

#### **Through Traffic Factor**
- **What it does**: Multiplies the base traffic count for vehicles that pass through the network without stopping
- **Range**: 0.0 to 2.0 (typically 0.1 to 1.0)
- **Meaning**:
  - **0.0**: No through traffic (all vehicles have origin/destination in network)
  - **0.3-0.5**: Moderate through traffic (30-50% of vehicles pass through)
  - **0.7-1.0**: High through traffic (most vehicles are just passing through)
  - **1.5-2.0**: Very high through traffic (highway/arterial scenario)

- **Recommendation by Scenario**:
  - **Single intersection (your case)**: **0.2-0.4**
    - Most vehicles are local traffic
    - Some vehicles pass through intersection
  - **Urban grid**: **0.3-0.5**
    - Mix of local and through traffic
  - **Highway/Arterial**: **0.7-1.0**
    - Most traffic is passing through
  - **Residential area**: **0.1-0.2**
    - Mostly local traffic

#### **Count (Base Traffic Count)**
- **What it does**: Base number of vehicles per hour for each vehicle type
- **How it works**: 
  - This is the **base count** before through traffic factor is applied
  - Final traffic = base_count × (1 + through_traffic_factor)
  - Example: 1000 cars/h with 0.3 through factor = 1000 × 1.3 = 1300 veh/h

## Recommended Configurations

### Configuration 1: Low Traffic (Baseline)
**Use for**: Testing control strategies under light load
```
Cars:        300 veh/h
Trucks:       30 veh/h (10%)
Buses:        10 veh/h
Motorcycles:  20 veh/h
Pedestrians:  50 veh/h
Through Traffic Factor: 0.2
Total: ~400 veh/h
```

### Configuration 2: Moderate Traffic (Your Current)
**Use for**: Standard comparison experiments
```
Cars:        1000 veh/h
Trucks:      100 veh/h (10%)
Buses:        20 veh/h
Motorcycles:  50 veh/h
Pedestrians: 100 veh/h
Through Traffic Factor: 0.3
Total: ~1,300 veh/h (with through traffic)
```

### Configuration 3: High Traffic (Stress Test)
**Use for**: Testing under congestion
```
Cars:        2000 veh/h
Trucks:      200 veh/h (10%)
Buses:        40 veh/h
Motorcycles: 100 veh/h
Pedestrians: 200 veh/h
Through Traffic Factor: 0.4
Total: ~2,600 veh/h (with through traffic)
```

### Configuration 4: Academic Experiment (Recommended)
**Use for**: Multiple scenarios for statistical analysis
```
Scenario A (Low):
  Cars: 400, Trucks: 40, Through: 0.2

Scenario B (Moderate):
  Cars: 800, Trucks: 80, Through: 0.3

Scenario C (High):
  Cars: 1200, Trucks: 120, Through: 0.4
```

## Parameter Selection Strategy

### Step 1: Determine Your Research Question
- **Question**: "How do control strategies perform under different traffic loads?"
  - **Answer**: Use 3-5 different traffic levels (low, moderate, high)
  
- **Question**: "What is the impact of truck percentage?"
  - **Answer**: Keep car count constant, vary truck percentage (5%, 10%, 15%, 20%)

- **Question**: "How does through traffic affect performance?"
  - **Answer**: Keep vehicle counts constant, vary through factor (0.1, 0.3, 0.5, 0.7)

### Step 2: Match Real-World Data (If Available)
If you have real-world traffic data:
- **Peak hour volume**: Use as base car count
- **Truck percentage**: Calculate from data (typically 5-15% in urban areas)
- **Through traffic**: Estimate from origin-destination data

### Step 3: Start Conservative
- **Begin with lower values** (e.g., 500 cars/h)
- **Run test simulation** to check:
  - Does simulation complete without errors?
  - Are waiting times realistic? (should be > 0, < 60s typically)
  - Is throughput reasonable?
- **Gradually increase** until you find the right balance

### Step 4: Consider Network Size
- **Small network** (single intersection): Lower counts (300-800 veh/h)
- **Medium network** (4 intersections): Moderate counts (800-1500 veh/h)
- **Large network** (city-scale): Higher counts (1500-3000+ veh/h)

## Practical Examples

### Example 1: Single Intersection (Your Current Setup)
**Goal**: Match your current traffic flow (~1,150 veh/h)

**OSM Web Wizard Settings**:
```
Cars: 900 veh/h
Trucks: 90 veh/h
Buses: 0 veh/h (optional: 10-20)
Motorcycles: 0 veh/h (optional: 20-30)
Pedestrians: 0 veh/h (optional: 50-100)
Through Traffic Factor: 0.3

Calculation:
  Base: 900 + 90 = 990 veh/h
  With through traffic: 990 × 1.3 = 1,287 veh/h
  This matches your ~1,150 veh/h target
```

### Example 2: Urban Grid Network
**Goal**: Realistic urban traffic

**OSM Web Wizard Settings**:
```
Cars: 1200 veh/h
Trucks: 120 veh/h (10%)
Buses: 30 veh/h
Motorcycles: 50 veh/h
Pedestrians: 150 veh/h
Through Traffic Factor: 0.4

Total: ~1,500 veh/h base × 1.4 = ~2,100 veh/h
```

### Example 3: Highway/Arterial
**Goal**: High-speed through traffic

**OSM Web Wizard Settings**:
```
Cars: 2000 veh/h
Trucks: 300 veh/h (15%)
Buses: 20 veh/h
Motorcycles: 30 veh/h
Pedestrians: 0 veh/h
Through Traffic Factor: 0.8

Total: ~2,350 veh/h base × 1.8 = ~4,230 veh/h
```

## Common Mistakes to Avoid

### ❌ Mistake 1: Too High Through Traffic Factor
- **Problem**: Setting through factor > 1.0 for single intersection
- **Result**: Unrealistic traffic patterns, all vehicles just passing through
- **Fix**: Use 0.2-0.4 for single intersections

### ❌ Mistake 2: Ignoring Vehicle Type Ratios
- **Problem**: Setting trucks = cars (50% trucks)
- **Result**: Unrealistic scenario (urban areas typically 5-15% trucks)
- **Fix**: Use 5-15% trucks relative to cars

### ❌ Mistake 3: Not Considering Network Capacity
- **Problem**: Setting 5000 veh/h for single-lane intersection
- **Result**: Extreme congestion, simulation may crash
- **Fix**: Check network capacity first (typically 500-2000 veh/h per lane)

### ❌ Mistake 4: Inconsistent Parameters
- **Problem**: Using different traffic levels without documentation
- **Result**: Cannot reproduce or compare experiments
- **Fix**: Document all parameters in experiment config

## Academic Best Practices

### 1. **Systematic Variation**
Test multiple parameter combinations:
```
Experiment Matrix:
  Traffic Levels: [Low, Moderate, High]
  Truck Percentages: [5%, 10%, 15%]
  Through Factors: [0.2, 0.3, 0.4]
  
Total: 3 × 3 × 3 = 27 experiments
```

### 2. **Document Everything**
Record all parameters:
- Vehicle counts (cars, trucks, buses, etc.)
- Through traffic factor
- Network characteristics
- Simulation duration
- Random seed (for reproducibility)

### 3. **Validate Against Real Data**
If possible:
- Compare simulation results to real-world data
- Adjust parameters to match observed patterns
- Report how parameters were chosen

### 4. **Use Multiple Scenarios**
For robust results:
- **Low traffic**: Baseline performance
- **Moderate traffic**: Typical conditions
- **High traffic**: Stress test
- **Peak hour**: Maximum capacity test

## Quick Reference Table

| Scenario | Cars/h | Trucks/h | Through Factor | Total veh/h |
|----------|--------|----------|----------------|-------------|
| Low (Single) | 300 | 30 | 0.2 | ~400 |
| Moderate (Single) | 800 | 80 | 0.3 | ~1,150 |
| High (Single) | 1200 | 120 | 0.4 | ~1,850 |
| Low (Grid) | 500 | 50 | 0.3 | ~715 |
| Moderate (Grid) | 1000 | 100 | 0.4 | ~1,540 |
| High (Grid) | 1500 | 150 | 0.5 | ~2,475 |

## Integration with Your Academic Evaluator

After generating networks with OSM Web Wizard:

1. **Run multiple experiments** with different parameters
2. **Use `AcademicEvaluator`** to analyze results statistically
3. **Compare strategies** across different traffic scenarios
4. **Generate reports** showing how parameters affect performance

Example:
```python
from src.academic_evaluator import ExperimentConfig

# Low traffic experiment
config_low = ExperimentConfig(
    experiment_name="low_traffic_comparison",
    strategies=["FIXED_TIME", "ADAPTIVE", "DQN"],
    num_runs=10,
    notes="Cars: 300/h, Trucks: 30/h, Through: 0.2"
)

# Moderate traffic experiment
config_moderate = ExperimentConfig(
    experiment_name="moderate_traffic_comparison",
    strategies=["FIXED_TIME", "ADAPTIVE", "DQN"],
    num_runs=10,
    notes="Cars: 800/h, Trucks: 80/h, Through: 0.3"
)
```

## Summary

**Key Takeaways**:
1. **Cars**: Start with 300-1000 veh/h for single intersections
2. **Trucks**: Use 5-15% of car count (typically 10%)
3. **Through Traffic Factor**: Use 0.2-0.4 for single intersections
4. **Test systematically**: Vary one parameter at a time
5. **Document everything**: Record all parameters for reproducibility
6. **Validate**: Compare with real-world data if available

**Your Current Setup** (for reference):
- Total: ~1,150 veh/h
- Cars: ~1,050 veh/h (91%)
- Trucks: ~100 veh/h (9%)
- Through factor: Not explicitly set (estimated 0.2-0.3)


