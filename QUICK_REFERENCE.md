# Quick Reference: TAPASCologne Dataset

## 🚀 Quick Start

### Switch Dataset
```python
from src.config import SimulationConfig

# Use TAPASCologne (real Cologne network)
SimulationConfig.set_dataset('tapas_cologne')

# Use Custom (single intersection)
SimulationConfig.set_dataset('custom')
```

### Run Simulation
```python
from src.traffic_simulator import TrafficSimulator

# Create simulator
simulator = TrafficSimulator(use_gui=True, dataset='tapas_cologne')

# Run simulation
metrics = simulator.run_simulation(
    duration=600,         # 10 minutes
    strategy='FIXED_TIME' # or 'ADAPTIVE' or 'DQN'
)
```

## 📋 Common Tasks

### 1. Test Setup
```bash
python test_tapas_cologne.py
```

### 2. Run Examples
```bash
python example_tapas_cologne.py
```

### 3. Start Streamlit App
```bash
python start.py
# Select dataset from dropdown in browser
```

### 4. Get Network Info
```python
simulator = TrafficSimulator(dataset='tapas_cologne')
simulator.start_simulation()

print(f"Traffic lights: {len(simulator.traffic_lights)}")
print(f"Controlled: {len(simulator.controlled_traffic_lights)}")
print(f"Lanes: {len(simulator.lanes)}")
```

### 5. Train RL Agent
```python
from src.rl_agent import RLAgent

simulator = TrafficSimulator(dataset='tapas_cologne')
simulator.start_simulation()

# Agent auto-calculates state dimensions
agent = RLAgent(simulator=simulator)

# Your training loop here...
```

## ⚙️ Configuration

### File: `src/config.py`

```python
# Line 11: Choose default dataset
DATASET = 'tapas_cologne'  # or 'custom'

# Line 69: Limit traffic lights for performance
'max_controlled_lights': 10,  # Reduce for faster training

# Line 85: RL parameters
RL_CONFIG = {
    'memory_size': 50000,  # Increase for complex scenarios
    'hidden_dims': [256, 256, 128],  # Network architecture
}
```

## 📊 Key Differences

| Feature | Custom | TAPASCologne |
|---------|--------|--------------|
| Traffic Lights | 1 | 245 (10 controlled by default) |
| Lanes | 8 | ~2000 (auto-detected) |
| State Dimension | 37 (fixed) | Variable (auto-calculated) |
| Simulation Time | 5-10 min | 10-60 min recommended |
| Use Case | Algorithm development | Real-world validation |

## 🎯 Best Practices

### Development Phase
- Use **Custom** dataset for fast iteration
- No GUI (`use_gui=False`) for training
- Short simulation duration (300-600s)

### Validation Phase
- Use **TAPASCologne** for real-world testing
- GUI (`use_gui=True`) for visualization
- Longer simulation (600-3600s)
- Limit `max_controlled_lights` to 5-10

### Production/Publication
- **TAPASCologne** for credible results
- Multiple runs for statistical significance
- Document configuration parameters

## 🔍 Debugging

### Check Current Dataset
```python
print(f"Current dataset: {SimulationConfig.DATASET}")
print(f"Config: {SimulationConfig.get_current_config()}")
```

### Verify State Dimensions
```python
agent = RLAgent(simulator=simulator)
print(f"State dim: {agent.config['state_dim']}")
print(f"Action dim: {agent.config['action_dim']}")
```

### Get Sample State
```python
state_dict = simulator.get_current_state()
print(f"Lanes: {len(state_dict['lane_vehicle_counts'])}")
print(f"Detectors: {len(state_dict['detector_occupancy'])}")
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `src/config.py` | Dataset configuration |
| `src/traffic_simulator.py` | Simulation engine |
| `src/rl_agent.py` | DQN agent |
| `test_tapas_cologne.py` | Test script |
| `example_tapas_cologne.py` | Examples |
| `TAPAS_COLOGNE_GUIDE.md` | Full guide |
| `MIGRATION_SUMMARY.md` | Change details |

## ⚡ Performance Tips

1. **Reduce Controlled Lights**: Edit `config.py`, set `max_controlled_lights: 5`
2. **Disable GUI**: Use `use_gui=False` for training
3. **Shorter Simulations**: Start with 300-600 seconds
4. **Monitor Memory**: TAPASCologne uses more RAM
5. **Use GPU**: DQN training benefits from CUDA if available

## 🐛 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| "SUMO not found" | Install SUMO, add to PATH |
| "Config file not found" | Check TAPASCologne-0.32.0/ folder exists |
| "State dimension mismatch" | Pass `simulator` to `RLAgent()` |
| "Memory error" | Reduce `max_controlled_lights` |
| Slow simulation | Disable GUI, reduce controlled lights |

## 📚 Learn More

- Full guide: `TAPAS_COLOGNE_GUIDE.md`
- Changes: `MIGRATION_SUMMARY.md`
- SUMO docs: https://sumo.dlr.de/docs/

## 💡 Pro Tips

✅ Start with custom dataset to debug your algorithm  
✅ Switch to TAPASCologne for validation  
✅ Use GUI to understand the network visually  
✅ Compare results between both datasets  
✅ Document which dataset was used in results  

---

**Need Help?** Check the guides or examine `example_tapas_cologne.py` for working code examples.
