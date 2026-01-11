#!/usr/bin/env python3
"""
Simple Intelligent Traffic Signal Control System
Select strategy → Click Start → Simulation starts → View data
Supports both custom single intersection and TAPASCologne dataset
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
import os
import json
import time

# Import configuration
sys.path.append(".")
from src.config import SimulationConfig

# Page configuration
st.set_page_config(
    page_title="Intelligent Traffic Signal Control System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Signal Control System")

# Dataset selection at the top
col1, col2 = st.columns([1, 3])
with col1:
    dataset_option = st.selectbox(
        "Select Dataset:",
        options=["custom", "4intersection", "tapas_cologne"],
        index=0 if SimulationConfig.DATASET == "custom" else (1 if SimulationConfig.DATASET == "4intersection" else 2),
        format_func=lambda x: {
            "custom": "Custom Single Intersection",
            "4intersection": "4-Intersection Grid Network",
            "tapas_cologne": "TAPASCologne (Real Cologne Network)"
        }.get(x, x)
    )
    if dataset_option != SimulationConfig.DATASET:
        # When switching dataset, clear previous simulation state and history
        st.session_state.simulation_running = False
        st.session_state.simulation_results = None
        st.session_state.all_strategy_results = {}
        SimulationConfig.set_dataset(dataset_option)
        st.rerun()

with col2:
    current_config = SimulationConfig.get_current_config()
    st.info(f"📍 **Current Dataset:** {current_config['description']}")

st.markdown("**Select strategy → Click Start → Simulation starts → View results**")
st.markdown("---")

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'all_strategy_results' not in st.session_state:
    st.session_state.all_strategy_results = {}  # Store real results for all strategies

# Functions
def run_simulation(strategy):
    """Run simulation"""
    try:
        # Clean up old temporary result/script files before starting a new run
        # to avoid showing stale results from previous simulations
        if os.path.exists("temp_results.json"):
            try:
                os.remove("temp_results.json")
            except Exception:
                # If removal fails, continue; new results will overwrite it later
                pass
        if os.path.exists("temp_simulation.py"):
            try:
                os.remove("temp_simulation.py")
            except Exception:
                pass

        # Get current dataset
        dataset = SimulationConfig.DATASET
        
        # Create simulation script - avoid nested f-string issues
        script_content = '''import sys
import os
sys.path.append(".")
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig

# Switch to project root directory
os.chdir(".")

print("Starting SUMO GUI simulation...")
print("Strategy: ''' + strategy + '''")
print("Dataset: ''' + dataset + '''")

# Set dataset
SimulationConfig.set_dataset("''' + dataset + '''")

# Start simulation with correct dataset
simulator = TrafficSimulator(use_gui=True, dataset="''' + dataset + '''")

# For TAPASCologne, run shorter simulation (600s = 10 min) due to larger network
duration = 600 if SimulationConfig.is_tapas_cologne() else 600
print(f"Running simulation for {duration} seconds...")

metrics = simulator.run_simulation(duration=duration, strategy="''' + strategy + '''")

# Save results - fix JSON serialization
import json

def clean_metrics_for_json(data):
    """Clean metrics data to ensure it is JSON serializable"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Convert tuple keys to strings
            if isinstance(key, tuple):
                str_key = str(key)
            else:
                str_key = str(key)
            cleaned[str_key] = clean_metrics_for_json(value)
        return cleaned
    elif isinstance(data, (list, tuple)):
        return [clean_metrics_for_json(item) for item in data]
    elif hasattr(data, 'to_dict'):  # pandas objects
        return clean_metrics_for_json(data.to_dict())
    else:
        return data

cleaned_metrics = clean_metrics_for_json(metrics)

with open("temp_results.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_metrics, f, indent=2, ensure_ascii=False, default=str)

simulator.export_data()
print("Simulation finished, results saved")
'''
        
        # Write script
        with open("temp_simulation.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # Start simulation (run in background)
        subprocess.Popen([sys.executable, "temp_simulation.py"])
        
    except Exception as e:
        st.error(f"Failed to start: {e}")
        st.session_state.simulation_running = False

def check_simulation_results():
    """Check simulation results"""
    if os.path.exists("temp_results.json"):
        try:
            with open("temp_results.json", "r", encoding="utf-8") as f:
                results = json.load(f)
            st.session_state.simulation_results = results
            st.session_state.simulation_running = False
            
            # Save into all strategies' results
            current_strategy = st.session_state.get('selected_strategy', 'UNKNOWN')
            strategy_names = {
                'FIXED_TIME': '🕐 Fixed-time Control',
                'ADAPTIVE': '🧠 Adaptive Control', 
                'DQN': '🤖 DQN Reinforcement Learning',
                'MAX_PRESSURE': '⚖️ MaxPressure Control'
            }
            strategy_display_name = strategy_names.get(current_strategy, current_strategy)
            
            # Save results with strategy info and timestamp
            st.session_state.all_strategy_results[current_strategy] = {
                'display_name': strategy_display_name,
                'results': results,
                'timestamp': time.strftime('%H:%M:%S')
            }
            
            # Clean up temporary files
            try:
                os.remove("temp_results.json")
                os.remove("temp_simulation.py")
            except:
                pass
                
        except Exception as e:
            pass

def show_results(results):
    """Display real simulation results (no preset data)"""
    
    # Get current strategy info
    current_strategy = st.session_state.get('selected_strategy', 'UNKNOWN')
    strategy_names = {
        'FIXED_TIME': '🕐 Fixed-time Control',
        'ADAPTIVE': '🧠 Adaptive Control', 
        'DQN': '🤖 DQN Reinforcement Learning'
    }
    current_name = strategy_names.get(current_strategy, 'Unknown strategy')
    
    # Strategy history management
    st.markdown("### 🗂️ Simulation history management")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Show executed strategies
        all_results = st.session_state.all_strategy_results
        if all_results:
            st.success(f"✅ Number of executed strategies: {len(all_results)}")
            for strategy_key, strategy_data in all_results.items():
                st.write(f"- {strategy_data['display_name']} (Run at: {strategy_data['timestamp']})")
        else:
            st.info("📝 No strategies have been executed yet")
    
    with col2:
        if st.button("🗑️ Clear history", help="Clear all historical simulation results"):
            st.session_state.all_strategy_results = {}
            st.success("✅ History cleared")
            st.rerun()
    
    st.markdown("---")
    
    # Current strategy results
    st.markdown(f"### 📈 Current simulation results: {current_name}")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_wait = results.get('avg_waiting_time', 0)
        st.metric("Average waiting time", f"{avg_wait:.1f}s", 
                 delta=None, delta_color="inverse")
    
    with col2:
        avg_speed = results.get('avg_speed', 0) 
        st.metric("Average speed", f"{avg_speed:.1f}m/s")
    
    with col3:
        throughput = results.get('throughput_per_hour', 0)
        st.metric("Throughput per hour", f"{throughput:.0f} vehicles")
    
    with col4:
        co2 = results.get('total_co2', 0)
        st.metric("CO₂ emissions", f"{co2:.1f}mg", 
                 delta=None, delta_color="inverse")
    
    # Detailed data table
    st.markdown("### 📋 Detailed metrics")
    
    metrics_data = {
        'Metric': [
            'Average waiting time (s)', 'Maximum waiting time (s)', 'Average speed (m/s)', 'Maximum speed (m/s)',
            'Vehicle throughput', 'Throughput per hour', 'Congestion index', 'Total CO₂ emissions (mg)',
            'Fuel consumption (ml)', 'Average lane occupancy'
        ],
        'Value': [
            f"{results.get('avg_waiting_time', 0):.2f}",
            f"{results.get('max_waiting_time', 0):.2f}",
            f"{results.get('avg_speed', 0):.2f}",
            f"{results.get('max_speed', 0):.2f}",
            f"{results.get('throughput', 0):.0f}",
            f"{results.get('throughput_per_hour', 0):.0f}",
            f"{results.get('congestion_index', 0):.3f}",
            f"{results.get('total_co2', 0):.2f}",
            f"{results.get('total_fuel', 0):.2f}",
            f"{results.get('avg_lane_occupancy', 0):.3f}"
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Real strategy comparison (only shown when multiple strategies have been run)
    st.markdown("### 📊 Real strategy comparison")
    
    all_results = st.session_state.all_strategy_results
    
    if len(all_results) < 2:
        st.info(f"📝 **More data needed for comparison**\n\nCurrently only {len(all_results)} strategy result(s). To see comparisons:\n\n1. Select another strategy (Fixed-time, Adaptive, DQN)\n2. Click \"Start Simulation\" to run\n3. Once multiple results are collected, charts will appear here")
    else:
        st.success(f"✅ **Real data comparison** - based on {len(all_results)} executed strategies")
        
        # Prepare real comparison data
        comparison_data = {
            'Strategy': [],
            'Waiting Time': [],
            'Speed': [],
            'Throughput': []
        }
        
        for strategy_key, strategy_data in all_results.items():
            results_data = strategy_data['results']
            comparison_data['Strategy'].append(strategy_data['display_name'])
            comparison_data['Waiting Time'].append(results_data.get('avg_waiting_time', 0))
            comparison_data['Speed'].append(results_data.get('avg_speed', 0))
            comparison_data['Throughput'].append(results_data.get('throughput_per_hour', 0))
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_comparison, x='Strategy', y='Waiting Time',
                        title="Average waiting time comparison (lower is better)",
                        color='Strategy',
                        text='Waiting Time')
            fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_comparison, x='Strategy', y='Throughput',
                        title="System throughput comparison (higher is better)", 
                        color='Strategy',
                        text='Throughput')
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("#### 📋 Detailed comparison data")
        
        comparison_table = pd.DataFrame({
            'Strategy': df_comparison['Strategy'],
            'Waiting time (s)': [f"{x:.2f}" for x in df_comparison['Waiting Time']],
            'Average speed (m/s)': [f"{x:.2f}" for x in df_comparison['Speed']],
            'Throughput (veh/h)': [f"{x:.0f}" for x in df_comparison['Throughput']]
        })
        
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
        
        # Performance analysis
        st.markdown("#### 🎯 Performance analysis")
        
        # Find best strategies
        best_wait_idx = df_comparison['Waiting Time'].idxmin()
        best_throughput_idx = df_comparison['Throughput'].idxmax()
        best_speed_idx = df_comparison['Speed'].idxmax()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"🏆 **Shortest waiting time**\n{df_comparison.iloc[best_wait_idx]['Strategy']}\n{df_comparison.iloc[best_wait_idx]['Waiting Time']:.1f}s")
        
        with col2:
            st.success(f"🚀 **Highest throughput**\n{df_comparison.iloc[best_throughput_idx]['Strategy']}\n{df_comparison.iloc[best_throughput_idx]['Throughput']:.0f} veh/h")
        
        with col3:
            st.success(f"⚡ **Fastest speed**\n{df_comparison.iloc[best_speed_idx]['Strategy']}\n{df_comparison.iloc[best_speed_idx]['Speed']:.1f}m/s")
    
    # Conclusions and suggestions
    st.markdown("### 🎯 Conclusion")
    
    current_wait = results.get('avg_waiting_time', 0)
    if current_wait < 30:
        st.success("✅ Excellent! Short waiting time and smooth traffic flow")
    elif current_wait < 45:
        st.info("🔶 Good! Moderate waiting time, room for improvement")  
    else:
        st.warning("🔻 Needs improvement! Long waiting time, consider optimizing strategy")
    
    # Data file links
    st.markdown("### 📁 Detailed data")
    st.info("🗂️ Full simulation data saved to the `output/` directory, including:\n- Vehicle trajectory data (CSV)\n- Traffic light state data (CSV)\n- Performance metrics report (JSON)")

# Main control area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Control panel")
    
    # Strategy selection
    strategy_options = {
        'FIXED_TIME': '🕐 Fixed-time Control',
        'ADAPTIVE': '🧠 Adaptive Control', 
        'DQN': '🤖 DQN Reinforcement Learning',
        'MAX_PRESSURE': '⚖️ MaxPressure Control'
    }
    
    selected_strategy = st.selectbox(
        "Select control strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x]
    )
    
    st.markdown("**Strategy description:**")
    if selected_strategy == 'FIXED_TIME':
        st.info("⏰ Switch signals at fixed intervals\nSuitable for regular traffic patterns")
    elif selected_strategy == 'ADAPTIVE': 
        st.info("🧠 Adjust intelligently based on vehicle counts\nSuitable for varying traffic flows")
    elif selected_strategy == 'MAX_PRESSURE':
        st.info("⚖️ Serve the phase with the highest queued demand (MaxPressure)\nStrong queue-based baseline for comparison")
    else:
        st.info("🤖 AI learns optimal control strategies\nSuitable for complex traffic conditions")
    
    # Start button
    if not st.session_state.simulation_running:
        if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
            st.session_state.simulation_running = True
            st.session_state.simulation_results = None
            st.session_state.selected_strategy = selected_strategy  # Save selected strategy
            
            # Run simulation
            run_simulation(selected_strategy)
    else:
        st.warning("⏳ Simulation is running, please wait...")
        if st.button("⏹️ Restart", use_container_width=True):
            st.session_state.simulation_running = False
            st.rerun()

with col2:
    st.subheader("📊 Simulation status")
    
    if st.session_state.simulation_running:
        st.info("🔄 Simulation is running...")
        st.info("🖥️ SUMO GUI window is open; you can watch the traffic simulation")
        st.info("⏱️ The simulation will run for 2 minutes; please wait")
        
        # Auto-refresh to check results
        time.sleep(3)
        check_simulation_results()
        st.rerun()
        
    elif st.session_state.simulation_results:
        st.success("✅ Simulation completed!")
        show_results(st.session_state.simulation_results)
    else:
        st.markdown("""
        ### 🎯 Steps
        1. **Select a strategy** - choose on the left
        2. **Start simulation** - click the "Start Simulation" button
        3. **Watch** - SUMO GUI window will open automatically
        4. **View results** - data will appear here after completion
        
        ### 💡 Tips
        - During the simulation you will see vehicles moving through an intersection
        - Observe how signals change under different strategies
        - Results include waiting time, speed, emissions, and more
        """)

if __name__ == "__main__":
    pass 