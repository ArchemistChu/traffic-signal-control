#!/usr/bin/env python3
"""
Flask-based Intelligent Traffic Signal Control System
Converted from Streamlit to Flask
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import subprocess
import sys
import os
import time
import threading
from datetime import datetime

# Import signal for process termination (Unix only)
if sys.platform != 'win32':
    import signal

# Import configuration
sys.path.append(".")
from src.config import SimulationConfig
from src.academic_evaluator import (
    AcademicEvaluator, ExperimentConfig, run_batch_experiments
)
from src.traffic_simulator import TrafficSimulator

app = Flask(__name__)
app.secret_key = 'traffic_signal_control_secret_key_2024'  # Change this in production

# Store large results on disk (avoid cookie-session overflow)
WEB_RESULTS_DIR = os.path.join(SimulationConfig.OUTPUT_DIR, "web_results")

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _safe_json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Initialize session variables
def init_session():
    if 'simulation_running' not in session:
        session['simulation_running'] = False
    # NOTE: Don't store full results in session (cookie size limit). Store a file path instead.
    if 'simulation_results_path' not in session:
        session['simulation_results_path'] = None
    if 'all_strategy_results' not in session:
        session['all_strategy_results'] = {}
    if 'selected_strategy' not in session:
        session['selected_strategy'] = 'FIXED_TIME'
    if 'current_dataset' not in session:
        session['current_dataset'] = SimulationConfig.DATASET
    if 'academic_experiment_running' not in session:
        session['academic_experiment_running'] = False
    if 'academic_experiment_results' not in session:
        session['academic_experiment_results'] = None
    if 'academic_experiment_progress' not in session:
        session['academic_experiment_progress'] = {}
    if 'marl_eval_running' not in session:
        session['marl_eval_running'] = False
    if 'marl_eval_results_path' not in session:
        session['marl_eval_results_path'] = None
    if 'marl_eval_status_path' not in session:
        session['marl_eval_status_path'] = None

@app.route('/')
def index():
    """Main page"""
    try:
        init_session()
        current_config = SimulationConfig.get_current_config()
        
        strategy_options = {
            'FIXED_TIME': '🕐 Fixed-time Control',
            'ADAPTIVE': '🧠 Adaptive Control', 
            'GA': '🧬 Genetic Algorithm (GA) baseline',
            'PRESSLIGHT': '📌 PressLight-inspired (pressure DQN)',
            'MAX_PRESSURE': '⚖️ MaxPressure Control',
            'SOTL': '🚦 SOTL (Self-Organizing)',
            'MARL_DQN': '🤝 Multi-Agent DQN (trained)'
        }
        
        # Render-time results: load from disk (not session cookie)
        simulation_results = None
        results_path = session.get('simulation_results_path')
        if results_path and os.path.exists(results_path):
            simulation_results = _safe_json_load(results_path)

        # If session says running but the temp script is gone AND there's no temp results,
        # clear stale running flag to avoid "stuck running" UI.
        if session.get('simulation_running', False):
            if not os.path.exists("temp_simulation.py") and not os.path.exists("temp_results.json"):
                session['simulation_running'] = False
                session.modified = True

        use_gui = session.get('use_gui', True)
        return render_template('index.html',
                             current_dataset=session.get('current_dataset', 'custom'),
                             dataset_description=current_config['description'],
                             strategy_options=strategy_options,
                             selected_strategy=session.get('selected_strategy', 'FIXED_TIME'),
                             simulation_running=session.get('simulation_running', False),
                             simulation_results=simulation_results,
                             all_strategy_results=session.get('all_strategy_results', {}),
                             use_gui=use_gui)
    except Exception as e:
        print(f"Error in index route: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple error page instead of blank
        return f"<h1>Error loading page</h1><p>{str(e)}</p><p><a href='/'>Reload</a></p>", 500

@app.route('/set_dataset', methods=['POST'])
def set_dataset():
    """Change dataset"""
    data = request.get_json()
    dataset_option = data.get('dataset', 'custom')
    
    if dataset_option != SimulationConfig.DATASET:
        session['simulation_running'] = False
        session['simulation_results_path'] = None
        session['all_strategy_results'] = {}
        SimulationConfig.set_dataset(dataset_option)
        session['current_dataset'] = dataset_option
        session.modified = True
    
    return jsonify({'status': 'success', 'dataset': dataset_option})

def kill_existing_sumo_processes():
    """Kill any existing SUMO processes to prevent duplicate windows"""
    try:
        if sys.platform == 'win32':
            # Windows: Kill all sumo-gui.exe and sumo.exe processes
            subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                         capture_output=True, timeout=3)
            subprocess.run(['taskkill', '/F', '/IM', 'sumo.exe'], 
                         capture_output=True, timeout=3)
        else:
            # Linux/Mac: Use pkill
            subprocess.run(['pkill', '-f', 'sumo-gui'], 
                         capture_output=True, timeout=3)
            subprocess.run(['pkill', '-f', 'sumo'], 
                         capture_output=True, timeout=3)
    except Exception as e:
        # Ignore errors - processes might not exist
        pass

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Start simulation"""
    data = request.get_json()
    strategy = data.get('strategy', 'FIXED_TIME')
    use_gui_option = data.get('use_gui', None)  # Optional: allow user to override
    requested_duration = int(data.get('duration', 600))  # Get duration from request, default 600s
    
    if session.get('simulation_running', False):
        return jsonify({'status': 'error', 'message': 'Simulation already running'})
    
    try:
        # Kill any existing SUMO processes first to prevent duplicate windows
        kill_existing_sumo_processes()
        time.sleep(0.5)  # Give processes time to close
        
        # Clean up old temporary files (but keep temp_results.json if it exists - might be from previous run)
        for temp_file in ["temp_simulation.py"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
        # Also clean up old results file if it's stale (older than 5 minutes)
        if os.path.exists("temp_results.json"):
            try:
                file_age = time.time() - os.path.getmtime("temp_results.json")
                if file_age > 300:  # 5 minutes
                    os.remove("temp_results.json")
            except:
                pass
        
        # Get current dataset
        dataset = session.get('current_dataset', SimulationConfig.DATASET)
        
        # Determine if we should use GUI based on map size
        # Large maps (Cologne, Vancouver) are slower with GUI
        large_maps = ['cologne', 'vancouver', 'los_angeles']
        if use_gui_option is None:
            # Auto-detect: use GUI for small maps, headless for large maps
            use_gui = dataset not in large_maps
        else:
            use_gui = use_gui_option
        
        session['selected_strategy'] = strategy
        session['use_gui'] = use_gui  # Store GUI preference
        session['simulation_duration'] = requested_duration  # Store duration
        session['simulation_running'] = True
        session['simulation_results_path'] = None
        session.modified = True
        
        # Validate duration
        if requested_duration < 60 or requested_duration > 7200:
            return jsonify({'status': 'error', 'message': 'Duration must be between 60 and 7200 seconds'})
        
        # Create simulation script
        gui_text = "GUI" if use_gui else "headless (faster)"
        script_content = f'''import sys
import os
sys.path.append(".")
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig

os.chdir(".")

print("Starting SUMO {gui_text} simulation...")
print("Strategy: {strategy}")
print("Dataset: {dataset}")

SimulationConfig.set_dataset("{dataset}")

simulator = TrafficSimulator(use_gui={use_gui}, dataset="{dataset}")

duration = {requested_duration}
print(f"Running simulation for {{duration}} seconds ({requested_duration/60:.1f} minutes)...")

metrics = simulator.run_simulation(duration=duration, strategy="{strategy}")

import json

def clean_metrics_for_json(data):
    """Clean metrics data to ensure it is JSON serializable"""
    if isinstance(data, dict):
        cleaned = {{}}
        for key, value in data.items():
            if isinstance(key, tuple):
                str_key = str(key)
            else:
                str_key = str(key)
            cleaned[str_key] = clean_metrics_for_json(value)
        return cleaned
    elif isinstance(data, (list, tuple)):
        return [clean_metrics_for_json(item) for item in data]
    elif hasattr(data, 'to_dict'):
        return clean_metrics_for_json(data.to_dict())
    else:
        return data

cleaned_metrics = clean_metrics_for_json(metrics)

with open("temp_results.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_metrics, f, indent=2, ensure_ascii=False, default=str)

simulator.export_data()
print("Simulation finished, results saved")

# Force close SUMO GUI to prevent dialog from appearing
import time
time.sleep(0.5)  # Give time for results to be saved
if {use_gui}:
    import subprocess
    import sys
    try:
        if sys.platform == 'win32':
            # Force kill SUMO GUI to avoid dialog
            subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                         capture_output=True, timeout=3)
        else:
            subprocess.run(['pkill', '-f', 'sumo-gui'], 
                         capture_output=True, timeout=3)
    except:
        pass
'''
        
        # Write script
        with open("temp_simulation.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # Start simulation in background and track process
        process = subprocess.Popen([sys.executable, "temp_simulation.py"])
        session['simulation_process_id'] = process.pid
        session['simulation_start_time'] = datetime.now().isoformat()
        
        return jsonify({'status': 'success', 'message': 'Simulation started'})
        
    except Exception as e:
        session['simulation_running'] = False
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/check_simulation', methods=['GET'])
def check_simulation():
    """Check simulation status and results"""
    try:
        # If we already have a results file saved, treat as completed.
        existing_path = session.get('simulation_results_path')
        if existing_path and os.path.exists(existing_path):
            if session.get('simulation_running', False):
                session['simulation_running'] = False
                session.modified = True
            return jsonify({'status': 'completed'})
        
        # Check for results file
        if os.path.exists("temp_results.json"):
            try:
                # Move results to a persistent location (avoid storing in session cookie)
                _ensure_dir(WEB_RESULTS_DIR)
                dataset = session.get('current_dataset', SimulationConfig.DATASET)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_dir = os.path.join(WEB_RESULTS_DIR, dataset)
                _ensure_dir(dest_dir)
                current_strategy = session.get('selected_strategy', 'UNKNOWN')
                dest_path = os.path.join(dest_dir, f"{ts}_{current_strategy}.json")
                try:
                    os.replace("temp_results.json", dest_path)
                except Exception:
                    # Fallback: keep temp file and point to it
                    dest_path = os.path.abspath("temp_results.json")

                session['simulation_running'] = False
                session['simulation_results_path'] = dest_path
                
                # Save into all strategies' results
                strategy_names = {
                    'FIXED_TIME': '🕐 Fixed-time Control',
                    'ADAPTIVE': '🧠 Adaptive Control', 
                    'GA': '🧬 Genetic Algorithm (GA) baseline',
                    'PRESSLIGHT': '📌 PressLight-inspired (pressure DQN)',
                    'MAX_PRESSURE': '⚖️ MaxPressure Control',
                    'SOTL': '🚦 SOTL (Self-Organizing)',
                    'MARL_DQN': '🤝 Multi-Agent DQN (trained)'
                }
                strategy_display_name = strategy_names.get(current_strategy, current_strategy)
                
                # Save results with strategy info and timestamp
                if 'all_strategy_results' not in session:
                    session['all_strategy_results'] = {}
                
                session['all_strategy_results'][current_strategy] = {
                    'display_name': strategy_display_name,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'results_path': dest_path,
                    'dataset': dataset
                }
                session.modified = True  # Mark session as modified
                
                print(f"✅ Simulation completed! Results saved. Strategy: {current_strategy}")
                
                # Clean up simulation script only
                try:
                    if os.path.exists("temp_simulation.py"):
                        os.remove("temp_simulation.py")
                except:
                    pass
                
                # Don’t return full results payload (can be large); frontend will reload.
                return jsonify({'status': 'completed'})
                
            except Exception as e:
                print(f"Error reading results: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        # Check if simulation process is still running
        if session.get('simulation_running', False):
            # Check for timeout (30 minutes max)
            start_time_str = session.get('simulation_start_time')
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 1800:  # 30 minutes timeout
                        session['simulation_running'] = False
                        session['simulation_results_path'] = None
                        session.modified = True
                        # Clean up stuck files
                        for temp_file in ["temp_simulation.py", "temp_results.json"]:
                            if os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                except:
                                    pass
                        # Force close SUMO GUI
                        try:
                            if sys.platform == 'win32':
                                subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                                             capture_output=True, timeout=3)
                            else:
                                subprocess.run(['pkill', '-f', 'sumo-gui'], 
                                             capture_output=True, timeout=3)
                        except:
                            pass
                        return jsonify({
                            'status': 'error',
                            'message': 'Simulation timed out after 30 minutes. Please try again.'
                        })
                except Exception:
                    pass
            
            # Check if process is actually still running
            process_id = session.get('simulation_process_id')
            process_still_running = False
            
            if process_id:
                try:
                    # Check if process exists (Windows)
                    if sys.platform == 'win32':
                        result = subprocess.run(['tasklist', '/FI', f'PID eq {process_id}'], 
                                              capture_output=True, text=True, timeout=2)
                        process_still_running = str(process_id) in result.stdout
                    else:
                        # Linux/Mac: try to send signal 0 (doesn't kill, just checks)
                        try:
                            os.kill(process_id, 0)
                            process_still_running = True
                        except OSError:
                            process_still_running = False
                except Exception as e:
                    print(f"Error checking process: {e}")
                    # Assume still running if we can't check
                    process_still_running = True
            
            if process_still_running:
                return jsonify({'status': 'running'})
            else:
                # Process has ended - check for results immediately
                # Wait a moment for file to be fully written
                time.sleep(0.2)
                
                if os.path.exists("temp_results.json"):
                    # Let the main temp_results.json handler above process it
                    return jsonify({'status': 'running'})
                else:
                    # Process ended but no results - simulation failed
                    session['simulation_running'] = False
                    session['simulation_results_path'] = None
                    session.modified = True
                    
                    # Force close SUMO GUI if still open
                    try:
                        if sys.platform == 'win32':
                            subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                                         capture_output=True, timeout=3)
                        else:
                            subprocess.run(['pkill', '-f', 'sumo-gui'], 
                                         capture_output=True, timeout=3)
                    except:
                        pass
                    
                    # Clean up
                    for temp_file in ["temp_simulation.py", "temp_results.json"]:
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                    
                    return jsonify({
                        'status': 'error',
                        'message': 'Simulation ended but no results were saved. Check console for errors.'
                    })
        else:
            # No results file and not running - check session state
            if not session.get('simulation_running', False):
                return jsonify({'status': 'idle'})
            else:
                # Session says running but no process and no results - might be stuck
                # Give it a moment and check again
                return jsonify({'status': 'running'})
            
    except Exception as e:
        print(f"Error in check_simulation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear simulation history"""
    session['all_strategy_results'] = {}
    session['simulation_results_path'] = None
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/reset_simulation', methods=['POST'])
def reset_simulation():
    """Reset stuck simulation state and kill running processes"""
    try:
        # Kill the Python simulation process if it exists
        process_id = session.get('simulation_process_id')
        if process_id:
            try:
                if sys.platform == 'win32':
                    # Windows: Use taskkill
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process_id)], 
                                 capture_output=True, timeout=5)
                else:
                    # Linux/Mac: Use kill
                    try:
                        os.kill(process_id, signal.SIGTERM)
                        time.sleep(1)
                        # Force kill if still running
                        try:
                            os.kill(process_id, signal.SIGKILL)
                        except:
                            pass
                    except ProcessLookupError:
                        pass  # Process already dead
            except Exception as e:
                print(f"Error killing process {process_id}: {e}")
        
        # Kill SUMO GUI processes (sumo-gui.exe on Windows)
        try:
            if sys.platform == 'win32':
                # Kill all sumo-gui.exe processes
                subprocess.run(['taskkill', '/F', '/IM', 'sumo-gui.exe'], 
                             capture_output=True, timeout=5)
                # Also kill sumo.exe just in case
                subprocess.run(['taskkill', '/F', '/IM', 'sumo.exe'], 
                             capture_output=True, timeout=5)
            else:
                # Linux/Mac: Use pkill
                subprocess.run(['pkill', '-f', 'sumo-gui'], 
                             capture_output=True, timeout=5)
                subprocess.run(['pkill', '-f', 'sumo'], 
                             capture_output=True, timeout=5)
        except Exception as e:
            print(f"Error killing SUMO processes: {e}")
            # Continue even if killing SUMO fails
        
    except Exception as e:
        print(f"Error in process termination: {e}")
    
    # Reset session state
    session['simulation_running'] = False
    session['simulation_results_path'] = None
    session['simulation_process_id'] = None
    session['simulation_start_time'] = None
    
    # Clean up temporary files
    for temp_file in ["temp_simulation.py", "temp_results.json"]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    session.modified = True
    return jsonify({'status': 'success', 'message': 'Simulation stopped and state reset'})

@app.route('/get_comparison_charts', methods=['GET'])
def get_comparison_charts():
    """Generate comparison charts"""
    all_results = session.get('all_strategy_results', {})
    
    if len(all_results) < 2:
        return jsonify({'status': 'insufficient_data', 'message': 'Need at least 2 strategies for comparison'})
    
    # Prepare comparison data
    comparison_data = {
        'Strategy': [],
        'Waiting Time': [],
        'Speed': [],
        'Throughput': [],
        'CO2_g': [],
        'Fuel_L': [],
        'AQL': [],
        'Pressure': []
    }
    
    for strategy_key, strategy_data in all_results.items():
        results_path = strategy_data.get('results_path')
        if not results_path or not os.path.exists(results_path):
            continue
        results_data = _safe_json_load(results_path) or {}
        comparison_data['Strategy'].append(strategy_data.get('display_name', strategy_key))
        comparison_data['Waiting Time'].append(results_data.get('avg_waiting_time', 0))
        comparison_data['Speed'].append(results_data.get('avg_speed', 0))
        comparison_data['Throughput'].append(results_data.get('throughput_per_hour', 0))
        # Emissions (totals are typically in mg and ml; convert for nicer plots)
        total_co2_mg = float(results_data.get('total_co2', 0) or 0)
        total_fuel_ml = float(results_data.get('total_fuel', 0) or 0)
        comparison_data['CO2_g'].append(total_co2_mg / 1000.0)     # mg -> g
        comparison_data['Fuel_L'].append(total_fuel_ml / 1000.0)   # ml -> L
        comparison_data['AQL'].append(float(results_data.get('avg_queue_length', 0) or 0))
        comparison_data['Pressure'].append(float(results_data.get('avg_pressure', 0) or 0))
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Debug: print the data to verify
    print(f"Comparison data: {comparison_data}")
    print(f"DataFrame:\n{df_comparison}")
    
    # Get actual values for proper scaling
    waiting_times = df_comparison['Waiting Time'].tolist()
    throughputs = df_comparison['Throughput'].tolist()
    
    # Create charts using graph_objects for better control
    # For waiting time chart
    waiting_text = [f"{val:.2f}s" for val in waiting_times]
    strategies = df_comparison['Strategy'].tolist()
    
    # Create bar trace with explicit values
    fig_waiting = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=waiting_times,
            text=waiting_text,
            textposition='outside',
            textfont=dict(size=12, color='black'),
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb', '#f5576c'][:len(strategies)],
                line=dict(color='rgba(0,0,0,0.5)', width=1)
            ),
            name='Waiting Time'
        )
    ])
    
    # Set proper Y-axis range
    max_wait = max(waiting_times) if waiting_times else 1
    min_wait = min(waiting_times) if waiting_times else 0
    y_range_wait = [min(0, min_wait * 0.9), max_wait * 1.2] if max_wait > 0 else [0, 1]
    
    fig_waiting.update_layout(
        title="Average waiting time comparison (lower is better)",
        xaxis_title="Strategy",
        yaxis_title="Waiting Time (seconds)",
        yaxis=dict(range=y_range_wait),
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # For throughput chart
    throughput_text = [f"{int(val)}" for val in throughputs]
    
    fig_throughput = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=throughputs,
            text=throughput_text,
            textposition='outside',
            textfont=dict(size=12, color='black'),
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb', '#f5576c'][:len(strategies)],
                line=dict(color='rgba(0,0,0,0.5)', width=1)
            ),
            name='Throughput'
        )
    ])
    
    # Set proper Y-axis range
    max_throughput = max(throughputs) if throughputs else 1
    min_throughput = min(throughputs) if throughputs else 0
    y_range_throughput = [min(0, min_throughput * 0.9), max_throughput * 1.1] if max_throughput > 0 else [0, 1]
    
    fig_throughput.update_layout(
        title="System throughput comparison (higher is better)",
        xaxis_title="Strategy",
        yaxis_title="Throughput (vehicles/hour)",
        yaxis=dict(range=y_range_throughput),
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    waiting_chart = json.dumps(fig_waiting, cls=plotly.utils.PlotlyJSONEncoder)
    throughput_chart = json.dumps(fig_throughput, cls=plotly.utils.PlotlyJSONEncoder)

    # --- Emissions chart (CO2 + Fuel) ---
    co2_g = df_comparison['CO2_g'].tolist()
    fuel_l = df_comparison['Fuel_L'].tolist()

    fig_emissions = go.Figure()
    fig_emissions.add_trace(go.Bar(
        x=strategies,
        y=co2_g,
        name='Total CO₂ (g)',
        marker=dict(color='#2b6cb0'),
        text=[f"{v:.1f}g" for v in co2_g],
        textposition='outside'
    ))
    fig_emissions.add_trace(go.Bar(
        x=strategies,
        y=fuel_l,
        name='Total Fuel (L)',
        marker=dict(color='#c05621'),
        text=[f"{v:.2f}L" for v in fuel_l],
        textposition='outside',
        yaxis='y2'
    ))

    max_co2 = max(co2_g) if co2_g else 1
    max_fuel = max(fuel_l) if fuel_l else 1
    fig_emissions.update_layout(
        title="Emissions comparison (lower is better)",
        xaxis_title="Strategy",
        yaxis=dict(title="CO₂ (g)", range=[0, max_co2 * 1.2 if max_co2 > 0 else 1]),
        yaxis2=dict(
            title="Fuel (L)",
            overlaying='y',
            side='right',
            range=[0, max_fuel * 1.2 if max_fuel > 0 else 1]
        ),
        barmode='group',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    emissions_chart = json.dumps(fig_emissions, cls=plotly.utils.PlotlyJSONEncoder)

    # --- AQL chart ---
    aql_vals = df_comparison['AQL'].tolist()
    fig_aql = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=aql_vals,
            text=[f"{v:.2f}" for v in aql_vals],
            textposition='outside',
            marker=dict(color='#38a169'),
            name='AQL'
        )
    ])
    max_aql = max(aql_vals) if aql_vals else 1
    fig_aql.update_layout(
        title="Average queue length (AQL) comparison (lower is better)",
        xaxis_title="Strategy",
        yaxis_title="AQL (vehicles)",
        yaxis=dict(range=[0, max_aql * 1.2 if max_aql > 0 else 1]),
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    aql_chart = json.dumps(fig_aql, cls=plotly.utils.PlotlyJSONEncoder)

    # --- Pressure chart ---
    p_vals = df_comparison['Pressure'].tolist()
    fig_pressure = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=p_vals,
            text=[f"{v:.2f}" for v in p_vals],
            textposition='outside',
            marker=dict(color='#805ad5'),
            name='Pressure'
        )
    ])
    max_p = max(p_vals) if p_vals else 1
    fig_pressure.update_layout(
        title="Average pressure comparison (lower is better)",
        xaxis_title="Strategy",
        yaxis_title="Pressure (weighted queued vehicles)",
        yaxis=dict(range=[0, max_p * 1.2 if max_p > 0 else 1]),
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    pressure_chart = json.dumps(fig_pressure, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Convert comparison_data to ensure JSON serializable format
    # Convert numpy types to native Python types
    serializable_data = {
        'Strategy': comparison_data['Strategy'],
        'Waiting Time': [float(x) for x in comparison_data['Waiting Time']],
        'Speed': [float(x) for x in comparison_data['Speed']],
        'Throughput': [float(x) for x in comparison_data['Throughput']],
        'CO2_g': [float(x) for x in comparison_data['CO2_g']],
        'Fuel_L': [float(x) for x in comparison_data['Fuel_L']],
        'AQL': [float(x) for x in comparison_data['AQL']],
        'Pressure': [float(x) for x in comparison_data['Pressure']]
    }
    
    return jsonify({
        'status': 'success',
        'waiting_chart': waiting_chart,
        'throughput_chart': throughput_chart,
        'emissions_chart': emissions_chart,
        'aql_chart': aql_chart,
        'pressure_chart': pressure_chart,
        'comparison_data': serializable_data
    })

@app.route('/academic_experiments')
def academic_experiments():
    """Academic experiments page"""
    init_session()
    current_config = SimulationConfig.get_current_config()
    
    return render_template('academic_experiments.html',
                         current_dataset=session.get('current_dataset', 'custom'),
                         dataset_description=current_config['description'])

@app.route('/start_academic_experiment', methods=['POST'])
def start_academic_experiment():
    """Start academic experiment"""
    data = request.get_json()
    
    # Get experiment parameters
    dataset = data.get('dataset', 'custom')
    strategies = data.get('strategies', ['FIXED_TIME', 'ADAPTIVE', 'MAX_PRESSURE', 'DQN'])
    num_runs = int(data.get('num_runs', 10))
    duration = int(data.get('duration', 600))
    experiment_name = data.get('experiment_name', 'traffic_signal_control_comparison')
    random_seed = data.get('random_seed', 42)
    
    if session.get('academic_experiment_running', False):
        return jsonify({'status': 'error', 'message': 'Academic experiment already running'})
    
    try:
        # Create experiment config
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            dataset=dataset,
            strategies=strategies,
            num_runs=num_runs,
            duration=duration,
            random_seed=random_seed,
            traffic_flow_level="moderate",
            notes="Academic experiment with statistical analysis"
        )
        
        # Store in session (convert dataclass to dict)
        from dataclasses import asdict
        results_file = f"temp_academic_results_{experiment_name.replace(' ', '_')}.json"
        
        # Clean up old results file
        if os.path.exists(results_file):
            try:
                os.remove(results_file)
            except:
                pass
        
        session['academic_experiment_running'] = True
        session['academic_experiment_file'] = results_file
        session['academic_experiment_config'] = asdict(experiment_config)
        session.modified = True
        
        # Start experiment in background thread
        # Use file-based storage for results (sessions don't work well in background threads)
        
        def run_experiment():
            try:
                # Set dataset
                SimulationConfig.set_dataset(dataset)
                
                # Factory function for simulator
                def create_simulator_and_run(strategy: str, duration: int):
                    simulator = TrafficSimulator(use_gui=False, dataset=dataset)
                    metrics = simulator.run_simulation(duration=duration, strategy=strategy)
                    simulator.close_simulation()
                    return metrics
                
                # Run batch experiments
                all_results = run_batch_experiments(experiment_config, create_simulator_and_run)
                
                # Analyze results
                evaluator = AcademicEvaluator()
                
                # Generate report
                report = evaluator.generate_academic_report(experiment_config, all_results)
                
                # Export results
                json_path = evaluator.export_results_json(experiment_config, all_results)
                
                # Store results in file (simplified for JSON serialization)
                from dataclasses import asdict
                results_data = {
                    'all_results_summary': {},
                    'report': report,
                    'json_path': json_path,
                    'config': asdict(experiment_config),
                    'status': 'completed'
                }
                
                # Create summary of results
                for strategy, result_list in all_results.items():
                    if result_list and len(result_list) > 0:
                        results_data['all_results_summary'][strategy] = {
                            'num_runs': len(result_list),
                            'sample_result': result_list[0] if result_list else {}
                        }
                
                # Write to file
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
                
            except Exception as e:
                print(f"Academic experiment error: {e}")
                import traceback
                traceback.print_exc()
                # Write error to file
                error_data = {'status': 'error', 'message': str(e)}
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2)
        
        # Start thread
        import threading
        thread = threading.Thread(target=run_experiment, daemon=True)
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Academic experiment started'})
        
    except Exception as e:
        session['academic_experiment_running'] = False
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/check_academic_experiment', methods=['GET'])
def check_academic_experiment():
    """Check academic experiment status"""
    running = session.get('academic_experiment_running', False)
    results_file = session.get('academic_experiment_file', None)
    
    if not running:
        return jsonify({'status': 'idle'})
    
    # Check if results file exists
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            if results_data.get('status') == 'completed':
                # Mark as not running in session
                session['academic_experiment_running'] = False
                session.modified = True
                
                return jsonify({
                    'status': 'completed',
                    'results': results_data
                })
            elif results_data.get('status') == 'error':
                session['academic_experiment_running'] = False
                session.modified = True
                return jsonify({
                    'status': 'error',
                    'message': results_data.get('message', 'Unknown error')
                })
        except Exception as e:
            print(f"Error reading results file: {e}")
    
    # Still running
    return jsonify({
        'status': 'running',
        'progress': {'message': 'Experiment in progress...'}
    })


@app.route('/start_marl_evaluation', methods=['POST'])
def start_marl_evaluation():
    """Start MARL evaluation in background."""
    init_session()
    data = request.get_json() or {}

    if session.get('marl_eval_running', False):
        return jsonify({'status': 'error', 'message': 'MARL evaluation already running'})

    dataset = data.get('dataset', SimulationConfig.DATASET)
    model_path = data.get('model_path', '')
    episodes = int(data.get('episodes', 5))
    duration = int(data.get('duration', 600))
    decision_interval = int(data.get('decision_interval', 5))
    ratio = float(data.get('controlled_lights_ratio', 0.2))
    lanes_per_tl = int(data.get('lanes_per_tl', 8))
    collect_emissions = bool(data.get('collect_emissions', False))
    calc_metrics = bool(data.get('calc_metrics', True))

    _ensure_dir(WEB_RESULTS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(WEB_RESULTS_DIR, f"marl_eval_{dataset}_{ts}.json")
    status_path = os.path.join(WEB_RESULTS_DIR, f"marl_eval_{dataset}_{ts}.status.json")

    def run_eval():
        try:
            cmd = [
                sys.executable,
                "evaluate_marl_osm.py",
                "--dataset", dataset,
                "--episodes", str(episodes),
                "--duration", str(duration),
                "--decision-interval", str(decision_interval),
                "--controlled-lights-ratio", str(ratio),
                "--lanes-per-tl", str(lanes_per_tl),
                "--out", results_path,
            ]
            if model_path:
                cmd.extend(["--model", model_path])
            if collect_emissions:
                cmd.append("--collect-emissions")
            if calc_metrics:
                cmd.append("--calc-metrics")

            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({"status": "running", "cmd": " ".join(cmd)}, f, indent=2)

            subprocess.run(cmd, check=True)
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({"status": "completed", "results_path": results_path}, f, indent=2)
        except Exception as e:
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({"status": "error", "message": str(e)}, f, indent=2)
        finally:
            session['marl_eval_running'] = False
            session.modified = True

    session['marl_eval_running'] = True
    session['marl_eval_results_path'] = results_path
    session['marl_eval_status_path'] = status_path
    session.modified = True

    thread = threading.Thread(target=run_eval, daemon=True)
    thread.start()

    return jsonify({'status': 'success', 'message': 'MARL evaluation started', 'status_path': status_path})


@app.route('/check_marl_evaluation', methods=['GET'])
def check_marl_evaluation():
    """Check MARL evaluation status."""
    init_session()
    status_path = session.get('marl_eval_status_path')
    if not status_path or not os.path.exists(status_path):
        return jsonify({'status': 'idle'})

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_marl_evaluation_results', methods=['GET'])
def get_marl_evaluation_results():
    """Fetch MARL evaluation JSON results."""
    init_session()
    path = request.args.get('path') or session.get('marl_eval_results_path')
    if not path or not os.path.exists(path):
        return jsonify({'status': 'error', 'message': 'Results file not found'})
    data = _safe_json_load(path)
    if data is None:
        return jsonify({'status': 'error', 'message': 'Failed to load results'})
    return jsonify({'status': 'success', 'data': data})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

