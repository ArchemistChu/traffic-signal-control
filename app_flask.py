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

# Initialize session variables
def init_session():
    if 'simulation_running' not in session:
        session['simulation_running'] = False
    if 'simulation_results' not in session:
        session['simulation_results'] = None
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

@app.route('/')
def index():
    """Main page"""
    try:
        init_session()
        current_config = SimulationConfig.get_current_config()
        
        strategy_options = {
            'FIXED_TIME': '🕐 Fixed-time Control',
            'ADAPTIVE': '🧠 Adaptive Control', 
            'DQN': '🤖 DQN Reinforcement Learning',
            'MAX_PRESSURE': '⚖️ MaxPressure Control'
        }
        
        # Ensure session is properly initialized
        if 'simulation_running' not in session:
            session['simulation_running'] = False
        if 'simulation_results' not in session:
            session['simulation_results'] = None
        if 'all_strategy_results' not in session:
            session['all_strategy_results'] = {}
        if 'selected_strategy' not in session:
            session['selected_strategy'] = 'FIXED_TIME'
        if 'current_dataset' not in session:
            session['current_dataset'] = SimulationConfig.DATASET
        
        # Check if simulation is actually still running (not just stuck in session)
        if session.get('simulation_running', False):
            # Check if results file exists (simulation completed)
            if os.path.exists("temp_results.json"):
                # Simulation completed, update session
                try:
                    with open("temp_results.json", "r", encoding="utf-8") as f:
                        results = json.load(f)
                    session['simulation_results'] = results
                    session['simulation_running'] = False
                    session.modified = True
                except:
                    pass
            # Check if simulation script still exists
            elif not os.path.exists("temp_simulation.py"):
                # Script deleted but no results - simulation crashed or completed
                # Check timeout
                start_time_str = session.get('simulation_start_time')
                if start_time_str:
                    try:
                        start_time = datetime.fromisoformat(start_time_str)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed > 5:  # At least 5 seconds passed, likely crashed
                            session['simulation_running'] = False
                            session.modified = True
                    except:
                        session['simulation_running'] = False
                        session.modified = True
                else:
                    # No start time recorded, likely stale state
                    session['simulation_running'] = False
                    session.modified = True
        
        session.modified = True
        
        return render_template('index.html',
                             current_dataset=session.get('current_dataset', 'custom'),
                             dataset_description=current_config['description'],
                             strategy_options=strategy_options,
                             selected_strategy=session.get('selected_strategy', 'FIXED_TIME'),
                             simulation_running=session.get('simulation_running', False),
                             simulation_results=session.get('simulation_results'),
                             all_strategy_results=session.get('all_strategy_results', {}))
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
        session['simulation_results'] = None
        session['all_strategy_results'] = {}
        SimulationConfig.set_dataset(dataset_option)
        session['current_dataset'] = dataset_option
        session.modified = True
    
    return jsonify({'status': 'success', 'dataset': dataset_option})

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Start simulation"""
    data = request.get_json()
    strategy = data.get('strategy', 'FIXED_TIME')
    
    if session.get('simulation_running', False):
        return jsonify({'status': 'error', 'message': 'Simulation already running'})
    
    try:
        # Clean up old temporary files
        for temp_file in ["temp_results.json", "temp_simulation.py"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
        
        # Get current dataset
        dataset = session.get('current_dataset', SimulationConfig.DATASET)
        session['selected_strategy'] = strategy
        session['simulation_running'] = True
        session['simulation_results'] = None
        session.modified = True
        
        # Create simulation script
        script_content = f'''import sys
import os
sys.path.append(".")
from src.traffic_simulator import TrafficSimulator
from src.config import SimulationConfig

os.chdir(".")

print("Starting SUMO GUI simulation...")
print("Strategy: {strategy}")
print("Dataset: {dataset}")

SimulationConfig.set_dataset("{dataset}")

simulator = TrafficSimulator(use_gui=True, dataset="{dataset}")

duration = 600
print(f"Running simulation for {{duration}} seconds...")

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
        if os.path.exists("temp_results.json"):
            try:
                with open("temp_results.json", "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                session['simulation_results'] = results
                session['simulation_running'] = False
                
                # Save into all strategies' results
                current_strategy = session.get('selected_strategy', 'UNKNOWN')
                strategy_names = {
                    'FIXED_TIME': '🕐 Fixed-time Control',
                    'ADAPTIVE': '🧠 Adaptive Control', 
                    'DQN': '🤖 DQN Reinforcement Learning',
                    'MAX_PRESSURE': '⚖️ MaxPressure Control'
                }
                strategy_display_name = strategy_names.get(current_strategy, current_strategy)
                
                # Save results with strategy info and timestamp
                if 'all_strategy_results' not in session:
                    session['all_strategy_results'] = {}
                
                session['all_strategy_results'][current_strategy] = {
                    'display_name': strategy_display_name,
                    'results': results,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                session.modified = True  # Mark session as modified
                
                # Clean up temporary files
                try:
                    os.remove("temp_results.json")
                    if os.path.exists("temp_simulation.py"):
                        os.remove("temp_simulation.py")
                except:
                    pass
                
                return jsonify({
                    'status': 'completed',
                    'results': results,
                    'strategy': current_strategy
                })
                
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
                        session['simulation_results'] = None
                        session.modified = True
                        # Clean up stuck files
                        for temp_file in ["temp_simulation.py", "temp_results.json"]:
                            if os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                except:
                                    pass
                        return jsonify({
                            'status': 'error',
                            'message': 'Simulation timed out after 30 minutes. Please try again.'
                        })
                except Exception:
                    pass
            
            # Check if temp_simulation.py still exists (it gets deleted when simulation completes)
            if os.path.exists("temp_simulation.py"):
                return jsonify({'status': 'running'})
            else:
                # Script file deleted but no results - simulation might have crashed
                # Wait a bit more in case results are still being written
                if not os.path.exists("temp_results.json"):
                    # Give it 5 more seconds
                    if start_time_str:
                        try:
                            start_time = datetime.fromisoformat(start_time_str)
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed > 5:  # At least 5 seconds have passed
                                session['simulation_running'] = False
                                session.modified = True
                                return jsonify({
                                    'status': 'error',
                                    'message': 'Simulation process ended without creating results. Please try again.'
                                })
                        except:
                            pass
                return jsonify({'status': 'running'})  # Still waiting
        else:
            return jsonify({'status': 'idle'})
            
    except Exception as e:
        print(f"Error in check_simulation: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear simulation history"""
    session['all_strategy_results'] = {}
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
    session['simulation_results'] = None
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
        'Throughput': []
    }
    
    for strategy_key, strategy_data in all_results.items():
        results_data = strategy_data['results']
        comparison_data['Strategy'].append(strategy_data['display_name'])
        comparison_data['Waiting Time'].append(results_data.get('avg_waiting_time', 0))
        comparison_data['Speed'].append(results_data.get('avg_speed', 0))
        comparison_data['Throughput'].append(results_data.get('throughput_per_hour', 0))
    
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
    
    # Convert comparison_data to ensure JSON serializable format
    # Convert numpy types to native Python types
    serializable_data = {
        'Strategy': comparison_data['Strategy'],
        'Waiting Time': [float(x) for x in comparison_data['Waiting Time']],
        'Speed': [float(x) for x in comparison_data['Speed']],
        'Throughput': [float(x) for x in comparison_data['Throughput']]
    }
    
    return jsonify({
        'status': 'success',
        'waiting_chart': waiting_chart,
        'throughput_chart': throughput_chart,
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

