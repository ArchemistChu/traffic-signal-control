import json
from pathlib import Path

# Check detailed metrics for one strategy
with open('output/eval_cologne_MAX_PRESSURE_ep120.json', 'r') as f:
    data = json.load(f)

print('Detailed analysis of MAX_PRESSURE strategy:')
print('=' * 50)

results = data['results'][:3]  # Check first 3 episodes

for i, result in enumerate(results):
    metrics = result.get('metrics', {})
    print(f'Episode {i+1}:')
    print(f'  Waiting time: {metrics.get("avg_waiting_time", "N/A")}')
    print(f'  Queue length: {metrics.get("avg_queue_length", "N/A")}')
    print(f'  Speed: {metrics.get("avg_speed", "N/A")}')
    print(f'  Throughput: {metrics.get("throughput_per_hour", "N/A")}')
    print()

# Check ranges
all_waiting = [r.get('metrics', {}).get('avg_waiting_time', 0) for r in data['results']]
all_speeds = [r.get('metrics', {}).get('avg_speed', 0) for r in data['results']]
all_queues = [r.get('metrics', {}).get('avg_queue_length', 0) for r in data['results']]

print('Range analysis:')
print(f'Waiting time range: {min(all_waiting):.1f} - {max(all_waiting):.1f} seconds')
print(f'Speed range: {min(all_speeds):.2f} - {max(all_speeds):.2f} m/s')
print(f'Queue length range: {min(all_queues):.1f} - {max(all_queues):.1f} vehicles')
print(f'Episodes with waiting > 200s: {sum(1 for w in all_waiting if w > 200)}')
print(f'Episodes with speed < 0.1 m/s: {sum(1 for s in all_speeds if s < 0.1)}')

print()
print('Checking MARL model performance:')
print('=' * 35)

# Check MARL model results
with open('output/eval_cologne_from_vancouver_regionaware_ep120_seeded_emissions.json', 'r') as f:
    marl_data = json.load(f)

marl_results = marl_data['results'][:3]
for i, result in enumerate(marl_results):
    metrics = result.get('metrics', {})
    print(f'MARL Episode {i+1}:')
    print(f'  Waiting time: {metrics.get("avg_waiting_time", "N/A")}')
    print(f'  Speed: {metrics.get("avg_speed", "N/A")}')
    print()

# Overall MARL stats
all_waiting_marl = [r.get('metrics', {}).get('avg_waiting_time', 0) for r in marl_data['results']]
all_speeds_marl = [r.get('metrics', {}).get('avg_speed', 0) for r in marl_data['results']]

print(f'MARL Avg waiting time: {sum(all_waiting_marl)/len(all_waiting_marl):.1f}s')
print(f'MARL Avg speed: {sum(all_speeds_marl)/len(all_speeds_marl):.2f} m/s')
print(f'MARL Episodes: {len(marl_data["results"])}')