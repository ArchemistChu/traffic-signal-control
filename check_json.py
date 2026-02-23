import json
from pathlib import Path

OUT = Path('output')

SELECT_FILES = [
    'eval_cologne_from_vancouver_regionaware_ep120_seeded_emissions.json',
    'eval_cologne_FIXED_TIME_ep120.json',
    'eval_cologne_MAX_PRESSURE_ep120.json',
    'eval_cologne_ADAPTIVE_ep120.json',
    'eval_cologne_SOTL_ep120.json'
]

paths = []
for name in SELECT_FILES:
    p = OUT / name
    if p.exists():
        paths.append(p)
        print(f'Found: {p}')
    else:
        print(f'Missing: {p}')

for i, p in enumerate(paths):
    try:
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)
        print(f'Success: {p.name} ({len(data.get("results", []))} episodes)')
    except json.JSONDecodeError as e:
        print(f'JSON Error in {p.name}: line {e.lineno}, col {e.colno}: {e.msg}')
        with p.open('r', encoding='utf-8') as f:
            lines = f.readlines()
            if e.lineno <= len(lines):
                line = lines[e.lineno-1]
                start = max(0, e.colno-20)
                end = min(len(line), e.colno+20)
                print(f'Context: ...{line[start:end]}...')
        break
    except Exception as e:
        print(f'Other error in {p.name}: {e}')
        break