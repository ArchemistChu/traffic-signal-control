# Flask Polling Issue - Analysis and Fix

## Problem
The Flask app is continuously polling `/check_simulation` every 3 seconds, showing logs like:
```
127.0.0.1 - - [27/Dec/2025 14:25:23] "GET /check_simulation HTTP/1.1" 200 -
127.0.0.1 - - [27/Dec/2025 14:25:26] "GET /check_simulation HTTP/1.1" 200 -
127.0.0.1 - - [27/Dec/2025 14:25:29] "GET /check_simulation HTTP/1.1" 200 -
```

## Root Cause
1. **Simulation process is stuck or crashed**: The Python process running the simulation may have:
   - Crashed without creating `temp_results.json`
   - Hung waiting for SUMO GUI interaction
   - Encountered an error that wasn't caught

2. **Session state never resets**: `session['simulation_running']` stays `True` even if the process died

3. **No timeout mechanism**: The frontend keeps polling indefinitely

4. **No process tracking**: The Flask app doesn't check if the actual Python process is still running

## Fixes Applied

### 1. Process Tracking
- Store `simulation_process_id` and `simulation_start_time` in session
- Track when simulation started for timeout detection

### 2. Timeout Mechanism
- Added 30-minute timeout
- If simulation runs longer than 30 minutes, automatically reset state
- Clean up stuck temporary files

### 3. Better Error Detection
- Check if `temp_simulation.py` still exists (it gets deleted on completion)
- If script is gone but no results, detect crash after 5 seconds
- Return error status to stop polling

### 4. Manual Reset Endpoint
- Added `/reset_simulation` endpoint to manually reset stuck state
- Cleans up temporary files
- Resets all simulation-related session variables

### 5. Frontend Error Handling
- Stop polling interval on error
- Show alert to user
- Reload page to reset state

## How to Use

### If Simulation Gets Stuck:
1. **Wait for timeout**: The system will automatically detect and reset after 30 minutes
2. **Manual reset**: Call `/reset_simulation` endpoint (can add button in UI)
3. **Check processes**: Look for stuck Python/SUMO processes and kill them manually

### Debugging:
- Check if `temp_simulation.py` exists (simulation still running)
- Check if `temp_results.json` exists (simulation completed)
- Check Python processes: `Get-Process python` (Windows) or `ps aux | grep python` (Linux)
- Check SUMO processes: `Get-Process sumo` (Windows)

## Next Steps (Optional Improvements)
1. Add a "Cancel Simulation" button in the UI
2. Use `psutil` library for better process tracking (optional dependency)
3. Add logging to track simulation progress
4. Add progress indicator showing elapsed time

