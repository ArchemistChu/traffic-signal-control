#!/usr/bin/env python3
"""
Start Intelligent Traffic Light Control System
Flask Web Interface
"""

import os
import subprocess
import sys


def main():
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "1").lower() not in ("0", "false", "no")
    print("🚀 Starting Flask application...")
    print("📍 Local browser: http://localhost:5000")
    print("=" * 45)
    print("🎯 Interface highlights:")
    print("  • Select strategy → Click Start → Watch simulation → View results")
    print("  • Comparative evaluation: uploaded model vs baselines")
    print("    → Progress + SUMO output: web UI (live log) and output/web_results/.../worker.log")
    print("    → This console hides noisy status-poll requests (GET /check_...)")
    if not use_reloader:
        print("  • FLASK_USE_RELOADER=0 — auto-reload OFF (saving .py files won't restart mid-demo)")
    else:
        print("  • Tip: server auto-reloads when project .py files change (can interrupt long evals)")
    print("  • Multiple datasets (Cologne, Vancouver, Los Angeles)")
    print("  • SUMO GUI starts automatically")
    print("=" * 45)
    
    try:
        # Run Flask app
        from app_flask import app
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=use_reloader)
    except KeyboardInterrupt:
        print("\n⏹️ System stopped")
    except ImportError:
        print(f"❌ Failed to import Flask app")
        print("Please ensure Flask is installed: pip install flask")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 