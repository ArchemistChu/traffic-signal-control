#!/usr/bin/env python3
"""
Start Intelligent Traffic Light Control System
Flask Web Interface
"""

import subprocess
import sys

def main():
    print("🚀 Starting Flask application...")
    print("📍 Local browser: http://localhost:5000")
    print("=" * 45)
    print("🎯 Interface highlights:")
    print("  • Select strategy → Click Start → Watch simulation → View results")
    print("  • Comparative evaluation: uploaded model vs baselines")
    print("  • Multiple datasets (Cologne, Vancouver, Los Angeles)")
    print("  • SUMO GUI starts automatically")
    print("=" * 45)
    
    try:
        # Run Flask app
        from app_flask import app
        app.run(debug=True, host='0.0.0.0', port=5000)
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