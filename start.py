#!/usr/bin/env python3
"""
Start Intelligent Traffic Light Control System
Simple Version, Avoid complex config
"""

import subprocess
import sys

def main():
    print("🚀 Starting now...")
    print("📍 Local browser will open: http://localhost:8501")
    print("=" * 45)
    print("🎯 New interface highlights:")
    print("  • Select strategy → Click Start → Watch simulation → View results")
    print("  • Single Start button, simple operation")
    print("  • Data presentation is clear and intuitive")
    print("  • SUMO GUI starts automatically")
    print("=" * 45)
    
    try:
        cmd = ['streamlit', 'run', 'app.py']
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ System stopped")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("Please ensure streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main() 