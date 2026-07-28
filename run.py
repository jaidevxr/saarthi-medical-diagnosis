"""
Saarthi Medical Diagnosis AI — Master Entry Point
=================================================
Launches the Streamlit web application or executes pipeline verification.

Usage:
  python run.py --app      # Launches Streamlit Web Interface
  python run.py --test     # Runs 1,000 QA Test Suite
"""

import sys, os, subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def launch_app():
    app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
    if not os.path.exists(app_path):
        app_path = os.path.join(PROJECT_ROOT, "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"[LAUNCH] Starting Streamlit app: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_tests():
    test_script = os.path.join(PROJECT_ROOT, "scripts", "test_1000_qa_suite.py")
    if os.path.exists(test_script):
        cmd = [sys.executable, test_script]
        print(f"[TEST] Executing test suite: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print("[FAIL] Test script not found.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        launch_app()
