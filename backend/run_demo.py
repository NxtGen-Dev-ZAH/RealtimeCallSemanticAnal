#!/usr/bin/env python3
"""
Run the Call Analysis System Demo (command-line, no web server).
"""

import sys
import os

# Ensure both backend package root and src/ are on sys.path (same as run_web_app.py)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BACKEND_DIR, "src")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Run from backend dir so relative paths (e.g. demo_audio_*.wav, output/) work
os.chdir(BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv()

from call_analysis.demo import main as demo_main


if __name__ == "__main__":
    demo_main()
