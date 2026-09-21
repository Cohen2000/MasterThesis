#!/usr/bin/env python3
"""Stage 1 (offline): graphs, surrogates, budgets, observations, prompts, requests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.prepare import run

if __name__ == '__main__':
    run()
