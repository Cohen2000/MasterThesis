#!/usr/bin/env python3
"""Stage 1 (offline): graphs, surrogates, budgets, observations, prompts, requests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
# First pipeline step, fully offline: load the raw graphs, build surrogates and synthetic
# graphs, compute sampling budgets, draw the observations and write the model prompts and
# request manifest. All logic is in src/study/prepare.py.
from study.prepare import run

if __name__ == '__main__':
    run()
