#!/usr/bin/env python3
"""Stages 2-4 (offline): synthetic pool, ExtraTrees folds, reference predictions.

usage: build_references.py pool|train|references
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import REFERENCES
from main_experiment.references import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pool', 'train', 'references'])
    run(REFERENCES, parser.parse_args().stage)
