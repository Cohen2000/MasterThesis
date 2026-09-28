#!/usr/bin/env python3
"""Orchestrator of the v11 extension (ET replicates, four new real test sources).

  python scripts/v11_ext.py --dry-run       planned arrays, CPU/GPU hours and wall estimate
  python scripts/v11_ext.py --status        completed tasks, Qwen answers and queued jobs
  python scripts/v11_ext.py --submit        submit every missing task as chained SLURM arrays
  python scripts/v11_ext.py task --plan F --index I     (inside an array job)
  python scripts/v11_ext.py task --name NAME            (one task in-process, e.g. a smoke test)
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from v11_ext import dag  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', choices=('task',))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--replicates', type=int)
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--index', type=int)
    ap.add_argument('--name')
    a = ap.parse_args()
    if a.command == 'task':
        if a.name: dag.run_named(a.name, a.replicates)
        else: dag.run_planned(a.plan, a.index)
        return
    replicates, tasks, est = dag.plan(a.replicates)
    print(dag.describe(tasks, est, replicates))
    if a.status: print(dag.status())
    if a.submit: dag.submit(replicates, tasks)


if __name__ == '__main__':
    main()
