#!/usr/bin/env python3
"""Orchestrator of pipeline stage 2 (stage-2 sources, surrogates, ET replicates, diagnostics, report).

  python production/run_pipeline.py --dry-run       planned arrays, CPU/GPU hours and wall estimate
  python production/run_pipeline.py --status        completed tasks, Qwen answers and queued jobs
  python production/run_pipeline.py --submit        submit every missing task as chained SLURM arrays
  python production/run_pipeline.py task --plan F --index I     (inside an array job)
  python production/run_pipeline.py task --name NAME            (one task in-process, e.g. a smoke test)
"""
# Command-line entry point for the cluster pipeline (bwUniCluster, SLURM).
# The real work lives in pipeline/task_graph.py: it builds a task graph (which step needs
# which other step), finds the tasks whose outputs are still missing and submits them
# as SLURM job arrays. Tasks that are already done are never recomputed.
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pipeline import task_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', choices=('task',))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--attach', action='store_true', help='with --submit: depend on still-queued v11x arrays')
    ap.add_argument('--replicates', type=int)
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--index', type=int)
    ap.add_argument('--name')
    a = ap.parse_args()
    # Inside a SLURM job: run exactly one task from a saved plan file (or one named task).
    if a.command == 'task':
        if a.name: task_graph.run_named(a.name, a.replicates)
        else: task_graph.run_planned(a.plan, a.index)
        return
    # On the login node: build the plan, print it, and optionally show the queue or submit.
    replicates, tasks, est = task_graph.plan(a.replicates)
    print(task_graph.describe(tasks, est, replicates))
    if a.status: print(task_graph.status())
    if a.submit: task_graph.submit(replicates, tasks, allow_queued=a.attach)


if __name__ == '__main__':
    main()
