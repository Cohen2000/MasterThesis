#!/usr/bin/env bash
# Walk the final target panel at nested budgets and report coverage/estimators.
#
# Standard strategies draw one trajectory at the largest budget and truncate it
# for every smaller one, so the whole ladder costs a single pass and the
# budgets are exact prefixes of each other. On the 32-graph panel this takes
# well under a minute on a laptop -- worth measuring before reaching for SLURM.
#
# The walk plan is written to a temporary config derived from the committed
# preset, so config/ stays untouched and cannot drift from this probe.
#
#   bash scripts/run_panel_budget_probe.sh
#   BUDGETS="[400, 800]" OUT_DIR=results/my_probe bash scripts/run_panel_budget_probe.sh
#   SEEDS=8 BUDGETS="[800]" OUT_DIR=results/panel_seed_probe bash scripts/run_panel_budget_probe.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CONFIG="${BASE_CONFIG:-config/benchmark.yaml}"
BASE_PRESET="${BASE_PRESET:-v2}"
MANIFEST="${MANIFEST:-results/final_target_panel/panel32_final.csv}"
OUT_DIR="${OUT_DIR:-results/panel_budget_probe}"
BUDGETS="${BUDGETS:-[200, 400, 800, 1600, 3200]}"
STRATEGIES="${STRATEGIES:-[time_agnostic_t, time_respecting, recent_history_k20]}"
DETAIL_BUDGET="${DETAIL_BUDGET:-800}"
SEEDS="${SEEDS:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"

[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
mkdir -p "$OUT_DIR"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CONFIG="$WORK/panel_budget.yaml"
BASE_CONFIG="$BASE_CONFIG" BASE_PRESET="$BASE_PRESET" CONFIG="$CONFIG" \
BUDGETS="$BUDGETS" STRATEGIES="$STRATEGIES" SEEDS="$SEEDS" "$PYTHON" - <<'PY'
import copy, os, yaml

cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"]))
preset = copy.deepcopy(cfg["presets"][os.environ["BASE_PRESET"]])
walks = preset.get("walks", {})
preset["walks"] = {
    "strategies": yaml.safe_load(os.environ["STRATEGIES"]),
    "budgets": yaml.safe_load(os.environ["BUDGETS"]),
    "seeds": int(os.environ["SEEDS"]),
    "recency_decay_scale": walks.get("recency_decay_scale", 0.10),
    "recent_events_json_limit": walks.get("recent_events_json_limit", 100),
    "recent_history_k": 20,
    # the sentinel only applies to the `time_agnostic` strategy, which this
    # probe does not run; keeping it at 0 makes that explicit
    "time_agnostic_sentinel_fraction": 0.0,
}
out = {"presets": {"panel_budget": preset}}
out.update({k: v for k, v in cfg.items() if k != "presets"})
with open(os.environ["CONFIG"], "w") as fh:
    yaml.safe_dump(out, fh, sort_keys=False)
print(f"walk plan: W={preset['W']} seed={preset['seed']} "
      f"walk_seeds={preset['walks']['seeds']} "
      f"strategies={preset['walks']['strategies']} "
      f"budgets={preset['walks']['budgets']}")
PY

echo "== walks =="
PYTHONPATH=src "$PYTHON" src/run_benchmark_walks.py \
    --config "$CONFIG" --preset panel_budget \
    --manifest "$MANIFEST" --out "$OUT_DIR/cases.csv.gz"

echo
echo "== report =="
PYTHONPATH=src "$PYTHON" src/report_panel_budget.py \
    --cases "$OUT_DIR/cases.csv.gz" --panel "$MANIFEST" \
    --budget "$DETAIL_BUDGET" | tee "$OUT_DIR/REPORT.txt"

echo
echo "wrote $OUT_DIR/cases.csv.gz and $OUT_DIR/REPORT.txt"
