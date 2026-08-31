#!/usr/bin/env bash
# G0d: re-budget the two whole-entity access arms to dyad-coverage parity with
# the walks, then re-measure everything the budget change can move.
#
# Everything here is local read-and-compute on the 32-graph panel and the
# 745-instance regenerated benchmark.  No SLURM, no API call, no LLM.  The
# frozen walk artifacts are read only and never rewritten.
#
# Usage:
#   bash scripts/run_g0d_headroom.sh ladder     # budget search (~25 min)
#   bash scripts/run_g0d_headroom.sh sample     # panels + benchmark (~25 min)
#   bash scripts/run_g0d_headroom.sh report     # tables + docs/HEADROOM_G0D
#   bash scripts/run_g0d_headroom.sh all
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
OUT="$REPO/results/g0d_headroom_2026_09"
SHARDS="${G0D_SHARDS:-4}"

# The grid the adopted budgets were chosen from.  Both whole-entity samplers
# stop before the first entity that does not fit, so every rung of the grid is
# resolved exactly by one cumulative pass -- the grid width is nearly free.
GRID=800,1600,2000,2200,2300,2400,2500,2600,2700,2800,3000,3200,3600,4000
GRID=$GRID,4500,5000,6000,8000,9000,9200,9400,9600,9800,10000,10200,10500
GRID=$GRID,11000,12000

ladder() {
    mkdir -p "$OUT"
    # --verify-budget replays the production sampler at 2,500 and fails loudly
    # if the ladder and the sampler ever disagree.
    PYTHONPATH=src "$PYTHON_BIN" src/g0d_budget_ladder.py \
        --budgets "$GRID" --arms A,B --seeds 8 --verify-budget 2500 \
        --out "$OUT/budget_ladder.csv"
}

sample() {
    local preset shard
    # 16 seed slots on the panels: the empty-sample rule advances along the
    # seed sequence, so it needs indices beyond the eight it will accept.
    for preset in g0d_panel_a g0d_panel_b g0d_panel_b_10500 \
                  g0d_benchmark_a g0d_benchmark_b; do
        for ((shard=0; shard<SHARDS; shard++)); do
            PYTHONPATH=src "$PYTHON_BIN" src/run_nonwalk_screen.py \
                --preset "$preset" --num-shards "$SHARDS" --shard-id "$shard" &
        done
        wait
        echo "done: $preset"
    done
}

report() {
    PYTHONPATH=src "$PYTHON_BIN" src/report_g0d_headroom.py "$@"
}

case "${1:-all}" in
    ladder) ladder ;;
    sample) sample ;;
    report) shift; report "$@" ;;
    all)    ladder; sample; report ;;
    *)
        echo "usage: $0 {ladder|sample|report|all}" >&2
        exit 2
        ;;
esac
