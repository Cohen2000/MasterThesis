#!/usr/bin/env bash
# Run the disclosed_examples / mask cell for Claude Code, paired to the
# existing disclosed / mask answers.
#
# Sibling of scripts/codex_screen/run_examples_cell.sh and the same question:
# the worked-examples effect has only ever been measured on models that read
# the answer off the observed data. Claude Code is the run that most clearly
# does not, so whether the examples help or hurt it is genuinely open.
#
# The case set comes from the EXISTING disclosed/mask answer file, not from the
# prompt list, so the two cells are paired by construction. src/select_paired_cell.py
# subsamples stratified over strategy and coverage band, and its ordering is a
# prefix, so this subset nests inside the Codex one.
#
# The output file carries no case count: raising CC_EXAMPLES_LIMIT later
# resumes into the same file and extends the cell, since the treatment is
# unchanged and resume is by prompt_id.
#
# Usage:
#   bash scripts/cc_screen/run_examples_cell.sh notools
#   CC_EXAMPLES_LIMIT=30 bash scripts/cc_screen/run_examples_cell.sh notools
#   CC_EXAMPLES_DRY_RUN=1 bash scripts/cc_screen/run_examples_cell.sh notools

set -euo pipefail

ARM="${1:-notools}"
if [[ "$ARM" != "notools" && "$ARM" != "tools" ]]; then
    echo "usage: $0 {notools|tools}" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PROMPTS="$REPO/results/llm_v2/prompts.jsonl"
CASES="$REPO/results/llm_v2/llm_cases.csv"
RUNNER="$REPO/scripts/cc_screen/run_cc_screen.py"
SELECTOR="$REPO/src/select_paired_cell.py"
OUT_DIR="${CC_SCREEN_OUT_DIR:-$HOME/Dokumente/cc_screen}"
# Ask for the full 30-case screen. The selector automatically uses fewer when
# the paired baseline has fewer complete answers (currently 28 for notools).
LIMIT="${CC_EXAMPLES_LIMIT:-30}"

BASE_LIVE="$OUT_DIR/answers_claude-code-opus_${ARM}.jsonl"
BASE_SNAP="$REPO/results/cc_screen_snapshot/answers_claude-code-opus_${ARM}.jsonl"
OUT="$OUT_DIR/answers_claude-code-opus_${ARM}_disclosed_examples_mask_paired.jsonl"
# Cumulative caps for the two answer files.  They include all earlier runs;
# tools needs substantially more headroom because its prompts make many turns.
if [[ "$ARM" == "notools" ]]; then
    DEFAULT_CAP=50
else
    DEFAULT_CAP=70
fi
CAP="${CC_EXAMPLES_COST_CAP:-$DEFAULT_CAP}"
# Several remaining long-running cases reached the former 2 USD budget and
# exited without an answer.  Four dollars lets those finish while the
# cumulative per-arm caps above still bound the total run.
PER_PROMPT="${CC_EXAMPLES_PER_PROMPT_CAP:-4.0}"
# A pass tries every pending case once. Later passes contain only failures from
# earlier passes, so bad cases are retried after all fresh cases were attempted.
MAX_ATTEMPTS="${CC_EXAMPLES_MAX_ATTEMPTS:-5}"
RETRY_PASSES="${CC_EXAMPLES_RETRY_PASSES:-4}"
MAX_WAITS="${CC_EXAMPLES_MAX_WAITS:-24}"

EXTRA_ARGS=()
if [[ "${CC_EXAMPLES_DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi

BASE="$BASE_LIVE"
[[ -f "$BASE" ]] || BASE="$BASE_SNAP"
if [[ ! -f "$BASE" ]]; then
    echo "no existing disclosed/mask answers for arm=$ARM; looked at:" >&2
    echo "  $BASE_LIVE" >&2
    echo "  $BASE_SNAP" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

IDS=$(PYTHONPATH="$REPO/src" "$PYTHON_BIN" "$SELECTOR" \
    --prompts "$PROMPTS" --cases "$CASES" --answers "$BASE" \
    --condition disclosed_examples --input-kind mask \
    --limit "$LIMIT" --min-cases 15 --report)
N=$(awk -F, '{print NF}' <<<"$IDS")

echo "=== Claude Code disclosed_examples / mask, arm=$ARM ==="
echo "paired case set from : $BASE"
echo "cases                : $N (limit $LIMIT)"
echo "out                  : $OUT"
echo "cost cap             : ${CAP} USD total, ${PER_PROMPT} USD per prompt"
echo "retry policy         : ${RETRY_PASSES} passes, ${MAX_ATTEMPTS} real attempts/prompt"
echo "usage-limit policy   : wait for reported reset (${MAX_WAITS} waits maximum)"
echo

PASSES="$RETRY_PASSES"
if [[ "${CC_EXAMPLES_DRY_RUN:-0}" == "1" ]]; then
    PASSES=1
fi

for ((pass = 1; pass <= PASSES; pass++)); do
    echo "--- arm=$ARM pass $pass/$PASSES ---"
    "$PYTHON_BIN" "$RUNNER" \
        --arm "$ARM" \
        --condition disclosed_examples \
        --input-kind mask \
        --ids "$IDS" \
        --out "$OUT" \
        --max-cost-usd "$CAP" \
        --per-prompt-budget-usd "$PER_PROMPT" \
        --wait-for-reset \
        --max-waits "$MAX_WAITS" \
        --max-attempts "$MAX_ATTEMPTS" \
        --max-consecutive-errors 0 \
        "${EXTRA_ARGS[@]}"
done
