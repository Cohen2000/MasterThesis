#!/usr/bin/env bash
# Run the disclosed_examples / mask cell for Codex, paired to the existing
# disclosed / mask answers.
#
# Why this cell exists: every model that has a worked-examples condition sits
# in the read-off regime, where the examples improve the level and degrade the
# ranking. Codex and Claude Code are the only runs that demonstrably estimate
# rather than read off, and neither has an examples cell -- so it is unknown
# whether the degradation is a property of the examples or of the models.
#
# The case set comes from the EXISTING disclosed/mask answer file, not from the
# prompt list, so the two cells are paired by construction; a different case
# set would confound the contrast with case difficulty. src/select_paired_cell.py
# does the selection and subsamples stratified over strategy and coverage band.
#
# --prefer-from points at the Claude Code answers, whose 28 cases are a subset
# of these 56. The two screens therefore draw nested subsets and stay
# comparable to each other, not only each to its own baseline.
#
# The output file carries no case count: raising CODEX_EXAMPLES_LIMIT later
# resumes into the same file and simply extends the cell, since the treatment
# is unchanged and resume is by prompt_id.
#
# Usage:
#   bash scripts/codex_screen/run_examples_cell.sh notools
#   bash scripts/codex_screen/run_examples_cell.sh tools
#   CODEX_EXAMPLES_LIMIT=56 bash scripts/codex_screen/run_examples_cell.sh notools
#   CODEX_EXAMPLES_LIMIT=56 bash scripts/codex_screen/run_examples_cell.sh tools
#   CODEX_EXAMPLES_VERIFY=1 bash scripts/codex_screen/run_examples_cell.sh notools
#   CODEX_EXAMPLES_DRY_RUN=1 bash scripts/codex_screen/run_examples_cell.sh tools

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
RUNNER="$REPO/scripts/codex_screen/run_codex_screen.py"
SELECTOR="$REPO/src/select_paired_cell.py"
OUT_DIR="${CODEX_SCREEN_OUT_DIR:-$HOME/Dokumente/codex_screen}"
CC_OUT_DIR="${CC_SCREEN_OUT_DIR:-$HOME/Dokumente/cc_screen}"
# 30 of the 56 available cases: enough to resolve a collapse of the ranking,
# which is the binary question, at roughly half the token bill.
LIMIT="${CODEX_EXAMPLES_LIMIT:-30}"

BASE_LIVE="$OUT_DIR/answers_codex-gpt-5.6-sol_${ARM}_high.jsonl"
BASE_SNAP="$REPO/results/codex_screen_snapshot/answers_codex-gpt-5.6-sol_${ARM}_high.jsonl"
CC_LIVE="$CC_OUT_DIR/answers_claude-code-opus_${ARM}.jsonl"
CC_SNAP="$REPO/results/cc_screen_snapshot/answers_claude-code-opus_${ARM}.jsonl"
OUT="$OUT_DIR/answers_codex-gpt-5.6-sol_${ARM}_high_disclosed_examples_mask_paired.jsonl"
# The full 56-case cell cost 6.9M tokens; examples add ~1k tokens of prompt per
# call against ~99k total, so the cap only needs headroom over the subset.
CAP="${CODEX_EXAMPLES_TOKEN_CAP:-9000000}"

EXTRA_ARGS=()
if [[ "${CODEX_EXAMPLES_DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
# Free, local, no API call. Worth doing for this cell in particular: the
# worked examples carry real answer values, so the leakage check should be
# read rather than assumed.
if [[ "${CODEX_EXAMPLES_VERIFY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--verify)
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
PREFER="$CC_LIVE"
[[ -f "$PREFER" ]] || PREFER="$CC_SNAP"
PREFER_ARGS=()
[[ -f "$PREFER" ]] && PREFER_ARGS=(--prefer-from "$PREFER")
mkdir -p "$OUT_DIR"

IDS=$(PYTHONPATH="$REPO/src" "$PYTHON_BIN" "$SELECTOR" \
    --prompts "$PROMPTS" --cases "$CASES" --answers "$BASE" \
    --condition disclosed_examples --input-kind mask \
    --limit "$LIMIT" --min-cases 15 --report "${PREFER_ARGS[@]}")
N=$(awk -F, '{print NF}' <<<"$IDS")

echo "=== Codex disclosed_examples / mask, arm=$ARM ==="
echo "paired case set from : $BASE"
[[ -f "$PREFER" ]] && echo "nested with          : $PREFER"
echo "cases                : $N (limit $LIMIT)"
echo "out                  : $OUT"
echo "token cap            : $CAP"
echo

"$PYTHON_BIN" "$RUNNER" \
    --arm "$ARM" \
    --condition disclosed_examples \
    --input-kind mask \
    --ids "$IDS" \
    --out "$OUT" \
    --max-total-tokens "$CAP" \
    --wait-for-reset \
    --max-attempts 2 \
    "${EXTRA_ARGS[@]}"
