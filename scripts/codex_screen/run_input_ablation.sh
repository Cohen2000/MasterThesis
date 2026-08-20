#!/usr/bin/env bash
# Run the frozen, paired 36-case Codex input-information ablation.
#
# Usage:
#   bash scripts/codex_screen/run_input_ablation.sh notools
#   bash scripts/codex_screen/run_input_ablation.sh tools
#   bash scripts/codex_screen/run_input_ablation.sh all

set -euo pipefail

ARM="${1:-all}"
if [[ "$ARM" != "notools" && "$ARM" != "tools" && "$ARM" != "all" ]]; then
    echo "usage: $0 {notools|tools|all}" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PROMPTS="$REPO/results/llm_v2/prompts.jsonl"
RUNNER="$REPO/scripts/codex_screen/run_codex_screen.py"
OUT_DIR="${CODEX_SCREEN_OUT_DIR:-$HOME/Dokumente/codex_screen}"
EXTRA_ARGS=()
if [[ "${CODEX_ABLATION_DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

prompt_ids_for_kind() {
    "$PYTHON_BIN" -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
case_ids = {
    row["case_id"] for row in rows
    if row["condition"] == "disclosed"
    and row["input_kind"] == "mask_crawl_full"
}
ids = sorted(
    row["prompt_id"] for row in rows
    if row["condition"] == "disclosed"
    and row["input_kind"] == sys.argv[2]
    and row["case_id"] in case_ids
)
if len(case_ids) != 36 or len(ids) != 36:
    raise SystemExit(
        f"expected 36 paired cases, got case_set={len(case_ids)}, ids={len(ids)}"
    )
print(",".join(ids))
' "$PROMPTS" "$1"
}

run_one() {
    local arm="$1"
    local kind="$2"
    local ids out cap

    ids=$(prompt_ids_for_kind "$kind")
    if [[ "$kind" == "mask" ]]; then
        out="$OUT_DIR/answers_codex-gpt-5.6-sol_${arm}_high.jsonl"
        if [[ "$arm" == "tools" ]]; then cap=11000000; else cap=8500000; fi
    else
        out="$OUT_DIR/answers_codex-gpt-5.6-sol_${arm}_high_disclosed_${kind}_ablation36.jsonl"
        if [[ "$arm" == "tools" ]]; then cap=9000000; else cap=6500000; fi
    fi
    # Resume override for an answer file whose measured cumulative usage has
    # already exceeded the conservative initial screening cap.
    cap="${CODEX_ABLATION_TOKEN_CAP:-$cap}"

    echo
    echo "=== arm=$arm input=$kind cases=36 ==="
    "$PYTHON_BIN" "$RUNNER" \
        --arm "$arm" \
        --condition disclosed \
        --input-kind "$kind" \
        --ids "$ids" \
        --out "$out" \
        --max-total-tokens "$cap" \
        --wait-for-reset \
        --max-attempts 2 \
        "${EXTRA_ARGS[@]}"
}

run_notools() {
    run_one notools mask
    run_one notools nw
    run_one notools mask_crawl_full
    run_one notools mask_crawl_temporal
    run_one notools mask_crawl_temporal_recent
}

run_tools() {
    run_one tools mask
    run_one tools mask_crawl_temporal
    run_one tools mask_crawl_temporal_recent
}

case "$ARM" in
    notools) run_notools ;;
    tools) run_tools ;;
    all)
        run_notools
        run_tools
        ;;
esac
