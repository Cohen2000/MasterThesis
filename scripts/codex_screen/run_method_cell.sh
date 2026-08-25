#!/usr/bin/env bash
# Run the frozen 24-prompt `method` condition through the Codex CLI.
#
# The method cell states an estimation procedure in the prompt instead of
# leaving the model to find one. Five API configurations answered it in the
# frozen suite; Codex never did, which leaves the one arm that reasons longest
# missing from the only cell that tells the model *how* to reason. Whether a
# stated method helps or constrains is exactly the question that needs the
# strongest arm present.
#
# All 24 prompts share input_kind `mask_crawl_temporal`, so condition plus
# input kind selects the cell exactly; the runner verifies the count.
#
# Usage:
#   bash scripts/codex_screen/run_method_cell.sh            # notools (default)
#   bash scripts/codex_screen/run_method_cell.sh tools
#   CODEX_METHOD_DRY_RUN=1 bash scripts/codex_screen/run_method_cell.sh
#
# Extra arguments are passed through to the runner, so a single stuck prompt
# can be retried with, for example, --ids <id> --timeout 2400.

set -euo pipefail

ARM="${1:-notools}"
if [[ "$ARM" == "notools" || "$ARM" == "tools" ]]; then
    shift || true
else
    ARM="notools"
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PROMPTS="$REPO/results/llm_v2/prompts.jsonl"
RUNNER="$REPO/scripts/codex_screen/run_codex_screen.py"
OUT_DIR="${CODEX_SCREEN_OUT_DIR:-$HOME/Dokumente/codex_screen}"
DEST="$REPO/results/codex_screen_snapshot"
OUT="$OUT_DIR/answers_codex-gpt-5.6-sol_${ARM}_high_method24.jsonl"

# 36 mask_crawl_temporal prompts ran under a 6.5M cap in the input ablation.
# 24 prompts of the same kind scale to roughly 4.3M; 5.5M leaves room for the
# longer reasoning a stated method tends to produce without removing the stop.
CAP="${CODEX_METHOD_TOKEN_CAP:-5500000}"

# Version pin. Every existing Codex arm -- the noise probe, the non-walk run,
# the input ablation -- was produced with 0.146.0. On 0.149.1 this cell failed
# all three attempts with "Code Mode is unavailable because code-mode host is
# disabled": the newer CLI reaches for code mode even though the notools arm
# disables it, and then fails closed. Pinning fixes that, and it is what makes
# the cell comparable to the arms it is meant to sit beside. Running a paired
# cell on a different harness version measures the version as much as the cell.
PINNED="${CODEX_METHOD_BIN:-$HOME/.codex/packages/standalone/releases/0.146.0-x86_64-unknown-linux-musl/bin/codex}"

EXTRA=()
[[ "${CODEX_METHOD_DRY_RUN:-0}" == "1" ]] && EXTRA+=(--dry-run)
if [[ -x "$PINNED" ]]; then
    EXTRA+=(--codex-bin "$PINNED")
    echo "codex binary pinned: $("$PINNED" --version 2>&1 | head -1)"
else
    echo "WARNING: pinned Codex 0.146.0 not found at $PINNED." >&2
    echo "         Falling back to whatever is on PATH; on 0.149.x the notools" >&2
    echo "         arm fails closed on code mode, and the result would not be" >&2
    echo "         comparable to the other Codex cells either." >&2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -s "$PROMPTS" ]]; then
    echo "frozen prompts not found: $PROMPTS" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

expected=$("$PYTHON_BIN" -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = [r["prompt_id"] for r in rows
       if r["condition"] == "method" and r["input_kind"] == "mask_crawl_temporal"]
if len(ids) != 24:
    raise SystemExit(f"expected 24 method prompts, found {len(ids)}")
print(len(ids))
' "$PROMPTS")

echo "=== Codex method cell: arm=$ARM prompts=$expected cap=$CAP ==="
echo "out: $OUT"

"$PYTHON_BIN" "$RUNNER" \
    --arm "$ARM" \
    --condition method \
    --input-kind mask_crawl_temporal \
    --out "$OUT" \
    --max-total-tokens "$CAP" \
    --wait-for-reset \
    --max-attempts 2 \
    "${EXTRA[@]}" \
    "$@"

# The snapshot directory is what the evaluators glob; leaving the only copy in
# a home directory outside the repository is how a finished run goes missing.
if [[ -s "$OUT" && "${CODEX_METHOD_DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$DEST"
    cp -f "$OUT" "$DEST/"
    echo "copied -> $DEST/$(basename "$OUT")"
fi
