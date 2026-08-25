#!/usr/bin/env bash
# Codex arm of the reduced noise probe (160 prompts, 32 panel graphs).
#
# The completed Gemini and DeepSeek arms both have a calibration slope near
# zero on the frozen suite, so "redrawing the walk adds nothing" has so far
# only been measured on models that barely read the case. Codex is the one
# available run with a slope around 0.7, which makes it the arm that can
# actually falsify the result.
#
# The pinned 0.146.0 release is the same one that produced the OFAT and screen
# answers. Measuring response noise on a different CLI build would mix a
# harness change into the variance component being estimated.
#
# Resumable: complete prompt_ids are skipped, so an interrupted run continues
# where it stopped. Prompts are ordered graph-blocked across all 12 groups, so
# a partial run is still analysable.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PROBE="$REPO/results/llm_noise_probe"
PROMPTS="${NOISE_PROMPTS:-$PROBE/prompts_subset160.jsonl}"
OUT_DIR="${CODEX_NOISE_OUT_DIR:-$HOME/Dokumente/codex_noise}"
ANSWER="$OUT_DIR/answers_codex-gpt-5.6-sol_notools_high_noise.jsonl"
CODEX_BIN="${CODEX_NOISE_BIN:-$HOME/.codex/packages/standalone/releases/0.146.0-x86_64-unknown-linux-musl/bin/codex}"
# 160 prompts at the OFAT median of ~24.7k total tokens is about 4M. The cap is
# a runaway brake, not a budget target.
CAP="${CODEX_NOISE_TOKEN_CAP:-8000000}"

if [[ ! -f "$PROMPTS" ]]; then
    echo "prompt subset missing: $PROMPTS" >&2
    echo "build it with: $PYTHON_BIN src/make_noise_subset.py" >&2
    exit 2
fi
if [[ ! -x "$CODEX_BIN" ]]; then
    echo "Pinned Codex CLI 0.146.0 not found: $CODEX_BIN" >&2
    echo "Set CODEX_NOISE_BIN to the matching 0.146.0 executable." >&2
    exit 2
fi
if [[ "$("$CODEX_BIN" --version 2>/dev/null)" != "codex-cli 0.146.0" ]]; then
    echo "Refusing to mix Codex CLI versions; expected codex-cli 0.146.0" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
total=$(wc -l <"$PROMPTS")
echo "== Codex noise arm: $total prompts, pinned 0.146.0 =="
echo "   answers: $ANSWER"

"$PYTHON_BIN" scripts/codex_screen/run_codex_screen.py \
    --arm notools \
    --codex-bin "$CODEX_BIN" \
    --prompts "$PROMPTS" \
    --condition disclosed \
    --input-kind mask \
    --out "$ANSWER" \
    --max-total-tokens "$CAP" \
    --wait-for-reset \
    --max-attempts 2 \
    "$@"

# Copying is part of the run, not a separate step someone has to remember.
# The OFAT answers sat outside the repository for three days and silently
# evaluated as zero coverage because this line did not exist there.
cp "$ANSWER" "$PROBE/"
echo
echo "copied -> $PROBE/$(basename "$ANSWER")"
echo "status:  bash scripts/noise_subset_status.sh"
