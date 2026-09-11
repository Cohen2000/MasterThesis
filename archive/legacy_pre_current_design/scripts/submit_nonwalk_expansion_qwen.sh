#!/usr/bin/env bash
# Submit the expanded non-walk screen for Qwen3.6-27B non-thinking.
#
# Only 448 of the 512 prompts go to the cluster: the expansion is nested, so
# the 64 answers from the original screen are reused rather than regenerated.
#
# Non-thinking only.  Thinking left 31 of 64 answers truncated at 32k in the
# original screen and would cost ~233 GPU-hours for the remainder; a screen
# whose invalid rate is that high measures where the model breaks, not how
# hard the access strategy is.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

SHARDS="${NW_QWEN_SHARDS:-8}"
JOB="slurm/nonwalk_qwen36_screen.sbatch"
DRY_RUN="${NW_SUBMIT_DRY_RUN:-0}"
PROMPTS="${NW_PROMPTS:-prompts_nonwalk_expansion_qwen.jsonl}"

dep_args=()
if [[ -n "${NW_AFTER:-}" ]]; then
    dep_args=(--dependency="afterany:$NW_AFTER")
    echo "waits for: $NW_AFTER"
fi

# Tasks 0..SHARDS-1 are all non-thinking, so the array stops there and no
# thinking task is even created.
command=(sbatch --parsable
    --job-name="nwexp_q36"
    --array="0-$((SHARDS - 1))"
    --export="ALL,SHARDS=$SHARDS,RUN_MODE=nothink,OUT_TAG=nwexp,PROMPTS_FILE=$PROMPTS"
    "${dep_args[@]}" "$JOB")

echo "Submitting expanded non-walk screen: 448 prompts, nothink, shards=$SHARDS"
if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY RUN:' >&2; printf ' %q' "${command[@]}" >&2; echo >&2
    exit 0
fi
job=$("${command[@]}")
echo "  submitted -> ${job%%;*}"
echo "Monitor with: squeue --me"
