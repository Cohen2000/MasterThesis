#!/usr/bin/env bash
# Submit the resumable Qwen noise-probe chain on the UC3 login node.
#
# 160 prompts per mode, 32 panel graphs, both arms. The escalation ladder is
# shorter than the OFAT's: at 32k the frozen suite left only a handful of
# think prompts open, and a truncated answer is scored as invalid rather than
# retried forever.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

MODE="${1:-nothink}"
SHARDS="${NOISE_QWEN_SHARDS:-8}"
JOB="slurm/llm_noise_qwen36.sbatch"
DRY_RUN="${NOISE_SUBMIT_DRY_RUN:-0}"

if [[ ! "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NOISE_QWEN_SHARDS must be a positive integer" >&2
    exit 2
fi

PRIMARY_PASSES="${NOISE_PRIMARY_PASSES:-1}"
N16_PASSES="${NOISE_N16_PASSES:-1}"
N32_PASSES="${NOISE_N32_PASSES:-1}"
T64_PASSES="${NOISE_T64_PASSES:-1}"
T128_PASSES="${NOISE_T128_PASSES:-1}"

submit_one() {
    local stage="$1" dependency="$2" array="$3"
    local dep_args=()
    [[ -n "$dependency" ]] && dep_args=(--dependency="afterany:$dependency")
    local command=(sbatch --parsable
        --job-name="noise_q36_${stage}"
        --array="$array"
        --export="ALL,STAGE=$stage,SHARDS=$SHARDS"
        "${dep_args[@]}" "$JOB")
    local raw
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY RUN:' >&2
        printf ' %q' "${command[@]}" >&2
        echo >&2
        raw=99999999
    else
        raw=$("${command[@]}")
    fi
    echo "${raw%%;*}"
}

submit_passes() {
    local stage="$1" passes="$2" dependency="$3" array="$4"
    local i job="$dependency"
    for ((i=1; i<=passes; i++)); do
        job=$(submit_one "$stage" "$job" "$array")
        echo "  $stage pass $i/$passes -> $job" >&2
    done
    echo "$job"
}

if [[ "$MODE" == "think" ]]; then
    # One pass at the full window instead of 32k -> 64k -> 128k.  Measured on
    # the OFAT: the ladder costs 1.9 GPU-hours per usable answer, a single pass
    # about 1.0, because every truncated rung is discarded in full.
    #
    # 160 prompts at a measured ~3600 s each is roughly 160 GPU-hours.  Sixteen
    # shards put that at ~10 h of wall clock, so the walltime is set well above
    # it: a think generation that gets killed at the limit is not resumable
    # work, it is a discarded hour.
    THINK_SHARDS="${NOISE_THINK_SHARDS:-16}"
    THINK_TIME="${NOISE_THINK_TIME:-20:00:00}"
    echo "Submitting Qwen think: 160 prompts, single pass at 126976 tokens"
    echo "  shards=$THINK_SHARDS  walltime=$THINK_TIME"
    # Sixteen think shards would otherwise compete with the eight the
    # non-thinking chain still needs for its escalation stages, and that chain
    # is the arm the argument actually rests on. NOISE_THINK_AFTER=<jobid>
    # queues think behind it instead.
    dep_args=()
    if [[ -n "${NOISE_THINK_AFTER:-}" ]]; then
        dep_args=(--dependency="afterany:$NOISE_THINK_AFTER")
        echo "  waits for: $NOISE_THINK_AFTER"
    fi
    command=(sbatch --parsable
        --job-name="noise_q36_tfull"
        --array="0-$((THINK_SHARDS - 1))"
        --time="$THINK_TIME"
        --export="ALL,STAGE=tfull,SHARDS=$THINK_SHARDS"
        "${dep_args[@]}" "$JOB")
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY RUN:' >&2; printf ' %q' "${command[@]}" >&2; echo >&2
        exit 0
    fi
    job=$("${command[@]}")
    echo "  submitted -> ${job%%;*}"
    echo "Monitor with: squeue --me"
    exit 0
fi

echo "Submitting Qwen noise probe: 160 prompts/mode, shards/mode=$SHARDS"

PRIMARY_LAST=$(submit_passes primary "$PRIMARY_PASSES" "" \
    "0-$((2 * SHARDS - 1))")

N16_LAST=$(submit_passes n16 "$N16_PASSES" "$PRIMARY_LAST" "0-$((SHARDS - 1))")
N32_LAST=$(submit_passes n32 "$N32_PASSES" "$N16_LAST" "0-$((SHARDS - 1))")
T64_LAST=$(submit_passes t64 "$T64_PASSES" "$PRIMARY_LAST" "0-$((SHARDS - 1))")
T128_LAST=$(submit_passes t128 "$T128_PASSES" "$T64_LAST" "0-$((SHARDS - 1))")

echo
echo "Submitted. Primary tail: $PRIMARY_LAST"
echo "Non-thinking final tail: $N32_LAST"
echo "Thinking final tail:     $T128_LAST"
echo "Monitor with: squeue --me"
