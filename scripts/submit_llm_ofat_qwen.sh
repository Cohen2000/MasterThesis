#!/usr/bin/env bash
# Submit the resumable Qwen OFAT chain on the UC3 login node.
#
# The prompt file contains only the two missing cells: 72 generations per
# mode.  The old four cells are reused and are never submitted again.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

SHARDS="${OFAT_QWEN_SHARDS:-8}"
JOB="slurm/llm_ofat_qwen36.sbatch"
DRY_RUN="${OFAT_SUBMIT_DRY_RUN:-0}"

if [[ ! "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "OFAT_QWEN_SHARDS must be a positive integer" >&2
    exit 2
fi

PRIMARY_PASSES="${OFAT_PRIMARY_PASSES:-1}"
N16_PASSES="${OFAT_N16_PASSES:-1}"
N32_PASSES="${OFAT_N32_PASSES:-1}"
T64_PASSES="${OFAT_T64_PASSES:-2}"
T128_PASSES="${OFAT_T128_PASSES:-2}"

submit_one() {
    local stage="$1" dependency="$2" array="$3"
    local dep_args=()
    [[ -n "$dependency" ]] && dep_args=(--dependency="afterany:$dependency")
    local command=(sbatch --parsable
        --job-name="ofat_q36_${stage}"
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

echo "Submitting Qwen OFAT: 72 missing prompts/mode, shards/mode=$SHARDS"
echo "Passes: primary=$PRIMARY_PASSES n16=$N16_PASSES n32=$N32_PASSES "
echo "        t64=$T64_PASSES t128=$T128_PASSES"

PRIMARY_LAST=$(submit_passes primary "$PRIMARY_PASSES" "" \
    "0-$((2 * SHARDS - 1))")

# The non-thinking and thinking escalation ladders are independent after the
# primary pass, so submit them as two parallel dependency branches.
N16_LAST=$(submit_passes n16 "$N16_PASSES" "$PRIMARY_LAST" \
    "0-$((SHARDS - 1))")
N32_LAST=$(submit_passes n32 "$N32_PASSES" "$N16_LAST" \
    "0-$((SHARDS - 1))")
T64_LAST=$(submit_passes t64 "$T64_PASSES" "$PRIMARY_LAST" \
    "0-$((SHARDS - 1))")
T128_LAST=$(submit_passes t128 "$T128_PASSES" "$T64_LAST" \
    "0-$((SHARDS - 1))")

echo
echo "Submitted. Primary tail: $PRIMARY_LAST"
echo "Non-thinking final tail: $N32_LAST"
echo "Thinking final tail:     $T128_LAST"
echo "Monitor with: squeue --me"
echo "Accounting:  sacct -j $PRIMARY_LAST,$N32_LAST,$T128_LAST --format=JobID%24,State,Elapsed,MaxRSS"
