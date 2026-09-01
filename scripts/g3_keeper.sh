#!/usr/bin/env bash
# Keep the G3 runs moving without supervision.
#
# Two jobs. On the cluster, resubmit any Qwen generation that is neither
# complete nor in the queue -- shards are killed at their wall clock by design,
# and resume makes a resubmission cost one model load. Locally, start Step 2
# once Step 1 has both its generations, so Codex quota is never idle while a
# report is being written.
#
# Every action is printed; silence means nothing needed doing. Safe to run
# repeatedly: it only ever submits work that resume would skip anyway.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

CODEX_BIN="${CODEX_BIN:-$HOME/.codex/packages/standalone/releases/0.146.0-x86_64-unknown-linux-musl/bin/codex}"
STEP2_GENERATIONS="${STEP2_GENERATIONS:-1}"   # Codex is quota-bound and
                                              # reproduces itself at 0.07, so
                                              # one generation is the allocation
                                              # the evidence supports.

qwen_keeper() {
    local report
    report=$(timeout 90 ssh -o BatchMode=yes uc3 'bash -s' <<'REMOTE' 2>/dev/null
WS=$(ws_find llm_pilot); cd "$WS" || exit 1
for tag in think nothink; do
  # `short` must match the submitted job name: nothink jobs are g3nt*, not
  # g3no*, so deriving it from the first two letters of $tag is wrong.
  [ "$tag" = think ] && { total=976; shards=4; maxtok=98304; short=th; }                      || { total=736; shards=2; maxtok=24576; short=nt; }
  for g in 0 1 2; do
    have=$(cat llm_g3/answers_vllm_qwen36-27b_${tag}_g${g}.shard*.jsonl 2>/dev/null \
           | grep -c '"finish_reason":"stop"')
    running=$(squeue -u "$USER" -h -o "%j" | grep -c "^g3${short}.*_g${g}$")
    echo "$tag $g $have $total $running $shards $maxtok $short"
  done
done
REMOTE
)
    [ -z "$report" ] && { echo "keeper: cluster unreachable"; return; }
    while read -r tag g have total running shards maxtok short; do
        [ -z "$tag" ] && continue
        if [ "$have" -lt "$total" ] && [ "$running" -eq 0 ]; then
            echo "keeper: resubmit qwen $tag gen$g ($have/$total complete)"
            timeout 90 ssh -o BatchMode=yes uc3 \
                "WS=\$(ws_find llm_pilot); cd \$WS && GEN=$g SHARDS=$shards \
                 SEQS=32 CHUNK=64 MAXTOK=$maxtok sbatch --array=0-$((shards-1)) \
                 --time=12:00:00 --job-name=g3${short}_g${g} \
                 g3_vllm_${tag}.sbatch" >/dev/null 2>&1
        fi
    done <<<"$report"
}

step2_keeper() {
    local a b
    a=$(grep -c '"finish_reason": *"stop"' results/final_run_g2/answers/step1_codex_gen0.jsonl 2>/dev/null || echo 0)
    b=$(grep -c '"finish_reason": *"stop"' results/final_run_g2/answers/step1_codex_gen1.jsonl 2>/dev/null || echo 0)
    # Step 1 is 128 prompts per generation; only start Step 2 once both are in.
    [ "$a" -lt 128 ] || [ "$b" -lt 128 ] && return
    pgrep -f "run_codex_screen.py.*prompts_codex" >/dev/null && return
    pgrep -f "run_codex_screen.py.*prompts_step1" >/dev/null && return
    local i
    for ((i=0; i<STEP2_GENERATIONS; i++)); do
        local out="results/final_run_g2/answers/step2_codex_gen${i}.jsonl"
        echo "keeper: starting Step 2 generation $i"
        nohup env PYTHONPATH=src .venv/bin/python -u \
            scripts/codex_screen/run_codex_screen.py \
            --arm notools --codex-bin "$CODEX_BIN" \
            --prompts results/final_run_g2/prompts_codex.jsonl \
            --condition "" --input-kind "" --out "$out" \
            --max-total-tokens 200000000 --wait-for-reset --max-waits 128 \
            --max-attempts 3 \
            >> "results/final_run_g2/logs/step2_gen${i}.log" 2>&1 &
        echo $! > "results/final_run_g2/pids/step2_gen${i}.pid"
    done
}

case "${1:-all}" in
    qwen)  qwen_keeper ;;
    step2) step2_keeper ;;
    all)   qwen_keeper; step2_keeper ;;
    *) echo "usage: $0 {qwen|step2|all}" >&2; exit 2 ;;
esac
