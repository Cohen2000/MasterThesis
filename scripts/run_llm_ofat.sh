#!/usr/bin/env bash
# Prepare and run the six-cell LLM input OFAT ablation.
#
# Local/API work:
#   bash scripts/run_llm_ofat.sh prepare
#   bash scripts/run_llm_ofat.sh gemini
#   bash scripts/run_llm_ofat.sh deepseek
#   bash scripts/run_llm_ofat.sh codex
#   bash scripts/run_llm_ofat.sh status
#   bash scripts/run_llm_ofat.sh evaluate
#
# Qwen runs on the cluster; see docs/LLM_OFAT_RUNBOOK.md and
# scripts/submit_llm_ofat_qwen.sh.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
OUT="$REPO/results/llm_v21_ofat"
GEMINI_SHARDS="${OFAT_GEMINI_SHARDS:-3}"
DEEPSEEK_SHARDS="${OFAT_DEEPSEEK_SHARDS:-8}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi

prepare() {
    mkdir -p "$OUT"
    "$PYTHON_BIN" src/make_llm_prompts_v2.py \
        --cases-dir results/llm_v2 \
        --out "$OUT/prompts_ofat_new_cells.jsonl" \
        --only-ablation \
        --ablation-inputs mask_temporal,mask_recent \
        --no-tool-subset
    PYTHONPATH=src "$PYTHON_BIN" src/make_llm_ofat.py \
        --reps 3 \
        --deepseek-shards "$DEEPSEEK_SHARDS" \
        --gemini-shards "$GEMINI_SHARDS"
}

need_prepared() {
    if [[ ! -f "$OUT/prompts_ofat.jsonl" ]]; then
        echo "OFAT prompts are absent; running prepare first"
        prepare
    fi
}

start_bg() {
    local name="$1" pidfile="$2"
    shift 2
    if [[ -f "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then
        echo "$name already runs as PID $(<"$pidfile")"
        return
    fi
    nohup "$@" >/dev/null 2>&1 &
    echo "$!" > "$pidfile"
    echo "$name started as PID $!"
}

run_codex() {
    need_prepared
    local out_dir="${CODEX_OFAT_OUT_DIR:-$HOME/Dokumente/codex_ofat}"
    # The four existing OFAT cells were produced with 0.146.0.  Pin the two
    # missing cells to that same cached release: 0.148.0 changed Code Mode
    # handling and fails the deliberately tool-free arm after spending tokens.
    local codex_bin="${CODEX_OFAT_BIN:-$HOME/.codex/packages/standalone/releases/0.146.0-x86_64-unknown-linux-musl/bin/codex}"
    local prompt="$OUT/prompts_ofat_codex.jsonl"
    local answer="$out_dir/answers_codex-gpt-5.6-sol_notools_high_ofat.jsonl"
    local cap="${CODEX_OFAT_TOKEN_CAP:-13000000}"
    local kinds=(mask_temporal mask_recent)
    if [[ ! -x "$codex_bin" ]]; then
        echo "Pinned Codex CLI 0.146.0 not found: $codex_bin" >&2
        echo "Set CODEX_OFAT_BIN to the matching 0.146.0 executable." >&2
        exit 2
    fi
    if [[ "$("$codex_bin" --version 2>/dev/null)" != "codex-cli 0.146.0" ]]; then
        echo "Refusing to mix Codex CLI versions; expected codex-cli 0.146.0 at $codex_bin" >&2
        exit 2
    fi
    mkdir -p "$out_dir"
    for kind in "${kinds[@]}"; do
        echo "== Codex OFAT: $kind (36 cases) =="
        "$PYTHON_BIN" scripts/codex_screen/run_codex_screen.py \
            --arm notools \
            --codex-bin "$codex_bin" \
            --prompts "$prompt" \
            --condition disclosed \
            --input-kind "$kind" \
            --out "$answer" \
            --max-total-tokens "$cap" \
            --wait-for-reset \
            --max-attempts 2
    done
    echo "Codex output: $answer"
    echo "Copy it into the experiment directory before evaluation:"
    echo "  cp '$answer' '$OUT/'"
}

run_gemini() {
    need_prepared
    : "${GEMINI_API_KEY:?set GEMINI_API_KEY first (the script never prints it)}"
    mkdir -p "$OUT/logs" "$OUT/pids"
    local i prompt answer log pidfile
    for ((i=0; i<GEMINI_SHARDS; i++)); do
        prompt="$OUT/prompts_ofat_gemini.shard${i}.jsonl"
        answer="$OUT/answers_gemini-3.1-flash-lite_minimal.shard${i}.jsonl"
        log="$OUT/logs/gemini_minimal.shard${i}.log"
        pidfile="$OUT/pids/gemini.shard${i}.pid"
        [[ -f "$prompt" ]] || { echo "missing $prompt; rerun prepare" >&2; exit 2; }
        start_bg "Gemini shard $i" "$pidfile" env \
            PROMPTS_FILE="$prompt" OUT_FILE="$answer" LOG_FILE="$log" \
            GEMINI_SLEEP="${GEMINI_SLEEP:-13}" \
            GEMINI_MAXTOK="${GEMINI_MAXTOK:-8192}" \
            GEMINI_MAX_WAIT="${GEMINI_MAX_WAIT:-3600}" \
            bash scripts/run_llm_v21_gemini.sh minimal
    done
    echo "Gemini logs: $OUT/logs/gemini_minimal.shard*.log"
}

retryable_count() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
sys.path.insert(0, "src")
import run_llm_v2 as r
rows = r.load_prompts(sys.argv[1], 0, 1, [])
done, todo, _ = r.select_todo(rows, sys.argv[2], max_length_attempts=1)
print(len(todo))
PY
}

deepseek_worker() {
    local i="${1:?shard index required}"
    local prompt="$OUT/prompts_ofat_deepseek.shard${i}.jsonl"
    local answer="$OUT/answers_deepseek-v4-flash-0731_nothink.shard${i}.jsonl"
    local api_log="$OUT/logs/deepseek_flash.shard${i}.api.log"
    local max_passes="${OFAT_DEEPSEEK_MAX_PASSES:-3}"
    local pass left
    for ((pass=1; pass<=max_passes; pass++)); do
        left=$(retryable_count "$prompt" "$answer")
        echo "DeepSeek shard $i pass $pass/$max_passes: $left retryable prompts"
        [[ "$left" -eq 0 ]] && break
        PROMPTS_FILE="$prompt" OUT_FILE="$answer" LOG_FILE="$api_log" \
        NIM_MAXTOK="${NIM_MAXTOK:-0}" \
        NIM_MAX_WAIT="${NIM_MAX_WAIT:-120}" \
        NIM_TOTAL_WAIT="${NIM_TOTAL_WAIT:-600}" \
        NIM_RETRIES="${NIM_RETRIES:-12}" \
            bash scripts/run_llm_v21_nim.sh dsv4-flash \
                --max-length-attempts 1 --sleep 0
    done
    left=$(retryable_count "$prompt" "$answer")
    echo "DeepSeek shard $i finished worker: $left retryable prompts remain"
}

run_deepseek() {
    need_prepared
    : "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first (the script never prints it)}"
    mkdir -p "$OUT/logs" "$OUT/pids"
    local i prompt pidfile worker_log
    for ((i=0; i<DEEPSEEK_SHARDS; i++)); do
        prompt="$OUT/prompts_ofat_deepseek.shard${i}.jsonl"
        pidfile="$OUT/pids/deepseek.shard${i}.pid"
        worker_log="$OUT/logs/deepseek_flash.shard${i}.worker.log"
        [[ -f "$prompt" ]] || { echo "missing $prompt; rerun prepare" >&2; exit 2; }
        if [[ -f "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then
            echo "DeepSeek shard $i already runs as PID $(<"$pidfile")"
            continue
        fi
        nohup bash "$0" _deepseek_worker "$i" >>"$worker_log" 2>&1 &
        echo "$!" > "$pidfile"
        echo "DeepSeek shard $i started as PID $!"
    done
    echo "DeepSeek worker logs: $OUT/logs/deepseek_flash.shard*.worker.log"
}

cmd="${1:-}"
case "$cmd" in
    prepare) prepare ;;
    codex) run_codex ;;
    gemini) run_gemini ;;
    deepseek) run_deepseek ;;
    all-local)
        run_gemini
        run_deepseek
        run_codex
        ;;
    _deepseek_worker) deepseek_worker "${2:-}" ;;
    status)
        PYTHONPATH=src "$PYTHON_BIN" src/llm_ofat_status.py
        ;;
    evaluate)
        PYTHONPATH=src "$PYTHON_BIN" src/evaluate_llm_ofat.py
        ;;
    *)
        echo "usage: $0 {prepare|gemini|deepseek|codex|all-local|status|evaluate}" >&2
        exit 2
        ;;
esac
