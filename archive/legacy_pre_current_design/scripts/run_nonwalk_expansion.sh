#!/usr/bin/env bash
# Expanded non-walk LLM screen: 32 graphs instead of 4 per strategy, on
# several models.
#
# Why the case count and not the model count is the expansion: measured on the
# existing Qwen answers, the spread between strategies is 0.0177 while the
# spread between cases inside one strategy is 0.0313.  At 4 cases the standard
# error of a strategy mean is 0.0156, i.e. 88% of the entire signal, and a
# second model does not shrink it -- that variance sits between cases and every
# model sees the same ones.  At 32 cases the standard error is 0.0055.
#
# Models are added on top for a different reason: the current screen cannot
# separate "this access strategy is hard" from "Qwen broke on it".
#
# Usage:
#   bash scripts/run_nonwalk_expansion.sh prepare
#   bash scripts/run_nonwalk_expansion.sh gemini
#   bash scripts/run_nonwalk_expansion.sh deepseek
#   bash scripts/run_nonwalk_expansion.sh codex
#   bash scripts/run_nonwalk_expansion.sh status
#   bash scripts/run_nonwalk_expansion.sh evaluate
#
# Qwen runs on the cluster; see the runbook.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
OUT="$REPO/results/nonwalk_llm_expansion"
PREV="$REPO/results/nonwalk_llm_qwen36_screen"
CASES="$REPO/results/nonwalk_screen/panel32_cases.csv.gz"
STRATEGIES="uniform_event_reservoir,time_prefix_events,time_random_window_events,node_panel_full_history,ego_recent_k1,ego_recent_k5,ego_recent_kall,ego_recent_k20"
GEMINI_SHARDS="${NW_GEMINI_SHARDS:-4}"
DEEPSEEK_SHARDS="${NW_DEEPSEEK_SHARDS:-8}"
# Instances per data block for the Codex subset: 2 -> 8 graphs -> 128
# prompts -> about 7 h of quota at the measured 206 s per prompt.
CODEX_BLOCKS="${NW_CODEX_BLOCKS:-2}"

need_prepared() {
    if [[ ! -f "$OUT/prompts_all.jsonl" ]]; then
        echo "prompts missing; run: bash $0 prepare" >&2
        exit 2
    fi
}

prepare() {
    mkdir -p "$OUT"
    # Selection seed and strategy list are copied from the original screen so
    # the hash order -- and therefore the nesting -- is preserved.
    PYTHONPATH=src "$PYTHON_BIN" src/make_nonwalk_llm_prompts.py \
        --cases "$CASES" \
        --out "$OUT/prompts_full.jsonl" \
        --selected-cases-out "$OUT/selected_cases.csv" \
        --strategies "$STRATEGIES" \
        --budget 800 --sample-seed 0 \
        --instances-per-block 0 \
        --selection-seed 20260820 \
        --conditions sample,metadata_only_no_sample

    # Codex runs a smaller block-stratified subset: the full set would be 29
    # hours of quota at the measured 206 s per prompt.  Both conditions are
    # kept rather than more graphs, because "does it use the sample at all"
    # is the question the strongest reader in the panel is there to answer.
    # The subset stays nested inside the 16 and the 32, so it pairs directly.
    PYTHONPATH=src "$PYTHON_BIN" src/make_nonwalk_llm_prompts.py \
        --cases "$CASES" \
        --out "$OUT/prompts_codex_full.jsonl" \
        --selected-cases-out "$OUT/selected_cases_codex.csv" \
        --strategies "$STRATEGIES" \
        --budget 800 --sample-seed 0 \
        --instances-per-block "$CODEX_BLOCKS" \
        --selection-seed 20260820 \
        --conditions sample,metadata_only_no_sample

    "$PYTHON_BIN" src/split_nonwalk_expansion.py \
        --all "$OUT/prompts_full.jsonl" \
        --codex "$OUT/prompts_codex_full.jsonl" \
        --previous "$PREV/prompts.jsonl" \
        --out-dir "$OUT" \
        --gemini-shards "$GEMINI_SHARDS" \
        --deepseek-shards "$DEEPSEEK_SHARDS"
}

run_gemini() {
    need_prepared
    : "${GEMINI_API_KEY:?set GEMINI_API_KEY first (the script never prints it)}"
    mkdir -p "$OUT/logs" "$OUT/pids"
    local i prompt answer log
    for ((i=0; i<GEMINI_SHARDS; i++)); do
        prompt="$OUT/prompts_gemini.shard${i}.jsonl"
        answer="$OUT/answers_gemini-3.1-flash-lite_minimal.shard${i}.jsonl"
        log="$OUT/logs/gemini.shard${i}.log"
        if pgrep -f "run_llm_v2\.py.*$(basename "$answer")" >/dev/null; then
            echo "shard $i already running; skipped"
            continue
        fi
        PROMPTS_FILE="$prompt" OUT_FILE="$answer" LOG_FILE="$log" \
            nohup bash scripts/run_llm_v21_gemini.sh minimal >>"$log" 2>&1 &
        echo "gemini shard $i -> $answer (pid $!)"
    done
}

run_deepseek() {
    need_prepared
    : "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first (the script never prints it)}"
    mkdir -p "$OUT/logs"
    local i prompt answer log
    for ((i=0; i<DEEPSEEK_SHARDS; i++)); do
        prompt="$OUT/prompts_deepseek.shard${i}.jsonl"
        answer="$OUT/answers_deepseek-v4-flash_nothink.shard${i}.jsonl"
        log="$OUT/logs/deepseek.shard${i}.log"
        if pgrep -f "run_llm_v2\.py.*$(basename "$answer")" >/dev/null; then
            echo "shard $i already running; skipped"
            continue
        fi
        nohup bash scripts/run_llm_v21_nim.sh dsv4-flash \
            --prompts "$prompt" --out "$answer" >>"$log" 2>&1 &
        echo "deepseek shard $i -> $answer (pid $!)"
    done
}

run_deepseek_official() {
    need_prepared
    if [[ "${DEEPSEEK_API_KEY+x}" != x ]]; then
        echo "set DEEPSEEK_API_KEY first (the script never prints it)" >&2
        exit 2
    fi
    if pgrep -f "run_llm_v2\.py.*nonwalk_llm_expansion.*deepseek-v4-flash_nothink" >/dev/null; then
        echo "Refusing overlapping providers: the NIM run is still going." >&2
        echo "Stop it first:  bash $0 stop-deepseek-nim" >&2
        exit 2
    fi
    mkdir -p "$OUT/logs" "$OUT/pids"
    # Die Shard-Dateien werden von `prepare` mit DEEPSEEK_SHARDS erzeugt.  Ein
    # eigener Default hier hat genau einen Effekt: laeuft er niedriger, bleibt
    # der Rest des Designs stumm liegen -- der Lauf meldet trotzdem "done, 0
    # errors", weil jeder gestartete Shard fuer sich vollstaendig ist.  Genau
    # das ist mit 4 statt 8 passiert und hat die Haelfte der 512 Prompts
    # verschluckt.  Der Default folgt deshalb der Vorbereitung.
    local shards="${NW_DSAPI_SHARDS:-$DEEPSEEK_SHARDS}"
    local prepared
    prepared=$(ls "$OUT"/prompts_deepseek.shard*.jsonl 2>/dev/null | wc -l)
    if (( shards < prepared )); then
        echo "Warnung: $prepared Shard-Dateien vorbereitet, aber nur $shards" \
             "gestartet -- $((prepared - shards)) Shards blieben unbearbeitet." >&2
    fi
    # Estimated from the NIM run's measured usage (mean 1191 in / 1984 out):
    # about 0.37 USD for all 512.  The cap is per process, so the total ceiling
    # is shards x cap -- a runaway brake, not a budget target.
    local cap="${NW_DSAPI_MAX_USD:-0.35}"
    local max_tokens="${NW_DSAPI_MAX_TOKENS:-8192}"
    local i prompt answer log
    for ((i=0; i<shards; i++)); do
        prompt="$OUT/prompts_deepseek.shard${i}.jsonl"
        answer="$OUT/answers_deepseek-v4-flash_official_nothink.shard${i}.jsonl"
        log="$OUT/logs/deepseek_official.shard${i}.log"
        if [[ ! -f "$prompt" ]]; then
            echo "missing shard file: $prompt" >&2
            exit 2
        fi
        nohup env PYTHONPATH=src "$PYTHON_BIN" src/run_llm_v2.py \
            --backend api \
            --prompts "$prompt" \
            --out "$answer" \
            --base-url https://api.deepseek.com \
            --api-key-env DEEPSEEK_API_KEY \
            --model deepseek-v4-flash \
            --thinking off \
            --api-thinking-format deepseek \
            --temperature 1.0 \
            --max-tokens "$max_tokens" \
            --max-length-attempts 1 \
            --timeout 1200 --retries 5 --sleep 1 \
            --max-usd "$cap" \
            --price-input-cache-hit 0.0028 \
            --price-input-cache-miss 0.14 \
            --price-output 0.28 \
            >>"$log" 2>&1 &
        echo "$!" >"$OUT/pids/deepseek_official.shard${i}.pid"
        echo "deepseek(api) shard $i -> $answer (pid $!)"
    done
    # awk rather than a nested-quote python f-string, which the shell mangles.
    echo "Kostendeckel: \$$cap je Shard, $shards Shards -> hoechstens \$$(
        awk -v c="$cap" -v n="$shards" 'BEGIN{printf "%.2f", c*n}')"
    echo "Output-Budget: $max_tokens Tokens (wie im NIM-Lauf und im Qwen-Screen)"
}

stop_deepseek_nim() {
    # The NIM answers stay on disk. They are a different provider name
    # (deepseek-v4-flash-0731) and simply drop out of the evaluation, which
    # globs the official file for the DeepSeek row.
    local pids
    pids=$(pgrep -f "run_llm_v2\.py.*nonwalk_llm_expansion.*deepseek-v4-flash_nothink" || true)
    if [[ -z "$pids" ]]; then
        echo "kein NIM-Lauf aktiv"
        return
    fi
    echo "beende NIM-Shards: $pids"
    kill $pids 2>/dev/null || true
    echo "Antwortdateien bleiben liegen (answers_deepseek-v4-flash_nothink.shard*.jsonl)"
}

run_codex() {
    need_prepared
    local out_dir="${CODEX_NW_OUT_DIR:-$HOME/Dokumente/codex_nonwalk}"
    local codex_bin="${CODEX_NW_BIN:-$HOME/.codex/packages/standalone/releases/0.146.0-x86_64-unknown-linux-musl/bin/codex}"
    local answer="$out_dir/answers_codex-gpt-5.6-sol_notools_high_nonwalk.jsonl"
    if [[ ! -x "$codex_bin" ]] || \
       [[ "$("$codex_bin" --version 2>/dev/null)" != "codex-cli 0.146.0" ]]; then
        echo "Pinned Codex CLI 0.146.0 not found at $codex_bin" >&2
        exit 2
    fi
    mkdir -p "$out_dir"
    # The screen's own input kind, not the walk suite's `mask`.
    "$PYTHON_BIN" scripts/codex_screen/run_codex_screen.py \
        --arm notools \
        --codex-bin "$codex_bin" \
        --prompts "$OUT/prompts_codex.jsonl" \
        --condition "" --input-kind "" \
        --out "$answer" \
        --max-total-tokens "${CODEX_NW_TOKEN_CAP:-12000000}" \
        --wait-for-reset --max-attempts 2 "$@"
    cp "$answer" "$OUT/"
    echo "copied -> $OUT/$(basename "$answer")"
}

status() {
    need_prepared
    PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import collections, glob, json
from pathlib import Path
from run_llm_v2 import is_complete_record

out = Path("results/nonwalk_llm_expansion")
prompts = {json.loads(l)["prompt_id"]: json.loads(l)
           for l in open(out / "prompts_all.jsonl") if l.strip()}
codex = {json.loads(l)["prompt_id"] for l in open(out / "prompts_codex.jsonl")
         if l.strip()}
runs = [
    ("gemini-3.1-flash-lite", 512, [str(out / "answers_gemini*.jsonl")]),
    ("deepseek-v4-flash (API)", 512,
     [str(out / "answers_deepseek-v4-flash_official*.jsonl")]),
    ("deepseek-v4-flash (NIM)", 512,
     [str(out / "answers_deepseek-v4-flash_nothink*.jsonl")]),
    ("qwen3.6-27b nothink", 512, [
        str(out / "answers_*qwen36_nothink*.jsonl"),
        "results/nonwalk_llm_qwen36_screen/answers/answers_nonwalk_qwen36_nothink.shard*.jsonl"]),
    ("codex-5.6 notools", len(codex), [str(out / "answers_codex*.jsonl")]),
]
for label, expected, patterns in runs:
    done, seen = set(), set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if "smoke" in Path(path).name:
                continue
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("prompt_id")
                if pid not in prompts:
                    continue
                seen.add(pid)
                if is_complete_record(rec):
                    done.add(pid)
    if label.startswith("codex"):
        done &= codex
        seen &= codex
    by = collections.Counter(prompts[p]["strategy"] for p in done)
    print(f"{label}")
    print(f"  complete {len(done):4d}/{expected}  ({100*len(done)/expected:5.1f}%)"
          f" | attempted {len(seen)}")
    if by:
        # Four of the eight strategies start with "ego_recent_", so a
        # prefix-truncated label collapses them into one indistinguishable row.
        short = {"uniform_event_reservoir": "reservoir",
                 "time_prefix_events": "prefix",
                 "time_random_window_events": "window",
                 "node_panel_full_history": "panel"}
        print("  per strategy: " + "  ".join(
            f"{short.get(k, k.replace('ego_recent_', 'ego'))}={v}"
            for k, v in sorted(by.items())))
    print()
PY
}

evaluate() {
    need_prepared
    mkdir -p "$OUT/eval"
    PYTHONPATH=src "$PYTHON_BIN" src/evaluate_nonwalk_llm.py \
        --prompts "$OUT/prompts_all.jsonl" \
        --cases "$OUT/selected_cases.csv" \
        --answers "$OUT/answers_*.jsonl" \
        --answers "$PREV/answers/answers_nonwalk_qwen36_*.shard*.jsonl" \
        --baseline-predictions results/nonwalk_screen/baselines/predictions.csv.gz \
        --out-dir "$OUT/eval"
    echo "report -> $OUT/eval/SUMMARY.md"
}

case "${1:-}" in
    prepare)  prepare ;;
    gemini)   run_gemini ;;
    deepseek) run_deepseek ;;
    deepseek-official) run_deepseek_official ;;
    stop-deepseek-nim) stop_deepseek_nim ;;
    codex)    shift; run_codex "$@" ;;
    status)   status ;;
    evaluate) evaluate ;;
    *)
        echo "usage: $0 {prepare|gemini|deepseek|deepseek-official|stop-deepseek-nim|codex|status|evaluate}" >&2
        exit 2
        ;;
esac
