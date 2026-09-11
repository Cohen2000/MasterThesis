#!/usr/bin/env bash
# One read-only view of everything currently running: both noise-probe arms
# and the expanded non-walk screen, local and on the cluster.
#
#   bash scripts/live_all.sh          once
#   watch -n 60 bash scripts/live_all.sh
#
# Qwen answers live on the cluster until they are downloaded, so their counts
# are fetched over SSH rather than read from the local tree. Globbing locally
# would report a healthy cluster run as 0/160 -- a progress display that says
# "not running" while it runs is worse than no display at all. When the
# multiplexed session has expired those rows say so instead of showing zero.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=6 uc3 '
    WS=$(ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws)
    cd "$WS/llm_v21" 2>/dev/null && python3 ~/MasterArbeit/scripts/cluster_counts.py
    echo "---QUEUE---"
    squeue --me -h -o "%.14i %.16j %.2t %.10M %.16P %R"
' >"$TMP/cluster" 2>/dev/null && CLUSTER_OK=1 || CLUSTER_OK=0

alive() {
    # pgrep -f matches the whole command line, so any shell that merely
    # mentions the pattern -- an editor, a heredoc, this script's own parent --
    # counts as a hit. Requiring the process to actually be a python
    # interpreter removes that class of false positive.
    local pid n=0 first=""
    for pid in $(pgrep -f "$1" 2>/dev/null); do
        case "$(ps -o comm= -p "$pid" 2>/dev/null)" in
            python*|codex*)
                n=$((n + 1))
                [[ -z "$first" ]] && \
                    first="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
                ;;
        esac
    done
    if (( n == 0 )); then
        printf 'gestoppt'
    elif (( n == 1 )); then
        printf 'laeuft (%s)' "$first"
    else
        # Ein einzelnes etime waere hier irrefuehrend: die Shards starten
        # gemeinsam, aber gemeldet wuerde nur der erste Treffer von pgrep.
        printf 'laeuft (%d Shards, seit %s)' "$n" "$first"
    fi
}

echo "=============================================================="
echo " Stand $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo
echo "--- Prozesse lokal ---"
printf '  Codex Rauschprobe    : %s\n' "$(alive 'run_codex_screen\.py.*noise')"
codex_nw="$(alive 'run_codex_screen\.py.*nonwalk')"
if [[ "$codex_nw" == gestoppt ]] && \
   pgrep -f 'run_nonwalk_codex_after_noise' >/dev/null 2>&1; then
    codex_nw='wartet auf die Rauschprobe'
fi
printf '  Codex Non-Walk       : %s\n' "$codex_nw"
printf '  Codex-Waechter       : %s\n' \
    "$(pgrep -f 'run_nonwalk_codex_after_noise' >/dev/null && echo wartet || echo gestoppt)"
printf '  Gemini Non-Walk      : %s\n' "$(alive 'run_llm_v2\.py.*nonwalk_llm_expansion.*gemini')"
printf '  DeepSeek Non-Walk    : %s\n' "$(alive 'run_llm_v2\.py.*nonwalk_llm_expansion.*deepseek')"
echo

CLUSTER_OK="$CLUSTER_OK" CLUSTER_FILE="$TMP/cluster" \
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import collections, glob, json, os
from pathlib import Path
from run_llm_v2 import is_complete_record

cluster = {}
if os.environ.get("CLUSTER_OK") == "1":
    for line in open(os.environ["CLUSTER_FILE"]):
        parts = line.split()
        if len(parts) == 4 and parts[0].isupper():
            cluster[parts[0]] = tuple(int(x) for x in parts[1:])


def count(patterns, wanted):
    done, seen = set(), set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if "smoke" in Path(path).name:
                continue
            try:
                fh = open(path)
            except OSError:
                continue
            with fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue   # half-written last line, not evidence
                    pid = rec.get("prompt_id")
                    if pid not in wanted:
                        continue
                    seen.add(pid)
                    if is_complete_record(rec):
                        done.add(pid)
    return len(done), len(seen)


def bar(done, total, width=28):
    filled = 0 if not total else int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def row(label, done, seen, total, note=""):
    if done is None:
        print(f"  {label:<22} {'(Cluster nicht erreichbar)':<28}")
        return
    pct = 0.0 if not total else 100 * done / total
    extra = f"  (+{seen - done} offen)" if seen and seen > done else ""
    print(f"  {label:<22} {bar(done, total)} {done:4d}/{total:<4d} "
          f"{pct:5.1f}%{extra}{note}")


def cl(key, add_done=0, add_seen=0, add_total=0):
    """Cluster row, optionally plus locally reused answers."""
    if key not in cluster:
        return None, 0, 0
    d, s, t = cluster[key]
    return d + add_done, s + add_seen, t + add_total


probe = Path("results/llm_noise_probe")
if (probe / "prompts_subset160.jsonl").exists():
    ids = {json.loads(l)["prompt_id"]
           for l in open(probe / "prompts_subset160.jsonl") if l.strip()}
    print("--- Rauschprobe (160 Prompts, 32 Graphen) ---")
    d, s = count([str(probe / "answers_codex*.jsonl"),
                  str(Path.home() / "Dokumente/codex_noise/answers_codex*.jsonl")], ids)
    row("Codex", d, s, len(ids))
    row("Qwen nothink *", *cl("NOISE_NOTHINK"))
    row("Qwen think *", *cl("NOISE_THINK"))
    d, s = count([str(probe / "answers_gemini*.jsonl")], ids)
    row("Gemini (fertig)", d, s, len(ids))
    d, s = count([str(probe / "answers_dsv4flash*.jsonl")], ids)
    row("DeepSeek (fertig)", d, s, len(ids))
    print()

exp = Path("results/nonwalk_llm_expansion")
if (exp / "prompts_all.jsonl").exists():
    ids = {json.loads(l)["prompt_id"]
           for l in open(exp / "prompts_all.jsonl") if l.strip()}
    codex_ids = {json.loads(l)["prompt_id"]
                 for l in open(exp / "prompts_codex.jsonl") if l.strip()}
    print("--- Non-Walk-Erweiterung (512 Prompts, 32 Graphen) ---")
    d, s = count([str(exp / "answers_gemini*.jsonl")], ids)
    row("Gemini", d, s, len(ids))
    # NIM and the official API answer the same prompt_ids, so one glob over
    # both would union them and overstate whichever is being watched.
    d, s = count([str(exp / "answers_deepseek-v4-flash_official*.jsonl")], ids)
    if d or s:
        row("DeepSeek (API)", d, s, len(ids))
    d, s = count([str(exp / "answers_deepseek-v4-flash_nothink*.jsonl")], ids)
    if d or s:
        row("DeepSeek (NIM, ersetzt)", d, s, len(ids))
    # 64 of the 512 come from the original screen and are reused, not rerun.
    reused = count(["results/nonwalk_llm_qwen36_screen/answers/"
                    "answers_nonwalk_qwen36_nothink.shard*.jsonl"], ids)
    row("Qwen nothink *", *cl("NWEXP_NOTHINK", reused[0], reused[1], 64),
        note=f"   [{reused[0]} wiederverwendet, 448 auf dem Cluster]")
    d, s = count([str(exp / "answers_codex*.jsonl"),
                  str(Path.home() / "Dokumente/codex_nonwalk/answers_codex*.jsonl")],
                 codex_ids)
    row("Codex", d, s, len(codex_ids))
    print()
    print("  * Cluster; Zahlen per SSH geholt, Antworten noch nicht heruntergeladen")
    print()
PY

echo "--- Cluster-Queue ---"
if [[ "$CLUSTER_OK" == "1" ]]; then
    sed -n '/---QUEUE---/,$p' "$TMP/cluster" | tail -n +2
else
    echo "  (keine SSH-Verbindung -- 'ssh -fN uc3' oeffnet sie neu, einmal mit OTP)"
fi
