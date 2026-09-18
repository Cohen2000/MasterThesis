#!/bin/bash
# After the cluster chain has finished: pull answers and archive, check
# completeness, collect (parser v2 main, v3 sensitivity), evaluate, audit.
# Idempotent; needs an open ssh ControlMaster to uc3.
#   bash scripts/finish_cells10.sh
set -euo pipefail
echo "Historical entry point: use run_json_revision_offline.sh and the revised runbook." >&2
exit 2

cd "$(dirname "$0")/.."
EXP=cells10
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
RUN=results/main_experiment/cells10_20260917
Q=results/main_experiment/cells10_20260917_qwen
PB=results/baseline_revision_cells10_20260917/primary_baselines.json
mkdir -p "$Q/answers"
rsync -a "uc3:$WS/$EXP/mainexp/answers/" "$Q/answers/"
rsync -a "uc3:$WS/$EXP/mainexp/logs/" "$Q/cluster_logs/"
ssh -o BatchMode=yes uc3 "cat $WS/$EXP/mainexp/status_final.json 2>/dev/null" > "$Q/status_final_cluster.json" || true
# archive, if the archive job has finished
if ssh -o BatchMode=yes uc3 "test -f $WS/$EXP/mainexp/archive/ARCHIVE_REPORT.json"; then
  ssh -o BatchMode=yes uc3 "cd $WS/$EXP/mainexp && tar czf ../${EXP}_qwen_archive.tgz archive && cd .. && sha256sum ${EXP}_qwen_archive.tgz > ${EXP}_qwen_archive.tgz.sha256 && mkdir -p \$HOME/${EXP}_archive && cp ${EXP}_qwen_archive.tgz* \$HOME/${EXP}_archive/"
  scp -q "uc3:$WS/$EXP/${EXP}_qwen_archive.tgz" "uc3:$WS/$EXP/${EXP}_qwen_archive.tgz.sha256" results/main_experiment/
  (cd results/main_experiment && sha256sum -c ${EXP}_qwen_archive.tgz.sha256)
fi
.venv/bin/python scripts/collect_qwen_answers.py --run "$RUN" --answers "$Q/answers" --out "$Q/responses_v2.jsonl"
.venv/bin/python scripts/collect_qwen_answers.py --run "$RUN" --answers "$Q/answers" --out "$Q/responses_v3.jsonl" --extract-trailing-json
.venv/bin/python scripts/evaluate_main_responses.py --run "$RUN" --baselines "$PB" --responses "$Q/responses_v2.jsonl" --out "$Q/evaluation_v2"
.venv/bin/python scripts/evaluate_main_responses.py --run "$RUN" --baselines "$PB" --responses "$Q/responses_v3.jsonl" --out "$Q/evaluation_v3"
.venv/bin/python scripts/audit_parser_rules.py --answers "$Q/answers" --out "$Q/parser_rule_audit.json"
echo FINISH_DONE
