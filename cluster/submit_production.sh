#!/bin/bash -l
# Submit a complete, self-finishing production chain for one experiment directory:
#   round 1   the generation array
#   round 2   the same array again after round 1 ends, whatever its outcome; shards
#             that finished exit before loading the model (NOTHING_TO_DO), shards
#             that were cut resume only unstarted requests; admitted ones are never retried
#   round 3   one more safety round after round 2
#   archive   a CPU job after round 3 that builds and verifies the archive
# Nothing here depends on an interactive session. Usage:
#   cd $WS/<EXP>/mainexp && bash submit_production.sh <EXP> [SHARDS]
set -euo pipefail
EXP="${1:?experiment directory under \$WS}"
SHARDS="${2:-6}"
ARMS="${3:-S}"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
cd "$WS/$EXP/mainexp"
mkdir -p logs
LAST=$((SHARDS-1))
r1=$(sbatch --parsable --job-name="${EXP}_r1" --array=0-$LAST qwen_engine.sbatch "$EXP" "$SHARDS" 16 "$ARMS")
r2=$(sbatch --parsable --job-name="${EXP}_r2" --dependency=afterany:$r1 --array=0-$LAST qwen_engine.sbatch "$EXP" "$SHARDS" 16 "$ARMS")
r3=$(sbatch --parsable --job-name="${EXP}_r3" --dependency=afterany:$r2 --array=0-$LAST qwen_engine.sbatch "$EXP" "$SHARDS" 16 "$ARMS")
ar=$(sbatch --parsable --job-name="${EXP}_archive" --dependency=afterany:$r3 -p cpu -t 02:00:00 -c 4 --mem=16G \
     -o "logs/archive_%j.out" \
     --wrap "bash -lc 'module load devel/python/3.12.3-gnu-14.2; source $WS/venv_mainexp/bin/activate; export HF_HUB_OFFLINE=1; cd $WS/$EXP/mainexp; pip freeze > pip_freeze_job.txt; python status.py . answers $ARMS > status_final.json; python build_archive.py --exp $EXP'")
printf 'round1=%s\nround2=%s\nround3=%s\narchive=%s\n' "$r1" "$r2" "$r3" "$ar" | tee production_jobs.txt
