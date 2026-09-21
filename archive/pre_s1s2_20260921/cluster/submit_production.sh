#!/bin/bash -l
# Submit a complete, self-finishing production chain for one experiment directory:
#   round 1   the generation array
#   round 2   the same array again after round 1 ends, whatever its outcome; shards
#             that finished exit before loading the model (NOTHING_TO_DO), shards
#             that were cut resume only unstarted requests; admitted ones are never retried
#   round 3   one more safety round after round 2
#   archive   a CPU job after round 3 that builds and verifies the archive
# Nothing here depends on an interactive session. Usage:
#   cd $WS/<EXP>/mainexp && bash submit_production.sh <EXP> <SHARDS> all <VERIFIED_COMMIT>
set -euo pipefail
EXP="${1:?experiment directory under \$WS}"
SHARDS="${2:-6}"
ARMS="${3:-all}"
EXPECTED_COMMIT="${4:?verified source commit required}"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
cd "$WS/$EXP/mainexp"
# One chain only. A partial submission must be reconciled from the durable log,
# never blindly resubmitted.
exec 9>production_submission.lock
flock -n 9
if [ -e production_submission_started.txt ] || [ -e production_jobs.txt ]; then
    echo 'Refusing duplicate production submission; reconcile existing job IDs.' >&2
    exit 1
fi
(cd .. && sha256sum -c --quiet BUNDLE_SHA256SUMS)
test "$(cat ../SPEC_COMMIT)" = "$EXPECTED_COMMIT"
test ! -s ../WORKTREE_STATUS
date -u +%FT%TZ > production_submission_started.txt
printf 'workspace=%s\ncommit=%s\n' "$WS/$EXP" "$EXPECTED_COMMIT" >> production_submission_started.txt
mkdir -p logs
LAST=$((SHARDS-1))
r1=$(sbatch --parsable --job-name="${EXP}_r1" --array=0-$LAST qwen_engine.sbatch "$EXP" "$SHARDS" 16 "$ARMS")
printf 'round1=%s\n' "$r1" >> production_jobs.txt
r2=$(sbatch --parsable --job-name="${EXP}_r2" --dependency=afterany:$r1 --array=0-$LAST qwen_engine.sbatch "$EXP" "$SHARDS" 16 "$ARMS")
printf 'round2=%s\n' "$r2" >> production_jobs.txt
r3=$(sbatch --parsable --job-name="${EXP}_r3" --dependency=afterany:$r2 --array=0-$LAST qwen_engine.sbatch "$EXP" "$SHARDS" 16 "$ARMS")
printf 'round3=%s\n' "$r3" >> production_jobs.txt
ar=$(sbatch --parsable --job-name="${EXP}_archive" --dependency=afterany:$r3 -p cpu -t 02:00:00 -c 4 --mem=16G \
     -o "logs/archive_%j.out" \
     --wrap "bash -lc 'module load devel/python/3.12.3-gnu-14.2; source $WS/venv_mainexp/bin/activate; export HF_HUB_OFFLINE=1; cd $WS/$EXP/mainexp; pip freeze > pip_freeze_job.txt; python status.py . answers $ARMS > status_final.json; python build_archive.py --exp $EXP'")
printf 'archive=%s\n' "$ar" >> production_jobs.txt
cat production_jobs.txt
