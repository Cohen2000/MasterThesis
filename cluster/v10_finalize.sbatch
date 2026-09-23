#!/bin/bash -l
#SBATCH --job-name=v10_finalize
#SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/v10_finalize_%j.out
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
PROJECT="$WS/panel888_v10_main"
ARCHIVE="$WS/panel888_access_v10_main/mainexp/archive"
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_offline/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PROJECT"
OUT="$PROJECT/docs/results/panel888_v10_main_20260923"
mkdir -p "$OUT"
python scripts/verify_v10_answers.py \
  --prepared "$PROJECT/results/panel888_v10/prepared" \
  --answers "$ARCHIVE/../answers" \
  --output "$OUT/QWEN_ANSWER_VERIFICATION.json"
python scripts/build_v10_results.py \
  --prepared "$PROJECT/results/panel888_v10/prepared" \
  --et "$PROJECT/results/panel888_v10/et" \
  --gate "$WS/panel888_v10_walk_audit/results_amended/walk_gate.csv" \
  --answers "$WS/panel888_access_v10_main/mainexp/answers" --out "$OUT"
