#!/bin/bash -l
# One task of a stage-2 job array: python scripts/pipeline.py task --plan <plan> --index <array index>.
# Submitted by scripts/pipeline.py --submit (partition, resources, output and dependencies on the command line).
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
VENV="${1:?venv}"
PLAN="${2:?plan file}"
module load devel/python/3.12.3-gnu-14.2
source "$WS/$VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/pipeline.py task --plan "$PLAN" --index "$SLURM_ARRAY_TASK_ID"
