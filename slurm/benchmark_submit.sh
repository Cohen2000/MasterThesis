#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
J1=$(sbatch --parsable slurm/benchmark_01_data.sbatch)
J2=$(sbatch --parsable --dependency="afterok:$J1" slurm/benchmark_02_walks.sbatch)
J3=$(sbatch --parsable --dependency="afterok:$J2" slurm/benchmark_03_eval.sbatch)
echo "submitted data=$J1 walks=$J2 eval=$J3"
