#!/bin/bash -l
# Submit every Qwen pass: two modes x three repeats, four shards each.
#
# Each repeat writes into its own directory. Resume matches on request id, so a
# repeat written into another repeat's directory would see every id as done and
# silently collapse the repeat-to-repeat variation the study measures.
#
# Sizing comes from the measured probe on one H100: 819 tok/s at max_num_seqs=16,
# 11.4 s per answer, median 9.5k output tokens, no output-limit hits. Eight shards
# put thirty-five requests in each, about seven minutes of generation, which fits
# the short partition's thirty-minute limit even after a slow model load.
set -euo pipefail
cd "$(dirname "$0")"
for MODE in thinking nonthinking; do
    for REPEAT in 1 2 3; do
        sbatch --array=0-7 qwen_main.sbatch "$MODE" "$REPEAT" 8 16 16
    done
done
squeue -u "$USER" -o "%.10i %.14j %.10P %.8T %.6M %R"
