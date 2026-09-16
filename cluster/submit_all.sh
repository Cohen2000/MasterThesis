#!/bin/bash -l
# Submit every Qwen pass: two modes x three repeats, four shards each.
#
# Each repeat writes into its own directory. Resume matches on request id, so a
# repeat written into another repeat's directory would see every id as done and
# silently collapse the repeat-to-repeat variation the study measures.
#
# Sizing comes from the measured probe on one H100: 819 tok/s at max_num_seqs=16,
# 11.4 s per answer, median 9.5k output tokens, no output-limit hits. Seventy
# requests per shard is roughly fifteen minutes of generation plus two minutes of
# model load, so the eight-hour wall time is generous and exists only so that a
# slow shard still finishes rather than to be used.
set -euo pipefail
cd "$(dirname "$0")"
for MODE in thinking nonthinking; do
    for REPEAT in 1 2 3; do
        sbatch --array=0-3 qwen_main.sbatch "$MODE" "$REPEAT" 4 16 16
    done
done
squeue -u "$USER" -o "%.10i %.14j %.10P %.8T %.6M %R"
