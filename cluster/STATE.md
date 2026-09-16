# Execution state — Qwen main run

Purpose: let this work be continued and checked without the session that started it.

## Where things are

| What | Where |
| --- | --- |
| Offline run (source of truth) | `results/main_experiment/budget10_20261001` (local), copied to `$WS/mainexp/run` |
| Package on the cluster | `$WS/src/main_experiment` (uploaded from this repo) |
| Pinned model | `$WS/models/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Pinned environment | `$WS/venv_mainexp`, frozen in `$WS/mainexp/requirements.pinned.txt` |
| Answers | `$WS/mainexp/answers/<mode>_r<repeat>/<request_id>.json` |
| Job logs | `$WS/mainexp/logs/` |

`$WS` is `/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot`. The workspace expires
2026-10-24; extend with `ws_extend llm_pilot 30` before then.

## What has to exist when this is finished

1 680 answer files: 2 modes × 3 repeats × 280 observations. Empty observations
would be skipped, but there are none in this run, so the count is exact.

```bash
# completeness by mode and repeat
for m in thinking nonthinking; do for r in 1 2 3; do
  echo "$m r$r: $(ls $WS/mainexp/answers/${m}_r${r} 2>/dev/null | wc -l) / 280"
done; done
```

## How to continue

Resume is per request id: rerunning a pass skips the answers already on disk and
costs one model reload. Nothing else is needed.

```bash
cd $WS/mainexp
sbatch --array=0-3 qwen_main.sbatch <mode> <repeat> 4 16 16   # one pass
bash submit_all.sh                                            # all six
```

Never write a repeat into another repeat's directory: resume matches on request id
and would treat every id as done, silently collapsing the repeat-to-repeat
variation that the study measures.

## How to finish

```bash
python scripts/collect_qwen_answers.py \
    --run results/main_experiment/budget10_20261001 \
    --answers <answers dir> --out responses.jsonl
python scripts/evaluate_main_responses.py \
    --run results/main_experiment/budget10_20261001 \
    --responses responses.jsonl --out results/main_experiment/qwen_evaluation
```

The collector refuses to emit a file for an incomplete set and writes the missing
ids to `<out>.missing.txt`. An incomplete or partly missing result set is never a
finished main run.

## Configuration in force

One H100, `tensor_parallel_size=1`, `max_model_len=262144`, `max_tokens=258048`,
`max_num_seqs=16`, `chunk=16`, BF16, text-only. Sampling from the model card:
thinking `1.0 / 0.95 / 20 / presence 1.5`, non-thinking `0.7 / 0.80 / 20 / 1.5`.
`VLLM_USE_FLASHINFER_SAMPLER=0` and `CUDA_HOME` pointing at the venv's CUDA 13.4
are both required; see the runbook for why.

Measured on the probe: 819 tok/s at 16 concurrent sequences, 11.4 s per answer,
median 9 546 output tokens, no output-limit hits, 124 s model load. Roughly fifteen
minutes of generation per shard of seventy requests.
