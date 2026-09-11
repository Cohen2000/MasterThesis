#!/bin/bash
# Build a small paired Qwen screen: 8 strategies x 4 data blocks x
# (sample + metadata-only) = 64 prompts per model mode.
set -euo pipefail

OUT_DIR="results/nonwalk_llm_qwen36_screen"
STRATEGIES="uniform_event_reservoir,time_prefix_events,time_random_window_events,node_panel_full_history,ego_recent_k1,ego_recent_k5,ego_recent_k20,ego_recent_kall"

PYTHONPATH=src python3 src/make_nonwalk_llm_prompts.py \
    --cases results/nonwalk_screen/panel32_cases.csv.gz \
    --out "$OUT_DIR/prompts.jsonl" \
    --selected-cases-out "$OUT_DIR/selected_cases.csv" \
    --strategies "$STRATEGIES" \
    --budget 800 \
    --sample-seed 0 \
    --instances-per-block 1 \
    --selection-seed 20260820 \
    --conditions sample,metadata_only_no_sample

python3 - "$OUT_DIR/prompts.jsonl" "$OUT_DIR/selected_cases.csv" <<'PY'
import json
import sys

import pandas as pd

prompt_path, case_path = sys.argv[1:]
with open(prompt_path) as fh:
    prompts = [json.loads(line) for line in fh if line.strip()]
cases = pd.read_csv(case_path, low_memory=False)
assert len(cases) == 32, len(cases)
assert len(prompts) == 64, len(prompts)
assert cases.groupby("strategy").size().eq(4).all()
counts = pd.DataFrame(prompts).groupby(["strategy", "condition"]).size()
assert counts.eq(4).all(), counts
assert len({p["prompt_id"] for p in prompts}) == len(prompts)
print("screen ready: 32 cases, 64 prompts per Qwen mode")
print(cases.groupby(["data_block", "instance_id"]).size().reset_index(
    name="strategy_count").to_string(index=False))
PY
