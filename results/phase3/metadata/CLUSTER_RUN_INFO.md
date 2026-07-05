# Cluster Run Info

## Cluster

- Cluster: bwUniCluster 3.0
- User: tu_zxokn55
- Workspace: /pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
- Pilot directory: /pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/pilot
- GPU type used: NVIDIA H100
- Main partition: gpu_h100_short

## Models

Final comparison models:

- Qwen/Qwen2.5-14B-Instruct
- Qwen/Qwen2.5-32B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

## Files

Main result files:

- results/answers_qwen14b.jsonl
- results/answers_qwen32b.jsonl
- results/answers_r1_32b.jsonl

Supporting files:

- results/pilot_cases.csv
- results/prompts.jsonl

## Task

Estimate temporal edge persistence rho.

rho = fraction of true edges active in at least two time windows.

Observed w>=2 is only a lower bound because limited temporal crawling can observe an edge in only one window even if the edge is truly active in multiple windows.

## Prompt conditions

Each model has 180 outputs:

- 60 hidden
- 60 disclosed
- 60 disclosed_calib

Total:

- 60 cases × 3 prompt conditions = 180 outputs per model

## Execution setup

The gpu_h100_short partition rejected time limits above 30 minutes.

Full runs therefore used:

- time limit: 00:30:00

Jobs were submitted as sequential dependency chains using afterany dependencies. Jobs for the same model were not run in parallel.

Submitted chains:

Qwen14B:
- 5735792
- 5735793

Qwen32B:
- 5735794
- 5735795
- 5735796
- 5735797
- 5735798

R1-32B:
- 5735799
- 5735800
- 5735801
- 5735802
- 5735803
- 5735804
- 5735805
- 5735806
- 5735807
- 5735808
- 5735809
- 5735810
- 5735811
- 5735812

R1 jobs reached TIMEOUT repeatedly because each 30-minute job generated as many prompts as possible and the next dependency job resumed from the JSONL output file. Final result files are complete.

## Final technical validation

Final squeue was empty.

Final row counts:

- Qwen14B: 180
- Qwen32B: 180
- R1-32B: 180

Validation:

Qwen14B:
- rows: 180
- conditions: hidden 60, disclosed 60, disclosed_calib 60
- pred None: 0
- duplicates: 0
- forced: 0
- pred range: 0.0 to 0.5

Qwen32B:
- rows: 180
- conditions: hidden 60, disclosed 60, disclosed_calib 60
- pred None: 0
- duplicates: 0
- forced: 0
- pred range: 0.0 to 0.45

R1-32B:
- rows: 180
- conditions: hidden 60, disclosed 60, disclosed_calib 60
- pred None: 0
- duplicates: 0
- forced: 178
- pred range: 0.0 to 0.568

## R1-specific note

DeepSeek-R1-Distill-Qwen-32B used a v3 runner with budget forcing.

- think budget: 2048 tokens
- if no FINAL answer appeared, the runner forced finalization with a FINAL continuation

R1 required forced finalization in 178/180 prompts. This should be considered a relevant inference/runtime behavior.
