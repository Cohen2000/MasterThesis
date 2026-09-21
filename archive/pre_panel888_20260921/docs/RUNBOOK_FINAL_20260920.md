# Final-only completion and reproduction

Design/generation `cells10-final-20260920`; source freeze
`aaaac491006aa6981aeebc95b913b8847da76120`. No further submission is needed.
The completed workspace is
`uc3:/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_final_20260920/mainexp`.

Download its `archive/` into `results/imported_final_20260920/`. Never mix in H/S
development imports or use `integrate_srw_qwen.py` for this final design.

```bash
.venv/bin/python scripts/audit_final_study.py
.venv/bin/python scripts/collect_qwen_answers.py --run results/main_experiment/cells10_final_20260920 --answers results/imported_final_20260920/answers --out results/main_experiment/cells10_final_20260920_qwen/responses.jsonl
.venv/bin/python scripts/evaluate_main_responses.py --run results/main_experiment/cells10_final_20260920 --baselines results/main_experiment/cells10_final_20260920_qwen/primary_baselines.json --responses results/main_experiment/cells10_final_20260920_qwen/responses.jsonl --out results/main_experiment/cells10_final_20260920_qwen/evaluation
.venv/bin/python scripts/verify_main_offline.py --run results/main_experiment/cells10_final_20260920 > /tmp/final_offline_verify.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v > /tmp/final_study_tests.log 2>&1
.venv/bin/python scripts/audit_prompt_freeze.py
# Refresh /tmp/final_slurm.psv with sacct for 7066386..7066389;
# retain results/final_bundle_provenance/{BUNDLE_SHA256SUMS,WORKTREE_STATUS,SPEC_COMMIT}.
.venv/bin/python scripts/report_final_study.py
```

The final baseline file is a verified metadata adaptation of the existing fixed
SRW-reference predictions, not a refit. Do not rerun the old revision pipeline
under changed constants. See [audit exceptions and release requirements](AUDIT_FINAL_20260920.md).
