# Pre-cleanup source snapshot (2026-09-21)

Exact copies of `src/`, `scripts/`, `cluster/`, `config/`, `docs/`, `tests/` and the
root README as they stood after the panel888 implementation and before the code
cleanup (uncommitted working tree on top of commit b95fa3e). Raw data, model
weights and result trees are not duplicated.

The cleanup changed structure, not science. After it, the full offline study was
recomputed from scratch and compared with the pre-cleanup run: all graphs,
budgets, 608 observation blocks and prompts, the 3,456-request manifest, all
18 ExtraTrees pickles and all 1,800 main reference predictions were byte- or
value-identical (see `CLEANUP_EQUIVALENCE.json`).

Replaced or removed from the active tree:
- `src/main_experiment/pipeline.py` (prepare stage with a duplicate ExtraTrees fit) -> `prepare.py`
- `scripts/run_baseline_revision.py` -> `src/main_experiment/references.py` + `scripts/build_references.py`
  (stages decompose/decompose_main -> `scripts/diagnose_decomposition.py`)
- `scripts/run_panel888_offline.sh`, `finalize_panel888_offline.sh`, `run_main_offline.py`,
  `run_main_offline_cpu.sbatch` -> `scripts/run_offline.sh`
- `scripts/verify_main_offline.py`, `audit_panel888.py`, `audit_prompt_freeze.py`,
  `build_panel888_seed_manifest.py` -> `scripts/audit_offline.py`
- `scripts/analyze_htime.py`, `analyze_srw.py`, `analyze_panel888_controls.py`,
  `analyze_panel888_census.py` -> `scripts/diagnose_{history,srw,null_model,windows,mixture_bounds}.py`
- `scripts/seal_panel888.py`, `check_panel888_seal.py` -> `scripts/seal_offline.py [--verify]`
- `scripts/evaluate_main_responses.py`, `evaluate_panel888_pairs.py`, `check_main_evaluation_mocks.py`,
  `run_main_api.py`, `fetch_main_tokenizers.py` -> renamed `evaluate_responses.py`,
  `evaluate_paired_controls.py`, `check_evaluation_mocks.py`, `run_api.py`, `fetch_tokenizers.py`
- `scripts/integrate_panel888_qwen.sh`, `verify_panel888_archive.py`, `report_panel888.py`
  -> `integrate_qwen.sh`, `verify_qwen_archive.py`, `audit_qwen.py`, `report_qwen.py`
- legacy modules `src/walks.py`, `nonwalk_samplers.py`, `generator.py`, `benchmark_generators.py`,
  `persistence_prompt.py`, `persistence_evaluation.py` and their tests
- legacy arm H_suffix_v1, event budget, recent-cap/reservoir helpers, suffix mixture candidate,
  resume/checkpoint machinery, `config/main_experiment/rule_H_suffix_panel_v1.txt`
- `tests/frozen_main/`, `tests/{data,targets,sampling,generators,prompting,evaluation}/`
  -> one flat test module per scientific module

`implementation_attempts/` holds the two earlier panel888 implementation passes
(their result trees stay local, git-ignored). Nothing here is final evidence.
