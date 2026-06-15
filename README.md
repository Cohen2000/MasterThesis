# Masterarbeit

Current focus: estimating temporal graph properties from walk-based summaries.

## Current frozen experiment

Main Run 1 is the current frozen synthetic benchmark.

## Structure

- src/      code for census, generator, walks, benchmark creation, walk summaries, evaluation
- tests/    unit tests for generator and walks
- config/   dataset registry
- data/     raw datasets and generated synthetic instances
- results/  census results and Main Run 1 outputs
- materials/ papers and notes/email material for NotebookLM
- figures/  pipeline and presentation figures

## Main Run 1 files

- data/synthetic/main_run1/manifest.csv
- data/synthetic/main_run1/instances/
- results/main_run1/calibration_report.md
- results/main_run1/summaries_main.csv
- results/main_run1/main_run1_results.csv
- results/main_run1/main_run1_mae_vs_budget.png
- results/main_run1/main_run1_r2_vs_budget.png
