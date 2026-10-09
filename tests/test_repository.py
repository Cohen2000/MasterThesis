"""The parts of the repository fit together: every program imports, and the README shows the final numbers."""
import csv
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# fetch_tokenizers.py and build_training_pool.py start working when imported, so they are left out.
PRODUCTION = ('add_panel_size', 'api_runner', 'check_random_walk', 'extratrees', 'model_requests', 'prepare_study',
              'run_guards', 'run_pipeline', 'run_qwen_engine', 'score_stage1', 'verify_qwen_answers',
              'pipeline.task_graph', 'pipeline.report')
SCRIPTS = ('evaluate', 'window_sensitivity', 'analysis_figures', 'analysis_tables')


@pytest.mark.parametrize('module', PRODUCTION + SCRIPTS)
def test_every_program_imports(module):
    importlib.import_module(module)


def test_readme_shows_the_final_errors():
    """The main table of the README repeats SUMMARY.csv (real networks, MAE_2, three decimals)."""
    order = ('plugin', 'median', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking', 'deepseek_flash', 'gpt_6_sol',
             'gpt_6_sol_tools')
    final = {(r['arm'], r['method']): f"{float(r['MAE_2']):.3f}"
             for r in csv.DictReader((ROOT/'docs/results/final/SUMMARY.csv').open()) if r['group'] == 'real'}
    readme = (ROOT/'README.md').read_text()
    for arm in 'RSHB':
        row = re.search(rf'^\| {arm} \|(.*)\|$', readme, re.M).group(1)
        assert [c.strip() for c in row.split('|')] == [final[arm, m] for m in order]
