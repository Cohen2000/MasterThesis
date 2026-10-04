"""Scientific checks for the derived analysis, separate from frozen results."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analysis_metrics import correction_residuals, noise_cases, noise_components


def predictions(method, samples, source='example'):
    return pd.DataFrame([dict(source=source, group='real', arm='B', method=method,
                              observation_id=f'{source}_{j}', valid=True, replicate=np.nan,
                              truth_rho2=.2, prediction=json.dumps([v/100, 0., 0., 0.]))
                         for j, answers in enumerate(samples) for v in answers])


def test_stable_estimator_can_vary_across_samples_and_remain_biased():
    p = predictions('mle', [[30], [32], [34]])
    c = noise_cases(p).query('target == 2').iloc[0]
    assert c.answer_sd == 0
    assert c.between_sd == pytest.approx(2)
    assert c.sample_variance_estimate == pytest.approx(4)
    assert correction_residuals(pd.concat([p, predictions('plugin', [[20], [20], [20]])])).residual.mean() == pytest.approx(12)


def test_variance_attribution_does_not_call_answer_noise_sample_noise():
    # Variation of [-1, 0, 1] around each mean has sample variance 1.
    p = predictions('gpt_6_sol', [[19, 20, 21], [29, 30, 31], [39, 40, 41]])
    c = noise_cases(p).query('target == 2')
    result = noise_components(c).iloc[0]
    assert result.answer_sd == pytest.approx(1)
    assert result.between_sd == pytest.approx(10)
    assert result.sample_variance_estimate == pytest.approx(100 - 1/3)
    assert result.sample_sd_estimate == pytest.approx(np.sqrt(100 - 1/3))


def test_negative_variance_is_retained_and_incomplete_repeats_are_excluded():
    p = predictions('gpt_6_sol', [[19, 20, 21]]*3)
    q = predictions('gpt_6_sol', [[19, 20, 21]]*3, source='incomplete')
    q.loc[q.index[-1], 'valid'] = False
    c = noise_cases(pd.concat([p, q], ignore_index=True)).query('target == 2')
    assert c.set_index('source').loc['incomplete', 'complete_answer_samples'] == 2
    result = noise_components(c).iloc[0]
    assert result.networks == 1
    assert result.sample_variance_estimate == pytest.approx(-1/3)
    assert result.sample_sd_estimate == 0


def test_extra_training_fits_do_not_enter_sampling_or_answer_variation():
    p = predictions('et', [[30], [32], [34]])
    p['replicate'] = 0
    q = p.copy()
    q['replicate'] = 1
    q['prediction'] = json.dumps([.99, .99, .99, .99])
    c = noise_cases(pd.concat([p, q], ignore_index=True)).query('target == 2').iloc[0]
    assert c.samples == 3
    assert c.answer_sd == 0
    assert c.between_sd == pytest.approx(2)


def test_derived_errors_and_sampling_spreads_match_frozen_reports():
    final = Path(__file__).resolve().parents[1]/'docs/results/final'
    p = pd.read_csv(final/'PREDICTIONS.csv')
    a = correction_residuals(p)
    e = a.groupby(['arm', 'method', 'source']).residual.apply(lambda v: v.abs().mean()).groupby(['arm', 'method']).mean()
    expected = pd.read_csv(final/'SUMMARY.csv').query("group == 'real'").set_index(['arm', 'method']).MAE_2*100
    np.testing.assert_allclose(e, expected.loc[e.index], atol=1e-10)
    c = noise_cases(p).query('target == 2').groupby(['arm', 'method']).between_sd.median()
    expected = pd.read_csv(final/'VARIABILITY_SAMPLING.csv').query("group == 'real'").set_index(['arm', 'method']).median_graph_SD_rho2_across_draws*100
    np.testing.assert_allclose(c, expected.loc[c.index], atol=1e-10)
