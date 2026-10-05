"""Scientific checks for the derived analysis, separate from frozen results."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analysis_metrics import (answer_averaging, correction_residuals, noise_cases, noise_components, paired_comparisons,
                              performance_spread, variance_shares)


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


def test_performance_sd_measures_network_differences_with_equal_weights():
    per = pd.DataFrame([
        dict(group='real', arm='R', method='mle', source='small', MAE_2=.1, answers=3),
        dict(group='real', arm='R', method='mle', source='large', MAE_2=.3, answers=900),
        dict(group='real', arm='R', method='mle', source='invalid', MAE_2=np.nan, answers=0)])
    a = performance_spread(per).iloc[0]
    assert a['mean'] == pytest.approx(20)
    assert a.sd == pytest.approx(np.sqrt(200))
    assert a.networks == 2


def test_performance_means_match_frozen_real_twin_and_synthetic_scores():
    final = Path(__file__).resolve().parents[1]/'docs/results/final'
    a = performance_spread(pd.read_csv(final/'PER_SOURCE.csv'))
    expected = pd.read_csv(final/'SUMMARY.csv').set_index(['group','arm','method']).MAE_2*100
    np.testing.assert_allclose(a['mean'], expected.loc[a.index], atol=1e-10)


def test_paired_comparisons_match_the_frozen_sign_flip_tests():
    final = Path(__file__).resolve().parents[1]/'docs/results/final'
    a = paired_comparisons(pd.read_csv(final/'PER_SOURCE.csv')).set_index(['group', 'arm', 'first', 'second'])
    frozen = pd.read_csv(final/'PAIRED_METHODS.csv')
    for row in frozen.itertuples():
        mine = a.loc[(row.group, row.arm, *row.comparison.split(' vs '))]
        assert mine.networks == row.sources and mine.first_better == row.first_better
        assert mine.mean_difference_pp == pytest.approx(100*row.mean_difference)
        assert mine.exact_signflip_p == pytest.approx(row.exact_signflip_p)


def test_averaging_three_answers_scores_their_mean_and_never_exceeds_the_single_error():
    a = answer_averaging(predictions('gpt_6_sol', [[10, 20, 60]])).iloc[0]   # truth 20
    assert a.one_answer == pytest.approx(50/3)
    assert a.mean_of_3 == pytest.approx(10)
    final = Path(__file__).resolve().parents[1]/'docs/results/final'
    frozen = answer_averaging(pd.read_csv(final/'PREDICTIONS.csv'))
    assert (frozen.mean_of_3 <= frozen.one_answer + 1e-12).all()
    assert frozen.samples.between(35, 36).all()


def test_variance_shares_tell_network_from_method():
    rows = [dict(group='real', arm=arm, source=s, method=m, MAE_2=(i if arm == 'R' else j))
            for arm in 'RS' for i, s in enumerate('abc') for j, m in enumerate(('mle', 'et'))]
    a = variance_shares(pd.DataFrame(rows), ['mle', 'et'])
    assert a.loc['R'].round(9).tolist() == [1, 0, 0]      # errors differ only between networks
    assert a.loc['S'].round(9).tolist() == [0, 1, 0]      # errors differ only between methods
