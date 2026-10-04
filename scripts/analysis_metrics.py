"""Small analysis tables from frozen predictions; all quantities are in pp.

ExtraTrees fit 0 is the reported model. Other fits measure training variation,
never repeated answers or sampling variation. Missing/invalid LLM answers are
excluded; an answer SD requires all three valid repeats.
"""
import json

import numpy as np
import pandas as pd

METHODS = ['mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash',
           'qwen_thinking', 'qwen_nonthinking']
LLMS = METHODS[2:]
KEY = ['source', 'arm', 'method']


def performance_spread(per_source):
    """Mean and SD of network MAEs, in pp; every network has equal weight.

    This measures variation across networks, not sampling or answer noise.
    PER_SOURCE already averages the valid production estimates per network.
    """
    p = per_source.assign(error_pp=100*per_source.MAE_2)
    return p.groupby(['group', 'arm', 'method']).error_pp.agg(
        mean='mean', sd='std', networks='count')


def primary_predictions(pred, group='real'):
    p = pred[(pred.group == group) & pred.valid & pred.prediction.notna()
             & (pred.replicate.isna() | pred.replicate.eq(0))].copy()
    profile = np.array([json.loads(x) for x in p.prediction])*100
    for k in range(4):
        p[f'rho{k + 2}'] = profile[:, k]
    return p


def noise_cases(pred):
    """Per network, sampler, method and target: controlled contrasts.

    A random-effects estimate subtracts within-sample variance / 3 from the
    variance of the three sample means. It assumes independent repeats and
    samples; three of each make this a rough estimate. Keep negative variance
    estimates in the CSV, and truncate only after pooling when displaying SD.
    """
    p = primary_predictions(pred)
    rows = []
    for (source, arm, method), a in p[p.method.isin(METHODS)].groupby(KEY):
        for k in range(2, 6):
            col = f'rho{k}'
            obs = a.groupby('observation_id')[col].agg(['count', 'mean', 'var'])
            stochastic = method in LLMS
            full = obs['count'].eq(3 if stochastic else 1)
            answer_var = obs.loc[full, 'var'].mean() if stochastic else 0.
            answer_sd = np.sqrt(obs.loc[full, 'var']).median() if stochastic else 0.
            # Require the complete 3 x 3 design for variance attribution.
            complete = len(obs) == 3 and full.all()
            between_var = obs['mean'].var() if len(obs) >= 2 else np.nan
            balanced_between = between_var if complete else np.nan
            sample_var = balanced_between - answer_var/3 if stochastic else balanced_between
            rows.append(dict(source=source, arm=arm, method=method, target=k,
                             samples=len(obs), complete_answer_samples=int(full.sum()),
                             answer_sd=answer_sd, between_sd=np.sqrt(between_var),
                             answer_variance=answer_var, balanced_between_variance=balanced_between,
                             sample_variance_estimate=sample_var, complete=complete))
    return pd.DataFrame(rows)


def noise_components(cases):
    """Pool variances with equal network weights; return square roots (RMS SD)."""
    rows = []
    for (arm, method, target), g in cases[cases.complete].groupby(['arm', 'method', 'target']):
        sample = g.sample_variance_estimate.mean()
        rows.append(dict(arm=arm, method=method, target=target, networks=len(g),
                         between_sd=np.sqrt(g.balanced_between_variance.mean()),
                         answer_sd=np.sqrt(g.answer_variance.mean()),
                         sample_variance_estimate=sample,
                         sample_sd_estimate=np.sqrt(max(0., sample)),
                         negative_sample_variances=int((g.sample_variance_estimate < 0).sum())))
    return pd.DataFrame(rows)


def correction_residuals(pred):
    """One row per valid production prediction; residual = estimate - truth."""
    p = primary_predictions(pred)
    naive = p[p.method.eq('plugin')].set_index('observation_id').rho2
    a = p[p.method.isin(METHODS)].copy()
    a['needed'] = a.truth_rho2*100 - a.observation_id.map(naive)
    a['made'] = a.rho2 - a.observation_id.map(naive)
    a['residual'] = a.rho2 - a.truth_rho2*100
    return a
