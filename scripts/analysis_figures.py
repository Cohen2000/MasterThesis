"""Figures for docs/analysis/ANALYSIS.md.

In plain words: draws the figures of the written analysis from the frozen results in
docs/results/final. Nothing in docs/results/final is changed.

Two steps:
  python scripts/analysis_figures.py --inputs   rebuild docs/analysis/data/*.csv
  python scripts/analysis_figures.py            draw docs/analysis/figures/*.png and *.pdf

The --inputs step needs files that are not in the repository: the raw networks in
data/raw, the frozen API observations (default location
~/.local/share/masterthesis). Its small output tables are kept in the repository, so the
figures can be redrawn without them.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
FINAL = ROOT/'docs/results/final'
OUT = ROOT/'docs/analysis'
DATA, FIGS = OUT/'data', OUT/'figures'
EXTERNAL = Path.home()/'.local/share/masterthesis'

ARMS = {'R': 'R · random nodes', 'S': 'S · random walk', 'H': 'H · late time only', 'B': 'B · event loss'}
METHODS = {'plugin': 'No correction', 'mle': 'MLE', 'et': 'ExtraTrees', 'gpt_6_sol': 'GPT',
           'gpt_6_sol_tools': 'GPT + Python', 'deepseek_flash': 'DeepSeek', 'qwen_thinking': 'Qwen thinking',
           'qwen_nonthinking': 'Qwen no thinking'}
MAIN = ['mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']
LLMS = ['gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
NAMES = {'copenhagen_bluetooth': 'Copenhagen (Bluetooth)', 'lkml_reply': 'Linux mailing list',
         'nr_digg_reply': 'Digg replies', 'nr_radoslaw_email': 'Radoslaw (email)', 'reality_mining': 'Reality Mining',
         'snap_collegemsg': 'College messages', 'snap_email_eu': 'Email EU', 'snap_mathoverflow': 'MathOverflow',
         'sp_highschool2013': 'High school', 'sp_hospital': 'Hospital', 'sp_malawi': 'Malawi (village)',
         'sp_workplace': 'Workplace'}
GREY, BLUE, ORANGE, VIOLET = '#9c9b95', '#2a78d6', '#eb6834', '#4a3aa7'
EASY, MEDIUM, HARD = '#bfe5c9', '#f7dc8a', '#f08a86'
COLOUR = {'plugin': GREY, 'mle': BLUE, 'et': BLUE}


# ---------------------------------------------------------------- inputs (needs data outside the repo)
def build_inputs(external):
    """Network features and answer types, as small CSV tables."""
    from study.data import prepare_real
    from study.observation import parse
    from study.estimators import design_estimate
    from pipeline.core import CFG
    from pipeline.real_networks import load_raw, checked_graph
    DATA.mkdir(parents=True, exist_ok=True)
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    obs = {p.stem: json.loads(p.read_text()) for p in sorted((external/'api_observations').glob('*.json'))}

    # Network features from the raw data; the rebuilt truth must equal TRUTH.json exactly.
    rows = []
    tmp = tempfile.TemporaryDirectory()
    for key in NAMES:
        spec = CFG['stage2_sources'].get(key, {})
        if 'file' in spec:
            g = checked_graph(key, load_raw(key, spec, ROOT/'data/raw')[0], spec['proximity'])[0]
        else:
            g = prepare_real(key, ROOT/'data/raw', Path(tmp.name)/key)
        if abs(g.truth[0] - truth[key][0]) > 1e-12: raise ValueError(f'{key}: rebuilt truth differs from TRUTH.json')
        m = g.m
        blocks = {a: [parse(o['block']) for o in obs.values() if o['graph_id'] == key and o['arm'] == a] for a in 'RB'}
        rows.append({'source': key, 'nodes': g.N, 'pairs': g.D, 'events': g.M, 'rho2': g.truth[0],
                     'rho2_shuffled': truth[key+'__pwt'][0], 'events_per_pair': g.M/g.D,
                     'one_off_pairs': float(np.mean(m == 1)),
                     'only_early_pairs': float(np.mean(g.counts[:, 2:].sum(axis=1) == 0)),
                     'observed_pairs_R': float(np.mean([b['D_obs'] for b in blocks['R']])),
                     'kept_share_B': float(np.mean([b['parameter'] for b in blocks['B']]))})
    walk = pd.read_csv(FINAL/'WALK.csv').set_index('graph_id')
    f = pd.DataFrame(rows).set_index('source')
    f['walk_distinct_pairs'] = walk.loc[f.index, 'distinct_dyads_mean']
    f['walk_revisit_share'] = walk.loc[f.index, 'revisit_rate_mean']
    f.to_csv(DATA/'network_features.csv', float_format='%.6g')

    # Every language-model answer: near which simple reference value does its rho_2 lie (within 0.5 pp)?
    # Checked in this order: the S reweighting (each walk traversal counts 1 / events of its pair),
    # the observed share, the MLE estimate; otherwise "other".
    pred = pd.read_csv(FINAL/'PREDICTIONS.csv')
    pred = pred[(pred.group == 'real') & pred.prediction.notna()]
    first = lambda v: json.loads(v)[0]
    pred['r2'] = pred.prediction.map(first)
    design = {i: design_estimate(parse(o['block']))[0] for i, o in obs.items() if o['arm'] == 'S'}
    ref = {m: pred[pred.method == m].groupby('observation_id').r2.first() for m in ('plugin', 'mle')}
    out = []
    for method in LLMS:
        for arm in ARMS:
            a = pred[(pred.method == method) & (pred.arm == arm) & (pred.valid == True)]
            v, oid = a.r2.to_numpy(), a.observation_id
            near_rw = np.abs(v - oid.map(design).to_numpy()) <= .005 if arm == 'S' else np.zeros(len(v), bool)
            near_obs = (np.abs(v - oid.map(ref['plugin']).to_numpy()) <= .005) & ~near_rw
            near_mle = (np.abs(v - oid.map(ref['mle']).to_numpy()) <= .005) & ~near_rw & ~near_obs
            out.append({'method': method, 'arm': arm, 'answers': len(a), 'reweighting': near_rw.mean(),
                        'observed_share': near_obs.mean(), 'mle': near_mle.mean(),
                        'other': 1 - near_rw.mean() - near_obs.mean() - near_mle.mean()})
    pd.DataFrame(out).to_csv(DATA/'answer_types.csv', index=False, float_format='%.4f')


# ---------------------------------------------------------------- figures
def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False, 'axes.edgecolor': '#bbbbbb',
                         'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                         'xtick.color': '#555555', 'ytick.color': '#222222', 'ytick.left': False,
                         'savefig.dpi': 200, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})
    return plt


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIGS/f'{name}.{ext}')


def colour(method): return COLOUR.get(method, ORANGE)


def band(v): return EASY if v < 3 else MEDIUM if v <= 10 else HARD


def legend_bands(fig, plt, y=-.02):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (EASY, MEDIUM, HARD)]
    fig.legend(handles, ['error < 3 pp', 'error 3–10 pp', 'error > 10 pp'], loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, y))


def unframe(ax):
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.xaxis.tick_top()


def value_grid(plt, ax, v, rows, cols, vmax, show_rows=True):
    """Table-like heatmap with the value written in every cell; NaN cells stay empty."""
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            x = v[i, j]
            if np.isnan(x): continue
            a = min(1, x/vmax)
            ax.add_patch(plt.Rectangle((j-.47, i-.43), .94, .86, color=plt.cm.Purples(.06 + .8*a), lw=0))
            ax.text(j, i, f'{x:.1f}', ha='center', va='center', fontsize=10, color='white' if a > .55 else '#222222')
    ax.set_xlim(-.5, v.shape[1]-.5); ax.set_ylim(v.shape[0]-.5, -.5); unframe(ax)
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows if show_rows else ['']*len(rows))


# Fig. 1: the problem. Direction of the error without correction and with MLE.
def fig_bias(plt, summary):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    arms = list(ARMS)[::-1]
    y = np.arange(len(arms))
    for k, (m, c) in enumerate((('plugin', GREY), ('mle', BLUE))):
        v = [100*summary.loc[(a, m), 'signed_rho_2'] for a in arms]
        yy = y + (.19 if k == 0 else -.19)
        ax.barh(yy, v, height=.36, color=c, label=METHODS[m] + (' (observed share)' if m == 'plugin' else ''))
        for yi, vi in zip(yy, v):
            ax.text(vi + (.7 if vi >= 0 else -.7), yi, f'{vi:+.1f}', va='center', ha='left' if vi >= 0 else 'right', fontsize=9.5)
    ax.axvline(0, color='#333333', lw=1)
    ax.set_yticks(y, [ARMS[a] for a in arms]); ax.set_xlim(-24, 34)
    ax.set_xlabel('← estimate too low          mean signed error (pp)          estimate too high →')
    ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig1_bias')


# Fig. 2: mean error per arm; dashed line = no correction.
def fig_error_by_arm(plt, summary):
    methods = ['plugin', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        v = [100*summary.loc[(arm, m), 'MAE_2'] for m in methods]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, v, color=[colour(m) for m in methods], height=.68)
        ax.axvline(v[0], color='#555555', lw=1, ls=(0, (4, 3)), zorder=3)
        for yi, vi in zip(y, v): ax.text(vi+.6, yi, f'{vi:.1f}', va='center', fontsize=9.5, zorder=4,
                                         bbox=dict(facecolor='white', edgecolor='none', pad=.6))
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlim(0, 36); ax.set_title(ARMS[arm])
        ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True); ax.set_xlabel('error (pp)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, BLUE, ORANGE)] + \
              [plt.Line2D([], [], color='#555555', lw=1, ls=(0, (4, 3)))]
    fig.legend(handles, ['no correction (observed share)', 'classical method', 'language model', 'level without correction'],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.0))
    save(fig, 'fig2_error_by_arm')


# Fig. 3: near which simple reference value do the language-model answers lie?
def fig_answer_types(plt, types):
    t = types.set_index(['arm', 'method'])
    cats = (('observed_share', GREY, 'observed share'), ('reweighting', BLUE, 'simple reweighting (S)'),
            ('mle', '#86b6ef', 'MLE estimate'), ('other', '#e4e2da', 'none of these'))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    y = np.arange(len(LLMS))[::-1]
    for ax, arm in zip(axes, ARMS):
        left = np.zeros(len(LLMS))
        for col, c, _ in cats:
            v = np.array([100*t.loc[(arm, m), col] for m in LLMS])
            ax.barh(y, v, left=left, color=c, height=.66)
            for yi, l, vi in zip(y, left, v):
                if vi >= 12: ax.text(l+vi/2, yi, f'{vi:.0f}', ha='center', va='center', fontsize=9,
                                     color='white' if c in (GREY, BLUE) else '#333333')
            left += v
        ax.set_yticks(y, [METHODS[m] for m in LLMS]); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
        ax.set_title(ARMS[arm]); ax.set_xlabel('share of answers (%)'); ax.spines['bottom'].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in cats]
    fig.legend(handles, [lab for _, _, lab in cats], loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, 1.0), title='answer lies within 0.5 pp of', title_fontsize=10.5)
    save(fig, 'fig3_answer_types')


# Fig. 4: same sample, asked again (MLE recomputed, ExtraTrees retrained, language models answering again).
def fig_repeat(plt, resp, train):
    rows = ['mle', 'et'] + LLMS
    labels = ['MLE (recomputed)', 'ExtraTrees (retrained)'] + [METHODS[m] for m in LLMS]
    v = np.array([[0.]*4 if m == 'mle' else [100*train.loc[a, 'median_observation_SD_rho2'] for a in ARMS] if m == 'et'
                  else [100*resp.loc[(a, m), 'median_observation_SD_rho2'] for a in ARMS] for m in rows])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    value_grid(plt, ax, v, labels, list(ARMS), 22)
    ax.set_xlabel('spread of repeated estimates for the identical sample (median SD, pp)'); ax.xaxis.set_label_position('bottom')
    save(fig, 'fig4_repeat_spread')


# Fig. 5: arm S, error against the number of distinct pairs one walk sees.
def fig_walk(plt, per, f):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = f.walk_distinct_pairs
    for m, c in (('mle', BLUE), ('gpt_6_sol', ORANGE)):
        y = per[(per.method == m) & (per.arm == 'S')].set_index('source').MAE_2.loc[f.index]*100
        ax.scatter(x, y, s=50, color=c, edgecolor='white', lw=.8, zorder=3, label=METHODS[m])
    top = np.maximum(*(per[(per.method == m) & (per.arm == 'S')].set_index('source').MAE_2.loc[f.index]*100
                       for m in ('mle', 'gpt_6_sol')))
    for s, dx, dy, ha in (('sp_malawi', 7, 4, 'left'), ('sp_hospital', 7, 4, 'left'), ('reality_mining', 7, 4, 'left'),
                          ('copenhagen_bluetooth', -8, 10, 'right'), ('nr_digg_reply', 0, 16, 'center'),
                          ('lkml_reply', -8, 4, 'right')):
        ax.annotate(NAMES[s].split(' (')[0], (x[s], top[s]), xytext=(dx, dy), textcoords='offset points', fontsize=9.5, ha=ha)
    ax.set_xscale('log'); ax.set_ylim(0, 35); ax.spines['left'].set_visible(True)
    ax.set_xlabel('distinct pairs one walk sees (mean, log scale)'); ax.set_ylabel('error in arm S (pp)')
    ax.grid(color='#eeeeee'); ax.set_axisbelow(True); ax.legend(frameon=False, loc='upper right')
    save(fig, 'fig5_walk_pairs')


def method_heat(plt, per, order, rows, name, header, breaks=()):
    """One small heatmap per method: rows = networks, columns = arms, colour = error level."""
    fig, axes = plt.subplots(1, len(MAIN), figsize=(15.5, .42*len(rows)+.9))
    for k, (ax, m) in enumerate(zip(axes, MAIN)):
        v = per[per.method == m].pivot(index='source', columns='arm', values='MAE_2').loc[order, list(ARMS)].to_numpy()*100
        for i in range(v.shape[0]):
            for j in range(4):
                ax.add_patch(plt.Rectangle((j-.46, i-.42), .92, .84, color=band(v[i, j]), lw=0))
                ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=8.5,
                        fontweight='bold' if v[i, j] > 10 else 'normal')
        for b in breaks: ax.axhline(b, color='#333333', lw=1.1)
        ax.set_xlim(-.5, 3.5); ax.set_ylim(len(rows)-.5, -.5); unframe(ax)
        ax.set_xticks(range(4), list(ARMS)); ax.set_title(METHODS[m], pad=22)
        ax.set_yticks(range(len(rows)), rows if k == 0 else ['']*len(rows))
    fig.text(.005, 1 - .55/fig.get_figheight(), header, fontsize=9.5, color='#666666')
    legend_bands(fig, plt, -.3/fig.get_figheight())
    save(fig, name)


def fig_networks(plt, per, f):
    order = f.sort_values('rho2').index.tolist()
    rows = [f"{NAMES[s]}  ({100*f.loc[s, 'rho2']:.0f} %)" if f.loc[s, 'rho2'] >= .01 else f"{NAMES[s]}  (0.3 %)" for s in order]
    method_heat(plt, per, order, rows, 'figA1_networks', 'network (true ρ₂)')


def fig_synthetic(plt, per):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    variants = (('dar_a0', 'DAR, no memory (α = 0)'), ('dar_a08', 'DAR, memory (α = 0.8)'),
                ('ad_memoryless', 'Activity-driven, no memory'), ('ad_memory', 'Activity-driven, memory'))
    order = [f'{v}_r{i}' for v, _ in variants for i in (1, 2)]
    rows = [f'{label} · {i}  ({100*truth[f"{v}_r{i}"][0]:.0f} %)' for v, label in variants for i in (1, 2)]
    method_heat(plt, per, order, rows, 'figA2_synthetic', 'variant · instance (true ρ₂)', breaks=(1.5, 3.5, 5.5))


def fig_real_vs_shuffled(plt, perall):
    methods = ['mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']
    per = perall.assign(family=perall.source.str.replace('__pwt', '', regex=False))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
    rows = []
    for ax, arm in zip(axes, ARMS):
        y = np.arange(len(methods))[::-1]
        for yi, m in zip(y, methods):
            p = per[(per.arm == arm) & (per.method == m)].pivot(index='family', columns='group', values='MAE_2')*100
            r, s = p.real.mean(), p.surrogate.mean(); better = int((p.real < p.surrogate).sum())
            rows.append({'arm': arm, 'method': m, 'real': r, 'shuffled': s, 'real_lower': better})
            c = colour(m)
            ax.plot([r, s], [yi, yi], color=c, lw=2)
            ax.scatter([s], [yi], s=52, facecolor='white', edgecolor=c, lw=2, zorder=3)
            ax.scatter([r], [yi], s=56, color=c, zorder=3)
            ax.text(45, yi, f'{better}/12', va='center', ha='right', fontsize=9.5, color='#444444')
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlim(0, 45); ax.set_xticks([0, 10, 20, 30]); ax.set_title(ARMS[arm])
        ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True); ax.set_xlabel('error (pp)')
    h = [plt.Line2D([], [], marker='o', ls='', color='#555555', markersize=8),
         plt.Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=2, markersize=8)]
    fig.legend(h, ['real network', 'time-shuffled copy'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    pd.DataFrame(rows).to_csv(DATA/'real_vs_shuffled.csv', index=False, float_format='%.2f')
    save(fig, 'figA3_real_vs_shuffled')


def fig_sample_spread(plt, samp):
    rows = ['plugin', 'mle', 'et'] + LLMS
    v = np.array([[100*samp.loc[(a, m), 'median_graph_SD_rho2_across_draws'] for a in ARMS] for m in rows])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    value_grid(plt, ax, v, [METHODS[m] for m in rows], list(ARMS), 22)
    ax.set_xlabel('spread across the 3 samples of a network (median SD, pp)'); ax.xaxis.set_label_position('bottom')
    save(fig, 'figA4_sample_spread')


def draw():
    plt = setup()
    for old in FIGS.glob('*'): old.unlink()
    summary = pd.read_csv(FINAL/'SUMMARY.csv').query("group == 'real'").set_index(['arm', 'method'])
    perall = pd.read_csv(FINAL/'PER_SOURCE.csv')
    per = perall.query("group == 'real'")
    resp = pd.read_csv(FINAL/'VARIABILITY_RESPONSE.csv').query("group == 'real'").set_index(['arm', 'method'])
    samp = pd.read_csv(FINAL/'VARIABILITY_SAMPLING.csv').query("group == 'real'").drop_duplicates().set_index(['arm', 'method'])
    train = pd.read_csv(FINAL/'VARIABILITY_TRAINING.csv').query("group == 'real'").set_index('arm')
    f = pd.read_csv(DATA/'network_features.csv', index_col=0)
    types = pd.read_csv(DATA/'answer_types.csv')
    fig_bias(plt, summary); fig_error_by_arm(plt, summary); fig_answer_types(plt, types)
    fig_repeat(plt, resp, train); fig_walk(plt, per, f)
    fig_networks(plt, per, f); fig_synthetic(plt, perall.query("group == 'synthetic'"))
    fig_real_vs_shuffled(plt, perall[perall.group.isin(['real', 'surrogate'])]); fig_sample_spread(plt, samp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
