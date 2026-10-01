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
import re
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
METHODS = {'plugin': 'Naive share', 'median': 'Training median', 'mle': 'MLE', 'et': 'ExtraTrees', 'gpt_6_sol': 'GPT',
           'gpt_6_sol_tools': 'GPT + Python', 'deepseek_flash': 'DeepSeek', 'qwen_thinking': 'Qwen thinking',
           'qwen_nonthinking': 'Qwen no thinking'}
MAIN = ['mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']
LLMS = ['gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
NAMES = {'copenhagen_bluetooth': 'Copenhagen (Bluetooth)', 'lkml_reply': 'Linux mailing list',
         'nr_digg_reply': 'Digg replies', 'nr_radoslaw_email': 'Radoslaw (email)', 'reality_mining': 'Reality Mining',
         'snap_collegemsg': 'College messages', 'snap_email_eu': 'Email EU', 'snap_mathoverflow': 'MathOverflow',
         'sp_highschool2013': 'High school', 'sp_hospital': 'Hospital', 'sp_malawi': 'Malawi (village)',
         'sp_workplace': 'Workplace'}
GREY, BLUE, ORANGE, VIOLET, AQUA = '#9c9b95', '#2a78d6', '#eb6834', '#4a3aa7', '#1baf7a'
EASY, MEDIUM, HARD = '#bfe5c9', '#f7dc8a', '#f08a86'
COLOUR = {'plugin': GREY, 'mle': BLUE, 'et': AQUA}


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
        blocks = {a: [parse(o['block']) for o in obs.values() if o['graph_id'] == key and o['arm'] == a] for a in ARMS}
        rows.append({'source': key, 'nodes': g.N, 'pairs': g.D, 'events': g.M, 'rho2': g.truth[0],
                     'rho2_shuffled': truth[key+'__pwt'][0], 'events_per_pair': g.M/g.D,
                     'one_off_pairs': float(np.mean(m == 1)),
                     'only_early_pairs': float(np.mean(g.counts[:, 2:].sum(axis=1) == 0)),
                     **{f'pairs_seen_{a}': float(np.mean([b['D_obs'] for b in blocks[a]])) for a in ARMS},
                     'kept_share_B': float(np.mean([b['parameter'] for b in blocks['B']]))})
    walk = pd.read_csv(FINAL/'WALK.csv').set_index('graph_id')
    f = pd.DataFrame(rows).set_index('source')
    f['walk_distinct_pairs'] = walk.loc[f.index, 'distinct_dyads_mean']
    f['walk_revisit_share'] = walk.loc[f.index, 'revisit_rate_mean']
    f.to_csv(DATA/'network_features.csv', float_format='%.6g')

    # Synthetic test networks: regenerated from their fixed seeds; the truth must equal TRUTH.json.
    from study.synthetic import generate_pair
    syn = []
    for family in ('dar', 'ad'):
        for replicate in (1, 2):
            for g, _, _ in generate_pair(family, replicate):
                if abs(g.truth[0] - truth[g.key][0]) > 1e-12: raise ValueError(f'{g.key}: truth differs')
                syn.append({'source': g.key, 'nodes': g.N, 'pairs': g.D, 'events': g.M,
                            'events_per_pair': g.M/g.D, 'rho2': g.truth[0]})
    pd.DataFrame(syn).set_index('source').to_csv(DATA/'synthetic_features.csv', float_format='%.6g')

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

    # Under the hood (real networks): reasoning length, DeepSeek's full traces, GPT + Python's executed code.
    rows = []
    for run, method in (('openai', 'gpt_6_sol'), ('openai_tools', 'gpt_6_sol_tools'), ('deepseek', 'deepseek_flash')):
        for line in (external/'api_runs'/run/'responses.jsonl').read_text().splitlines():
            d = json.loads(line)
            oid = d['id'].rsplit('__', 2)[0]
            if d.get('kind') != 'main' or obs[oid]['stratum'] != 'real': continue
            code = '\n'.join(o.get('code', '') for o in (d.get('raw_response') or {}).get('output', [])
                             if o.get('type') == 'code_interpreter_call')
            rows.append({'method': method, 'arm': obs[oid]['arm'], 'reasoning_tokens': d.get('reasoning_tokens') or 0,
                         'trace_says_guess': 'guess' in (d.get('reasoning_content') or '').lower(),
                         'code_fits_model': 'optimize' in code or 'minimize' in code})
    # GPT + Python in B: which model ingredients each answer's code uses (by the distributions it names), and
    # whether the three answers to the same sample use the same set.
    kinds = {'gamma': r'gamma', 'log-normal': r'hermgauss|lognorm|roots_herm', 'beta': r'\bbeta\b|betaln',
             'dirichlet': r'dirichlet', 'latent classes': r'classes|latent class|n_classes|multi_mon'}
    sets = {}
    for line in (external/'api_runs/openai_tools/responses.jsonl').read_text().splitlines():
        d = json.loads(line)
        oid = d['id'].rsplit('__', 2)[0]
        if d.get('kind') != 'main' or obs[oid]['stratum'] != 'real' or obs[oid]['arm'] != 'B': continue
        code = '\n'.join(o.get('code', '') for o in (d.get('raw_response') or {}).get('output', [])
                         if o.get('type') == 'code_interpreter_call')
        if code: sets.setdefault(oid, []).append(frozenset(k for k, pat in kinds.items() if re.search(pat, code, re.I)))
    full = [v for v in sets.values() if len(v) == 3]
    pd.DataFrame([{'samples_with_code_in_all_3_answers': len(full),
                   'samples_whose_3_answers_differ': sum(len(set(v)) > 1 for v in full)}]
                 ).to_csv(DATA/'python_models_B.csv', index=False)
    r = pd.DataFrame(rows).groupby(['method', 'arm'])
    pd.DataFrame({'answers': r.size(), 'median_reasoning_tokens': r.reasoning_tokens.median(),
                  'trace_says_guess': r.trace_says_guess.mean(), 'code_fits_model': r.code_fits_model.mean()}
                 ).to_csv(DATA/'under_the_hood.csv', float_format='%.4f')


# ---------------------------------------------------------------- figures
LIGHT = '#cdc9bf'   # training median, a constant guess


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


def colour(method): return {'median': LIGHT}.get(method, COLOUR.get(method, ORANGE))


def band(v): return EASY if v < 3 else MEDIUM if v <= 10 else HARD


def legend_bands(fig, plt, y=-.02):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (EASY, MEDIUM, HARD)]
    fig.legend(handles, ['error < 3 pp', 'error 3–10 pp', 'error > 10 pp'], loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, y))


def unframe(ax):
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.xaxis.tick_top()


def value_grid(plt, ax, v, rows, cols, vmax, show_rows=True, muted=()):
    """Table-like heatmap with the value written in every cell."""
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]): continue
            a = min(1, v[i, j]/vmax)
            if i in muted:
                ax.text(j, i, f'{v[i, j]:.0f}', ha='center', va='center', fontsize=10, color='#bbbbbb'); continue
            ax.add_patch(plt.Rectangle((j-.47, i-.43), .94, .86, color=plt.cm.Purples(.06 + .8*a), lw=0))
            ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=10, color='white' if a > .55 else '#222222')
    ax.set_xlim(-.5, v.shape[1]-.5); ax.set_ylim(v.shape[0]-.5, -.5); unframe(ax)
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows if show_rows else ['']*len(rows))


# Fig. 0: a toy network of six pairs and what each arm shows of it.
TOY = {'A': [3, 3, 2, 3, 3], 'B': [1, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 1],
       'D': [1, 0, 0, 0, 0], 'E': [0, 0, 1, 0, 0], 'F': [0, 0, 0, 0, 1]}
TOY_ARMS = (('R · random nodes', {'B': TOY['B'], 'C': TOY['C'], 'D': TOY['D'], 'E': TOY['E']}, (), 'only pairs between drawn nodes'),
            ('S · random walk', {'A': TOY['A'], 'B': TOY['B'], 'C': TOY['C'], 'E': TOY['E']}, (), 'the walk mostly meets busy pairs'),
            ('H · late time only', {k: [0, 0] + v[2:] for k, v in TOY.items() if sum(v[2:])}, (0, 1), 'windows 1–2 hidden (node sampling not shown)'),
            ('B · event loss', {'A': [1, 0, 1, 0, 1], 'B': [0, 0, 1, 0, 0], 'E': [0, 0, 1, 0, 0]}, (), 'most events are lost'))


def toy_panel(plt, ax, pairs, hidden, title, note):
    for j in hidden: ax.add_patch(plt.Rectangle((j-.5, -.5), 1, 6, color='#e9e7e1', lw=0))
    seen = 0; persistent = 0
    for i, k in enumerate(TOY):
        row = pairs.get(k)
        ax.text(-.9, i, f'pair {k}', ha='right', va='center', fontsize=9.5, color='#222222' if row else '#c4c1b8')
        for j in range(5):
            ax.scatter(j, i, s=14, color='#dddad2', zorder=1)
            if row and row[j]: ax.scatter(j, i, s=40 + 35*row[j], color='#333333', zorder=3)
        if row:
            seen += 1; persistent += sum(x > 0 for x in row) >= 2
    ax.set_xlim(-.6, 4.6); ax.set_ylim(5.6, -.6); ax.axis('off')
    ax.set_title(title, loc='left', fontsize=11, pad=6)
    ax.text(2, 6.25, note, ha='center', fontsize=9, color='#666666')
    return seen, persistent


def fig_toy(plt):
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=.75, wspace=.55)
    top = fig.add_subplot(gs[0, 1:3])
    toy_panel(plt, top, TOY, (), 'Full network: 6 pairs, 5 time windows', 'dot = active in that window; bigger dot = more events')
    top.set_title('Full network: 6 pairs, 5 time windows', loc='left', fontsize=11, pad=24)
    for j in range(5): top.text(j, -.95, f'window {j+1}', ha='center', fontsize=9, color='#666666')
    top.text(2, 7.0, 'true ρ₂ = 3 of 6 pairs active in ≥ 2 windows = 50 %', ha='center', fontsize=10.5, fontweight='bold')
    for k, (title, pairs, hidden, note) in enumerate(TOY_ARMS):
        ax = fig.add_subplot(gs[1, k])
        seen, persistent = toy_panel(plt, ax, pairs, hidden, title, note)
        ax.text(2, 7.0, f'naive share = {persistent}/{seen} = {100*persistent/seen:.0f} %', ha='center', fontsize=10.5, fontweight='bold')
    save(fig, 'fig0_toy')


# Fig. 1: how the sample looks. Signed error of the naive share, per arm and per network.
def fig_sample(plt, summary, per):
    fig, ax = plt.subplots(figsize=(8, 3.3))
    arms = list(ARMS)[::-1]
    for y, a in enumerate(arms):
        v = 100*summary.loc[(a, 'plugin'), 'signed_rho_2']
        ax.barh(y, v, height=.55, color=GREY)
        dots = 100*per[(per.method == 'plugin') & (per.arm == a)].signed_rho_2
        ax.scatter(dots, y + np.linspace(-.12, .12, len(dots)), s=16, color='#333333', zorder=3, lw=0)
        ax.text(v + (1 if v >= 0 else -1), y + .38, f'{v:+.1f}', ha='left' if v >= 0 else 'right', fontsize=10, fontweight='bold')
    ax.axvline(0, color='#333333', lw=1)
    ax.set_yticks(range(4), [ARMS[a] for a in arms]); ax.set_xlim(-45, 45)
    ax.set_xlabel('naive share − true ρ₂ (pp)')
    ax.text(-44, 3.55, '← sample looks less persistent', fontsize=9.5, color='#555555', va='center')
    ax.text(44, 3.55, 'sample looks more persistent →', fontsize=9.5, color='#555555', va='center', ha='right')
    ax.set_ylim(-.5, 3.8)
    ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Line2D([], [], marker='o', ls='', color='#333333', markersize=4)]
    ax.legend(h, ['mean of 12 networks', 'one network'], frameon=False, loc='lower right', fontsize=9.5)
    save(fig, 'fig1_sample')


# Fig. 2: ranking per arm. Everything above the naive share improves on doing nothing.
def fig_ranking(plt, summary, per):
    methods = ['plugin', 'median', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    for ax, arm in zip(axes, ARMS):
        ranked = sorted(methods, key=lambda m: summary.loc[(arm, m), 'MAE_2'])
        for i, m in enumerate(ranked):
            v, c = 100*summary.loc[(arm, m), 'MAE_2'], colour(m)
            base = m == 'plugin'
            e = per[per.arm == arm].pivot(index='source', columns='method', values='MAE_2')
            wins = '' if base else f'{int((e[m] < e.plugin - .005).sum())}/12'
            ax.add_patch(plt.Rectangle((0, i-.42), 1, .84, color=GREY if base else '#f4f3f0', lw=0))
            ax.add_patch(plt.Rectangle((.03, i+.2), .94*min(v, 55)/55, .14, color='white' if base else c, lw=0))
            ax.text(.03, i-.08, METHODS[m], va='center', fontsize=10,
                    color='white' if base else '#222222', fontweight='bold' if base else 'normal')
            ax.text(.74, i-.08, wins, va='center', ha='right', fontsize=9.5, color='#777777')
            ax.text(.97, i-.08, f'{v:.1f}', va='center', ha='right', fontsize=10,
                    color='white' if base else '#222222', fontweight='bold' if base else 'normal')
        ax.text(.74, -.75, 'beats naive*', ha='right', fontsize=8.5, color='#777777')
        ax.text(.97, -.75, 'error', ha='right', fontsize=8.5, color='#777777')
        ax.set_xlim(0, 1); ax.set_ylim(len(ranked)-.5, -.5); ax.axis('off')
        ax.set_title(ARMS[arm], loc='left', pad=16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, LIGHT, BLUE, AQUA, ORANGE)]
    fig.legend(handles, ['naive share', 'constant guess', 'statistical model', 'trained model', 'language model'],
               loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.text(.5, -.075, 'sorted by error (pp) · bar length = error · *networks (of 12) where the method beats the naive share by more than 0.5 pp',
             ha='center', fontsize=9.5, color='#666666')
    save(fig, 'fig2_ranking')


# Fig. 6: error against the number of pairs in the sample, every arm; dot = median of the six methods.
def fig_sample_size(plt, per, f):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        e = per[(per.arm == arm) & per.method.isin(MAIN)].groupby('source').MAE_2
        mid = e.median()*100
        x = f.loc[mid.index, f'pairs_seen_{arm}']
        ax.scatter(x, mid, s=44, color='#333333', zorder=3)
        for s in ('sp_malawi', 'copenhagen_bluetooth'):
            ax.annotate(NAMES[s].split(' (')[0], (x[s], mid[s]), xytext=(6, 2), textcoords='offset points', fontsize=9)
        ax.set_xscale('log'); ax.set_xlim(10, 50000); ax.set_ylim(0, 35); ax.set_title(ARMS[arm])
        ax.set_xlabel('pairs in the sample (log)'); ax.grid(color='#eeeeee'); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('error (pp)')
    fig.legend([plt.Line2D([], [], marker='o', ls='', color='#333333', markersize=6)],
               ['one network: median error of MLE, ExtraTrees, GPT, GPT + Python, DeepSeek, Qwen thinking'],
               loc='lower center', frameon=False, bbox_to_anchor=(.5, .98))
    save(fig, 'fig6_sample_size')


# Fig. 3: near which simple estimate do the language-model answers lie?
def fig_answer_types(plt, types):
    t = types.set_index(['arm', 'method'])
    cats = (('observed_share', GREY, 'naive share'), ('reweighting', BLUE, 'simple reweighting (S)'),
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


# Fig. 4: how far the language models correct, relative to the correction the sample needs.
CORRECTION = (('wrong direction', '#d9383a'), ('too little', '#f2b46d'), ('about right', '#1f9d55'), ('too much', '#86b6ef'))


def correction_types(pred):
    """Per answer: needed = truth − naive share, done = answer − naive share; samples off by ≥ 5 pp only."""
    p = pred[(pred.group == 'real') & pred.prediction.notna()].copy()
    p['r2'] = p.prediction.map(lambda v: json.loads(v)[0])
    naive = p[p.method == 'plugin'].groupby('observation_id').r2.first()
    a = p[p.method.isin(LLMS) & (p.valid == True)].copy()
    a['need'] = a.truth_rho2 - a.observation_id.map(naive)
    a['done'] = a.r2 - a.observation_id.map(naive)
    a = a[a.need.abs() >= .05]
    ratio = a.done/a.need
    a['type'] = pd.cut(ratio, [-np.inf, 0, .5, 1.5, np.inf], right=False, labels=[c for c, _ in CORRECTION])
    t = a.groupby(['arm', 'method']).type.value_counts(normalize=True).unstack()
    t.to_csv(DATA/'correction_types.csv', float_format='%.4f')
    return t


def fig_correction(plt, pred):
    t = correction_types(pred)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    y = np.arange(len(LLMS))[::-1]
    for ax, arm in zip(axes, 'SHB'):
        left = np.zeros(len(LLMS))
        for cat, c in CORRECTION:
            v = np.array([100*t.loc[(arm, m), cat] for m in LLMS])
            ax.barh(y, v, left=left, color=c, height=.66)
            for yi, l, vi in zip(y, left, v):
                if vi >= 12: ax.text(l+vi/2, yi, f'{vi:.0f}', ha='center', va='center', fontsize=9,
                                     color='white' if cat in ('wrong direction', 'about right') else '#222222')
            left += v
        ax.set_yticks(y, [METHODS[m] for m in LLMS]); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
        ax.set_title(ARMS[arm]); ax.set_xlabel('share of answers (%)'); ax.spines['bottom'].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CORRECTION]
    fig.legend(handles, ['wrong direction', 'too little (< 50 %)', 'about right (50–150 %)', 'too much (> 150 %)'],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.0),
               title='correction made, as a share of the correction needed', title_fontsize=10.5)
    save(fig, 'fig4_correction')


# Fig. 5: stability. Left: the identical sample again. Right: a new sample (deterministic methods only; with 3 samples x 3 answers
# the language models' sample effect cannot be separated from their answer noise).
def fig_stability(plt, resp, train, samp):
    rows = ['plugin', 'mle', 'et'] + LLMS
    labels = ['Naive share', 'MLE', 'ExtraTrees*'] + [METHODS[m] for m in LLMS]
    same = np.array([[0.]*4 if m in ('plugin', 'mle') else [100*train.loc[a, 'median_observation_SD_rho2'] for a in ARMS]
                     if m == 'et' else [100*resp.loc[(a, m), 'median_observation_SD_rho2'] for a in ARMS] for m in rows])
    new = np.array([[100*samp.loc[(a, m), 'median_graph_SD_rho2_across_draws'] if m in ('plugin', 'mle', 'et')
                     else np.nan for a in ARMS] for m in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    titles = ('Same sample, asked again', 'New sample of the same network')
    for k, (ax, v, title) in enumerate(zip(axes, (same, new), titles)):
        value_grid(plt, ax, v, labels, list(ARMS), 22, show_rows=k == 0, muted=(0, 1) if k == 0 else ())
        ax.set_title(title, pad=24)
    axes[1].text(1.5, 5, 'language models: not separable\nfrom their answer noise', ha='center', va='center', fontsize=10, color='#999999')
    fig.text(.5, -.03, 'spread of the ρ₂ estimate (median standard deviation, pp) · *left: ExtraTrees retrained on new training data',
             ha='center', fontsize=10.5, color='#444444')
    save(fig, 'fig5_stability')


def method_heat(plt, per, order, rows, name, header):
    """One small heatmap per method: rows = networks, columns = arms, colour = error level."""
    fig, axes = plt.subplots(1, len(MAIN), figsize=(15.5, .42*len(rows)+.9))
    for k, (ax, m) in enumerate(zip(axes, MAIN)):
        v = per[per.method == m].pivot(index='source', columns='arm', values='MAE_2').loc[order, list(ARMS)].to_numpy()*100
        for i in range(v.shape[0]):
            for j in range(4):
                ax.add_patch(plt.Rectangle((j-.46, i-.42), .92, .84, color=band(v[i, j]), lw=0))
                ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=8.5,
                        fontweight='bold' if v[i, j] > 10 else 'normal')
        ax.set_xlim(-.5, 3.5); ax.set_ylim(len(rows)-.5, -.5); unframe(ax)
        ax.set_xticks(range(4), list(ARMS)); ax.set_title(METHODS[m], pad=22)
        ax.set_yticks(range(len(rows)), rows if k == 0 else ['']*len(rows))
    fig.text(.005, 1 - .55/fig.get_figheight(), header, fontsize=9.5, color='#666666')
    legend_bands(fig, plt, -.3/fig.get_figheight())
    save(fig, name)


# Fig. 8: error per real network, arm and method.
def fig_networks(plt, per, f):
    order = f.sort_values('rho2').index.tolist()
    rows = [f"{NAMES[s]}  ({100*f.loc[s, 'rho2']:.0f} %)" if f.loc[s, 'rho2'] >= .01 else f"{NAMES[s]}  (0.3 %)" for s in order]
    method_heat(plt, per, order, rows, 'fig8_networks', 'network (true ρ₂)')


# Fig. 7: error against true rho_2 for all 32 networks (y = median error of the six methods).
def fig_persistence(plt, perall):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    med = perall[perall.method.isin(MAIN)].groupby(['source', 'arm']).MAE_2.median().unstack()*100
    group = perall.groupby('source').group.first()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for g, kw in (('real', dict(color='#333333')), ('surrogate', dict(facecolor='white', edgecolor='#333333', lw=1.5)),
                      ('synthetic', dict(color=ORANGE, marker='^', s=60))):
            idx = group[group == g].index
            ax.scatter([100*truth[s][0] for s in idx], med.loc[idx, arm], zorder=3, **{'s': 40, **kw})
        ax.set_xlim(0, 90); ax.set_ylim(0, 35); ax.set_title(ARMS[arm]); ax.set_xlabel('true ρ₂ (%)')
        ax.grid(color='#eeeeee'); ax.set_axisbelow(True); ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('median error of six methods (pp)')
    h = [plt.Line2D([], [], marker='o', ls='', color='#333333', markersize=7),
         plt.Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='#333333', markeredgewidth=1.5, markersize=7),
         plt.Line2D([], [], marker='^', ls='', color=ORANGE, markersize=8)]
    fig.legend(h, ['real network', 'time-shuffled copy', 'synthetic network'],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .98))
    save(fig, 'fig7_persistence')
    # Numbers for the timing table: median error of the six methods, real against shuffled copy.
    rows = []
    for arm in ARMS:
        real = med.loc[list(NAMES), arm]; shuf = med.loc[[s+'__pwt' for s in NAMES], arm].to_numpy()
        rows.append({'arm': arm, 'real': real.mean(), 'shuffled': shuf.mean(), 'shuffled_harder': int((shuf > real.to_numpy()).sum())})
    pd.DataFrame(rows).to_csv(DATA/'real_vs_shuffled.csv', index=False, float_format='%.2f')


def draw():
    plt = setup()
    for old in FIGS.glob('*'): old.unlink()
    summary = pd.read_csv(FINAL/'SUMMARY.csv').query("group == 'real'").set_index(['arm', 'method'])
    perall = pd.read_csv(FINAL/'PER_SOURCE.csv')
    per = perall.query("group == 'real'")
    pred = pd.read_csv(FINAL/'PREDICTIONS.csv')
    resp = pd.read_csv(FINAL/'VARIABILITY_RESPONSE.csv').query("group == 'real'").set_index(['arm', 'method'])
    samp = pd.read_csv(FINAL/'VARIABILITY_SAMPLING.csv').query("group == 'real'").drop_duplicates().set_index(['arm', 'method'])
    train = pd.read_csv(FINAL/'VARIABILITY_TRAINING.csv').query("group == 'real'").set_index('arm')
    f = pd.read_csv(DATA/'network_features.csv', index_col=0)
    types = pd.read_csv(DATA/'answer_types.csv')
    fig_toy(plt); fig_sample(plt, summary, per); fig_ranking(plt, summary, per); fig_answer_types(plt, types)
    fig_correction(plt, pred); fig_stability(plt, resp, train, samp); fig_sample_size(plt, per, f)
    fig_persistence(plt, perall); fig_networks(plt, per, f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations and api_runs')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
