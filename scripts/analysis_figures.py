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
                     # what the pairs look like: one event, several events in one window, active in 2..5 windows
                     'share_one_event': float(np.mean(m == 1)),
                     'share_one_window_several': float(np.mean((g.K == 1) & (m > 1))),
                     **{f'share_{k}_windows': float(np.mean(g.K == k)) for k in range(2, 6)},
                     # pairs that carry the events: as many equally busy pairs would hold them (inverse Simpson index of event shares)
                     'effective_pairs': float(1/np.sum((m/m.sum())**2)),
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
                            'events_per_pair': g.M/g.D, 'rho2': g.truth[0],
                            'effective_pairs': float(1/np.sum((g.m/g.M)**2)),
                            'share_one_event': float(np.mean(g.m == 1)),
                            'share_one_window_several': float(np.mean((g.K == 1) & (g.m > 1))),
                            **{f'share_{k}_windows': float(np.mean(g.K == k)) for k in range(2, 6)}})
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
    s = pred[(pred.arm == 'S') & pred.method.isin(LLMS) & (pred.valid == True)]
    near = (s.r2 - s.observation_id.map(design)).abs() <= .005
    near.groupby([s.method, s.source]).mean().rename('share').to_csv(DATA/'textbook_by_network.csv', float_format='%.4f')

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
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIGS/f'{name}.{ext}')
    plt.close(fig)


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


# ---------------------------------------------------------------- shared pieces
REAL_NOTE = '12 real networks · 3 samples each'
LM3 = (('gpt_6_sol', ORANGE), ('deepseek_flash', VIOLET), ('qwen_thinking', '#e87ba4'))


def note(fig, text, y=-.04):
    """Small grey line under a figure saying which networks and samples the numbers rest on."""
    fig.text(.01, y, text, fontsize=9, color='#888888', ha='left', va='top')


def network_order(f):
    """Real networks from the most persistent (top) to the least persistent (bottom)."""
    return f.sort_values('rho2').index.tolist()


def short(s): return NAMES[s].split(' (')[0]


def num(x):
    """Compact count: 347, 4,274, 14k, 2.4M."""
    return f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k' if x >= 1e4 else f'{x:,.0f}'


def pct(r): return f'{100*r:.1f} %' if r < .01 else f'{100*r:.0f} %'


def label(s, f):
    """Row label of a real network: name · pairs · persistence."""
    return f"{short(s)} · {num(f.loc[s, 'pairs'])} pairs · ρ₂ {pct(f.loc[s, 'rho2'])}"


def specs(s, f, sep=' · '):
    """Central numbers of a network: nodes, pairs, events, persistence."""
    return (f"{num(f.loc[s, 'nodes'])} nodes · {num(f.loc[s, 'pairs'])} pairs{sep}"
            f"{num(f.loc[s, 'events'])} events · ρ₂ {pct(f.loc[s, 'rho2'])}")


def typical_error(perall):
    """Median error of the six main methods, per network and arm (pp)."""
    return perall[perall.method.isin(MAIN)].groupby(['source', 'arm']).MAE_2.median().unstack()*100


def r2_answers(pred, group='real'):
    p = pred[(pred.group == group) & pred.prediction.notna()].copy()
    p['r2'] = p.prediction.map(lambda v: json.loads(v)[0])*100
    return p


def network_dots(plt, ax, order, values, xlab, xlim, f):
    """One row per network; one coloured dot per language model."""
    y = np.arange(len(order))
    for yi in y: ax.axhline(yi, color='#f0efeb', lw=6, zorder=0)
    for k, (m, c) in enumerate(LM3):
        v = [values[m].get(s, np.nan) for s in order]
        ax.scatter(v, y + (k - 1)*.2, s=42, color=c, edgecolor='white', lw=.7, zorder=3, label=METHODS[m])
    ax.set_yticks(y, [label(s, f) for s in order]); ax.set_xlim(*xlim); ax.set_xlabel(xlab)
    ax.tick_params(axis='y', length=0); ax.set_ylim(-.6, len(order)-.4)


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
    note(fig, REAL_NOTE, -.08)
    save(fig, 'fig1_sample')


# Fig. 2: error per arm, methods sorted from best to worst; the naive share is the grey row.
def fig_ranking(plt, summary):
    methods = ['plugin', 'median', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, arm in zip(axes, ARMS):
        ranked = sorted(methods, key=lambda m: summary.loc[(arm, m), 'MAE_2'])[::-1]
        v = [100*summary.loc[(arm, m), 'MAE_2'] for m in ranked]
        y = np.arange(len(ranked))
        ax.barh(y, v, color=[colour(m) for m in ranked], height=.7)
        for yi, vi, m in zip(y, v, ranked):
            ax.text(vi + .8, yi, f'{vi:.1f}', va='center', fontsize=9.5, fontweight='bold' if m == 'plugin' else 'normal')
        ax.set_yticks(y, [METHODS[m] for m in ranked])
        for lab, m in zip(ax.get_yticklabels(), ranked):
            if m == 'plugin': lab.set_fontweight('bold')
        ax.set_xlim(0, 60); ax.set_xticks([]); ax.spines['bottom'].set_visible(False); ax.set_title(ARMS[arm])
    fig.subplots_adjust(wspace=.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, LIGHT, BLUE, AQUA, ORANGE)]
    fig.legend(handles, ['naive share', 'constant guess', 'statistical model', 'trained model', 'language model'],
               loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.06))
    note(fig, 'error (pp) · ' + REAL_NOTE + ' (language models: 3 answers per sample)', -.1)
    save(fig, 'fig2_ranking')


def grouped_bars(plt, ax, groups, values, ymax, fmt='{:.0f}', title=''):
    """values[model][group] -> bars; one colour per model, values written on top."""
    w = .8/len(LM3)
    for k, (m, c) in enumerate(LM3):
        xs = np.arange(len(groups)) + (k - (len(LM3)-1)/2)*w
        vs = [values[m][g] for g in groups]
        ax.bar(xs, vs, width=w*.92, color=c, label=METHODS[m])
        for x, v in zip(xs, vs): ax.text(x, v + ymax*.015, fmt.format(v), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(groups)), [ARMS[g] for g in groups]); ax.set_ylim(0, ymax); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0); ax.set_title(title, pad=12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.12))


# Fig. 3: share of answers that equal the textbook answer (R: the naive share; S: the simple reweighting).
def fig_textbook(plt, types):
    t = types.set_index(['arm', 'method'])
    vals = {m: {'R': 100*t.loc[('R', m), 'observed_share'], 'S': 100*t.loc[('S', m), 'reweighting']} for m, _ in LM3}
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    grouped_bars(plt, ax, ['R', 'S'], vals, 112, '{:.0f} %', 'Answers equal to the textbook answer')
    note(fig, REAL_NOTE + ' × 3 answers', -.2)
    save(fig, 'fig3_textbook')


def design_values(external):
    from study.observation import parse
    from study.estimators import design_estimate
    out = {}
    for path in (external/'api_observations').glob('*__S-*.json'):
        o = json.loads(path.read_text())
        out[o['id']] = 100*design_estimate(parse(o['block']))[0]
    return out


# Fig. 3b: arm S per network: share of answers equal to the simple reweighting.
def fig_textbook_networks(plt, pred, f):
    m = pd.read_csv(DATA/'textbook_by_network.csv').set_index(['method', 'source']).share
    order = network_order(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    network_dots(plt, ax, order, {mm: {s: 100*m[(mm, s)] for s in order} for mm, _ in LM3},
                 'answers equal to the simple reweighting (%)', (-5, 105), f)
    ax.set_title('S · random walk, per network', pad=12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.45, -.14))
    note(fig, 'networks sorted by true ρ₂ (top: highest) · 3 samples × 3 answers each', -.18)
    save(fig, 'fig3b_textbook_networks')


# Fig. 4: share of answers that correct by about the right amount (50–150 % of the needed correction).
def correction_table(pred):
    """Per answer: needed = truth − naive share, done = answer − naive share; samples off by ≥ 5 pp only."""
    p = r2_answers(pred)
    naive = p[p.method == 'plugin'].groupby('observation_id').r2.first()
    a = p[p.method.isin(LLMS) & (p.valid == True)].copy()
    a['need'] = 100*a.truth_rho2 - a.observation_id.map(naive)
    a['done'] = a.r2 - a.observation_id.map(naive)
    a = a[a.need.abs() >= 5]
    a['type'] = pd.cut(a.done/a.need, [-np.inf, 0, .5, 1.5, np.inf], right=False,
                       labels=['wrong direction', 'too little', 'about right', 'too much'])
    a.groupby(['arm', 'method']).type.value_counts(normalize=True).unstack().to_csv(DATA/'correction_types.csv', float_format='%.4f')
    return a


def fig_correction(plt, pred):
    a = correction_table(pred)
    t = a.groupby(['arm', 'method']).type.value_counts(normalize=True).unstack()
    vals = {m: {arm: 100*t.loc[(arm, m), 'about right'] for arm in 'SHB'} for m, _ in LM3}
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    grouped_bars(plt, ax, ['S', 'H', 'B'], vals, 95, '{:.0f} %', 'Answers that correct by about the right amount')
    note(fig, REAL_NOTE + ' × 3 answers · only samples off by ≥ 5 pp', -.2)
    save(fig, 'fig4_correction')


# Fig. 4b: per network, H and B together: share of answers that correct by about the right amount.
def fig_correction_networks(plt, pred, f):
    a = correction_table(pred)
    a = a[a.arm.isin(['H', 'B'])]
    share = a.groupby(['method', 'source']).type.apply(lambda s: 100*(s == 'about right').mean())
    count = a.groupby(['method', 'source']).size()
    order = network_order(f)
    vals = {m: {s: share[(m, s)] for s in order if (m, s) in share.index and count[(m, s)] >= 3} for m, _ in LM3}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    network_dots(plt, ax, order, vals, 'answers that correct by about the right amount (%)', (-5, 105), f)
    ax.set_title('H and B together, per network', pad=12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.45, -.14))
    note(fig, 'networks sorted by true ρ₂ · only samples off by ≥ 5 pp (Digg and Linux: none)', -.18)
    save(fig, 'fig4b_correction_networks')


# Fig. 4c: mean estimates of rho_2..rho_5 against the truth, per arm (12 real networks).
def fig_profile(plt, pred):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    p = pred[(pred.group == 'real') & pred.prediction.notna() & ((pred.valid == True) | (pred.method == 'plugin'))]
    p = p[p.method.isin(['plugin', 'gpt_6_sol'])]
    prof = pd.DataFrame([json.loads(v) for v in p.prediction], index=p.index, columns=[2, 3, 4, 5])*100
    mean = prof.groupby([p.arm, p.method, p.source]).mean().groupby(level=[0, 1]).mean()
    t = pd.DataFrame({s: truth[s] for s in p.source.unique()}, index=[2, 3, 4, 5]).T.mean()*100
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    k = np.arange(4)
    for ax, arm in zip(axes, ARMS):
        ax.plot(k, t.to_numpy(), color='#333333', lw=2.5, marker='o', label='truth')
        ax.plot(k, mean.loc[(arm, 'gpt_6_sol')].to_numpy(), color=ORANGE, lw=2.2, marker='o', label='GPT')
        ax.plot(k, mean.loc[(arm, 'plugin')].to_numpy(), color=GREY, lw=2.2, marker='o', ls=(0, (4, 2)), label='naive share')
        ax.set_xticks(k, ['ρ₂', 'ρ₃', 'ρ₄', 'ρ₅']); ax.set_title(ARMS[arm]); ax.set_ylim(0, 75)
        ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('share of pairs (%)')
    axes[0].legend(frameon=False, loc='upper right')
    note(fig, 'mean over 12 real networks · 3 samples (× 3 answers) each', -.06)
    save(fig, 'fig4c_profile')


# Fig. 5: spread of the three answers to the identical sample (median SD, pp).
def fig_stability(plt, resp):
    vals = {m: {a: 100*resp.loc[(a, m), 'median_observation_SD_rho2'] for a in ARMS} for m, _ in LM3}
    fig, ax = plt.subplots(figsize=(9, 3.3))
    grouped_bars(plt, ax, list(ARMS), vals, 25, '{:.1f}', 'Spread of 3 answers to the same sample (pp)')
    note(fig, REAL_NOTE + ' · MLE and ExtraTrees: always the same answer', -.2)
    save(fig, 'fig5_stability')


# Fig. 5b: per network, the spread of 3 answers to the same sample (mean over the four arms).
def fig_noise_networks(plt, pred, f):
    p = r2_answers(pred)
    p = p[p.method.isin([m for m, _ in LM3]) & (p.valid == True)]
    sd = p.groupby(['method', 'arm', 'source', 'observation_id']).r2.std().groupby(['method', 'arm', 'source']).median()
    v = sd.groupby(['method', 'source']).mean()
    order = network_order(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    network_dots(plt, ax, order, {m: {s: v[(m, s)] for s in order} for m, _ in LM3}, 'spread of 3 answers (pp)', (-1, 22), f)
    ax.set_title('Spread of 3 answers to the same sample, per network', pad=12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.45, -.14))
    note(fig, 'networks sorted by true ρ₂ · median over samples, mean over the four arms', -.18)
    save(fig, 'fig5b_noise_networks')


# Fig. 5c: a new sample of the same network: MLE's estimate from each of the 3 samples against the truth.
def fig_sample_noise(plt, pred, f):
    p = r2_answers(pred)
    p = p[p.method == 'mle']
    order = network_order(f)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    y = {s: i for i, s in enumerate(order)}
    for ax, arm in zip(axes, ARMS):
        for s in order:
            ax.axhline(y[s], color='#f0efeb', lw=6, zorder=0)
            ax.plot([100*f.loc[s, 'rho2']]*2, [y[s]-.35, y[s]+.35], color='#333333', lw=2.2, zorder=2)
            est = p[(p.arm == arm) & (p.source == s)].r2
            ax.scatter(est, [y[s]]*len(est), s=30, color=BLUE, edgecolor='white', lw=.6, zorder=3)
        ax.set_xlim(-2, 95); ax.set_title(ARMS[arm]); ax.set_xlabel('ρ₂ (%)'); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(order)), [label(s, f) for s in order]); axes[0].set_ylim(-.6, len(order)-.4)
    h = [plt.Line2D([], [], color='#333333', lw=2.2), plt.Line2D([], [], marker='o', ls='', color=BLUE, markersize=6)]
    fig.legend(h, ['true ρ₂', 'MLE estimate from one sample'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, '12 real networks · 3 independent samples each · MLE gives the same answer for the same sample', -.04)
    save(fig, 'fig5c_sample_noise')


# Fig. 6: GPT with and without Python, error per arm.
def fig_python(plt, summary):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    w = .38
    for k, (m, c) in enumerate((('gpt_6_sol', ORANGE), ('gpt_6_sol_tools', VIOLET))):
        xs = np.arange(4) + (k - .5)*w
        vs = [100*summary.loc[(a, m), 'MAE_2'] for a in ARMS]
        ax.bar(xs, vs, width=w*.92, color=c, label=METHODS[m])
        for x, v in zip(xs, vs): ax.text(x, v + .3, f'{v:.1f}', ha='center', va='bottom', fontsize=9.5)
    ax.set_xticks(range(4), [ARMS[a] for a in ARMS]); ax.set_ylim(0, 18); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, ncol=2, loc='upper left'); ax.set_title('Error (pp)', pad=12)
    note(fig, REAL_NOTE + ' × 3 answers', -.1)
    save(fig, 'fig6_python')


# Fig. 6b: per network, how much Python changes GPT's error (pp; right = worse with Python).
def fig_python_networks(plt, per, f):
    e = per[per.method.isin(['gpt_6_sol', 'gpt_6_sol_tools'])].pivot_table(index='source', columns=['arm', 'method'], values='MAE_2')*100
    order = network_order(f)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    y = np.arange(len(order))
    for ax, arm in zip(axes, ARMS):
        d = (e[(arm, 'gpt_6_sol_tools')] - e[(arm, 'gpt_6_sol')]).loc[order]
        for yi in y: ax.axhline(yi, color='#f0efeb', lw=6, zorder=0)
        ax.axvline(0, color='#333333', lw=1)
        ax.scatter(d, y, s=46, color=['#d9383a' if v > .5 else '#1f9d55' if v < -.5 else '#bbbbbb' for v in d], zorder=3,
                   edgecolor='white', lw=.7)
        ax.set_xlim(-8, 25); ax.set_title(ARMS[arm]); ax.set_xlabel('Python − GPT (pp)'); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, [label(s, f) for s in order]); axes[0].set_ylim(-.6, len(order)-.4)
    h = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=7) for c in ('#d9383a', '#bbbbbb', '#1f9d55')]
    fig.legend(h, ['worse with Python', 'about the same (±0.5 pp)', 'better with Python'], loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, 'networks sorted by true ρ₂ · ' + REAL_NOTE + ' × 3 answers', -.04)
    save(fig, 'fig6b_python_networks')


# Fig. 7: is a network hard for every method? MLE against GPT, one dot per real network.
def fig_agreement(plt, per, f):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    e = lambda m, arm: per[(per.method == m) & (per.arm == arm)].set_index('source').MAE_2*100
    for ax, arm in zip(axes, ARMS):
        x, y = e('mle', arm), e('gpt_6_sol', arm)
        y = y.loc[x.index]
        ax.plot([0, 35], [0, 35], color='#bbbbbb', lw=1, zorder=1)
        ax.scatter(x, y, s=46, color=ORANGE, edgecolor='white', lw=.7, zorder=3)
        if arm == 'B':
            ax.annotate(f"Copenhagen\n({num(f.loc['copenhagen_bluetooth', 'pairs'])} pairs, ρ₂ {pct(f.loc['copenhagen_bluetooth', 'rho2'])})",
                        (x['copenhagen_bluetooth'], y['copenhagen_bluetooth']), xytext=(4, 30),
                        textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='-', color='#888888', lw=.8))
        ax.set_xlim(0, 35); ax.set_ylim(0, 35); ax.set_aspect('equal'); ax.set_title(ARMS[arm])
        ax.set_xlabel('error of MLE (pp)'); ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('error of GPT (pp)')
    note(fig, 'one dot per real network · 3 samples each', -.02)
    save(fig, 'fig7_agreement')


# Fig. 8: typical error against the number of events (top) and the number of pairs that carry them (bottom), per arm.
def fig_structure(plt, perall, f):
    t = typical_error(perall).loc[f.index]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharey=True)
    rows = (('events', 'events in the network', [10**5, 10**6]),
            ('effective_pairs', 'pairs that carry the events*', [100, 1000, 10000, 100000]))
    for r, (col, xlab, ticks) in enumerate(rows):
        for ax, arm in zip(axes[r], ARMS):
            ax.scatter(f[col], t[arm], s=40, color='#333333', zorder=3)
            txt = 'Malawi (102k events)' if col == 'events' else 'Malawi (55 pairs carry them)'
            ax.annotate(txt, (f.loc['sp_malawi', col], t.loc['sp_malawi', arm]), xytext=(3, 9), textcoords='offset points',
                        fontsize=8.5, va='bottom')
            ax.set_xscale('log'); ax.set_xticks(ticks, [f'{x:,}' for x in ticks]); ax.minorticks_off()
            ax.set_xlabel(xlab, fontsize=9.5); ax.set_ylim(0, 36); ax.set_yticks([0, 10, 20, 30]); ax.spines['left'].set_visible(True)
            if r == 0: ax.set_title(ARMS[arm])
        axes[r][0].set_ylabel('typical error (pp)')
    fig.text(.005, .97, 'More events: no help', fontsize=11, fontweight='bold')
    fig.text(.005, .48, 'Events spread over more pairs: easier with random nodes and the walk', fontsize=11, fontweight='bold')
    fig.subplots_adjust(hspace=.65)
    note(fig, '12 real networks · typical error = median of six methods · *effective number of pairs (see definitions)', -.02)
    save(fig, 'fig8_structure')


# Fig. 9: typical error against true rho_2, all 32 networks, per arm.
def fig_persistence(plt, perall):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    t = typical_error(perall)
    rho = pd.Series({s: 100*truth[s][0] for s in t.index})
    groups = (('real', [s for s in t.index if s in NAMES], dict(color='#333333')),
              ('time-shuffled twin', [s for s in t.index if s.endswith('__pwt')], dict(color='white', edgecolor='#333333', lw=1.1)),
              ('synthetic', [s for s in t.index if s not in NAMES and not s.endswith('__pwt')], dict(color=BLUE)))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for lab, keys, style in groups: ax.scatter(rho[keys], t.loc[keys, arm], s=34, zorder=3, label=lab, **style)
        ax.set_xlim(0, 90); ax.set_ylim(0, 35); ax.set_title(ARMS[arm]); ax.set_xlabel('true ρ₂ (%)')
        ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('typical error (pp)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .98))
    note(fig, '32 networks: 12 real, their 12 time-shuffled twins, 8 synthetic · typical error = median of six methods', -.06)
    save(fig, 'fig9_persistence')


# Fig. 10: time-shuffled twins. How much each method raises its estimate from a network to its twin (truth: +27 pp).
def fig_twins(plt, pred):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    methods = ['plugin', 'mle', 'et', 'gpt_6_sol', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
    p = pred[pred.group.isin(['real', 'surrogate']) & pred.prediction.notna() & ((pred.valid == True) | pred.method.isin(['plugin', 'mle', 'et']))].copy()
    p = p[p.method.isin(methods) & (p.replicate.isna() | (p.replicate == 0))]
    p['r2'] = p.prediction.map(lambda v: json.loads(v)[0])*100
    p['family'] = p.source.str.replace('__pwt', '', regex=False)
    est = p.groupby(['arm', 'method', 'group', 'family']).r2.mean().unstack('group')
    change = (est.surrogate - est.real).groupby(['arm', 'method']).mean()
    true = np.mean([100*(truth[s+'__pwt'][0] - truth[s][0]) for s in NAMES])
    change.unstack().to_csv(DATA/'twins_change.csv', float_format='%.2f')
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, arm in zip(axes, ARMS):
        v = [change[(arm, m)] for m in methods]
        ax.barh(y, v, color=[colour(m) for m in methods], height=.65)
        ax.axvline(true, color='#333333', lw=1.5, ls=(0, (4, 2)))
        for yi, vi in zip(y, v): ax.text(max(vi, 0) + .8, yi, f'{vi:+.0f}', va='center', fontsize=9, zorder=4,
                                         bbox=dict(facecolor='white', edgecolor='none', pad=.5))
        ax.set_xlim(-14, 40); ax.set_title(ARMS[arm]); ax.set_xticks([0, 10, 20, 30])
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlabel('change in estimated ρ₂ (pp)')
    fig.legend([plt.Line2D([], [], color='#333333', lw=1.5, ls=(0, (4, 2)))], [f'true change (+{true:.0f} pp)'],
               loc='lower center', frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, '12 real networks and their time-shuffled twins · 3 samples (× 3 answers) each', -.05)
    save(fig, 'fig10_twins')


# Fig. 11: every real network on its own: typical error per arm.
def fig_cards(plt, perall, f):
    t = typical_error(perall)
    order = network_order(f)[::-1]
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.2), sharey=True)
    for ax, s in zip(axes.flat, order):
        v = [t.loc[s, a] for a in ARMS]
        ax.bar(range(4), v, color=['#7d7c76', '#5598e7', '#e0a100', '#d9383a'], width=.7)
        for i, x in enumerate(v): ax.text(i, x + .6, f'{x:.0f}', ha='center', fontsize=9)
        ax.set_xticks(range(4), list(ARMS)); ax.set_ylim(0, 36); ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.set_title(short(s), fontsize=10.5, loc='left', pad=30)
        ax.text(0, 1.04, specs(s, f, '\n'), transform=ax.transAxes, fontsize=8.5, color='#555555', linespacing=1.3)
    fig.subplots_adjust(hspace=.95, wspace=.25)
    note(fig, 'typical error (pp, median of six methods) per arm · 3 samples each', -.01)
    save(fig, 'fig11_cards')


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


# Detail figure (linked, not embedded): error per real network, arm and method.
def fig_networks(plt, per, f):
    order = f.sort_values('rho2').index.tolist()
    method_heat(plt, per, order, [label(s, f) for s in order], 'fig_networks_detail', 'error (pp) per network')


# Fig. 0b: what the pairs of each network look like (share of pairs, %).
PORTRAIT = (('share_one_event', '#d9d6cf', 'one event'), ('share_one_window_several', '#b3afa6', 'several events, one window'),
            ('share_2_windows', '#9ec5f4', '2 windows'), ('share_3_windows', '#5598e7', '3 windows'),
            ('share_4_windows', '#256abf', '4 windows'), ('share_5_windows', '#104281', '5 windows'))


def fig_portrait(plt, f):
    order = f.sort_values('rho2').index.tolist()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for col, c, lab in PORTRAIT:
        v = 100*f.loc[order, col].to_numpy()
        ax.barh(y, v, left=left, color=c, height=.72, label=lab)
        left += v
    for yi, s in zip(y, order):
        ax.text(101.5, yi, specs(s, f), va='center', fontsize=9, color='#444444')
    ax.set_yticks(y, [short(s) for s in order]); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('share of pairs (%)'); ax.spines['bottom'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0))
    note(fig, '12 real networks, complete (not sampled)', -.06)
    save(fig, 'fig0b_portrait')


# Synthetic networks: four variants (two generators, without and with memory), two instances each.
VARIANTS = (('dar_a0', 'DAR, no memory'), ('dar_a08', 'DAR, memory'),
            ('ad_memoryless', 'Activity-driven, no memory'), ('ad_memory', 'Activity-driven, memory'))


def variant_table():
    sy = pd.read_csv(DATA/'synthetic_features.csv', index_col=0)
    return pd.DataFrame({v: sy.loc[[f'{v}_r1', f'{v}_r2']].mean() for v, _ in VARIANTS}).T


# Fig. 12: what the pairs of the synthetic variants look like (same layout as fig. 1).
def fig_synthetic_portrait(plt):
    sv = variant_table()
    keys = [v for v, _ in VARIANTS][::-1]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    y = np.arange(len(keys)); left = np.zeros(len(keys))
    for col, c, lab in PORTRAIT:
        v = 100*sv.loc[keys, col].to_numpy()
        ax.barh(y, v, left=left, color=c, height=.68, label=lab)
        left += v
    for yi, k in zip(y, keys):
        ax.text(101.5, yi, f"{num(sv.loc[k, 'nodes'])} nodes · {num(sv.loc[k, 'pairs'])} pairs · {num(sv.loc[k, 'events'])} events · "
                           f"ρ₂ {pct(sv.loc[k, 'rho2'])}", va='center', fontsize=9, color='#444444')
    ax.set_yticks(y, [dict(VARIANTS)[k] for k in keys]); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('share of pairs (%)'); ax.spines['bottom'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0))
    note(fig, 'mean of the 2 instances per variant, complete networks (not sampled)', -.16)
    save(fig, 'fig12_synthetic_portrait')


# Fig. 13: what memory does to the estimation task: typical error per arm, without and with memory.
def fig_synthetic_arms(plt, perall):
    t = typical_error(perall)
    sv = variant_table()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    w = .38
    for ax, (name, without, with_) in zip(axes, (('DAR', 'dar_a0', 'dar_a08'), ('Activity-driven', 'ad_memoryless', 'ad_memory'))):
        for k, (key, c, lab) in enumerate(((without, '#b8b5ad', 'no memory'), (with_, '#333333', 'memory'))):
            v = [t.loc[[f'{key}_r1', f'{key}_r2'], a].mean() for a in ARMS]
            xs = np.arange(4) + (k - .5)*w
            ax.bar(xs, v, width=w*.92, color=c, label=f"{lab} · {num(sv.loc[key, 'pairs'])} pairs · ρ₂ {pct(sv.loc[key, 'rho2'])}")
            for x, vi in zip(xs, v): ax.text(x, vi + .3, f'{vi:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(4), [ARMS[a].replace(' · ', '\n') for a in ARMS]); ax.set_ylim(0, 19); ax.set_yticks([])
        ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0); ax.set_title(name, pad=10)
        ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    note(fig, 'typical error (pp, median of six methods) · 500 nodes and about 10k events each · 2 instances per variant · 3 samples each', -.04)
    save(fig, 'fig13_synthetic_arms')


# Fig. 14: true rho_2 of each real network for other numbers of time windows (pairs active in ≥ 2 windows).
def fig_windows(plt):
    w = pd.read_csv(FINAL/'W_SENSITIVITY_NETWORKS.csv').query("group == 'real'")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s, g in w.groupby('network'):
        g = g.sort_values('W')
        ax.plot(g.W, 100*g.rho_2, color='#9c9b95', lw=1.4, marker='o', markersize=3)
    mean = w.groupby('W').rho_2.mean()*100
    ax.plot(mean.index, mean.to_numpy(), color='#333333', lw=2.6, label='mean of 12 networks')
    ax.axvline(5, color=ORANGE, lw=1.2, ls=(0, (4, 2))); ax.text(5.3, 88, 'used: 5 windows', color=ORANGE, fontsize=9)
    ax.set_xlabel('number of time windows'); ax.set_ylabel('true ρ₂ (%)'); ax.set_ylim(0, 95); ax.set_xticks([2, 5, 10, 15, 20])
    ax.spines['left'].set_visible(True); ax.legend(frameon=False, loc='lower right')
    note(fig, 'one grey line per real network, complete networks (no sampling)', -.06)
    save(fig, 'fig14_windows')


def draw():
    plt = setup()
    for old in FIGS.glob('*'): old.unlink()
    summary = pd.read_csv(FINAL/'SUMMARY.csv').query("group == 'real'").set_index(['arm', 'method'])
    perall = pd.read_csv(FINAL/'PER_SOURCE.csv')
    per = perall.query("group == 'real'")
    pred = pd.read_csv(FINAL/'PREDICTIONS.csv')
    resp = pd.read_csv(FINAL/'VARIABILITY_RESPONSE.csv').query("group == 'real'").set_index(['arm', 'method'])
    f = pd.read_csv(DATA/'network_features.csv', index_col=0)
    types = pd.read_csv(DATA/'answer_types.csv')
    fig_toy(plt); fig_portrait(plt, f); fig_sample(plt, summary, per); fig_ranking(plt, summary)
    fig_textbook(plt, types); fig_textbook_networks(plt, pred, f)
    fig_correction(plt, pred); fig_correction_networks(plt, pred, f); fig_profile(plt, pred)
    fig_stability(plt, resp); fig_noise_networks(plt, pred, f); fig_sample_noise(plt, pred, f)
    fig_python(plt, summary); fig_python_networks(plt, per, f)
    fig_agreement(plt, per, f); fig_structure(plt, perall, f); fig_persistence(plt, perall); fig_twins(plt, pred)
    fig_cards(plt, perall, f); fig_networks(plt, per, f)
    fig_synthetic_portrait(plt); fig_synthetic_arms(plt, perall); fig_windows(plt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations and api_runs')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
