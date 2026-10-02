"""Figures for docs/analysis/ANALYSIS.md.

In plain words: draws the figures of the written analysis from the frozen results in
docs/results/final. Nothing in docs/results/final is changed.

Two steps:
  python scripts/analysis_figures.py --inputs   rebuild docs/analysis/data/*.csv
  python scripts/analysis_figures.py            draw docs/analysis/figures/*.png and *.pdf

The --inputs step needs files that are not in the repository: the raw networks in
data/raw, and the frozen samples and the answers of the language models (default location
~/.local/share/masterthesis: api_observations, api_runs, qwen_runs). Its small output tables
are kept in the repository, so the figures can be redrawn without them.
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
GREY, BLUE, ORANGE, VIOLET, AQUA, PINK = '#9c9b95', '#2a78d6', '#eb6834', '#4a3aa7', '#1baf7a', '#e87ba4'
RUST, PLUM = '#8f3410', '#9e3562'   # GPT + Python and Qwen no thinking: a darker step of GPT's orange and of Qwen's pink
LIGHT = '#cdc9bf'                   # training median, a constant guess
EASY, MEDIUM, HARD = '#bfe5c9', '#f7dc8a', '#f08a86'
BETTER, SAME, WORSE = '#0f6b3a', '#bbbbbb', '#ee6a5f'   # a change for the better, none (within 0.5 pp), for the worse;
                                                        # dark green and lighter red also differ in lightness (colour-blind safe)
# One colour per method, the same in every figure.
COLOUR = {'plugin': GREY, 'median': LIGHT, 'mle': BLUE, 'et': AQUA, 'gpt_6_sol': ORANGE, 'gpt_6_sol_tools': RUST,
          'deepseek_flash': VIOLET, 'qwen_thinking': PINK, 'qwen_nonthinking': PLUM}
# How persistent a pair is: active in 1, in 2–3 or in 4–5 of the five time windows.
KINDS = (('pairs active in 1 window', [1], '#c6c3bb'), ('in 2–3 windows', [2, 3], '#7aafee'), ('in 4–5 windows', [4, 5], '#1b56a0'))


# ---------------------------------------------------------------- inputs (needs data outside the repo)
def activity_rows(g):
    """When the pairs of one network are active: for pairs active in 1, 2, ... 5 windows, the share of all
    pairs of the network that are of this kind and active in window 1, ... 5."""
    on, K = g.counts > 0, g.K
    return [{'source': g.key, 'windows_of_pair': k, **{f'window_{j + 1}': float(on[K == k][:, j].sum()/g.D) for j in range(5)}}
            for k in range(1, 6)]


def build_inputs(external):
    """Network features and answer types, as small CSV tables."""
    from study.data import prepare_real
    from study.observation import parse
    from study.estimators import design_estimate
    from study.surrogates import shuffle
    from pipeline.core import CFG
    from pipeline.real_networks import load_raw, checked_graph
    DATA.mkdir(parents=True, exist_ok=True)
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    obs = {p.stem: json.loads(p.read_text()) for p in sorted((external/'api_observations').glob('*.json'))}

    # Network features from the raw data; the rebuilt truth must equal TRUTH.json exactly.
    rows, activity = [], []
    tmp = tempfile.TemporaryDirectory()
    for key in NAMES:
        spec = CFG['stage2_sources'].get(key, {})
        if 'file' in spec:
            g = checked_graph(key, load_raw(key, spec, ROOT/'data/raw')[0], spec['proximity'])[0]
        else:
            g = prepare_real(key, ROOT/'data/raw', Path(tmp.name)/key)
        if abs(g.truth[0] - truth[key][0]) > 1e-12: raise ValueError(f'{key}: rebuilt truth differs from TRUTH.json')
        twin = shuffle(g)   # the time-shuffled twin, rebuilt from its fixed seed
        if abs(twin.truth[0] - truth[twin.key][0]) > 1e-12: raise ValueError(f'{twin.key}: rebuilt truth differs from TRUTH.json')
        activity += activity_rows(g) + activity_rows(twin)
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
                activity += activity_rows(g)
                syn.append({'source': g.key, 'nodes': g.N, 'pairs': g.D, 'events': g.M,
                            'events_per_pair': g.M/g.D, 'rho2': g.truth[0],
                            'effective_pairs': float(1/np.sum((g.m/g.M)**2)),
                            'share_one_event': float(np.mean(g.m == 1)),
                            'share_one_window_several': float(np.mean((g.K == 1) & (g.m > 1))),
                            **{f'share_{k}_windows': float(np.mean(g.K == k)) for k in range(2, 6)}})
    pd.DataFrame(syn).set_index('source').to_csv(DATA/'synthetic_features.csv', float_format='%.6g')
    # Per-window activity of the 12 real networks, their 12 twins and the 8 synthetic networks.
    pd.DataFrame(activity).to_csv(DATA/'window_activity.csv', index=False, float_format='%.6g')

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

    # Under the hood (real networks): reasoning length, and whether the full trace says "guess" (GPT shows no trace).
    rows = []
    for run, method in (('openai', 'gpt_6_sol'), ('openai_tools', 'gpt_6_sol_tools'), ('deepseek', 'deepseek_flash')):
        for line in (external/'api_runs'/run/'responses.jsonl').read_text().splitlines():
            d = json.loads(line)
            oid = d['id'].rsplit('__', 2)[0]
            if d.get('kind') != 'main' or obs[oid]['stratum'] != 'real': continue
            trace = d.get('reasoning_content')
            rows.append({'method': method, 'arm': obs[oid]['arm'], 'reasoning_tokens': d.get('reasoning_tokens') or 0,
                         'trace_says_guess': 'guess' in trace.lower() if trace else np.nan})
    # Qwen ran on the cluster; a copy of its answer files (one JSON per answer) lies in qwen_runs. Its output tokens
    # are the thinking plus the answer of about 55 tokens.
    for path in sorted((external/'qwen_runs').glob('**/answers/thinking_r*/*.json')):
        d = json.loads(path.read_text())
        if obs.get(d['observation_id'], {}).get('stratum') != 'real': continue
        rows.append({'method': 'qwen_thinking', 'arm': d['arm'], 'reasoning_tokens': d['output_tokens'],
                     'trace_says_guess': 'guess' in d['reasoning_text'].lower()})
    r = pd.DataFrame(rows).astype({'trace_says_guess': float}).groupby(['method', 'arm'])
    pd.DataFrame({'answers': r.size(), 'median_reasoning_tokens': r.reasoning_tokens.median(),
                  'trace_says_guess': r.trace_says_guess.mean()}
                 ).to_csv(DATA/'under_the_hood.csv', float_format='%.4f')


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
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIGS/f'{name}.{ext}')
    plt.close(fig)


def colour(method): return COLOUR[method]


def band(v): return EASY if v < 3 else MEDIUM if v <= 10 else HARD


def legend_bands(fig, plt, y=-.02):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (EASY, MEDIUM, HARD)]
    fig.legend(handles, ['error < 3 pp', 'error 3–10 pp', 'error > 10 pp'], loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, y))


def unframe(ax):
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.xaxis.tick_top()


# Fig. 0: a toy network of six pairs and what each arm shows of it (events per pair and window).
TOY = {'A': [3, 3, 2, 3, 3], 'B': [1, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 1],
       'D': [1, 0, 0, 0, 0], 'E': [0, 0, 1, 0, 0], 'F': [0, 0, 0, 0, 1]}
TOY_NODES = 'BCDE'   # the pairs between the drawn nodes (R and H)
TOY_PANELS = (('Full network', TOY, (), 'true ρ₂', ''),
              ('R · random nodes', {k: TOY[k] for k in TOY_NODES}, (), 'naive share', 'only pairs between drawn nodes'),
              ('S · random walk', {k: TOY[k] for k in 'ABCE'}, (), 'naive share', 'busy pairs are met more often'),
              ('H · late time only', {k: [0, 0] + TOY[k][2:] for k in TOY_NODES if sum(TOY[k][2:])}, (0, 1), 'naive share',
               'same nodes, windows 1–2 hidden'),
              ('B · event loss', {'A': [1, 0, 1, 0, 1], 'B': [0, 0, 1, 0, 0], 'E': [0, 0, 1, 0, 0]}, (), 'naive share', 'most events are lost'))


def fig_toy(plt):
    kind = {k: c for _, ks, c in KINDS for k in ks}
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), gridspec_kw=dict(width_ratios=[1.12, 1, 1, 1, 1], wspace=.32))
    for ax, (title, pairs, hidden, what, caption) in zip(axes, TOY_PANELS):
        for j in hidden: ax.add_patch(plt.Rectangle((j - .5, -.5), 1, 6, color='#efede8', lw=0))
        seen = persistent = 0
        for i, k in enumerate(TOY):
            row = pairs.get(k)
            ax.text(-.95, i, k, ha='center', va='center', fontsize=9.5, color='#222222' if row else '#c9c6bd')
            ax.plot([-.35, 4.35], [i, i], color='#eeece7', lw=1, zorder=0)
            if not row: continue
            windows = sum(x > 0 for x in row); seen += 1; persistent += windows >= 2
            for j in range(5):
                if row[j]: ax.scatter(j, i, s=40 + 35*row[j], color=kind[windows], zorder=3)
        ax.set_xlim(-1.3, 4.6); ax.set_ylim(5.6, -1.0); ax.axis('off')
        for j in range(5): ax.text(j, -.85, j + 1, ha='center', fontsize=8.5, color='#999999')
        ax.set_title(title, loc='left', fontsize=11, pad=8)
        ax.text(1.65, 6.35, f'{what} = {persistent}/{seen} = {100*persistent/seen:.0f} %', ha='center', fontsize=10.5, fontweight='bold')
        ax.text(1.65, 7.15, caption, ha='center', fontsize=9, color='#666666')
    axes[0].text(-.95, -.85, 'pair', ha='center', fontsize=8.5, color='#999999')
    fig.add_artist(plt.Line2D([.2755, .2755], [.02, .95], color='#dddad2', lw=1))
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=8) for _, _, c in KINDS]
    fig.legend(handles, [lab.replace('pairs', 'pair') for lab, _, _ in KINDS], loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.0), fontsize=9.5)
    note(fig, 'columns: the five time windows · bigger dot = more events', -.16)
    save(fig, 'fig0_toy')


# ---------------------------------------------------------------- shared pieces
REAL_NOTE = '12 real networks · 3 samples each'
LM3 = tuple((m, COLOUR[m]) for m in ('gpt_6_sol', 'deepseek_flash', 'qwen_thinking'))
MLE_LM3 = (('mle', BLUE),) + LM3
LEVELS = ['ρ₂', 'ρ₃', 'ρ₄', 'ρ₅']
FIVE = ['mle', 'et', 'gpt_6_sol', 'deepseek_flash', 'qwen_thinking']   # the methods shown side by side


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


SPEC = '#9a9a9a'   # small grey specs next to a network's name


def pair_spec(s, f):
    """The two numbers shown next to a network's name: size in pairs and persistence."""
    return f"{num(f.loc[s, 'pairs'])} pairs · ρ₂ {pct(f.loc[s, 'rho2'])}"


def twin_spec(s, f):
    """The same for a time-shuffled twin: its persistence next to the real network's."""
    return f"{num(f.loc[s, 'pairs'])} pairs · ρ₂ {pct(f.loc[s, 'rho2'])[:-2]} → {pct(f.loc[s, 'rho2_shuffled'])}"


def full_spec(row):
    """All central numbers of a network (overview figures only)."""
    return f"{num(row['nodes'])} nodes · {num(row['pairs'])} pairs · {num(row['events'])} events · ρ₂ {pct(row['rho2'])}"


def two_tone(ax, x, y, name, spec, transform, ha='right', size=10, small=8.5, weight='normal', dx=0, dy=-3.5):
    """A name in black followed by small grey specs in parentheses, on one line."""
    from matplotlib.transforms import offset_copy
    fig = ax.figure
    at = lambda off: offset_copy(transform, fig=fig, x=dx + off, y=dy, units='points')
    width = lambda txt: txt.get_window_extent(fig.canvas.get_renderer()).width*72/fig.dpi
    kw = dict(va='baseline', clip_on=False)
    if ha == 'right':
        s = ax.text(x, y, f'({spec})', transform=at(0), ha='right', fontsize=small, color=SPEC, **kw)
        ax.text(x, y, name, transform=at(-width(s) - 3), ha='right', fontsize=size, fontweight=weight, **kw)
    else:
        n = ax.text(x, y, name, transform=at(0), ha='left', fontsize=size, fontweight=weight, **kw)
        ax.text(x, y, f'({spec})', transform=at(width(n) + 3), ha='left', fontsize=small, color=SPEC, **kw)


def row_names(ax, order, f, x=-.02, spec=pair_spec):
    """Row labels of the real networks: name (pairs · ρ₂)."""
    ax.set_yticks(range(len(order)), ['']*len(order))
    for i, s in enumerate(order): two_tone(ax, x, i, short(s), spec(s, f), ax.get_yaxis_transform())


def network_rows(plt, f, spec=pair_spec):
    """Four panels, one per arm, that share one row per real network (the most persistent on top)."""
    order = network_order(f)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for y in range(len(order)): ax.axhline(y, color='#f0efeb', lw=6, zorder=0)
        ax.set_title(ARMS[arm]); ax.tick_params(axis='y', length=0)
    row_names(axes[0], order, f, spec=spec); axes[0].set_ylim(-.6, len(order) - .4)
    return fig, axes, order


def change_dots(plt, f, change, methods, labels, spec=pair_spec):
    """One dot per network (row), method (column) and arm (panel): green where the error falls by more than 0.5 pp,
    red where it rises, the area grows with the size of the change. change: columns (arm, method), rows networks."""
    fig, axes, order = network_rows(plt, f, spec)
    scale = 14
    for ax, arm in zip(axes, ARMS):
        for x, m in enumerate(methods):
            v = change[(arm, m)].loc[order].to_numpy()
            ax.scatter([x]*len(order), range(len(order)), s=np.clip(np.abs(v), .5, 40)*scale, zorder=3, edgecolor='white', lw=.5,
                       color=[BETTER if c < -.5 else WORSE if c > .5 else SAME for c in v])
        ax.set_xticks(range(len(methods)), [METHODS[m] for m in methods], rotation=35, ha='right', fontsize=9)
        ax.set_xlim(-.6, len(methods) - .4); ax.tick_params(axis='x', length=0); ax.spines['bottom'].set_visible(False)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=7) for c in (BETTER, WORSE)] + \
              [plt.Line2D([], [], marker='o', ls='', color='#999999', markersize=np.sqrt(v*scale)) for v in (5, 20)]
    fig.legend(handles, labels + ['by 5 pp', 'by 20 pp'], loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .97))
    return fig


def method_pairs(plt, axes, before, after, methods, xlim):
    """One row per method and one panel per arm: two errors joined by a line, red where the second is higher by
    more than 0.5 pp, green where it is lower. The first dot is filled, the second hollow."""
    y = np.arange(len(methods))[::-1]
    for ax, arm in zip(axes, ARMS):
        for yi, m in zip(y, methods):
            b, a = before[(arm, m)], after[(arm, m)]
            ax.plot([b, a], [yi, yi], color=BETTER if a < b - .5 else WORSE if a > b + .5 else SAME, lw=2.6, zorder=2, solid_capstyle='round')
            ax.scatter(b, yi, s=44, color=colour(m), edgecolor='white', lw=.7, zorder=3)
            ax.scatter(a, yi, s=44, color='white', edgecolor=colour(m), lw=1.4, zorder=3)
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlim(*xlim); ax.tick_params(axis='y', length=0)


def pair_legend(plt, fig, labels, y):
    handles = [plt.Line2D([], [], marker='o', ls='', markersize=7, color='#555555'),
               plt.Line2D([], [], marker='o', ls='', markersize=7, markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=1.4),
               plt.Line2D([], [], color=WORSE, lw=2.6), plt.Line2D([], [], color=BETTER, lw=2.6)]
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, y))


def typical_error(perall):
    """Median error of the six main methods, per network and arm (pp)."""
    return perall[perall.method.isin(MAIN)].groupby(['source', 'arm']).MAE_2.median().unstack()*100


def r2_answers(pred, group='real'):
    p = pred[(pred.group == group) & pred.prediction.notna()].copy()
    p['r2'] = p.prediction.map(lambda v: json.loads(v)[0])*100
    return p


def level_errors(pred):
    """Error of every level of the profile (pp), per group, network, arm and method (mean over samples and answers)."""
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    p = pred[pred.prediction.notna() & (pred.valid == True) & (pred.replicate.isna() | (pred.replicate == 0))]
    e = 100*np.abs(np.array([json.loads(v) for v in p.prediction]) - np.array([truth[s] for s in p.source]))
    return pd.DataFrame(e, index=p.index, columns=LEVELS).groupby([p.group, p.source, p.arm, p.method]).mean()


def network_dots(plt, ax, order, values, xlab, xlim, f, models=LM3):
    """One row per network; one coloured dot per model."""
    y = np.arange(len(order))
    for yi in y: ax.axhline(yi, color='#f0efeb', lw=6, zorder=0)
    for k, (m, c) in enumerate(models):
        v = [values[m].get(s, np.nan) for s in order]
        ax.scatter(v, y + (k - (len(models) - 1)/2)*.6/len(models), s=42, color=c, edgecolor='white', lw=.7, zorder=3, label=METHODS[m])
    row_names(ax, order, f); ax.set_xlim(*xlim); ax.set_xlabel(xlab)
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
    note(fig, 'error (pp) · ' + REAL_NOTE + ' (language models: 3 answers per sample)', -.02)
    save(fig, 'fig2_ranking')


def value_column(ax, y, v, x):
    """The bars' values as a column at the right edge, clear of any reference line."""
    for yi, vi in zip(y, v): ax.text(x, yi, f'{round(vi):+d}' if round(vi) else '0', va='center', ha='right', fontsize=9)


# Fig. 2c: how much each method corrects: its estimate minus the naive share; dashed line: what the sample needs.
def fig_amount(plt, pred, summary):
    methods = ['mle', 'et', 'gpt_6_sol', 'deepseek_flash', 'qwen_thinking']
    p = r2_answers(pred)
    p = p[((p.valid == True) | p.method.isin(['plugin', 'mle', 'et'])) & (p.replicate.isna() | (p.replicate == 0))]
    naive = p[p.method == 'plugin'].groupby('observation_id').r2.first()
    made = (p.r2 - p.observation_id.map(naive)).groupby([p.arm, p.method, p.source]).mean().groupby(level=[0, 1]).mean()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.1), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, arm in zip(axes, ARMS):
        need = -100*summary.loc[(arm, 'plugin'), 'signed_rho_2']
        v = [made[(arm, m)] for m in methods]
        ax.barh(y, v, color=[colour(m) for m in methods], height=.65)
        ax.axvline(0, color='#bbbbbb', lw=1)
        ax.axvline(need, color='#333333', lw=1.5, ls=(0, (4, 2)))
        value_column(ax, y, v, 33)
        ax.set_xlim(-33, 34); ax.set_xticks([-30, -20, -10, 0, 10, 20]); ax.set_title(f'{ARMS[arm]}: needs {need:+.0f}')
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlabel('estimate − naive share (pp)')
    fig.legend([plt.Line2D([], [], color='#333333', lw=1.5, ls=(0, (4, 2)))], ['correction the sample needs (truth − naive share)'],
               loc='lower center', frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, REAL_NOTE + ' (× 3 answers) · mean over the networks', -.07)
    save(fig, 'fig2c_amount')


# Fig. 2d: when correction pays. Error of each method against the error of the naive share, one dot per network and arm.
def fig_breakeven(plt, per):
    e = per.pivot_table(index=['source', 'arm'], columns='method', values='MAE_2')*100
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
    for ax, m in zip(axes, FIVE):
        ax.fill([0, 46, 46], [0, 0, 46], color='#eaf5ee', lw=0, zorder=0); ax.fill([0, 0, 46], [0, 46, 46], color='#fbecec', lw=0, zorder=0)
        ax.plot([0, 46], [0, 46], color='#bbbbbb', lw=1, zorder=1)
        ax.scatter(e.plugin, e[m], s=26, color=colour(m), edgecolor='white', lw=.5, zorder=3)
        ax.set_xlim(0, 46); ax.set_ylim(0, 46); ax.set_aspect('equal'); ax.set_title(METHODS[m])
        ax.set_xlabel('error of the naive share (pp)'); ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('error of the method (pp)')
    fig.canvas.draw()   # the square panels are placed now
    box, inch = axes[0].get_position(), 1/fig.get_figheight()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#eaf5ee', '#fbecec')]
    fig.legend(handles, ['correction pays: the method is better than the naive share', 'correction hurts'], loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.5, box.y1 + .35*inch))
    note(fig, REAL_NOTE + ' · one dot per network and arm (48 per method)', box.y0 - .65*inch)
    save(fig, 'fig2d_breakeven')


# Fig. 2e: on which networks correction pays, per method: error of the method minus error of the naive share.
def fig_gain(plt, per, f):
    e = per.pivot_table(index='source', columns=['arm', 'method'], values='MAE_2')*100
    gain = pd.DataFrame({(a, m): e[(a, m)] - e[(a, 'plugin')] for a in ARMS for m in FIVE})
    fig = change_dots(plt, f, gain, FIVE, ['better than the naive share', 'worse'])
    note(fig, 'networks sorted by true ρ₂ · ' + REAL_NOTE + ' · error of the method minus error of the naive share', -.12)
    save(fig, 'fig2e_networks')


# Fig. 2b: the error at every level of the profile, per arm (12 real networks).
def fig_levels(plt, pred):
    e = level_errors(pred).loc['real'].groupby(['arm', 'method']).mean()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for m, c in (('plugin', GREY), ('mle', BLUE), ('et', AQUA)) + LM3:
            ax.plot(range(4), e.loc[(arm, m)], color=c, lw=2, marker='o', markersize=5, ls=(0, (4, 2)) if m == 'plugin' else '-',
                    label=METHODS[m])
        ax.set_xticks(range(4), LEVELS); ax.set_title(ARMS[arm]); ax.set_ylim(0, 32); ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('error (pp)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, .98))
    note(fig, REAL_NOTE + ' (language models: 3 answers per sample)', -.06)
    save(fig, 'fig2b_levels')


def grouped_bars(plt, ax, groups, values, ymax, fmt='{:.0f}', title='', models=LM3):
    """values[model][group] -> bars; one colour per model, values written on top."""
    w = .8/len(models)
    for k, (m, c) in enumerate(models):
        xs = np.arange(len(groups)) + (k - (len(models)-1)/2)*w
        vs = [values[m][g] for g in groups]
        ax.bar(xs, vs, width=w*.92, color=c, label=METHODS[m])
        for x, v in zip(xs, vs): ax.text(x, v + ymax*.015, fmt.format(v), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(groups)), [ARMS[g] for g in groups]); ax.set_ylim(0, ymax); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0); ax.set_title(title, pad=12)
    ax.legend(frameon=False, ncol=len(models), loc='upper center', bbox_to_anchor=(.5, -.12))


# Fig. 3: share of answers that equal the textbook answer (R: the naive share; S: the simple reweighting).
def fig_textbook(plt, types):
    t = types.set_index(['arm', 'method'])
    vals = {m: {'R': 100*t.loc[('R', m), 'observed_share'], 'S': 100*t.loc[('S', m), 'reweighting']} for m, _ in LM3}
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    grouped_bars(plt, ax, ['R', 'S'], vals, 112, '{:.0f} %', 'Answers equal to the textbook answer')
    note(fig, REAL_NOTE + ' × 3 answers', -.2)
    save(fig, 'fig3_textbook')


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


# Fig. 4: share of answers that correct by about the right amount (50–150 % of the needed correction), in H and B.
def correction_table(pred):
    """Per answer: needed = truth − naive share, done = answer − naive share; samples off by ≥ 5 pp only."""
    p = r2_answers(pred)
    naive = p[p.method == 'plugin'].groupby('observation_id').r2.first()
    a = p[p.method.isin(LLMS + ['mle']) & (p.valid == True)].copy()
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
    vals = {m: {arm: 100*t.loc[(arm, m), 'about right'] for arm in 'HB'} for m, _ in MLE_LM3}
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    grouped_bars(plt, ax, ['H', 'B'], vals, 92, '{:.0f} %', 'Answers that correct by about the right amount', MLE_LM3)
    note(fig, REAL_NOTE + ' × 3 answers (MLE: 1) · only samples off by ≥ 5 pp', -.2)
    save(fig, 'fig4_correction')


# Fig. 4b: per network, H and B together: share of answers that correct by about the right amount.
def fig_correction_networks(plt, pred, f):
    a = correction_table(pred)
    a = a[a.arm.isin(['H', 'B'])]
    share = a.groupby(['method', 'source']).type.apply(lambda s: 100*(s == 'about right').mean())
    count = a.groupby(['method', 'source']).size()
    order = network_order(f)
    vals = {m: {s: share[(m, s)] for s in order if (m, s) in share.index and count[(m, s)] >= 3} for m, _ in MLE_LM3}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    network_dots(plt, ax, order, vals, 'answers that correct by about the right amount (%)', (-5, 105), f, MLE_LM3)
    ax.set_title('H and B together, per network', pad=12)
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.4, -.14), columnspacing=1.2)
    note(fig, 'networks sorted by true ρ₂ · only samples off by ≥ 5 pp (Digg and Linux: none)', -.18)
    save(fig, 'fig4b_correction_networks')


# Fig. 4c: mean estimates of rho_2..rho_5 against the truth, per arm (12 real networks).
def fig_profile(plt, pred):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    p = pred[(pred.group == 'real') & pred.prediction.notna() & (pred.valid == True)]
    prof = pd.DataFrame([json.loads(v) for v in p.prediction], index=p.index, columns=[2, 3, 4, 5])*100
    mean = prof.groupby([p.arm, p.method, p.source]).mean().groupby(level=[0, 1]).mean()
    t = pd.DataFrame({s: truth[s] for s in p.source.unique()}, index=[2, 3, 4, 5]).T.mean()*100
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    k = np.arange(4)
    for ax, arm in zip(axes, ARMS):
        ax.plot(k, t.to_numpy(), color='#333333', lw=3, marker='o', label='truth', zorder=5)
        for m, c in (('plugin', GREY),) + MLE_LM3:
            ax.plot(k, mean.loc[(arm, m)].to_numpy(), color=c, lw=1.8, marker='o', markersize=4.5,
                    ls=(0, (4, 2)) if m == 'plugin' else '-', label=METHODS[m])
        ax.set_xticks(k, LEVELS); ax.set_title(ARMS[arm]); ax.set_ylim(0, 75)
        ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('share of pairs (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, .98))
    note(fig, 'mean over 12 real networks · 3 samples (× 3 answers) each', -.06)
    save(fig, 'fig4c_profile')


# Fig. 4d: how long the models think (median reasoning tokens per answer) against their error, one dot per model and arm.
def fig_thinking(plt, summary):
    tokens = pd.read_csv(DATA/'under_the_hood.csv').set_index(['method', 'arm']).median_reasoning_tokens
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for m, c in LM3:
        x = [tokens[(m, a)] for a in ARMS]; y = [100*summary.loc[(a, m), 'MAE_2'] for a in ARMS]
        ax.scatter(x, y, s=70, color=c, edgecolor='white', lw=.8, zorder=3, label=METHODS[m])
        for xi, yi, a in zip(x, y, ARMS):
            ax.annotate(a, (xi, yi), xytext=(6, 4), textcoords='offset points', fontsize=9, color='#444444')
    ax.set_xscale('log'); ax.set_xlim(250, 100000); ax.set_ylim(0, 25); ax.minorticks_off()
    ax.set_xticks([300, 1000, 3000, 10000, 30000, 100000], ['300', '1,000', '3,000', '10,000', '30,000', '100,000'])
    ax.set_xlabel('median reasoning tokens per answer'); ax.set_ylabel('error (pp)'); ax.spines['left'].set_visible(True)
    ax.grid(axis='y', color='#eeeeee'); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0))
    note(fig, REAL_NOTE + ' × 3 answers · letters: the arm · Qwen: all output tokens', -.08)
    save(fig, 'fig4d_thinking')


# Fig. 5: spread of the three answers to the identical sample (median SD, pp).
def fig_stability(plt, resp):
    vals = {m: {a: 100*resp.loc[(a, m), 'median_observation_SD_rho2'] for a in ARMS} for m, _ in LM3}
    fig, ax = plt.subplots(figsize=(9, 3.3))
    grouped_bars(plt, ax, list(ARMS), vals, 25, '{:.1f}', 'Spread of 3 answers to the same sample (pp)')
    note(fig, REAL_NOTE + ' · MLE and ExtraTrees: always the same answer', -.2)
    save(fig, 'fig5_stability')


# Fig. 5b: per network, the spread of 3 answers to the same sample, H and B together (R and S hardly vary).
def fig_noise_networks(plt, pred, f):
    p = r2_answers(pred)
    p = p[p.method.isin([m for m, _ in LM3]) & (p.valid == True) & p.arm.isin(['H', 'B'])]
    sd = p.groupby(['method', 'arm', 'source', 'observation_id']).r2.std().groupby(['method', 'arm', 'source']).median()
    v = sd.groupby(['method', 'source']).mean()
    order = network_order(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    network_dots(plt, ax, order, {m: {s: v[(m, s)] for s in order} for m, _ in LM3}, 'spread of 3 answers (pp)', (-1, 33), f)
    ax.set_title('H and B together, per network', pad=12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.45, -.14))
    note(fig, 'networks sorted by true ρ₂ · median over samples, mean of H and B', -.18)
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
    row_names(axes[0], order, f); axes[0].set_ylim(-.6, len(order)-.4)
    h = [plt.Line2D([], [], color='#333333', lw=2.2), plt.Line2D([], [], marker='o', ls='', color=BLUE, markersize=6)]
    fig.legend(h, ['true ρ₂', 'MLE estimate from one sample'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, '12 real networks · 3 independent samples each · MLE gives the same answer for the same sample', -.04)
    save(fig, 'fig5c_sample_noise')


def python_bars(ax, summary, title):
    """GPT with and without Python: error per arm (pp)."""
    w = .38
    for k, m in enumerate(('gpt_6_sol', 'gpt_6_sol_tools')):
        xs = np.arange(4) + (k - .5)*w
        vs = [100*summary.loc[(a, m), 'MAE_2'] for a in ARMS]
        ax.bar(xs, vs, width=w*.92, color=colour(m), label=METHODS[m])
        for x, v in zip(xs, vs): ax.text(x, v + .3, f'{v:.1f}', ha='center', va='bottom', fontsize=9.5)
    ax.set_xticks(range(4), [ARMS[a] for a in ARMS]); ax.set_ylim(0, 18); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, ncol=2, loc='upper left'); ax.set_title(title, pad=12)


# Fig. 6: GPT with and without Python, error per arm.
def fig_python(plt, summary):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    python_bars(ax, summary, 'Error (pp)')
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
        ax.scatter(d, y, s=46, color=[WORSE if v > .5 else BETTER if v < -.5 else SAME for v in d], zorder=3,
                   edgecolor='white', lw=.7)
        ax.set_xlim(-8, 25); ax.set_title(ARMS[arm]); ax.set_xlabel('Python − GPT (pp)'); ax.tick_params(axis='y', length=0)
    row_names(axes[0], order, f); axes[0].set_ylim(-.6, len(order)-.4)
    h = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=7) for c in (WORSE, SAME, BETTER)]
    fig.legend(h, ['worse with Python', 'about the same (±0.5 pp)', 'better with Python'], loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, 'networks sorted by true ρ₂ · ' + REAL_NOTE + ' × 3 answers', -.04)
    save(fig, 'fig6b_python_networks')


# Fig. 6c: GPT with and without Python on the time-shuffled twins and the synthetic networks.
def fig_python_groups(plt):
    s = pd.read_csv(FINAL/'SUMMARY.csv').set_index(['group', 'arm', 'method'])
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.2))
    for ax, (group, title) in zip(axes, (('surrogate', 'Time-shuffled twins: error (pp)'), ('synthetic', 'Synthetic networks: error (pp)'))):
        python_bars(ax, s.loc[group], title)
    fig.subplots_adjust(wspace=.1)
    note(fig, '12 time-shuffled twins and 8 synthetic networks · 3 samples each × 3 answers', -.1)
    save(fig, 'fig6c_python_groups')


# Fig. 7: is a network hard for every method? MLE against GPT, one dot per real network.
def fig_agreement(plt, per, f):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    e = lambda m, arm: per[(per.method == m) & (per.arm == arm)].set_index('source').MAE_2*100
    for ax, arm in zip(axes, ARMS):
        x, y = e('mle', arm), e('gpt_6_sol', arm)
        y = y.loc[x.index]
        ax.plot([0, 35], [0, 35], color='#bbbbbb', lw=1, zorder=1)
        ax.scatter(x, y, s=46, color='#333333', edgecolor='white', lw=.7, zorder=3)
        if arm == 'B':
            xy = (x['copenhagen_bluetooth'], y['copenhagen_bluetooth'])
            ax.annotate('Copenhagen', xy, xytext=(4, 34), textcoords='offset points', fontsize=9,
                        arrowprops=dict(arrowstyle='-', color='#888888', lw=.8, relpos=(0, 0), shrinkA=0))
            ax.annotate(f"({pair_spec('copenhagen_bluetooth', f)})", xy, xytext=(12, 23), textcoords='offset points',
                        fontsize=8, color=SPEC)
        ax.set_xlim(0, 35); ax.set_ylim(0, 35); ax.set_aspect('equal'); ax.set_title(ARMS[arm])
        ax.set_xlabel('error of MLE (pp)'); ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('error of GPT (pp)')
    note(fig, 'one dot per real network · 3 samples each', -.02)
    save(fig, 'fig7_agreement')


# Fig. 8 and 8b: typical error against the number of events, and against the number of pairs that carry them, per arm.
def fig_structure(plt, perall, f):
    t = typical_error(perall).loc[f.index]
    figures = (('fig8_events', 'events', 'events in the network', [10**5, 10**6], '102k events', ''),
               ('fig8b_pairs', 'effective_pairs', 'pairs that carry the events', [100, 1000, 10000, 100000], '55 pairs carry them',
                ' · pairs that carry the events = effective number of pairs (see definitions)'))
    for name, col, xlab, ticks, malawi, extra in figures:
        fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)
        for ax, arm in zip(axes, ARMS):
            ax.scatter(f[col], t[arm], s=40, color='#333333', zorder=3)
            two_tone(ax, f.loc['sp_malawi', col], t.loc['sp_malawi', arm], 'Malawi', malawi, ax.transData, ha='left',
                     size=8.5, small=8, dx=3, dy=9)
            ax.set_xscale('log'); ax.set_xticks(ticks, [f'{x:,}' for x in ticks]); ax.minorticks_off()
            ax.set_xlabel(xlab, fontsize=9.5); ax.set_ylim(0, 36); ax.set_yticks([0, 10, 20, 30]); ax.spines['left'].set_visible(True)
            ax.set_title(ARMS[arm])
        axes[0].set_ylabel('typical error (pp)')
        note(fig, '12 real networks · typical error = median of six methods' + extra, -.12)
        save(fig, name)


# Fig. 9: typical error against true rho_2, all 32 networks, per arm.
def fig_persistence(plt, perall):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    t = typical_error(perall)
    rho = pd.Series({s: 100*truth[s][0] for s in t.index})
    groups = (('real', [s for s in t.index if s in NAMES], dict(color='#333333')),
              ('time-shuffled twin', [s for s in t.index if s.endswith('__pwt')], dict(color='white', edgecolor='#333333', lw=1.1)),
              ('synthetic', [s for s in t.index if s not in NAMES and not s.endswith('__pwt')], dict(color='#333333', marker='s')))
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


# Fig. 9b: the same per method: mean error on the networks with rho_2 below and above 30 %.
def fig_persistence_methods(plt, perall):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    e = perall.pivot_table(index='source', columns=['arm', 'method'], values='MAE_2')*100
    high = np.array([truth[s][0] >= .3 for s in e.index])
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)
    method_pairs(plt, axes, e[~high].mean(), e[high].mean(), ['plugin'] + FIVE, (-2, 40))
    for ax, arm in zip(axes, ARMS): ax.set_title(ARMS[arm]); ax.set_xlabel('error (pp)')
    pair_legend(plt, fig, [f'ρ₂ below 30 % ({(~high).sum()} networks)', f'ρ₂ above 30 % ({high.sum()} networks)', 'harder', 'easier'], .98)
    note(fig, '32 networks: 12 real, their 12 time-shuffled twins, 8 synthetic · 3 samples (× 3 answers) each', -.1)
    save(fig, 'fig9b_methods')


# Fig. 9c: the same for the higher levels of the profile: typical error of rho_3, rho_4, rho_5 against their true values.
def fig_persistence_levels(plt, pred):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    e = level_errors(pred)
    t = e[e.index.get_level_values('method').isin(MAIN)].groupby(['source', 'arm']).median()
    sources = t.index.get_level_values('source').unique()
    groups = (('real', [s for s in sources if s in NAMES], dict(color='#333333')),
              ('time-shuffled twin', [s for s in sources if s.endswith('__pwt')], dict(color='white', edgecolor='#333333', lw=1.1)),
              ('synthetic', [s for s in sources if s not in NAMES and not s.endswith('__pwt')], dict(color='#333333', marker='s')))
    fig, axes = plt.subplots(3, 4, figsize=(14, 7.4), sharex=True, sharey=True)
    for r, k in enumerate((1, 2, 3)):
        for ax, arm in zip(axes[r], ARMS):
            for lab, keys, style in groups:
                ax.scatter([100*truth[s][k] for s in keys], t.loc[[(s, arm) for s in keys], LEVELS[k]], s=30, zorder=3, label=lab, **style)
            ax.set_xlim(0, 90); ax.set_ylim(0, 35); ax.spines['left'].set_visible(True)
            if r == 0: ax.set_title(ARMS[arm])
            if r == 2: ax.set_xlabel('true value (%)')
        axes[r][0].set_ylabel(f'{LEVELS[k]}: typical error (pp)')
    fig.subplots_adjust(hspace=.16, wspace=.08)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .93))
    note(fig, '32 networks: 12 real, their 12 time-shuffled twins, 8 synthetic · typical error = median of six methods · '
         'each row: one level against its own true value', axes[2][0].get_position().y0 - .75/fig.get_figheight())
    save(fig, 'fig9c_levels')


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
        value_column(ax, y, v, 40)
        ax.set_xlim(-14, 41); ax.set_title(ARMS[arm]); ax.set_xticks([0, 10, 20, 30])
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlabel('change in estimated ρ₂ (pp)')
    fig.legend([plt.Line2D([], [], color='#333333', lw=1.5, ls=(0, (4, 2)))], [f'true change (+{true:.0f} pp)'],
               loc='lower center', frameon=False, bbox_to_anchor=(.5, .97))
    note(fig, '12 real networks and their time-shuffled twins · 3 samples (× 3 answers) each', -.05)
    save(fig, 'fig10_twins')


# Fig. 10c: the twins per method: mean error on the 12 real networks and on their twins.
def fig_twin_methods(plt):
    e = pd.read_csv(FINAL/'SUMMARY.csv').set_index(['group', 'arm', 'method']).MAE_2*100
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)
    method_pairs(plt, axes, e.loc['real'], e.loc['surrogate'], ['plugin'] + FIVE, (-2, 36))
    for ax, arm in zip(axes, ARMS): ax.set_title(ARMS[arm]); ax.set_xlabel('error (pp)')
    pair_legend(plt, fig, ['real networks', 'their time-shuffled twins', 'twins harder', 'twins easier'], .98)
    note(fig, '12 real networks and their time-shuffled twins · 3 samples (× 3 answers) each', -.1)
    save(fig, 'fig10c_twin_methods')


# Fig. 10d: the twins per network and method: error on the twin minus error on its network.
def fig_twin_networks(plt, perall, f):
    e = perall[perall.group.isin(['real', 'surrogate'])].pivot_table(index='source', columns=['arm', 'method'], values='MAE_2')*100
    methods = ['plugin'] + FIVE
    change = pd.DataFrame({(a, m): e.loc[[s + '__pwt' for s in f.index], (a, m)].to_numpy() - e.loc[f.index, (a, m)].to_numpy()
                           for a in ARMS for m in methods}, index=f.index)
    fig = change_dots(plt, f, change, methods, ['twin easier than its network', 'twin harder'], twin_spec)
    note(fig, 'networks sorted by true ρ₂ · 3 samples each · error on the twin minus error on its network', -.12)
    save(fig, 'fig10d_twin_networks')


# Fig. 11: every real network on its own: typical error per arm.
def fig_cards(plt, perall, f):
    t = typical_error(perall)
    order = network_order(f)[::-1]
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.2), sharey=True)
    for ax, s in zip(axes.flat, order):
        v = [t.loc[s, a] for a in ARMS]
        ax.bar(range(4), v, color='#333333', width=.7)
        for i, x in enumerate(v): ax.text(i, x + .6, f'{x:.0f}', ha='center', fontsize=9)
        ax.set_xticks(range(4), list(ARMS)); ax.set_ylim(0, 36); ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.text(0, 1.2, short(s), transform=ax.transAxes, fontsize=10.5, fontweight='bold')
        ax.text(0, 1.06, f'({pair_spec(s, f)})', transform=ax.transAxes, fontsize=8.5, color=SPEC)
    fig.subplots_adjust(hspace=.8, wspace=.25)
    note(fig, 'typical error (pp, median of six methods) per arm · 3 samples each', -.01)
    save(fig, 'fig11_cards')


def method_heat(plt, per, order, f, name, header):
    """One small heatmap per method: rows = networks, columns = arms, colour = error level."""
    rows = order
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
        if k == 0: row_names(ax, order, f, x=-.04)
        else: ax.set_yticks(range(len(rows)), ['']*len(rows))
    fig.text(.005, 1 - .55/fig.get_figheight(), header, fontsize=9.5, color='#666666')
    legend_bands(fig, plt, -.3/fig.get_figheight())
    save(fig, name)


# Detail figure (linked, not embedded): error per real network, arm and method.
def fig_networks(plt, per, f):
    order = f.sort_values('rho2').index.tolist()
    method_heat(plt, per, order, f, 'fig_networks_detail', 'error (pp) per network')


# Synthetic networks: four variants (two generators, without and with memory), two instances each.
VARIANTS = (('dar_a0', 'DAR, no memory'), ('dar_a08', 'DAR, memory'),
            ('ad_memoryless', 'Activity-driven, no memory'), ('ad_memory', 'Activity-driven, memory'))


def variant_table():
    sy = pd.read_csv(DATA/'synthetic_features.csv', index_col=0)
    return pd.DataFrame({v: sy.loc[[f'{v}_r1', f'{v}_r2']].mean() for v, _ in VARIANTS}).T


# Fig. 0b, 10b, 12: what the networks look like. Per time window, the share of the network's pairs that are active,
# split by how persistent the pair is.
def fig_active(plt, name, panels, text):
    """panels: (keys in window_activity.csv to average, title, grey specs), one per network."""
    w = pd.read_csv(DATA/'window_activity.csv').set_index(['source', 'windows_of_pair'])
    n = -(-len(panels)//4)
    fig, axes = plt.subplots(n, 4, figsize=(15, 2.45*n + .5), sharey=True, squeeze=False)
    for ax, (keys, title, spec) in zip(axes.flat, panels):
        share = 100*sum(w.loc[k] for k in keys)/len(keys)   # rows: windows of the pair; columns: the five time windows
        bottom = np.zeros(5)
        for lab, kinds, c in KINDS[::-1]:                   # the most persistent pairs at the bottom
            v = share.loc[kinds].sum().to_numpy()
            ax.bar(range(5), v, bottom=bottom, color=c, width=.78, label=lab); bottom += v
        ax.set_xticks(range(5), range(1, 6)); ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100])
        ax.tick_params(axis='x', length=0, labelsize=8.5, colors='#888888', pad=2); ax.tick_params(axis='y', labelsize=8.5, colors='#888888')
        ax.grid(axis='y', color='#eeeeee'); ax.set_axisbelow(True)
        above = dict(xy=(0, 1), xycoords='axes fraction', textcoords='offset points')
        ax.annotate(title, xytext=(0, 20), fontsize=10.5, fontweight='bold', **above)
        ax.annotate(f'({spec})', xytext=(0, 6), fontsize=8.2, color=SPEC, **above)
    for ax in axes[:, 0]: ax.set_ylabel('pairs active (%)', fontsize=9)
    for ax in axes[-1]: ax.set_xlabel('time window', fontsize=9, color='#888888')
    fig.subplots_adjust(hspace=.62, wspace=.1)
    top, bottom, inch = axes[0, 0].get_position(), axes[-1, 0].get_position(), 1/fig.get_figheight()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, top.y1 + .55*inch))
    note(fig, text + ' · share of all pairs of the network that are active in the window', bottom.y0 - .55*inch)
    save(fig, name)


def fig_actives(plt, f):
    order = network_order(f)[::-1]
    fig_active(plt, 'fig0b_active', [([s], short(s), full_spec(f.loc[s])) for s in order], '12 real networks, complete (not sampled)')
    fig_active(plt, 'fig10b_twin_active', [([s + '__pwt'], short(s), twin_spec(s, f)) for s in order], 'the 12 time-shuffled twins, complete')
    sv = variant_table()
    fig_active(plt, 'fig12_synthetic_active', [([f'{v}_r1', f'{v}_r2'], lab, full_spec(sv.loc[v])) for v, lab in VARIANTS],
               'mean of the 2 instances per variant, complete networks')


# Fig. 13: what memory does to the estimation task, per method: mean error without and with memory.
def fig_synthetic_arms(plt, perall):
    e = perall[perall.group == 'synthetic'].pivot_table(index='source', columns=['arm', 'method'], values='MAE_2')*100
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.4), sharex=True, sharey=True)
    for row, (name, without, with_) in zip(axes, (('DAR', 'dar_a0', 'dar_a08'), ('Activity-driven', 'ad_memoryless', 'ad_memory'))):
        mean = lambda key: e.loc[[f'{key}_r1', f'{key}_r2']].mean()
        method_pairs(plt, row, mean(without), mean(with_), ['plugin'] + FIVE, (-2, 68))
        row[0].set_ylabel(name, fontsize=11, fontweight='bold', labelpad=12)
    for ax, arm in zip(axes[0], ARMS): ax.set_title(ARMS[arm])
    for ax in axes[1]: ax.set_xlabel('error (pp)')
    fig.subplots_adjust(hspace=.12)
    pair_legend(plt, fig, ['no memory', 'memory', 'harder with memory', 'easier'], .93)
    note(fig, '8 synthetic networks (2 instances per variant) · 3 samples (× 3 answers) each', .0)
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
    ax.axvline(5, color='#555555', lw=1.2, ls=(0, (4, 2))); ax.text(5.3, 88, 'used: 5 windows', color='#555555', fontsize=9)
    ax.set_xlabel('number of time windows'); ax.set_ylabel('true ρ₂ (%)'); ax.set_ylim(0, 95); ax.set_xticks([2, 5, 10, 15, 20])
    ax.spines['left'].set_visible(True); ax.legend(frameon=False, loc='lower right')
    note(fig, 'one grey line per real network, complete networks (no sampling)', -.06)
    save(fig, 'fig14_windows')


# ---------------------------------------------------------------- the two generators as small animations (GIF)
PERSIST = '#dbe8f9'   # background of pairs active in >= 2 windows


def gif(frames, durations, name):
    """Write matplotlib figures as one looping GIF with a shared palette."""
    from PIL import Image
    import matplotlib.pyplot as plt
    imgs = []
    for fig in frames:
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    w, h = imgs[0].size
    strip = Image.new('RGB', (w, h*len(imgs)))
    for k, im in enumerate(imgs): strip.paste(im, (0, h*k))
    palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    colours = palette.getpalette()
    for c in range(len(colours)//3):   # near-white entries become white, so the background stays exactly white
        if min(colours[3*c:3*c + 3]) >= 246: colours[3*c:3*c + 3] = [255, 255, 255]
    palette.putpalette(colours)
    out = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    out[0].save(FIGS/f'{name}.gif', save_all=True, append_images=out[1:], duration=durations, loop=0)


def dar_example(seed=35, P=8, chi=.2, alpha=.8):
    """A small DAR run with and without memory on the same random draws, as in the study.
    The seed is picked so that this small example shows the effect of memory clearly."""
    r = np.random.default_rng(seed)
    first = r.random(P) < chi
    keep, fresh = r.random((4, P)), r.random((4, P)) < chi
    events = 1 + r.poisson(1, (5, P))
    runs = {}
    for name, a in (('no memory', 0.), ('memory', alpha)):
        on, kept = np.zeros((5, P), bool), np.zeros((5, P), bool)
        on[0] = first
        for j in range(1, 5):
            kept[j] = keep[j-1] < a
            on[j] = np.where(kept[j], on[j-1], fresh[j-1])
        runs[name] = (on, kept)
    return runs, events


def callout(ax, text, xy, xytext):
    """A short orange label with an arrow, pointing at what happens right now."""
    ax.annotate(text, xy, xytext=xytext, fontsize=10, color=ORANGE, va='center', ha='left', zorder=6,
                annotation_clip=False, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1, shrinkA=3, shrinkB=7,
                                                       mutation_scale=9))


def dar_frame(plt, runs, events, step):
    """step 0: start; 1-5: windows; 6: events; 7: rho_2."""
    P = events.shape[1]
    fig = plt.figure(figsize=(10.8, 5.0))
    fig.text(.03, .92, 'DAR: every pair is on or off in each window', fontsize=13.5, fontweight='bold')
    where = {0: 'same random draws on both sides', 6: 'events', 7: 'result'}.get(step, f'window {step} of 5')
    fig.text(.97, .92, where, fontsize=12, color=ORANGE, ha='right', fontweight='bold')
    j = step - 1 if 1 <= step <= 5 else None
    for k, name in enumerate(('no memory', 'memory')):
        on, kept = runs[name]
        ax = fig.add_axes([.08 + .5*k, .04, .41, .76])
        ax.set_xlim(-.6, 7.6); ax.set_ylim(P + .7, -1.6); ax.axis('off')
        ax.text(-1.5, -1.45, name, fontsize=12, fontweight='bold')
        shown = min(step, 5)
        K = on[:shown].sum(0)
        if j is not None: ax.add_patch(plt.Rectangle((j - .45, -.55), .9, P + .1, color='#f3efe6', lw=0))
        ax.text(-.75, -.95, 'window', ha='right', fontsize=9, color='#999999')
        for c in range(5):
            ax.text(c, -.95, c + 1, ha='center', fontsize=10, color='#222222' if c == j else '#999999',
                    fontweight='bold' if c == j else 'normal')
        new = []
        for i in range(P):
            if step == 7 and K[i] >= 2: ax.add_patch(plt.Rectangle((-.55, i - .42), 5.1, .84, color=PERSIST, lw=0))
            ax.text(-.75, i, f'pair {"ABCDEFGH"[i]}', ha='right', va='center', fontsize=9.5,
                    color='#c4c1b8' if step == 7 and K[i] == 0 else '#222222')
            for c in range(5):
                if c >= shown: ax.scatter(c, i, s=10, color='#efede8', zorder=2); continue
                if c and kept[c, i]:
                    ax.plot([c-1, c], [i, i], color=ORANGE if c == j else '#bdb8ad', lw=2.4, zorder=1, solid_capstyle='round')
                if on[c, i]: ax.scatter(c, i, s=40 + 35*events[c, i] if step >= 6 else 70, color='#333333', zorder=3)
                else: ax.scatter(c, i, s=14, color='#d9d6cf', zorder=2)
                if c == j and not kept[c, i]:
                    ax.scatter(c, i, s=230, facecolor='none', edgecolor=ORANGE, lw=1.4, zorder=4); new.append(i)
        rows = range(P)
        if step == 1:
            i = next(i for i in rows if on[0, i]); callout(ax, 'on: chance 0.2', (0, i), (5.1, i))
        elif j is not None:
            stay = [i for i in rows if kept[j, i]]
            target = [i for i in new if on[j, i]] or new
            if target: callout(ax, 'new draw', (j, target[0]), (5.1, target[0]))
            if stay:
                i = ([i for i in stay if on[j, i]] or stay)[0]
                y = i if not target or abs(i - target[0]) > 1 else target[0] + (2 if target[0] < P - 2 else -2)
                callout(ax, 'kept: chance 0.8', (j, i), (5.1, y))
        elif step == 6:
            c, i = np.unravel_index(np.argmax(np.where(on, events, 0)), on.shape)
            callout(ax, f'{events[c, i]} events', (c, i), (5.1, i))
        elif step == 7:
            i = int(np.argmax(K >= 2)); callout(ax, '≥ 2 windows', (4.5, i), (5.1, i))
            if (K == 0).any():
                i = int(np.argmax(K == 0)); callout(ax, 'never on: no pair', (4.5, i), (5.1, i))
            a, b = int((K >= 2).sum()), int((K >= 1).sum())
            ax.text(2, P + .25, f'ρ₂ = {a} of {b} pairs = {100*a/b:.0f} %', ha='center', fontsize=12, fontweight='bold')
    return fig


def gif_dar(plt):
    runs, events = dar_example()
    gif([dar_frame(plt, runs, events, s) for s in range(8)], [3000, 2600, 2800, 2800, 2800, 2800, 2800, 6500], 'gif_dar')


AD_ACTIVITY = np.array([.6, .35, .25, .15, .1, .1, .05, .05, .05, .05])


def ad_example(seed=88, rounds=10):
    """A small activity-driven run with and without memory on the same random draws, as in the study.
    The seed is picked so that this small example shows the effect of memory clearly."""
    r = np.random.default_rng(seed)
    N = len(AD_ACTIVITY)
    activation, decision, partner = r.random((rounds, N)), r.random((rounds, N)), r.random((rounds, N))
    runs = {}
    for name in ('no memory', 'memory'):
        known, out = [set() for _ in range(N)], []
        for t in range(rounds):
            picks = []
            for i in np.flatnonzero(activation[t] < AD_ACTIVITY):
                if name == 'no memory':
                    k = int(partner[t, i]*(N-1)); j = k if k < i else k+1
                else:
                    old = known[i]
                    new = len(old) == 0 or (len(old) < N-1 and decision[t, i] < 1/(len(old) + 1))
                    choices = [j for j in range(N) if j != i and j not in old] if new else sorted(old)
                    j = choices[int(partner[t, i]*len(choices))]
                picks.append((int(i), int(j)))
            pairs = sorted({(min(i, j), max(i, j)) for i, j in picks})
            for i, j in pairs: known[i].add(j); known[j].add(i)
            out.append((picks, pairs))
        runs[name] = out
    return runs


def ad_frame(plt, runs, step, rounds=10):
    """step 0: start; 1-10: rounds (2 per window); 11: rho_2."""
    N = len(AD_ACTIVITY)
    angle = np.pi/2 - 2*np.pi*np.arange(N)/N
    pos = np.c_[np.cos(angle), np.sin(angle)]
    fig = plt.figure(figsize=(10.8, 5.4))
    fig.text(.03, .925, 'Activity-driven: active nodes create events with a partner', fontsize=13.5, fontweight='bold')
    where = {0: 'dot size = activity', rounds + 1: 'result'}.get(step, f'round {step} of {rounds} · window {(step + 1)//2}')
    fig.text(.97, .925, where, fontsize=12, color=ORANGE, ha='right', fontweight='bold')
    for k, name in enumerate(('no memory', 'memory')):
        ax = fig.add_axes([.03 + .5*k, .03, .44, .8])
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.45, 1.35); ax.set_aspect('equal'); ax.axis('off')
        ax.text(-1.75, 1.3, name, fontsize=12, fontweight='bold')
        past = runs[name][:max(step - 1, 0)] if step <= rounds else runs[name]
        count, windows = {}, {}
        for t, (_, pairs) in enumerate(past):
            for p in pairs: count[p] = count.get(p, 0) + 1; windows.setdefault(p, set()).add(t//2)
        for (i, j), c in count.items():
            colour = '#bdb8ad' if step <= rounds else BLUE if len(windows[(i, j)]) >= 2 else '#d9d6cf'
            ax.plot(*pos[[i, j]].T, color=colour, lw=1.2 + 1.6*(c - 1), zorder=1, solid_capstyle='round')
        active = set()
        if 1 <= step <= rounds:
            picks, pairs = runs[name][step - 1]
            active = {i for i, _ in picks}
            for i, j in pairs:
                ax.plot(*pos[[i, j]].T, color=ORANGE, lw=3, zorder=2)
                tag = 'random' if name == 'no memory' else 'known' if (i, j) in count else 'new'
                ax.text(*pos[[i, j]].mean(0), tag, fontsize=9.5, color=ORANGE, ha='center', va='center', zorder=5,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
        ax.scatter(*pos.T, s=60 + 900*AD_ACTIVITY, zorder=3, edgecolor='white', lw=1,
                   color=[ORANGE if i in active else '#7d7a73' for i in range(N)])
        if step == 0 and k == 0:
            callout(ax, 'often active', pos[0], (.35, 1.22))
            callout(ax, 'rarely active', pos[6], (-1.75, -1.3))
        if 1 <= step <= rounds:
            done = runs[name][:step]
            n_pairs, n_events = len({p for _, prs in done for p in prs}), sum(len(prs) for _, prs in done)
            ax.text(0, -1.35, f'{n_events} events on {n_pairs} pairs', ha='center', fontsize=10.5, color='#444444')
        if step == rounds + 1:
            a, b = sum(len(w) >= 2 for w in windows.values()), len(windows)
            ax.text(0, -1.35, f'ρ₂ = {a} of {b} pairs = {100*a/b:.0f} %', ha='center', fontsize=12, fontweight='bold')
            (i, j) = max((p for p in windows if len(windows[p]) >= 2), key=lambda p: count[p])
            ax.text(*pos[[i, j]].mean(0), '≥ 2 windows', fontsize=9.5, color=BLUE, ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
    return fig


def gif_activity(plt):
    runs = ad_example()
    gif([ad_frame(plt, runs, s) for s in range(12)], [3500] + [2000]*10 + [6500], 'gif_activity')


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
    fig_toy(plt); fig_actives(plt, f); fig_sample(plt, summary, per)
    fig_ranking(plt, summary); fig_levels(plt, pred); fig_amount(plt, pred, summary); fig_breakeven(plt, per); fig_gain(plt, per, f)
    fig_textbook(plt, types); fig_textbook_networks(plt, pred, f)
    fig_correction(plt, pred); fig_correction_networks(plt, pred, f); fig_profile(plt, pred); fig_thinking(plt, summary)
    fig_stability(plt, resp); fig_noise_networks(plt, pred, f); fig_sample_noise(plt, pred, f)
    fig_python(plt, summary); fig_python_networks(plt, per, f); fig_python_groups(plt)
    fig_agreement(plt, per, f); fig_structure(plt, perall, f); fig_cards(plt, perall, f); fig_networks(plt, per, f)
    fig_twins(plt, pred); fig_twin_methods(plt); fig_twin_networks(plt, perall, f)
    fig_synthetic_arms(plt, perall); fig_persistence(plt, perall); fig_persistence_methods(plt, perall)
    fig_persistence_levels(plt, pred); fig_windows(plt)
    gif_dar(plt); gif_activity(plt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations, api_runs and qwen_runs')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
