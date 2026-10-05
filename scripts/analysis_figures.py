"""Figures for docs/analysis/ANALYSIS.md.

In plain words: draws the figures of the written analysis from the frozen results in
docs/results/final. Nothing in docs/results/final is changed.

Two steps:
  python scripts/analysis_figures.py --inputs   rebuild docs/analysis/data/*.csv
  python scripts/analysis_figures.py            redraw figures and marked tables in ANALYSIS.md

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
from analysis_metrics import (primary_predictions, correction_residuals, noise_cases, noise_components, performance_spread,
                              paired_comparisons, answer_averaging)

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
BEFORE, AFTER = '#b8b5ad', '#333333'   # typical error of the networks as they are, and of their more persistent version
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


def graph_features(g):
    """Size, persistence and event concentration of one complete network (twins and synthetic networks)."""
    m = g.m
    return {'source': g.key, 'nodes': g.N, 'pairs': g.D, 'events': g.M, 'events_per_pair': g.M/g.D, 'rho2': g.truth[0],
            'effective_pairs': float(1/np.sum((m/g.M)**2)), 'share_one_event': float(np.mean(m == 1)),
            'share_one_window_several': float(np.mean((g.K == 1) & (m > 1))),
            **{f'share_{k}_windows': float(np.mean(g.K == k)) for k in range(2, 6)}}


def build_inputs(external):
    """Network features and answer types, as small CSV tables."""
    from study.data import prepare_real
    from study.observation import parse
    from study.estimators import design_estimate
    from study.surrogates import shuffle
    from pipeline.core import CFG
    from pipeline.real_networks import load_raw, checked_graph
    DATA.mkdir(parents=True, exist_ok=True)
    build_formula_references(external)
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    obs = {p.stem: json.loads(p.read_text()) for p in sorted((external/'api_observations').glob('*.json'))}

    # Network features from the raw data; the rebuilt truth must equal TRUTH.json exactly.
    rows, activity, twins = [], [], []
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
        twins.append(graph_features(twin))
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
    pd.DataFrame(twins).set_index('source').to_csv(DATA/'twin_features.csv', float_format='%.6g')

    # Synthetic test networks: regenerated from their fixed seeds; the truth must equal TRUTH.json.
    from study.synthetic import generate_pair
    syn = []
    for family in ('dar', 'ad'):
        for replicate in (1, 2):
            for g, _, _ in generate_pair(family, replicate):
                if abs(g.truth[0] - truth[g.key][0]) > 1e-12: raise ValueError(f'{g.key}: truth differs')
                activity += activity_rows(g)
                syn.append(graph_features(g))
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
            other, error = ~(near_rw | near_obs | near_mle), 100*a.AE2.to_numpy()
            out.append({'method': method, 'arm': arm, 'answers': len(a), 'reweighting': near_rw.mean(),
                        'observed_share': near_obs.mean(), 'mle': near_mle.mean(), 'other': other.mean(),
                        'error_observed_share_pp': error[near_obs].mean() if near_obs.any() else np.nan,
                        'error_other_pp': error[other].mean() if other.any() else np.nan})
    pd.DataFrame(out).to_csv(DATA/'answer_types.csv', index=False, float_format='%.4f')
    s = pred[(pred.arm == 'S') & pred.method.isin(LLMS) & (pred.valid == True)]
    near = (s.r2 - s.observation_id.map(design)).abs() <= .005
    near.groupby([s.method, s.source]).mean().rename('share').to_csv(DATA/'textbook_by_network.csv', float_format='%.4f')

    # Under the hood (real networks): reasoning length, whether the full trace says "guess" (GPT releases no full
    # trace), and what GPT + Python does with its code runs.
    rows, answers = [], []
    for run, method in (('openai', 'gpt_6_sol'), ('openai_tools', 'gpt_6_sol_tools'), ('deepseek', 'deepseek_flash')):
        for line in (external/'api_runs'/run/'responses.jsonl').read_text().splitlines():
            d = json.loads(line)
            oid = d['id'].rsplit('__', 2)[0]
            if d.get('kind') != 'main': continue
            if d.get('validity') == 'valid':   # every valid answer of all 32 networks, for the comparison within a sample
                answers.append({'method': method, 'arm': obs[oid]['arm'], 'sample': oid, 'tokens': d.get('reasoning_tokens') or 0,
                                'error': abs(d['prediction'][0] - truth[obs[oid]['graph_id']][0])})
            if obs[oid]['stratum'] != 'real': continue
            trace = d.get('reasoning_content')
            row = {'method': method, 'arm': obs[oid]['arm'], 'reasoning_tokens': d.get('reasoning_tokens') or 0,
                   'trace_says_guess': 'guess' in trace.lower() if trace else np.nan}
            if run == 'openai_tools':
                code = '\n'.join(x.get('code') or '' for x in d['raw_response']['output'] if x['type'] == 'code_interpreter_call')
                row.update(code_runs=d['tool_calls'], fits_numerically=bool(re.search(OPTIMISER, code)))
            rows.append(row)
    # Qwen ran on the cluster; a copy of its answer files (one JSON per answer) lies in qwen_runs. Its output tokens
    # are the thinking plus the answer of about 55 tokens.
    for path in sorted((external/'qwen_runs').glob('**/answers/thinking_r*/*.json')):
        d = json.loads(path.read_text())
        if obs.get(d['observation_id'], {}).get('stratum') != 'real': continue
        rows.append({'method': 'qwen_thinking', 'arm': d['arm'], 'reasoning_tokens': d['output_tokens'],
                     'trace_says_guess': 'guess' in d['reasoning_text'].lower()})
    r = pd.DataFrame(rows).astype({'trace_says_guess': float, 'fits_numerically': float}).groupby(['method', 'arm'])
    pd.DataFrame({'answers': r.size(), 'median_reasoning_tokens': r.reasoning_tokens.median(),
                  'trace_says_guess': r.trace_says_guess.mean(), 'mean_code_runs': r.code_runs.mean(),
                  'fits_numerically': r.fits_numerically.mean()}).to_csv(DATA/'under_the_hood.csv', float_format='%.4f')

    # Within one sample (all 32 networks): is the answer with the longest reasoning the best or the worst of the three?
    def longest(g):
        g = g[g.groupby('sample').error.transform('size').eq(3)]
        top = g.loc[g.groupby('sample').tokens.idxmax()].set_index('sample').error
        low, high = g.groupby('sample').error.min(), g.groupby('sample').error.max()
        differ = high > low
        return pd.Series({'samples': int(differ.sum()), 'longest_is_best': (top[differ] == low[differ]).mean(),
                          'longest_is_worst': (top[differ] == high[differ]).mean()})
    pd.DataFrame(answers).groupby(['method', 'arm']).apply(longest).to_csv(DATA/'reasoning_within_sample.csv', float_format='%.4f')


# Does the code that GPT + Python ran call a numerical optimiser, i.e. fit a model of its own? (keyword search)
OPTIMISER = (r'scipy\.optimize|minimize\(|least_squares\(|curve_fit\(|fsolve\(|brentq\(|root\(|differential_evolution|'
             r'nnls\(|lsq_linear|linprog')


def build_formula_references(external):
    """Freeze the R/S formula values, so MLE comparisons redraw without raw inputs."""
    from study.estimators import plugin, design_estimate
    from study.observation import parse
    rows = []
    for path in sorted((external/'api_observations').glob('*.json')):
        o = json.loads(path.read_text())
        if o['stratum'] != 'real' or o['arm'] not in 'RS': continue
        b = parse(o['block'])
        value = plugin(b)[0] if o['arm'] == 'R' else design_estimate(b)[0]
        rows.append({'observation_id': path.stem, 'reference': 100*value})
    if len(rows) != 72: raise ValueError(f'expected 72 real R/S samples, got {len(rows)}')
    pd.DataFrame(rows).to_csv(DATA/'formula_references.csv', index=False, float_format='%.12g')


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
    for ext in ('png', 'pdf'):
        kw = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {}
        fig.savefig(FIGS/f'{name}.{ext}', **kw)
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
# A possible underlying graph: R/H select nodes 2..5; S follows 0→2→3→4→5.
# Thus these views can arise from the actual node-panel and walk sampling rules.
TOY_EDGES = {'A': (0, 2), 'B': (2, 3), 'C': (3, 4), 'D': (2, 4), 'E': (4, 5), 'F': (0, 1)}
TOY_NODES = ''.join(k for k, ends in TOY_EDGES.items() if all(n in {2, 3, 4, 5} for n in ends))
TOY_PANELS = (('Full network', TOY, (), 'true ρ₂', ''),
              ('R · random nodes', {k: TOY[k] for k in TOY_NODES}, (), 'naive share', 'only pairs between drawn nodes'),
              ('S · random walk', {k: TOY[k] for k in 'ABCE'}, (), 'naive share', 'busy pairs are met more often'),
              ('H · late time only', {k: [0, 0] + TOY[k][2:] for k in TOY_NODES if sum(TOY[k][2:])}, (0, 1), 'naive share',
               'same nodes, windows 1–2 hidden'),
              ('B · event loss', {'A': [1, 0, 1, 0, 1], 'B': [0, 0, 1, 0, 0], 'E': [0, 0, 1, 0, 0]}, (), 'naive share', 'most events are lost'))


def fig_toy(plt):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), gridspec_kw=dict(wspace=.36))
    fig.suptitle('Same graph, different samples, different apparent persistence', x=.04, ha='left',
                 fontsize=14, fontweight='bold', y=1.13)
    fig.text(.04, 1.02, 'ρ₂ = share of pairs active in ≥ 2 windows',
             fontsize=11, color='#444444')
    for ax, (title, pairs, hidden, what, caption) in zip(axes, TOY_PANELS):
        for j in hidden: ax.add_patch(plt.Rectangle((j - .5, -.5), 1, 6, color='#efede8', lw=0))
        seen = persistent = 0
        for i, k in enumerate(TOY):
            row = pairs.get(k)
            ax.text(-.95, i, k, ha='center', va='center', fontsize=9.5, color='#222222' if row else '#c9c6bd')
            ax.plot([-.35, 4.35], [i, i], color='#eeece7', lw=1, zorder=0)
            if not row:
                ax.text(5.05, i, '–', ha='center', va='center', fontsize=10, color='#c9c6bd')
                continue
            windows = sum(x > 0 for x in row); seen += 1; persistent += windows >= 2
            ax.text(5.05, i, '1' if windows >= 2 else '0', ha='center', va='center', fontsize=12,
                    fontweight='bold', color=BLUE if windows >= 2 else GREY)
            for j in range(5):
                if row[j]: ax.scatter(j, i, s=40 + 35*row[j], color=BLUE if windows >= 2 else GREY, zorder=3)
        ax.set_xlim(-1.3, 5.6); ax.set_ylim(5.6, -1.0); ax.axis('off')
        for j in range(5): ax.text(j, -.85, j + 1, ha='center', fontsize=8.5, color='#999999')
        ax.text(5.05, -.85, 'count', ha='center', fontsize=8.5, color='#666666')
        ax.set_title(title, loc='left', fontsize=11, pad=8)
        share = persistent/seen
        ax.text(2.1, 6.35, f'{persistent}/{seen} = {100*share:.0f} %', ha='center', fontsize=15, fontweight='bold')
        delta = 100*(share - .5)
        label = 'true persistence' if what == 'true ρ₂' else f'sample: {delta:+.0f} pp' if delta else 'sample: same share'
        ax.text(2.1, 7.05, label, ha='center', fontsize=10, color='#444444')
    axes[0].text(-.95, -.85, 'pair', ha='center', fontsize=8.5, color='#999999')
    note(fig, 'share = sum of counts / pairs seen · –: unseen, excluded · columns: time windows · bigger dot: more events', -.2)
    save(fig, 'fig0_toy')


# ---------------------------------------------------------------- shared pieces
REAL_NOTE = '12 real networks · 3 samples each'
LM3 = tuple((m, COLOUR[m]) for m in ('gpt_6_sol', 'deepseek_flash', 'qwen_thinking'))
MLE_LM3 = (('mle', BLUE),) + LM3
LEVELS = ['ρ₂', 'ρ₃', 'ρ₄', 'ρ₅']


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
    """Keep only persistence beside network names; sizes live in the overview table."""
    return f"true ρ₂ {pct(f.loc[s, 'rho2'])}"


def twin_spec(s, f):
    """The same for a time-shuffled twin: its persistence next to the real network's."""
    return f"ρ₂ {pct(f.loc[s, 'rho2'])[:-2]} → {pct(f.loc[s, 'rho2_shuffled'])}"


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


def row_names(ax, order, f, x=-.02):
    """Row labels of the real networks: name (true ρ₂)."""
    ax.set_yticks(range(len(order)), ['']*len(order))
    for i, s in enumerate(order): two_tone(ax, x, i, short(s), pair_spec(s, f), ax.get_yaxis_transform())


def pair_bars(ax, first, second, labels, colours, title, ymax, spreads=None):
    """Two bars per arm with their values on top: the same quantity under two conditions.
    spreads: per condition a second, small grey number written as '± x' above the value."""
    w = .38
    for k, (values, c, lab) in enumerate(zip((first, second), colours, labels)):
        xs = np.arange(4) + (k - .5)*w
        ax.bar(xs, values, width=w*.92, color=c, label=lab)
        for i, (x, v) in enumerate(zip(xs, values)):
            value = ax.annotate(f'{v:.1f}', (x, v), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9.5)
            if spreads: ax.annotate(f'± {spreads[k][i]:.1f}', (.5, 1), xycoords=value, xytext=(0, 1), textcoords='offset points',
                                    ha='center', va='bottom', fontsize=8, color=SPEC)
    ax.set_xticks(range(4), [ARMS[a] for a in ARMS]); ax.set_ylim(0, ymax); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, ncol=2, loc='upper left'); ax.set_title(title, pad=12)


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


def network_dots(plt, ax, order, values, xlab, xlim, f, models=LM3, names=True):
    """One row per network; one coloured dot per model."""
    y = np.arange(len(order))
    for yi in y: ax.axhline(yi, color='#f0efeb', lw=6, zorder=0)
    for k, (m, c) in enumerate(models):
        v = [values[m].get(s, np.nan) for s in order]
        ax.scatter(v, y + (k - (len(models) - 1)/2)*.6/len(models), s=42, color=c, edgecolor='white', lw=.7, zorder=3, label=METHODS[m])
    if names: row_names(ax, order, f)
    ax.set_xlim(*xlim); ax.set_xlabel(xlab)
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
def fig_ranking(plt, summary, spread, name='fig2_ranking', label='12 real networks'):
    methods = ['plugin', 'median', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    xmax = 100*summary.MAE_2.max() + 16
    for ax, arm in zip(axes.flat, ARMS):
        ranked = sorted(methods, key=lambda m: summary.loc[(arm, m), 'MAE_2'])[::-1]
        v = [100*summary.loc[(arm, m), 'MAE_2'] for m in ranked]
        y = np.arange(len(ranked))
        ax.barh(y, v, color=[colour(m) for m in ranked], height=.7)
        for yi, vi, m in zip(y, v, ranked):
            mean = ax.annotate(f'{vi:.1f}', (vi, yi), xytext=(3, 0), textcoords='offset points', va='center',
                               fontsize=10, fontweight='bold' if m == 'plugin' else 'normal')
            ax.annotate(f"± {spread.loc[(arm, m), 'sd']:.1f}", (1, .5), xycoords=mean, xytext=(3, 0),
                        textcoords='offset points', va='center', fontsize=8, color=SPEC)
        ax.set_yticks(y, [METHODS[m] for m in ranked])
        for lab, m in zip(ax.get_yticklabels(), ranked):
            if m == 'plugin': lab.set_fontweight('bold')
        ax.set_xlim(0, xmax); ax.set_xticks([]); ax.spines['bottom'].set_visible(False); ax.set_title(ARMS[arm])
    fig.subplots_adjust(wspace=.62, hspace=.28)
    note(fig, 'mean error (pp) · grey: ± SD across ' + label, .04)
    save(fig, name)


def value_column(ax, y, v, x):
    """The bars' values as a column at the right edge, clear of any reference line."""
    for yi, vi in zip(y, v): ax.text(x, yi, f'{round(vi):+d}' if round(vi) else '0', va='center', ha='right', fontsize=9)


# Fig. 2c: residual after correction; mean signed error and middle 80 % of estimates.
def fig_amount(plt, pred):
    methods = ['mle', 'et', 'gpt_6_sol', 'deepseek_flash', 'qwen_thinking']
    a = correction_residuals(pred)
    # A residual of zero means precisely the required correction was made.
    # Average within each network, then give all networks equal weight.
    stats = a.groupby(['arm', 'method', 'source']).agg(bias=('residual', 'mean'), mae=('residual', lambda v: v.abs().mean()))
    stats.groupby(['arm', 'method']).mean().to_csv(DATA/'correction_reliability.csv', float_format='%.6g')
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, arm in zip(axes, ARMS):
        ax.axvline(0, color='#333333', lw=1.2)
        for yi, m in zip(y, methods):
            v = a[(a.arm == arm) & a.method.eq(m)].residual
            lo, hi = v.quantile([.1, .9])
            mean = stats.loc[(arm, m)].bias.mean()
            ax.plot([lo, hi], [yi, yi], color=colour(m), lw=3, alpha=.6)
            ax.scatter(mean, yi, color=colour(m), s=45, zorder=3, edgecolor='white', lw=.6)
        ax.set_xlim(-50, 50); ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_title(ARMS[arm])
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlabel('estimate − truth (pp)')
    fig.legend([plt.Line2D([], [], marker='o', ls='', color='#555555'), plt.Line2D([], [], color='#555555', lw=3)],
               ['mean error with sign', 'middle 80 % of individual estimates'],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .98))
    note(fig, '0 = exactly the needed correction · ' + REAL_NOTE + ' · spread, not a confidence interval', -.07)
    save(fig, 'fig2c_amount')


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
    fig, ax = plt.subplots(figsize=(7, 3.1))
    grouped_bars(plt, ax, ['R', 'S'], vals, 112, '{:.0f} %', 'R/S: LLM answers often match the formula', LM3)
    note(fig, 'match: within 0.5 pp · 12 networks × 3 samples × 3 answers', -.2)
    save(fig, 'fig3_textbook')


# Fig. 3b: arm S per network: share of answers equal to the simple reweighting.
def fig_textbook_networks(plt, pred, f):
    m = pd.read_csv(DATA/'textbook_by_network.csv').set_index(['method', 'source']).share
    order = network_order(f)
    vals = {mm: {s: 100*m[(mm, s)] for s in order} for mm, _ in LM3}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    network_dots(plt, ax, order, vals,
                 'answers matching the simple reweighting (%)', (-5, 105), f, LM3)
    ax.set_title('S · random walk, per network', pad=12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.45, -.14))
    note(fig, '9 answers per network · match: within 0.5 pp', -.18)
    save(fig, 'fig3b_textbook_networks')


# Fig. 4: share of answers that correct by about the right amount (50–150 % of the needed correction), in H and B.
def correction_table(pred):
    """Per answer: needed = truth − naive share, done = answer − naive share; samples off by ≥ 5 pp only."""
    p = r2_answers(pred)
    naive = p[p.method == 'plugin'].groupby('observation_id').r2.first()
    a = p[p.method.isin(LLMS + ['mle']) & (p.valid == True) & (p.replicate.isna() | p.replicate.eq(0))].copy()
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
    grouped_bars(plt, ax, ['H', 'B'], vals, 92, '{:.0f} %', 'Corrections within 50–150 % of the needed correction', MLE_LM3)
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
    network_dots(plt, ax, order, vals, 'corrections within 50–150 % of the needed correction (%)', (-5, 105), f, MLE_LM3)
    ax.set_title('H and B together, per network', pad=12)
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.4, -.14), columnspacing=1.2)
    note(fig, 'networks sorted by true ρ₂ · only samples off by ≥ 5 pp (Digg and Linux: none)', -.18)
    save(fig, 'fig4b_correction_networks')


# Fig. 5a/b: simple bars, one controlled contrast per figure, on the same scale.
def noise_bars(plt, values, name, title, footer, methods, xmax, labels=METHODS):
    fig, axes = plt.subplots(1, 4, figsize=(14, max(3., .45*len(methods) + 1.2)), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, arm in zip(axes, ARMS):
        v = [values[(arm, m)] for m in methods]
        ax.barh(y, v, height=.58, color=[colour(m) for m in methods])
        for yi, x in zip(y, v):
            ax.text(x + .015*xmax, yi, f'{x:.1f}', fontsize=10, va='center')
        ax.set_yticks(y, [labels[m] for m in methods])
        ax.set_xlim(0, xmax); ax.set_xticks(np.linspace(0, xmax, 6))
        ax.set_xlabel('spread (SD, pp)'); ax.set_title(ARMS[arm])
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#f0efeb'); ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.05)
    note(fig, footer, -.08)
    save(fig, name)


def answer_spread(pred):
    """Median SD of the three answers to the same sample (pp), per arm and language model; complete repeats only."""
    p = primary_predictions(pred)
    a = p[p.method.isin(LLMS)].groupby(['arm', 'method', 'observation_id']).rho2.agg(['count', 'std'])
    return a[a['count'].eq(3)].groupby(['arm', 'method'])['std'].median()


def fig_stability(plt, pred):
    """Two controlled contrasts, directly measured, with one fixed ExtraTrees fit."""
    cases = noise_cases(pred)
    cases.to_csv(DATA/'noise_by_network.csv', index=False, float_format='%.8g')
    noise_components(cases).to_csv(DATA/'noise_components.csv', index=False, float_format='%.8g')
    c = cases[cases.target.eq(2)]
    # Match the frozen variability report: median of per-network between-sample
    # SDs, and median of per-observation answer SDs (complete repeats only).
    between = c.groupby(['arm', 'method']).between_sd.median()
    response = answer_spread(pred)
    noise_bars(plt, between, 'fig5_sample_variation', 'Sample-redraw noise · 3 draws · MLE / ExtraTrees',
               'median across 12 networks · fixed estimators · LLMs left out: their redraws also contain answer noise',
               ('mle', 'et'), 5)
    # ExtraTrees gives one answer per sample; its counterpart is a new training fit on the same sample.
    training = pd.read_csv(FINAL/'VARIABILITY_TRAINING.csv').query("group == 'real'").set_index('arm')
    for arm in ARMS: response[(arm, 'et')] = 100*training.loc[arm, 'median_observation_SD_rho2']
    noise_bars(plt, response, 'fig5_stability', 'Answer-repeat noise · 3 LLM answers to the same sample',
               'median SD across samples · LLMs: 3 valid answers · ExtraTrees: 11 training fits instead', LLMS + ['et'], 25,
               {**METHODS, 'et': 'ExtraTrees, retrained'})


# Fig. 5c: does asking three times help? Error of one answer and of the mean of the three answers to the same sample.
def fig_averaging(plt, pred):
    from matplotlib.colors import to_rgba
    a = answer_averaging(pred)
    pd.concat({g: answer_averaging(pred, g) for g in ('real', 'surrogate', 'synthetic')}, names=['group']
              ).to_csv(DATA/'answer_averaging.csv', float_format='%.4g')
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, (m, c) in zip(axes, LM3):
        pair_bars(ax, a.loc[m].one_answer[list(ARMS)], a.loc[m].mean_of_3[list(ARMS)], ['one answer', 'mean of 3 answers'],
                  [to_rgba(c, .4), c], METHODS[m] + ': error (pp)', 27)
        ax.set_xticks(range(4), list(ARMS))   # three panels side by side: the letters only
    fig.subplots_adjust(wspace=.08)
    note(fig, REAL_NOTE + ' × 3 answers · the 3 answers to the same sample are averaged, then scored · '
         + ' · '.join(ARMS.values()), -.1)
    save(fig, 'fig5_averaging')


def write_relations(perall, f, pred):
    """Numbers behind the general statements of the analysis, as one small table, so every 'why' can be looked up.
    Rank correlations are Spearman; group says which networks they rest on."""
    per = perall.query("group == 'real'")
    rows = []
    add = lambda relation, arm, value, method='', group='real': rows.append(
        dict(relation=relation, group=group, method=method, arm=arm, value=value))
    rho = lambda a, b: pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)), method='spearman')
    cases = noise_cases(pred).query('target == 2')
    typical = typical_error(perall)
    seen = {a: f.walk_distinct_pairs if a == 'S' else f[f'pairs_seen_{a}'] for a in ARMS}
    for arm in ARMS:
        e = per[per.arm == arm].pivot(index='source', columns='method', values='MAE_2').loc[f.index, MAIN]
        c = e.corr(method='spearman').to_numpy()
        add('six methods agree on which networks are hard (mean pairwise rank correlation)', arm, c[np.triu_indices(len(MAIN), 1)].mean())
        add('typical error vs effective pairs', arm, rho(f.effective_pairs, typical.loc[f.index, arm]))
        redraw = cases[cases.arm.eq(arm) & cases.method.isin(['mle', 'et'])].groupby('source').between_sd.mean().loc[f.index]
        add('redraw SD (MLE, ExtraTrees) vs pairs in the sample', arm, rho(seen[arm], redraw))
    answers = cases[cases.method.isin([m for m, _ in LM3])].set_index(['arm', 'method', 'source']).answer_sd.sort_index()
    weights = pd.read_csv(DATA/'textbook_by_network.csv').set_index(['method', 'source']).share
    for m, _ in LM3:
        for arm in 'HB': add('answer SD vs true rho_2', arm, rho(f.rho2, answers.loc[(arm, m)].loc[f.index]), m)
        add('answer SD: same networks noisy in H and B', 'H/B', rho(answers.loc[('H', m)].loc[f.index], answers.loc[('B', m)].loc[f.index]), m)
        add('share of S answers using the weights vs pairs in the network', 'S', rho(f.pairs, weights.loc[m].loc[f.index]), m)
    add('pairs seen by a walk relative to random nodes (median ratio)', 'S', (f.walk_distinct_pairs/f.pairs_seen_R).median())
    add('true rho_2 vs keep rate p', 'B', rho(f.rho2, f.kept_share_B))
    # What makes a network hard: in every sampler (effective pairs) and per sampler, on all 32 networks, within each
    # group, and the weakest value when any one network is left out.
    a = all_features()
    groups = {lab: keys for lab, keys, _ in network_groups(typical.index)}
    for properties in (EFFECTIVE_PAIRS, HARD_PROPERTY):
        for arm, (col, label, _) in properties.items():
            relation = 'typical error vs ' + ('effective pairs' if col == 'effective_pairs' else label)
            for lab, keys in {'all 32': list(typical.index), **groups}.items():
                if not (lab == 'real' and col == 'effective_pairs'): add(relation, arm, rho(a.loc[keys, col], typical.loc[keys, arm]), group=lab)
            left_out = [rho(a.loc[typical.index.drop(s), col], typical.loc[typical.index.drop(s), arm]) for s in typical.index]
            add(relation + ', weakest when one network is left out', arm, min(left_out, key=abs), group='all 32')
    for lab, keys in groups.items():
        for col, label in (('events_per_pair', 'events per pair'), ('bursty_pairs', 'bursty pairs (%)'), ('rho2', 'true rho_2 (%)')):
            add(f'median {label}', '', a.loc[keys, col].median(), group=lab)
    high = a.rho2 > 30
    e = perall[perall.arm == 'B'].pivot(index='source', columns='method', values='MAE_2').loc[a.index]*100
    for m in MAIN: add(f'networks with true rho_2 > 30 % ({int(high.sum())}) where the B error stays below 10 pp', 'B', int((e.loc[high, m] < 10).sum()), m, 'all 32')
    # Single estimates of all networks: order of the networks, large errors, and what is left when noise is averaged out.
    est = pd.concat([primary_predictions(pred, g) for g in ('real', 'surrogate', 'synthetic')])
    est = est[est.method.isin(['plugin'] + MAIN)].assign(truth=lambda d: 100*d.truth_rho2)
    net = est.groupby(['group', 'arm', 'method', 'source']).agg(estimate=('rho2', 'mean'), truth=('truth', 'first'))
    for arm in ARMS:
        for m in ('plugin', 'mle', 'et', 'gpt_6_sol'):
            for lab, d in (('real', net.loc[('real', arm, m)]), ('all 32', net.xs((arm, m), level=('arm', 'method')))):
                add('order of the networks: estimate vs truth', arm, rho(d.estimate, d.truth), m, lab)
        for m in MAIN:
            single = est[est.group.eq('real') & est.arm.eq(arm) & est.method.eq(m)]
            add('share of single estimates more than 20 pp off (%)', arm, 100*((single.rho2 - single.truth).abs() > 20).mean(), m)
            d = net.loc[('real', arm, m)]
            add('error left when all estimates of a network are averaged (pp)', arm, (d.estimate - d.truth).abs().mean(), m)
    very = net.xs('B', level='arm'); very = very[very.truth > 60]
    for m in ('mle', 'et', 'gpt_6_sol'):
        d = very.xs(m, level='method')
        add(f'mean signed error on the {len(d)} networks with true rho_2 > 60 % (pp)', 'B', (d.estimate - d.truth).mean(), m, 'all 32')
    pd.DataFrame(rows).to_csv(DATA/'relations.csv', index=False, float_format='%.2f')


def fig_noise_by_graph(plt, pred, f):
    """Two compact details: pure redraw SDs, then pooled LLM answer SDs."""
    c = noise_cases(pred).query('target == 2')
    order = network_order(f)
    y = np.arange(len(order))
    v = c[c.method.isin(['mle', 'et'])].groupby(['method', 'source']).between_sd.mean()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for k, m in enumerate(('mle', 'et')):
        xs = v.loc[m].reindex(order)
        ys = y + (.5 - k)*.4
        ax.barh(ys, xs, height=.30, color=colour(m), label=METHODS[m])
        for yi, x in zip(ys, xs): ax.text(x + .08, yi, f'{x:.1f}', va='center', fontsize=8.5)
    row_names(ax, order, f)
    ax.set_ylim(-.6, len(order)-.4); ax.set_xlim(0, v.max() + .8)
    ax.set_xlabel('sample-redraw noise (mean SD, pp)'); ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0))
    note(fig, 'mean over R/S/H/B · each SD: 3 redraws · fixed estimators', -.06)
    save(fig, 'fig5_redraw_networks')

    v = c[c.method.isin([m for m, _ in LM3])].set_index(['arm', 'method', 'source']).answer_sd.sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True, gridspec_kw=dict(wspace=.08))
    for ax, arm in zip(axes, 'HB'):
        network_dots(plt, ax, order, {m: v.loc[(arm, m)] for m, _ in LM3}, 'answer-repeat noise (SD, pp)', (-1.5, 45), f, names=arm == 'H')
        ax.set_title(ARMS[arm], pad=12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.6, .97))
    note(fig, 'networks sorted by true ρ₂ · SD of 3 answers to the same sample, median over samples · R/S: almost none', -.04)
    save(fig, 'fig5_answers_networks')

    # The same numbers as H against B: a stable pattern would put the networks on the diagonal.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, (m, col) in zip(axes, LM3):
        x, yv = v.loc[('H', m)].loc[order], v.loc[('B', m)].loc[order]
        ax.plot([0, 45], [0, 45], color='#bbbbbb', lw=1, zorder=1)
        ax.scatter(x, yv, s=46, color=col, edgecolor='white', lw=.7, zorder=3)
        ax.set_xlim(-1.5, 45); ax.set_ylim(-1.5, 45); ax.set_aspect('equal'); ax.set_title(METHODS[m])
        ax.set_xlabel('answer noise in H (SD, pp)'); ax.spines['left'].set_visible(True)
        ax.text(.97, .04, f'rank correlation {x.corr(yv, method="spearman"):.2f}', transform=ax.transAxes, ha='right', fontsize=9, color=SPEC)
    axes[0].set_ylabel('answer noise in B (SD, pp)')
    note(fig, 'one dot per real network · on the line: equally noisy in H and B', -.04)
    save(fig, 'fig5_answers_h_vs_b')


def python_bars(ax, summary, title):
    """GPT with and without Python: error per arm (pp)."""
    pair = ('gpt_6_sol', 'gpt_6_sol_tools')
    first, second = ([100*summary.loc[(a, m), 'MAE_2'] for a in ARMS] for m in pair)
    pair_bars(ax, first, second, [METHODS[m] for m in pair], [colour(m) for m in pair], title, 18)


# Fig. 6: GPT with and without Python: error per arm; in grey how much the three answers to a sample differ.
def fig_python(plt, summary, pred):
    pair = ('gpt_6_sol', 'gpt_6_sol_tools')
    spread = answer_spread(pred)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    pair_bars(ax, *([100*summary.loc[(a, m), 'MAE_2'] for a in ARMS] for m in pair), [METHODS[m] for m in pair],
              [colour(m) for m in pair], 'Error (pp)', 21, spreads=[[spread[(a, m)] for a in ARMS] for m in pair])
    note(fig, REAL_NOTE + ' × 3 answers · grey: ± spread of the 3 answers to a sample (SD)', -.1)
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


# Fig. 7b: on which networks MLE is closer and on which GPT, in H and B (error per network, pp).
def fig_method_networks(plt, per, f):
    order = network_order(f)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True, gridspec_kw=dict(wspace=.08))
    for ax, arm in zip(axes, 'HB'):
        e = per[per.arm == arm].pivot(index='source', columns='method', values='MAE_2')*100
        network_dots(plt, ax, order, {m: e[m] for m in ('mle', 'gpt_6_sol')}, 'error (pp)', (-1, 22), f, MLE_LM3[:2],
                     names=arm == 'H')
        ax.set_title(ARMS[arm], pad=12)
    axes[1].legend(frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1, 1.0), columnspacing=1, handletextpad=.2)
    note(fig, 'networks sorted by true ρ₂ · ' + REAL_NOTE + ' (GPT: × 3 answers)', -.04)
    save(fig, 'fig7b_method_networks')


# Fig. 8b: typical error against the number of pairs that carry the events, per arm.
def fig_structure(plt, perall, f):
    t = typical_error(perall).loc[f.index]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)
    for ax, arm in zip(axes, ARMS):
        ax.scatter(f.effective_pairs, t[arm], s=40, color='#333333', zorder=3)
        two_tone(ax, f.loc['sp_malawi', 'effective_pairs'], t.loc['sp_malawi', arm], 'Malawi', '55 pairs carry them', ax.transData,
                 ha='left', size=8.5, small=8, dx=3, dy=9)
        ticks = [100, 1000, 10000, 100000]
        ax.set_xscale('log'); ax.set_xticks(ticks, [f'{x:,}' for x in ticks]); ax.minorticks_off()
        ax.set_xlabel('pairs that carry the events', fontsize=9.5); ax.set_ylim(0, 36); ax.set_yticks([0, 10, 20, 30])
        ax.spines['left'].set_visible(True); ax.set_title(ARMS[arm])
        correlation_label(ax, f.effective_pairs.corr(t[arm], method='spearman'), right=True, y=.8)   # clear of the Malawi label
    axes[0].set_ylabel('typical error (pp)')
    note(fig, '12 real networks · typical error = median of six methods'
         ' · pairs that carry the events = effective number of pairs (see definitions)', -.12)
    save(fig, 'fig8b_pairs')


def all_features():
    """Size, events per pair, bursty pairs and true rho_2 of all 32 networks (real, time-shuffled twins, synthetic)."""
    cols = ['nodes', 'pairs', 'events', 'events_per_pair', 'rho2', 'effective_pairs', 'share_one_window_several']
    a = pd.concat([pd.read_csv(DATA/f'{name}_features.csv', index_col=0)[cols] for name in ('network', 'twin', 'synthetic')])
    a['bursty_pairs'] = 100*a.share_one_window_several   # pairs with several events, all in one window (% of pairs)
    a['rho2'] *= 100
    return a


def correlation_label(ax, r, right=False, y=.95):
    """The rank correlation of a panel, small and grey in its upper corner."""
    ax.text(.97 if right else .03, y, f'rank correlation {r:+.2f}'.replace('-', '−'), transform=ax.transAxes, va='top',
            ha='right' if right else 'left', fontsize=9, color=SPEC)


# What makes a network hard: (column of all_features, axis label, ticks of a log axis or None), per sampler.
HARD_PROPERTY = {'R': ('nodes', 'nodes in the network', [100, 1000, 10000]), 'S': ('events_per_pair', 'events per pair', [1, 10, 100]),
                 'H': ('bursty_pairs', 'bursty pairs (%)', None), 'B': ('rho2', 'true ρ₂ (%)', None)}
EFFECTIVE_PAIRS = {arm: ('effective_pairs', 'pairs that carry the events', [100, 1000, 10000, 100000]) for arm in ARMS}


# Fig. 8d, 9e: typical error against a network property per sampler, on the real networks or on all 32.
def fig_property(plt, perall, name, properties, real_only):
    a, t = all_features(), typical_error(perall)
    keys = [s for s in t.index if s in NAMES] if real_only else list(t.index)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        col, label, ticks = properties[arm]
        for lab, group, style in network_groups(keys): ax.scatter(a.loc[group, col], t.loc[group, arm], s=34, zorder=3, label=lab, **style)
        if ticks: ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(ticks, [f'{x:,}' for x in ticks])
        ax.set_ylim(0, 36); ax.set_title(ARMS[arm]); ax.set_xlabel(label); ax.spines['left'].set_visible(True)
        correlation_label(ax, a.loc[keys, col].corr(t.loc[keys, arm], method='spearman'), right=col == 'effective_pairs')
    axes[0].set_ylabel('typical error (pp)')
    if not real_only:
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .98))
    bursty = ' · bursty pairs = pairs with several events, all in one time window' if properties is HARD_PROPERTY else ''
    note(fig, ('12 real networks' if real_only else '32 networks: 12 real, their 12 time-shuffled twins, 8 synthetic')
         + ' · typical error = median of six methods' + bursty, -.08)
    save(fig, name)


def network_groups(sources, c='#333333'):
    """The three groups of networks with their marker: real filled, time-shuffled twin hollow, synthetic square."""
    return (('real', [s for s in sources if s in NAMES], dict(color=c)),
            ('time-shuffled twin', [s for s in sources if s.endswith('__pwt')], dict(color='white', edgecolor=c, lw=1.1)),
            ('synthetic', [s for s in sources if s not in NAMES and not s.endswith('__pwt')], dict(color=c, marker='s')))


# Fig. 9: every main method (rows) in every sampler (columns): error against true rho_2, all 32 networks.
def fig_all_networks(plt, perall):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    e = perall.pivot_table(index='source', columns=['arm', 'method'], values='MAE_2')*100
    rho = pd.Series({s: 100*truth[s][0] for s in e.index})
    fig, axes = plt.subplots(len(MAIN), 4, figsize=(13, 13.5), sharex=True, sharey=True, gridspec_kw=dict(hspace=.16, wspace=.08))
    for row, m in zip(axes, MAIN):
        for ax, arm in zip(row, ARMS):
            for lab, keys, style in network_groups(e.index, colour(m)):
                ax.scatter(rho[keys], e.loc[keys, (arm, m)], s=22, zorder=3, **style)
            ax.axhline(10, color='#bbbbbb', lw=.8, ls=(0, (4, 2)), zorder=1)
            ax.set_xlim(0, 90); ax.set_ylim(0, 56); ax.spines['left'].set_visible(True)
        row[0].set_ylabel(METHODS[m], fontsize=11.5, fontweight='bold', color=colour(m))
    for ax, arm in zip(axes[0], ARMS): ax.set_title(ARMS[arm])
    for ax in axes[-1]: ax.set_xlabel('true ρ₂ (%)')
    h = [plt.Line2D([], [], marker=mk, ls='', color='#555555', markerfacecolor=fc, markersize=6)
         for mk, fc in (('o', '#555555'), ('o', 'white'), ('s', '#555555'))]
    fig.legend(h, ['real', 'time-shuffled twin', 'synthetic'], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .9))
    note(fig, 'y-axis: error (pp) · 32 networks: 12 real, their 12 time-shuffled twins, 8 synthetic · dashed: 10 pp', .075)
    save(fig, 'fig9_all_networks')


# Fig. 9b: the same for the higher levels of the profile: typical error of rho_3, rho_4, rho_5 against their true values.
def fig_persistence_levels(plt, pred):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    e = level_errors(pred)
    t = e[e.index.get_level_values('method').isin(MAIN)].groupby(['source', 'arm']).median()
    sources = t.index.get_level_values('source').unique()
    groups = network_groups(sources)
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
    save(fig, 'fig9b_levels')


# Fig. 10: time-shuffled twins. How much each method raises its estimate from a network to its twin (truth: +27 pp).
def fig_twins(plt, pred):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    methods = ['plugin', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
    p = pred[pred.group.isin(['real', 'surrogate']) & pred.prediction.notna() & ((pred.valid == True) | pred.method.isin(['plugin', 'mle', 'et']))].copy()
    p = p[p.method.isin(methods) & (p.replicate.isna() | (p.replicate == 0))]
    p['r2'] = p.prediction.map(lambda v: json.loads(v)[0])*100
    p['family'] = p.source.str.replace('__pwt', '', regex=False)
    est = p.groupby(['arm', 'method', 'group', 'family']).r2.mean().unstack('group')
    change = (est.surrogate - est.real).groupby(['arm', 'method']).mean()
    true = np.mean([100*(truth[s+'__pwt'][0] - truth[s][0]) for s in NAMES])
    change.unstack().to_csv(DATA/'twins_change.csv', float_format='%.2f')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
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


# Fig. 10c: are the twins harder? Typical error on the 12 real networks and on their twins, per arm.
def fig_twin_error(plt, perall, f):
    t = typical_error(perall)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    pair_bars(ax, t.loc[list(f.index)].mean()[list(ARMS)], t.loc[[s + '__pwt' for s in f.index]].mean()[list(ARMS)],
              ['real networks', 'their time-shuffled twins'], [BEFORE, AFTER], 'Typical error (pp)', 20)
    note(fig, '12 real networks and their twins · 3 samples each · typical error = median of six methods', -.1)
    save(fig, 'fig10c_twin_error')


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
    fig_active(plt, 'fig0b_active', [([s], short(s), pair_spec(s, f)) for s in order], '12 real networks, complete (not sampled)')
    fig_active(plt, 'fig10b_twin_active', [([s + '__pwt'], short(s), twin_spec(s, f)) for s in order], 'the 12 time-shuffled twins, complete')
    sv = variant_table()
    fig_active(plt, 'fig12_synthetic_active', [([f'{v}_r1', f'{v}_r2'], lab,
               f"{num(sv.loc[v, 'pairs'])} pairs · ρ₂ {pct(sv.loc[v, 'rho2'])}") for v, lab in VARIANTS],
               'mean of the 2 instances per variant, complete networks')


# Fig. 13: what memory does to the estimation task: typical error without and with memory, per arm and generator.
def fig_memory(plt, perall):
    t = typical_error(perall)
    mean = lambda key: t.loc[[f'{key}_r1', f'{key}_r2']].mean()[list(ARMS)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.2))
    for ax, (name, without, with_) in zip(axes, (('DAR', 'dar_a0', 'dar_a08'), ('Activity-driven', 'ad_memoryless', 'ad_memory'))):
        pair_bars(ax, mean(without), mean(with_), ['no memory', 'memory'], [BEFORE, AFTER], f'{name}: typical error (pp)', 20)
    fig.subplots_adjust(wspace=.1)
    note(fig, '8 synthetic networks (2 instances per variant) · 3 samples each · typical error = median of six methods', -.1)
    save(fig, 'fig13_memory')


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


def dar_example(seed=13, P=6, chi=.2, alpha=.8):
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
    """One binary state per pair/window; copying OFF is as explicit as copying ON."""
    P = events.shape[1]
    fig = plt.figure(figsize=(10.8, 5.1))
    fig.text(.035, .95, 'DAR · copy the last ON or OFF state', fontsize=15, fontweight='bold')
    fig.text(.035, .875, 'New draw: ON with 20 %, otherwise OFF', fontsize=11, color='#555555')
    where = {0: 'start', 6: 'events', 7: 'result'}.get(step, f'window {step} / 5')
    fig.text(.965, .95, where, fontsize=12, color=ORANGE, ha='right', fontweight='bold')
    j = step - 1 if 1 <= step <= 5 else None
    for k, name in enumerate(('no memory', 'memory')):
        on, kept = runs[name]
        ax = fig.add_axes([.06 + .49*k, .22, .41, .49])
        ax.set_xlim(-1.2, 4.7); ax.set_ylim(P - .4, -1.0); ax.axis('off')
        rule = 'No memory · new draw' if k == 0 else 'Memory · 80 % same state, 20 % new draw'
        fig.text(.045 + .49*k, .78, rule, fontsize=12, fontweight='bold')
        shown = min(step, 5)
        K = on[:shown].sum(0)
        if j is not None: ax.add_patch(plt.Rectangle((j - .45, -.55), .9, P + .1, color='#f3efe6', lw=0))
        for c in range(5):
            ax.text(c, -.65, c + 1, ha='center', fontsize=10, color='#222222' if c == j else '#999999',
                    fontweight='bold' if c == j else 'normal')
        for i in range(P):
            if step == 7 and K[i] >= 2: ax.add_patch(plt.Rectangle((-.55, i - .42), 5.1, .84, color=PERSIST, lw=0))
            ax.text(-.65, i, f'pair {"ABCDEF"[i]}', ha='right', va='center', fontsize=10,
                    color='#c4c1b8' if step == 7 and K[i] == 0 else '#222222')
            for c in range(5):
                if c >= shown: continue
                if c and kept[c, i]:
                    ax.annotate('', (c - .18, i), (c-1 + .18, i),
                                arrowprops=dict(arrowstyle='->', lw=1.4, color=ORANGE if c == j else '#bbbbbb'))
                if on[c, i]: ax.scatter(c, i, s=90, color=BLUE, zorder=3)
                else: ax.scatter(c, i, s=42, marker='x', color=GREY, lw=1.7, zorder=3)
        if k and j is not None and j > 0:
            # Show one copied OFF state explicitly; arrows in the grid show all copies.
            copied_off = np.flatnonzero(kept[j] & ~on[j])
            copied_on = np.flatnonzero(kept[j] & on[j])
            if len(copied_off):
                i = copied_off[0]
                ax.text((j - .5), i - .22, 'OFF → OFF', fontsize=8, color=ORANGE, ha='center',
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
            if len(copied_on):
                i = copied_on[0]
                ax.text((j - .5), i - .22, 'ON → ON', fontsize=8, color=ORANGE, ha='center',
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
        if step == 7:
            a, b = int((K >= 2).sum()), int((K >= 1).sum())
            fig.text(.045 + .49*k, .155, f'ρ₂ = {a}/{b} = {100*a/b:.0f} %', fontsize=13, fontweight='bold')
    key = 'Blue rows: ≥ 2 windows · never-ON pairs excluded' if step == 7 else '● ON    × OFF    → keep previous state'
    fig.text(.045, .065, key, fontsize=11, color='#555555')
    return fig


def gif_dar(plt):
    runs, events = dar_example()
    steps = [1, 2, 3, 4, 5, 7]
    gif([dar_frame(plt, runs, events, s) for s in steps], [3500]*5 + [7000], 'gif_dar')


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
    fig = plt.figure(figsize=(10.8, 6.1))
    fig.text(.03, .955, 'Activity-driven · repeat a known contact', fontsize=15, fontweight='bold')
    where = {0: 'dot size = activity', rounds + 1: 'result'}.get(step, f'round {step} of {rounds} · window {(step + 1)//2}')
    fig.text(.97, .90, where, fontsize=11, color=ORANGE, ha='right', fontweight='bold')
    for k, name in enumerate(('no memory', 'memory')):
        ax = fig.add_axes([.03 + .5*k, .15, .44, .67])
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.45, 1.35); ax.set_aspect('equal'); ax.axis('off')
        rule = 'No memory · random partner' if k == 0 else 'Memory · known: n/(n+1)'
        ax.text(-1.75, 1.3, rule, fontsize=12, fontweight='bold')
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
        if step == rounds + 1:
            a, b = sum(len(w) >= 2 for w in windows.values()), len(windows)
            ax.text(0, -1.35, f'ρ₂ = {a}/{b} = {100*a/b:.0f} %', ha='center', fontsize=13, fontweight='bold')
            (i, j) = max((p for p in windows if len(windows[p]) >= 2), key=lambda p: count[p])
            ax.text(*pos[[i, j]].mean(0), '≥ 2 windows', fontsize=9.5, color=BLUE, ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
    key = ('Blue: ≥ 2 windows · grey: 1 window' if step == rounds + 1 else
           'Orange: active now · grey lines: past contacts')
    fig.text(.03, .09, key, fontsize=10, color='#555555')
    fig.text(.03, .047, 'n = known partners · first contact: always new', fontsize=10, color='#555555')
    return fig


def gif_activity(plt):
    runs = ad_example()
    gif([ad_frame(plt, runs, s) for s in range(12)], [3500] + [2000]*10 + [6500], 'gif_activity')


def draw():
    plt = setup()
    for old in FIGS.glob('*'): old.unlink()
    summaries = pd.read_csv(FINAL/'SUMMARY.csv')
    summary = summaries.query("group == 'real'").set_index(['arm', 'method'])
    perall = pd.read_csv(FINAL/'PER_SOURCE.csv')
    per = perall.query("group == 'real'")
    spread = performance_spread(perall)
    spread.to_csv(DATA/'performance_spread.csv', float_format='%.8g')
    paired_comparisons(perall).to_csv(DATA/'paired_comparisons.csv', index=False, float_format='%.4g')
    pred = pd.read_csv(FINAL/'PREDICTIONS.csv')
    f = pd.read_csv(DATA/'network_features.csv', index_col=0)
    types = pd.read_csv(DATA/'answer_types.csv')
    write_relations(perall, f, pred)
    fig_toy(plt); fig_actives(plt, f); fig_sample(plt, summary, per)
    fig_ranking(plt, summary, spread.loc['real']); fig_levels(plt, pred); fig_amount(plt, pred)
    for group, name, label in [('surrogate', 'fig2_twins_ranking', '12 time-shuffled twins'),
                               ('synthetic', 'fig2_synthetic_ranking', '8 synthetic networks')]:
        fig_ranking(plt, summaries[summaries.group.eq(group)].set_index(['arm','method']),
                    spread.loc[group], name, label)
    fig_textbook(plt, types); fig_textbook_networks(plt, pred, f)
    fig_correction(plt, pred); fig_correction_networks(plt, pred, f)
    fig_stability(plt, pred); fig_noise_by_graph(plt, pred, f); fig_averaging(plt, pred)
    fig_python(plt, summary, pred); fig_python_networks(plt, per, f); fig_python_groups(plt)
    fig_agreement(plt, per, f); fig_method_networks(plt, per, f); fig_structure(plt, perall, f)
    fig_property(plt, perall, 'fig8d_hard_networks', HARD_PROPERTY, True); fig_cards(plt, perall, f); fig_networks(plt, per, f)
    fig_twins(plt, pred); fig_twin_error(plt, perall, f)
    fig_memory(plt, perall); fig_property(plt, perall, 'fig9e_hard_networks', HARD_PROPERTY, False)
    fig_property(plt, perall, 'fig9e_pairs', EFFECTIVE_PAIRS, False); fig_all_networks(plt, perall)
    fig_persistence_levels(plt, pred); fig_windows(plt)
    gif_dar(plt); gif_activity(plt)
    from analysis_tables import write_tables
    write_tables(pred, f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations, api_runs and qwen_runs')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
