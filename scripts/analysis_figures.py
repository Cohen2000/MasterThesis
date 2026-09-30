"""Figures for docs/analysis/ANALYSIS.md.

In plain words: draws the figures of the written analysis from the frozen results in
docs/results/final. Nothing in docs/results/final is changed.

Two steps:
  python scripts/analysis_figures.py --inputs   rebuild docs/analysis/data/*.csv
  python scripts/analysis_figures.py            draw docs/analysis/figures/*.png and *.pdf

The --inputs step needs files that are not in the repository: the raw networks in
data/raw, the frozen API observations and the GPT + Python answers (default location
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
MAIN = ['mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']   # "typical error"
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
    """Network features, S answer types and GPT + Python tool use, as small CSV tables."""
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
                     'observed_pairs_R': float(np.mean([b['D_obs'] for b in blocks['R']])),
                     'kept_share_B': float(np.mean([b['parameter'] for b in blocks['B']]))})
    walk = pd.read_csv(FINAL/'WALK.csv').set_index('graph_id')
    f = pd.DataFrame(rows).set_index('source')
    f['walk_distinct_pairs'] = walk.loc[f.index, 'distinct_dyads_mean']
    f['walk_revisit_share'] = walk.loc[f.index, 'revisit_rate_mean']
    f['type'] = np.where(f.events_per_pair >= 15, 'intense', 'fleeting')
    f.to_csv(DATA/'network_features.csv', float_format='%.6g')

    # Arm S: does an answer equal the simple reweighting, the uncorrected share, or neither (within 0.5 pp)?
    pred = pd.read_csv(FINAL/'PREDICTIONS.csv')
    s = pred[(pred.group == 'real') & (pred.arm == 'S')]
    first = lambda v: json.loads(v)[0]
    design = {i: design_estimate(parse(o['block']))[0] for i, o in obs.items() if o['arm'] == 'S'}
    plug = s[s.method == 'plugin'].set_index('observation_id').prediction.map(first)
    out = []
    for method in LLMS:
        a = s[(s.method == method) & (s.valid == True)]
        v = a.prediction.map(first).to_numpy()
        near_design = np.abs(v - a.observation_id.map(design).to_numpy()) <= .005
        near_plug = np.abs(v - a.observation_id.map(plug).to_numpy()) <= .005
        out.append({'method': method, 'answers': len(a), 'reweighting': near_design.mean(),
                    'uncorrected': (near_plug & ~near_design).mean(), 'other': (~near_plug & ~near_design).mean()})
    r = pred[(pred.group == 'real') & (pred.arm == 'R')]
    rplug = r[r.method == 'plugin'].set_index('observation_id').prediction.map(first)
    for row in out:
        a = r[(r.method == row['method']) & (r.valid == True)]
        row['R_copies_observed_share'] = float(np.mean(np.abs(a.prediction.map(first) - a.observation_id.map(rplug)) <= .005))
    pd.DataFrame(out).to_csv(DATA/'answer_types.csv', index=False, float_format='%.4f')

    # GPT + Python: how often it fits its own model, and how many tool calls it uses (real networks).
    ae = pred[pred.method == 'gpt_6_sol_tools'].set_index(['observation_id', 'repeat_index'])
    tool = []
    for line in (external/'api_runs/openai_tools/responses.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d.get('kind') != 'main': continue
        oid, rep = d['id'].rsplit('__', 2)[0], int(d['id'][-1])
        if (oid, rep) not in ae.index or ae.loc[(oid, rep), 'group'] != 'real': continue
        code = '\n'.join(o.get('code', '') for o in (d.get('raw_response') or {}).get('output', [])
                         if o.get('type') == 'code_interpreter_call')
        tool.append({'arm': obs[oid]['arm'], 'tool_calls': d.get('tool_calls') or 0,
                     'fits_own_model': 'optimize' in code or 'minimize' in code, 'AE2': ae.loc[(oid, rep), 'AE2']})
    t = pd.DataFrame(tool)
    t = t[t.AE2.notna()]
    summary = [{'arm': a, 'answers': len(g), 'fits_own_model': g.fits_own_model.mean(),
                'median_tool_calls': g.tool_calls.median(), 'share_10_or_more_calls': (g.tool_calls >= 10).mean(),
                'error_under_10_calls': g[g.tool_calls < 10].AE2.mean(), 'error_10_or_more_calls': g[g.tool_calls >= 10].AE2.mean()}
               for a, g in t.groupby('arm')]
    pd.DataFrame(summary).to_csv(DATA/'python_tool_use.csv', index=False, float_format='%.4f')


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


def fig_error_by_arm(plt, summary):
    methods = ['plugin', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        v = [100*summary.loc[(arm, m), 'MAE_2'] for m in methods]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, v, color=[colour(m) for m in methods], height=.68)
        for yi, vi in zip(y, v): ax.text(vi+.6, yi, f'{vi:.1f}', va='center', fontsize=9.5)
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlim(0, 36); ax.set_title(ARMS[arm])
        ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True); ax.set_xlabel('error (pp)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, BLUE, ORANGE)]
    fig.legend(handles, ['no correction', 'classical method', 'language model'], loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0))
    save(fig, 'fig1_error_by_arm')


def fig_bias(plt, summary):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    arms = list(ARMS)[::-1]
    y = np.arange(len(arms))
    for k, (m, c) in enumerate((('plugin', GREY), ('mle', BLUE))):
        v = [100*summary.loc[(a, m), 'signed_rho_2'] for a in arms]
        yy = y + (.19 if k == 0 else -.19)
        ax.barh(yy, v, height=.36, color=c, label=METHODS[m])
        for yi, vi in zip(yy, v):
            ax.text(vi + (.7 if vi >= 0 else -.7), yi, f'{vi:+.1f}', va='center', ha='left' if vi >= 0 else 'right', fontsize=9.5)
    ax.axvline(0, color='#333333', lw=1)
    ax.set_yticks(y, [ARMS[a] for a in arms]); ax.set_xlim(-24, 34)
    ax.set_xlabel('← estimate too low          signed error (pp)          estimate too high →')
    ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig2_bias_mle')


def fig_answer_types(plt, types, summary):
    t = types.set_index('method').loc[LLMS[::-1]]
    fig, ax = plt.subplots(figsize=(9.5, 3.1))
    y = np.arange(len(t)); left = np.zeros(len(t))
    for col, c, lab in (('reweighting', BLUE, 'applies the simple reweighting'),
                        ('uncorrected', GREY, 'leaves the bias uncorrected'), ('other', '#e4e2da', 'other value')):
        v = 100*t[col].to_numpy()
        ax.barh(y, v, left=left, color=c, height=.62, label=lab)
        for yi, l, vi in zip(y, left, v):
            if vi >= 7: ax.text(l+vi/2, yi, f'{vi:.0f} %', ha='center', va='center', fontsize=9.5,
                                color='white' if c != '#e4e2da' else '#333333')
        left += v
    for yi, m in zip(y, t.index):
        ax.text(103, yi, f'error {100*summary.loc[("S", m), "MAE_2"]:.1f} pp', va='center', fontsize=9.5)
    ax.set_yticks(y, [METHODS[m] for m in t.index]); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('share of answers (%)'); ax.spines['bottom'].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0))
    save(fig, 'fig3_answer_types_S')


def fig_stability(plt, resp):
    rows = LLMS + ['mle_et']
    labels = [METHODS[m] for m in LLMS] + ['MLE, ExtraTrees']
    v = np.array([[0.]*4 if m == 'mle_et' else
                   [100*resp.loc[(a, m), 'median_observation_SD_rho2'] for a in ARMS] for m in rows])
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.imshow(v, cmap='Purples', vmin=0, vmax=25, aspect='auto')
    for i in range(v.shape[0]):
        for j in range(4):
            ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=10, color='white' if v[i, j] > 13 else '#222222')
    ax.set_xticks(range(4), list(ARMS)); ax.set_yticks(range(len(rows)), labels)
    ax.xaxis.tick_top(); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel('spread of 3 answers to the same sample (median SD, pp)'); ax.xaxis.set_label_position('bottom')
    save(fig, 'fig4_answer_spread')


def fig_python(plt, summary, resp):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    arms = list(ARMS)[::-1]; y = np.arange(4)
    for ax, (title, get) in zip(axes, (('Error (pp)', lambda a, m: 100*summary.loc[(a, m), 'MAE_2']),
                                       ('Spread of 3 answers (pp)', lambda a, m: 100*resp.loc[(a, m), 'median_observation_SD_rho2']))):
        for k, (m, c) in enumerate((('gpt_6_sol', ORANGE), ('gpt_6_sol_tools', VIOLET))):
            v = [get(a, m) for a in arms]; yy = y + (.19 if k == 0 else -.19)
            ax.barh(yy, v, height=.36, color=c, label=METHODS[m])
            for yi, vi in zip(yy, v): ax.text(vi+.25, yi, f'{vi:.1f}', va='center', fontsize=9.5)
        ax.set_title(title); ax.set_xlim(0, 18); ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True)
    axes[0].set_yticks(y, [ARMS[a] for a in arms])
    axes[0].legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.08))
    save(fig, 'fig5_python')


def network_order(f):
    return f.sort_values(['type', 'rho2'], ascending=[True, True]).index.tolist()


def heat(ax, v, rows, show_rows=True, fontsize=10):
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.add_patch(__import__('matplotlib').patches.FancyBboxPatch((j-.46, i-.42), .92, .84, boxstyle='round,pad=0,rounding_size=.08',
                                                                           color=band(v[i, j]), lw=0))
            ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=fontsize,
                    fontweight='bold' if v[i, j] > 10 else 'normal')
    ax.set_xlim(-.5, v.shape[1]-.5); ax.set_ylim(v.shape[0]-.5, -.5)
    ax.set_yticks(range(len(rows)), rows if show_rows else [''] * len(rows))
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.xaxis.tick_top()


def legend_bands(fig, plt, y=-.02):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (EASY, MEDIUM, HARD)]
    fig.legend(handles, ['easy (< 3 pp)', 'medium (3–10 pp)', 'hard (> 10 pp)'], loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, y))


def fig_networks(plt, per, f):
    order = network_order(f)
    typical = per[per.method.isin(MAIN)].groupby(['source', 'arm']).MAE_2.mean().unstack()[list(ARMS)]*100
    v = typical.loc[order].to_numpy(); v = np.c_[v, v.mean(axis=1)]
    rows = [NAMES[s] for s in order]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    heat(ax, v, rows)
    ax.set_xticks(range(5), list(ARMS) + ['mean']); ax.axvline(3.5, color='white', lw=4)
    split = sum(f.loc[order, 'type'] == 'fleeting') - .5
    ax.axhline(split, color='#333333', lw=1.2)
    for i, s in enumerate(order):
        ax.text(4.75, i, f"{f.loc[s, 'type']:<9} {100*f.loc[s, 'rho2']:5.1f} %  {f.loc[s, 'events_per_pair']:7.1f}",
                va='center', fontsize=9.5, family='DejaVu Sans Mono', color='#444444')
    ax.text(4.75, -.75, 'type       true ρ₂  events\n                    per pair', fontsize=9, family='DejaVu Sans Mono',
            color='#777777', va='bottom')
    legend_bands(fig, plt, -.03)
    save(fig, 'fig6_networks')


def fig_networks_by_method(plt, per, f):
    order = network_order(f); rows = [NAMES[s] for s in order]
    fig, axes = plt.subplots(1, len(MAIN), figsize=(15, 5.4))
    for k, (ax, m) in enumerate(zip(axes, MAIN)):
        v = per[per.method == m].pivot(index='source', columns='arm', values='MAE_2').loc[order, list(ARMS)].to_numpy()*100
        heat(ax, v, rows, show_rows=k == 0, fontsize=8.5)
        ax.set_xticks(range(4), list(ARMS)); ax.set_title(METHODS[m], pad=22)
        ax.axhline(sum(f.loc[order, 'type'] == 'fleeting') - .5, color='#333333', lw=1.2)
    legend_bands(fig, plt, -.04)
    save(fig, 'figA1_networks_by_method')


def fig_causes(plt, per, f):
    typical = per[per.method.isin(MAIN)].groupby(['source', 'arm']).MAE_2.mean().unstack()*100
    panels = (('R', 'observed_pairs_R', 'pairs in the sample', 'R · fewer pairs seen'),
              ('S', 'walk_distinct_pairs', 'distinct pairs the walk sees', 'S · walk sees few distinct pairs'),
              ('B', 'kept_share_B', 'share of events kept', 'B · stronger thinning'))
    labels = {'sp_malawi', 'sp_hospital', 'nr_digg_reply', 'reality_mining'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (arm, col, xlab, title) in zip(axes, panels):
        x, y = f[col], typical.loc[f.index, arm]
        ax.scatter(x, y, s=46, color=ORANGE, edgecolor='white', lw=.8, zorder=3)
        for s in f.index:
            if s in labels: ax.annotate(NAMES[s].split(' (')[0], (x[s], y[s]), xytext=(6, 4), textcoords='offset points', fontsize=9)
        ax.set_xscale('log'); ax.set_xlabel(xlab + ' (log scale)'); ax.set_title(title)
        ax.set_ylim(0, max(10, y.max()*1.2)); ax.grid(color='#eeeeee'); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(True)
    axes[0].set_ylabel('typical error (pp)')
    save(fig, 'fig7_causes')


def draw():
    plt = setup()
    summary = pd.read_csv(FINAL/'SUMMARY.csv').query("group == 'real'").set_index(['arm', 'method'])
    per = pd.read_csv(FINAL/'PER_SOURCE.csv').query("group == 'real'")
    resp = pd.read_csv(FINAL/'VARIABILITY_RESPONSE.csv').query("group == 'real'").set_index(['arm', 'method'])
    f = pd.read_csv(DATA/'network_features.csv', index_col=0)
    types = pd.read_csv(DATA/'answer_types.csv')
    fig_error_by_arm(plt, summary); fig_bias(plt, summary); fig_answer_types(plt, types, summary)
    fig_stability(plt, resp); fig_python(plt, summary, resp); fig_networks(plt, per, f)
    fig_networks_by_method(plt, per, f); fig_causes(plt, per, f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations and api_runs')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
