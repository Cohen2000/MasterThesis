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
METHODS = {'plugin': 'No correction', 'median': 'Training median', 'mle': 'MLE', 'et': 'ExtraTrees', 'gpt_6_sol': 'GPT',
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


def value_grid(plt, ax, v, rows, cols, vmax, show_rows=True):
    """Table-like heatmap with the value written in every cell."""
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            a = min(1, v[i, j]/vmax)
            ax.add_patch(plt.Rectangle((j-.47, i-.43), .94, .86, color=plt.cm.Purples(.06 + .8*a), lw=0))
            ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=10, color='white' if a > .55 else '#222222')
    ax.set_xlim(-.5, v.shape[1]-.5); ax.set_ylim(v.shape[0]-.5, -.5); unframe(ax)
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows if show_rows else ['']*len(rows))


# Fig. 1: how the sample looks. Signed error of the observed share, per arm and per network.
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
    ax.set_xlabel('observed share − true ρ₂ (pp)')
    ax.text(-44, 3.55, '← sample looks less persistent', fontsize=9.5, color='#555555', va='center')
    ax.text(44, 3.55, 'sample looks more persistent →', fontsize=9.5, color='#555555', va='center', ha='right')
    ax.set_ylim(-.5, 3.8)
    ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Line2D([], [], marker='o', ls='', color='#333333', markersize=4)]
    ax.legend(h, ['mean of 12 networks', 'one network'], frameon=False, loc='lower right', fontsize=9.5)
    save(fig, 'fig1_sample')


# Fig. 2: error per arm; no correction on top, all other methods ranked; dashed line = no correction.
def fig_error_by_arm(plt, summary):
    others = ['median', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking', 'qwen_nonthinking']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    for ax, arm in zip(axes, ARMS):
        ranked = ['plugin'] + sorted(others, key=lambda m: summary.loc[(arm, m), 'MAE_2'])
        v = [100*summary.loc[(arm, m), 'MAE_2'] for m in ranked]
        y = np.arange(len(ranked))[::-1]
        ax.barh(y, v, color=[colour(m) for m in ranked], height=.7)
        ax.axvline(v[0], color='#555555', lw=1, ls=(0, (4, 3)), zorder=3)
        for yi, vi in zip(y, v): ax.text(vi+.8, yi, f'{vi:.1f}', va='center', fontsize=9.5, zorder=4,
                                         bbox=dict(facecolor='white', edgecolor='none', pad=.6))
        ax.set_yticks(y, [METHODS[m] for m in ranked]); ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40])
        ax.set_title(ARMS[arm]); ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True); ax.set_xlabel('error (pp)')
    fig.subplots_adjust(wspace=.75)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, LIGHT, BLUE, ORANGE)] + \
              [plt.Line2D([], [], color='#555555', lw=1, ls=(0, (4, 3)))]
    fig.legend(handles, ['no correction', 'training median (constant guess)', 'classical method', 'language model',
                         'error without correction'], loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, .98))
    save(fig, 'fig2_error_ranked')


# Fig. 3: near which simple estimate do the language-model answers lie?
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


# Fig. 4: stability. Left: the identical sample again. Right: a new sample of the same network.
def fig_stability(plt, resp, train, samp):
    rows = ['plugin', 'mle', 'et'] + LLMS
    labels = ['No correction', 'MLE', 'ExtraTrees'] + [METHODS[m] for m in LLMS]
    same = np.array([[0.]*4 if m in ('plugin', 'mle') else [100*train.loc[a, 'median_observation_SD_rho2'] for a in ARMS]
                     if m == 'et' else [100*resp.loc[(a, m), 'median_observation_SD_rho2'] for a in ARMS] for m in rows])
    new = np.array([[100*samp.loc[(a, m), 'median_graph_SD_rho2_across_draws'] for a in ARMS] for m in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (ax, v, title) in enumerate(zip(axes, (same, new), ('Same sample, asked again', 'New sample of the same network'))):
        value_grid(plt, ax, v, labels, list(ARMS), 22, show_rows=k == 0)
        ax.set_title(title, pad=24)
    fig.text(.5, -.03, 'spread of the ρ₂ estimate (median standard deviation, pp)', ha='center', fontsize=10.5, color='#444444')
    save(fig, 'fig4_stability')


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


# Fig. 5: error per real network, arm and method.
def fig_networks(plt, per, f):
    order = f.sort_values('rho2').index.tolist()
    rows = [f"{NAMES[s]}  ({100*f.loc[s, 'rho2']:.0f} %)" if f.loc[s, 'rho2'] >= .01 else f"{NAMES[s]}  (0.3 %)" for s in order]
    method_heat(plt, per, order, rows, 'fig5_networks', 'network (true ρ₂)')


# Fig. 6: every real network against its time-shuffled copy.
def fig_real_vs_shuffled(plt, perall):
    methods = ['plugin', 'mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']
    per = perall.assign(family=perall.source.str.replace('__pwt', '', regex=False))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    rows = []
    for ax, arm in zip(axes, ARMS):
        y = np.arange(len(methods))[::-1]
        for yi, m in zip(y, methods):
            p = per[(per.arm == arm) & (per.method == m)].pivot(index='family', columns='group', values='MAE_2')*100
            r, s = p.real.mean(), p.surrogate.mean()
            rows.append({'arm': arm, 'method': m, 'real': r, 'shuffled': s, 'real_lower': int((p.real < p.surrogate).sum())})
            c = colour(m)
            ax.annotate('', xy=(s, yi), xytext=(r, yi), arrowprops=dict(arrowstyle='->', color=c, lw=1.8, shrinkA=4, shrinkB=4))
            ax.scatter([r], [yi], s=46, color=c, zorder=3)
            ax.scatter([s], [yi], s=46, facecolor='white', edgecolor=c, lw=1.8, zorder=3)
        ax.set_yticks(y, [METHODS[m] for m in methods]); ax.set_xlim(0, 36); ax.set_title(ARMS[arm])
        ax.grid(axis='x', color='#eeeeee'); ax.set_axisbelow(True); ax.set_xlabel('mean error (pp)')
    h = [plt.Line2D([], [], marker='o', ls='', color='#555555', markersize=7),
         plt.Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=1.8, markersize=7)]
    fig.legend(h, ['real networks', 'time-shuffled copies'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    pd.DataFrame(rows).to_csv(DATA/'real_vs_shuffled.csv', index=False, float_format='%.2f')
    save(fig, 'fig6_real_vs_shuffled')


# Fig. 7: arm B, error against true rho_2 over all 32 networks (real, shuffled, synthetic).
def fig_b_vs_rho(plt, perall):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MAIN):
        e = perall[(perall.arm == 'B') & (perall.method == m)].set_index('source')
        c = colour(m)
        for fam in NAMES:
            ax.plot([100*truth[fam][0], 100*truth[fam+'__pwt'][0]], [100*e.MAE_2[fam], 100*e.MAE_2[fam+'__pwt']],
                    color='#cccccc', lw=1, zorder=1)
        for group, kw in (('real', dict(color=c)), ('surrogate', dict(facecolor='white', edgecolor=c, lw=1.6)),
                          ('synthetic', dict(color=c, marker='^'))):
            g = e[e.group == group]
            ax.scatter([100*truth[s][0] for s in g.index], 100*g.MAE_2, s=40, zorder=3, **kw)
        ax.set_title(METHODS[m]); ax.grid(color='#eeeeee'); ax.set_axisbelow(True); ax.spines['left'].set_visible(True)
        ax.set_xlim(0, 90); ax.set_ylim(0, 55)
    for ax in axes[1]: ax.set_xlabel('true ρ₂ (%)')
    for ax in axes[:, 0]: ax.set_ylabel('error in arm B (pp)')
    h = [plt.Line2D([], [], marker='o', ls='', color='#555555', markersize=7),
         plt.Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=1.6, markersize=7),
         plt.Line2D([], [], marker='^', ls='', color='#555555', markersize=7),
         plt.Line2D([], [], color='#cccccc', lw=1)]
    fig.legend(h, ['real network', 'time-shuffled copy', 'synthetic network', 'same network, before and after shuffling'],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .98))
    save(fig, 'fig7_event_loss_vs_rho')


# Appendix: synthetic networks, all arms and methods.
def fig_synthetic(plt, per):
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    order = [f'{v}_r{i}' for v, _ in SYNTHETIC for i in (1, 2)]
    rows = [f'{label} · {i}  ({100*truth[f"{v}_r{i}"][0]:.0f} %)' for v, label in SYNTHETIC for i in (1, 2)]
    method_heat(plt, per, order, rows, 'figA1_synthetic', 'variant · instance (true ρ₂)', breaks=(1.5, 3.5, 5.5))


SYNTHETIC = (('dar_a0', 'DAR, no memory'), ('dar_a08', 'DAR, memory'),
             ('ad_memoryless', 'Activity-driven, no memory'), ('ad_memory', 'Activity-driven, memory'))


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
    fig_sample(plt, summary, per); fig_error_by_arm(plt, summary); fig_answer_types(plt, types)
    fig_stability(plt, resp, train, samp); fig_networks(plt, per, f)
    fig_real_vs_shuffled(plt, perall[perall.group.isin(['real', 'surrogate'])]); fig_b_vs_rho(plt, perall)
    fig_synthetic(plt, perall.query("group == 'synthetic'"))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', action='store_true', help='rebuild docs/analysis/data (needs data outside the repo)')
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations')
    a = ap.parse_args()
    if a.inputs: build_inputs(a.external)
    draw()
