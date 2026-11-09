#!/usr/bin/env python3
"""Every figure of the thesis, drawn from the frozen data in data/.

  python scripts/figures.py

Writes thesis/figures/*.png. Calls no model and needs no raw data. Figure 1.1 is a drawn toy
example; all other figures plot data/predictions.csv, data/networks.csv and the samples.
Labels are English, as the captions of the thesis.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis as T  # noqa: E402  (data loading and error conventions)
from study.common import ONLINE, ROOT  # noqa: E402
from study.estimators import ratio  # noqa: E402
from study.sample import parse  # noqa: E402

OUT = ROOT/'thesis/figures'
CM = 1/2.54
GRAY = '#6b6a66'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3de'
COLOR = {'observed': '#6b6a66', 'statistical_model': '#2a78d6', 'gpt_6_sol': '#eb6834',
         'learned_reference': '#1baf7a', 'deepseek_v4_1_flash': '#6250d6', 'qwen3_6': '#e87ba4'}
MARKER = {'observed': 'o', 'statistical_model': 's', 'gpt_6_sol': 'o', 'learned_reference': 'D',
          'deepseek_v4_1_flash': '^', 'qwen3_6': 'v'}
LABEL = {'observed': 'Observed persistence', 'statistical_model': 'Statistical model', 'gpt_6_sol': 'GPT-6 Sol',
         'deepseek_v4_1_flash': 'DeepSeek V4.1', 'qwen3_6': 'Qwen3.6', 'learned_reference': 'Learned reference'}
SAMPLER = {'node': 'Node sampling', 'walk': 'Random walk', 'truncation': 'Time truncation', 'event': 'Event sampling'}
SUB = {'node': 'no bias', 'walk': 'sampling bias', 'truncation': 'lost history', 'event': 'both'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .6, 'xtick.major.width': .6,
                     'ytick.major.width': .6, 'legend.frameon': False, 'savefig.dpi': 300, 'axes.titlesize': 8.5,
                     'axes.titleweight': 'bold'})


def de(x, digits=1):
    """A number as printed in the figures (decimal point, minus sign)."""
    return f'{x:.{digits}f}'.replace('-', '−')


COMMA = FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−'))


def clean(ax, grid='x'):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color=GRID, lw=.6); ax.set_axisbelow(True)
    if ax.get_xscale() == 'linear': ax.xaxis.set_major_formatter(COMMA)
    if ax.get_yscale() == 'linear': ax.yaxis.set_major_formatter(COMMA)


def save(fig, name):
    fig.savefig(OUT/name, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5.2: mean error by method and sampler
def figure_overview(pred):
    real = pred[pred.shuffled == 0]
    per = T.per_network(real)
    methods = ['observed', 'statistical_model', 'learned_reference', 'gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6']
    fig, axes = plt.subplots(1, 4, figsize=(16*CM, 6.4*CM), sharey=True)
    for ax, arm in zip(axes, T.SAMPLERS):
        for i, key in enumerate(methods):
            x = per.loc[key, arm]
            m, se = x.mean(), x.std(ddof=1)/np.sqrt(len(x))
            y = len(methods)-1-i
            hollow = key == 'learned_reference'
            ax.errorbar(m, y, xerr=se, fmt=MARKER[key], ms=5, color=COLOR[key], mfc='white' if hollow else COLOR[key],
                        mew=1.2, elinewidth=1, capsize=0)
            ax.text(m+se+1, y, de(m), va='center', fontsize=6.5, color=MUTED)
        ax.set_title(SAMPLER[arm]); ax.set_xlim(0, 40); ax.set_xticks([0, 10, 20, 30, 40])
        clean(ax)
        ax.set_xlabel('Error (pp)')
    axes[0].set_yticks(range(len(methods))); axes[0].set_yticklabels([LABEL[k] for k in reversed(methods)])
    axes[0].tick_params(axis='y', length=0)
    fig.subplots_adjust(wspace=.12)
    save(fig, 'fig5_2_overview.png')


# ---------------------------------------------------------------- Figure 5.1: error per network
def figure_networks(pred, nets):
    names = nets[nets.shuffled == 0].set_index('network')
    d = pred[(pred.shuffled == 0) & pred.valid]
    order = list(names.rho_2.sort_values().index)
    methods = [('observed', 0.0), ('statistical_model', .27), ('gpt_6_sol', -.27)]
    fig, axes = plt.subplots(1, 4, figsize=(16*CM, 9*CM), sharey=True)
    for ax, arm in zip(axes, T.SAMPLERS):
        ax.axhspan(-.5, 3.5, color='#f4f3ef', zorder=0)
        ax.axvline(0, color=INK, lw=.7, zorder=1)
        sub = d[d.sampler == arm]
        for key, off in methods:
            g = sub[sub.method == key].groupby('network').err
            mean, lo, hi = g.mean(), g.min(), g.max()
            for i, s in enumerate(order):
                y = i+off
                if key != 'observed':
                    ax.plot([max(lo[s], -42), min(hi[s], 47)], [y, y], color=COLOR[key], alpha=.45, lw=1, solid_capstyle='round', zorder=2)
                hollow = key == 'observed'
                ax.plot(mean[s], y, MARKER[key], ms=3.6 if key != 'observed' else 3.4, color=COLOR[key],
                        mfc='white' if hollow else COLOR[key], mew=.9, zorder=3)
        ax.set_xlim(-42, 47); ax.set_xticks([-30, 0, 30]); ax.set_xticks([-15, 15], minor=True)
        ax.grid(axis='x', which='minor', color=GRID, lw=.6)
        ax.set_title(f'{SAMPLER[arm]}\n', loc='left')
        ax.text(0, 1.015, SUB[arm], transform=ax.transAxes, fontsize=7, color=MUTED, style='italic', va='bottom')
        clean(ax)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f'{names.name[s]} · {100*names.rho_2[s]:.0f}%' if names.rho_2[s] >= .01
                             else f'{names.name[s]} · {de(100*names.rho_2[s])}%' for s in order])
    axes[0].tick_params(axis='y', length=0)
    fig.text(.5, .0, 'Estimate minus true persistence (pp)', ha='center', fontsize=7.5, color=MUTED)
    handles = [plt.Line2D([], [], marker=MARKER[k], ls='', color=COLOR[k], mfc='white' if k == 'observed' else COLOR[k], ms=4.5)
               for k, _ in methods]+[plt.Line2D([], [], color=MUTED, lw=1, alpha=.6)]
    fig.legend(handles, [LABEL[k] for k, _ in methods]+['Range over samples and answers'], loc='lower center',
               ncol=4, bbox_to_anchor=(.5, -.07), fontsize=7)
    fig.subplots_adjust(wspace=.1, bottom=.13)
    save(fig, 'fig5_1_networks.png')


# ---------------------------------------------------------------- Figure 5.3: random walk, LLM answers vs textbook correction
def figure_textbook(pred, samples):
    design = {i: ratio(parse(x['block']))[0] for i, x in samples.items() if x['arm'] == 'S'}
    walk = pred[(pred.shuffled == 0) & (pred.sampler == 'walk')]
    fig, axes = plt.subplots(1, 3, figsize=(16*CM, 5.8*CM), sharex=True, sharey=True)
    for ax, key in zip(axes, T.LLMS):
        a = walk[(walk.method == key) & walk.valid]
        x, y = 100*a.sample_id.map(design), 100*a.rho_2
        near = (y-x).abs() <= .5
        n_all = len(walk[walk.method == key])
        ax.plot([0, 100], [0, 100], color=GRID, lw=1, zorder=0)
        ax.scatter(x[~near], y[~near], s=12, facecolors='white', edgecolors=COLOR[key], linewidths=.9, zorder=2)
        ax.scatter(x[near], y[near], s=12, color=COLOR[key], edgecolors='white', linewidths=.4, zorder=3)
        ax.set_title(f'{LABEL[key]}\n{int(near.sum())} of {n_all} match')
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.set_aspect('equal')
        clean(ax, grid=None)
        ax.set_xlabel('Textbook correction (%)')
    axes[0].set_ylabel('LLM answer (%)')
    fig.subplots_adjust(wspace=.15)
    save(fig, 'fig5_3_textbook.png')


# ---------------------------------------------------------------- Figure 5.4: spread of three answers to the same sample
def figure_answer_spread(pred):
    real = pred[(pred.shuffled == 0) & pred.valid]
    fig, ax = plt.subplots(figsize=(16*CM, 5.4*CM))
    width = .24
    for j, key in enumerate(T.LLMS):
        g = real[real.method == key]
        var = g.groupby(['network', 'sampler', 'sample']).rho_2.var(ddof=0)
        sd = (np.sqrt(var.groupby(['network', 'sampler']).mean())*100).groupby('sampler').median().reindex(T.SAMPLERS)
        xs = np.arange(4)+(j-1)*width
        ax.bar(xs, sd.values, width=width*.88, color=COLOR[key], label=LABEL[key])
        for xx, v in zip(xs, sd.values): ax.text(xx, v+.4, de(v), ha='center', fontsize=6.5, color=MUTED)
    clean(ax, grid='y')
    ax.set_xticks(range(4)); ax.set_xticklabels([SAMPLER[a] for a in T.SAMPLERS])
    ax.set_ylabel('Answer spread (pp)')
    ax.legend(loc='upper left', fontsize=7)
    save(fig, 'fig5_4_answer_spread.png')


# ---------------------------------------------------------------- Figure 5.5: what makes a network hard
def figure_hard(pred, nets):
    names = nets[nets.shuffled == 0].set_index('network')
    net = pd.read_csv(T.DATA/'network_properties.csv')
    net = net[net.shuffled == 0].set_index('network')
    real = pred[pred.shuffled == 0]
    e = T.per_network(real)
    med = e[e.index.get_level_values('method').isin(T.FIVE)].groupby(['sampler', 'network']).median().unstack('sampler')
    panels = [('walk', 'events_per_pair', 'Interactions per pair (log)', True),
              ('truncation', 'abs_early_late_mismatch_2', 'Error of extrapolating from\nthe three late windows (pp)', False),
              ('event', 'rho2', 'True persistence', False)]
    fig, axes = plt.subplots(1, 3, figsize=(16*CM, 5.8*CM))
    for ax, (arm, prop, xl, log) in zip(axes, panels):
        x, y = net[prop].reindex(med.index), med[arm]
        r = T.spearman(x, y)
        online = x.index.isin(ONLINE)
        ax.scatter(x[~online], y[~online], s=16, color='#2a78d6', edgecolors='white', linewidths=.5, zorder=3, label='Other graphs')
        ax.scatter(x[online], y[online], s=16, color='#eb6834', edgecolors='white', linewidths=.5, zorder=3, label='Online platforms')
        if log:
            ax.set_xscale('log'); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_title(f'{SAMPLER[arm]}\nr = {de(r, 2) if r < 0 else "+"+de(r, 2)} (Spearman)')
        ax.set_xlabel(xl)
        clean(ax, grid='y')
        top = y.sort_values().index[-2:]
        for s in top: ax.annotate(names.name[s], (x[s], y[s]), xytext=(-3, 3), textcoords='offset points', fontsize=6, color=MUTED, ha='right')
    axes[0].set_ylabel('Median error of the five methods (pp)')
    axes[0].legend(loc='upper left', fontsize=6.5, handletextpad=.2)
    fig.subplots_adjust(wspace=.3)
    save(fig, 'fig5_5_hard.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred, nets, samples = T.load()
    figure_overview(pred)
    figure_networks(pred, nets)
    figure_textbook(pred, samples)
    figure_answer_spread(pred)
    figure_hard(pred, nets)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
