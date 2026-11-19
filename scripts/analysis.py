#!/usr/bin/env python3
"""The tables and results of the thesis, computed from the frozen data in data/.

  python scripts/analysis.py

Writes the tables of the thesis as thesis/tables/*.csv and prints the further results the text
reports (Chapters 3 to 5). It calls no model and needs no raw data: everything comes from data/
(see README.md for how data/ was made). The figures are drawn by scripts/figures.py.

Conventions: the error is estimate minus true persistence rho_2, in percentage points (pp). A mean
error is averaged over the samples and answers of a network first and then over networks, so every
network counts equally. Invalid answers (2 of 4,536) are left out.
"""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import BUDGET, DATA, ONLINE, ROOT, TRAIN, TRAIN_DRAWS  # noqa: E402
from study.estimators import mle_counts, ratio  # noqa: E402
from study.sample import parse  # noqa: E402

OUT = ROOT/'thesis'
SAMPLERS = ['node', 'walk', 'truncation', 'event']
MAIN = ['observed', 'statistical_model', 'gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6', 'learned_reference']
FIVE = ['statistical_model', 'learned_reference', 'gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6']   # "the five methods"
LLMS = ['gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6']
NAME_EN = {'observed': 'Observed persistence', 'statistical_model': 'Statistical model', 'gpt_6_sol': 'GPT-6 Sol',
           'deepseek_v4_1_flash': 'DeepSeek V4.1', 'qwen3_6': 'Qwen3.6', 'learned_reference': 'Learned reference',
           'gpt_6_sol_python': 'GPT-6 Sol with Python', 'gpt_6_sol_no_walk_log': 'GPT-6 Sol without walk log'}
SAMPLER_EN = {'node': 'Node sampling', 'walk': 'Random walk', 'truncation': 'Time truncation', 'event': 'Event sampling'}
KIND_EN = {'face-to-face': 'Contact sensor', 'Bluetooth': 'Bluetooth', 'email': 'Email'}
ONLINE_KIND_EN = {'lkml_reply': 'Online, replies', 'snap_collegemsg': 'Online, messages',
                  'snap_mathoverflow': 'Online, replies', 'nr_digg_reply': 'Online, replies'}


# ---------------------------------------------------------------- formatting and output
def en(x, digits=1, sign=False):
    """A number as printed in the thesis (decimal point, minus sign, optional plus)."""
    s = f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'
    if s in ('-0.0', '+-0.0'): s = '0.0'
    return s.replace('-', '−')


def thousands_en(n):
    return f'{int(n):,}'


def section(title):
    print(f'\n{title}')


def report(claim, value):
    print(f'  {claim}: {value}')


# ---------------------------------------------------------------- data
def load():
    pred = pd.read_csv(DATA/'predictions.csv')
    truth = json.loads((DATA/'truth.json').read_text())
    key = pred.network+np.where(pred.shuffled == 1, '__pwt', '')
    for k in range(2, 6):
        pred[f'truth_{k}'] = [truth[x][k-2] for x in key]
        pred[f'err_{k}'] = 100*(pred[f'rho_{k}']-pred[f'truth_{k}'])
    pred['err'], pred['abs'] = pred.err_2, pred.err_2.abs()
    nets = pd.read_csv(DATA/'networks.csv')
    samples = {p.stem: json.loads(p.read_text()) for p in sorted((DATA/'samples').glob('*.json'))}
    return pred, nets, samples


def per_network(pred, value='abs'):
    """Mean of `value` per method, sampler and network (valid answers only)."""
    return pred[pred.valid].groupby(['method', 'sampler', 'network'])[value].mean()


def mae(pred, networks=None, value='abs'):
    """Mean error per method and sampler, every network counting equally."""
    per = per_network(pred, value)
    if networks is not None: per = per[per.index.get_level_values('network').isin(networks)]
    return per.groupby(['method', 'sampler']).mean()


def spread(pred, networks=None):
    """Per network the SD of all rho_2 estimates of a method (samples x answers), median over networks."""
    sd = pred[pred.valid].groupby(['method', 'sampler', 'network']).rho_2.std(ddof=0)*100
    if networks is not None: sd = sd[sd.index.get_level_values('network').isin(networks)]
    return sd.groupby(['method', 'sampler']).median()


def answer_spread(pred, key):
    """Per network the SD of the three answers to one sample (root of the mean variance), median over networks."""
    g = pred[(pred.method == key) & pred.valid]
    var = g.groupby(['network', 'sampler', 'sample']).rho_2.var(ddof=0)
    return (np.sqrt(var.groupby(['network', 'sampler']).mean())*100).groupby('sampler').median()


def mean_of_three(pred, key, value='abs'):
    """Error when the three answers to a sample are averaged first."""
    g = pred[(pred.method == key) & pred.valid]
    m3 = g.groupby(['network', 'sampler', 'sample']).agg(est=('rho_2', 'mean'), truth=('truth_2', 'first'))
    err = 100*(m3.est-m3.truth)
    if value == 'abs': err = err.abs()
    return err.groupby(['sampler', 'network']).mean().groupby('sampler').mean()


def spearman(x, y):
    return pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float)), method='spearman')


def textbook(samples):
    """The textbook correction of every random-walk sample (rho_2)."""
    return {i: ratio(parse(x['block']))[0] for i, x in samples.items() if x['arm'] == 'S'}


def matches(walk, key, design):
    a = walk[walk.method == key]
    near = a.valid & ((a.rho_2-a.sample_id.map(design)).abs() <= .005)
    return int(near.sum()), len(a)


# ---------------------------------------------------------------- tables (thesis/tables/*.csv)
def write_table(name, header, rows):
    """A table as printed in the thesis. A row ['## text'] is a group header across all columns."""
    path = OUT/'tables'/f'{name}.csv'
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows: w.writerow(r)


def window_text(days):
    if days < 1: return f'{24*days:.0f} hours'
    if days < 60: return f'{en(days)} days'
    return f'{days/30.44:.0f} months'


def table_networks(nets):
    r = nets[nets.shuffled == 0].copy()
    rows = []
    for online, title in ((0, '## Other graphs'), (1, '## Online platforms')):
        rows.append([title])
        for _, x in r[r.online == online].sort_values('rho_2', ascending=False).iterrows():
            kind = ONLINE_KIND_EN[x.network] if online else KIND_EN[x.kind]
            rho = f'{en(100*x.rho_2)}%'
            rows.append([x['name'], kind, thousands_en(x.nodes), thousands_en(x.pairs), en(x.interactions_per_pair),
                         window_text(x.window_days), f'{100*x.share_one_interaction:.0f}%', rho])
    write_table('networks', ['Graph', 'Kind', 'Nodes', 'Pairs', 'Events per pair', 'Window', 'One-off pairs',
                             'Persistence'], rows)


def block(m, s, keys, bold_from):
    """Rows 'error (spread)' with the smallest error per column in bold (learned reference excluded)."""
    best = {a: min(round(m[k, a], 1) for k in bold_from) for a in SAMPLERS}
    rows = []
    for k in keys:
        cells = []
        for a in SAMPLERS:
            text = en(m[k, a])+(f' ({en(s[k, a])})' if s is not None else '')
            if k == 'learned_reference': text = f'*{text}*'
            elif abs(round(m[k, a], 1)-best[a]) < 1e-9:
                text = f'**{en(m[k, a])}**'+(f' ({en(s[k, a])})' if s is not None else '')
            cells.append(text)
        rows.append([f'*{NAME_EN[k]}*' if k == 'learned_reference' else NAME_EN[k]]+cells)
    return rows


def table_results(real):
    m, s = mae(real), spread(real)
    mo, so = mae(real, ONLINE), spread(real, ONLINE)
    no_train = [k for k in MAIN if k != 'learned_reference']
    rows = block(m, s, MAIN, no_train)
    write_table('results', ['Method']+[SAMPLER_EN[a] for a in SAMPLERS], rows)
    return m, s, mo, so


def table_walk(real, design):
    walk = real[real.sampler == 'walk']
    m, sg = mae(real), mae(real, value='err')
    tb = walk[walk.method == 'observed'].assign(est=lambda d: d.sample_id.map(design))
    tb_abs = (100*(tb.est-tb.truth_2)).abs().groupby(tb.network).mean().mean()
    tb_sign = (100*(tb.est-tb.truth_2)).groupby(tb.network).mean().mean()
    rows = [[NAME_EN['observed'], en(m['observed', 'walk']), en(sg['observed', 'walk'], sign=True), '–'],
            [NAME_EN['statistical_model'], en(m['statistical_model', 'walk']), en(sg['statistical_model', 'walk'], sign=True), '–']]
    for k in ['gpt_6_sol', 'gpt_6_sol_python', 'gpt_6_sol_no_walk_log', 'deepseek_v4_1_flash', 'qwen3_6']:
        n, total = matches(walk, k, design)
        rows.append([NAME_EN[k], en(m[k, 'walk']), en(sg[k, 'walk'], sign=True), f'{n} of {total}'])
    rows.append([f"*{NAME_EN['learned_reference']}*", f"*{en(m['learned_reference', 'walk'])}*",
                 f"*{en(sg['learned_reference', 'walk'], sign=True)}*", '–'])
    write_table('walk', ['Method', 'Error (pp)', 'Signed error (pp)', 'Matches textbook correction'], rows)
    return tb_abs


def table_history(real):
    m, sg = mae(real), mae(real, value='err')
    rows = []
    for k in ['observed', 'statistical_model', 'gpt_6_sol', 'gpt3', 'gpt_6_sol_python', 'deepseek_v4_1_flash', 'ds3',
              'qwen3_6', 'qw3', 'learned_reference']:
        if k in ('gpt3', 'ds3', 'qw3'):
            base = {'gpt3': 'gpt_6_sol', 'ds3': 'deepseek_v4_1_flash', 'qw3': 'qwen3_6'}[k]
            a, s = mean_of_three(real, base), mean_of_three(real, base, 'err')
            label = f'{NAME_EN[base]}, mean of three answers'
            cells = [en(a['truncation']), en(s['truncation'], sign=True), en(a['event']), en(s['event'], sign=True)]
        else:
            label = NAME_EN[k]
            cells = [en(m[k, 'truncation']), en(sg[k, 'truncation'], sign=True), en(m[k, 'event']), en(sg[k, 'event'], sign=True)]
        if k == 'learned_reference':
            label, cells = f'*{label}*', [f'*{c}*' for c in cells]
        rows.append([label]+cells)
    write_table('history', ['Method', 'Time truncation: error', 'Time truncation: signed', 'Event sampling: error',
                            'Event sampling: signed'], rows)


def table_windows():
    ws = pd.read_csv(DATA/'window_sensitivity.csv')
    ws = ws[ws.shuffled == 0]
    rows = [[str(int(x.W)), f'{en(100*x.mean_rho_2)}%', en(100*x.mean_absolute_change_vs_W5), en(x.spearman_vs_W5, 2)]
            for _, x in ws.iterrows()]
    write_table('windows', ['Windows', 'Mean persistence', 'Change (pp)', 'Rank correlation'], rows)
    return ws


def table_thresholds(real):
    rows = []
    for k in MAIN:
        cells = []
        for a in SAMPLERS:
            sub = real[(real.sampler == a) & (real.method == k) & real.valid]
            vals = [sub.assign(x=sub[f'err_{j}'].abs()).groupby('network').x.mean().mean() for j in (3, 4, 5)]
            cells.append(' / '.join(en(v) for v in vals))
        name = f'*{NAME_EN[k]}*' if k == 'learned_reference' else NAME_EN[k]
        rows.append([name]+([f'*{c}*' for c in cells] if k == 'learned_reference' else cells))
    write_table('thresholds', ['Method']+[f'{SAMPLER_EN[a]}: ρ₃ / ρ₄ / ρ₅' for a in SAMPLERS], rows)


def table_copies(shuf):
    ms = mae(shuf)
    no_train = [k for k in MAIN if k != 'learned_reference']
    write_table('copies', ['Method']+[SAMPLER_EN[a] for a in SAMPLERS], block(ms, None, MAIN, no_train))
    return ms


def table_per_network(real, nets):
    names = nets[nets.shuffled == 0].set_index('network')
    per = per_network(real)
    order = list(names.rho_2.sort_values(ascending=False).index)
    rows = []
    for a in SAMPLERS:
        rows.append([f'## {SAMPLER_EN[a]}'])
        for n in order:
            rows.append([names.name[n]]+[en(per[k, a, n]) for k in MAIN])
    write_table('per_network', ['Graph', 'Observed', 'Model', 'GPT-6 Sol', 'DeepSeek', 'Qwen', 'Learned'], rows)


# ---------------------------------------------------------------- tables and results
def main():
    (OUT/'tables').mkdir(parents=True, exist_ok=True)
    pred, nets, samples = load()
    real, shuf = pred[pred.shuffled == 0], pred[pred.shuffled == 1]
    nr = nets[nets.shuffled == 0].set_index('network')
    other = [n for n in nr.index if n not in ONLINE]
    design = textbook(samples)
    walk = real[real.sampler == 'walk']

    table_networks(nets)
    m, s, mo, so = table_results(real)
    tb_mae = table_walk(real, design)
    table_history(real)
    ws = table_windows()
    table_thresholds(real)
    ms = table_copies(shuf)
    table_per_network(real, nets)
    sg, sg_s = mae(real, value='err'), mae(shuf, value='err')
    obs = m.loc['observed']
    gpt_n, total = matches(walk, 'gpt_6_sol', design)

    section('Main result')
    report('GPT-6 Sol answers that match the textbook correction (random walk)', f'{gpt_n} of {total}')
    report('random walk: error of the observed persistence and of GPT-6 Sol (pp)', f'{en(obs["walk"])} and {en(m["gpt_6_sol", "walk"])}')
    others = min(m[k, a] for k in MAIN for a in ('walk', 'truncation', 'event'))
    report('node sampling, observed persistence vs. best method under any other sampler (pp)', f'{en(obs["node"])} vs. {en(others)}')

    section('The graphs')
    one = nr.share_one_interaction
    report('pairs that interact only once: online platforms', f'{100*one[list(ONLINE)].min():.0f} to {100*one[list(ONLINE)].max():.0f}%')
    report('pairs that interact only once: other networks', f'{100*one[other].min():.0f} to {100*one[other].max():.0f}%')
    report('persistence of the online platforms', f'{en(100*nr.rho_2[list(ONLINE)].min())}% to {100*nr.rho_2[list(ONLINE)].max():.0f}%')
    report('persistence of the other networks', f'{100*nr.rho_2[other].min():.0f}% and {100*nr.rho_2[other].max():.0f}%')
    hours, months = 24*nr.window_days, nr.window_days/30.44
    report('shortest window: hospital (hours); longest: Linux mailing list (months)', f'{hours.min():.0f} hours, {months.max():.0f} months')
    w = ws[ws.W.between(3, 20) & (ws.W != 5)]
    report('with 3 to 20 windows: largest mean change of persistence (pp)', en(100*w.mean_absolute_change_vs_W5.max()))
    report('with 3 to 20 windows: smallest rank correlation with W = 5', en(w.spearman_vs_W5.min(), 3))
    section('The samples')
    smp = pd.read_csv(DATA/'sample_properties.csv')
    smp = smp[smp.shuffled == 0].assign(share=lambda d: d.M_obs/d.network.map(nr.interactions))
    seen = 100*smp.groupby(['sampler', 'network']).share.mean().groupby('sampler').median()
    report("share of the pairs' active windows each sampler sees", f'{100*BUDGET:.0f}%')
    report('share of all events seen, median over networks: walk, event sampling', f'{seen["S"]:.0f}%, about {seen["B"]:.0f}%')
    n_samples = real.drop_duplicates('sample_id').shape[0]
    report('samples of the twelve networks', str(n_samples))
    report('mean persistence: networks, time-shuffled copies', f'{100*nr.rho_2.mean():.0f}%, {100*nets[nets.shuffled == 1].rho_2.mean():.0f}%')
    report('samples in total', str(pred.drop_duplicates('sample_id').shape[0]))
    # the hospital sample under event sampling shown as the example prompt (Figure 3.2, Appendix A)
    hospital = [x for x in samples.values() if x['graph_id'] == 'sp_hospital' and x['arm'] == 'B'
                and x['sample_index'] == 1]
    o = parse(hospital[0]['block'])
    persistent = sum(r[1] for r in o['table'] if r[0].count('1') >= 2)
    one_window = sum(r[1] for r in o['table'] if r[0].count('1') == 1)
    rows = {r[0]: (r[1], r[2]) for r in o['table']}
    report('hospital sample: p, pairs, events, per window', f"{o['parameter']} (about {100*o['parameter']:.1f}%), {o['D_obs']}, {thousands_en(o['M_obs'])}, {','.join(str(x) for x in o['Events_per_window'])}")
    report('row 01000 (pairs,events)', f"{rows['01000'][0]},{rows['01000'][1]}")
    report('persistent pairs, observed persistence, truth', f"{persistent} of {o['D_obs']}, {100*persistent/o['D_obs']:.0f}%, {100*nr.rho_2['sp_hospital']:.0f}%")
    report('pairs with only one visible window', f"{one_window} of {o['D_obs']}")
    section('The methods')
    report('estimates in the study, invalid answers', f'{thousands_en(len(pred))}, {int((~pred.valid).sum())}')
    report('samplers of the invalid answers', 'both under the random walk' if set(pred[~pred.valid].sampler) == {'walk'} else 'no')
    from study.common import REAL  # noqa: E402
    report('real training networks, of them test networks', f"{len(TRAIN)}, {9 if len(set(TRAIN) & set(REAL)) == 9 else len(set(TRAIN) & set(REAL))}")
    report('training samples per sampler of the learned reference', thousands_en((len(TRAIN)+400)*TRAIN_DRAWS))
    se = per_network(real).groupby(['method', 'sampler']).std(ddof=1)/math.sqrt(12)
    se = se[se.index.get_level_values('method').isin(MAIN)]
    report('largest standard error of a mean error over networks (pp)', en(se.max()))

    section('How the samplers distort persistence')
    report('mean signed error of the observed persistence: node, walk, truncation, event (pp)', ', '.join(en(sg['observed', a], sign=True) for a in SAMPLERS))
    walk_obs = per_network(real, 'err')['observed', 'walk']
    report('random walk: networks where the observed persistence is too high by more than 5 pp', f'{(walk_obs > 5).sum()} of 12')
    eu = real[(real.network == 'snap_email_eu') & (real.sampler == 'walk') & (real.method == 'observed')]
    report('a crawl: Email EU looks this persistent (truth)', f'{100*eu.rho_2.mean():.0f}% ({100*nr.rho_2["snap_email_eu"]:.0f}%)')
    report('time truncation: persistent pairs that lose their return; pairs hidden entirely (mean)', f'{100*nr.truncation_persistent_pairs_losing_return.mean():.0f}%, {100*nr.truncation_pairs_hidden.mean():.0f}%')
    report('event sampling: kept pairs persistent vs. overall vs. seen as persistent (mean)', f'{100*nr.event_rho_2_of_pairs_still_seen.mean():.0f}%, {100*nr.rho_2.mean():.0f}%, {100*nr.event_rho_2_seen.mean():.0f}%')
    section('Accuracy overview')
    report('best method without training: walk, truncation, event', ', '.join(NAME_EN[k] for k in
                    (min(MAIN[:-1], key=lambda k: m[k, a]) for a in ('walk', 'truncation', 'event'))))
    cheap = all(m[k, a] > obs[a] for k in ('deepseek_v4_1_flash', 'qwen3_6') for a in ('truncation', 'event'))
    report('cheaper LLMs worse than the observed persistence under truncation and event sampling', 'yes' if cheap else 'no')
    cheaper = all(m[k, a] > m['gpt_6_sol', a] for k in ('deepseek_v4_1_flash', 'qwen3_6') for a in ('walk', 'truncation', 'event'))
    report('cheaper LLMs err more than GPT-6 Sol whenever a correction is needed', 'yes' if cheaper else 'no')
    no_llm = max(s[k, a] for k in ('observed', 'statistical_model', 'learned_reference') for a in ('truncation', 'event'))
    report('largest spread of the methods without LLM under truncation and event sampling (pp)', en(no_llm))
    report('spread of GPT-6 Sol: truncation, event (pp)', f'{en(s["gpt_6_sol", "truncation"])} and {en(s["gpt_6_sol", "event"])}')
    rng_ = [s[k, a] for k in ('deepseek_v4_1_flash', 'qwen3_6') for a in ('truncation', 'event')]
    report('spread of DeepSeek V4.1 and Qwen3.6: truncation, event (pp)', f'{min(rng_):.0f} to {max(rng_):.0f}')
    section('Correcting sampling bias')
    ref = real[real.method == 'observed'].set_index('sample_id').rho_2
    llm_node = real[(real.sampler == 'node') & real.method.isin(LLMS) & real.valid]
    same = (llm_node.rho_2-llm_node.sample_id.map(ref)).abs() <= 5e-5
    report('node sampling: LLM answers equal to the observed persistence', f'{same.sum()} of {len(llm_node)}')
    report('node sampling: statistical model vs. observed persistence (pp)', f'{en(m["statistical_model", "node"])} vs. {en(obs["node"])}')
    report('random walk: textbook correction itself, statistical model (pp)', f'{en(tb_mae)}, {en(m["statistical_model", "walk"])}')
    for k in ('deepseek_v4_1_flash', 'qwen3_6', 'gpt_6_sol_python'):
        n, t = matches(walk, k, design)
        report(f'{NAME_EN[k]}: answers that match the textbook correction (an invalid answer counts as not)', f'{n} of {t}')
    qw = walk[(walk.method == 'qwen3_6') & walk.valid]
    unc = ((qw.rho_2-qw.sample_id.map(ref)).abs() <= .005) & ((qw.rho_2-qw.sample_id.map(design)).abs() > .005)
    report('Qwen3.6 answers equal to the uncorrected observed persistence (random walk)', str(int(unc.sum())))
    g = walk[(walk.method == 'gpt_6_sol') & walk.valid].assign(tb=lambda d: d.sample_id.map(design))
    off = g[(g.rho_2-g.tb).abs() > .005]
    closer = int(((off.rho_2-off.truth_2).abs() < (off.tb-off.truth_2).abs()).sum())
    report('GPT-6 Sol answers that deviate from the textbook correction, and of them closer to the truth', f'{closer} of {len(off)}')
    village = off[off.tb > .999]
    report('Village: textbook correction, GPT-6 Sol answers, truth', f'{100*village.tb.max():.0f}%, {100*village.rho_2.min():.0f} to {100*village.rho_2.max():.0f}%, {100*nr.rho_2["sp_malawi"]:.0f}%')
    report('GPT-6 Sol with Python, random walk (pp)', en(m['gpt_6_sol_python', 'walk']))
    nolog = m['gpt_6_sol_no_walk_log', 'walk']
    report('random walk without the walk log: GPT-6 Sol (pp)', en(nolog))
    gain = (obs['walk']-nolog)/(obs['walk']-m['gpt_6_sol', 'walk'])
    report('share of the gain that needs only the description of the walk', f'{100*gain:.0f}%')
    # the statistical model without the walk's log: unweighted, and with weights a person built from the table alone
    # (each pattern weighted by dyads^2 / events, the inverse of its mean events per pair, times its dyads)
    rows = []
    for i, x in samples.items():
        if x['arm'] != 'S' or x['graph_id'] not in nr.index or '__pwt' in i: continue
        tab = parse(x['block'])['table']
        none, table = [0.]*6, [0.]*6
        for r in tab:
            none[r[0].count('1')] += r[1]
            table[r[0].count('1')] += r[1]**2/r[2] if r[2] else 0.
        rows.append((x['graph_id'], mle_counts(none, 5)[0], mle_counts(table, 5)[0]))
    nolog_model = pd.DataFrame(rows, columns=['network', 'none', 'table']).assign(truth=lambda d: d.network.map(nr.rho_2))
    nm = {k: (100*(nolog_model[k]-nolog_model.truth)).abs().groupby(nolog_model.network).mean().mean() for k in ('none', 'table')}
    report('random walk without the log: statistical model unweighted, with weights from the table alone (pp)', f"{en(nm['none'])}, {en(nm['table'])}")
    sw = shuf[shuf.sampler == 'walk']
    report('time-shuffled copies: GPT-6 Sol answers that match the textbook correction', '%d of %d' % matches(sw, 'gpt_6_sol', design))
    others_s = min(ms[k, a] for k in MAIN for a in ('walk', 'truncation', 'event'))
    report('time-shuffled copies: node sampling vs. best method under any other sampler (pp)', f'{en(ms["observed", "node"])} vs. {en(others_s)}')
    section('Inferring lost history')
    report('event sampling: statistical model (pp)', en(m['statistical_model', 'event']))
    report('time truncation: statistical model signed error (pp)', en(sg['statistical_model', 'truncation']))
    report('time truncation: statistical model vs. observed persistence (pp)', f'{en(m["statistical_model", "truncation"])} vs. {en(obs["truncation"])}')
    report('time-shuffled copies, time truncation: statistical model vs. observed persistence (pp)', f'{en(ms["statistical_model", "truncation"])} vs. {en(ms["observed", "truncation"])}')
    untrained = MAIN[:-1]
    least = min(untrained, key=lambda k: m[k, 'truncation'])
    report('time truncation: least error without training data', f'{NAME_EN[least]}, {en(m[least, "truncation"])}')
    report('event sampling: GPT-6 Sol (pp)', en(m['gpt_6_sol', 'event']))
    pn = per_network(real)
    wins = {a: int((pn['gpt_6_sol', a] < pn['statistical_model', a]).sum()) for a in SAMPLERS}
    report('networks where GPT-6 Sol errs less than the model: truncation, event, walk', f'{wins["truncation"]}, {wins["event"]}, {wins["walk"]}')
    asd = {k: answer_spread(real, k) for k in LLMS+['gpt_6_sol_python']}
    report('answer spread of GPT-6 Sol: event sampling, walk (pp)', f'{en(asd["gpt_6_sol"]["event"])}, {en(asd["gpt_6_sol"]["walk"])}')
    g6 = asd['gpt_6_sol']
    report('answer spread of GPT-6 Sol: zero under node sampling, growing with what the sampler hides', 'yes' if g6['node'] < .05 and g6['node'] < g6['walk'] < g6['truncation'] < g6['event'] else 'no')
    ratio_ = [asd[k]['event']/asd['gpt_6_sol']['event'] for k in ('deepseek_v4_1_flash', 'qwen3_6')]
    report('event sampling: answer spread of DeepSeek and Qwen relative to GPT-6 Sol', f'{min(ratio_):.1f}- to {max(ratio_):.1f}-fold')
    m3 = mean_of_three(real, 'gpt_6_sol')
    report('GPT-6 Sol, mean of its three answers: event, truncation (pp)', f'{en(m3["event"])}, {en(m3["truncation"])}')
    report('with Python: truncation, event (pp)', f'{en(m["gpt_6_sol_python", "truncation"])}, {en(m["gpt_6_sol_python", "event"])}')
    section('Online platforms')
    lk = per_network(real, 'err').loc['statistical_model', :, 'lkml_reply']
    report('statistical model overshoots on the Linux mailing list (pp, sampler)', f'{lk.max():.0f} ({SAMPLER_EN[lk.idxmax()]})')
    lower = all(mo['gpt_6_sol', a] < mo['statistical_model', a] for a in SAMPLERS)
    on = list(ONLINE)
    combos = sum(int((pn['gpt_6_sol', a][on] < pn['statistical_model', a][on]).sum()) for a in SAMPLERS)
    report('online: platform-sampler combinations where GPT-6 Sol errs less than the model', f'{combos} of 16')
    diff = sorted(pn['gpt_6_sol', a][n]-pn['statistical_model', a][n] for a in SAMPLERS for n in on
                  if not pn['gpt_6_sol', a][n] < pn['statistical_model', a][n])
    gaps = {k: max(mo[k, a]-min(mo[x, a] for x in MAIN) for a in SAMPLERS) for k in MAIN}
    arg = {k: max(SAMPLERS, key=lambda a: mo[k, a]-min(mo[x, a] for x in MAIN)) for k in MAIN}
    report('online: largest gap of GPT-6 Sol to the best method under any sampler (pp)', en(gaps['gpt_6_sol']))
    report('online: every method other than GPT-6 Sol misses the best by at least 4 pp under some sampler', 'yes' if all(gaps[k] >= 4 for k in MAIN if k != 'gpt_6_sol') else 'no')
    section('What makes a graph hard')
    net = pd.read_csv(DATA/'network_properties.csv').set_index(['shuffled', 'network'])
    smp2 = pd.read_csv(DATA/'sample_properties.csv')
    smp2['sampler'] = smp2.sampler.map({'R': 'node', 'S': 'walk', 'H': 'truncation', 'B': 'event'})
    smp2 = smp2.drop(columns='sample_id').groupby(['shuffled', 'network', 'sampler']).mean()
    props = set()
    med = {}
    for sh in (0, 1):
        e = per_network(pred[pred.shuffled == sh])
        med[sh] = e[e.index.get_level_values('method').isin(FIVE)].groupby(['sampler', 'network']).median().unstack('sampler')
        props |= set(net.columns)
        for col in smp2.columns:
            for a in SAMPLERS:
                x = smp2.loc[sh].xs(a, level='sampler')[col].reindex(med[sh].index)
                if x.notna().sum() >= 12 and x.nunique() >= 3: props.add(col)
    props |= {'observed_bias_signed', 'observed_bias_abs'}
    report('properties tested', str(len(props)))
    for a, prop, text in (('walk', 'events_per_pair', '+0,77'), ('truncation', 'abs_early_late_mismatch_2', '+0,78'),
                          ('event', 'rho2', '+0,85')):
        x, y = net.loc[0][prop].reindex(med[0].index), med[0][a]
        r = spearman(x, y)
        loo = min(abs(spearman(x.drop(n), y.drop(n))) for n in x.index)
        report(f'{SAMPLER_EN[a]}: Spearman of the median error with {prop}', en(r, 2, sign=True))
        report(f'{SAMPLER_EN[a]}: smallest |r| when one network is left out', en(loo, 2))
    cm = pn.loc['statistical_model', :, 'snap_collegemsg']
    report('CollegeMsg, statistical model: node, walk, event below 1 pp; truncation (pp)', f'{en(max(cm[a] for a in ("node", "walk", "event")))}; {en(cm["truncation"])}')
    section('Robustness')
    report('time-shuffled copies: signed error of the observed persistence under event sampling (pp)', en(sg_s['observed', 'event'], sign=True))
    th = {}
    for a in ('walk', 'truncation', 'event'):
        for k in ('gpt_6_sol', 'statistical_model'):
            sub = real[(real.sampler == a) & (real.method == k) & real.valid]
            th[k, a] = [sub.assign(x=sub[f'err_{j}'].abs()).groupby('network').x.mean().mean() for j in (3, 4, 5)]
    ok = all(th['gpt_6_sol', a][i] <= th['statistical_model', a][i]+.25 for a in ('walk', 'truncation') for i in range(3)) and \
        all(th['gpt_6_sol', 'event'][i] > th['statistical_model', 'event'][i] for i in range(3))
    report('stricter thresholds: GPT-6 Sol level with or ahead of the model (walk, truncation), behind under event sampling', 'yes' if ok else 'no')
    report('with 2 windows: mean change (pp), rank correlation', f'{en(100*ws[ws.W == 2].mean_absolute_change_vs_W5.iloc[0])}, {en(ws[ws.W == 2].spearman_vs_W5.iloc[0], 2)}')

    print(f'\nTables written to {OUT/"tables"}')


if __name__ == '__main__':
    main()
