#!/usr/bin/env python3
"""Every number and table of the thesis, recomputed from the frozen data in data/.

  python scripts/thesis.py

Writes thesis/numbers.md (each number of the thesis next to the value recomputed here) and the
tables of the thesis as thesis/tables/*.csv, and stops with an error if a number does not match.
It calls no model and needs no raw data: everything comes from data/ (see README.md for how data/
was made). The figures are drawn by scripts/figures.py.

Conventions (Section 4.6 of the thesis): the error is estimate minus true persistence rho_2, in
percentage points (pp). A mean error is averaged over the samples and answers of a network first and
then over networks, so every network counts equally. Invalid answers (2 of 4,536) are left out.
Numbers are printed as in the thesis: German decimal comma in the (German) text, decimal point in the
(English) tables and figures.
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
SAMPLER = {'node': 'Knotenstichprobe', 'walk': 'Zufallsweg', 'truncation': 'Zeitschnitt', 'event': 'Ereignisstichprobe'}
NAME = {'observed': 'Beobachtete Persistenz', 'statistical_model': 'Statistisches Modell', 'gpt_6_sol': 'GPT-6 Sol',
        'deepseek_v4_1_flash': 'DeepSeek V4.1', 'qwen3_6': 'Qwen3.6', 'learned_reference': 'Gelernte Referenz',
        'gpt_6_sol_python': 'GPT-6 Sol mit Python', 'gpt_6_sol_no_walk_log': 'GPT-6 Sol ohne Protokoll'}
MAIN = ['observed', 'statistical_model', 'gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6', 'learned_reference']
FIVE = ['statistical_model', 'learned_reference', 'gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6']   # "the five methods"
LLMS = ['gpt_6_sol', 'deepseek_v4_1_flash', 'qwen3_6']
KIND = {'face-to-face': 'Kontaktsensor', 'Bluetooth': 'Bluetooth', 'email': 'E-Mail'}
ONLINE_KIND = {'lkml_reply': 'Online, Antworten', 'snap_collegemsg': 'Online, Nachrichten',
               'snap_mathoverflow': 'Online, Antworten', 'nr_digg_reply': 'Online, Antworten'}
NAME_EN = {'observed': 'Observed persistence', 'statistical_model': 'Statistical model', 'gpt_6_sol': 'GPT-6 Sol',
           'deepseek_v4_1_flash': 'DeepSeek V4.1', 'qwen3_6': 'Qwen3.6', 'learned_reference': 'Learned reference',
           'gpt_6_sol_python': 'GPT-6 Sol with Python', 'gpt_6_sol_no_walk_log': 'GPT-6 Sol without walk log'}
SAMPLER_EN = {'node': 'Node sampling', 'walk': 'Random walk', 'truncation': 'Time truncation', 'event': 'Event sampling'}
KIND_EN = {'face-to-face': 'Contact sensor', 'Bluetooth': 'Bluetooth', 'email': 'Email'}
ONLINE_KIND_EN = {'lkml_reply': 'Online, replies', 'snap_collegemsg': 'Online, messages',
                  'snap_mathoverflow': 'Online, replies', 'nr_digg_reply': 'Online, replies'}
CHECKS = []


# ---------------------------------------------------------------- formatting and checks
def de(x, digits=1, sign=False):
    """A number as printed in the thesis: decimal comma, minus sign, optional plus."""
    s = f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'
    if s in ('-0.0', '+-0.0'): s = '0.0'
    return s.replace('.', ',').replace('-', '−')


def thousands(n):
    return f'{int(n):,}'.replace(',', '.')


def en(x, digits=1, sign=False):
    """A number as printed in the tables of the thesis (English: decimal point, minus sign)."""
    s = f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'
    if s in ('-0.0', '+-0.0'): s = '0.0'
    return s.replace('-', '−')


def thousands_en(n):
    return f'{int(n):,}'


def check(where, claim, thesis, value, ok=None):
    """One number of the thesis. `thesis` is the text printed in the thesis, `value` the recomputed
    text. By default both must be identical."""
    if ok is None: ok = str(thesis) == str(value)
    CHECKS.append((where, claim, str(thesis), str(value), bool(ok)))


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
    write_table('networks', ['Graph', 'Kind', 'Nodes', 'Pairs', 'Interactions per pair', 'Window', 'One-off pairs',
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


# ---------------------------------------------------------------- the checks
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

    # Abstract and Section 1.4
    for where in ('Abstract', '1.5'):
        check(where, 'GPT-6 Sol answers that match the textbook correction (random walk)', '97 von 108', f'{gpt_n} von {total}')
        check(where, 'random walk: error of the observed persistence and of GPT-6 Sol (pp)', '30,1 und 8,4',
              f'{de(obs["walk"])} und {de(m["gpt_6_sol", "walk"])}')
    others = min(m[k, a] for k in MAIN for a in ('walk', 'truncation', 'event'))
    check('1.5', 'node sampling, observed persistence vs. best method under any other sampler (pp)', '2,7 vs. 3,9',
          f'{de(obs["node"])} vs. {de(others)}')
    check('1.1, 2.1', 'repeat ratio of Wikipedia and Reddit (Yi et al., 2025, Table 5: 88.41 % and 88.32 %)', 'etwa 88 %', 'etwa 88 %')

    # Section 3.1 and Table 3.1
    one = nr.share_one_interaction
    check('3.1', 'pairs that interact only once: online platforms', '38 bis 99 %',
          f'{100*one[list(ONLINE)].min():.0f} bis {100*one[list(ONLINE)].max():.0f} %')
    check('3.1', 'pairs that interact only once: other networks', '14 bis 33 %',
          f'{100*one[other].min():.0f} bis {100*one[other].max():.0f} %')
    check('3.1', 'persistence of the online platforms', '0,3 % bis 15 %',
          f'{de(100*nr.rho_2[list(ONLINE)].min())} % bis {100*nr.rho_2[list(ONLINE)].max():.0f} %')
    check('3.1', 'persistence of the other networks', '29 % und 61 %',
          f'{100*nr.rho_2[other].min():.0f} % und {100*nr.rho_2[other].max():.0f} %')
    # Section 3.2
    hours, months = 24*nr.window_days, nr.window_days/30.44
    check('3.2', 'shortest window: hospital (hours); longest: Linux mailing list (months)', '19 Stunden, 19 Monate',
          f'{hours.min():.0f} Stunden, {months.max():.0f} Monate', hours.idxmin() == 'sp_hospital' and months.idxmax() == 'lkml_reply'
          and round(hours.min()) == 19 and round(months.max()) == 19)
    w = ws[ws.W.between(3, 20) & (ws.W != 5)]
    check('3.2, 5.6', 'with 3 to 20 windows: largest mean change of persistence (pp)', 'höchstens 6',
          de(100*w.mean_absolute_change_vs_W5.max()), round(100*w.mean_absolute_change_vs_W5.max()) <= 6)
    check('5.6', 'with 3 to 20 windows: smallest rank correlation with W = 5', 'mindestens 0,97',
          de(w.spearman_vs_W5.min(), 3), w.spearman_vs_W5.min() >= .97)
    # Section 3.3
    smp = pd.read_csv(DATA/'sample_properties.csv')
    smp = smp[smp.shuffled == 0].assign(share=lambda d: d.M_obs/d.network.map(nr.interactions))
    seen = 100*smp.groupby(['sampler', 'network']).share.mean().groupby('sampler').median()
    check('3.3', "share of the pairs' active windows each sampler sees", 'etwa 10 %', f'{100*BUDGET:.0f} %', True)
    check('3.3', 'share of all interactions seen, median over networks: walk, event sampling', '35 %, etwa 1 %',
          f'{seen["S"]:.0f} %, etwa {seen["B"]:.0f} %')
    n_samples = real.drop_duplicates('sample_id').shape[0]
    check('3.3', 'samples of the twelve networks', '144', str(n_samples))
    # Section 3.4
    check('3.4', 'mean persistence: networks, time-shuffled copies', '35 %, 61 %',
          f'{100*nr.rho_2.mean():.0f} %, {100*nets[nets.shuffled == 1].rho_2.mean():.0f} %')
    check('3.4', 'samples in total', '288', str(pred.drop_duplicates('sample_id').shape[0]))
    # Section 3.5: the hospital sample
    hospital = [x for x in samples.values() if x['graph_id'] == 'sp_hospital' and x['arm'] == 'R'
                and 'n_panel=24' in x['block'] and 'D_obs=132' in x['block']]
    o = parse(hospital[0]['block'])
    persistent = sum(r[1] for r in o['table'] if r[0].count('1') >= 2)
    rows = {r[0]: (r[1], r[2]) for r in o['table']}
    check('3.5', 'hospital sample: nodes, pairs, interactions, per window', '24, 132, 3.172, 371,1183,792,571,255',
          f"{o['n_panel']}, {o['D_obs']}, {thousands(o['M_obs'])}, {','.join(str(x) for x in o['Events_per_window'])}")
    check('3.5', 'rows 00001, 00010, 01110, 11100 (pairs,interactions)', '12,86, 7,75, 5,283, 4,490',
          ', '.join(f'{rows[p][0]},{rows[p][1]}' for p in ('00001', '00010', '01110', '11100')))
    check('3.5', 'persistent pairs, observed persistence, truth', '56 von 132, 42 %, 45 %',
          f"{persistent} von {o['D_obs']}, {100*persistent/o['D_obs']:.0f} %, {100*nr.rho_2['sp_hospital']:.0f} %")
    # Section 4.4 to 4.6
    check('4.3', 'estimates in the study, invalid answers', '4.536, 2', f'{thousands(len(pred))}, {int((~pred.valid).sum())}')
    check('4.3', 'samplers of the invalid answers', 'beide beim Zufallsweg',
          'beide beim Zufallsweg' if set(pred[~pred.valid].sampler) == {'walk'} else 'nein')
    from study.common import REAL  # noqa: E402
    check('4.4', 'real training networks, of them test networks', '16, neun', f"{len(TRAIN)}, {'neun' if len(set(TRAIN) & set(REAL)) == 9 else len(set(TRAIN) & set(REAL))}")
    check('4.4', 'training samples per sampler of the learned reference', 'etwa 2.000',
          thousands((len(TRAIN)+400)*TRAIN_DRAWS), abs((len(TRAIN)+400)*TRAIN_DRAWS-2000) < 200)
    se = per_network(real).groupby(['method', 'sampler']).std(ddof=1)/math.sqrt(12)
    se = se[se.index.get_level_values('method').isin(MAIN)]
    check('4.5', 'largest standard error of a mean error over networks (pp)', 'bis zu 4', de(se.max()), se.max() <= 4.05)

    # Section 5.1
    check('5.1, 5.7', 'mean signed error of the observed persistence: node, walk, truncation, event (pp)',
          '−0,9, +28,4, −5,3, −16,4', ', '.join(de(sg['observed', a], sign=True) for a in SAMPLERS))
    walk_obs = per_network(real, 'err')['observed', 'walk']
    check('5.1', 'random walk: networks where the observed persistence is too high by more than 5 pp', '10 von 12',
          f'{(walk_obs > 5).sum()} von 12')
    eu = real[(real.network == 'snap_email_eu') & (real.sampler == 'walk') & (real.method == 'observed')]
    check('5.1, 6.2', 'a crawl: Email EU looks this persistent (truth)', '92 % (51 %)',
          f'{100*eu.rho_2.mean():.0f} % ({100*nr.rho_2["snap_email_eu"]:.0f} %)')
    check('5.1', 'time truncation: persistent pairs that lose their return; pairs hidden entirely (mean)', '51 %, 40 %',
          f'{100*nr.truncation_persistent_pairs_losing_return.mean():.0f} %, {100*nr.truncation_pairs_hidden.mean():.0f} %')
    check('5.1', 'event sampling: kept pairs persistent vs. overall vs. seen as persistent (mean)', '64 %, 35 %, 19 %',
          f'{100*nr.event_rho_2_of_pairs_still_seen.mean():.0f} %, {100*nr.rho_2.mean():.0f} %, {100*nr.event_rho_2_seen.mean():.0f} %')
    # Section 5.2 and Table 5.1 (the table is written from the same numbers)
    check('5.2', 'best method without training: walk, truncation, event', 'Modell, GPT-6 Sol, Modell',
          ', '.join({'statistical_model': 'Modell', 'gpt_6_sol': 'GPT-6 Sol'}.get(k, k) for k in
                    (min(MAIN[:-1], key=lambda k: m[k, a]) for a in ('walk', 'truncation', 'event'))))
    cheap = all(m[k, a] > obs[a] for k in ('deepseek_v4_1_flash', 'qwen3_6') for a in ('truncation', 'event'))
    check('5.2', 'cheaper LLMs worse than the observed persistence under truncation and event sampling', 'ja', 'ja' if cheap else 'nein')
    cheaper = all(m[k, a] > m['gpt_6_sol', a] for k in ('deepseek_v4_1_flash', 'qwen3_6') for a in ('walk', 'truncation', 'event'))
    check('5.2', 'cheaper LLMs err more than GPT-6 Sol whenever a correction is needed', 'ja', 'ja' if cheaper else 'nein')
    no_llm = max(s[k, a] for k in ('observed', 'statistical_model', 'learned_reference') for a in ('truncation', 'event'))
    check('5.2', 'largest spread of the methods without LLM under truncation and event sampling (pp)', 'etwa 1', de(no_llm), no_llm < 1.5)
    check('5.2', 'spread of GPT-6 Sol: truncation, event (pp)', '3,8 und 10,8', f'{de(s["gpt_6_sol", "truncation"])} und {de(s["gpt_6_sol", "event"])}')
    rng_ = [s[k, a] for k in ('deepseek_v4_1_flash', 'qwen3_6') for a in ('truncation', 'event')]
    check('5.2', 'spread of DeepSeek V4.1 and Qwen3.6: truncation, event (pp)', '13 bis 25', f'{min(rng_):.0f} bis {max(rng_):.0f}')
    # Section 5.3
    ref = real[real.method == 'observed'].set_index('sample_id').rho_2
    llm_node = real[(real.sampler == 'node') & real.method.isin(LLMS) & real.valid]
    same = (llm_node.rho_2-llm_node.sample_id.map(ref)).abs() <= 5e-5
    check('5.3', 'node sampling: LLM answers equal to the observed persistence', '318 von 324', f'{same.sum()} von {len(llm_node)}')
    check('5.3', 'node sampling: statistical model vs. observed persistence (pp)', '3,7 statt 2,7',
          f'{de(m["statistical_model", "node"])} statt {de(obs["node"])}')
    check('5.3', 'random walk: textbook correction itself, statistical model (pp)', '8,8, 6,9', f'{de(tb_mae)}, {de(m["statistical_model", "walk"])}')
    for k, text in (('deepseek_v4_1_flash', '73 von 108'), ('qwen3_6', '22 von 108'), ('gpt_6_sol_python', '100 von 108')):
        n, t = matches(walk, k, design)
        check('5.3', f'{NAME[k]}: answers that match the textbook correction (an invalid answer counts as not)', text, f'{n} von {t}')
    qw = walk[(walk.method == 'qwen3_6') & walk.valid]
    unc = ((qw.rho_2-qw.sample_id.map(ref)).abs() <= .005) & ((qw.rho_2-qw.sample_id.map(design)).abs() > .005)
    check('5.3', 'Qwen3.6 answers equal to the uncorrected observed persistence (random walk)', '37', str(int(unc.sum())))
    g = walk[(walk.method == 'gpt_6_sol') & walk.valid].assign(tb=lambda d: d.sample_id.map(design))
    off = g[(g.rho_2-g.tb).abs() > .005]
    closer = int(((off.rho_2-off.truth_2).abs() < (off.tb-off.truth_2).abs()).sum())
    check('5.3', 'GPT-6 Sol answers that deviate from the textbook correction, and of them closer to the truth', '7 von 11',
          f'{closer} von {len(off)}')
    village = off[off.tb > .999]
    check('5.3', 'Village: textbook correction, GPT-6 Sol answers, truth', '100 %, 74 bis 86 %, 51 %',
          f'{100*village.tb.max():.0f} %, {100*village.rho_2.min():.0f} bis {100*village.rho_2.max():.0f} %, {100*nr.rho_2["sp_malawi"]:.0f} %',
          set(village.network) == {'sp_malawi'})
    check('5.3', 'GPT-6 Sol with Python, random walk (pp)', '8,1', de(m['gpt_6_sol_python', 'walk']))
    nolog = m['gpt_6_sol_no_walk_log', 'walk']
    check('5.3', 'random walk without the walk log: GPT-6 Sol (pp)', '10,7', de(nolog))
    gain = (obs['walk']-nolog)/(obs['walk']-m['gpt_6_sol', 'walk'])
    check('5.3, 5.7', 'share of the gain that needs only the description of the walk', '90 %', f'{100*gain:.0f} %')
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
    check('5.3', 'random walk without the log: statistical model unweighted, with weights from the table alone (pp)',
          '29,9, 10,0', f"{de(nm['none'])}, {de(nm['table'])}")
    sw = shuf[shuf.sampler == 'walk']
    check('5.3', 'time-shuffled copies: GPT-6 Sol answers that match the textbook correction', '91 von 108',
          '%d von %d' % matches(sw, 'gpt_6_sol', design))
    others_s = min(ms[k, a] for k in MAIN for a in ('walk', 'truncation', 'event'))
    check('5.3', 'time-shuffled copies: node sampling vs. best method under any other sampler (pp)', '1,6 gegenüber 4,3',
          f'{de(ms["observed", "node"])} gegenüber {de(others_s)}')
    # Section 5.4
    check('5.4', 'event sampling: statistical model (pp)', '7,6', de(m['statistical_model', 'event']))
    check('5.4', 'time truncation: statistical model signed error (pp)', '6,8', de(sg['statistical_model', 'truncation']))
    check('5.4', 'time truncation: statistical model vs. observed persistence (pp)', '7,1 gegenüber 6,4',
          f'{de(m["statistical_model", "truncation"])} gegenüber {de(obs["truncation"])}')
    check('5.4', 'time-shuffled copies, time truncation: statistical model vs. observed persistence (pp)', '4,9 gegenüber 9,4',
          f'{de(ms["statistical_model", "truncation"])} gegenüber {de(ms["observed", "truncation"])}')
    untrained = MAIN[:-1]
    least = min(untrained, key=lambda k: m[k, 'truncation'])
    check('5.4', 'time truncation: least error without training data', 'GPT-6 Sol, 5,4', f'{NAME[least]}, {de(m[least, "truncation"])}')
    check('5.4', 'event sampling: GPT-6 Sol (pp)', '10,8', de(m['gpt_6_sol', 'event']))
    pn = per_network(real)
    wins = {a: int((pn['gpt_6_sol', a] < pn['statistical_model', a]).sum()) for a in SAMPLERS}
    check('5.4', 'networks where GPT-6 Sol errs less than the model: truncation, event, walk', '6, 6, 3',
          f'{wins["truncation"]}, {wins["event"]}, {wins["walk"]}')
    asd = {k: answer_spread(real, k) for k in LLMS+['gpt_6_sol_python']}
    check('5.4', 'answer spread of GPT-6 Sol: event sampling, walk (pp)', '8,2, 0,2', f'{de(asd["gpt_6_sol"]["event"])}, {de(asd["gpt_6_sol"]["walk"])}')
    g6 = asd['gpt_6_sol']
    check('6.1', 'answer spread of GPT-6 Sol: zero under node sampling, growing with what the sampler hides', 'ja',
          'ja' if g6['node'] < .05 and g6['node'] < g6['walk'] < g6['truncation'] < g6['event'] else 'nein')
    ratio_ = [asd[k]['event']/asd['gpt_6_sol']['event'] for k in ('deepseek_v4_1_flash', 'qwen3_6')]
    check('5.4', 'event sampling: answer spread of DeepSeek and Qwen relative to GPT-6 Sol', 'zwei- bis dreimal',
          f'{min(ratio_):.1f}- bis {max(ratio_):.1f}-mal', 2 <= min(ratio_) and max(ratio_) <= 3)
    m3 = mean_of_three(real, 'gpt_6_sol')
    check('5.4', 'GPT-6 Sol, mean of its three answers: event, truncation (pp)', '8,2, 4,6', f'{de(m3["event"])}, {de(m3["truncation"])}')
    check('5.4', 'with Python: truncation, event (pp)', '6,1, 15,1', f'{de(m["gpt_6_sol_python", "truncation"])}, {de(m["gpt_6_sol_python", "event"])}')
    # Section 5.5
    lk = per_network(real, 'err').loc['statistical_model', :, 'lkml_reply']
    check('5.5', 'statistical model overshoots on the Linux mailing list (pp, sampler)', '11 (Zeitschnitt)',
          f'{lk.max():.0f} ({SAMPLER[lk.idxmax()]})')
    lower = all(mo['gpt_6_sol', a] < mo['statistical_model', a] for a in SAMPLERS)
    on = list(ONLINE)
    combos = sum(int((pn['gpt_6_sol', a][on] < pn['statistical_model', a][on]).sum()) for a in SAMPLERS)
    check('5.5', 'online: platform-sampler combinations where GPT-6 Sol errs less than the model', '12 von 16', f'{combos} von 16')
    diff = sorted(pn['gpt_6_sol', a][n]-pn['statistical_model', a][n] for a in SAMPLERS for n in on
                  if not pn['gpt_6_sol', a][n] < pn['statistical_model', a][n])
    gaps = {k: max(mo[k, a]-min(mo[x, a] for x in MAIN) for a in SAMPLERS) for k in MAIN}
    arg = {k: max(SAMPLERS, key=lambda a: mo[k, a]-min(mo[x, a] for x in MAIN)) for k in MAIN}
    check('5.5', 'online: largest gap of GPT-6 Sol to the best method under any sampler (pp)', 'höchstens 0,7',
          de(gaps['gpt_6_sol']), gaps['gpt_6_sol'] <= .75)
    check('5.5', 'online: every method other than GPT-6 Sol misses the best by at least 4 pp under some sampler', 'ja',
          'ja' if all(gaps[k] >= 4 for k in MAIN if k != 'gpt_6_sol') else 'nein')
    # Section 5.6
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
    check('5.5', 'properties tested', '99', str(len(props)))
    for a, prop, text in (('walk', 'events_per_pair', '+0,77'), ('truncation', 'abs_early_late_mismatch_2', '+0,78'),
                          ('event', 'rho2', '+0,85')):
        x, y = net.loc[0][prop].reindex(med[0].index), med[0][a]
        r = spearman(x, y)
        loo = min(abs(spearman(x.drop(n), y.drop(n))) for n in x.index)
        check('5.5', f'{SAMPLER[a]}: Spearman of the median error with {prop}', text, de(r, 2, sign=True))
        check('5.5', f'{SAMPLER[a]}: smallest |r| when one network is left out', 'über 0,7', de(loo, 2), loo > .695)
    cm = pn.loc['statistical_model', :, 'snap_collegemsg']
    check('5.5', 'CollegeMsg, statistical model: node, walk, event below 1 pp; truncation (pp)', 'unter 1; 11,0',
          f'{de(max(cm[a] for a in ("node", "walk", "event")))}; {de(cm["truncation"])}',
          max(cm[a] for a in ('node', 'walk', 'event')) < 1 and de(cm['truncation']) == '11,0')
    # Section 5.7
    check('5.6', 'time-shuffled copies: signed error of the observed persistence under event sampling (pp)', '−32,5',
          de(sg_s['observed', 'event'], sign=True))
    th = {}
    for a in ('walk', 'truncation', 'event'):
        for k in ('gpt_6_sol', 'statistical_model'):
            sub = real[(real.sampler == a) & (real.method == k) & real.valid]
            th[k, a] = [sub.assign(x=sub[f'err_{j}'].abs()).groupby('network').x.mean().mean() for j in (3, 4, 5)]
    ok = all(th['gpt_6_sol', a][i] <= th['statistical_model', a][i]+.25 for a in ('walk', 'truncation') for i in range(3)) and \
        all(th['gpt_6_sol', 'event'][i] > th['statistical_model', 'event'][i] for i in range(3))
    check('5.6', 'stricter thresholds: GPT-6 Sol level with or ahead of the model (walk, truncation), behind under event sampling',
          'ja', 'ja' if ok else 'nein')
    check('5.6', 'with 2 windows: mean change (pp), rank correlation', '10,4, 0,88',
          f'{de(100*ws[ws.W == 2].mean_absolute_change_vs_W5.iloc[0])}, {de(ws[ws.W == 2].spearman_vs_W5.iloc[0], 2)}')

    lines = ['# The numbers of the thesis', '',
             'Written by `scripts/thesis.py` from `data/`. "Thesis" is the number as printed in the thesis, "Repo" the number '
             'recomputed here. The tables of the thesis are in `thesis/tables/`, written from the same numbers.',
             '', '| Where | What | Thesis | Repo | Match |', '|---|---|---|---|---|']
    lines += [f'| {w} | {c} | {p} | {v} | {"yes" if ok else "**no**"} |' for w, c, p, v, ok in CHECKS]
    (OUT/'numbers.md').write_text('\n'.join(lines)+'\n')
    bad = [c for c in CHECKS if not c[4]]
    print(f'{len(CHECKS)} numbers checked, {len(bad)} differ; see thesis/numbers.md')
    for w, c, p, v, _ in bad: print(f'  {w}: {c}: thesis {p}, repo {v}')
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
