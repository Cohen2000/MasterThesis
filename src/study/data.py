"""Networks: from a raw event list to one clean graph with five time windows.

In plain words: turns a raw event list (who interacted with whom, when) into one clean
graph object: pairs are undirected, self-loops removed, time is cut into 5 windows, and
the true rho_2..rho_5 of the complete graph are computed. "Dyad" = node pair.

A graph is a list of event records (dyad, timestamp). The archive horizon
[t_start, t_end] is cut into W=5 equal windows; K_e is the number of windows in
which dyad e has at least one event, and the estimand is rho_k = mean_e[K_e >= k]
for k = 2..5 over all dyads with at least one event.

The 16 training networks (nine of them are also test networks) are described in
config/datasets.yaml and read by prepare_real(); the three test-only networks are described
in config/pipeline.yaml and read by load_raw() and checked_graph(). real_network() returns
any of the 12 real test networks.
"""
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pandas as pd
import yaml
from .common import ROOT, W, sha, write_json, read_json

CNS_ORIGINAL_MD5 = '98892459f73e774cf79e7977edfeee3e'   # Figshare article 7267433 v1, file 14000795
_PIPELINE = yaml.safe_load((ROOT/'config/pipeline.yaml').read_text())
EXTRA_SOURCES = _PIPELINE['stage2_sources']    # test networks that are not training networks, plus nr_radoslaw_email
WINDOW_RULE = _PIPELINE['windows']             # when a nearly empty time window is trimmed (see checked_graph)


@dataclass
class Graph:
    key: str
    u: np.ndarray        # node index of every canonical (u < v) event record
    v: np.ndarray
    t: np.ndarray        # timestamp of every event record
    w: np.ndarray        # window index 0..4 of every event record
    pair: np.ndarray     # dyad index of every event record
    ends: np.ndarray     # (D, 2) node indices of every dyad
    counts: np.ndarray   # (D, 5) events per dyad and window
    horizon: tuple       # (t_start, t_end)

    @property
    def N(self): return int(self.ends.max())+1
    @property
    def D(self): return len(self.ends)
    @property
    def M(self): return len(self.t)
    @property
    def m(self): return self.counts.sum(axis=1)            # events per dyad
    @property
    # Reference: K is the temporal support of one edge, Lahiri & Berger-Wolf (2007), defs. 2-3.
    def K(self): return (self.counts > 0).sum(axis=1)      # active windows per dyad
    @property
    def cells(self): return int((self.counts > 0).sum())   # sum_e K_e, the matched quantity
    @property
    def truth(self): return [float(np.mean(self.K >= k)) for k in range(2, 6)]


def window_of(t, horizon):
    """Window index of each timestamp; a timestamp on a cut belongs to the later window."""
    cuts = np.array([horizon[0]+(horizon[1]-horizon[0])*j/W for j in range(1, W)])
    return np.searchsorted(cuts, t, side='right').astype(np.int64)


def window_counts(pair, w, D):
    return np.bincount(pair*W+w, minlength=D*W).reshape(-1, W).astype(np.int64)


def canonical(key, frame, proximity=False, horizon=None):
    """Undirected event records sorted by (u, v, t); self-events removed.

    Proximity sources (SocioPatterns, Copenhagen) record the same contact once per
    scan, so identical (u, v, t) records are removed there only. The horizon is
    the observed time range unless a fixed one (synthetic [0, 1]) is given.
    """
    x = frame[['u', 'v', 't']].copy()
    if x[['u', 'v']].isna().any().any() or not np.isfinite(x.t.to_numpy(float)).all():
        raise ValueError('non-finite or missing canonical event')
    x.u = x.u.astype(str); x.v = x.v.astype(str)
    self_events = int((x.u == x.v).sum())
    x = x[x.u != x.v].copy()
    lo = np.minimum(x.u.to_numpy(), x.v.to_numpy()); hi = np.maximum(x.u.to_numpy(), x.v.to_numpy())
    x['u'] = lo; x['v'] = hi
    duplicates = int(x.duplicated(['u', 'v', 't']).sum())
    if proximity: x = x.drop_duplicates(['u', 'v', 't'])
    if x.empty: raise ValueError('empty full archive')
    x = x.sort_values(['u', 'v', 't'], kind='stable').reset_index(drop=True)
    bounds = tuple(map(float, horizon or (x.t.min(), x.t.max())))
    if not bounds[0] < bounds[1]: raise ValueError('degenerate horizon')
    if x.t.min() < bounds[0] or x.t.max() > bounds[1]: raise ValueError('event outside fixed horizon')
    labels = np.unique(np.r_[x.u.to_numpy(), x.v.to_numpy()])
    u = np.searchsorted(labels, x.u).astype(np.int64)
    v = np.searchsorted(labels, x.v).astype(np.int64)
    ends, pair = np.unique(np.column_stack([u, v]), axis=0, return_inverse=True)
    pair = pair.astype(np.int64)
    t = x.t.to_numpy(float)
    w = window_of(t, bounds)
    g = Graph(key, u, v, t, w, pair, ends, window_counts(pair, w, len(ends)), bounds)
    if g.N < 2: raise ValueError(f'{key}: fewer than two nodes')
    cleaning = {'self_events_removed': self_events, 'same_dyad_time_duplicates': duplicates,
                'duplicates_removed': duplicates if proximity else 0, 'deduplicate_proximity': proximity}
    return g, x, cleaning


def save_graph(out, g, manifest, frame=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'graph.npz', u=g.u, v=g.v, t=g.t, w=g.w, pair=g.pair, ends=g.ends,
                        counts=g.counts, horizon=np.array(g.horizon))
    if frame is not None:
        # Stable sorted canonical export with source IDs and unmodified timestamps.
        frame.to_csv(out/'canonical.csv', index=False, float_format='%.17g', lineterminator='\n')
        manifest['canonical_sha256'] = sha(out/'canonical.csv')
    manifest.update(key=g.key, N_full=g.N, D_full=g.D, M_full=g.M, active_dyad_windows=g.cells,
                    horizon=list(g.horizon), truth=g.truth, events_per_window=g.counts.sum(0).tolist(),
                    graph_sha256=sha(out/'graph.npz'))
    write_json(out/'manifest.json', manifest)


def load_graph(out):
    out = Path(out)
    manifest = read_json(out/'manifest.json')
    if sha(out/'graph.npz') != manifest['graph_sha256']: raise ValueError('graph checksum mismatch')
    with np.load(out/'graph.npz') as z:
        arrays = {k: z[k] for k in ('u', 'v', 't', 'w', 'pair', 'ends', 'counts')}
        return Graph(manifest['key'], **arrays, horizon=tuple(z['horizon']))


def _numeric_sorted(frame):
    f = frame.copy()
    f['u'] = pd.to_numeric(f.u); f['v'] = pd.to_numeric(f.v)
    f[['u', 'v']] = np.sort(f[['u', 'v']].to_numpy(), axis=1)
    return f.sort_values(['u', 'v', 't']).reset_index(drop=True)


def copenhagen_original(raw_dir, local_export):
    """Check the local CNS export against the original Figshare release bytes.

    Returns the original cleaned frame (numeric endpoints) and the provenance record.
    """
    original = Path(raw_dir)/'bt_symmetric.csv'
    if not original.exists(): raise FileNotFoundError('CNS original bt_symmetric.csv required')
    md5 = hashlib.md5(original.read_bytes()).hexdigest()
    if md5 != CNS_ORIGINAL_MD5: raise ValueError('CNS original bytes differ from Figshare v1 file 14000795')
    bt = pd.read_csv(original)
    bt.columns = [c.lstrip('# ').strip() for c in bt.columns]
    bt = bt[bt.user_b >= 0]          # negative IDs encode empty scans / external devices
    source = bt.rename(columns={'user_a': 'u', 'user_b': 'v', 'timestamp': 't'})[['u', 'v', 't']]
    _, cleaned, cleaning = canonical('copenhagen_bluetooth', source, proximity=True)
    original_frame = _numeric_sorted(cleaned)
    same = np.array_equal(_numeric_sorted(local_export).to_numpy(), original_frame.to_numpy())
    if not same: raise ValueError('CNS original and local export differ after frozen cleaning')
    record = {'article_id': 7267433, 'version': 1, 'file_id': 14000795, 'release_md5_verified': md5,
              'original_sha256': sha(original), 'original_cleaning': cleaning, 'legacy_export_matches': same}
    return original_frame, record


# ---------------------------------------------------------------- raw files
def parse_audited(path: Path, fmt: dict, audit: dict | None = None):
    """Parse declared columns and count exclusions without guessing a layout."""
    audit = audit or {}
    columns = fmt["columns"]
    delimiter = fmt.get("delimiter", "whitespace")
    skip = fmt.get("skiprows", 0)
    comments = tuple(fmt.get("comment", "#%"))
    widths = Counter()
    counts = dict(data_rows=0, invalid_rows_removed=0, header_rows_skipped=0,
                  comment_or_blank_rows_skipped=0)
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream):
            text = line.strip()
            if line_number < skip:
                if line_number == 0 and audit.get("expected_header"):
                    if text != audit["expected_header"]:
                        raise ValueError("header differs from reviewed column layout")
                counts["header_rows_skipped"] += 1
                continue
            if not text or text.startswith(comments):
                counts["comment_or_blank_rows_skipped"] += 1
                continue
            counts["data_rows"] += 1
            fields = text.split() if delimiter == "whitespace" else text.split(delimiter)
            widths[len(fields)] += 1
            try:
                if len(fields) <= max(columns.values()):
                    raise ValueError("too few columns")
                if audit.get("expected_fields") and len(fields) != audit["expected_fields"]:
                    raise ValueError("row width differs from reviewed layout")
                u, v = fields[columns["u"]].strip(), fields[columns["v"]].strip()
                t = float(fields[columns["t"]])
                if not u or not v or not math.isfinite(t):
                    raise ValueError("empty endpoint or non-finite timestamp")
                if fmt.get("bipartite", False):
                    u, v = "u:" + u, "i:" + v
                rows.append((u, v, t))
            except (ValueError, IndexError):
                counts["invalid_rows_removed"] += 1
    counts["observed_row_widths"] = json.dumps(dict(sorted(widths.items())))
    raw = pd.DataFrame(rows, columns=["u", "v", "t"])
    counts["raw_min_timestamp"] = float(raw.t.min()) if len(raw) else None
    counts["raw_max_timestamp"] = float(raw.t.max()) if len(raw) else None
    return raw, counts


def prepare_real(key, raw_dir, out):
    spec = yaml.safe_load((ROOT/'config/datasets.yaml').read_text())['datasets'][key]
    path = Path(raw_dir)/spec['file']
    raw, parser = parse_audited(path, spec['format'], spec.get('census_audit', {}))
    proximity = key.startswith('sp_') or key == 'copenhagen_bluetooth'
    g, frame, cleaning = canonical(key, raw, proximity=proximity)
    manifest = dict(source_family=key, raw_file=path.name, raw_sha256=sha(path), provider=spec['page_url'],
                    format=spec['format'], parser=parser, cleaning=cleaning,
                    timestamp_unit=spec['census_audit']['timestamp_unit'])
    if key == 'copenhagen_bluetooth':
        frame, manifest['copenhagen'] = copenhagen_original(raw_dir, frame)
    save_graph(out, g, manifest, frame)
    return g


# ---------------------------------------------------------------- raw records
# Read the downloaded file of one source into (u, v, t) records, as described in config/pipeline.yaml (stage2_sources).
def load_raw(key, spec, raw_dir):
    """Timestamped records (u, v, t) of one source plus parser facts; fails on any
    layout that is not one event per record."""
    path = Path(raw_dir)/spec['file']
    if sha(path) != spec['sha256']: raise ValueError(f'{key}: raw file hash differs from config')
    cols = spec['columns']
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z, z.open(spec['member']) as f:
            text = io.TextIOWrapper(f, encoding='utf-8')
            header = text.readline().strip()
            if header != '# source, target, weight, time': raise ValueError(f'{key}: unexpected edge header {header!r}')
            x = pd.read_csv(text, header=None, dtype={cols['u']: str, cols['v']: str})
        if x.shape[1] != 4: raise ValueError(f'{key}: expected four columns')
        weight = x[cols['weight']].to_numpy()
        # The netzschleuder weight column counts multiplicity; every record must be one event.
        if not np.all(weight == 1): raise ValueError(f'{key}: weight column is not 1 throughout; expansion undefined')
        facts = {'weight_values': sorted(map(int, np.unique(weight)))}
    else:
        with gzip.open(path, 'rt') as f:
            header = f.readline().strip()
            if header != spec['header']: raise ValueError(f'{key}: unexpected header {header!r}')
            x = pd.read_csv(f, header=None, dtype={cols['u']: str, cols['v']: str})
        t = x[cols['t']].to_numpy()
        # SocioPatterns 20 s contact records: one row per active 20 s interval, no onset/duration.
        if not np.all(t % 20 == 0): raise ValueError(f'{key}: timestamps are not 20 s records')
        facts = {'record_resolution_seconds': 20, 'onset_duration_expansion': False}
    frame = pd.DataFrame({'u': x[cols['u']].astype(str), 'v': x[cols['v']].astype(str),
                          't': x[cols['t']].astype(float)})
    facts.update(records=int(len(frame)), raw_sha256=spec['sha256'], raw_file=spec['file'],
                 min_timestamp=float(frame.t.min()), max_timestamp=float(frame.t.max()))
    return frame, facts


# ---------------------------------------------------------------- window shares
# Share of all events that falls into each of the 5 windows.
def window_shares(g):
    return (g.counts.sum(0)/g.M).tolist()


# Find a few stray timestamps at the very start or end that would leave a window almost empty.
def end_outliers(t, max_share):
    """(low, high) bounds that drop isolated end-of-range timestamps: at most
    max_share of the records at either end, each dropped group separated from the
    rest by a gap longer than one window (a fifth) of the remaining range."""
    s = np.sort(np.asarray(t, float)); n = len(s); limit = int(max_share*n)
    lead = trail = 0
    for _ in range(2):          # alternate once so each end sees the other's trimmed range
        hi = s[n-1-trail]
        lead = max([k for k in range(1, limit+1) if s[k]-s[k-1] > (hi-s[k])/5], default=0)
        lo = s[lead]
        trail = max([k for k in range(1, limit+1) if s[n-k]-s[n-k-1] > (s[n-k-1]-lo)/5], default=0)
    return float(s[lead]), float(s[n-1-trail]), lead, trail


# Build the graph; trim stray end timestamps only under the documented rule, otherwise keep and flag.
def checked_graph(key, frame, proximity, windows=None):
    """Canonical graph after the documented window-share rule; returns (g, frame, cleaning, report)."""
    windows = windows or WINDOW_RULE
    g, clean, cleaning = canonical(key, frame, proximity=proximity)
    shares = window_shares(g)
    report = {'shares': shares, 'min_share': min(shares), 'threshold': windows['min_share'],
              'trimmed_records': 0, 'flagged': False, 'rule': 'none needed'}
    if min(shares) >= windows['min_share']:
        return g, clean, cleaning, report
    lo, hi, lead, trail = end_outliers(clean.t, windows['max_trim_share'])
    if lead or trail:
        kept = frame[(frame.t >= lo) & (frame.t <= hi)]
        g2, clean2, cleaning2 = canonical(key, kept, proximity=proximity)
        if min(window_shares(g2)) >= windows['min_share']:
            report.update(shares_before=shares, shares=window_shares(g2), min_share=min(window_shares(g2)),
                          trimmed_records=int(len(frame)-len(kept)), trimmed_leading=lead, trimmed_trailing=trail,
                          rule='isolated end-of-range timestamp outliers trimmed', kept_range=[lo, hi])
            return g2, clean2, cleaning2, report
    report.update(flagged=True, rule='window below threshold not caused by isolated end outliers; kept unchanged')
    return g, clean, cleaning, report


# ---------------------------------------------------------------- one call for any real test network
def real_network(key, raw_dir=ROOT/'data/raw'):
    """The complete graph of one of the 12 real test networks, built from its raw file."""
    spec = EXTRA_SOURCES.get(key, {})
    if 'file' in spec:
        return checked_graph(key, load_raw(key, spec, raw_dir)[0], spec['proximity'])[0]
    with tempfile.TemporaryDirectory() as tmp:
        return prepare_real(key, raw_dir, Path(tmp)/key)
