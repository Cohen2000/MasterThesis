"""Canonical temporal graphs: undirected dyads, event records and five windows.

A graph is a list of event records (dyad, timestamp). The archive horizon
[t_start, t_end] is cut into W=5 equal windows; K_e is the number of windows in
which dyad e has at least one event, and the estimand is rho_k = mean_e[K_e >= k]
for k = 2..5 over all dyads with at least one event.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dataset_census import parse_audited
from .common import ROOT, W, sha, write_json, read_json

CNS_ORIGINAL_MD5 = '98892459f73e774cf79e7977edfeee3e'   # Figshare article 7267433 v1, file 14000795


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
