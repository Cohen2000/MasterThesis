"""P[w,t] timestamp-shuffled temporal surrogates of the real sources.

The timestamp vector of the parent's canonical event records is permuted
uniformly at random. Record i keeps its dyad, so nodes, dyad support, events per
dyad, the total event count, the timestamp multiset and the archive bounds are
preserved exactly. Records are never deduplicated after the shuffle: two records
that now share dyad and timestamp remain two events (relevant for arm B thinning).

Shuffle index 0 is the single productive surrogate. Indices 1..99 are offline
null-distribution diagnostics only; they are never selected, trained on or sent
to a model.
"""
from dataclasses import replace
import numpy as np
from .common import DESIGN_VERSION, seed, rng
from .data import save_graph, window_of, window_counts

NULL_SHUFFLES = 99


def shuffle(parent, index=0):
    domain = 'pwt_productive' if index == 0 else 'pwt_null_diagnostic'
    t = rng(domain, parent.key, '', index).permutation(parent.t)
    w = window_of(t, parent.horizon)
    return replace(parent, key=parent.key+'__pwt', t=t, w=w, counts=window_counts(parent.pair, w, parent.D))


def collisions(g):
    """Number of event records that repeat an earlier record's (dyad, timestamp)."""
    order = np.lexsort((g.t, g.pair))
    p = g.pair[order]; t = g.t[order]
    return int(np.sum((p[1:] == p[:-1]) & (t[1:] == t[:-1])))


def audit(parent, child):
    """Assert every P[w,t] invariant; returns the audit record."""
    checks = {
        'nodes': parent.N == child.N and np.array_equal(parent.u, child.u) and np.array_equal(parent.v, child.v),
        'support': np.array_equal(parent.ends, child.ends),
        'record_dyad_identity': np.array_equal(parent.pair, child.pair),
        'dyad_event_multiplicity': np.array_equal(parent.m, child.m),
        'total_events': parent.M == child.M,
        'timestamp_multiset': np.array_equal(np.sort(parent.t), np.sort(child.t)),
        'archive_bounds': parent.horizon == child.horizon,
        'count_representation': np.array_equal(child.counts, window_counts(child.pair, child.w, child.D))}
    if not all(checks.values()): raise AssertionError(checks)
    return {'checks': checks, 'passed': True,
            'parent_collisions': collisions(parent), 'surrogate_collisions': collisions(child),
            'events_removed_after_shuffle': 0,
            'event_identity': 'zero-based index of the parent canonical event records; order unchanged',
            'representation': 'labeled event multiset; simultaneous dyad records retain multiplicity'}


def prepare_surrogate(parent, out):
    g = shuffle(parent)
    report = audit(parent, g)
    save_graph(out, g, {'design_version': DESIGN_VERSION, 'parent': parent.key,
                        'seed': seed('pwt_productive', parent.key), 'null_model': 'P[w,t]',
                        'invariants': report, 'selection': 'single pre-fixed shuffle, no rejection/resampling'})
    return g
