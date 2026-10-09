"""Animation examples must follow the study's actual generator rules."""
import numpy as np

from analysis_figures import AD_ACTIVITY, ad_example, dar_example, TOY_EDGES, TOY_NODES, TOY_PANELS
from study.synthetic import ad_rows, dar_rows


def test_dar_animation_copies_active_and_inactive_states_like_production():
    seed, P = 13, 6
    runs, counts = dar_example(seed=seed, P=P)
    r = np.random.default_rng(seed)
    initial = r.random(P) < .2
    copy = r.random((4, P))
    refresh = r.random((4, P)) < .2
    event_counts = 1 + r.poisson(1, (5, P))
    offsets = np.r_[0, np.cumsum(event_counts.ravel())]
    L = dict(E=P, edges=np.array([[i, i + 1] for i in range(P)]), initial=initial,
             copy=copy, refresh=refresh, counts=event_counts, offsets=offsets,
             pos=np.full(offsets[-1], .5))
    np.testing.assert_array_equal(counts, event_counts)
    for name, alpha in [('no memory', 0.), ('memory', .8)]:
        _, production_states = dar_rows(L, alpha)
        on, kept = runs[name]
        np.testing.assert_array_equal(on, production_states)
        np.testing.assert_array_equal(on[1:][kept[1:]], on[:-1][kept[1:]])
    on, kept = runs['memory']
    assert (kept[1:] & ~on[1:]).any()  # copying OFF really occurs in the shown example
    assert (kept[1:] & on[1:]).any()


def test_partner_memory_animation_matches_synchronous_production_contacts():
    seed, rounds, N = 88, 10, len(AD_ACTIVITY)
    r = np.random.default_rng(seed)
    L = dict(N=N, rounds=rounds, activities=AD_ACTIVITY,
             activation=r.random((rounds, N)), decision=r.random((rounds, N)),
             partner=r.random((rounds, N)))
    runs = ad_example(seed, rounds)
    for name, mode in [('no memory', 'memoryless'), ('memory', 'memory')]:
        production, _ = ad_rows(L, mode, 1.)
        expected = [(i, j, (t + .5)/rounds) for t, (_, pairs) in enumerate(runs[name]) for i, j in pairs]
        assert production == expected


def test_toy_node_panel_and_walk_are_possible_views_of_one_graph():
    assert TOY_NODES == 'BCDE'
    walk_pairs = [next(k for k, edge in TOY_EDGES.items() if set(edge) == {u, v})
                  for u, v in zip([0, 2, 3, 4], [2, 3, 4, 5])]
    assert set(walk_pairs) == set(TOY_PANELS[2][1])
