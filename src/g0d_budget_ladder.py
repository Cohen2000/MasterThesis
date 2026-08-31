#!/usr/bin/env python3
"""Budget ladder for the two whole-entity access arms.

Both whole-entity samplers stop *before* the first entity whose complete
response would exceed the budget, so the sample at budget ``b`` is a prefix of
one fixed random entity order.  A single cumulative pass over that order
therefore prices an entire budget grid exactly, which is what makes a coverage
search over many candidate budgets affordable.

The ladder is a search instrument only.  Every number that enters the G0d
report is recomputed by ``run_nonwalk_screen.py`` through the production
samplers; ``--verify`` checks the two agree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_benchmark_data import stable_seed
from nonwalk_samplers import (
    event_sample_then_full_history,
    node_panel_full_history,
    prepare_dyad_histories,
    prepare_events,
)

NODE_PANEL = "node_panel_full_history"
TWO_PHASE = "event_sample_then_full_history"
ARM_KEYS = {"A": NODE_PANEL, "B": TWO_PHASE}


def _event_path(manifest_path: Path, rel: str) -> Path:
    path = Path(str(rel))
    return path if path.is_absolute() else manifest_path.parent / path


def _stops(cum_events: np.ndarray, cum_dyads: list[int],
           budgets: list[int]) -> list[tuple[int, int, int, int]]:
    """Resolve a cumulative entity ladder at every requested budget.

    The sampler breaks at the first entity that does not fit, so the realized
    sample is the prefix *before* that entity -- not the longest fitting
    subsequence.
    """
    out = []
    for budget in budgets:
        over = np.nonzero(cum_events > budget)[0]
        stop = int(over[0]) if len(over) else len(cum_events)
        if stop == 0:
            out.append((budget, 0, 0, 0))
        else:
            out.append((budget, int(cum_events[stop - 1]),
                        int(cum_dyads[stop - 1]), stop))
    return out


def ladder_node_panel(prepared, seed: int,
                      budgets: list[int]) -> list[tuple[int, int, int, int]]:
    """Cumulative unique events and dyads along the uniform node order."""
    order = np.random.default_rng(seed).permutation(prepared.active_nodes)
    events = prepared.events
    u = events["u"].to_numpy(int)
    v = events["v"].to_numpy(int)
    seen_events: set[int] = set()
    seen_dyads: set[tuple[int, int]] = set()
    cum_events, cum_dyads = [], []
    for raw_node in order:
        node = int(raw_node)
        new = [int(e) for e in prepared.incident_event_ids[node]
               if int(e) not in seen_events]
        seen_events.update(new)
        seen_dyads.update((u[e], v[e]) for e in new)
        cum_events.append(len(seen_events))
        cum_dyads.append(len(seen_dyads))
    return _stops(np.asarray(cum_events, dtype=np.int64), cum_dyads, budgets)


def ladder_two_phase(dyads, seed: int,
                     budgets: list[int]) -> list[tuple[int, int, int, int]]:
    """Cumulative unique events along the uniform dyad discovery order."""
    events = dyads.prepared.events
    history = {tuple(map(int, edge)): np.asarray(ids, dtype=np.int64)
               for edge, ids in zip(dyads.edges, dyads.event_ids)}
    u = events["u"].to_numpy(int)
    v = events["v"].to_numpy(int)
    order = np.random.default_rng(seed).permutation(len(events))
    discovered: set[tuple[int, int]] = set()
    seen_events: set[int] = set()
    cum_events, cum_dyads = [], []
    for raw_event in order:
        edge = (u[int(raw_event)], v[int(raw_event)])
        if edge in discovered:
            continue
        discovered.add(edge)
        seen_events.update(map(int, history[edge]))
        cum_events.append(len(seen_events))
        cum_dyads.append(len(discovered))
    return _stops(np.asarray(cum_events, dtype=np.int64), cum_dyads, budgets)


def _verify(arm: str, sample_input, seed: int, budget: int,
            expected: tuple[int, int, int, int]) -> None:
    result = (node_panel_full_history(sample_input, budget, seed) if arm == "A"
              else event_sample_then_full_history(sample_input, budget, seed))
    log = result.log
    dyads = (len(set(map(tuple, log[["u", "v"]].astype(int).to_numpy())))
             if len(log) else 0)
    if (len(log), dyads) != (expected[1], expected[2]):
        raise AssertionError(
            f"ladder disagrees with sampler for arm {arm} at budget {budget}: "
            f"{(len(log), dyads)} vs {(expected[1], expected[2])}")


def sweep(manifest_path: Path, budgets: list[int], arms: list[str],
          seeds: int, base_seed: int, verify_budget: int | None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for position, meta in enumerate(manifest.itertuples(index=False), 1):
        events = pd.read_csv(_event_path(manifest_path, str(meta.path)))
        prepared = prepare_events(events)
        dyads = prepare_dyad_histories(prepared) if "B" in arms else None
        total_dyads = int(meta.n_edges)
        for arm in arms:
            sample_input = prepared if arm == "A" else dyads
            for index in range(seeds):
                seed = stable_seed(base_seed, meta.instance_id,
                                   ARM_KEYS[arm], index)
                ladder = (ladder_node_panel(prepared, seed, budgets)
                          if arm == "A"
                          else ladder_two_phase(dyads, seed, budgets))
                for budget, realized, observed_dyads, entities in ladder:
                    if verify_budget is not None and budget == verify_budget:
                        _verify(arm, sample_input, seed, budget,
                                (budget, realized, observed_dyads, entities))
                    rows.append({
                        "instance_id": meta.instance_id,
                        "group_id": meta.group_id,
                        "arm": ARM_KEYS[arm],
                        "sample_seed": index,
                        "target_budget": budget,
                        "realized_events": realized,
                        "observed_dyads": observed_dyads,
                        "natural_units": entities,
                        "total_dyads": total_dyads,
                        "coverage": observed_dyads / max(1, total_dyads),
                        "empty": int(realized == 0),
                    })
        print(f"[g0d ladder] {position}/{len(manifest)} {meta.instance_id}",
              flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",
                        default="results/final_target_panel/panel32_final.csv")
    parser.add_argument("--budgets", required=True,
                        help="comma-separated candidate budgets")
    parser.add_argument("--arms", default="A,B",
                        help="A = node panel, B = two-phase event sample")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260831)
    parser.add_argument("--verify-budget", type=int, default=None,
                        help="replay the production sampler at this budget")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    budgets = sorted({int(b) for b in args.budgets.split(",")})
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    if set(arms) - set(ARM_KEYS):
        raise ValueError(f"unknown arm in {arms}")
    frame = sweep(Path(args.manifest), budgets, arms, args.seeds,
                  args.base_seed, args.verify_budget)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}: {len(frame)} rows", flush=True)


if __name__ == "__main__":
    main()
