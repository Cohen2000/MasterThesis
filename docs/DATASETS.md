> Historical pre-freeze documentation. The 2026-09-16 main experiment is defined by
> [the current runbook](MAIN_EXPERIMENT_RUNBOOK.md) and its linked freeze.

# Dataset registry and pending census

[config/datasets.yaml](../config/datasets.yaml) remains the complete authoritative
registry, unchanged by cleanup. All 18 entries are retained. The following
availability was checked from filenames only, not by running a census:

| Collection | Candidates | Local status |
|---|---|---|
| SocioPatterns | Hospital, Primary School, High School 2013, Hypertext 2009, Workplace | Present |
| SNAP | CollegeMsg, Email-Eu, MathOverflow, Bitcoin-OTC | Present |
| SNAP | WikiTalk | Registered, absent |
| Network Repository | Radoslaw Email, Enron Employees, Digg Reply | Present |
| Copenhagen | Bluetooth (existing canonical export) | Present |
| JODIE | Wikipedia, Reddit, LastFM, MOOC | Present |

Bitcoin-OTC remains a candidate. WikiTalk has not been downloaded, and no
replacement dataset or final eight-dataset panel has been selected.

The next census should record `|V|`, `|E_full|` after the repository's undirected
preprocessing, number of events `M`, `M/|E_full|`, `P(m_e>=2)`, `P(m_e>=3)`,
`P(m_e>=4)`, `P(m_e>=5)`, `rho_2..rho_5`, observation span and the physical
duration of one of the five windows. The current census adapter exposes these
columns and a derived occupancy column. It reports absent inputs explicitly.

**Census status:** computed for all 17 local datasets; outputs, conventions
and warnings are in [results/dataset_census/](../results/dataset_census/README.md).
No panel has been selected from it. To reproduce into a new directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/census_datasets.py --realized-twins --out-dir <new-dir>
```

The adapter uses the registry's column positions, headers, comment markers and
bipartite namespaces. Input events are not deduplicated. Raw files are never
rewritten. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the retained boundary
convention and other preprocessing caveats.

Each registry entry's `census_audit` block records the reviewed field width,
header, original direction and timestamp unit. Physical durations are reported
only where the unit is documented as seconds; otherwise they are `UNKNOWN`.
The Digg file was verified locally: `u v weight t`, weight 1 on every row, so
the configured time column 3 is correct.

Matched timing variants will be checked for node-set, topology and per-edge
event-count equality with each chosen empirical backbone. Event-count ceilings
and temporal capacity constraints can make 0.15/0.55 infeasible. The old panel
and census outputs remain in the archive as historical evidence and are not
used as a substitute for the pending census.
