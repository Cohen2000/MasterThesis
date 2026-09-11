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

Run later, into a new output path:

```bash
.venv/bin/python scripts/census_datasets.py --out results/dataset_census.csv
```

The adapter uses the existing registry parser and census window helpers.
Comments, skipped headers, bipartite namespaces and column positions remain
registry controlled. Input events are not deduplicated. Raw files are never
rewritten. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the retained boundary
convention and other preprocessing caveats.

Duration columns are in **source timestamp units**. The registry currently
describes units in prose rather than a complete machine-readable field; the
census review must verify each unit before converting durations to seconds or
days. The Digg registry's prose note also warns that releases can differ in
column layout; its configured time column is unchanged and should be verified
against the local file during that review.

Matched timing variants will be checked for node-set, topology and per-edge
event-count equality with each chosen empirical backbone. Event-count ceilings
and temporal capacity constraints can make 0.15/0.55 infeasible. The old panel
and census outputs remain in the archive as historical evidence and are not
used as a substitute for the pending census.
