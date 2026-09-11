# Reproducibility

## Immutable inputs and provenance

`data/raw/` is immutable. Its paths, bytes and timestamps were inventoried
before cleanup. The complete dataset registry is unchanged. Duplicate event
records remain events; no deduplication, filtering policy, dataset selection,
download or empirical recomputation occurred during cleanup.

The starting commit was `277d820b75b5d828639bd9c98709372dcb1bcc76`.
Thirteen G4 summary CSVs had pre-existing modifications. The checksum inventory
records their working-tree contents, not an assumed clean-commit version.
Those contents moved intact into the archive. No commit or Git index write was
performed. See the [manifest](REPO_CLEANUP_MANIFEST.md).

## Preprocessing and windows

The preserved parser reads the columns, delimiters, comment markers, header
counts and bipartite namespaces from `config/datasets.yaml`. Non-numeric
timestamps are dropped as before. `census.normalize` excludes self loops and
factorizes undirected dyads, sorting by pair and timestamp. The current census
adapter counts nodes after that self-loop exclusion so `|V|`, `|E_full|` and `M`
refer to the same retained interaction population.

The current design specifies `[0,1)` and W=5. Two inherited implementation
details remain explicitly unresolved:

1. `benchmark_generators.normalize_event_stream` maps the last event to **1.0**,
   producing a closed endpoint. `census.window_index` assigns that endpoint to
   the last window. It also uses `WINDOW_EPS=1e-9` in window coordinates to guard
   exact boundaries. An event just below a boundary can consequently advance
   a window. No endpoint transform, epsilon or preprocessing rule was changed.
2. `census.census_row` sets its horizon from the minimum and maximum event times.
   The controlled generator uses `Family.T` when assigning slots, then calls
   `census_row` for achieved targets. Sparse mechanistic streams can therefore
   have a different realized horizon from their generation horizon. Resolve
   this explicitly before constructing the new experiment's truth table.

The current census adapter preserves these helpers and publishes their W=5
rho values under `rho_2..rho_5`. It does not silently substitute a new boundary
rule. It has only been exercised on miniature test fixtures. Physical duration
columns preserve source units pending unit verification in the census.

`make_instance` retains its `span_layout="legacy"` default and the previous
allocation/capacity behavior. The intended controlled use must choose layout
explicitly and report achieved profiles/deviations. Homogeneous DAR retains its
existing sparse-stream fallback and event-emission layer. No generator formula,
event-rate model or random-number call order changed.

## Seeds, identities and raw repeats

The current plan has four sampler-seed slots per graph × mechanism and three
LLM repeats per identical observation. The master seed is deliberately unset in
`config/study.yaml`. No prior master seed is silently adopted.

Record graph/backbone identity and any generator seed separately from the
sampler seed. Sampler randomness must identify the graph, mechanism and seed
slot; LLM repeats share the resulting observation and its content hash. Preserve
the actual numeric seeds, repeat IDs, model/settings, raw responses and attempts
without overwriting or collapsing them. If the backend accepts a generation
seed, record it separately. The final seed derivation contract remains to be
chosen for the current pipeline. Existing generator/sampler functions continue
using their original `numpy.random.default_rng(seed)` behavior.

The archived builders used a CRC32 hash of a pipe-separated base seed and
identity parts. That historical helper and its callers are preserved; changing
the identity components would change the realizations, so those builders were
not relabeled as current wrappers.

The new evaluator parses the final non-empty line as strict JSON with exactly
four finite numeric components. Duplicate keys, missing/extra keys, booleans,
numeric strings and non-finite numbers are rejected. Bounds/monotonicity errors
remain visible and are not repaired. It scores one response at a time, retains
the raw text, and performs no retry selection, pooling, median or repeat
aggregation. Full storage/orchestration for the future experiment is not yet
implemented.

## Environment and checks

The existing `.venv` runs Python 3.10.12. Installed scientific packages at
cleanup were NumPy 2.2.6, pandas 2.3.3, NetworkX 3.4.2, scikit-learn 1.7.2,
matplotlib 3.10.9, PyYAML 6.0.3, SciPy 1.15.3 and tabulate 0.10.0.
`requirements.txt` is preserved unchanged. Nothing was installed or downloaded.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests
```

The first command passes the current suite. The pytest invocation fails before
collection because pytest is not installed. `pytest.ini` is provided for an
environment where pytest is available and limits collection to `tests/`.
The tests use tiny fixtures and temporary files; they do not read raw empirical
datasets, call APIs, run a panel, or write experimental outputs. Missing current
samplers are documented rather than covered by misleading historical tests.

The package migration to `src/masterthesis/` is deferred. Current flat imports
have no dependencies on archived-only modules. Historical scripts, command
defaults, SLURM jobs and links are preserved as provenance; they are not a
supported current execution workflow.
