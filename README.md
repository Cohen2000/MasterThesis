# Persistence estimation in partially observable temporal graphs

This Master's thesis studies whether a language model can estimate persistence
in a complete temporal interaction network from a partial observation. It
separates which relationships are discovered from how much of their history is
visible, using four observation mechanisms and empirical, controlled, and
mechanistic networks.

The complete stream is normalized to `[0,1)` and divided into **W=5** equal
non-overlapping windows. `E_full` contains every undirected dyad with at least
one event in that stream. If `K_e` counts the windows in which dyad `e` is
active, the model jointly predicts

```text
rho_k = |{e in E_full : K_e >= k}| / |E_full|,  k = 2,3,4,5
1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0
ProfileMAE = mean(|rho_hat_k - rho_k| for k=2..5)
```

Mean occupancy is derived as `(1 + rho_2 + rho_3 + rho_4 + rho_5) / 5`.
Raw predictions are never clipped, reordered, or repaired.

| Role | Observation mechanism | History access |
|---|---|---|
| Reference | Uniform random node/participant panel | Full history |
| Selection dominated | One Simple Random Walk on the collapsed graph | Full histories of discovered relations |
| History loss dominated | Recency/time truncation with the same conceptual population access | Recent history only |
| Both | Sampled event stream | Selected events only |

The data strategy combines empirical originals; matched timing variants that
preserve nodes, topology and per-edge event counts; DAR(1) on DCSBM substrates
with intended sizes 500/1500 and low/high memory; and activity-driven networks
with and without tie reinforcement. Controlled `rho_2` targets near 0.15/0.55
remain provisional. The final empirical panel is **pending the dataset census**.
The complete [registry](config/datasets.yaml), including Bitcoin-OTC and the
locally absent WikiTalk entry, is preserved.

Each graph × mechanism is planned to have four independent sampler seeds and
three LLM repeats of each identical observation. Every raw repeat is retained.
The [study design](docs/STUDY_DESIGN.md) and [configuration](config/study.yaml)
define the current scope.

## Repository

| Path | Purpose |
|---|---|
| `config/` | Complete dataset registry and current design parameters |
| `data/raw/` | Immutable local empirical inputs |
| `src/` | Current adapters and preserved shared utilities; see [module guide](src/README.md) |
| `scripts/census_datasets.py` | Local census entry point for the next task |
| `tests/` | Current invariants on small fixtures |
| `docs/` | Current design, data, sampling, reproducibility and cleanup audit |
| `results/`, `figures/` | Landing pages for future outputs |
| `literature/` | Preserved research collection |
| `archive/legacy_pre_current_design/` | Historical code, configs, tests, workflows, results and logs |

This is a conservative **Phase-1 cleanup**. Five mixed utility modules remain
unchanged because scientific and historical functions share internals. The
full-history RW and current recency sampler are not yet implemented, and the
existing node panel has a budget-dependent stopping rule. The implementation
boundary is documented in [sampling](docs/SAMPLING.md) and
[reproducibility](docs/REPRODUCIBILITY.md); the archive contains the old pipelines.

## Local commands

Use the existing Python environment. No dependencies were installed during the
cleanup. Run the focused suite with either runner:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests
```

Pytest is not installed in this workspace; the standard-library runner executes
the same current test cases. `pytest.ini` excludes the historical archive.

The following census command is for the **next task**; it was not run during
cleanup. It reads locally available registry entries, reports missing entries,
never downloads inputs, and refuses to overwrite its output:

```bash
.venv/bin/python scripts/census_datasets.py --help
.venv/bin/python scripts/census_datasets.py --out results/dataset_census.csv
```

Future panel construction follows the census and timing-feasibility review.
There is no current executable panel-construction command: the old builder
hard-codes a previous panel and was archived. The preserved
`family_from_events`, `make_instance`, `make_dcsbm_graph`, `dar_event_stream`
and `activity_memory_event_stream` functions provide its reusable components.
No replacement panel, budgets, master seed, or final memory settings were chosen
as part of the cleanup.

See the [documentation index](docs/README.md),
[cleanup manifest](docs/REPO_CLEANUP_MANIFEST.md), and
[validation report](docs/REPO_CLEANUP_VALIDATION.md).
Historical experiments live in [archive/](archive/README.md).

Original code and documentation use the [MIT License](LICENSE).
[Citation metadata](CITATION.cff) and the
[third-party material notice](docs/THIRD_PARTY.md) cover attribution and inputs.
