# Active source and retained shared code

The active interface remains flat in Phase 1. No source package migration was
attempted while sampler and window semantics remain unresolved.

| Module | Current use |
|---|---|
| `dataset_census.py` | Current census columns and complete-registry local loading |
| `persistence_prompt.py` | Generic zero-shot four-component prompt renderer |
| `persistence_evaluation.py` | Final JSON parsing, raw ProfileMAE, violation diagnostics, derived occupancy |

Five mixed modules remain **UNCERTAIN for relocation**, byte-identical to their
pre-cleanup versions. The current-use portions are:

| Mixed module | Reusable definitions |
|---|---|
| `census.py` | `parse_events`, `load_registry`, `normalize`, `count_nodes`, `window_index`, `spans`, `profile`; `census_row` is retained as a compatibility dependency |
| `generator.py` | `make_dcsbm_graph`, `Family`, `Instance`, allocation helpers, `make_instance` |
| `benchmark_generators.py` | `normalize_event_stream`, `family_from_events`, homogeneous `dar_event_stream`, `activity_memory_event_stream` |
| `nonwalk_samplers.py` | Prepared event indices, log conversion, `uniform_event_reservoir`; full-history panel primitive pending its stopping-rule review |
| `walks.py` | `TemporalGraphIndex`, `build_index`, simple time-agnostic RW transitions |

Legacy branches and return fields in those modules remain compatibility code;
they do not define current targets or additional main sampling mechanisms.
Their originals are also snapshotted under the archive's `src/` so archived
source imports can still find their historical support modules.

All old panel builders, experiment runners, prompt conditions, feature screens
and baseline evaluation experiments moved to the archive. No current baseline
configuration has been selected. See [sampling](../docs/SAMPLING.md),
[reproducibility](../docs/REPRODUCIBILITY.md) and the
[cleanup manifest](../docs/REPO_CLEANUP_MANIFEST.md) before reusing a primitive.
