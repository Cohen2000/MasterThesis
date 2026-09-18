# Repository cleanup validation

Completed as a conservative Phase-1 archival refactor on 2026-09-11.
Starting and final commit: `277d820b75b5d828639bd9c98709372dcb1bcc76`. No staging or commit.

## A. Final repository tree (depth three)

Generated/local environments and protected directories are shown but not
expanded. Ignored bytecode directories remain in place and are omitted here.
The local literature collection is also not expanded.

```text
MasterArbeit/
├── .agents/
├── .claude/
├── .codex/
├── .git/
├── .venv/
├── archive/
│   ├── legacy_pre_current_design/
│   │   ├── config/
│   │   ├── docs/
│   │   ├── figures/
│   │   ├── logs/
│   │   ├── misc/
│   │   ├── results/
│   │   ├── results_summary/
│   │   ├── scripts/
│   │   ├── slurm/
│   │   ├── src/
│   │   ├── tests/
│   │   └── README.md
│   └── README.md
├── config/
│   ├── datasets.yaml
│   └── study.yaml
├── data/
│   └── raw/
│       ├── .gitkeep
│       ├── CollegeMsg.txt.gz
│       ├── HighSchool2013_proximity_net.csv.gz
│       ├── copenhagen_bluetooth.csv.gz
│       ├── email-Eu-core-temporal.txt.gz
│       ├── hospital_lyon_contacts.dat.gz
│       ├── ht2009_contact_list.dat.gz
│       ├── ia-digg-reply.edges
│       ├── ia-enron-employees.edges
│       ├── ia-radoslaw-email.edges
│       ├── lastfm.csv
│       ├── mooc.csv
│       ├── primaryschool.csv.gz
│       ├── reddit.csv
│       ├── soc-sign-bitcoinotc.csv.gz
│       ├── sx-mathoverflow.txt.gz
│       ├── wikipedia.csv
│       └── workplace_InVS15_tij.dat.gz
├── docs/
│   ├── figures/
│   ├── DATASETS.md
│   ├── README.md
│   ├── REPO_CLEANUP_MANIFEST.md
│   ├── REPO_CLEANUP_VALIDATION.md
│   ├── REPRODUCIBILITY.md
│   ├── SAMPLING.md
│   ├── STUDY_DESIGN.md
│   ├── THIRD_PARTY.md
│   ├── repo_cleanup_git_stat.txt
│   ├── repo_cleanup_git_status.txt
│   ├── repo_cleanup_inventory.json
│   └── repo_cleanup_validation.json
├── figures/
│   └── README.md
├── literature/
│   ├── papers/
│   └── README.md
├── results/
│   └── README.md
├── scripts/
│   └── census_datasets.py
├── src/
│   ├── README.md
│   ├── benchmark_generators.py
│   ├── census.py
│   ├── dataset_census.py
│   ├── generator.py
│   ├── nonwalk_samplers.py
│   ├── persistence_evaluation.py
│   ├── persistence_prompt.py
│   └── walks.py
├── tests/
│   ├── data/
│   │   ├── __init__.py
│   │   └── test_census.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── test_metrics.py
│   ├── generators/
│   │   ├── __init__.py
│   │   └── test_generators.py
│   ├── prompting/
│   │   ├── __init__.py
│   │   └── test_prompt.py
│   ├── sampling/
│   │   ├── __init__.py
│   │   └── test_primitives.py
│   ├── targets/
│   │   ├── __init__.py
│   │   └── test_persistence.py
│   ├── README.md
│   └── __init__.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── pytest.ini
└── requirements.txt
```

## B. Active modules

| Module | Active purpose |
|---|---|
| `src/dataset_census.py` | Generic local registry loading and current census columns |
| `src/persistence_prompt.py` | Current zero-shot architecture and four-key result request |
| `src/persistence_evaluation.py` | Final-line JSON parsing, ProfileMAE and unrepaired output diagnostics |
| `scripts/census_datasets.py` | Future census CLI; only --help was run |

Five unchanged mixed modules support these interfaces and the current invariant
tests: `census.py`, `generator.py`, `benchmark_generators.py`,
`nonwalk_samplers.py`, and `walks.py`. Their reusable functions are enumerated in
[the source guide](../src/README.md). The files remain UNCERTAIN for relocation,
and their legacy branches are not additional current mechanisms or targets.

The complete dataset registry and existing requirements are unchanged.
`config/study.yaml` states W=5, the four targets/mechanisms and 4×3 replication;
it does not select the final panel or a master seed.

## C. Archived experiment families

V1/V2 benchmarks; V2.1 and lifetime work; pre-V2.1 backups; phase-3 pilots;
G0/G0b/G0c/G0d headroom; G1/G2 prompt/freeze experiments; G3/G4 runs and reports;
walk/nonwalk/crawl screens; OFAT, noise, mismatch, wrong-direction and
target-isolation experiments; browser probes; Codex/CC screen snapshots;
coverage studies and old synthetic grids; old SLURM jobs, runbooks, figures,
status documents and runtime logs.

Mixed-module snapshots preserve renewal, shuffle, rewiring, chunk, ER/BA/LFR,
heterogeneous DAR and community-correlated DAR implementations. Their original
root containers are retained under the Phase-1 exception; nothing was removed
from a shared scientific implementation.

## D. Every old → new path

The [manifest](REPO_CLEANUP_MANIFEST.md) gives all 192 logical move mappings.
[repo_cleanup_inventory.json](repo_cleanup_inventory.json) expands them to
**every one of the 2,198 moved files/symlinks**, with original and final paths,
classifications, reasons, tracked status and checksums. It also inventories all
359 retained original files/symlinks and records the five small exact support
snapshots. Directory renames moved 2,295,402,436 bytes of historical material;
no result directory was copied.

All 13 originally modified G4 summary CSVs were preserved from the working
tree. Their pre-cleanup contents are the checksummed archived versions.

## E. UNCERTAIN files left untouched

- `.claude/RESUME.md`: Local agent settings/session state: checked for project references but environment-owned preferences and permissions must not be relocated as experiment code.
- `.claude/settings.local.json`: Local agent settings/session state: checked for project references but environment-owned preferences and permissions must not be relocated as experiment code.
- `src/benchmark_generators.py`: Mixed module: empirical normalization/family extraction, homogeneous DAR and activity-driven functions coexist with legacy surrogates and DAR screens. Preserve unchanged pending separation.
- `src/census.py`: Mixed module: parsing, dyad normalization and rho/window helpers are shared by both generator modules; legacy metrics and plotting CLI remain coupled. Preserve unchanged in Phase 1.
- `src/generator.py`: Mixed module: controlled-timing Family/make_instance and DCSBM share allocation and census internals with historical ER/BA/LFR controls. Preserve implementation and seeds unchanged.
- `src/nonwalk_samplers.py`: Mixed module: uniform event reservoir and prepared-event indices are reusable; node-panel budget stopping differs from a fixed-size reference, and historical crawl families share internals. Preserve unchanged.
- `src/walks.py`: Mixed module: collapsed-graph index and simple RW transitions are reusable; timestamp-return and reverse/forward-time strategies do not implement the new full-history RW or recency truncation. Preserve unchanged.

The remaining generated bytecode was also left untouched and ignored.
Protected `.git/`, `.agents/`, `.codex/` and the installed `.venv/`
were not modified.

## F. Tests and integrity

| Check | Result |
|---|---|
| Relevant pytest invocation | Exit 1 before collection: `No module named pytest` |
| Same current cases via unittest | **29 passed**, no failures or skips |
| Python syntax | 22 current source/script/test files parsed successfully |
| Active local import closure | No imports into archived-only modules |
| Current Markdown links | All resolve |
| Census CLI `--help` | Passed; no dataset loaded and no census run |
| Original regular files | All 2,549 SHA-256 values unchanged at mapped destinations |
| Original symlinks | All 8 link targets and link metadata unchanged; targets resolve |
| Original file sizes/mtimes | All 2,557 entries unchanged using lstat |
| Raw empirical inputs | All 17 datasets plus .gitkeep unchanged in their original paths |
| Shared support snapshots | All 5 byte-identical to their untouched root source |
| Previously tracked archived files | All 482 destinations visible under the new ignore rules |
| `git diff --check` | No whitespace errors |
| Staged diff / commit | Empty / unchanged |

Test commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

No test dependencies were installed because pytest was unavailable and network
access was prohibited. Both runners target the same unittest-compatible cases.
Historical suites were archived intact rather than changed to pass against a
different scientific design. Current tests validate parsing, dyad/window
definitions, controlled timing, DAR, activity-driven dynamics, sampling
primitives, determinism, prompt/schema, raw output preservation and ProfileMAE.
They do not claim coverage of missing current sampler implementations.

The first integrity-check script used `stat` on symlinks after inventorying
them with `lstat`, producing eight false size/mtime discrepancies. The verifier
was corrected to compare link metadata; all final checks passed without changing
those links. Move preflight checks likewise stopped before any move when their
guards needed correction, as recorded in the inventory's execution notes.

The complete test output and verification counts are in
[repo_cleanup_validation.json](repo_cleanup_validation.json).

## G. Unresolved references and implementation limits

There are no broken imports in active code. Package migration is deferred:
the shared utilities mix current and historical functionality, and the
scientific boundary issues below need explicit review.

- The existing panel sampler stops on an event budget; a current reference
  panel's operational size/stopping rule remains to be specified.
- A full-history Simple RW wrapper and recency/time-truncation sampler are
  absent. Old timestamp sampling and backward walks were not renamed to imply
  the new semantics.
- The empirical normalizer includes t=1, the window helper has an EPS boundary
  guard, and achieved generator truth can use a realized horizon differing
  from the generation horizon. These details are documented unchanged.
- The old panel builder freezes a previous empirical panel and inherits old
  presets. It is archived. A current panel-construction command and final
  timing feasibility are pending the census.
- Complete future experiment orchestration and repeat storage are not yet
  implemented; existing raw repeat files remain intact in the archive.
- Historical cwd-dependent paths, cluster paths, shell commands, output
  metadata and links are preserved. They may require explicit path resolution
  for historical reproduction and are not supported current workflows.

The registry's physical timestamp units and Digg format note require
verification during the next census, not an unrecorded preprocessing change.

## H. Exact Git diff/stat

```text
482 files changed, 239 insertions(+), 82218 deletions(-)
```

`git status --porcelain=v1 --untracked-files=all` reports:

```text
4 modified tracked paths
478 deleted original tracked paths
521 untracked files
0 staged changes
```

**Git's unstaged diff does not include untracked destinations.** The 478
`D` entries describe old paths whose contents now exist in the archive;
the four modified paths are replaced landing pages/ignore rules whose originals
were also archived. The 521 untracked files consist of 482 previously tracked
archived originals, 5 exact support snapshots and 34 new files (excluding
the 4 tracked-path replacements). Previously ignored outputs remain ignored.
No historical content was deleted.

The exact full outputs are available in
[repo_cleanup_git_stat.txt](repo_cleanup_git_stat.txt) and
[repo_cleanup_git_status.txt](repo_cleanup_git_status.txt). These were obtained
with optional Git locks disabled; no staging was used to force rename detection.

## I. Safety confirmations

- No raw data or registry contents were modified, moved or regenerated.
- No original historical code, config, test, document, result, response or log
  was deleted; all original file/link contents were verified.
- No external API or network calls, dataset/model downloads, dependency installs,
  cluster actions or LLM experiments occurred.
- No existing scientific formulas, sampling semantics, seeds, preprocessing,
  generator implementations or experimental outputs were changed.
- New census/prompt/scoring adapters are explicitly documented; they do not
  overwrite historical targets or predictions, introduce clipping/repair, or
  import historical failure penalties.
- The dataset census and new experiments were not run. No empirical panel was
  selected, nothing was committed, and Phase 2 was not started.

Cleanup and validation stop here.
