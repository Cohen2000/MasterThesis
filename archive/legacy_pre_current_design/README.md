# Legacy material before the current design

These files belong to previous thesis designs, including V1/V2/V2.1, G0/G0b/
G0c/G0d headroom work, G1/G2/G3/G4 experiments, phase-3 pilots, lifetime and
coverage studies, walk/nonwalk screens, prompt OFAT, noise probes, wrong-direction
and target-isolation experiments, browser probes, and Codex/CC snapshots.
They are preserved for provenance, not selected as the current main study.

The archive retains source, configs, tests, scripts, SLURM jobs, docs, results,
summary tables, figures, logs, backups and the old resume patch. Original bytes
were preserved, including all raw responses/repeats and the 13 G4 CSV edits
that already existed before cleanup. Complete results directories were renamed
on the same filesystem; no gigabytes were duplicated.

The five shared modules that remain unchanged in the root `src/` are also
stored here as small exact support snapshots, so their historical implementations
are preserved alongside their old importers. These are the only source copies
made for archive support. The old root README, agent instructions and ignore
file are in `misc/`; the ignore file is named `gitignore.pre_cleanup.txt` so its
rules do not hide files inside the archive.

## Reading and execution caveats

Historical “current”, “final” and “frozen” labels refer to earlier designs.
The original numerical outputs and conclusions were not edited to agree with
the new design. Source-page metadata remains in `../../config/datasets.yaml`;
raw empirical inputs remain in `../../data/raw/`.

Relative imports among archived source files have their original neighboring
modules. Runners are **not automatically runnable in the new location**:
cwd-dependent `src/`, `results/`, `data/raw/`, registry, and cluster paths may
refer to the old layout, and some historic inputs were absent before cleanup.
No old script, SLURM job, hard-coded absolute path, results metadata, or shell
cleanup command was rewritten. Preserving them avoids changing provenance or
accidentally resuming an experiment. For future historical reproduction, resolve
paths in a separate reviewed workspace using the manifest and starting commit;
do not regenerate into this archive.

The generic empirical loading/census helpers and current design documents are
at the root. A complete list of retained mixed modules, unresolved semantics,
and all path mappings is in
[the cleanup manifest](../../docs/REPO_CLEANUP_MANIFEST.md).
