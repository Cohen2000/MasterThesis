# Third-party material notice

The local research workspace may contain third-party research material under:

- `data/raw/`: temporal-network datasets used by the benchmark;
- `literature/papers/`: papers collected during the literature review.

These files are not distributed as part of the current public repository. They
are not authored by the repository owner and are not covered by the project's
MIT License. Copyright, database rights, attribution requirements, and
redistribution terms remain with the respective authors and data providers.

Source-page metadata for benchmark datasets is maintained in
[`../config/datasets.yaml`](../config/datasets.yaml). Bibliographic filenames in
`literature/papers/` identify the corresponding publications, but the directory
is a working reading collection rather than a curated redistributable corpus.

## Public-release status

The `.gitignore` keeps raw datasets and literature PDFs local. Files that were
tracked during development have been removed from the current Git index without
deleting the researcher's local copies. Dataset source pages are retained in
the configuration so that inputs can be obtained from their original providers.

Earlier Git revisions may still contain historical copies. Removing them from
the complete repository history requires a separate, coordinated history
rewrite; excluding them from the current revision does not retroactively change
old commits.
