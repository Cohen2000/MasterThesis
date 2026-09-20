# Source code

`main_experiment/` contains the executable implementation of the current
`cells10-json-20260918` experiment.

Primary offline entry point: `scripts/run_main_offline.py`.

The package contains data loading, observation construction, sampling, synthetic
generation, baseline training, request construction, evaluation, integrity checks
and execution support.

A small number of top-level modules are retained because the current experiment
imports reusable dataset or graph primitives from them.
