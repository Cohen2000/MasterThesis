# Tests

    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

One module per scientific module: data and surrogates, sampling and walk kernel,
observations and features, references, B mixture, training pool and folds,
evaluation and paired controls, transports (Qwen runner, disabled API ledger),
census parsers. Two integration tests use the prepared study when it exists and
are skipped otherwise. Full artifact checks are in scripts/audit_offline.py.
