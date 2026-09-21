# Verification

Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` after the offline build. Artifact checks use the current run. Historical sampler-helper tests remain regression checks only. New tests cover multiplicity-preserving MRRM, paired random streams, training/test draw separation and temporal-control errors. Independent full artifact verification: scripts/verify_main_offline.py and scripts/audit_panel888.py.
