# Merge 2026-07-12
- src/config/tests/slurm/scripts: Stand MasterArbeit_benchmark_v2 (Legacy-Generatorpfad bit-exakt verifiziert).
- results/benchmark_v1, results/benchmark_v2: entpackte *_results_to_share.zip (data/logs/results je Lauf).
- data/raw: Union; JODIE (wikipedia/reddit/lastfm/mooc) und wiki-talk geloescht, Re-Download per config/datasets.yaml.
- Geloescht: MasterArbeit_benchmark_v1/, MasterArbeit_benchmark_v2/, MasterArbeit (Kopie)(.zip), .venv, __pycache__, generierte Event-Streams (deterministisch regenerierbar).
- Backups: MA_pre_merge_backup_*.tar.gz, MasterArbeit_cleanup_backup_20260705*.tar.gz

# Nachtrag 2026-07-12
- copenhagen_bluetooth.csv.gz aus figshare bt_symmetric.csv erzeugt (user_b>=0, u,v,t).
- Doku nach docs/, Run-Skripte nach scripts/ (ROOT +1 Ebene, venv -> .venv).
- v1 cases_shard_* geloescht (regenerierbar, preset full); phase3 slurm_logs als tar.gz.
