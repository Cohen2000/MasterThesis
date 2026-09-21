# Current state

Design panel888-pwt-srw-20260921 (the single final panel).

- Offline freeze: commit 54d5832ce450f8bad9ac557070df5324fcd7c8b1 (pushed). Full clean
  `scripts/run_offline.sh` run, independent audit and 104 tests green; evidence in
  `docs/results/panel888_offline/`. Code cleanup verified equivalent to the pre-cleanup
  implementation (`archive/pre_panel888_cleanup_20260921/CLEANUP_EQUIVALENCE.json`).
- Qwen production: ONE chain submitted 2026-09-21T04:58Z on uc3 from that commit,
  workspace `$WS/panel888_pwt_srw_20260921`: round1=7093897, round2=7093898,
  round3=7093899, archive=7093900 (also in `mainexp/production_jobs.txt`).
  Do not submit again; after the archive job: `bash scripts/integrate_qwen.sh 54d5832ce450f8bad9ac557070df5324fcd7c8b1`.
- Sol / DeepSeek: prepared only (864 requests each), not started.
