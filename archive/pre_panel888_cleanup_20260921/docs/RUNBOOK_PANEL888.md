# Reproduction

Use the pinned local environment and data/raw plus data/tokenizers.
Run `bash scripts/run_panel888_offline.sh`; every stage writes into a new
panel888 namespace and refuses changed checkpoint dependencies. Audit all stages
before `scripts/cluster_bundle.sh panel888_pwt_srw_20260921
results/main_experiment/panel888_pwt_srw_20260921`. Commit source before bundling.
Verify remote BUNDLE_SHA256SUMS and SPEC_COMMIT before one submit_production.sh
invocation. Consult CURRENT_STATE.md and production_jobs.txt before any submission.
Sol/DeepSeek dispatch remains prohibited.

After offline stages finish, run `bash scripts/finalize_panel888_offline.sh` to
prepare the disabled API ledger, paired baseline controls, independent audit and
compact evidence seal. This performs no inference. Source/ARTIFACT_CHECKSUMS and
FREEZE.json are in docs/results/panel888_20260921. Keep that directory immutable;
post-inference evidence belongs to its own evaluation directory.

Production submission requires the frozen commit as argument 4:
`bash submit_production.sh panel888_pwt_srw_20260921 6 all <commit>`.
The script verifies the remote bundle and clean source status, takes a lock and
writes each job ID immediately. A submission marker prohibits a second chain.
Rounds 2/3 admit only never-started requests; they never regenerate an answer.
