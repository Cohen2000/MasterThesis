# Current state: v10

Design `panel888-access-v10-20260923`, source commit `8ee94c7` for the frozen Qwen bundle. The 10% main observations (7143619), 500-graph synthetic pool (7143620), ET selection and 45 pooled forests (7143701–7143703), ET reproducibility check (7143977), and offline result build (7144027) completed on uc3. The Qwen chain was submitted as round arrays 7143823, 7143824, 7143825 and archive 7143826. Integration waits for the archive. The [generated main tables](results/panel888_v10_main_20260923/MAIN_RESULTS.md) currently contain the offline methods; the same script will add Qwen from the verified archive.

## Post-hoc gate amendment, 2026-09-23

The original absolute-bias threshold of 0.01 ignored Monte Carlo error and the first-order finite-sample bias of the Hajek ratio. The amended gate requires |design(S) bias| <= 0.1 |plugin bias| and design RMSE <= 0.5 plugin RMSE on each real or surrogate source whose interaction stationary shift exceeds 0.05. The S reference remains the plain ratio; no correction column was added. The 24-graph audit (7143961; 1,000 walks per graph) passes 13 of 14 applicable sources. sp_hospital__pwt remains in every result and is flagged "not correctable at this budget." The confirmation array (7143568) found that strength-proportional starts did not remove the bias, while 4L reduced it. [WALK_GATE](results/panel888_v10_walk_gate_20260923/WALK_GATE.md) contains the generated numbers.

## Request identity and deviation

The R/H/B blocks in the separate v9 CPU preparation match v10 byte for byte, but the sealed v9 Qwen production archive has different block and payload hashes. The [request freeze](results/panel888_v10_request_freeze/REQUEST_FREEZE.json) records 0 reusable production requests, 2,592 R/H/B requests matching the CPU preparation, and 864 S/S_obs Qwen requests. Accordingly the frozen v9 decoding protocol is used to generate all 2,160 v10 Qwen calls. No answer reuse is claimed. ET anchor and regularization selection used only the synthetic development pool; real test sources and surrogates did not enter selection or training.

The v9 offline, Qwen, budget, and model evidence remains sealed under `docs/results/panel888_offline/`, `panel888_qwen/`, `panel888_budget_sensitivity/`, and `panel888_shared_mle/`. Its obsolete scripts and narrative reports are in `archive/pre_v10_20260923/`.
