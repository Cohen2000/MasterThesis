# Estimator-screen summary

- Cases: 90,040
- Independent source/family groups: 38
- Case files: results/benchmark_v2/cases_shard_000.csv.gz, results/benchmark_v2/cases_shard_001.csv.gz, results/benchmark_v2/cases_shard_002.csv.gz, results/benchmark_v2/cases_shard_003.csv.gz, results/benchmark_v2/cases_shard_004.csv.gz, results/benchmark_v2/cases_shard_005.csv.gz, results/benchmark_v2/cases_shard_006.csv.gz, results/benchmark_v2/cases_shard_007.csv.gz, results/benchmark_v2/cases_shard_008.csv.gz, results/benchmark_v2/cases_shard_009.csv.gz, results/benchmark_v2/cases_shard_010.csv.gz, results/benchmark_v2/cases_shard_011.csv.gz
- Main split: GroupKFold by source/family; no variant-family leakage.
- Extra protocols: strategy-blind, block+group holdout, and synthetic-to-real when enabled.
- True coverage and EdgeBank AUC diagnostics are metadata, never model inputs.

## Best headline estimators by access model

| Access | Rank | Model | Group-macro MAE | Worst-group MAE |
|---|---:|---|---:|---:|
| recency_biased | 1.0 | extra_trees [combined] | 0.0729 | 0.1585 |
| recency_biased | 2.0 | extra_trees [combined_plus_estimators] | 0.0733 | 0.1694 |
| recency_biased | 3.0 | random_forest [combined_plus_estimators] | 0.0744 | 0.1705 |
| recency_biased | 4.0 | random_forest [combined] | 0.0744 | 0.1576 |
| recency_biased | 5.0 | hist_gradient_boosting [combined] | 0.0788 | 0.1661 |
| recent_history_k20 | 1.0 | random_forest [combined_plus_estimators] | 0.0648 | 0.1468 |
| recent_history_k20 | 2.0 | extra_trees [combined_plus_estimators] | 0.0653 | 0.1552 |
| recent_history_k20 | 3.0 | extra_trees [combined] | 0.0653 | 0.1617 |
| recent_history_k20 | 4.0 | random_forest [combined] | 0.0661 | 0.1544 |
| recent_history_k20 | 5.0 | hist_gradient_boosting [combined_plus_estimators] | 0.0700 | 0.1586 |
| recent_history_k5 | 1.0 | extra_trees [combined_plus_estimators] | 0.0623 | 0.1659 |
| recent_history_k5 | 2.0 | random_forest [combined_plus_estimators] | 0.0631 | 0.1658 |
| recent_history_k5 | 3.0 | extra_trees [combined] | 0.0634 | 0.1650 |
| recent_history_k5 | 4.0 | random_forest [combined] | 0.0671 | 0.1868 |
| recent_history_k5 | 5.0 | hist_gradient_boosting [combined_plus_estimators] | 0.0690 | 0.1740 |
| time_agnostic | 1.0 | mean_floor | 0.2312 | 0.4266 |
| time_agnostic | 2.0 | ridge [combined] | 0.2434 | 0.4837 |
| time_agnostic | 3.5 | random_forest [combined] | 0.2546 | 0.6384 |
| time_agnostic | 3.5 | random_forest [combined_plus_estimators] | 0.2546 | 0.6384 |
| time_agnostic | 5.0 | extra_trees [crawl] | 0.2550 | 0.6381 |
| time_agnostic_t | 1.0 | extra_trees [combined_plus_estimators] | 0.0566 | 0.0972 |
| time_agnostic_t | 2.0 | random_forest [combined_plus_estimators] | 0.0566 | 0.1000 |
| time_agnostic_t | 3.0 | hist_gradient_boosting [combined_plus_estimators] | 0.0610 | 0.0983 |
| time_agnostic_t | 4.0 | extra_trees [combined] | 0.0614 | 0.1353 |
| time_agnostic_t | 5.0 | random_forest [combined] | 0.0626 | 0.1477 |
| time_respecting | 1.0 | extra_trees [combined] | 0.0769 | 0.1609 |
| time_respecting | 2.0 | extra_trees [combined_plus_estimators] | 0.0776 | 0.1617 |
| time_respecting | 3.0 | random_forest [combined] | 0.0781 | 0.1609 |
| time_respecting | 4.0 | random_forest [combined_plus_estimators] | 0.0781 | 0.1573 |
| time_respecting | 5.0 | hist_gradient_boosting [combined_plus_estimators] | 0.0793 | 0.1671 |
| time_respecting_multistart3 | 1.0 | extra_trees [combined] | 0.0781 | 0.1595 |
| time_respecting_multistart3 | 2.0 | random_forest [combined] | 0.0783 | 0.1592 |
| time_respecting_multistart3 | 3.0 | extra_trees [combined_plus_estimators] | 0.0792 | 0.1634 |
| time_respecting_multistart3 | 4.0 | random_forest [combined_plus_estimators] | 0.0793 | 0.1575 |
| time_respecting_multistart3 | 5.0 | hist_gradient_boosting [combined] | 0.0808 | 0.1682 |

## Transfer protocols

| Protocol | Access | Model | Group-macro MAE |
|---|---|---|---:|
| leave_one_block_and_group_out | recency_biased | extra_trees_lobo [combined_plus_estimators] | 0.1290 |
| leave_one_block_and_group_out | recency_biased | extra_trees_lobo [combined] | 0.1304 |
| leave_one_block_and_group_out | recent_history_k20 | extra_trees_lobo [combined_plus_estimators] | 0.1334 |
| leave_one_block_and_group_out | recent_history_k20 | extra_trees_lobo [combined] | 0.1349 |
| leave_one_block_and_group_out | recent_history_k5 | extra_trees_lobo [combined_plus_estimators] | 0.1408 |
| leave_one_block_and_group_out | recent_history_k5 | extra_trees_lobo [combined] | 0.1421 |
| leave_one_block_and_group_out | time_agnostic | extra_trees_lobo [combined] | 0.2416 |
| leave_one_block_and_group_out | time_agnostic | extra_trees_lobo [combined_plus_estimators] | 0.2416 |
| leave_one_block_and_group_out | time_agnostic_t | extra_trees_lobo [combined_plus_estimators] | 0.0774 |
| leave_one_block_and_group_out | time_agnostic_t | hist_gradient_boosting_lobo [combined_plus_estimators] | 0.0804 |
| leave_one_block_and_group_out | time_respecting | extra_trees_lobo [combined_plus_estimators] | 0.1295 |
| leave_one_block_and_group_out | time_respecting | extra_trees_lobo [combined] | 0.1325 |
| leave_one_block_and_group_out | time_respecting_multistart3 | extra_trees_lobo [combined_plus_estimators] | 0.1289 |
| leave_one_block_and_group_out | time_respecting_multistart3 | extra_trees_lobo [combined] | 0.1344 |
| strategy_blind_group_kfold | recency_biased | extra_trees_strategy_blind [combined_plus_estimators] | 0.0733 |
| strategy_blind_group_kfold | recency_biased | extra_trees_strategy_blind [combined] | 0.0733 |
| strategy_blind_group_kfold | recent_history_k20 | extra_trees_strategy_blind [combined_plus_estimators] | 0.0667 |
| strategy_blind_group_kfold | recent_history_k20 | extra_trees_strategy_blind [combined] | 0.0677 |
| strategy_blind_group_kfold | recent_history_k5 | extra_trees_strategy_blind [combined_plus_estimators] | 0.0672 |
| strategy_blind_group_kfold | recent_history_k5 | extra_trees_strategy_blind [combined] | 0.0693 |
| strategy_blind_group_kfold | time_agnostic_t | extra_trees_strategy_blind [combined_plus_estimators] | 0.0569 |
| strategy_blind_group_kfold | time_agnostic_t | extra_trees_strategy_blind [combined] | 0.0600 |
| strategy_blind_group_kfold | time_respecting | extra_trees_strategy_blind [combined] | 0.0718 |
| strategy_blind_group_kfold | time_respecting | extra_trees_strategy_blind [combined_plus_estimators] | 0.0727 |
| strategy_blind_group_kfold | time_respecting_multistart3 | extra_trees_strategy_blind [combined] | 0.0726 |
| strategy_blind_group_kfold | time_respecting_multistart3 | extra_trees_strategy_blind [combined_plus_estimators] | 0.0730 |
| synthetic_to_real | recency_biased | extra_trees_sim2real [patterns] | 0.1256 |
| synthetic_to_real | recency_biased | extra_trees_sim2real [combined_plus_estimators] | 0.1359 |
| synthetic_to_real | recent_history_k20 | hist_gradient_boosting_sim2real [patterns] | 0.1161 |
| synthetic_to_real | recent_history_k20 | extra_trees_sim2real [patterns] | 0.1175 |
| synthetic_to_real | recent_history_k5 | extra_trees_sim2real [patterns] | 0.1089 |
| synthetic_to_real | recent_history_k5 | hist_gradient_boosting_sim2real [patterns] | 0.1241 |
| synthetic_to_real | time_agnostic | hist_gradient_boosting_sim2real [combined] | 0.4421 |
| synthetic_to_real | time_agnostic | hist_gradient_boosting_sim2real [combined_plus_estimators] | 0.4421 |
| synthetic_to_real | time_agnostic_t | extra_trees_sim2real [combined] | 0.0618 |
| synthetic_to_real | time_agnostic_t | extra_trees_sim2real [combined_plus_estimators] | 0.0653 |
| synthetic_to_real | time_respecting | extra_trees_sim2real [combined_plus_estimators] | 0.1246 |
| synthetic_to_real | time_respecting | extra_trees_sim2real [patterns] | 0.1267 |
| synthetic_to_real | time_respecting_multistart3 | extra_trees_sim2real [patterns] | 0.1311 |
| synthetic_to_real | time_respecting_multistart3 | extra_trees_sim2real [combined_plus_estimators] | 0.1442 |
