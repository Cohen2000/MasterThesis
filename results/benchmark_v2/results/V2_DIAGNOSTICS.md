# V2 diagnostic overview

- Instances: 745
- Independent groups: 38
- Cases: 90,040

## Data blocks

| data_block                     |   0 |
|:-------------------------------|----:|
| mechanistic_activity           |  18 |
| mechanistic_dar                | 108 |
| mechanistic_dar_correlated     |  48 |
| mechanistic_dar_heterogeneous  |  96 |
| mechanistic_renewal            |  36 |
| real_chunk                     |  11 |
| real_controlled                | 196 |
| real_empirical                 |   8 |
| real_shuffle_edge_rewire       |  14 |
| real_shuffle_global            |  14 |
| real_shuffle_lifetime_resample |  14 |
| real_shuffle_within_window     |  14 |
| synthetic_controlled           | 168 |

## Best headline rows per protocol/access

- **group_kfold / recency_biased:** extra_trees [combined] — group-macro MAE 0.0729, worst-group 0.1585
- **group_kfold / recent_history_k20:** random_forest [combined_plus_estimators] — group-macro MAE 0.0648, worst-group 0.1468
- **group_kfold / recent_history_k5:** extra_trees [combined_plus_estimators] — group-macro MAE 0.0623, worst-group 0.1659
- **group_kfold / time_agnostic:** mean_floor [none] — group-macro MAE 0.2312, worst-group 0.4266
- **group_kfold / time_agnostic_t:** extra_trees [combined_plus_estimators] — group-macro MAE 0.0566, worst-group 0.0972
- **group_kfold / time_respecting:** extra_trees [combined] — group-macro MAE 0.0769, worst-group 0.1609
- **group_kfold / time_respecting_multistart3:** extra_trees [combined] — group-macro MAE 0.0781, worst-group 0.1595
- **leave_one_block_and_group_out / recency_biased:** extra_trees_lobo [combined_plus_estimators] — group-macro MAE 0.1290, worst-group 0.2257
- **leave_one_block_and_group_out / recent_history_k20:** extra_trees_lobo [combined_plus_estimators] — group-macro MAE 0.1334, worst-group 0.2096
- **leave_one_block_and_group_out / recent_history_k5:** extra_trees_lobo [combined_plus_estimators] — group-macro MAE 0.1408, worst-group 0.2273
- **leave_one_block_and_group_out / time_agnostic:** extra_trees_lobo [combined] — group-macro MAE 0.2416, worst-group 0.5245
- **leave_one_block_and_group_out / time_agnostic_t:** extra_trees_lobo [combined_plus_estimators] — group-macro MAE 0.0774, worst-group 0.1078
- **leave_one_block_and_group_out / time_respecting:** extra_trees_lobo [combined_plus_estimators] — group-macro MAE 0.1295, worst-group 0.2028
- **leave_one_block_and_group_out / time_respecting_multistart3:** extra_trees_lobo [combined_plus_estimators] — group-macro MAE 0.1289, worst-group 0.2476
- **strategy_blind_group_kfold / recency_biased:** extra_trees_strategy_blind [combined_plus_estimators] — group-macro MAE 0.0733, worst-group 0.1651
- **strategy_blind_group_kfold / recent_history_k20:** extra_trees_strategy_blind [combined_plus_estimators] — group-macro MAE 0.0667, worst-group 0.1569
- **strategy_blind_group_kfold / recent_history_k5:** extra_trees_strategy_blind [combined_plus_estimators] — group-macro MAE 0.0672, worst-group 0.1680
- **strategy_blind_group_kfold / time_agnostic_t:** extra_trees_strategy_blind [combined_plus_estimators] — group-macro MAE 0.0569, worst-group 0.1090
- **strategy_blind_group_kfold / time_respecting:** extra_trees_strategy_blind [combined] — group-macro MAE 0.0718, worst-group 0.1478
- **strategy_blind_group_kfold / time_respecting_multistart3:** extra_trees_strategy_blind [combined] — group-macro MAE 0.0726, worst-group 0.1465
- **synthetic_to_real / recency_biased:** extra_trees_sim2real [patterns] — group-macro MAE 0.1256, worst-group 0.2798
- **synthetic_to_real / recent_history_k20:** hist_gradient_boosting_sim2real [patterns] — group-macro MAE 0.1161, worst-group 0.1885
- **synthetic_to_real / recent_history_k5:** extra_trees_sim2real [patterns] — group-macro MAE 0.1089, worst-group 0.2016
- **synthetic_to_real / time_agnostic:** hist_gradient_boosting_sim2real [combined] — group-macro MAE 0.4421, worst-group 0.4421
- **synthetic_to_real / time_agnostic_t:** extra_trees_sim2real [combined] — group-macro MAE 0.0618, worst-group 0.0758
- **synthetic_to_real / time_respecting:** extra_trees_sim2real [combined_plus_estimators] — group-macro MAE 0.1246, worst-group 0.2112
- **synthetic_to_real / time_respecting_multistart3:** extra_trees_sim2real [patterns] — group-macro MAE 0.1311, worst-group 0.2882
