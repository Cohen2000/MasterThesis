# Estimator-screen summary

- Cases: 12,084
- Independent source/family groups: 36
- Case files: results/benchmark_full/cases_shard_000.csv.gz, results/benchmark_full/cases_shard_001.csv.gz, results/benchmark_full/cases_shard_002.csv.gz, results/benchmark_full/cases_shard_003.csv.gz, results/benchmark_full/cases_shard_004.csv.gz, results/benchmark_full/cases_shard_005.csv.gz, results/benchmark_full/cases_shard_006.csv.gz, results/benchmark_full/cases_shard_007.csv.gz, results/benchmark_full/cases_shard_008.csv.gz, results/benchmark_full/cases_shard_009.csv.gz, results/benchmark_full/cases_shard_010.csv.gz, results/benchmark_full/cases_shard_011.csv.gz
- Split: GroupKFold by source/family; no variant-family leakage.
- True coverage is evaluation metadata and was never a model input.

## Best headline estimators by access model

| Access | Rank | Model | Group-macro MAE |
|---|---:|---|---:|
| recency_biased | 1.0 | extra_trees [combined] | 0.0664 |
| recency_biased | 2.0 | hist_gradient_boosting [combined] | 0.0679 |
| recency_biased | 3.0 | random_forest [combined] | 0.0688 |
| recency_biased | 4.0 | random_forest [patterns] | 0.0773 |
| recency_biased | 5.0 | extra_trees [patterns] | 0.0775 |
| recent_history | 1.0 | extra_trees [combined] | 0.0618 |
| recent_history | 2.0 | random_forest [combined] | 0.0649 |
| recent_history | 3.0 | hist_gradient_boosting [combined] | 0.0671 |
| recent_history | 4.0 | random_forest [occupancy] | 0.0753 |
| recent_history | 5.0 | extra_trees [occupancy] | 0.0756 |
| time_agnostic | 1.0 | mean_floor | 0.1504 |
| time_agnostic | 2.5 | ridge [combined] | 0.1666 |
| time_agnostic | 2.5 | ridge [crawl] | 0.1666 |
| time_agnostic | 4.5 | random_forest [combined] | 0.1821 |
| time_agnostic | 4.5 | random_forest [crawl] | 0.1821 |
| time_agnostic_t | 1.0 | extra_trees [combined] | 0.0594 |
| time_agnostic_t | 2.0 | random_forest [combined] | 0.0610 |
| time_agnostic_t | 3.0 | extra_trees [patterns] | 0.0624 |
| time_agnostic_t | 4.0 | hist_gradient_boosting [combined] | 0.0634 |
| time_agnostic_t | 5.0 | extra_trees [occupancy] | 0.0636 |
| time_respecting | 1.0 | hist_gradient_boosting [combined] | 0.0739 |
| time_respecting | 2.0 | extra_trees [combined] | 0.0746 |
| time_respecting | 3.0 | random_forest [combined] | 0.0750 |
| time_respecting | 4.0 | random_forest [patterns] | 0.0804 |
| time_respecting | 5.0 | hist_gradient_boosting [patterns] | 0.0805 |

## Real empirical slice

The CSVs contain all models and targets; this compact table shows the top three headline rows per access model.

| Access | Model | Input | MAE |
|---|---|---|---:|
| recency_biased | random_forest | combined | 0.0957 |
| recency_biased | extra_trees | combined | 0.1034 |
| recency_biased | random_forest | patterns | 0.1041 |
| recent_history | random_forest | occupancy | 0.0889 |
| recent_history | extra_trees | occupancy | 0.0925 |
| recent_history | hist_gradient_boosting | occupancy | 0.0951 |
| time_agnostic_t | extra_trees | occupancy | 0.0644 |
| time_agnostic_t | random_forest | occupancy | 0.0646 |
| time_agnostic_t | hist_gradient_boosting | occupancy | 0.0678 |
| time_respecting | mean_floor | none | 0.1061 |
| time_respecting | hist_gradient_boosting | crawl | 0.1093 |
| time_respecting | hist_gradient_boosting | combined | 0.1128 |
