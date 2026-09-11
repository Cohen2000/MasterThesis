# LLM leaderboard (rho_k2, identical case sets)

## condition=disclosed, input=mask

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 84 | 1.00 | 0.0215 | 0.0225 | 0.0746 | -0.002 | 1.00 | 0.0102 | 0.0198 | 0.0465 |
| baseline_et_combined | 84 | 1.00 | 0.0872 | 0.0770 | 0.3428 | +0.037 | - | - | - | - |
| baseline_et_stacked | 84 | 1.00 | 0.0829 | 0.0772 | 0.2635 | +0.031 | - | - | - | - |
| baseline_beta_block | 84 | 1.00 | 0.1358 | 0.1239 | 0.3341 | -0.008 | - | - | - | - |
| baseline_mask_mle | 84 | 1.00 | 0.1459 | 0.1547 | 0.5236 | -0.066 | - | 0.0687 | - | 0.1490 |
| baseline_occ_mle | 84 | 1.00 | 0.1558 | 0.1585 | 0.4260 | -0.079 | - | 0.0707 | - | - |
| baseline_floor | 84 | 1.00 | 0.1909 | 0.2164 | 0.4965 | +0.043 | - | 0.0940 | 0.1259 | - |
| demo_global_mean | 84 | 1.00 | 0.1868 | 0.2184 | 0.5281 | -0.000 | 1.00 | 0.1914 | 0.3014 | - |
| baseline_lt_plugin_cond | 84 | 1.00 | - | - | - | - | - | - | 0.1238 | - |
| baseline_plugin_obs | 84 | 1.00 | - | - | - | - | - | 0.1274 | 0.1831 | 0.2738 |

## condition=disclosed, input=mask_crawl_full

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 36 | 1.00 | 0.0231 | 0.0231 | 0.0526 | -0.006 | 1.00 | 0.0093 | 0.0236 | 0.0442 |
| baseline_et_stacked | 36 | 1.00 | 0.0846 | 0.0794 | 0.1731 | +0.017 | - | - | - | - |
| baseline_et_combined | 36 | 1.00 | 0.0913 | 0.0867 | 0.1784 | +0.029 | - | - | - | - |
| baseline_beta_block | 36 | 1.00 | 0.1519 | 0.1441 | 0.4491 | -0.033 | - | - | - | - |
| baseline_mask_mle | 36 | 1.00 | 0.1690 | 0.1757 | 0.5172 | -0.107 | - | 0.0732 | - | 0.1611 |
| baseline_occ_mle | 36 | 1.00 | 0.1829 | 0.1759 | 0.4500 | -0.101 | - | 0.0848 | - | - |
| baseline_floor | 36 | 1.00 | 0.1713 | 0.1919 | 0.4965 | +0.022 | - | 0.0881 | 0.1161 | - |
| demo_global_mean | 36 | 1.00 | 0.1720 | 0.1954 | 0.5281 | -0.019 | 1.00 | 0.1814 | 0.2899 | - |
| baseline_lt_plugin_cond | 36 | 1.00 | - | - | - | - | - | - | 0.1335 | - |
| baseline_plugin_obs | 36 | 1.00 | - | - | - | - | - | 0.1305 | 0.1883 | 0.2866 |

## condition=disclosed, input=mask_crawl_temporal

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 36 | 1.00 | 0.0256 | 0.0254 | 0.0496 | -0.001 | 1.00 | 0.0092 | 0.0241 | 0.0318 |
| baseline_et_stacked | 36 | 1.00 | 0.0846 | 0.0794 | 0.1731 | +0.017 | - | - | - | - |
| baseline_et_combined | 36 | 1.00 | 0.0913 | 0.0867 | 0.1784 | +0.029 | - | - | - | - |
| baseline_beta_block | 36 | 1.00 | 0.1519 | 0.1441 | 0.4491 | -0.033 | - | - | - | - |
| baseline_mask_mle | 36 | 1.00 | 0.1690 | 0.1757 | 0.5172 | -0.107 | - | 0.0732 | - | 0.1611 |
| baseline_occ_mle | 36 | 1.00 | 0.1829 | 0.1759 | 0.4500 | -0.101 | - | 0.0848 | - | - |
| baseline_floor | 36 | 1.00 | 0.1713 | 0.1919 | 0.4965 | +0.022 | - | 0.0881 | 0.1161 | - |
| demo_global_mean | 36 | 1.00 | 0.1720 | 0.1954 | 0.5281 | -0.019 | 1.00 | 0.1814 | 0.2899 | - |
| baseline_lt_plugin_cond | 36 | 1.00 | - | - | - | - | - | - | 0.1335 | - |
| baseline_plugin_obs | 36 | 1.00 | - | - | - | - | - | 0.1305 | 0.1883 | 0.2866 |

## condition=disclosed, input=mask_crawl_temporal_recent

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 36 | 1.00 | 0.0236 | 0.0233 | 0.0646 | -0.006 | 1.00 | 0.0072 | 0.0272 | 0.0379 |
| baseline_et_stacked | 36 | 1.00 | 0.0846 | 0.0794 | 0.1731 | +0.017 | - | - | - | - |
| baseline_et_combined | 36 | 1.00 | 0.0913 | 0.0867 | 0.1784 | +0.029 | - | - | - | - |
| baseline_beta_block | 36 | 1.00 | 0.1519 | 0.1441 | 0.4491 | -0.033 | - | - | - | - |
| baseline_mask_mle | 36 | 1.00 | 0.1690 | 0.1757 | 0.5172 | -0.107 | - | 0.0732 | - | 0.1611 |
| baseline_occ_mle | 36 | 1.00 | 0.1829 | 0.1759 | 0.4500 | -0.101 | - | 0.0848 | - | - |
| baseline_floor | 36 | 1.00 | 0.1713 | 0.1919 | 0.4965 | +0.022 | - | 0.0881 | 0.1161 | - |
| demo_global_mean | 36 | 1.00 | 0.1720 | 0.1954 | 0.5281 | -0.019 | 1.00 | 0.1814 | 0.2899 | - |
| baseline_lt_plugin_cond | 36 | 1.00 | - | - | - | - | - | - | 0.1335 | - |
| baseline_plugin_obs | 36 | 1.00 | - | - | - | - | - | 0.1305 | 0.1883 | 0.2866 |

## condition=disclosed, input=nw

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 36 | 1.00 | 0.0225 | 0.0199 | 0.0588 | -0.002 | 1.00 | 0.0086 | 0.0214 | 0.0436 |
| baseline_et_stacked | 36 | 1.00 | 0.0846 | 0.0794 | 0.1731 | +0.017 | - | - | - | - |
| baseline_et_combined | 36 | 1.00 | 0.0913 | 0.0867 | 0.1784 | +0.029 | - | - | - | - |
| baseline_beta_block | 36 | 1.00 | 0.1519 | 0.1441 | 0.4491 | -0.033 | - | - | - | - |
| baseline_mask_mle | 36 | 1.00 | 0.1690 | 0.1757 | 0.5172 | -0.107 | - | 0.0732 | - | 0.1611 |
| baseline_occ_mle | 36 | 1.00 | 0.1829 | 0.1759 | 0.4500 | -0.101 | - | 0.0848 | - | - |
| baseline_floor | 36 | 1.00 | 0.1713 | 0.1919 | 0.4965 | +0.022 | - | 0.0881 | 0.1161 | - |
| demo_global_mean | 36 | 1.00 | 0.1720 | 0.1954 | 0.5281 | -0.019 | 1.00 | 0.1814 | 0.2899 | - |
| baseline_lt_plugin_cond | 36 | 1.00 | - | - | - | - | - | - | 0.1335 | - |
| baseline_plugin_obs | 36 | 1.00 | - | - | - | - | - | 0.1305 | 0.1883 | 0.2866 |

## condition=disclosed_examples, input=mask

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 84 | 1.00 | 0.0239 | 0.0243 | 0.0522 | -0.003 | 1.00 | 0.0101 | 0.0217 | 0.0392 |
| baseline_et_combined | 84 | 1.00 | 0.0872 | 0.0770 | 0.3428 | +0.037 | - | - | - | - |
| baseline_et_stacked | 84 | 1.00 | 0.0829 | 0.0772 | 0.2635 | +0.031 | - | - | - | - |
| baseline_beta_block | 84 | 1.00 | 0.1358 | 0.1239 | 0.3341 | -0.008 | - | - | - | - |
| baseline_mask_mle | 84 | 1.00 | 0.1459 | 0.1547 | 0.5236 | -0.066 | - | 0.0687 | - | 0.1490 |
| baseline_occ_mle | 84 | 1.00 | 0.1558 | 0.1585 | 0.4260 | -0.079 | - | 0.0707 | - | - |
| baseline_floor | 84 | 1.00 | 0.1909 | 0.2164 | 0.4965 | +0.043 | - | 0.0940 | 0.1259 | - |
| demo_global_mean | 84 | 1.00 | 0.1868 | 0.2184 | 0.5281 | -0.000 | 1.00 | 0.1914 | 0.3014 | - |
| baseline_lt_plugin_cond | 84 | 1.00 | - | - | - | - | - | - | 0.1238 | - |
| baseline_plugin_obs | 84 | 1.00 | - | - | - | - | - | 0.1274 | 0.1831 | 0.2738 |

## condition=hidden, input=mask

| model | n | parse | MAE k2 | group-macro | worst-group | bias | cov90 | occ MAE | lifetime MAE | C MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demo_oracle_noise | 84 | 1.00 | 0.0235 | 0.0243 | 0.0537 | +0.000 | 1.00 | 0.0103 | 0.0226 | 0.0363 |
| baseline_et_combined | 84 | 1.00 | 0.0872 | 0.0770 | 0.3428 | +0.037 | - | - | - | - |
| baseline_et_stacked | 84 | 1.00 | 0.0829 | 0.0772 | 0.2635 | +0.031 | - | - | - | - |
| baseline_beta_block | 84 | 1.00 | 0.1358 | 0.1239 | 0.3341 | -0.008 | - | - | - | - |
| baseline_mask_mle | 84 | 1.00 | 0.1459 | 0.1547 | 0.5236 | -0.066 | - | 0.0687 | - | 0.1490 |
| baseline_occ_mle | 84 | 1.00 | 0.1558 | 0.1585 | 0.4260 | -0.079 | - | 0.0707 | - | - |
| baseline_floor | 84 | 1.00 | 0.1909 | 0.2164 | 0.4965 | +0.043 | - | 0.0940 | 0.1259 | - |
| demo_global_mean | 84 | 1.00 | 0.1868 | 0.2184 | 0.5281 | -0.000 | 1.00 | 0.1914 | 0.3014 | - |
| baseline_lt_plugin_cond | 84 | 1.00 | - | - | - | - | - | - | 0.1238 | - |
| baseline_plugin_obs | 84 | 1.00 | - | - | - | - | - | 0.1274 | 0.1831 | 0.2738 |

