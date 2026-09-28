# Walk diagnostics of the added graphs

The unchanged 1000-walk audit of `scripts/audit_v10_walk.py` (same seeds, calibration and gate rule:
applicable when |stationary shift| > 0.05; pass when the design ratio removes 90% of the plugin bias
and halves its RMSE) on the four added originals and their surrogates. Bias and SD of rho_2 are
reported separately for plugin, the design ratio (gate criterion only) and the MLE (S reference).
ESS: weight ESS of the traversal weights and ratio ESS. Coverage: discovered cells per cell of the
start component, and per cell of the Louvain communities touched (a dyad counts for the community
of its first endpoint). MLE failures: walks whose histogram the production MLE rejects (e.g. only
dyads active in all five windows discovered); MLE bias and SD use the remaining walks. The 24 v11 graphs: `docs/results/panel888_v10_walk_gate_20260923`.

| Graph | L | Shift | Plugin bias | Plugin SD | Design bias | Design SD | MLE bias | MLE SD | MLE failures | Gate appl. | Gate pass | Weight ESS | Ratio ESS | Revisit rate | Distinct dyads | Components | Largest comp. | Comp. coverage | Communities | Comm. touched | Comm. coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reality_mining | 245 | 0.3586 | 0.3432 | 0.0197 | 0.0370 | 0.1841 | 0.0160 | 0.1397 | 0/1000 | True | False | 16.0 | 16.4 | 0.356 | 157.7 | 1 | 1.000 | 0.100 | 8 | 0.898 | 0.104 |
| lkml_reply | 16436 | 0.4461 | 0.3028 | 0.1373 | -0.0119 | 0.0436 | 0.0245 | 0.0543 | 0/1000 | True | True | 2720.6 | 5356.9 | 0.389 | 10045.4 | 634 | 0.991 | 0.186 | 682 | 0.013 | 0.186 |
| sp_malawi | 178 | 0.4782 | 0.3629 | 0.1628 | 0.1208 | 0.2905 | 0.1863 | 0.2088 | 0/1000 | True | False | 6.4 | 20.5 | 0.888 | 19.9 | 2 | 0.999 | 0.118 | 21 | 0.177 | 0.494 |
| nr_radoslaw_email | 249 | 0.4033 | 0.3881 | 0.0197 | 0.0091 | 0.1077 | 0.0908 | 0.0552 | 0/1000 | True | True | 30.1 | 30.9 | 0.278 | 179.8 | 1 | 1.000 | 0.100 | 7 | 0.866 | 0.100 |
| reality_mining__pwt | 294 | 0.2015 | 0.2000 | 0.0048 | 0.0399 | 0.1786 | 0.0683 | 0.1039 | 0/1000 | True | False | 18.3 | 18.7 | 0.383 | 181.3 | 1 | 1.000 | 0.100 | 8 | 0.922 | 0.102 |
| lkml_reply__pwt | 14319 | 0.3791 | 0.3021 | 0.2148 | -0.0303 | 0.1345 | 0.0009 | 0.1358 | 0/1000 | True | False | 2369.6 | 4574.1 | 0.367 | 9059.5 | 634 | 0.994 | 0.176 | 672 | 0.015 | 0.176 |
| sp_malawi__pwt | 273 | 0.1637 | 0.1586 | 0.0151 | 0.0832 | 0.2008 | 0.0897 | 0.1181 | 2/1000 | True | False | 7.9 | 25.7 | 0.903 | 26.5 | 2 | 0.996 | 0.124 | 21 | 0.221 | 0.411 |
| nr_radoslaw_email__pwt | 312 | 0.1899 | 0.1853 | 0.0086 | 0.0048 | 0.1049 | -0.0139 | 0.0643 | 0/1000 | True | False | 37.0 | 38.2 | 0.309 | 215.7 | 1 | 1.000 | 0.100 | 7 | 0.869 | 0.100 |
