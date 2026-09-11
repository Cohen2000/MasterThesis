# Point 6: is the cluster bootstrap over twelve groups too optimistic?

Expectation going in: twelve clusters is few, so the percentile interval is
probably too narrow. **It is not.** Two of the three alternatives give a
comparable or narrower interval, and no single group carries the estimate.

Step 1 slice, 64 cases, slope of `Delta_i` on `delta_i`, point estimate 0.826.
4,000 draws, seed 20260901. `results_summary/g3/bootstrap_robustness.csv`.

| scheme | units | 95% interval | width | sd |
|---|---|---|---|---|
| **cluster over graph groups (preregistered)** | 12 | [0.734, 0.974] | 0.239 | 0.061 |
| wild cluster, Rademacher | 12 | [0.718, 0.937] | 0.219 | 0.057 |
| cluster over graph instances | 32 | [0.704, 0.950] | 0.246 | 0.063 |

All three exclude zero in every draw.

**The wild cluster bootstrap is slightly narrower, not wider.** It resamples the
sign of each cluster's residual block around the fitted line rather than
resampling clusters, which is the standard remedy when the cluster count is
small because the resampled design matrix cannot degenerate. With twelve
clusters there are 2^12 = 4096 sign vectors, so 4,000 draws nearly enumerate
them and the interval is not limited by resampling noise. That it comes out
tighter says the group-level residuals are not heavy enough to inflate the
percentile interval here.

**Resampling instances instead of groups does not narrow the interval either**
(0.246 against 0.239). That is mildly surprising — a finer unit usually buys
precision — and it means within-group dependence is weak on this slice, so the
conservative choice of clustering on the group costs almost nothing. The group
stays preregistered regardless: it is the right unit if instances inside a group
are dependent, and this measurement does not license switching after the fact.

**Leave-one-group-out**: the slope ranges 0.802 to 0.863 across the twelve
deletions, `results_summary/g3/bootstrap_leave_one_group_out.csv`. The spread is
a quarter of the bootstrap interval's width and the sign never moves. Dropping
`activity::n1500::f0` gives the smallest value at 0.802 and dropping
`real::sp_highschool2013` the largest at 0.863; no group is load-bearing.

## Limits

This is the Step 1 two-arm slice. The five-arm panel adds
`node_panel_full_history`, where `Var(delta)` collapses to 0.0026, and slope
intervals on that arm are not interpretable at all (freeze (b)). These
robustness rows are recomputed on the full panel before any of them is quoted
for the main result, and nothing here generalizes to it.

The preregistered variant is unchanged. This is a reported row beside it, never
a replacement.

## Reproducing

```bash
python src/report_bootstrap_robustness.py
```
