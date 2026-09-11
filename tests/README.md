# Current invariant suite

These are `unittest.TestCase` tests, discoverable by both unittest and pytest.
The 29 focused cases cover empirical parsing/normalization, complete-stream
census columns, dyad-weighted rho, W=5 boundaries, controlled timing invariants,
homogeneous DAR, activity-driven modes, deterministic sampling primitives, the
generic prompt, strict final JSON parsing, no clipping/repair and ProfileMAE.

The hand-calculated target values and timing-invariant checks are retained from
the historical suites using small fixtures. The event-reservoir test and the
whole-node-history test retain their original test bodies. Complete historical
test files are preserved in `archive/legacy_pre_current_design/tests/`.

The old `test_census.py` placed assertions in `main()` and the old generator
and walk test files executed at import time. Current invariants now use
discoverable cases; old experiment-only assertions are still in the archive.

The sampling tests validate **primitives**. They do not establish that an
adaptive whole-node budget stop is a fixed-size reference panel, that the old
timestamp-sampling walk returns full histories, or that a backward walk is
recency truncation. The missing current implementations and associated test
gaps are listed in `docs/SAMPLING.md`.
