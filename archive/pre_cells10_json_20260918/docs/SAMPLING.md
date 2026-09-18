> Historical pre-freeze documentation. The 2026-09-16 main experiment is defined by
> [the current runbook](MAIN_EXPERIMENT_RUNBOOK.md) and its linked freeze.

# Four observation mechanisms

The population and five-window time axis are defined from the complete stream
before sampling. Never rescale the observed timestamps to their own range.
Discovery of dyads and access to their histories are distinct parts of each
observation. Budgets and cutoff values remain to be specified for the current
experiment; historical screening budgets are not defaults.

| Mechanism | Selection/access | Returned history |
|---|---|---|
| Reference: `node_panel_full_history` | Uniform random panel of nodes/participants from the complete population | Full history for relations incident to selected participants |
| Selection: `simple_random_walk_full_history` | One local Simple Random Walk on the time-collapsed undirected graph; choose the next neighbor uniformly | Full history of each discovered relation |
| History loss: `recency_truncation` | Same conceptual population access as the reference | Only history in a recent suffix `[cutoff,1)` of the complete time axis |
| Both: `sampled_event_stream` | Select a subset of complete-stream event records | Only those selected records; no subsequent full-history lookup |

For the node panel, observing a dyad through either endpoint must not duplicate
its event records. Panel size must be specified in the operational design.
For the RW, repeated visits must not create extra empirical events, and the
start-node rule, discovery budget and disconnected-component policy must be
explicit. Only one main walk mechanism is planned, providing the EstGraph
bridge.

Recency is time truncation on the original axis, not selection of the last
fixed number of events. Full histories can contain empty early or late windows;
truncation must not erase the definition of those windows. The accessible
population/roster and treatment of dyads with no recent records must be recorded
when the sampler is implemented.

An existing uniform sample without replacement of event records is a reusable
implementation for the event-stream arm. Its sample size/fraction remains an
explicit observation parameter. More active dyads have more chances to appear,
and an observed dyad need not have its complete history returned.

## Phase-1 implementation boundary

These findings were checked in the source before archiving its callers:

- `nonwalk_samplers.uniform_event_reservoir` is a seeded uniform fixed-size
  event sample without replacement, implemented through a random-priority
  permutation. It is retained unchanged.
- `nonwalk_samplers.node_panel_full_history` recruits nodes in uniform random
  order but stops before the next whole node would exceed an event budget. Its
  adaptive stopping time is not a fixed-size uniform panel. Complete-history
  and seeding invariants are tested; a current reference wrapper is still needed.
- `walks.build_index` supplies collapsed adjacency and complete edge histories.
  The `time_agnostic` branch of `run_walk` supplies simple RW transitions. It
  currently records no timestamps. `time_agnostic_t` instead draws one timestamp
  per traversal. Neither is a completed full-history RW observation mechanism.
- `recent_history`/`recent_history_k20` is a backward temporal walk. The
  `time_prefix_events` and random event-count window functions also have
  different semantics from recent fixed-time truncation. None has been renamed
  to imply it implements the current history-loss arm.

The mixed files remain unchanged and marked UNCERTAIN for relocation. Their
historical strategies are not additional main mechanisms. The new config is a
design declaration, not a dispatch table claiming all four samplers exist.
No new sampling semantics or seed rules were introduced during cleanup.
