# G1 prompt contract

Prepared: **2026-09-01**  
Gate status: **G1 complete. No LLM calls were made. STOP for review before G2.**

Every text below is generated from `src/prompt_contract_g1.py`, not transcribed
beside it, so the contract and the strings G2 will render cannot drift apart.
Each block carries a truncated SHA-256 so a later prompt can be checked against
this document.

## Conditions

| condition | mechanism described | direction stated | sample shown |
|---|---|---|---|
| `hidden` | no | no | yes |
| `direction_only` | no | yes | yes |
| `mechanism` | yes | no | yes |
| `mechanism_direction` | yes | yes | yes |
| `mismatched` | wrong arm's neutral text | no | yes |
| `metadata_only` | no | no | no |

`mismatched` runs on `time_agnostic_t` <->
`event_sample_then_full_history` only, bidirectional.
`irrelevant_context` is a robustness subset, not a seventh condition.

Section order is identical in every condition and every arm — definitions,
context block, observation, task — so neither arm nor condition is confounded
with layout. Only the context block changes.

## The neutral `mechanism` texts

The acceptance criterion is that **a competent statistician can reconstruct
`P(observed | truth)` for the arm from the text plus the sample, without being
told the direction of the resulting bias.** A sketch follows each text showing
the reconstruction and stopping short of the direction, so the criterion can be
checked rather than asserted.

Each text is written from its own process. The wording that is right for a
walk — observations per pair are limited, so the number of distinct active
windows seen is bounded by the number of times that pair was seen — is simply
false for arms A and B, where every retrieved history is complete and the
entire error lives in which entities are retrieved at all. No sentence crosses
that boundary, which is enforced by test rather than by care.

`docs/HEADROOM_G0D_2026-09.md` measures the same point on the time axis:
temporal evenness runs from 0.27 on `time_respecting` to 0.82 on
`time_agnostic_t`. Even the three walks put their observations in measurably
different places, so a text describing one of them misdescribes the others.
That is the evidence for the rule, rather than an assertion of it.

The rule is "no shared phrasing where the processes differ", not "no shared
phrasing". Two sentences do recur, in both cases because the underlying process
fact is identical and inventing a difference would be its own confound: the
three walkers all record exactly the event they moved across, and both
full-history arms return `n` and a window mask that are the pair's true values.
Every arm is at least 60% prose that appears nowhere else, and the selection
rule — the thing that actually differs — is unique to each arm.

The `Consequence:` lines of the historical `disclosed` prompts ("later windows
are easier to record", "biased toward recent activity") are **not** present in
any text in this section. They name the direction and belong to
`mechanism_direction` only; their presence in the historical prompts is the
confound this condition split exists to resolve.

#### `time_agnostic_t` — walk, time-agnostic

```text
SAMPLING MECHANISM
The network was explored by a walker on the time-collapsed graph, in which all
events of a pair are merged into a single edge. From its current node the
walker stepped to a uniformly random neighbour. Each traversal of an edge
returned ONE event timestamp of that pair, drawn uniformly at random from that
pair's complete event history, independently on each traversal and with
replacement. Every step cost one budget unit and the walk continued until the
budget was spent. A pair appears in the data below only if the walker traversed
it, and n is the number of times it was traversed. The window mask shown for a
pair is the set of windows containing the n timestamps that were drawn for that
pair.
```

*Reconstruction sketch.*
A pair is in the sample iff the walk traversed it, and the traversal count n
is the pair's share of random-walk traffic on the collapsed graph. Given n,
the text states that each traversal draws one event uniformly and
independently from the pair's *complete* history. So the recorded timestamps
are n iid draws from that pair's event-time distribution, and the observed
window mask is the set of windows those n draws landed in. `P(mask | K, n)` is
therefore a classical occupancy problem over the pair's per-window event
shares, and `P(observed | truth)` factorizes into traffic-driven inclusion
times that occupancy term. Nothing in the text says which way the resulting
error points.

#### `time_respecting` — walk, forward in time

```text
SAMPLING MECHANISM
The network was explored by a walker carrying a clock. At node x with clock t
the walker chose uniformly at random among the events at x whose time is
strictly greater than t, moved across the chosen event to its other endpoint,
and set its clock to that event's time. When no event at the current node was
later than the clock, the walker restarted at a uniformly random node with a
clock drawn uniformly from [0, 1). Every move and every restart cost one budget
unit, and the walk continued until the budget was spent. Each traversal
recorded the single event it moved across. A pair appears in the data below
only if the walker traversed it, n is the number of traversals of that pair,
and the window mask shown is the set of windows containing those n recorded
events.
```

*Reconstruction sketch.*
A pair is in the sample iff the walk traversed it, and each traversal records
exactly the one event moved across. The text states the eligibility rule
explicitly: at clock t only events strictly later than t can be chosen. So the
set of a pair's events that could ever be recorded on a given visit is
determined by the arrival clock, and the restart rule (uniform node, uniform
clock in [0,1)) fixes the distribution of arrival clocks. `P(observed |
truth)` is inclusion by forward-time reachability times, for an included pair,
the distribution of which of its events were reachable given the arrival
clock. The direction of the resulting error is not stated.

#### `recent_history_k20` — walk, backward in time, k=20

```text
SAMPLING MECHANISM
The network was explored by a walker carrying a clock that started at the end
of the observation period. At node x the walker retrieved the up to 20 events
at x with the latest times strictly earlier than its clock, chose one of them
uniformly at random, moved across it to its other endpoint, and set its clock
to that event's time. When no event at the current node was earlier than the
clock, the walker restarted at a uniformly random node with its clock reset to
the end of the period. Every retrieval and every restart cost one budget unit,
and the walk continued until the budget was spent. Each traversal recorded the
single event it moved across. A pair appears in the data below only if the
walker traversed it, n is the number of traversals of that pair, and the window
mask shown is the set of windows containing those n recorded events.
```

*Reconstruction sketch.*
A pair is in the sample iff the walk traversed it, and each traversal records
exactly the one event moved across. The eligibility rule is stated: only
events strictly earlier than the clock, and only the 20 latest such events at
that node, are candidates, one chosen uniformly. The clock starts at the end
of the period and restarts reset it there. So `P(observed | truth)` is
inclusion by backward-time reachability under a 20-event retrieval window,
times the uniform choice within that window. The truncation at 20 and the
backward direction are both stated as process; neither the text nor the sample
is told which way the resulting error points.

#### `node_panel_full_history` — arm A, uniform node panel, complete histories

```text
SAMPLING MECHANISM
The network was accessed by recruiting a panel of nodes. Nodes were drawn one
at a time, uniformly at random and without replacement, from the set of all
nodes appearing anywhere in the event stream. Recruiting a node returned the
COMPLETE event history of every pair incident to that node, with no event
omitted. Recruitment continued down this random order and stopped before the
first node whose newly returned events would have taken the total past the
event budget, so no returned history was ever truncated. A pair appears in the
data below if and only if at least one of its two endpoints was recruited. For
every pair that does appear, n is that pair's true total number of events in
the full network and the window mask shown is that pair's true set of active
windows.
```

*Reconstruction sketch.*
The text states that nodes are drawn uniformly without replacement, that a
recruited node returns the complete history of every incident pair, and that a
pair is in the sample iff at least one endpoint was recruited. For a panel of
p nodes out of N, that inclusion probability is 1 - C(N-p, 2)/C(N, 2), which
does not depend on the pair. The text also states that for an included pair, n
and the window mask are the pair's true values. So `P(observed | truth)` is a
pair-independent inclusion probability times a point mass on the true mask --
no censoring and no pair-dependent selection. A statistician can read off that
the plug-in estimate is unbiased, but the text never says so; the stopping
rule is disclosed as the one thing that makes p a stopping time rather than
fixed.

#### `event_sample_then_full_history` — arm B, event sample then complete histories

```text
SAMPLING MECHANISM
The network was accessed through an event log in two phases. In phase one the
event records of the complete stream were examined in a uniformly random order,
so that any prefix of that order is a simple random sample of event records
drawn without replacement. Whenever a record named a pair that had not been
retrieved yet, phase two retrieved that pair's COMPLETE event history, with no
event omitted. The pass stopped before the first newly named pair whose
complete history would have taken the total past the event budget, so no
retrieved history was ever truncated. A pair appears in the data below only if
one of its own event records was reached in that random order. For every pair
that does appear, n is that pair's true total number of events in the full
network and the window mask shown is that pair's true set of active windows.
```

*Reconstruction sketch.*
The text states that event *records* are examined in uniformly random order,
that a pair is retrieved the first time one of its own records is reached, and
that the retrieved history is complete. A pair owning m of the M records is
therefore reached in proportion to m: its inclusion probability is an
increasing function of its own event count, derivable directly from the stated
uniform-random order over records. The text also states that for an included
pair, n and the window mask are the pair's true values. So `P(observed |
truth)` is activity-proportional inclusion times a point mass on the true mask
-- no censoring, strong pair-dependent selection. The direction the resulting
error points is a further step the text does not take.

## `direction_only`

One sentence, one template across all five arms, naming only the direction and
describing no part of the process. The wrap is applied after substitution so a
longer direction word cannot change the line layout. Directions are the
measured eight-slot naive signs from G0d. Arm A is **not** skipped: its correct
statement is that the naive estimate is approximately unbiased, and saying so is
as much a directional claim as the other four.

#### `time_agnostic_t`

```text
DIRECTION OF THE SAMPLING BIAS
For this access process, the share of pairs with K >= k computed directly from
the pairs shown below is on average an underestimate of the corresponding
share for the full network.
```

#### `time_respecting`

```text
DIRECTION OF THE SAMPLING BIAS
For this access process, the share of pairs with K >= k computed directly from
the pairs shown below is on average an underestimate of the corresponding
share for the full network.
```

#### `recent_history_k20`

```text
DIRECTION OF THE SAMPLING BIAS
For this access process, the share of pairs with K >= k computed directly from
the pairs shown below is on average an underestimate of the corresponding
share for the full network.
```

#### `node_panel_full_history`

```text
DIRECTION OF THE SAMPLING BIAS
For this access process, the share of pairs with K >= k computed directly from
the pairs shown below is on average approximately equal to the corresponding
share for the full network.
```

#### `event_sample_then_full_history`

```text
DIRECTION OF THE SAMPLING BIAS
For this access process, the share of pairs with K >= k computed directly from
the pairs shown below is on average an overestimate of the corresponding share
for the full network.
```

## `mechanism_direction`

Composed as `mechanism` + a blank line + the arm's `direction_only` sentence.
Nothing else changes, so the 2x2 is clean: the `direction stated` factor is
exactly that one sentence, in both of its levels, on every arm.

### One conflict, and how it is resolved

G1.4 asked that for the three walks `mechanism_direction` be the existing
`disclosed` prompt unchanged, so the historical comparison survives. That
cannot hold together with a clean factorial:

- `time_agnostic_t`'s historical `disclosed` text names **no** direction at
  all, so using it verbatim would make `mechanism` and `mechanism_direction`
  byte-identical for that arm and delete the `direction stated` contrast on one
  of the five arms;
- for the other two walks the historical directional wording is embedded in a
  `Consequence:` line whose phrasing differs from the `direction_only`
  template, so the factorial's second factor would not be the same manipulation
  across arms.

Resolution: the factorial is built compositionally, as above, and the exact
historical text is retained as a separate optional bridge condition,
`disclosed_historical`, byte-identical to `make_llm_prompts_v2.MECHANISM`. That
satisfies the stated purpose — the historical comparison survives, and exactly
rather than approximately — while keeping the 2x2 intact. It costs 3 arms x 32
cases = 96 extra prompts if run. **This is a decision for review**: the
factorial does not depend on it.

For the same reason the historical `MECHANISM` dict in
`src/make_llm_prompts_v2.py` is left untouched. It is the only remaining way to
regenerate the frozen 420-prompt V2.1 suite byte-identically, and editing it
would trade a reproducible frozen artifact for a change that is achieved here
by the neutral texts simply not carrying those lines.

## `hidden`, `metadata_only`, and the placebo

```text
SAMPLING MECHANISM
Not disclosed. The data below was collected by a budget-limited sampling
process on the network; the process is not described.
```

`metadata_only` gets its own context line, because `hidden`'s wording refers to
"the data below" and there is no data below:

```text
SAMPLING MECHANISM
Not disclosed, and no sampling was carried out for this network.
```

```text
NO SAMPLE WAS COLLECTED
No pairs were observed for this network. You are given only the summary facts
below about the FULL network, and must estimate the targets from them together
with whatever you know about temporal networks of this kind.
```

The metadata block is graph-level only — node count, event count, observation
period, window count — so **all five arms render an identical `metadata_only`
prompt for a given graph**: verified identical across 32 graphs.
It therefore runs **32 prompts, not 160**.

This is the **"metadata plus learned prior"** anchor. It is never to be called
a no-information baseline: size, event count and span are themselves
informative about the target, and a model's prior over networks of that size is
doing real work in the answer.

The placebo, `irrelevant_context` — same register and length band as a
`mechanism` text, content irrelevant to the estimation. Robustness subset only:
primary model, one walk arm and arm B. It guards against "more structured text
makes the model compute more carefully" as a rival explanation for a
`mechanism > hidden` effect.

```text
DATA HANDLING NOTES
The event records were held in a column-oriented store and were read once into
memory for this task. Node identifiers were renumbered to a contiguous range in
order of first appearance, and the mapping was discarded afterwards, so the
identifiers carry no meaning outside this record. Timestamps were converted
from their original units to the interval [0, 1) by an affine transform fixed
before any record was read. Records were checked for exact duplicates on the
triple (first node, second node, time), and none were found. The store was
verified against its checksum when it was opened. Within the store the records
were held sorted by time, and ties among records sharing an identical timestamp
were left in the order the original file listed them. Field widths were chosen
so that no value needed rounding when it was loaded. None of this affects which
pairs or which events are present in the data below.
```

## `mismatched` assignment

| actual_arm                     | stated_arm                     | direction_implied_by_stated_text   | direction_implied_by_actual_process   |   cases |
|:-------------------------------|:-------------------------------|:-----------------------------------|:--------------------------------------|--------:|
| time_agnostic_t                | event_sample_then_full_history | downward                           | upward                                |      32 |
| event_sample_then_full_history | time_agnostic_t                | upward                             | downward                              |      32 |

Bidirectional on this pair only. The two arms need corrections in opposite
directions, which is what makes the outcome space three-way and the reading
sharp — see the interpretation table pre-registered in
`docs/HEADROOM_G0D_2026-09.md` G0d.5 and in the G4 plan.

## Token counts

Context blocks alone:

| arm                            | condition            |   words |   characters |   portable_tokens |   qwen36_tokens_est | sha256_16        |
|:-------------------------------|:---------------------|--------:|-------------:|------------------:|--------------------:|:-----------------|
| time_agnostic_t                | hidden               |      22 |          145 |                23 |                  38 | 6ccf0702fb791f17 |
| time_agnostic_t                | direction_only       |      59 |          358 |                60 |                 100 | aeaab593eba073e4 |
| time_agnostic_t                | mechanism            |     127 |          719 |               133 |                 221 | 01aeb8d1cd721964 |
| time_agnostic_t                | mechanism_direction  |     164 |          932 |               170 |                 283 | af66a328e639da73 |
| time_agnostic_t                | metadata_only        |      13 |           83 |                14 |                  23 | 9848b779fba98fc4 |
| time_agnostic_t                | mismatched           |     153 |          861 |               161 |                 268 | 86965e9d64210a1d |
| time_agnostic_t                | disclosed_historical |      85 |          540 |                93 |                 155 | 3b3511e8f687bcb6 |
| time_agnostic_t                | irrelevant_context   |     157 |          931 |               166 |                 276 | 7efd50c5620dcbc8 |
| time_respecting                | hidden               |      22 |          145 |                23 |                  38 | 6ccf0702fb791f17 |
| time_respecting                | direction_only       |      59 |          358 |                60 |                 100 | aeaab593eba073e4 |
| time_respecting                | mechanism            |     145 |          791 |               154 |                 256 | 5b85c3e5a3d1552d |
| time_respecting                | mechanism_direction  |     182 |         1004 |               191 |                 318 | 1b8a25adf276e7e1 |
| time_respecting                | metadata_only        |      13 |           83 |                14 |                  23 | 9848b779fba98fc4 |
| time_respecting                | disclosed_historical |     110 |          646 |               124 |                 206 | 6a3cc41938237fe4 |
| time_respecting                | irrelevant_context   |     157 |          931 |               166 |                 276 | 7efd50c5620dcbc8 |
| recent_history_k20             | hidden               |      22 |          145 |                23 |                  38 | 6ccf0702fb791f17 |
| recent_history_k20             | direction_only       |      59 |          358 |                60 |                 100 | aeaab593eba073e4 |
| recent_history_k20             | mechanism            |     159 |          868 |               167 |                 278 | 12a5cc674160c51f |
| recent_history_k20             | mechanism_direction  |     196 |         1081 |               204 |                 339 | 5e0671bc17acf468 |
| recent_history_k20             | metadata_only        |      13 |           83 |                14 |                  23 | 9848b779fba98fc4 |
| recent_history_k20             | disclosed_historical |      98 |          577 |               106 |                 176 | ef810b123903cbc5 |
| recent_history_k20             | irrelevant_context   |     157 |          931 |               166 |                 276 | 7efd50c5620dcbc8 |
| node_panel_full_history        | hidden               |      22 |          145 |                23 |                  38 | 6ccf0702fb791f17 |
| node_panel_full_history        | direction_only       |      59 |          361 |                60 |                 100 | 8e96734148fa2bff |
| node_panel_full_history        | mechanism            |     140 |          796 |               147 |                 245 | cf61811826d32306 |
| node_panel_full_history        | mechanism_direction  |     177 |         1012 |               184 |                 306 | 29b4472476c68440 |
| node_panel_full_history        | metadata_only        |      13 |           83 |                14 |                  23 | 9848b779fba98fc4 |
| node_panel_full_history        | irrelevant_context   |     157 |          931 |               166 |                 276 | 7efd50c5620dcbc8 |
| event_sample_then_full_history | hidden               |      22 |          145 |                23 |                  38 | 6ccf0702fb791f17 |
| event_sample_then_full_history | direction_only       |      59 |          357 |                60 |                 100 | 03d3bccdc53a752d |
| event_sample_then_full_history | mechanism            |     153 |          861 |               161 |                 268 | 86965e9d64210a1d |
| event_sample_then_full_history | mechanism_direction  |     190 |         1073 |               198 |                 329 | 2084da5eb62fc887 |
| event_sample_then_full_history | metadata_only        |      13 |           83 |                14 |                  23 | 9848b779fba98fc4 |
| event_sample_then_full_history | mismatched           |     127 |          719 |               133 |                 221 | 01aeb8d1cd721964 |
| event_sample_then_full_history | irrelevant_context   |     157 |          931 |               166 |                 276 | 7efd50c5620dcbc8 |

Whole prompts, rendered on the 160 G0d panel samples:

| arm                            | condition           |   cases |   median_portable |   p90_portable |   median_qwen_est |   max_qwen_est |
|:-------------------------------|:--------------------|--------:|------------------:|---------------:|------------------:|---------------:|
| event_sample_then_full_history | hidden              |      32 |               860 |           1358 |              1431 |           2971 |
| event_sample_then_full_history | direction_only      |      32 |               897 |           1395 |              1492 |           3033 |
| event_sample_then_full_history | mechanism           |      32 |               998 |           1496 |              1660 |           3201 |
| event_sample_then_full_history | mechanism_direction |      32 |              1035 |           1533 |              1722 |           3263 |
| event_sample_then_full_history | mismatched          |      32 |               970 |           1468 |              1614 |           3154 |
| event_sample_then_full_history | metadata_only       |      32 |               447 |            447 |               744 |            744 |
| event_sample_then_full_history | irrelevant_context  |      32 |              1003 |           1501 |              1669 |           3209 |
| node_panel_full_history        | hidden              |      32 |               762 |            962 |              1268 |           1747 |
| node_panel_full_history        | direction_only      |      32 |               799 |            999 |              1329 |           1808 |
| node_panel_full_history        | mechanism           |      32 |               886 |           1086 |              1474 |           1953 |
| node_panel_full_history        | mechanism_direction |      32 |               923 |           1123 |              1536 |           2015 |
| node_panel_full_history        | metadata_only       |      32 |               447 |            447 |               744 |            744 |
| node_panel_full_history        | irrelevant_context  |      32 |               905 |           1105 |              1506 |           1985 |
| recent_history_k20             | hidden              |      32 |               654 |            727 |              1088 |           1288 |
| recent_history_k20             | direction_only      |      32 |               691 |            764 |              1150 |           1349 |
| recent_history_k20             | mechanism           |      32 |               798 |            871 |              1328 |           1527 |
| recent_history_k20             | mechanism_direction |      32 |               835 |            908 |              1389 |           1589 |
| recent_history_k20             | metadata_only       |      32 |               447 |            447 |               744 |            744 |
| recent_history_k20             | irrelevant_context  |      32 |               797 |            870 |              1326 |           1526 |
| time_agnostic_t                | hidden              |      32 |               562 |            652 |               935 |           1248 |
| time_agnostic_t                | direction_only      |      32 |               599 |            689 |               997 |           1309 |
| time_agnostic_t                | mechanism           |      32 |               672 |            762 |              1118 |           1431 |
| time_agnostic_t                | mechanism_direction |      32 |               709 |            799 |              1180 |           1492 |
| time_agnostic_t                | mismatched          |      32 |               700 |            790 |              1165 |           1477 |
| time_agnostic_t                | metadata_only       |      32 |               447 |            447 |               744 |            744 |
| time_agnostic_t                | irrelevant_context  |      32 |               705 |            795 |              1173 |           1486 |
| time_respecting                | hidden              |      32 |               596 |            638 |               992 |           1068 |
| time_respecting                | direction_only      |      32 |               633 |            675 |              1053 |           1130 |
| time_respecting                | mechanism           |      32 |               727 |            769 |              1210 |           1286 |
| time_respecting                | mechanism_direction |      32 |               764 |            806 |              1271 |           1348 |
| time_respecting                | metadata_only       |      32 |               447 |            447 |               744 |            744 |
| time_respecting                | irrelevant_context  |      32 |               739 |            781 |              1229 |           1306 |

**Qwen figures are estimates**, converted from the portable count at the
measured G0c ratio 1.6637. Decision 4 requires exact counts from the
BWUniCluster tokenizer before G3; until then no Qwen number here is to be
quoted as measured.

## Length band

The five neutral `mechanism` texts span **127-159 words** and **133-167
portable tokens**. Same four beats in the same order in every arm: what one
budget unit buys, how units are chosen, what one unit returns, and what `n`
and the window mask therefore mean. The placebo sits inside that band at 157
words, which is what lets it control for prose bulk rather than introduce a
length difference of its own.

Prompt length is near-constant across conditions within a case — the conditions
differ by a paragraph of prose, not by the data block — so it cancels in the
primary `mechanism - hidden` contrast. Across arms it does not cancel, because
the data block differs by arm and cannot be equalized without changing the
input contract. Cross-arm comparisons of absolute accuracy therefore stay
descriptive only, and G3 tracks response rate and validity rate per arm as well
as per condition.

## What is not settled here

- Whether `disclosed_historical` is run (96 prompts). The factorial does not
  depend on it.
- Exact Qwen token counts, pending the cluster tokenizer.
- The final output schema. The historical nine-key contract is **not**
  automatically final; `docs/TARGET_EVALUATION_FREEZE.md` lists it as an open
  gate, and the `TASK` block reused here still requests the nine keys. If the
  schema changes, every token count in this document shifts and the prompt hash
  changes with it.
- No language model has seen any of these texts. Nothing here says whether a
  model can operationalize a mechanism description.
