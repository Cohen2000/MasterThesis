#!/usr/bin/env python3
"""G1 prompt contract: the exact condition texts for the five-arm main run.

The design is a 2x2 factorial over {process described} x {bias direction
stated}, plus two controls and one robustness placebo.  The point of splitting
`mechanism` from `mechanism_direction` is that the historical `disclosed`
prompts conflated them: they described the process *and* named the direction of
the resulting error in the same breath, so a `disclosed > hidden` effect could
not distinguish "the model can derive the correction" from "the model was told
which way to move".

Acceptance criterion for every neutral `mechanism` text: a competent
statistician must be able to reconstruct `P(observed | truth)` for that arm from
the text plus the sample, *without* being told the direction of the resulting
bias.  `docs/PROMPT_CONTRACT_2026-09.md` carries a reconstruction sketch per arm
so that criterion can be checked rather than asserted.

Each arm's text is written from its own process.  Phrasing that is correct for
the walks -- observations per pair are limited, so the number of distinct active
windows seen is bounded by the number of times that pair was seen -- is simply
false for arms A and B, where retrieved histories are complete and the entire
error lives in which entities get retrieved at all.  Nothing is shared across
arms where the processes differ.  `docs/HEADROOM_G0D_2026-09.md` measures the
same point on the time axis: temporal evenness runs from 0.27 on
`time_respecting` to 0.82 on `time_agnostic_t`.

The historical `MECHANISM` dict in `make_llm_prompts_v2.py` is deliberately left
untouched.  It is the only remaining way to regenerate the frozen 420-prompt
V2.1 suite byte-identically, and `disclosed_historical` below reuses it verbatim
for exactly that reason.  The `Consequence:` lines the brief asked to remove are
removed *here*, by the neutral texts not carrying them.
"""

from __future__ import annotations

import json
import textwrap

from make_llm_prompts_v2 import DEFINITIONS, INPUT_MASK, MECHANISM, TASK

WALK_ARMS = ("time_agnostic_t", "time_respecting", "recent_history_k20")
NODE_PANEL = "node_panel_full_history"
TWO_PHASE = "event_sample_then_full_history"
ARMS = (*WALK_ARMS, NODE_PANEL, TWO_PHASE)

# Bidirectional, this pair only.  Chosen in G0d.5 as the eligible pair with the
# largest bidirectional bias penalty; its observable AUC of 0.8584 is recorded
# as a measured limitation rather than a disqualification.
MISMATCH_PAIR = ("time_agnostic_t", TWO_PHASE)

CONDITIONS = ("hidden", "direction_only", "mechanism", "mechanism_direction",
              "mismatched", "metadata_only")


# --- the neutral mechanism texts -------------------------------------------
# Same four beats in the same order in every arm: what one budget unit buys,
# how units are chosen, what one unit returns, and what `n` and the window mask
# therefore mean.  No arm names the direction of its own bias.

MECHANISM_NEUTRAL = {
    "time_agnostic_t": """SAMPLING MECHANISM
The network was explored by a walker on the time-collapsed graph, in which all
events of a pair are merged into a single edge. From its current node the
walker stepped to a uniformly random neighbour. Each traversal of an edge
returned ONE event timestamp of that pair, drawn uniformly at random from that
pair's complete event history, independently on each traversal and with
replacement. Every step cost one budget unit and the walk continued until the
budget was spent. A pair appears in the data below only if the walker traversed
it, and n is the number of times it was traversed. The window mask shown for a
pair is the set of windows containing the n timestamps that were drawn for that
pair.""",

    "time_respecting": """SAMPLING MECHANISM
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
events.""",

    "recent_history_k20": """SAMPLING MECHANISM
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
mask shown is the set of windows containing those n recorded events.""",

    NODE_PANEL: """SAMPLING MECHANISM
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
windows.""",

    TWO_PHASE: """SAMPLING MECHANISM
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
network and the window mask shown is that pair's true set of active windows.""",
}


# --- direction sentences ---------------------------------------------------
# One sentence, one template, the only moving part being the direction word.
# Directions come from the measured eight-slot naive bias in G0d: the three
# walks underestimate, arm B overestimates, and arm A is approximately
# unbiased, which is written as such rather than skipped.

_DIRECTION_SENTENCE = (
    "For this access process, the share of pairs with K >= k computed directly "
    "from the pairs shown below is on average {} the corresponding share for "
    "the full network.")


def _direction_text(word: str) -> str:
    """One sentence, one template, wrapped identically for every arm.

    The wrap is applied after substitution so that a longer direction word
    cannot change the line layout, which would otherwise make prose shape a
    silent function of the arm.
    """
    body = textwrap.fill(_DIRECTION_SENTENCE.format(word), width=78)
    return f"DIRECTION OF THE SAMPLING BIAS\n{body}"


_DIRECTION_WORD = {
    "time_agnostic_t": "an underestimate of",
    "time_respecting": "an underestimate of",
    "recent_history_k20": "an underestimate of",
    NODE_PANEL: "approximately equal to",
    TWO_PHASE: "an overestimate of",
}

DIRECTION_ONLY = {arm: _direction_text(word)
                  for arm, word in _DIRECTION_WORD.items()}

MECHANISM_DIRECTION = {
    arm: f"{MECHANISM_NEUTRAL[arm]}\n\n{DIRECTION_ONLY[arm]}"
    for arm in ARMS
}

# Byte-identical historical `disclosed` text, kept as an optional bridge
# condition.  G1.4 asked that the three walks' `mechanism_direction` be the
# historical prompt unchanged so the historical comparison survives; that
# conflicts with a clean 2x2, because `time_agnostic_t`'s historical text names
# no direction at all and would make `mechanism` and `mechanism_direction`
# identical for that arm.  The contract document explains the choice: the
# factorial is built compositionally above, and the historical text is offered
# here so the bridge can still be run byte-exactly if wanted.
DISCLOSED_HISTORICAL = {arm: MECHANISM[arm] for arm in WALK_ARMS}

HIDDEN = """SAMPLING MECHANISM
Not disclosed. The data below was collected by a budget-limited sampling
process on the network; the process is not described."""

# Same register and length band as a `mechanism` text, content deliberately
# irrelevant to the estimation.  Guards against "more structured text makes the
# model compute more carefully" as a rival explanation for a
# `mechanism > hidden` effect.  Robustness subset only.
IRRELEVANT_CONTEXT = """DATA HANDLING NOTES
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
pairs or which events are present in the data below."""

# `metadata_only` has no sample, so the `hidden` wording ("the data below")
# would be false.  It gets its own neutral line naming neither process nor
# direction.  This is the "metadata plus learned prior" anchor, never a
# no-information baseline: size, event count and span are informative.
METADATA_CONTEXT = """SAMPLING MECHANISM
Not disclosed, and no sampling was carried out for this network."""

METADATA_HEADER = """NO SAMPLE WAS COLLECTED
No pairs were observed for this network. You are given only the summary facts
below about the FULL network, and must estimate the targets from them together
with whatever you know about temporal networks of this kind."""

OBSERVATION_HEADER = "NOW THE ACTUAL OBSERVATION TO EVALUATE:"


def metadata_block(row) -> str:
    """Graph-level facts only, so every arm renders the same text."""
    return json.dumps({
        "n_nodes": int(row["n_nodes_true"]),
        "n_events": int(row["n_events_true"]),
        "observation_period": [0, 1],
        "windows": 5,
    }, separators=(",", ":"))


def context_block(arm: str, condition: str, stated_arm: str | None = None) -> str:
    """The one block that varies across conditions, for a given arm."""
    if condition == "hidden":
        return HIDDEN
    if condition == "direction_only":
        return f"{HIDDEN}\n\n{DIRECTION_ONLY[arm]}"
    if condition == "mechanism":
        return MECHANISM_NEUTRAL[arm]
    if condition == "mechanism_direction":
        return MECHANISM_DIRECTION[arm]
    if condition == "mismatched":
        if stated_arm is None:
            raise ValueError("mismatched needs the stated (wrong) arm")
        if {arm, stated_arm} != set(MISMATCH_PAIR):
            raise ValueError(
                f"mismatched runs on {MISMATCH_PAIR} only, got {(arm, stated_arm)}")
        return MECHANISM_NEUTRAL[stated_arm]
    if condition == "metadata_only":
        return METADATA_CONTEXT
    if condition == "irrelevant_context":
        return IRRELEVANT_CONTEXT
    if condition == "disclosed_historical":
        return DISCLOSED_HISTORICAL[arm]
    raise ValueError(f"unknown condition {condition!r}")


def build_prompt(row, condition: str, stated_arm: str | None = None) -> str:
    """Assemble one prompt.

    Section order is identical in every condition and every arm, so arm and
    condition are never confounded with layout: definitions, then the context
    block, then the observation, then the task.
    """
    arm = str(row["strategy"])
    parts = [DEFINITIONS, context_block(arm, condition, stated_arm)]
    if condition == "metadata_only":
        parts += [METADATA_HEADER, metadata_block(row)]
    else:
        parts += [OBSERVATION_HEADER, INPUT_MASK,
                  str(row["input__nmask_exact_json"])]
    parts.append(TASK)
    return "\n\n".join(parts)
