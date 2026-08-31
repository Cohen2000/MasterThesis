#!/usr/bin/env python3
"""Emit docs/PROMPT_CONTRACT_2026-09.md from the G1 texts themselves.

The document is generated from `prompt_contract_g1` rather than written beside
it, so the texts in the contract and the texts G2 will render cannot drift.
"""

from __future__ import annotations

import argparse
import hashlib
import textwrap
from pathlib import Path

import pandas as pd

import prompt_contract_g1 as C
from report_g0b_headroom import distribution, portable_token_count, read_globs
from report_g0d_headroom import NODE_PANEL, TWO_PHASE

# Measured on the G0c cases, where both the exact Qwen3.6 count and the same
# rendered text are stored.  Decision 4: exact counts replace this before G3.
QWEN_RATIO = 1.6637

ARM_LABEL = {
    "time_agnostic_t": "walk, time-agnostic",
    "time_respecting": "walk, forward in time",
    "recent_history_k20": "walk, backward in time, k=20",
    NODE_PANEL: "arm A, uniform node panel, complete histories",
    TWO_PHASE: "arm B, event sample then complete histories",
}

# Why each text meets the acceptance criterion, written so a reviewer can check
# it rather than take it on trust.  Each sketch derives P(observed | truth) from
# the text alone and stops *before* naming the direction of the rho_k error.
RECONSTRUCTION = {
    "time_agnostic_t": """A pair is in the sample iff the walk traversed it, and the traversal count n
is the pair's share of random-walk traffic on the collapsed graph. Given n, the
text states that each traversal draws one event uniformly and independently
from the pair's *complete* history. So the recorded timestamps are n iid draws
from that pair's event-time distribution, and the observed window mask is the
set of windows those n draws landed in. `P(mask | K, n)` is therefore a
classical occupancy problem over the pair's per-window event shares, and
`P(observed | truth)` factorizes into traffic-driven inclusion times that
occupancy term. Nothing in the text says which way the resulting error points.""",

    "time_respecting": """A pair is in the sample iff the walk traversed it, and each traversal records
exactly the one event moved across. The text states the eligibility rule
explicitly: at clock t only events strictly later than t can be chosen. So the
set of a pair's events that could ever be recorded on a given visit is
determined by the arrival clock, and the restart rule (uniform node, uniform
clock in [0,1)) fixes the distribution of arrival clocks. `P(observed | truth)`
is inclusion by forward-time reachability times, for an included pair, the
distribution of which of its events were reachable given the arrival clock.
The direction of the resulting error is not stated.""",

    "recent_history_k20": """A pair is in the sample iff the walk traversed it, and each traversal records
exactly the one event moved across. The eligibility rule is stated: only events
strictly earlier than the clock, and only the 20 latest such events at that
node, are candidates, one chosen uniformly. The clock starts at the end of the
period and restarts reset it there. So `P(observed | truth)` is inclusion by
backward-time reachability under a 20-event retrieval window, times the uniform
choice within that window. The truncation at 20 and the backward direction are
both stated as process; neither the text nor the sample is told which way the
resulting error points.""",

    NODE_PANEL: """The text states that nodes are drawn uniformly without replacement, that a
recruited node returns the complete history of every incident pair, and that a
pair is in the sample iff at least one endpoint was recruited. For a panel of p
nodes out of N, that inclusion probability is 1 - C(N-p, 2)/C(N, 2), which does
not depend on the pair. The text also states that for an included pair, n and
the window mask are the pair's true values. So `P(observed | truth)` is a
pair-independent inclusion probability times a point mass on the true mask --
no censoring and no pair-dependent selection. A statistician can read off that
the plug-in estimate is unbiased, but the text never says so; the stopping rule
is disclosed as the one thing that makes p a stopping time rather than fixed.""",

    TWO_PHASE: """The text states that event *records* are examined in uniformly random order,
that a pair is retrieved the first time one of its own records is reached, and
that the retrieved history is complete. A pair owning m of the M records is
therefore reached in proportion to m: its inclusion probability is an
increasing function of its own event count, derivable directly from the stated
uniform-random order over records. The text also states that for an included
pair, n and the window mask are the pair's true values. So `P(observed | truth)`
is activity-proportional inclusion times a point mass on the true mask -- no
censoring, strong pair-dependent selection. The direction the resulting error
points is a further step the text does not take.""",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _tokens(text: str) -> tuple[int, int]:
    portable = portable_token_count(text)
    return portable, round(portable * QWEN_RATIO)


def block_table() -> pd.DataFrame:
    """Every condition text, with its length and a hash to pin it."""
    rows = []
    for arm in C.ARMS:
        blocks = {
            "hidden": C.HIDDEN,
            "direction_only": f"{C.HIDDEN}\n\n{C.DIRECTION_ONLY[arm]}",
            "mechanism": C.MECHANISM_NEUTRAL[arm],
            "mechanism_direction": C.MECHANISM_DIRECTION[arm],
            "metadata_only": C.METADATA_CONTEXT,
        }
        if arm in C.MISMATCH_PAIR:
            other = [a for a in C.MISMATCH_PAIR if a != arm][0]
            blocks["mismatched"] = C.MECHANISM_NEUTRAL[other]
        if arm in C.WALK_ARMS:
            blocks["disclosed_historical"] = C.DISCLOSED_HISTORICAL[arm]
        blocks["irrelevant_context"] = C.IRRELEVANT_CONTEXT
        for condition, text in blocks.items():
            portable, qwen = _tokens(text)
            rows.append({
                "arm": arm, "condition": condition,
                "words": len(text.split()), "characters": len(text),
                "portable_tokens": portable, "qwen36_tokens_est": qwen,
                "sha256_16": _sha(text),
            })
    return pd.DataFrame(rows)


def full_prompt_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Whole-prompt length per arm and condition on the real G0d samples."""
    rows = []
    for arm, part in panel.groupby("strategy"):
        for condition in (*C.CONDITIONS, "irrelevant_context"):
            if condition == "mismatched" and arm not in C.MISMATCH_PAIR:
                continue
            stated = None
            if condition == "mismatched":
                stated = [a for a in C.MISMATCH_PAIR if a != arm][0]
            lengths = [portable_token_count(
                C.build_prompt(row, condition, stated))
                for row in part.to_dict("records")]
            stats = distribution(pd.Series(lengths))
            rows.append({
                "arm": arm, "condition": condition, "cases": len(part),
                "median_portable": round(stats["median"]),
                "p90_portable": round(stats["p90"]),
                "median_qwen_est": round(stats["median"] * QWEN_RATIO),
                "max_qwen_est": round(stats["max"] * QWEN_RATIO),
            })
    return pd.DataFrame(rows)


def mismatch_assignment(panel: pd.DataFrame) -> pd.DataFrame:
    a, b = C.MISMATCH_PAIR
    rows = []
    for actual, stated in ((a, b), (b, a)):
        cases = int((panel.strategy == actual).sum())
        rows.append({
            "actual_arm": actual, "stated_arm": stated,
            "direction_implied_by_stated_text": (
                "downward" if stated == b else "upward"),
            "direction_implied_by_actual_process": (
                "downward" if actual == b else "upward"),
            "cases": cases,
        })
    return pd.DataFrame(rows)


def metadata_identity(panel: pd.DataFrame) -> tuple[bool, int]:
    """Do the arms render the same metadata_only prompt for a graph?"""
    rendered = {}
    for row in panel.to_dict("records"):
        rendered.setdefault(row["instance_id"], set()).add(
            C.build_prompt(row, "metadata_only"))
    identical = all(len(v) == 1 for v in rendered.values())
    return identical, len(rendered)


def _fence(text: str) -> str:
    return "```text\n" + text + "\n```"


def build_document(*, blocks, prompts, assignment, identical, graphs,
                   length_band, panel_n) -> str:
    per_arm = "\n\n".join(
        f"#### `{arm}` — {ARM_LABEL[arm]}\n\n"
        f"{_fence(C.MECHANISM_NEUTRAL[arm])}\n\n"
        "*Reconstruction sketch.*\n"
        + textwrap.fill(" ".join(RECONSTRUCTION[arm].split()), width=78)
        for arm in C.ARMS)
    directions = "\n\n".join(
        f"#### `{arm}`\n\n{_fence(C.DIRECTION_ONLY[arm])}" for arm in C.ARMS)
    return f"""# G1 prompt contract

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

`mismatched` runs on `{C.MISMATCH_PAIR[0]}` <->
`{C.MISMATCH_PAIR[1]}` only, bidirectional.
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

{per_arm}

## `direction_only`

One sentence, one template across all five arms, naming only the direction and
describing no part of the process. The wrap is applied after substitution so a
longer direction word cannot change the line layout. Directions are the
measured eight-slot naive signs from G0d. Arm A is **not** skipped: its correct
statement is that the naive estimate is approximately unbiased, and saying so is
as much a directional claim as the other four.

{directions}

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

{_fence(C.HIDDEN)}

`metadata_only` gets its own context line, because `hidden`'s wording refers to
"the data below" and there is no data below:

{_fence(C.METADATA_CONTEXT)}

{_fence(C.METADATA_HEADER)}

The metadata block is graph-level only — node count, event count, observation
period, window count — so **all five arms render an identical `metadata_only`
prompt for a given graph**: verified {identical} across {graphs} graphs.
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

{_fence(C.IRRELEVANT_CONTEXT)}

## `mismatched` assignment

{_format(assignment)}

Bidirectional on this pair only. The two arms need corrections in opposite
directions, which is what makes the outcome space three-way and the reading
sharp — see the interpretation table pre-registered in
`docs/HEADROOM_G0D_2026-09.md` G0d.5 and in the G4 plan.

## Token counts

Context blocks alone:

{_format(blocks)}

Whole prompts, rendered on the {panel_n} G0d panel samples:

{_format(prompts)}

**Qwen figures are estimates**, converted from the portable count at the
measured G0c ratio {QWEN_RATIO}. Decision 4 requires exact counts from the
BWUniCluster tokenizer before G3; until then no Qwen number here is to be
quoted as measured.

## Length band

{length_band}

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
"""


def _format(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", nargs="+", default=[
        "results/g0d_headroom_2026_09/accepted_panel_cases.csv.gz"])
    parser.add_argument("--walk-panel",
                        default="results/panel_seed_probe/cases.csv.gz")
    parser.add_argument("--out", default="docs/PROMPT_CONTRACT_2026-09.md")
    parser.add_argument("--summary-dir", default="results_summary/g1")
    args = parser.parse_args()

    accepted = read_globs(args.panel)
    accepted = accepted[accepted.seed_slot == 0]
    walk = pd.read_csv(args.walk_panel)
    walk = walk[(walk.budget == 800) & (walk.walk_seed == 0) &
                walk.strategy.isin(C.WALK_ARMS)]
    panel = pd.concat([walk, accepted], ignore_index=True)

    blocks = block_table()
    prompts = full_prompt_table(panel)
    assignment = mismatch_assignment(panel)
    identical, graphs = metadata_identity(panel)

    mech = blocks[blocks.condition == "mechanism"]
    low, high = int(mech.words.min()), int(mech.words.max())
    placebo = int(blocks[blocks.condition == "irrelevant_context"].words.iloc[0])
    where = ("inside that band" if low <= placebo <= high else
             f"OUTSIDE that band, at {placebo} words")
    band = (f"The five neutral `mechanism` texts span **{low}-{high} words** "
            f"and **{int(mech.portable_tokens.min())}-"
            f"{int(mech.portable_tokens.max())} portable tokens**. Same four "
            f"beats in the same order in every arm: what one budget unit buys, "
            f"how units are chosen, what one unit returns, and what `n` and the "
            f"window mask therefore mean. The placebo sits {where} at "
            f"{placebo} words, which is what lets it control for prose bulk "
            f"rather than introduce a length difference of its own.")

    summary = Path(args.summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(summary / "condition_block_tokens.csv", index=False)
    prompts.to_csv(summary / "full_prompt_tokens.csv", index=False)
    assignment.to_csv(summary / "mismatch_assignment.csv", index=False)

    Path(args.out).write_text(build_document(
        blocks=blocks, prompts=prompts, assignment=assignment,
        identical=("identical" if identical else "NOT identical"),
        graphs=graphs, length_band=textwrap.fill(band, width=78),
        panel_n=len(panel)))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
