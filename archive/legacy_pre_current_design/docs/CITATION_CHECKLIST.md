# Citation checklist

**Rule: nothing enters the bibliography before the full text has been read by a
human and the claim attributed to it has been located in that text.**

The trigger for this file was a set of externally generated reading suggestions
— CRAFT (2025), TGPM (2026), TEACUPS (2024), Online-TNS (2021), and a Miritello
figure of "30–40% spurious decay" — none of which has been verified. Output from
NotebookLM, GPT, or any other model is a *pointer to look something up*, never a
citation. A plausible-looking reference with authors, venue and year is exactly
what a language model produces when it has nothing.

## Status values

| status | meaning |
|---|---|
| `verified` | full text read; the specific claim located; page or section noted |
| `exists-unread` | the work provably exists, but the claim has not been checked against it |
| `unverified` | provenance is a model suggestion or second-hand; existence not established |
| `rejected` | checked and does not support the claim, or does not exist |

A citation may be used in the thesis only at `verified`.

## Claims that currently carry weight in the design

| # | claim as it would appear | source | status | who checks |
|---|---|---|---|---|
| 1 | Four sampling strategies TS / NS / ES / RS; NS and ES estimate mean link lifetime well, TS underestimates, RS overestimates | Rocha, Masuda, Holme 2017, PRE 96, 052302 | `exists-unread` | — |
| 2 | CTDNE time-respecting walks sample the next event uniformly among strictly later events | Nguyen et al. 2018 | `exists-unread` | — |
| 3 | Greedy temporal walks terminate at a temporal dead end | Saramäki & Holme 2015 | `exists-unread` | — |
| 4 | Burstiness measure applied to node pairs | Holme & Saramäki 2012; Karsai et al. 2011 | `exists-unread` | — |
| 5 | Non-monotonic temporal walks | Ma et al. 2026 | `unverified` | — |
| 6 | "30–40% spurious decay" from sampling | Miritello et al. | `unverified` | — |
| 7 | CRAFT (2025) | — | `unverified` | — |
| 8 | TGPM (2026) | — | `unverified` | — |
| 9 | TEACUPS (2024) | — | `unverified` | — |
| 10 | Online-TNS (2021) | — | `unverified` | — |

Items 2, 3 and 4 are cited in code comments (`src/walks.py`, `src/census.py`,
`src/generator.py`) that describe implementation choices. They are marked
`exists-unread` rather than `verified` because being long-standing in the
codebase is not the same as having been checked; the implementations they
justify are correct on their own terms either way.

Item 5 appears in `src/walks.py` with a year of 2026 for a design decision made
earlier. That is worth checking before it reaches the thesis.

Items 7–10 have no author, no venue and no located claim. They are listed so
they are not silently reintroduced.

## Rocha, Masuda & Holme 2017 — attribution, drafted for the text

This is the closest prior work and the paragraph is drafted in
`docs/SAMPLING_RATIONALE.md`. It is written from the four strategy names and the
lifetime findings as supplied, which are **not yet verified against the paper**.
The correspondence claims — `node_panel_full_history` ~ NS, arm B a two-stage
extension that is *not* ES — are ours and follow from the mechanism definitions,
so they stand or fall on our own descriptions rather than on the paper. The
*direction* claims about L_s do depend on the paper and must reach `verified`
before publication.

The transferability caveat in that section is independent of verification: it
follows from the definitions. Rocha's L_s is `t_last - t_first` over links with
at least two events; ours is `rho_k = P(K >= k)`. Bias directions do not carry
across.

## Process

1. Obtain the full text. Record the DOI.
2. Locate the specific sentence, figure or table supporting the claim.
3. Record the page or section in the table above and set `verified`.
4. If the claim is not there, set `rejected` and remove it from the text — do
   not weaken the wording and keep the citation.
