# G3 Step 1: early signal slice

Prepared: **2026-09-01**  
Scope: **Codex `gpt-5.6-sol`, 1 generation(s), arms
`time_agnostic_t` and `event_sample_then_full_history`, conditions `hidden` and
`mechanism`, all 32 graphs.** 128/128 calls complete.

**This is not a go/no-go and it did not change anything.** The prompts are the
frozen set from `docs/FREEZE_2026-09.md`, verified here by per-prompt SHA-256
against the frozen file; the same cases enter the full run unchanged; and no
parameter, arm, condition, budget or metric was adjusted on what follows. Its
only purpose is to say which result the thesis leads with.

The two arms were chosen because their correct corrections point in opposite
directions -- `time_agnostic_t` needs an upward correction, arm B a downward
one -- so a model that simply shifts every estimate the same way scores zero on
the direction test rather than passing it by accident.

## Prespecified reading

**slope clearly positive** -- the language-model chapter leads: the model
moves its estimate in the case-specific correct direction when given a neutral
process description.

## The primary statistic

`delta_i = rho2_true_i - rho2_naive_i` is the correct signed correction for
case `i`. `Delta_i = rho2_model(mechanism) - rho2_model(hidden)` is the shift
the description produced. The primary statistic is the slope of `Delta_i` on
`delta_i`, with a cluster bootstrap over graph groups, because cases inside a
group share a backbone and 128 calls are not 128 independent observations.

Generations are averaged within (case, condition) **before** the pairing.

| scope                          |   slope |   ci_lo |   ci_hi |   share_positive_draws |   graph_groups |   cases |
|:-------------------------------|--------:|--------:|--------:|-----------------------:|---------------:|--------:|
| pooled (both arms)             |  0.8341 |  0.7507 |  0.9648 |                 1.0000 |             12 |      64 |
| event_sample_then_full_history |  0.6619 |  0.4119 |  0.9812 |                 1.0000 |             12 |      32 |
| time_agnostic_t                |  0.8544 |  0.7134 |  1.0039 |                 1.0000 |             12 |      32 |

## Direction and magnitude, separately

Deriving the right direction and sizing the correction are different failures
and are not collapsed into one number.

| scope                          |   cases |   direction_hit_rate |   mean_delta_i_true |   mean_Delta_i_model |   median_magnitude_ratio |   share_moved_at_all |
|:-------------------------------|--------:|---------------------:|--------------------:|---------------------:|-------------------------:|---------------------:|
| pooled (both arms)             |      64 |               0.8281 |              0.0582 |               0.0373 |                   0.8235 |               0.9219 |
| event_sample_then_full_history |      32 |               0.8125 |             -0.1631 |              -0.1537 |                   1.0735 |               1.0000 |
| time_agnostic_t                |      32 |               0.8438 |              0.2796 |               0.2282 |                   0.6868 |               0.8438 |

## The distribution of the shift, not only its mean

An effect carried by a handful of cases must be visible as one.

| scope                          |   cases |    mean |     p10 |     q25 |   median |     q75 |     p90 |   share_positive |   share_negative |
|:-------------------------------|--------:|--------:|--------:|--------:|---------:|--------:|--------:|-----------------:|-----------------:|
| pooled (both arms)             |      64 |  0.0373 | -0.2977 | -0.1182 |   0.0000 |  0.1757 |  0.4097 |           0.4688 |           0.4531 |
| event_sample_then_full_history |      32 | -0.1537 | -0.3321 | -0.2746 |  -0.1337 | -0.0192 | -0.0058 |           0.0938 |           0.9062 |
| time_agnostic_t                |      32 |  0.2282 |  0.0000 |  0.0691 |   0.1774 |  0.4005 |  0.4713 |           0.8438 |           0.0000 |

## Signed `rho_2` bias by condition

This is the axis of the main figure. The reference lines are the naive
read-off on the same fresh samples: **-0.278** on `time_agnostic_t` and
**+0.153** on `event_sample_then_full_history`.

| arm                            | condition   |   cases |   group_macro_rho2_bias |   group_macro_profile_mae |
|:-------------------------------|:------------|--------:|------------------------:|--------------------------:|
| event_sample_then_full_history | hidden      |      32 |                  0.1531 |                    0.0979 |
| event_sample_then_full_history | mechanism   |      32 |                 -0.0032 |                    0.0499 |
| time_agnostic_t                | hidden      |      32 |                 -0.2762 |                    0.1393 |
| time_agnostic_t                | mechanism   |      32 |                 -0.0437 |                    0.0577 |

## Response, validity and internal consistency

Tracked per arm and per condition. `invalid_profile_rate` is the share of
otherwise-valid answers violating `1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0`.
Nothing is repaired: a violation is recorded, never sorted.

| arm                            | condition   |   calls |   provider_refusals |   served_calls |   response_rate |   structural_completeness |   validity_rate |   invalid_profile_rate |   median_total_tokens |
|:-------------------------------|:------------|--------:|--------------------:|---------------:|----------------:|--------------------------:|----------------:|-----------------------:|----------------------:|
| event_sample_then_full_history | hidden      |      32 |                   0 |             32 |          1.0000 |                    1.0000 |          1.0000 |                 0.0000 |            47865.5000 |
| event_sample_then_full_history | mechanism   |      32 |                   0 |             32 |          1.0000 |                    1.0000 |          1.0000 |                 0.0000 |            63176.5000 |
| time_agnostic_t                | hidden      |      32 |                   0 |             32 |          1.0000 |                    1.0000 |          1.0000 |                 0.0000 |            15218.5000 |
| time_agnostic_t                | mechanism   |      32 |                   0 |             32 |          1.0000 |                    1.0000 |          1.0000 |                 0.0000 |            89104.5000 |

## What this does not show

- One model, and a product screen: the Codex harness injects instructions that
  are not part of the frozen prompt and cannot be version-pinned. Two arms and
  two conditions out of five and six.
- Prompt length is an arm-level confound. Arm B's data block is roughly three
  times the walks', so the two arms' absolute accuracy is not comparable. The
  paired contrast is within a case, where length is near-constant across the
  two conditions, so the primary statistic is unaffected.
- Nothing here separates mechanistic reasoning from a learned text heuristic.
  The design cannot, and the wording throughout stays at
  "mechanism-sensitive inference".
- `128` case-condition cells survived validity filtering out of
  `256` possible; a shift computed on a case whose
  other condition failed to parse is dropped rather than imputed.
