# Methodological notes

The panel, mechanisms, references and endpoints were fixed before inference and
without selection on LLM accuracy. This is not a retroactive preregistration.

## Timestamp collisions are real and retained

The productive proximity surrogates have newly coincident (dyad, timestamp)
records: hospital 399, highschool 7,541, workplace 517 and Copenhagen 146,100
(parents: 0, since proximity sources are deduplicated before shuffling). Digital
sources keep their input multiplicities as labeled records: email-EU 5,001 in the
parent and 627 in the surrogate, CollegeMsg 40/0, MathOverflow 27/0, Digg 0/0.
No record is deleted after shuffling; support, dyad multiplicities, total events,
timestamp multiset and archive endpoints pass the invariant audit
(`scripts/audit_offline.py`). Bernoulli sampling acts on individual records.

## Budgets and paired comparison

Each graph has its own T = .10 sum_e K_e and separately calibrated parameters.
Shared random numbers do not require identical p, L or H panel sizes. No main cell
is saturated or empty; all main budgets are within the 5% tolerance, so the
nominal 288 observations and 1,728 Qwen calls apply. The B retention probability
is kept to 12 significant digits, which makes every R/H/B parameter and draw
identical across machines (verified between the laptop and the cluster CPUs).

## Arm S: what the observation contains and what the reference uses

The degree-biased walk traverses dyad (u, v) in its stationary regime with
probability proportional to d_u d_v, so the distinct traversed dyads are a
degree-selected sample. The observation deliberately shows only these distinct
dyads and L, like R shows its panel dyads; the LLM, the plug-in and ExtraTrees
therefore see a degree-selected sample without the information needed to undo
the selection. The primary S reference uses the internal traversal log and
degrees (inverse-traversal weights) and is the design-aware benchmark.

Construct validity (`scripts/diagnose_walk.py`, 1,000 walks per graph): on the
real sources the degree-selection target lies 0.005–0.282 above the true rho_2
(mean +0.076); the plug-in rho_2 is biased by +0.064 at the production L and
+0.049 at 4L, while the design reference has mean bias -0.0003 at L (SD 0.025) and
-0.0005 at 4L (SD 0.013). Three real sources (and their surrogates) have several
components (CollegeMsg 4, MathOverflow 45, Digg 335); the design reference targets
the component mixture reached from a uniform start. On the 288 main observations
the S design reference has real-source MAE2 0.014 against 0.060 for the plug-in.

## Estimator interpretation and input checks

The S design reference is consistent for its component-mixture target, not
finite-sample unbiased. H is a homogeneous working model; cross-h fits use h = .60
training. B mixture-bound widening is descriptive only. Invalid LLM answers have
no imputed prediction; paired-control accuracy requires both answers valid.
Qwen token counts use the exact local chat template; DeepSeek counts message texts
and Sol uses the o200k proxy (their provider framing is verified only at a later
technical release).

## Null distribution of the productive surrogate

Among 99 further offline shuffles per parent, the productive surrogate's rho_2
lies at lower-rank fractions between .26 and .99; it was fixed beforehand and is
never selected by this diagnostic.
