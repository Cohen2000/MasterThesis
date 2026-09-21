# Methodological audit notes

The final panel and all primary endpoints were fixed before new Qwen inference.
The design choices do not depend on comparative new model accuracy. This is a
prospective revision of development work, not a retroactive preregistration.

## Timestamp collisions are real and retained

The productive proximity surrogates have newly coincident (dyad,timestamp)
records: hospital 399, highschool 7,541, workplace 517 and Copenhagen 146,100
excess records relative to unique dyad/timestamp keys (parents: 0, since
proximity sources are deduplicated before shuffling). Digital sources keep their
input multiplicities as labeled records: email-EU has 5,001 coincident records
in the parent and 627 in the surrogate, CollegeMsg 40/0, MathOverflow 27/0,
Digg 0/0. No record is deleted after shuffling (scripts/audit_offline.py). Support, dyad multiplicities, total events, global timestamps and
archive endpoints all pass the invariant audit. Bernoulli sampling acts on
individual records, including coincident ones. This is a timestamp-shuffled
event multiset, not a deduplicated binary contact sequence.

## Budgets and paired comparison

Each graph has its own T=.10*sum K and separately calibrated parameters. A
surrogate can have different persistence and active-cell volume without changing
support or event counts. Shared random numbers do not require identical p, L or
H panel sizes. No productive main cell is saturated or empty; all current main
budgets satisfy the inherited tolerance. The nominal 288 observations and
1,728 Qwen calls therefore apply without forced duplication.

## Null distribution of the productive surrogate

Among 99 further offline shuffles per parent, the productive surrogate's rho_2
lies at lower-rank fractions between .26 and .99 (no systematic extreme
position); it was fixed before and never selected by this diagnostic.

## Estimator interpretation and input checks

The inherited SRW reference remains a stationary/asymptotic working reference;
finite-walk, component and mixing effects are reported. H remains a homogeneous
working model; cross-h fits use h=.60 training. B mixture-bound widening is
strictly descriptive and cannot choose a new primary fit. Invalid LLM answers
have no imputed prediction. Paired-control accuracy requires both answers valid.

Qwen token counts use the exact local tokenizer chat template. DeepSeek counts
message texts and Sol uses the inherited o200k tokenizer proxy; provider framing
remains unverified until their later technical release. This does not release
or start either API. Requests and repeats already match Qwen's observations.
