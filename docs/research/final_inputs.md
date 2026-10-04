# Final inputs — which input set does the main model use?

Run `poc_long` (train 2011 – 2021, validate 2022, test 2023 and 2024). R3 had found, on five training years, that dropping
salinity and currents improved the Transformer by 0.027 °C. That had not been tested on eleven years, so before the final
run the headline model (CNN + Transformer, trained from scratch) was retrained with reduced input sets, three seeds each, and
the input set was chosen **on the validation year only**. Produced by `oceanembed research final-inputs` and
`final-inputs-report --config configs/poc_long.yaml`; full tables in `outputs/poc_long/research/final_inputs/summary.md`.

## Candidates

| Input set | Inputs |
|---|---|
| All seven inputs (the R2 model, same seeds) | SST, salinity, sea level, currents, winds |
| All but salinity and currents (= SST + sea level + winds: the same set) | SST, sea level, winds |
| SST + sea level | SST, sea level |

A dropped group is zeroed in every input channel (the training mean), as in R3.

## The rule

A reduced set is adopted only if its mean validation-year RMSE (the pooled RMSE over all depths of the checkpoint early
stopping kept) is lower than the full set's by more than the **seed spread**, the larger of the two sample standard deviations
over the three seeds. The lowest adopted set wins; otherwise all seven inputs are kept. No test-year number enters the rule.

## Validation year (2022)

| Input set | Val. RMSE per seed (°C) | Mean | SD | Gain vs full | Seed spread | Adopted |
|---|---|---|---|---|---|---|
| All seven inputs | 0.6996, 0.6967, 0.7003 | 0.6988 | 0.0019 | – | – | – |
| SST, sea level, winds | 0.6897, 0.7032, 0.6991 | 0.6973 | 0.0069 | +0.0015 | 0.0069 | no |
| **SST + sea level** | 0.6874, 0.6964, 0.6924 | **0.6921** | 0.0045 | +0.0068 | 0.0045 | **yes** |

**Decision: SST and sea level only.** The margin is small (0.0068 against a spread of 0.0045 °C).

## Test years (reported after the decision, for every candidate)

RMSE over 50 – 200 m against GLORYS, mean ± SD over three seeds:

| Input set | 2023 | 2024 | Both years | Arabian Sea (both) | Bay of Bengal (both) |
|---|---|---|---|---|---|
| All seven inputs | 0.975 ± 0.007 | 1.004 ± 0.014 | 0.989 ± 0.010 | 1.018 ± 0.018 | 0.938 ± 0.007 |
| SST, sea level, winds | 0.970 ± 0.008 | 1.010 ± 0.016 | 0.990 ± 0.005 | 1.015 ± 0.006 | 0.947 ± 0.004 |
| SST + sea level | 0.970 ± 0.009 | 1.002 ± 0.003 | 0.986 ± 0.003 | 1.010 ± 0.004 | 0.943 ± 0.002 |

SST + sea level minus all seven inputs (paired block bootstrap, 95 % interval): −0.004 [−0.012, +0.005] in 2023, −0.002
[−0.007, +0.006] in 2024, −0.003 [−0.007, +0.004] pooled; Arabian Sea pooled −0.008 [−0.015, −0.001]; Bay of Bengal pooled
+0.005 [−0.006, +0.027].

## What this means

- With eleven training years the extra inputs are **not needed**: the two-input model is as good as the seven-input one on
  both test years (differences within the uncertainty, except a small gain of 0.008 °C in the Arabian Sea), and it needs only the two best-established,
  lowest-latency satellite products. The five-year finding (a gain from dropping inputs) is therefore a small-data effect
  that shrinks to a tie with more data.
- Adding winds to SST + sea level does not help (validation gain +0.0015 °C, inside the spread).
- The seed spread is as large as the differences being decided (SD 0.002 – 0.007 °C on validation): the choice of two inputs is
  defensible by the stated rule, but equally defensible as "the inputs beyond SST and sea level make no measurable
  difference". Both statements are true.

## Limits

- Three seeds per set; the rule uses the validation RMSE pooled over all depths (the thermocline-only validation means, in
  `summary.md`, rank the sets the same way).
- Zeroing a group is "replaceable", not "carries no information" (as in R3).
- Scores are against GLORYS; the 2022 validation year also selected the checkpoints, so validation numbers are slightly optimistic for every set alike.
