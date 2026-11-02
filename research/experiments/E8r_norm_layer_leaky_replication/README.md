# E8r: Norm-layer leakage replication

**Status:** registered 2026-11-02, not yet run; first to be dropped if quota is short  ·  **Outcome:** pending

## Question
Does LayerNorm's advantage over InstanceNorm on leaky splits replicate on new seeds?

## Why this exists
E8's LN-IN result was exploratory: no prediction was written, E8c was added after E8a failed, and it used 3 leaky seeds.
It carries one of the audit's main claims, so it needs a test with a prediction written first. E4b is one seed and cannot
fill that role.

## Prediction (written by the AI assistant, 2026-11-02, before any run)
**What the writer had already seen:** E8's leaky BACC of 97.34 (E8b) and 96.80 (E8c) on seeds 42-44, a difference of
+.54. **Not seen:** anything on seeds 45-47.

1. **Primary:** mean paired LN-IN leaky BACC on seeds 45-47 stays positive but shrinks to about +.3, plausible range
   +.1 to +.6, as a selected result regresses.
2. **Secondary:** the same difference is positive on at least 2 of 3 seeds.

**Decision rule, fixed now:**
- **Replicated:** mean at least +.29 (E1's MDE), with the 95% CI above 0.
- **Direction only:** CI above 0, mean under +.29.
- **Not replicated:** CI includes 0. The claim that LN gains from leakage then goes back to exploratory.

**Falsified if:** the rule returns "not replicated", or the mean falls outside +.1 to +.6.

## Protocol
- **Runs:** `E8_paper_faithful` (E8b, LN) and `E8c_paper_instancenorm` (E8c, IN) on leaky seeds 45-47 (6 runs).
- **Split:** new file `splits/leaky_splits_replication.csv`, built by E0's function. The frozen split files are unchanged.
- **Primary metric:** mean over seeds of LN-IN test BACC. Each seed's arms share test cells, so the comparison is paired.
- **CI:** paired patient-cluster bootstrap, 2,000 draws. This replaces E8's approximate independent bootstraps.
- **Departure from E8's stated follow-up:** grouped repeats 1-2 are cut for time, so the grouped side stays at repeat 0.
  The gap difference is secondary and uses E8's grouped values.

## Runs
| Date | What | Notes |
|---|---|---|

## Result
Pending.

## Verdict and what we learned
Pending.
