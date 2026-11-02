# E8r: Norm-layer leakage replication

**Status:** done  ·  **Outcome:** not replicated under the registered rule; falsified

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
| 2026-11-02 | `e8r-leaky-replication` | 6/6 finished, return code 0, about 1.0 h each. No run stuck at chance |

## Result
| Seed | LN (E8b) BACC | IN (E8c) BACC | LN minus IN |
|---|---|---|---|
| 45 | 97.17 | 97.21 | -.04 |
| 46 | 97.19 | 97.04 | +.15 |
| 47 | 97.21 | 96.86 | +.34 |
| **Mean** | 97.19 | 97.04 | **+.15 [-.05, .36]** |

Secondary: gap difference against E8's grouped values +.22, compared with E8's +.61.

| Prediction | Outcome |
|---|---|
| 1. Mean about +.3, range +.1 to +.6 | Inside the range, below the central value |
| 2. Positive on at least 2 of 3 seeds | **Supported:** 2 of 3 |
| Decision rule | **Not replicated:** the CI includes 0 |

**Falsified** under the registered condition, because the rule returned "not replicated".

## Verdict and what we learned
**The claim that LayerNorm gains more from leakage than InstanceNorm is not confirmed, and goes back to exploratory.** E8's difference of +.54 on seeds 42-44 fell to +.15 on new seeds. LN's leaky score was stable (97.34 to 97.19), and most of the fall comes from IN moving up
(96.80 to 97.04). The result does not show the effect is zero: the CI runs to +.36, which is above the MDE, so a modest effect remains possible. Three seeds on each side cannot settle that. E4b's reading, which assumed an LN leakage effect on one seed, is weakened along with it.
