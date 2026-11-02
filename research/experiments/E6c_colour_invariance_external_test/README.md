# E6c: Does colour invariance close the external gap?

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** falsified; training-time colour transfer passes the rule, test-time normalisation harms

## Question
How much of the MP-IDB gap can colour invariance recover, at training time or at test time?

## Why this exists
E4b's registry row listed "external SP" as a metric, but its 15 checkpoints were never scored on MP-IDB. This is
also the gate for the Fourier-JEPA idea in `ideas.txt`: that idea assumes stain invariance helps outside NIH.
E4 and E4b showed no benefit on unseen NIH patients, so MP-IDB is the only place the premise can be tested.

## Prediction (written by the AI assistant, 2026-11-02, before any scoring)
**What the writer had already seen:** E6's fold-0 results, E4b's NIH results, and contact sheets showing
MP-IDB is more yellow-brown than NIH. **Not seen:** any score from these conditions.

Three conditions, each on 3 arms x 5 folds:
- **A, baseline:** E1/E8b/E8c checkpoints, unmodified crops (reused from E6b).
- **B, training-time:** E4b colour-transfer checkpoints, unmodified crops.
- **C, test-time:** baseline checkpoints, crops Reinhard-normalised to NIH colour statistics.

Predictions, pooled across the three arms:
1. **B-A is small:** AUC change below +0.02, BACC change within +/-2. E4b only transferred colour between NIH
   patients, and MP-IDB's stain looks outside that range.
2. **C-A is positive and larger:** AUC +0.01 to +0.05, BACC +2 to +6, mainly through sensitivity (+5 to +15 SE)
   at some cost to specificity.
3. **Colour is the minority of the gap:** condition C closes under a third of the gap to each arm's NIH level,
   and pooled BACC stays below 88. MP-IDB also differs in stage mix (1,230 of 1,297 parasites are rings) and
   crop pipeline (Cellpose).

**Decision rule, fixed now:**
- Colour invariance "passes" only if B-A or C-A reaches a pooled AUC gain of at least +0.02 with a hierarchical
  95% CI above 0.
- If neither passes, Fourier-JEPA and other colour-invariance remedies are rejected for MP-IDB.
- If one passes, the next step is a plain supervised Fourier-augmentation screen, not JEPA.
- If only C passes, the likely reading is that MP-IDB lies outside NIH's colour range.

**Falsified if:** prediction 1 fails (B-A at or above +0.02 AUC), or C closes at least a third of the gap.

## Protocol
- **B:** `E4b_{e1,e8b,e8c}_colourtransfer` grouped r0 f0-4, paired with A by arm and fold.
- **C, source stats:** foreground LAB mean and std pooled over all crops from the same MP-IDB field image. Field
  level, so a parasite's own colour does not drive its own crop's correction. Uses no labels.
- **C, target stats:** mean of `patient_lab_stats` over that fold's non-test NIH patients, the same population
  and code path as E4b's targets. No test-patient colour is used.
- **C, transform:** `colour.reinhard_transfer`, foreground only, background re-zeroed, at native resolution
  before the standard resize to 224.
- **Metrics, threshold, CI method and pairing:** identical to E6b.
- **Multiplicity:** the pooled AUC changes are primary. Per-arm results are secondary and do not trigger the rule.
- Condition C is transductive in the mild sense that it uses unlabelled MP-IDB colour statistics. Disclosed.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-02 | (local) `score_external.py --cond B`, `--cond C` | 30 checkpoint passes |
| 2026-11-02 | (local) `analysis.py` | Registered analysis, B=2000 |
| 2026-11-02 | (local) `score_external.py --cond D`, `sensitivity_condition_d.py` | Unregistered sensitivity rerun of C, see below |

## Result
**Mean over five folds on MP-IDB:**

| Condition | Arm | BACC | SE | SP | AUC |
|---|---|---|---|---|---|
| A, baseline | E1 / E8b / E8c | 72.8 / 74.4 / 70.6 | 48.2 / 51.9 / 44.1 | 97.3 / 96.9 / 97.1 | 0.888 / 0.881 / 0.884 |
| B, training-time | E1 / E8b / E8c | 82.1 / 79.4 / 77.4 | 65.6 / 59.9 / 56.9 | 98.6 / 99.0 / 98.0 | 0.952 / 0.950 / 0.930 |
| C, test-time | E1 / E8b / E8c | 54.5 / 54.7 / 52.3 | 9.6 / 9.7 / 5.2 | 99.4 / 99.7 / 99.4 | 0.738 / 0.743 / 0.733 |

**Pooled change, primary (hierarchical 95% CI):**

| Change | BACC | SE | SP | AUC |
|---|---|---|---|---|
| B-A | +7.05 [4.35, 9.65] | +12.69 [6.58, 18.23] | +1.42 [0.24, 2.71] | **+0.059 [0.036, 0.088]** |
| C-A | -18.75 [-22.73, -15.03] | -39.89 [-47.97, -32.09] | +2.40 [1.48, 3.20] | **-0.147 [-0.184, -0.112]** |

Per-arm B-A AUC, secondary: E1 +0.064 [0.031, 0.099], E8b +0.068 [0.047, 0.093], E8c +0.046 [0.012, 0.088].

**Prediction scorecard:**

| # | Prediction | Result |
|---|---|---|
| 1 | B-A small: AUC under +0.02, BACC within +/-2 | **Falsified:** +0.059 AUC, +7.05 BACC |
| 2 | C-A positive: AUC +0.01 to +0.05, BACC +2 to +6 | **Falsified:** strongly negative, and still negative in the sensitivity rerun |
| 3 | C closes under a third of the gap, BACC below 88 | Met only trivially, because C widened the gap. B closes 29% of it, with pooled BACC 79.6 |

## Verdict and what we learned
**Falsified under the registered condition (prediction 1). By the decision rule, colour invariance passes through
condition B, and C does not.** Training with colour transfer between NIH patients helps a lot on MP-IDB in all three
arms, with no MP-IDB data involved. That contradicts the reasoning behind prediction 1, that NIH-range recolouring
could not reach MP-IDB's stain. It also revises E4/E4b's reading: colour augmentation showed no benefit on unseen NIH
patients, but does outside NIH. Per the rule, the next colour step would be a supervised Fourier-augmentation screen,
not JEPA.

**Test-time normalisation harmed all three arms, and a protocol flaw explains only a small part of that.** The registered
protocol paired a field-pooled source std with a target std averaged within cells, which scaled the L channel by 0.51
instead of 0.72. Condition D, an unregistered sensitivity rerun with matching definitions, was still strongly negative:

| Change from baseline, pooled | BACC | SE | SP | AUC |
|---|---|---|---|---|
| C-A, registered | -18.75 [-22.73, -15.03] | -39.89 | +2.40 | -0.147 [-0.184, -0.112] |
| D-A, sensitivity | -15.37 [-18.73, -12.29] | -32.66 | +1.92 | -0.126 [-0.161, -0.093] |

Sensitivity collapses while specificity rises, so models stop calling parasites after normalisation. The likely
reason is lost parasite contrast, since D still scales L by 0.72, but this was not tested directly. Only this one
Reinhard recipe was tried, so the result is not a verdict on every test-time colour method.

**Exploratory, unregistered, computed after seeing these results:** 15-checkpoint logit ensembles. Baseline: AUC 0.928,
BACC 73.6. Colour-transfer: AUC 0.972, BACC 82.0. Ensembling lifts ranking far more than it lifts BACC at the fixed
threshold, because sensitivity stays low (47.9 and 64.7).

## Limitations
MP-IDB is now heavily reused, and B has only repeat 0. These results need confirming on BBBC041, which is untouched.
