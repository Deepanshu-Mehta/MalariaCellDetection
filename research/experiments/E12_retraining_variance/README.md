# E12: Retraining variance on external data

**Status:** done (Kaggle and local, 2026-11-02 to 11-03)  ·  **Outcome:** falsified on prediction 1's BACC range, because seed spread came out larger than predicted; predictions 2, 3 and 4 supported

## Question
How much of the checkpoint-to-checkpoint spread on MP-IDB comes from the training seed alone, with the training
patients held fixed?

## Why this exists
**Correction, 2026-11-02, after registration:** "differ by under 1 BACC" below is wrong. E1's repeat-0 NIH BACC has SD .94 across folds, with a range of 95.3 to 97.9. The argument
and the prediction are unchanged, since 8.4 is still about nine times .94.

E6b found between-fold BACC SDs of 5.3-8.4 on MP-IDB, while the same checkpoints differ by under 1 BACC on NIH. Each
fold changes both the training patients and the fitted model, so E6b cannot separate them. Exploratory checks on
E6b's scores (fold-mean SD 6.12, arm-mean SD 1.91) suggested possible underspecification, but no run has fixed the
data and varied only the seed. The same result decides how much weight a single checkpoint can carry on BBBC041.

## Prediction (written by the AI assistant, 2026-11-02, before any run or scoring in this experiment)
**What the writer had already seen:** E6b's per-fold scores for E1, E8b and E8c, including E1 fold 2 at 85.7 and fold
3 at 73.3, and NIH test results for all grouped runs. **Not seen:** any seed other than 0 on any fold, or any MP-IDB
score from a repeat 1-2 checkpoint or a ResNet-18.

1. **Seed alone spreads external BACC:** pooled within-fold SD, across 5 seeds on each of 2 folds, is 3 to 6 BACC.
   AUC SD is .01 to .04.
2. **NIH does not show it:** the same checkpoints' pooled within-fold SD on their NIH test fold is under .5 BACC, and at
   most a fifth of the MP-IDB SD.
3. **Regression on fold 2:** the mean of seeds 1-4 on fold 2 is at least 3 BACC below seed 0's 85.7.
4. **Unused checkpoints agree:** E1 repeats 1-2 (10 checkpoints) have mean BACC 70 to 76 and SD at least 5. ResNet-18
   (15 checkpoints, repeats 0-2) has SD at least 3 and mean BACC below 80.

**Interpretation rule, fixed now:**
- Seed SD at least 3 BACC: a single checkpoint's external BACC is unsafe even with training data fixed.
- Seed SD at most 1: external spread comes from which patients were trained on, not from the seed.
- In between: both matter, and the size is reported without choosing one.

**Falsified if:** prediction 1's BACC range is missed, or prediction 2's ratio fails.

## Protocol
- **Runs:** `configs/E1_danet.json`, grouped repeat 0, folds 2 and 3, `train_seed` 1-4 (8 runs). Seed 0 already exists.
- **Fold choice:** `np.random.default_rng(20261102).choice(5, 2, replace=False)` gave `[2, 3]`, before any run. Fold 2
  is E1's highest seed-0 value. It is kept, because redrawing would be a selection.
- **External scoring:** E6b's `score()` path, same 14,468 crops, threshold .5, no tuning. ResNet-18 uses 128 px, its
  training size. No selection or metric from this experiment feeds a BBBC041 candidate.
- **Primary metric:** pooled within-fold SD of MP-IDB BACC (8 df), with a chi-square 95% CI. AUC is reported alongside.
- **Secondary:** variance share of seed against E1's total single-checkpoint variance (8.41 SD, E6b), which is rough
  because it borrows a five-fold estimate.
- **Limitation:** two folds, five seeds each, one architecture. E12 cannot estimate data variance on its own.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-02 | (local) `score_checkpoints.py --set unused` | 25 unused checkpoints |
| 2026-11-02 to 11-03 | Kaggle `e10-e11-e12-batch`, E1 config, grouped r0 folds 2 and 3, `train_seed` 1-4 | 8 runs, all return code 0, about 1.05h each |
| 2026-11-03 | (local) `score_checkpoints.py --set seeds`, `analysis.py` | E1 r0f0 reproduced E6b exactly (max p difference 0) |

## Result
**Predictions 1-3 (2026-11-03, `analysis.py`, `results.json`).** MP-IDB, threshold .5:

| Fold | Seed | MP-IDB BACC | AUC | SE | SP | NIH test BACC |
|---|---|---|---|---|---|---|
| 2 | 0 | 85.70 | .945 | 74.4 | 97.0 | 96.63 |
| 2 | 1 | 82.15 | .943 | 65.4 | 98.9 | 97.12 |
| 2 | 2 | 67.86 | .880 | 38.0 | 97.7 | 97.27 |
| 2 | 3 | 72.33 | .898 | 48.1 | 96.6 | 97.11 |
| 2 | 4 | 74.01 | .889 | 51.1 | 96.9 | 97.13 |
| 3 | 0 | 73.29 | .878 | 50.2 | 96.4 | 97.89 |
| 3 | 1 | 71.67 | .856 | 47.9 | 95.5 | 97.89 |
| 3 | 2 | 66.87 | .817 | 36.2 | 97.5 | 97.73 |
| 3 | 3 | 73.36 | .860 | 52.2 | 94.5 | 97.92 |
| 3 | 4 | 59.61 | .845 | 19.9 | 99.3 | 97.74 |

| Prediction | Result | Outcome |
|---|---|---|
| 1. Pooled seed SD 3 to 6 BACC, .01 to .04 AUC | BACC SD **6.63 [4.48, 12.71]**, 8 df; AUC SD .027 [.018, .052] | **BACC range missed, above the top; falsified.** AUC point estimate in range |
| 2. NIH SD under .5 and at most a fifth of MP-IDB | NIH SD .18 [.12, .35]; ratio .03 | **Supported** |
| 3. Fold 2 seeds 1-4 at least 3 BACC below 85.7 | Mean 74.09, which is 11.61 below | **Supported** |

Secondary, rough because it borrows E6b's five-fold SD of 8.41: seed variance is 62% of E1's total single-checkpoint variance.

Unregistered, computed after seeing the table: fold 2's seed 0 was known to be high before registration, so dropping both seed 0
rows gives a pooled SD of 6.07 on 6 df. The excess over 6 does not depend on that row, though the difference is small and well inside the CI.

**Prediction 4 (2026-11-02, `analysis.py`, `results.json`).**

| Checkpoints | BACC mean +/- SD (range) | AUC mean +/- SD | SE | SP | Prediction 4 |
|---|---|---|---|---|---|
| E1 repeats 1-2 (10) | 71.56 +/- 6.53 (58.3-79.8) | 0.874 +/- 0.039 | 46.8 | 96.3 | **Supported:** mean in 70-76, SD at least 5 |
| ResNet-18 repeats 0-2 (15) | 67.41 +/- 5.81 (59.6-77.1) | 0.902 +/- 0.027 | 35.6 | 99.2 | **Supported:** SD at least 3, mean below 80 |

Unregistered observation: ResNet-18 has the lower BACC but the higher mean AUC, with SE 35.6 and SP 99.2. At threshold .5
both models under-call parasites on MP-IDB, so ranking and thresholding need separate reporting.

## Verdict and what we learned
**Falsified as registered, because the spread was larger than predicted, not smaller.** The point estimate is 6.63 against a
top of 6. The CI runs from 4.48 to 12.71, so the data do not exclude part of the predicted range, but the registered test was on
the range, and it was missed.

**By the interpretation rule, a single checkpoint's external BACC is unsafe even with the training data fixed.** Seed SD is at
least 3, and the CI's lower bound is too. The same ten checkpoints differ by an SD of .18 on NIH, so NIH could not have
warned of this: on fold 3, seeds 0 and 4 differ by .15 on NIH and by 13.7 on MP-IDB. This is direct evidence of underspecification.
Fold 2's 85.7, the best E1 value, regressed to 74.1 on new seeds, so that value owed more to the seed than to the fold's patients.

**Consequences.** E6b's between-fold spread cannot be read as data variance alone. BBBC041 candidates are ensembles, which is the
right choice given this result, though no selection from E12 fed them. Sensitivity drives the spread: SE runs from 19.9 to 74.4
while SP stays between 94.5 and 99.3, which fits E11's separation of ranking from threshold.

**Limits.** Two folds, five seeds each, one architecture, one external dataset. E12 cannot estimate data variance on its own, and
the 62% share is rough.
