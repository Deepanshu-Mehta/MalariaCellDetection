# E12: Retraining variance on external data

**Status:** registered 2026-11-02, not yet run  ·  **Outcome:** pending

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

## Result
**Prediction 4 (2026-11-02, `analysis.py`, `results.json`). Predictions 1-3 wait for the Kaggle seeds.**

| Checkpoints | BACC mean +/- SD (range) | AUC mean +/- SD | SE | SP | Prediction 4 |
|---|---|---|---|---|---|
| E1 repeats 1-2 (10) | 71.56 +/- 6.53 (58.3-79.8) | 0.874 +/- 0.039 | 46.8 | 96.3 | **Supported:** mean in 70-76, SD at least 5 |
| ResNet-18 repeats 0-2 (15) | 67.41 +/- 5.81 (59.6-77.1) | 0.902 +/- 0.027 | 35.6 | 99.2 | **Supported:** SD at least 3, mean below 80 |

Unregistered observation: ResNet-18 has the lower BACC but the higher mean AUC, with SE 35.6 and SP 99.2. At threshold .5
both models under-call parasites on MP-IDB, so ranking and thresholding need separate reporting.

## Verdict and what we learned
Pending.
