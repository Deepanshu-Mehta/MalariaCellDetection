# E5b: Five-checkpoint explanation check

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** primary prediction supported; rule met in all three arms; prediction 4 missed

## Question
Does a CAM land on the parasite more often than a no-model rule, across checkpoints rather than one per arm?

## Why this exists
E5 used one checkpoint per arm (fold 0). The 2026-11-02 review found a darkest-pixel rule hits 97.6-98.4% on the same true
positives, against CAM 77.7-94.1%, and that E5's chance figure was understated. E6b then showed that a single fold-0
checkpoint is a poor summary on MP-IDB. Both problems apply to E5.

## Prediction (written by the AI assistant, 2026-11-02, before any run)
**What the writer had already seen:** E5's fold-0 results and the fold-0 darkest-pixel and centre figures. **Not seen:**
any CAM from folds 1-4.

1. **Reproduction:** fold-0 CAM hits match E5's saved flags on at least 99% of crops in each arm.
2. **Primary:** darkest pixel beats CAM in every arm's five-fold mean, and on at least 13 of 15 checkpoints.
3. **Spread:** between-fold SD of the CAM hit rate is at least 5 points in at least two arms.
4. **Centre:** the image-centre rule stays between 40 and 60% on every checkpoint.

**Decision rule, fixed now:**
- If darkest minus CAM has a CI above 0 in all three arms, E5 reports that CAM hits show dependence on a dark object, and
  gives no evidence of learned morphology.
- If CAM reaches or beats the darkest pixel in any arm, the 2026-11-02 E5 correction is reopened for that arm.

**Falsified if:** prediction 2 fails.

## Protocol
- **Checkpoints:** E1, E8b and E8c, grouped repeat 0, folds 0-4 (15). E5's own files are not overwritten.
- **Crops and masks:** E5's Parasitized MP-IDB crops with `parasite_masks`, mask at least 4 pixels.
- **CAM and hit rule:** E5's `grad_cam_v2`, `pointing_game` and 6 px tolerance, on true positives, on 224 px input.
- **No-model rules:** `verify_corrections.py`'s darkest-pixel (min channel, sigma 3, erosion 8) and centre rules, scored
  with the same mask and tolerance on each checkpoint's true positives.
- **CI:** hierarchical bootstrap, crops and folds, 2,000 draws, as in E6b.
- **Origin:** the darkest-pixel rule was chosen after seeing E5 on fold 0. Folds 1-4 are new to it, but the rule is not
  independent of E5.
- **Inference:** E6b's `fast_norms` is used for speed. Prediction 1 checks it does not alter results.
- MP-IDB masks only. NIH has no parasite masks, so whether this holds on NIH stays open.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-02 | `run.py` | 13 of 15 checkpoints, then MPS out of memory. Finished in a fresh process with the script's resume path. Saved results are unaffected |

## Result
**Correction, 2026-11-02:** the first analysis resampled crops, not whole images as the registered "as in E6b" requires. It was fixed and rerun. CI limits moved by under .7 points, and no
conclusion changed.

Hit rates in percent, on each checkpoint's true positives, out of 963 usable crops. CI: hierarchical bootstrap over images and folds.

| Arm | CAM mean +/- SD | CAM by fold (0-4) | Darkest pixel | Darkest minus CAM | True positives |
|---|---|---|---|---|---|
| E1 | 71.5 +/- 13.9 | 77.7, 68.5, 57.0, 62.3, 92.2 | 98.9 | +27.3 [15.8, 38.0] | 296-716 |
| E8b | 94.5 +/- 2.3 | 94.1, 96.8, 95.0, 90.8, 96.0 | 98.9 | +4.4 [2.1, 6.9] | 373-640 |
| E8c | 42.8 +/- 43.6 | 90.4, 16.3, 9.0, 90.4, 7.9 | 98.4 | +55.6 [22.1, 88.6] | 279-635 |

| Prediction | Outcome |
|---|---|
| 1. Fold 0 reproduces at least 99% | **Supported:** hits and predictions match on 100% of crops in all arms |
| 2. Darkest pixel beats CAM in every arm, at least 13 of 15 checkpoints | **Supported:** every arm, 15 of 15 |
| 3. CAM SD at least 5 points in at least two arms | **Supported:** E1 13.9, E8c 43.6. E8b 2.3 |
| 4. Centre stays 40-60% on every checkpoint | **Missed:** 1 checkpoint(s) outside, range 49.6-63.2 |

**Decision rule:** met. The CI is above 0 in all three arms.

## Verdict and what we learned
**CAM hits show dependence on a dark object, and give no evidence of learned morphology.** A rule with no model beats CAM on every
checkpoint. The bigger lesson is instability: E5's fold 0 was not representative. E8c's CAM hit rate is about 90% on folds 0 and 3, and 8-16% on the
other three. E8b is stable at 91-97%. This experiment does not explain why E8c splits this way. Single-checkpoint Grad-CAM should not be reported as an
explanation of an architecture.
