# E5b: Five-checkpoint explanation check

**Status:** registered 2026-11-02, not yet run  ·  **Outcome:** pending

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

## Result
Pending.

## Verdict and what we learned
Pending.
