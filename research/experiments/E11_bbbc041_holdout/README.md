# E11: BBBC041 single-shot holdout

**Status:** registered 2026-11-02, amended the same day before any score, not scored  ·  **Outcome:** pending

> **Amendment, 2026-11-02, decided by the authors before any BBBC041 score or metric existed.** Reason: the MP-IDB rehearsal under Protocol showed the crop rule alone
> decides the outcome. The five-way rule below now runs twice, once on ellipse crops and once on Otsu crops, each with candidates 1 and 3 scored on those crops and its own bootstrap.
> **If both give the same outcome, that is E11's outcome. If they differ, the outcome is "Depends on crop rule".** This overrides the "ellipse primary, Otsu sensitivity" wording below, and
> the crop-rule check is superseded. Both score manifests are committed before either metric is calculated. Probes run on ellipse crops only and stay descriptive.
>
> **Addendum to prediction 1, written by the AI assistant before scoring, after seeing the rehearsal and the crop audit:** Depends on crop rule 45%, Inconclusive 20%, Feature holds 20%, Trained
> model wins 5%, Ranking and threshold disagree 5%, No transfer 5%. The original probabilities stand as written and are the ones scored against the five-way rule.

## Question
Does a one-feature dark-spot rule, fitted on NIH alone, hold against trained models on a new dataset, and does it keep its
fixed threshold there?

## Why this exists
MP-IDB has been reused for many development decisions, including the discovery of the feature, so it can no longer test
it. BBBC041 is a new test: different species (*P. vivax*), different stain, annotated boxes rather than segmentations. It
has hard negatives (leukocytes), and the writer has never seen a score on it. The result decides the paper's framing.

**State at registration:** the archive is downloaded (SHA-256 `46e47b02284fcab4d3f8cff291f08f0dbc1e9296c4ec974f3f4d989652446e81`).
The JSON annotations were read for counts, box sizes and image shapes. No pixel has been opened and no model or feature has
run on BBBC041.

## Prediction (written by the AI assistant, 2026-11-02, before any BBBC041 pixel was read)
**What the writer had already seen:** every MP-IDB result, the feature on three MP-IDB crop rules, MalariaNet's published
BBBC041 result (SE 96.4, SP 18.0, AUC .768), and BBBC041's annotation counts and box sizes. **Not seen:** any BBBC041 image.

1. **Most likely outcome under the rule:** Feature holds 45%, Inconclusive 25%, Trained model wins 15%, Ranking and
   threshold disagree 10%, No transfer 5%.
2. **Candidate 1:** AUC .85 to .95, BACC 78 to 90, SE 80 to 95, SP 70 to 92. Specificity falls well below MP-IDB's 98.4,
   because crowded fields put neighbouring edges inside the ellipse.
3. **Candidate 3:** AUC .82 to .93, BACC 65 to 85, and it beats candidate 2 on AUC by at least .02, repeating E6c.
4. **Candidate 4:** both probes have lower AUC than candidate 3.
5. **Hard negatives:** candidate 1 fires on at least 60% of leukocytes (16 of 20 on MP-IDB).
6. **Natural prevalence:** candidate 1's PPV at BBBC041's roughly 3% prevalence is under 60%.
7. **Acquisition groups:** candidate 1's BACC differs by at least 3 points between the PNG and JPG groups.

**Falsified if:** for prediction 2, candidate 1's AUC or BACC falls outside its range. The outcome rule below stands
regardless of what was predicted.

## Frozen candidates
All are fitted on NIH only. Nothing is refitted, recalibrated or thresholded on BBBC041.

| # | Candidate | Definition | Decision threshold |
|---|---|---|---|
| 1 | Dark-spot feature | `malaria/dark_spot.py`, hash below | score at least .3047 |
| 2 | E1 ensemble | Mean logit of E1 grouped r0 folds 0-4 | mean logit at least 0 |
| 3 | Colour-transfer ensemble | Mean logit of 15 E4b DANets and 5 E11 ResNet-18s, grouped r0 | mean logit at least 0 |
| 4 | Frozen probes (descriptive) | DINOv2 and DinoBloom, logistic regression refitted on all NIH cells with E9's settings | p at least .5 |

Logits are clipped as in the development checks (p in [1e-6, 1 - 1e-6]). RedDino is excluded, because its training list could
not be confirmed free of BBBC041. Checkpoint hashes go in `frozen_manifest.json`, with ResNet-18 hashes added when those runs
finish and before scoring.

## Outcome rule, fixed now
Delta = candidate 3 minus candidate 1, on the primary set. Two co-primary metrics, AUC and BACC, each with a 97.5% CI
(image-clustered percentile bootstrap, 5,000 draws, seed 20261102). The first condition that holds decides.

| Order | Outcome | Condition |
|---|---|---|
| 1 | No transfer | AUC below .80 for both candidates |
| 2 | Feature holds | AUC upper bound at most +.01 and BACC upper bound at most +2 |
| 3 | Trained model wins | AUC lower bound above +.01 and BACC lower bound above +2 |
| 4 | Ranking and threshold disagree | The AUC and BACC decisions point opposite ways |
| 5 | Inconclusive | Anything else |

For rule 4, a metric's decision is "feature" when its upper bound meets the rule 2 limit, and "trained" when its lower bound
meets the rule 3 limit. Any other case for either metric is no decision, so the result falls to Inconclusive.

## Protocol
**Data.** All 1,328 annotated images: 1,208 PNG at 1200x1600 and 120 JPG at 1383x1944. The BBBC041 page lists 1,364; only annotated
images can be scored. Format and shape give two acquisition groups, and a third source cannot be mapped from the metadata.

**Scoring set.** Every non-excluded box is scored, without label-based subsampling, so prevalence stays natural.
- Parasitized: ring, trophozoite, schizont, gametocyte.
- Uninfected: red blood cell.
- Excluded from the primary set, before any metric: boxes touching the image border (within 1 px, about 1.8%, matching MP-IDB),
  `difficult` boxes, and red blood cells whose box intersects a parasite box.
- Leukocytes are scored and kept apart as secondary hard negatives (103 boxes, PNG group only).

**Crops.** Primary: `crops.py` ellipse inscribed in the box, background zeroed. Sensitivity: `crops.py` Otsu rule, which does not
enter the outcome rule.

**Departure from the plan, decided on MP-IDB before any BBBC041 pixel.** ROADMAP section 8 named Otsu masks. Rehearsal showed
they lose specificity against Cellpose masks (feature AUC .969, BACC 90.3, SP 94.7), while the inscribed ellipse stays close
(AUC .982, BACC 93.9, SP 98.4; Cellpose AUC .987, BACC 93.0, SP 99.1). That choice was made on the feature, so it could favour
candidate 1. **Check, fixed now:** candidates 2 and 3 are also rehearsed on both crop rules. The ellipse stays primary unless
either candidate's AUC is more than .01 higher on Otsu crops, in which case Otsu becomes primary. The result is recorded here
before scoring.

**Rehearsal result, 2026-11-02, on MP-IDB, before any BBBC041 score. No BBBC041 metric exists.** Candidate 3 has 15 of its 20 members here, because the ResNet-18s are
still training.

| Crops | Feature AUC / BACC | Candidate 2 AUC / BACC | Candidate 3 AUC / BACC (SE) | Candidate 3 best-threshold BACC* | Delta BACC, 3 minus 1 | Rule on MP-IDB |
|---|---|---|---|---|---|---|
| Cellpose (reference) | .987 / 93.0 | .919 / 73.8 | .973 / 82.0 (64.7) | 92.6 | -11.0 | not run |
| Ellipse | .982 / 93.9 | .932 / 67.2 | .966 / 71.7 (43.9) | 91.7 | -22.3 [-26.4, -18.1] | Feature holds |
| Otsu | .970 / 90.3 | .909 / 78.4 | .973 / 91.2 (85.0) | 92.7 | +0.9 [-1.3, 3.0] | Inconclusive |

*Upper bound only, chosen on the same data.

**The check as registered passes, and the ellipse stays primary:** candidate 2 is higher on ellipse crops, and candidate 3 is +.007 on Otsu crops, within the .01 limit.

**The check was too narrow, which was a design error by the writer.** It tested AUC alone, while BACC is co-primary and is where the crop rule acts. Ranking barely moves. The trained
models' threshold does: candidate 3's median parasite logit is -.36 on ellipse, +.87 on Cellpose and +2.09 on Otsu. Against the Cellpose reference, the ellipse costs candidate 3 about
10 BACC and Otsu adds about 9, while the feature moves by under 3. Neither rule is neutral, and on MP-IDB the choice alone decides the outcome.

**Order of work.** (1) Rehearse the exact scoring script on MP-IDB crops. (2) Build BBBC041 crops. (3) Audit 100 crops by eye
with no model or feature scores in view, recording only whether a crop is usable. (4) Score, then save and hash every score file
before any metric exists. Labels stay in a separate file the scoring script does not read. (5) Calculate once and apply the rule.

**Integrity.** A rescore is allowed only for a failed automated check: row count, file hash or NaN rate above .5%. It is never
allowed because of a metric. Feature NaNs count as score 0.

**Secondary, none of which enters the rule.** SE, SP and PPV at natural prevalence, by stage and acquisition group; leukocyte
false-positive rate; E12 retrains; candidate 2 versus candidate 1.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-02 | `build_crops.py` | 86,035 boxes, 79,722 kept: 2,425 parasitized, 77,194 uninfected, 103 leukocytes; prevalence 3.05%. Excluded: 4,344 red cells intersecting a parasite box, 1,523 border boxes, 446 difficult. All image checksums match. Otsu fell back to the full box on 138 crops |
| 2026-11-02 | Crop audit, ellipse rule | 50 parasitized and 50 uninfected crops drawn by seeded RNG (`audit_sample.csv`), viewed with no scores. Done by the AI assistant, not an expert. All 100 contain their target cell, and none is a gross failure. About 8 of 50 uninfected crops show a dark neighbouring cell inside the ellipse. Parasitized crops show the same crowding. Two colour families are visible. Nothing was excluded or changed as a result |

## Result
Pending.

## Verdict and what we learned
Pending.
