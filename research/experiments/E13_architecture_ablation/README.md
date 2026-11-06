# E13: Which part of DANet's architecture earns its accuracy?

**Status:** done (Kaggle, 2026-11-03 to 11-04), NIH only  ·  **Outcome:** E13a predictions 1-2 supported; E13b prediction 4 missed on its CI clause, prediction 5′ falsified

## Question
Two architecture changes to DANet-S, each tested against the unchanged E1 model on the same folds and seeds:
- **E13a, no attention.** Does DANet's core contribution, the Dilated Attention Block, matter?
- **E13b, max-pool head.** Does pooling for "a parasite anywhere" raise sensitivity, which is the weak point on
  another lab's slides?

## Why this exists
For the course presentation, and because the audit so far changed DANet's gate, dilation order and norm layer
(E8) but never removed the attention itself. E1r showed a plain ResNet-18 matches DANet on unseen patients, and a
hand-made dark-spot feature reaches 95.6 BACC on NIH, so it is open whether the attention adds anything. On MP-IDB,
DANet's errors are mostly missed parasites (sensitivity about 48% at threshold .5; E6b, E12), while specificity stays
near 97%. DANet averages its final 7x7 feature map, so a parasite occupying one cell of that map contributes 1/49 of
the pooled signal.

## The two changes (`research/malaria/models.py`)
| Arm | Change | Parameters |
|---|---|---|
| E1 (baseline) | released DANet-S: nine Dilated Attention Blocks, final `AvgPool2d(7)` | 2,310,842 |
| **E13a** `attention="none"` | each block keeps only its 1x1 projection (conv, InstanceNorm, LeakyReLU); the dilated 3x3 convolutions, the avg/max pooling concat and the gate are removed. The only spatial filter left is the stem's 3x3 conv; spatial context comes from the max-pools between blocks | 42,778 (−98%) |
| **E13b** `head_pool="max"` | final `AvgPool2d(7)` replaced by `MaxPool2d(7)`: the head sees the strongest response anywhere in the cell | 2,310,842 |

Defaults are unchanged: the default model loads every existing E1 checkpoint strictly and reproduces E6b's saved
MP-IDB scores to 3.5e-6 (checked 2026-11-03, CPU vs MPS rounding).

## Prediction (written by the AI assistant, 2026-11-03, before any E13 run)
**What the writer had already seen:** all E1, E1r, E6b, E6c, E9, E12 results, the dark-spot feature's NIH score, and a
one-epoch smoke test of both arms on 2% of the data (BACC 50, uninformative). **Not seen:** any trained E13 result.

**E13a, no attention.**
1. **NIH (primary):** pooled out-of-fold grouped-r0 BACC is **0.5 to 2.5 points below** E1's 96.78, with the paired
   patient-bootstrap CI below 0. Reasoning: a per-pixel colour network with max-pooling can still find purple
   chromatin, so most of the accuracy survives (the dark-spot feature alone gets 95.6), but ring shape and texture
   context are lost.
2. **NIH:** sensitivity falls more than specificity.
3. **MP-IDB (secondary):** 5-fold mean AUC below E1's 0.888, since a colour-only model should transfer worse.

**E13b, max-pool head.**
4. **NIH (primary):** BACC changes by **less than the 0.29 MDE**, with the paired CI including 0. NIH is saturated.
5. **MP-IDB (secondary):** 5-fold mean sensitivity **at least +5 points** above E1's 48.2, specificity lower by at most
   3, and BACC change positive.

**Interpretation rule, fixed now:**
- E13a drop below 0.29 with CI including 0: the dilated attention adds nothing measurable on NIH.
- E13a drop above 2.5: the attention matters more than the dark-spot reading implies.
- E13b prediction 5 met: max pooling addresses the external under-calling. Missed: pooling is not its cause.

**Falsified if:** prediction 1 or prediction 5 misses its range.

**Power, stated now.** NIH: MDE 0.29 BACC (E1 A/A). (Withdrawn with the amendment below.) MP-IDB: E12 found a seed-only SD of 6.63 BACC per checkpoint, so a
fold-paired difference over 5 folds has a 95% CI half-width of roughly ±12 BACC. MP-IDB results can only show large
effects and are reported as secondary.

## Amendment, 2026-11-03, before any E13 result (authors' decision)
Written while the Kaggle job was still running; no E13 checkpoint, prediction file or metric had been downloaded or seen.
**E13 uses the NIH dataset only.** MP-IDB is dropped from this experiment:
- Predictions 3 and 5 (MP-IDB) are **withdrawn unscored**. No E13 checkpoint will be scored on MP-IDB.
- Predictions 1, 2 and 4 stand unchanged.
- **New prediction 5′ (E13b, NIH):** max pooling reacts to the single strongest response in the cell, so it trades
  specificity for sensitivity. E13b's NIH sensitivity change is **positive** and its specificity change **negative**,
  each under 1 point in size, with BACC still within the 0.29 MDE (prediction 4).
- **Falsified if:** prediction 1 misses its range, or prediction 5′ has the wrong sign on either SE or SP.
- The interpretation rule for E13b becomes: 5′ met means the max-pool head shifts the operating point toward catching
  parasites without changing overall accuracy; missed means pooling does not set that trade-off on NIH.

## Protocol
- **Runs:** each arm on grouped repeat 0, folds 0-4, `train_seed` 0, 50 epochs, everything else as `configs/E1_danet.json`.
  10 runs on Kaggle T4. The E1 grouped r0 checkpoints are the baseline; no baseline is retrained.
- **NIH primary metric:** pooled out-of-fold BACC over 27,558 cells, paired difference vs E1 r0 with the patient-cluster
  bootstrap (`research.malaria.stats.paired_diff`, B = 5000), as in E1r. SE, SP and AUC reported alongside.
- **MP-IDB:** dropped by the amendment above; not scored.
- **BBBC041:** not used. E11 is final.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-03 | (local) smoke test, both arms, 1 epoch, 2% data | runs end to end; E13a 2.0 s/epoch vs E13b 11.0 s/epoch on MPS |
| 2026-11-03 to 11-04 | Kaggle `e13-architecture-ablation`, 10 runs, 2 x T4 | all return code 0; E13b 1.13-1.17 h, E13a 0.87-0.96 h per run (CPU data loading dominates, 65-68 s/epoch even at 42,778 params); about 5.3 quota-h |
| 2026-11-04 | (local) `analysis.py`, `results.json` | registered NIH analysis, patient-cluster bootstrap B = 5000 |

## Result
**NIH, grouped repeat 0, pooled out-of-fold over 27,558 cells (2026-11-04, `analysis.py`, `results.json`).**

| Arm | Params | BACC | SE | SP | AUC |
|---|---|---|---|---|---|
| E1 (unchanged DANet) | 2,310,842 | 96.78 | 96.01 | 97.55 | .9924 |
| E13a, no attention | 42,778 | 96.04 | 94.54 | 97.54 | .9893 |
| E13b, max-pool head | 2,310,842 | 96.59 | 95.78 | 97.41 | .9932 |

**Paired difference vs E1, patient-cluster bootstrap 95% CI:**

| Change | BACC | SE | SP | AUC |
|---|---|---|---|---|
| E13a − E1 | **−0.74 [−1.08, −0.43]** | −1.47 [−2.18, −0.88] | −0.01 [−0.32, +0.30] | −.0031 [−.0050, −.0016] |
| E13b − E1 | **−0.19 [−0.37, −0.01]** | −0.23 [−0.55, +0.08] | −0.14 [−0.38, +0.10] | +.0007 [−.0001, +.0018] |

| Prediction | Result | Outcome |
|---|---|---|
| 1. E13a BACC 0.5 to 2.5 below E1, CI below 0 | −0.74 [−1.08, −0.43] | **Supported** |
| 2. E13a SE falls more than SP | SE −1.47, SP −0.01 | **Supported** |
| 4. E13b \|ΔBACC\| < 0.29 with CI including 0 | −0.19, below the MDE, but the CI [−0.37, −0.01] just excludes 0 | **Missed on the CI clause** |
| 5′. E13b SE up, SP down, each under 1 | SE −0.23, SP −0.14: both down | **Falsified** (wrong sign on SE) |
| 3, 5 (MP-IDB) | withdrawn unscored by the amendment | n/a |

Exploratory, computed after the registered analysis: per-fold BACC, E13a − E1 = −0.83, −0.68, −0.53, −0.86, −0.73, lower on
all 5 folds; E13b − E1 = −0.11, +0.11, −0.28, −0.28, −0.39, lower on 4 of 5.

## Verdict and what we learned
**E13a: removing DANet's attention costs 0.74 points, as predicted, and the cost is all sensitivity.** A network with 98%
fewer parameters, whose only spatial filter is the stem's 3x3 convolution, still reaches 96.0 BACC on unseen patients.
Specificity does not move at all, so the dilated attention is not what rejects healthy cells; it helps catch about 1.5% more
of the parasitized cells, which fits the reading that most cells are decided by a simple colour cue and the attention adds
context for the harder parasites. The drop is consistent across folds and its CI excludes 0, but it rests on one seed per
arm; E1's A/A pair differed by 0.15, well below 0.74.

**E13b: the max-pool head did not shift the operating point.** Sensitivity did not rise, so prediction 5′ is falsified:
pooling is not what sets DANet's sensitivity on NIH. The BACC change (−0.19) is below the 0.29 MDE, so by the registered
rule it is not a measurable change, but the patient-bootstrap CI just excludes 0, so prediction 4 is missed as worded. The
bootstrap resamples patients, not training runs, so it does not include seed variance, which is why the MDE is the
registered yardstick. Report E13b as "no measurable gain; at most a small loss".

**Limits.** One seed per arm, one repeat, NIH only. E13a removes the dilated convolutions and the gate together, so it cannot
say which of the two carries the 0.74.
