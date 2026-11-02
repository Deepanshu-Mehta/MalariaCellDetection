# E6b: Five-checkpoint external rescore

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** rule failed, as predicted: no robust architecture ranking on MP-IDB

## Question
Does E6's architecture ranking on MP-IDB survive once variation between trained models is counted?

## Why this exists
E6 scored one checkpoint per arm (grouped repeat 0, fold 0). Its bootstrap resampled images only, so its CIs
describe those three trained models, not the architectures. The 2026-11-02 ideas review also found that E6's
threshold-free result is weaker than its BACC result on the saved fold-0 predictions: AUC E8b-E1 +0.024
[0.004, 0.045], E8c-E1 +0.013 [-0.010, 0.034]. The Phase 2 gate/dilation ablation rests on E6's reading.

## Prediction (written by the AI assistant, 2026-11-02, before this experiment's scoring)
**What the writer had already seen:** E6's fold-0 results (BACC, AUC, SE, SP). **Not seen:** any MP-IDB score
from the other 12 checkpoints.
1. **Reproduction check:** rescored fold-0 checkpoints reproduce E6's saved AUCs within 0.001.
2. **Variation between checkpoints is large:** at least two of three arms show a between-fold SD of at least
   2 BACC points and at least 0.01 AUC.
3. **The E1 gap shrinks:** the five-fold mean differences E8b-E1 and E8c-E1 fall from +4.71 and +4.97 BACC to
   roughly +1 to +4.
4. **"E1 is robustly worst" is NOT supported** under the rule below, most likely because E8c-E1 AUC includes 0.
5. **Norm layer still does not explain the external result:** E8b-E8c includes 0 on both BACC and AUC.

**Decision rule, fixed now:**
- "E1 is robustly worst" holds only if E8b-E1 and E8c-E1 both have a hierarchical 95% CI above 0 on both BACC
  and AUC.
- If it holds, the gate/dilation ablation stays the top Phase 2 experiment.
- If it fails, the ablation loses that priority, and E6's README and ROADMAP are corrected to say no robust
  architecture ranking was shown on MP-IDB.

**Prediction 4 is falsified if** the rule holds.

## Protocol
- Score all 15 existing grouped repeat-0 checkpoints (E1, E8b, E8c x folds 0-4) on the same 14,468 MP-IDB crops,
  with the same eval transform. No training, no tuning, no threshold selection (threshold stays 0.5).
- **Deviation from E6's protocol, disclosed:** "score each test set once" is relaxed for these models. Nothing
  is selected on MP-IDB, and BBBC041 stays untouched as the final holdout.
- **Primary metrics:** BACC and AUC, co-primary. SE and SP reported.
- **Primary comparison:** mean of the fold-paired arm differences. All arms in one fold share training patients.
- **Primary CI:** hierarchical bootstrap, B=2000, resampling MP-IDB images and fold indices, with the same draw
  applied to every arm. A paired t-interval over the five fold differences (df=4) is reported as secondary.
- **Limitation:** the five folds share about 75% of their training patients, so this understates what fully
  independent training runs would show.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-02 | (local) `score_external.py --cond A` | 15 checkpoints. Inference-only `GroupNorm(C, C)` swap for `InstanceNorm2d`, checked equal to about 1e-5, about 9x faster on MPS. First passes were slowed by unrelated memory pressure on the laptop |
| 2026-11-02 | (local) `analysis.py` | Registered analysis, B=2000 |
| 2026-11-02 | (local) `exploratory_fold_effect.py` | Added after seeing the results, not registered |

## Result
**Reproduction:** fold-0 AUCs match E6 within 0.0000012 for all three arms (largest probability difference 0.00032).

**Five-checkpoint means on MP-IDB (mean +/- SD across folds):**

| Arm | BACC | Per-fold BACC | SE | SP | AUC |
|---|---|---|---|---|---|
| E1 | 72.75 +/- 8.41 | 74.1, 66.5, 85.7, 73.3, 64.2 | 48.2 +/- 17.6 | 97.3 +/- 1.5 | 0.888 +/- 0.041 |
| E8b | 74.42 +/- 5.33 | 78.8, 69.0, 80.9, 73.7, 69.7 | 51.9 +/- 11.9 | 96.9 +/- 2.0 | 0.881 +/- 0.025 |
| E8c | 70.61 +/- 6.45 | 79.1, 70.5, 74.3, 66.7, 62.5 | 44.1 +/- 14.6 | 97.1 +/- 3.1 | 0.884 +/- 0.060 |

**Fold-paired comparisons (hierarchical 95% CI; paired t-interval in brackets after it):**

| Comparison | BACC | AUC |
|---|---|---|
| E8b - E1 | +1.67 [-1.98, 5.42]; t [-3.45, 6.79] | -0.006 [-0.032, 0.020]; t [-0.036, 0.023] |
| E8c - E1 | -2.14 [-7.84, 3.24]; t [-10.84, 6.56] | -0.004 [-0.034, 0.026]; t [-0.048, 0.041] |
| E8b - E8c | +3.81 [-0.01, 7.37]; t [-1.55, 9.17] | -0.003 [-0.040, 0.040]; t [-0.058, 0.053] |

**Prediction scorecard:**

| # | Prediction | Result |
|---|---|---|
| 1 | Fold 0 reproduces E6 | Confirmed |
| 2 | SD at least 2 BACC and 0.01 AUC in two arms | Confirmed, far exceeded: all three arms, SD 5.3-8.4 BACC and 0.025-0.060 AUC |
| 3 | E8b-E1 and E8c-E1 fall to +1 to +4 | Half right: E8b-E1 +1.67 in range; E8c-E1 -2.14 is below it, so E1 is not worst on the mean |
| 4 | "E1 is robustly worst" not supported | Confirmed: every CI includes 0 |
| 5 | E8b-E8c includes 0 on both metrics | Confirmed, narrowly on BACC (lower bound -0.006); clearly on AUC |

## Verdict and what we learned
**The registered rule failed, so E6's architecture ranking does not stand.** E6's +4.71 and +4.97 BACC gaps came from
which fold-0 models happened to be trained. Across five checkpoints no arm is reliably better on either metric, and
the ordering changes: E8c, which tied for best in E6, has the lowest mean BACC here. Per the rule, the gate/dilation
ablation loses its top Phase 2 priority, and "gate/dilation drives external robustness" is withdrawn.

E6's narrower finding survives in weaker form: there is still no evidence that InstanceNorm generalises better than
LayerNorm on MP-IDB, which was the prediction E6 falsified.

**Exploratory, unregistered, added after seeing the results:** retraining the same architecture on a different 80% of
NIH patients changes external performance far more than changing the architecture.
- The SD of fold means is 6.12 BACC and 0.039 AUC. The SD of arm means is 1.91 BACC and 0.003 AUC.
- Fold profiles track each other across arms (r = 0.58-0.92 BACC, 0.73-0.85 AUC), for example fold 4 is the lowest or
  near-lowest in every arm. This rests on five points only, so treat it as suggestive.
- On NIH the same checkpoints have a BACC SD of 0.81-0.97. That figure includes different test folds, so it overstates
  checkpoint variation there. On MP-IDB every checkpoint sees the same crops, so its SD of 5.3-8.4 is all model-to-model.

This looks like underspecification: models that are equivalent on NIH behave very differently outside it. If it holds
up, it matters more to the paper than any architecture comparison. It would need its own registered test, with
independent retrains, before being claimed.

**Knock-on effect:** E5's explanation results also come from fold-0 checkpoints only, so they deserve the same
five-checkpoint check before the paper relies on them.

## Limitations
Five folds share about 75% of their training patients, and a bootstrap over five folds is coarse. The BACC threshold
stays 0.5, which is why SE swings so widely. This experiment shows a ranking is unsupported, not that the arms are
equal: the CIs are wide enough to hide a real difference of several BACC points.
