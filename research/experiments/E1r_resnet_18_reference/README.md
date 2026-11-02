# E1r: ResNet-18 reference

**Status:** done (18/18 runs)  ·  **Outcome:** no prior prediction; DANet ≈ ResNet-18 on unseen patients (within MDE)

## Question
Does DANet beat a standard ImageNet CNN on identical folds?

## Prediction (written before any run; date it)
**None was written before the first run (launched 2026-10-23 as part of the Phase-0 timing job, before the
prediction-before-running discipline was applied project-wide).** Recorded honestly as a gap, not scrubbed:
the result (below) is reported as an unregistered, exploratory comparison rather than a confirmed hypothesis
test.

## Protocol
- Splits / runs: same folds and seeds as E1
- Config: `research/configs/E1r_resnet18.json`
- Primary metric: BACC, ROC-AUC
- Estimated T4 GPU-h: 8

## Runs
Kaggle job slugs and dates (results are aggregated by `research/scripts/collect_results.py` into
`research/results/all_runs.csv`):

| Date | Job slug | Runs | Notes |
|---|---|---|---|
| 2026-10-23 | m3-rows-and-resnet | grouped r0 f1-4, r1-r2 f0-4; leaky s42-44 | 17/17 finished, ~8 min each |
| 2026-10-23 | danet-phase0-timing | grouped r0 f0 (15 ep, 128 px) | Phase 0 timing job; 24 s/epoch. Note: run before a written prediction existed |

## Result (`analysis.py` -> `results.json`)

| Model | Grouped BACC | Grouped SE | Grouped SP | Leaky BACC | Leaky - grouped gap |
|---|---|---|---|---|---|
| ResNet-18 (ImageNet, 11.2 M, 128 px) | 96.58 ± 0.05 | 95.51 | 97.66 | 97.26 ± 0.05 | **+0.68** |
| DANet-S (2.3 M, 224 px) | 96.71 ± 0.08 | 95.91 | 97.51 | 96.96 ± 0.20 | **+0.26** |

Paired DANet - ResNet on identical grouped OOF cells: BACC +0.13 [-0.05, +0.30] (below the 0.29 MDE),
SE +0.40 [+0.11, +0.72], SP -0.15 [-0.34, +0.05], AUC +0.0003.
Reading: on unseen patients DANet is level with an off-the-shelf ResNet-18 overall (slightly more sensitive,
slightly less specific). ResNet-18 benefits ~2.6x more from leakage, but it differs from DANet in pretraining,
resolution and normalisation (BatchNorm), so the cause is not isolated here; E8 tests the norm layer directly.

## Verdict and what we learned
_Positive / negative / invalid, and why. Negative results stay here and go into the paper's appendix._

## Notes

