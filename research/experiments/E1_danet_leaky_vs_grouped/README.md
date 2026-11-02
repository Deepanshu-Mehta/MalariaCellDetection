# E1: DANet leaky vs grouped

**Status:** draft - team to confirm  ·  **Outcome:** — (positive / negative / invalid)

## Question
How much of DANet's score depends on patient overlap?

## Prediction (written before any run; date it)
From M3 row B1 (registered 2026-10-05): grouped vs leaky BACC -1 to -2.5 pts, SE -1.5 to -3 pts, AUC 0.985-0.99

## Protocol
- Splits / runs: leaky seeds 42-44 + grouped 5 folds x 3 repeats
- Config: `research/configs/E1_danet.json`
- Primary metric: BACC, SE, ROC-AUC (pooled OOF) + patient-macro
- Estimated T4 GPU-h: 30

## Runs
Kaggle job slugs and dates (results are aggregated by `research/scripts/collect_results.py` into
`research/results/all_runs.csv`):

| Date | Job slug | Runs | Notes |
|---|---|---|---|
| 2026-10-23 | e1-danet-grid | grouped r0 f2-4, r1-r2 f0-4; leaky seeds 42-44 | 16/16 finished, ~1.0 h each; leakage assertions passed |
| 2026-10-23 | danet-phase0-timing | grouped r0 f0, f1 (50 ep) | Phase 0 timing job; 78–79 s/epoch on T4; best epochs 42, 34 |

## Result
`analysis.py` -> `results.json` (2026-10-23).

| Protocol | BACC | SE | SP | AUC | SP healthy-only pts | SP infected pts |
|---|---|---|---|---|---|---|
| Leaky (3 seeds) | 96.96 ± 0.20 | 96.25 | 97.68 | 0.994 | 98.36 | 97.37 |
| Grouped (pooled OOF, 3 repeats) | 96.71 ± 0.08 | 95.91 | 97.51 | 0.992 | 98.08 | 97.32 |
| Gap | +0.26 | +0.34 | +0.17 | +0.001 | | |

Patient-bootstrap 95 % CI: grouped r0 [96.18, 97.29]; leaky s42 [96.04, 97.38]. Per-fold BACC sd 0.72.
A/A test (repeat 0 vs 1, same cells): diff 0.15, SE 0.10 -> **MDE 0.29 pp BACC** (used for E4).
Prediction (B1: -1 to -2.5 BACC, -1.5 to -3 SE) **not supported**: the leakage penalty for DANet is ~0.3 pp,
smaller than published for ResNet-50 features (2.7 pp) and MalariaNet (1.45 pp).

## Verdict and what we learned
_Positive / negative / invalid, and why. Negative results stay here and go into the paper's appendix._

## Notes
A/A seed pairs give the MDE; write it here before E4 is analysed
