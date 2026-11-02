# E4: M3 colour augmentation (pre-registered)

**Status:** registered (M3)  ·  **Outcome:** — (positive / negative / invalid)

## Question
Does ColorJitter raise SE / BACC on unseen patients?

## Prediction (written before any run; date it)
M3 (2026-10-05): vs B1 SE +0.5 to +1.5, BACC +0.3 to +1.0; control C within +/-0.3

## Minimal detectable effect (fixed 2026-10-23, before any E4 run)
0.29 pp BACC (E1 A/A test). Effects smaller than this are reported as "no detectable change".

## Protocol
- Splits / runs: M3 rows on m3 splits; then grouped 5x3 + leaky control
- Config: `research/configs/E4_colourjitter.json`
- Primary metric: SE and BACC (paired patient bootstrap vs E1)
- Estimated T4 GPU-h: 40

## Runs
Kaggle job slugs and dates (results are aggregated by `research/scripts/collect_results.py` into
`research/results/all_runs.csv`):

| Date | Job slug | Runs | Notes |
|---|---|---|---|
| 2026-10-23 | m3-rows-and-resnet | B1 m3 s0-2, H m3 s0-2, C leaky s42 (all 75 ep) | 7/7 finished; ColorJitter runs 3.1-3.35 h vs 1.4-1.6 h (CPU hue jitter) |

## Result (M3 registered rows, `analysis_m3_rows.py` -> `results_m3_rows.json`)

| Row | Split | BACC | SE | SP | AUC |
|---|---|---|---|---|---|
| B0 | leaky s42 (M2 run) | 97.03 | 96.30 | 97.75 | 0.995 |
| B1 | M3 patient splits (3) | 96.80 ± 0.52 | 96.08 ± 1.49 | 97.52 | 0.994 |
| H | M3 patient splits (3) | 96.72 ± 0.74 | 96.05 ± 1.09 | 97.39 | 0.994 |
| C | leaky s42 | 96.84 | 95.72 | 97.97 | 0.994 |

Paired, same test cells (patients resampled within split seed):
- H - B1: **BACC -0.04 [-0.22, +0.14]**, **SE +0.06 [-0.17, +0.29]**, AUC -0.0003 [-0.0012, +0.0005].
  Predicted: BACC +0.3 to +1.0, SE +0.5 to +1.5, AUC +0.001 to +0.004 -> all CIs exclude the predicted ranges.
- C - B0: BACC -0.18 [-0.59, +0.21] -> within the predicted +/-0.3 (control behaves as predicted).
- B1 - B0: BACC -0.23 (predicted -1 to -2.5; consistent with E1's small leakage gap).
- Between-split sd did not shrink (BACC 0.52 -> 0.74).

## Verdict and what we learned
**Negative (hypothesis falsified by the registered rule).** Colour augmentation does not change DANet's
performance on unseen patients; the effect is bounded within about +/-0.3 pp BACC. This matches the M3 "why we
might be wrong" reasoning: with a leakage gap of only ~0.25 pp (E1), there is little stain shortcut left for
augmentation to remove, plausibly because InstanceNorm already normalises per-image colour (test: E8).
The 5x3 extension is low priority unless E6 (external data) shows a colour-related drop.

## Notes
protocol correction: 201 -> 200 patients (C47P8 merged), documented deviation; M3 rows keep the frozen 75 epochs (--set epochs=75), the 5x3 extension uses 50 like E1
