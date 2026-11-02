# E0: Data integrity and frozen splits

**Status:** done  ·  **Outcome:** positive (positive / negative / invalid)

## Question
Are the patient groups and folds leakage-free and balanced?

## Prediction (written before any run; date it)
200 patients (150+50); fold prevalence SD < 3 pp

## Protocol
- Splits / runs: research/scripts/phase0_build.py
- Config: `research/-`
- Primary metric: assertions + fold prevalence SD
- Estimated T4 GPU-h: 0

## Runs
Kaggle job slugs and dates (results are aggregated by `research/scripts/collect_results.py` into
`research/results/all_runs.csv`):

| Date | Job slug | Runs | Notes |
|---|---|---|---|

## Result
_Filled after the runs: effect size with CI, compared with the prediction._

## Verdict and what we learned
_Positive / negative / invalid, and why. Negative results stay here and go into the paper's appendix._

## Notes
splits hashed in research/splits/SHA256SUMS
