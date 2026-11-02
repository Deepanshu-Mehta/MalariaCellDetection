# E10: Leave-session-out split

**Status:** done, one repeat (Kaggle, 2026-11-02 to 11-03)  ·  **Outcome:** prediction 3 falsified under the rule (gap +.24, below the .29 MDE); no claim either way, because the registered limitation asks for a second repeat when the gap sits near the MDE

## Question
Does "unseen patient" overstate generalisation when patients imaged in the same session share train and test?

## Why this exists
Exploratory checks on 2026-11-02 (`exploratory_checks.py`, `exploratory_results.json`) found that filename dates
(`IMG_YYYYMMDD`) give 56 imaging dates, 31 of which span several E1 test folds. Infection status is also heavily
confounded with date: only 4 dates are shared, covering 7 of 50 healthy-only and 27 of 150 infected patients. E1's
small leakage gap may therefore reflect a split that still leaks session.

## Prediction (written by the AI assistant, 2026-11-02, before any run)
**What the writer had already seen:** date and patient counts, cell-count AUC, parasite-free AUCs over all
patients. **Not seen:** anything from a session-held-out split, or any analysis restricted to shared dates.
1. **Part A, stain:** imaging date explains at least 70% of the colour variance that patient explains.
2. **Part A, parasite-free signal:** restricted to patients on shared dates, RedDino's parasite-free patient AUC
   falls below 0.75 from the all-patient 0.93. The CI will be wide (7 healthy vs 27 infected), so this is
   indicative only.
3. **Part B, primary:** DANet (E1 config) loses BACC moving from patient-grouped to session-grouped folds, with
   a gap (grouped minus session) of about +0.35, plausible range 0 to +0.8.

**Interpretation rule:** a gap of at least 0.29 (E1's MDE) with a CI above 0 means session leakage is real and
the patient-grouped results must be reported as optimistic. A smaller gap means session leakage does not
materially change DANet's cell-level score, and that is reported plainly.

**Falsified if:** the gap is below the 0.29 MDE, for prediction 3. Part A predictions are judged separately.

## Protocol
**Addendum, 2026-11-02, before any run. The prediction above is unchanged.** Validation is also date-separated, so train,
validation and test share no date. The run estimate rises from about 3h to 5.5 run-hours. Blocks are balanced on healthy
patients and cells as well as prevalence, because the largest healthy blocks hold 13, 11 and 8 patients. Final split, from
`scripts/build_session_splits.py`, hashed in `splits/SHA256SUMS`: 50 blocks; healthy per test fold 11/8/13/9/9; test patients
40/41/41/39/39; test prevalence SD .10; validation 16 patients with 4 healthy in every fold. Date is a proxy for session.

- **Part A (no GPU):** variance in foreground colour explained by date vs patient; parasite-free patient AUC
  restricted to shared dates.
- **Part B:** patients linked by any shared date form one block (connected components, since 6 patients span
  several dates). Assign whole blocks to 5 folds, balancing prevalence. Report fold prevalence before any run,
  and hash the new split file as in `splits/SHA256SUMS`.
- Run configuration: `configs/E1_danet.json`, 5 folds, trained on Kaggle (about 3h of quota).
- **Primary metric:** pooled out-of-fold BACC, paired by patient against E1 grouped repeat 0, using the
  patient-cluster bootstrap.
- **Limitation:** one repeat only. If the gap sits near the MDE, a second repeat is needed before any claim.

## Runs
| Date | What | Notes |
|---|---|---|
| 2026-11-02 | (local) `part_a.py` | Part A, no GPU |
| 2026-11-02 to 11-03 | Kaggle `e10-e11-e12-batch`, E1 config, session split r0 f0-4 | 5 runs, all return code 0, about 1.04h each |
| 2026-11-03 | (local) `part_b.py`, `part_b_results.json` | Patient-cluster bootstrap, B=2000 |

## Result
**Part B (2026-11-03, `part_b.py`, `part_b_results.json`).** Pooled out-of-fold, 27,558 cells, E1 repeat 0 against the
session split, paired by cell:

| Split | BACC | SE | SP | AUC |
|---|---|---|---|---|
| Patient-grouped | 96.78 | 96.01 | 97.55 | .9924 |
| Session-grouped | 96.54 | 95.96 | 97.13 | .9925 |
| Gap, grouped minus session | **+.24 [-.06, .59]** | +.05 [-.48, .72] | +.42 [.11, .74] | -.0001 [-.0009, .0007] |

Session-grouped BACC by test fold: 97.58, 94.11, 97.06, 96.87, 97.13. Fold 1 is low because of its sensitivity (SE 90.8);
this was not investigated. False positives on healthy-only patients: 1.96% grouped, 2.20% session.

| Prediction | Outcome |
|---|---|
| 3. Gap about +.35, range 0 to +.8, falsified below .29 | **Falsified by the registered line:** +.24 is below .29 and its CI includes 0. It is inside the plausible range |

**Part A (2026-11-02, `part_a.py`, `part_a_results.json`).** Measures below were fixed in the script
before it ran. Cells: 13,665 Uninfected cells with E7 flags removed.

| Prediction | Outcome |
|---|---|
| 1. Date explains at least 70% of the colour variance patient explains | **Supported:** eta-squared .748 vs .864, ratio 86.5% (86.6% adjusted for group count). By channel: R 90%, G 61%, B 97% |
| 2. RedDino parasite-free AUC on shared dates below .75 | **Missed narrowly:** .767 [.519, .958], from .933 on all patients. n = 7 healthy, 27 infected |

Secondary, same patients: DINOv2 .783 [.513, .979], colour .677 [.434, .894]. Comparing only healthy-infected pairs from the
same date gives RedDino .641 [.313, .959], DINOv2 .667 [.263, 1.0], colour .333 [0, .643]. These rest on few pairs, so they
are indicative only: 39 pairs over 4 dates, three of which hold a single healthy patient. The CIs are too wide to say how much of the parasite-free signal is batch. Part A does show that most
of the colour signal sits at date level.

## Verdict and what we learned
**Prediction 3 is falsified, and the rule does not call session leakage real.** The gap is below E1's MDE and its CI
includes 0. On this one repeat, moving to session-held-out folds did not materially change DANet's cell-level BACC, and
ranking did not move at all (AUC gap -.0001).

**This is not evidence that session leakage is absent.** The estimate sits close to the MDE, the CI reaches +.59, and the
protocol's limitation says a gap near the MDE needs a second repeat before any claim. No second repeat was run. The patient-grouped
results are therefore not reported as optimistic, and they are not reported as cleared either. The only secondary interval that
excludes 0 is specificity (+.42 [.11, .74]). It was not a registered test and is not adjusted for multiple comparisons.

Part A still stands: date carries most of the colour signal patient carries. Together, A and B say the batch signal is real in
the data but costs DANet little on NIH at the cell level, within the limits of one repeat. Date is only a proxy for session.
