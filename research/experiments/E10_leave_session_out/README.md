# E10: Leave-session-out split

**Status:** registered 2026-11-02, not yet run (needs Kaggle quota)  ·  **Outcome:** pending

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

## Result
**Part A (2026-11-02, `part_a.py`, `part_a_results.json`). Part B pending.** Measures below were fixed in the script
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
Pending.
