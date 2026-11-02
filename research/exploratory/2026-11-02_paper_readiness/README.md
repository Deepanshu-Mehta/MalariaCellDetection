# Exploratory checks, 2026-11-02

Everything here is **exploratory and unregistered**. The dark-spot feature was chosen after E5 had been seen, and
MP-IDB has been reused many times, so none of it is confirmatory. BBBC041 is the clean test.

| File | What it is |
|---|---|
| `dark_spot_stress.py`, `.json` | Independent reimplementation of the dark-spot feature, with stress tests and the E6c condition C diagnosis |
| `feature_extremes.png` | Top: highest-scoring MP-IDB Uninfected crops. Bottom: lowest-scoring Parasitized crops |
| `verify_corrections.py`, `.json` | Recomputes the numbers behind the 2026-11-02 corrections to E4b, E5 and E9 |
| `other_session_verbatim/` | Scripts and written-first predictions from a parallel session, copied unchanged. Paths are absolute, so they do not run elsewhere |

The date, bag-size and receptive-field checks are in `experiments/E10_leave_session_out/exploratory_checks.py`.

**Definition checked here:** `(median - min) / median` of the smoothed green channel inside the eroded cell, at 128 px,
sigma 1.5, erosion 4. Threshold selection is sensitive to the search grid (.295 here, .316 in the parallel session), so
a registered test must freeze one definition and one threshold.
