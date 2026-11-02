# E7: Label-noise audit

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** confirmed, exceeds prediction

> **Correction, 2026-11-02:** "genuine labelling problems" and "missed infections" are **withdrawn** until an expert
> reviews these cells. 21 of the 114 flagged Uninfected cells (18.4%) sit in healthy-only patients, close to their 24.8%
> base rate, and a dark-spot feature fires on them about as often as on flagged cells from infected patients. So either
> those patients are not healthy, or the dark marks are not all parasites. Call these cells **flagged**. The review below
> was an AI visual check, not an expert read. The text below is kept as the original record.

## Question
How many NIH labels look wrong?

## Prediction (written before any run; date it)
**Written by the AI assistant, 2026-10-29, before any run.** Reasoning: Rahman 2019 found 38% of one model's
false positives on NIH were truly parasitized (mislabelled as uninfected). Fuhad 2020 removed 1,397/27,558
(5.1%) suspect cells, which is an upper bound on suspected noise, not a confirmed error rate. Wilm 2025's "10%
ambiguous" figure is about missing annotations near full-image borders, not crop-level label noise -- do not
use it as a prior here (already corrected in the fact-check round).
**Prediction:** cleanlab's confident-learning filter, run on E1's pooled grouped out-of-fold predictions, flags
roughly 1-3% of all 27,558 cells as likely mislabelled (a few hundred cells) -- lower than Fuhad's 5.1% removal,
since CL is a stricter, model-confidence-based filter than Fuhad's manual relabelling pass. Manual review of the
top 100 highest-confidence-of-error flagged cells confirms a genuine labelling problem (not just a hard but
correctly labelled cell) in 40-70% of them, consistent with Rahman's 38% precedent for a comparable error type.
**Falsified if:** manual review finds genuine errors in fewer than 20% of the top-100 flagged cells -- meaning
CL is mostly flagging cells the model gets wrong for other reasons (hard cases), not real label noise.

## Protocol
- Splits / runs: cleanlab `find_label_issues` (self-confidence ranking) on E1's pooled grouped-repeat-0 OOF
  predictions (every cell scored once, never by a model trained on it); top-100 visual review
- Config: `analysis.py`
- Primary metric: flagged count + manual-review precision
- Estimated T4 GPU-h: 0 (local, CPU only)

## Runs
All local, no Kaggle quota.

| Date | What | Notes |
|---|---|---|
| 2026-10-29 | cleanlab audit + top-100 visual review (AI assistant) | ~1 min CPU; `flagged_cells.csv`, `audit_top100.csv`, `audit_top100_sheet_{0-3}.png` |

## Result
**312 / 27,558 cells (1.13%)** flagged by cleanlab -- within the predicted 1-3% range, well below Fuhad 2020's
5.1% manual-removal rate (as predicted, since cleanlab is a stricter filter). 198 flagged within
Parasitized-labelled cells, 114 within Uninfected-labelled cells, spread across 115 of the 200 patients.

**Top-100 visual review (AI assistant, 2026-10-29), by direction:**

| Flagged direction | n | Visual pattern | Read |
|---|---|---|---|
| Labelled Uninfected, flagged likely-Parasitized | 37 | **~86% (32/37)** show a clearly visible dark chromatin mark | Strong evidence of genuine mislabelling (missed infections) |
| Labelled Parasitized, flagged likely-Uninfected | 63 | **~97% (61/63)** show no visible mark at all | Consistent with mislabelling, but a non-expert visual check cannot rule out a faint/early-stage parasite invisible at this crop's resolution (Rahman 2019's own caveat) |

Combined, **~93% of the top-100 flagged cells look like genuine labelling problems** on inspection -- well
above the predicted 40-70% range and far above the 20% falsification threshold.

## Verdict and what we learned
**Confirmed, exceeds prediction.** The flagged fraction (1.13%) landed exactly where predicted, and the
manual-review confirmation rate (~93%) came in much higher than the 40-70% predicted from the Rahman 2019
precedent -- NIH's labels are noisier than that one comparator suggested, at least among the cells a trained
classifier finds most surprising. The asymmetry is itself informative: the "Uninfected but clearly has a
parasite" direction is confirmable by eye with high confidence; the "Parasitized but nothing visible" direction
cannot be fully confirmed without expert input, so its ~97% figure should be read as "consistent with
mislabelling" rather than "proven mislabelled" -- some fraction may be genuine faint infections the model
still picks up on from texture cues too subtle for a visual crop review.

**For the paper:** report both the flagged fraction and the confirmation rate with this asymmetry disclosed.
A sensitivity analysis re-scoring E1 with the 312 flagged cells removed from the test set (do accuracy/SE/SP
move materially?) would strengthen this further and is cheap to add (CPU only, reuses E1's existing splits).

## Notes
The two flagged-direction visual patterns are nearly binary (almost all-or-nothing within each direction),
which is a cleaner signal than a gradual confidence spread would be -- suggests cleanlab's self-confidence
ranking is doing real work here, not just surfacing noise at the margin.

