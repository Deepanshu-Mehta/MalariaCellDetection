# E3: Random-patient-label permutation null

**Status:** done (10/10 valid runs)  ·  **Outcome:** directionally confirmed, magnitude far below prediction

> **Correction, 2026-11-02:** this experiment does not measure capacity. Final `train_acc` was at most 0.60 in 6 of 10
> runs, so the 30-epoch runs underfit. It also has no LayerNorm arm, so it cannot support any InstanceNorm claim. The
> statements below that DANet has "less capacity" and that this "strengthens" the InstanceNorm story are **withdrawn**.
> What stands: under this training budget, random-patient labels were not learned well on either split. The text below
> is kept as the original record.

## Question
Can DANet memorise patients when labels carry no parasite signal?

## Prediction (written before any run; date it)
**Written by the AI assistant, 2026-10-29, before any run.** Reasoning: E1 found DANet's real leaky-vs-grouped
gap is small (+0.26 BACC), but that measures RELIANCE on patient identity for the real task, not CAPACITY to
memorise it -- a different question (as the MRI analogue, Yagis 2021, shows: ~96% random-label accuracy on a
leaky split vs ~50% on a grouped one). DANet has 2.3M parameters and, under the leaky split, sees many of a
test patient's other cells during training.
**Prediction:** leaky-split random-patient-label accuracy well above chance, BACC > 80% (most likely 90%+),
because the network can fit a per-patient visual signature (colour, per E2's probe, and/or texture/background)
even when the label carries zero parasite information. Grouped-split random-label accuracy at or near chance,
BACC ~50% (+/- 5 pts), since grouped test patients are never seen in training under any label scheme.
**Falsified if:** grouped-split accuracy is also well above 50% -- that would point to a leak in the fold
construction itself (re-check `splits/grouped_folds.csv` immediately), not genuine capacity.

## Protocol
- Splits / runs: 5 label draws x (leaky + 1 grouped split)
- Config: `research/configs/E3_random_patient.json`
- Primary metric: BACC leaky vs grouped
- Estimated T4 GPU-h: 10

## Runs

| Date | Job slug | Runs | Notes |
|---|---|---|---|
| 2026-10-29 | `e3-random-patient-label` (first attempt, DISCARDED) | 10 jobs | **Invalid**: a real bug in `train.py` meant the output path didn't vary with `label_draw`, so draws 1-4 silently resumed draw 0's finished checkpoint and evaluated it against their own (different) labels. Fixed in `train.py` (path now includes a signature of every `--set` override; checkpoints store a config fingerprint and refuse to resume into a mismatched one). All outputs from this run were deleted. |
| 2026-10-29 | `e3-random-patient-label-v2` | 10/10 valid | ~0.6h each, confirmed distinct output dirs and correct `label_draw` per run (`analysis.py` checks this) |

## Result
`analysis.py` -> `results.json`. BACC is against each draw's own random labels (chance = 50%).

| Split | Draw 0 | Draw 1 | Draw 2 | Draw 3 | Draw 4 | Mean +/- SD |
|---|---|---|---|---|---|---|
| Grouped | 50.0 | 53.6 | 50.0 | **60.7** | 51.1 | 53.1 +/- 4.5 |
| Leaky | 57.7 | 54.9 | 57.0 | 59.7 | **74.5** | 60.8 +/- 7.9 |

Each draw uses the identical patient-label assignment on both splits, so the per-draw (leaky - grouped)
difference is a fair paired comparison: **+7.7, +1.3, +7.0, -0.9, +23.4** -- 4 of 5 draws positive (leaky
BACC higher), the predicted direction, but nowhere near the predicted leaky BACC > 80%.

**Draw 3's grouped outlier (60.7%) investigated, not a leak:** its training curve shows val_acc 93% against
train_acc only 64% -- the reverse of normal overfitting. With only 16 inner-validation patients and no real
signal to learn under this null, best-val-loss checkpoint selection is itself noisy; a small, non-representative
validation set can pick an epoch that doesn't generalise even to the 40 test patients in the same fold. This
does not recur in the other 4 draws, which argues against a genuine fold-construction leak (that would be
expected to show up consistently, not in one draw out of five).

## Verdict and what we learned
**Directionally confirmed, magnitude far below prediction.** DANet has much less capacity to memorise random
patient labels under the leaky split than predicted (mean 60.8% vs a predicted >80%, with only one of five
draws reaching the predicted range). This is a more informative result than a clean pass or fail: it
**strengthens** the InstanceNorm-suppresses-patient-shortcuts story running through E1/E2/E8, rather than
undermining the permutation-test design. If DANet's architecture makes patient identity hard to encode and
use (E2: embeddings carry much less patient-ID signal under InstanceNorm than LayerNorm; E8: InstanceNorm
arms show almost no leakage-gap profit), the network should also struggle to deliberately memorise patient
identity when forced to -- which is exactly what this experiment shows. The MRI analogue (Yagis 2021, ~96%
leaky vs ~50% grouped) used a very different architecture and task (slice-level MRI classification with no
equivalent of InstanceNorm's per-image colour normalisation); DANet's cell-level task with heavy per-image
normalisation looks like a genuinely harder substrate for memorisation.

**Caveat on checkpoint selection under a null:** future permutation-null experiments on this codebase should
consider a larger inner-validation set, or last-checkpoint rather than best-val-loss selection, since
best-val-loss selection is itself noisy when there is no real signal for the validation loss to track.

## Notes
Capacity test, not proof of reliance -- E1's small real-task leakage gap (+0.26 BACC) already showed DANet
does not *need* patient identity for its real decisions; this experiment asked whether it *could* use it if
forced to, and found a weaker capacity than predicted, for a plausible architectural reason (InstanceNorm).
Two bugs were found and fixed in the course of running this experiment: the path-collision bug above (real,
in `train.py`, affects any future experiment using `--set` overrides), and a checkpoint-selection noise issue
specific to interpreting a single outlier draw (not a code bug, a statistical-design caveat for future null
experiments).
