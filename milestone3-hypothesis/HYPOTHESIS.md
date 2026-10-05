# Milestone 3: Hypothesis and prediction (written before any M4 code)

**Team:** Agrapujya Lashkari, Deepanshu Mehta · **Date:** 2026-10-05
**Baseline:** DANet-S reproduced on NIH Malaria, cell-level split: ACC 97.02, SE 96.30, SP 97.75, F1 97.00, ROC-AUC 0.9945.
Paper: 97.95 on its single split, 97.25 as its 5-fold CV mean. See `milestone2-baseline/`.

---

## 1. What the baseline revealed

1. **Every test patient is also a training patient.** The paper's random cell-level split puts cells from all 201 patients into both
   train and test (measured: 100 % of test cells, 201/201 patients). So the reported 97–98 % is a *within-patient* score (M1 gap 01).
2. **Each slide carries a strong colour signature.** Patient ID explains **90 % (R), 74 % (G) and 90 % (B)** of the variance in a cell's mean colour.
   Between patients the stain varies mostly in **saturation (±48 % relative s.d.)** and **hue** (pink vs. lavender smears); brightness varies only ±4 %.
3. **There is a learnable patient shortcut.** 50 patients are healthy-only, and they supply 25 % of all uninfected cells. Under a cell-level split, a
   network can learn "this slide's stain → healthy patient → uninfected" without looking at the parasite.
4. **Sensitivity is the weak metric.** We missed 102 parasitized cells and produced 62 false positives. Our SE is 1.8 pts below the paper.
5. **Grad-CAM is not on the parasite.** It is cold on the visible parasite in correctly classified infected cells (M1 gap 04).

## 2. Hypothesis (one variable)

> **If we add stain-colour augmentation (random saturation, hue, brightness and contrast jitter, calibrated to the measured
> between-patient colour spread) to DANet's training, the model will generalise better to *unseen patients*.**
> **Mechanism:** each slide's stain is a near-unique fingerprint (patient ID explains 74–90 % of colour variance). Randomising colour during training
> removes that fingerprint as a usable cue, which pushes the network toward stain-independent evidence: the parasite's shape and chromatin.
> The gain should appear **only** when test patients are unseen. Under the paper's leaky cell-level split, the fingerprint is
> still valid at test time, so there is nothing to gain.

- **The one variable changed:** the training-time colour augmentation.
  `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.5, hue=0.1)`, applied to every training image.
  Saturation and hue get the large ranges because that is where patients actually differ (§1.2).
  Everything else stays frozen exactly as in the M2 baseline: architecture, optimiser, schedule, epochs and the geometric augmentation.
- **The evaluation protocol** gains a patient-level split. This is a *protocol correction*, measured as its own row (B1), not the hypothesis.

## 3. Experimental design and predictions (fixed now, before running)

Patient-level split: 70/10/20 **by patient**, stratified by patient type (151 infected, 50 healthy-only), giving 40 test patients.
Test-set class balance swings between splits (1,611 to 3,329 parasitized cells across 3 seeds), so we use **3 patient-split seeds** and report mean ± s.d.
The headline metrics are **sensitivity, balanced accuracy and ROC-AUC**, as the handout requires.

| Row | Config | Split | One change vs. previous row | Predicted outcome |
|---|---|---|---|---|
| B0 | Baseline (M2, done) | cell-level | — | ACC 97.02, SE 96.30, AUC 0.9945 (measured) |
| B1 | Baseline | **patient-level** | evaluation protocol | **Bal. ACC drops 1–2.5 pts (≈ 94.5–96 %), SE drops 1.5–3 pts, AUC ≈ 0.985–0.99.** Unseen stains and slide quality, with no patient shortcut |
| **H** | **+ colour augmentation** | patient-level | **colour augmentation** | **vs. B1: SE +0.5 to +1.5 pts, Bal. ACC +0.3 to +1.0 pt, AUC +0.001 to +0.004. The between-split s.d. should shrink** |
| C | + colour augmentation | cell-level | control | **vs. B0: within ±0.3 pt** (no gain when the stain cue is leaked anyway; a small drop is possible) |

**What would falsify the hypothesis:** H ≤ B1 on sensitivity and balanced accuracy, averaged over the 3 splits.

### Why we might be wrong (and what we would learn)
- **DANet may already be largely stain-invariant.** Every 3×3 conv in each Dilated Attention Block is followed by **InstanceNorm**, which removes each image's
  per-channel mean and scale. A global stain shift is then largely normalised away after the first block. If so, B1 will drop
  less than predicted and H will help little. That would itself be a finding: InstanceNorm acts as implicit stain normalisation.
- **Colour itself is a weak label cue.** Mean RGB alone predicts the label at only 61 %, so the leakage benefit may come more from slide-specific
  focus, illumination or cell morphology than from colour. Colour augmentation would not address those.
- **Strong hue jitter could erase real signal.** The parasite's purple chromatin is itself a colour feature. Too much hue shift could cost
  sensitivity, especially in the cell-level control (C).
- **Noise floor:** 40 test patients per split means a split-to-split s.d. of maybe 0.5–1 pt. Effects smaller than that won't be resolvable from 3 splits.

## 4. Alternatives considered (not chosen)

| Candidate | Why not chosen as *the* hypothesis |
|---|---|
| Sigmoid gate instead of channel-softmax (the paper's own Eq. 3) | Interesting paper-vs-code ablation, but the expected effect (≈ 0.2–0.5 pt) is below the single-seed noise floor (±0.45 pt CI) |
| Weight decay or dropout, against the observed overfitting | Best-checkpoint selection already avoids the overfit epochs; a small expected effect |
| Lower decision threshold to buy sensitivity | Just an operating-point choice, not a model change |
| Label cleaning (Fuhad et al.) | Changes the data, which makes comparison with the paper and the baseline unfair |

## 5. Compute plan for M4

6 training runs (B1 and H × 3 patient splits) plus 1 control (C) = 7 runs × ≈ 2.5 h on a Kaggle T4 ≈ 18 GPU-h. That fits in one week's
30 h quota. The notebook from M2 is reused unchanged except for two config switches (`split_mode`, `colour_aug`).
Only the M2 notebook's existing code paths are touched. No M4 code is written before this document is submitted.
