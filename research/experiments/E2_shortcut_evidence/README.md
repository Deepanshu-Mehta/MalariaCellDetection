# E2: Shortcut evidence

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** mixed / partially confirmed (see Verdict)

> **Note, 2026-11-02:** part B (colour separating healthy-only from infected patients) is confounded with acquisition:
> only 7 of 50 healthy-only patients share an imaging date with any infected patient. It may reflect batch rather than a
> biological or label-linked signal. E10 part A tests this. The text below is kept as the original record.

## Question
Do DANet features encode patient identity / stain?

## Prediction (written before any run; date it)
**Written 2026-10-29, before any E2 run, informed by E8's completed result (InstanceNorm vs LayerNorm):**
- Colour-only classifier (5 foreground colour features, grouped CV): well above chance (BACC > 65%), since
  M3 (2026-10-05) measured mean RGB alone at 61% on a smaller sample.
- Healthy-only vs infected-patient probe (uninfected cells only): separable above chance in RGB/HSV space.
- Patient-ID probe on frozen embeddings (within-patient split): **E8b (LayerNorm) embeddings predict patient
  ID much better than E8c (InstanceNorm)** embeddings, given E8's finding that LN profits from leakage while
  grouped accuracy is unchanged. E1 (code-faithful, InstanceNorm) should pattern with E8c.
- Stain-swap flip rate (Reinhard, foreground-only): **E8b flips more predictions than E8c/E1** when a cell's
  colour is remapped to another patient's stain statistics, especially for uninfected cells and especially for
  healthy-only -> infected-patient swaps (the empty-group case noted in the phased-plan report).
**Falsified if:** the patient-ID probe and the stain-swap flip rate do not differ between E8b and E8c beyond
their bootstrap CIs — in that case InstanceNorm is not the mechanism and E8's gap difference needs another
explanation (e.g. the gate or dilation structure, or noise).

## Protocol
Four parts, all run locally (no Kaggle quota needed), reusing E1/E8b/E8c's repeat-0 checkpoints:
- **A.** Colour-only label classifier (5 foreground colour features: R, G, B, HSV saturation, circular-mean
  hue), logistic regression, same patient-grouped folds as E1. `analysis_colour.py`.
- **B.** Healthy-only vs infected-patient probe on uninfected cells only (no parasite-colour confound).
  `analysis_colour.py`.
- **C.** Frozen-embedding extraction (DANet's 128-dim pool output, OOF per fold) for E1/E8b/E8c, then a
  within-patient 80/20 patient-ID probe (logistic regression + 5-NN). `extract_embeddings.py`,
  `analysis_embeddings.py`.
- **D.** Reinhard stain-swap counterfactual (foreground-only LAB stats, background re-zeroed), flip rate by
  patient type / label / condition, with patient-cluster bootstrap CIs (B=5000).
  `analysis_stainswap.py`, `analysis_stainswap_stats.py`.

## Runs
All local; no Kaggle jobs. Run log:

| Date | What | Notes |
|---|---|---|
| 2026-10-29 | A, B (colour-only analysis) | ~2 min, CPU only |
| 2026-10-29 | C (embedding extraction) | First attempt stalled ~9h on laptop sleep (fixed: plugged in, caffeinate); then hit a real bug — no MPS cache release between folds grew to 7.5GB RSS, caused system-wide swap thrashing and thermal throttling (pace degraded from 3 to 35+ min/fold). Fixed (release cache + per-fold checkpointing) and reran: all 15 folds (3 arms x 5) in ~25 min |
| 2026-10-29 | C (patient-ID probe) | First run crashed: `np.load(allow_pickle=False)` on a string patient-ID array. One-line fix (`allow_pickle=True`), reran successfully |
| 2026-10-29 | D (stain-swap) | ~35 min for all 3 arms, ~1,280 source cells/fold x 5 folds x 3 arms |

## Result
**A. Colour-only label classifier:** BACC 63.98 (SE 57.94, SP 70.03) predicting Parasitized/Uninfected from
5 colour numbers alone, patient-grouped CV. Matches M3's 61% estimate on a smaller sample.

**B. Healthy-only vs infected-patient probe (uninfected cells only, no parasite colour):** BACC **78.20**
(SE 64.98, SP 91.43) telling a healthy-only patient's red cell from an infected patient's red cell, from
colour alone. This is direct, parasite-free evidence of a stain/slide shortcut in the data.

**C. Patient-ID probe (within-patient 80/20 split, frozen embeddings, chance ~0.7% for 150 infected patients):**

| Arm | Norm | LR accuracy (infected patients) | LR accuracy (all 200) |
|---|---|---|---|
| E1 (code-faithful) | InstanceNorm | 25.1% | 22.0% |
| E8b (paper-faithful) | **LayerNorm** | **39.8%** | **37.0%** |
| E8c (paper-faithful) | InstanceNorm | 27.9% | 24.3% |

E8b >> E1 ~= E8c. **This confirms the predicted mechanism:** LayerNorm embeddings carry far more linearly
decodable patient identity than InstanceNorm embeddings, and the gate/dilation structure (shared by E8b/E8c,
different in E1) does not explain the gap -- the norm layer does.

**D. Stain-swap flip rate (patient-cluster bootstrap, B=5000), uninfected cells:**

| Comparison | All uninfected-cell conditions (control + cross-type, healthy and infected patients) |
|---|---|
| E8b vs E1 | **No detectable difference** -- every 95% CI includes 0 |
| E8b vs E8c | **No detectable difference** -- every 95% CI includes 0 |
| E8c vs E1 | **No detectable difference** -- every 95% CI includes 0 |

Flip rates themselves are low and similar across arms: 2.0-5.1% depending on condition. **This part of the
prediction is NOT supported.**

One unplanned, statistically significant result: on **parasitized** cells, E8c flips more than E1
(+0.033, CI [0.017, 0.049]) and more than E8b (~+0.029, analogous CI excludes 0). Not interpreted as shortcut
evidence: a Reinhard swap on a parasitized cell also recolours the parasite's own chromatin stain, so a flip
there conflates "lost the stain shortcut" with "lost real diagnostic evidence" -- exactly the confound the
protocol doc flagged when restricting parasitized cells to a control-only condition.

## Verdict and what we learned
**Mixed / partially confirmed.** The pre-registered falsification criterion was two-part (patient-ID probe
AND stain-swap flip rate must differ between E8b and E8c), and only the first part held:

- **Representational claim (confirmed):** LayerNorm's embeddings are measurably more patient-identifiable than
  InstanceNorm's, cleanly isolating the norm layer as the cause (E8c ~= E1, both InstanceNorm, despite E8c
  sharing E8b's gate and dilation structure).
- **Behavioural claim (not confirmed):** that extra patient-identity content in the embedding does not
  translate into a measurably different sensitivity to a global Reinhard stain-swap on uninfected cells.

**What this means for the paper's thesis:** the norm layer changes *what the network's representation
encodes* about patient identity (part C), and separately changes *how much the network profits from patient
leakage on the real task* (E8's accuracy-gap finding) -- but a simple global colour-swap counterfactual is not
sensitive enough to detect a corresponding behavioural difference. That dissociation is itself worth reporting:
it is evidence against treating "the model encodes X in its features" and "the model measurably uses X at
decision time" as interchangeable, and it is the kind of gap a channel-level or concept-based analysis (CRP/CAV,
not attempted here) could plausibly resolve, since it operates on what each embedding dimension contributes to
the actual classification decision rather than linear decodability of the whole embedding.

**Limits:** one grouped repeat (r0) per arm, Reinhard is one specific (global, LAB, foreground-only) colour
transform -- a texture-preserving or more aggressive transform might show a behavioural effect this one didn't.
Colour probes (A, B) have no parasite mask for NIH cells, so Parasitized-cell colour includes the parasite's
own chromatin stain for part A (disclosed in `analysis_colour.py`); part B avoids this by using uninfected
cells only.

## Notes
Three rival explanations for part B's 78% separability remain open, not yet distinguished: stain/slide
shortcut, label noise (early rings in "uninfected" cells from infected patients), host-RBC change (parasite-free
infection signal, per the Malaria Journal 2026 finding). E7 (label-noise audit) and a greyscale/stain-normalised
variant of part B would help separate these -- not yet run.
