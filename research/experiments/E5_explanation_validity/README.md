# E5: Explanation validity

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** confirmed directionally, magnitude far exceeds prediction -- important scope caveat (see Verdict)

**Correction, 2026-11-02 (E5b):** the fold-0 CAM rates below are not representative. Across five folds, CAM hits on true positives are E1 71.6 +/- 13.9, E8b 94.5 +/- 2.3, E8c
42.8 +/- 43.6, and the darkest-pixel rule beats CAM on all 15 checkpoints. See `../E5b_five_checkpoint_explanations/README.md`.

> **Correction, 2026-11-02:** the ~15% chance level understated the baseline, because the dilated mask covers about 27%
> of the cell foreground. A fixed image-centre point hits about 50% of true positives, and a no-model darkest-pixel rule
> hits about 98%, above Grad-CAM's 78-94% (`exploratory/2026-11-02_paper_readiness/verify_corrections.json`). The inpaint
> result fits any detector that responds to a dark object. So the supported claim is narrower: predictions depend on
> the annotated dark object. **This does not show the model learned parasite morphology**, and the "causal" wording
> below is withdrawn. Results come from one checkpoint per arm, so E6b's variation applies here too. The text below is
> kept as the original record.

## Question
Does saliency land on parasites?

## Prediction (written before any run; date it)
**Written by the AI assistant, 2026-10-29, before any run.** Reasoning: M2 already found Grad-CAM "cold on
parasites, hot on the cell rim." The nearest quantitative precedent, MalariAI (arXiv 2607.00385), found
Grad-CAM++ energy-in-box only +0.0485 over chance, and only at CELL level, not parasite level -- a much easier
target than this experiment's parasite-level pointing game.
**Prediction:** on MP-IDB expert masks, DANet's Grad-CAM pointing-game hit rate on true positives will be
modest, in the 30-55% range (clearly above geometric chance for a small parasite in a cell-sized box, but far
below what a clinically trustworthy map would need), and energy-in-parasite-mask will exceed chance by a small
margin, roughly +0.02 to +0.06 (smaller than MalariAI's cell-level +0.0485, because parasite-level targets are
harder). Parasite-inpaint vs random-inpaint flip rate: parasite-inpainting flips the prediction MORE often than
random-region inpainting (predict roughly 40-60% vs 10-20%), showing the model uses some real, if poorly
localised, parasite evidence. Cascading-randomisation sanity check: the map is NOT invariant to randomising the
model's weights (passes the basic Adebayo 2018 check), since most trained CNN Grad-CAMs do.
**Falsified if:** the pointing-game hit rate is at or below geometric chance (the map carries zero spatial
information about the parasite), or the parasite-inpaint and random-inpaint flip rates are statistically
indistinguishable (the model isn't using localised parasite evidence at all).

## Protocol
- **Masks:** `build_parasite_masks.py` builds a parasite-pixel mask for all 963 MP-IDB Parasitized crops, in
  the exact crop frame already saved by `scripts/extract_mpidb.py`, reusing the cached Cellpose segmentation
  (no re-segmentation). Alignment checked visually (see `colourtransfer_check.png`-style spot check in E4b's
  log; E5's own check confirmed masks sit on the visible dark chromatin in every sampled cell).
- **Grad-CAM + metrics:** `analysis.py`, run locally (CPU/MPS, no Kaggle quota), on E1/E8b/E8c's grouped
  repeat-0 fold-0 checkpoints (none of which ever trained on MP-IDB). For each Parasitized crop: Grad-CAM at
  the last DAB block, pointing game (CAM peak within a 6px-dilated mask), energy ratio (CAM energy inside the
  mask vs the mask's own area fraction = chance), parasite-inpaint vs random-inpaint flip rate (in-distribution
  fill = the cell's own foreground mean colour), and a single-layer cascading-randomisation sanity check
  (Adebayo 2018 style, last block only).
- **Primary metrics computed on TRUE POSITIVES ONLY** (crops where the model's own prediction is Parasitized),
  per the registered prediction wording -- a model's explanation for a wrong prediction isn't informative
  about parasite localisation.
- Config: `build_parasite_masks.py`, `analysis.py` (local, no JSON config)
- Primary metric: pointing game, energy-in-mask, inpaint flip rates
- Estimated T4 GPU-h: 0 (local)

## Runs

| Date | What | Notes |
|---|---|---|
| 2026-10-29 | `build_parasite_masks.py` | 963/963 crops got a mask, 0 skipped; alignment spot-checked visually |
| 2026-10-29 | `analysis.py`, 3 arms | ~2-3 min total; one bug found and fixed (see below) |

**Bug found and fixed:** the cascading-randomisation sanity check initially returned NaN for E8b. Traced to
one of the 20 sampled cells producing a constant (all-zero) CAM after an unlucky random reinitialisation of
the last block (a known ReLU edge case: a bad random draw can zero out every unit). Fixed by skipping
zero-variance CAM pairs and reporting how many of the 20 draws were valid, rather than letting one bad draw
silently NaN the whole statistic.

## Result

**Accuracy on MP-IDB Parasitized cells (external domain, not yet formally scored in E6):** 52.5% (E1), 60.2%
(E8b), 65.9% (E8c) -- roughly half to two-thirds correct, as expected for an out-of-domain test set.

**On true positives only** (506-635 cells depending on arm):

| Arm | n (TP) | Pointing game | Chance (mask area) | Energy | Energy - chance | Flip (parasite) | Flip (random) | Sanity (1-corr) |
|---|---|---|---|---|---|---|---|---|
| E1 | 506 | **0.777** | 0.146 | 0.400 | **+0.254** | **0.927** | 0.018 | 0.747 |
| E8b | 580 | **0.941** | 0.148 | 0.374 | **+0.227** | **0.953** | 0.019 | 0.381 |
| E8c | 635 | **0.904** | 0.143 | 0.383 | **+0.241** | **0.928** | 0.005 | 1.003 |

All three arms pass the sanity check (CAM changes after randomising the last block; none is near 0, which
would indicate the map ignores the model's weights). The parasite-inpaint vs random-inpaint flip rates show a
very clean dissociation: removing the parasite flips the prediction ~93-95% of the time, removing a random
same-size foreground region flips it under 2% of the time.

**Selection-effect check (does the model just get the obvious, easy-to-localise parasites right?):**
pointing-game success correlates weakly with parasite mask size (r = 0.09-0.12 across arms), and mean mask
size is nearly identical between true positives and missed cells (e.g. E1: 0.146 vs 0.153). Even in the
smallest-parasite quartile, pointing-game rate is still 68.5% (E1) to 89.7% (E8b) -- the result does not
reduce to "the model only localises the big, obvious cases."

## Verdict and what we learned
**Confirmed directionally; the predicted magnitude was far too conservative in the positive direction** (not
falsified by either of the two registered falsification conditions). Pointing-game hit rate (78-94%) is
roughly double the predicted 30-55% ceiling, energy-above-chance (+0.23-0.25) is about 5x the predicted
+0.02-0.06 and 5x MalariAI's cell-level +0.0485, and the inpaint flip-rate dissociation (93-95% vs <2%) is far
cleaner than the predicted 40-60%/10-20%.

**Important scope caveat, stated plainly:** this is the first quantitative parasite-level localisation result
for this model, and it is measured on **MP-IDB**, an external dataset with a different stain and source
institution, not NIH. **It does not directly confirm or refute the team's own M2 qualitative finding on NIH**
("Grad-CAM is cold on parasites, hot on the cell rim"), because no NIH parasite ground truth exists to run the
identical quantitative test there -- the protocol's secondary NIH SAM-mask pass (a 200-cell audit) was not
built in this pass. Two readings are both consistent with the data:
- DANet's explanations might genuinely be better/more localised than the earlier NIH impression suggested, and
  the M2 observation was an unrepresentative qualitative sample (only ~8 images in a grid).
- Or MP-IDB's parasites (which include trophozoite/schizont/gametocyte stages with visually larger, more
  distinct chromatin, not just NIH's typical early rings) may simply be an easier localisation target, with
  no bearing on NIH-domain explanation quality.
The weak size-correlation check argues somewhat against the second explanation within MP-IDB itself, but
cannot rule out a between-dataset difficulty gap. **This should be stated as an open question in the paper**,
not resolved by this experiment -- building the NIH pseudo-mask pass (SAM-family, 200-cell audit) would
settle it and is a natural, already-scoped next step.

**For the paper:** the flip-rate dissociation (93-95% vs <2%) is the strongest, least confound-prone piece of
evidence here -- a direct causal test that does not depend on CAM geometry or mask precision, and it says
clearly that DANet's prediction causally depends on the parasite region specifically, not just on general cell
appearance, at least on MP-IDB. The E1 vs E8b vs E8c comparison does not show a large norm-layer effect on
explanation quality (all three localise comparably well); if anything E8b (LayerNorm) scores a little higher,
the opposite of what a stain-shortcut story might predict, so this experiment does not extend the E8/E2 norm
narrative in a new direction -- it is a mostly independent, positive finding about MP-IDB specifically.

## Notes
The parasite-inpaint fill colour (cell's own foreground mean) is a simple in-distribution fill; it does not
match more careful ROAD-style infilling exactly, and a generative inpainting model would be a stronger version
of this test for a later phase. The NIH pseudo-mask pass remains open (Phase 2 candidate): SAM-family masks on
NIH cells, audited on 200 hand-checked cells, would let this exact pipeline run on the in-domain data and
directly answer the open question above.

