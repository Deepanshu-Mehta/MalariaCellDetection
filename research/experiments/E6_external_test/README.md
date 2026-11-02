# E6: External test

**Status:** done (local, no Kaggle quota)  ·  **Outcome:** FALSIFIED (norm layer does not explain external generalisation)

> **Correction, 2026-11-02 (E6b):** this experiment scored one checkpoint per arm, and its CIs resample images only.
> Scoring all five folds per arm shows no robust architecture ranking on MP-IDB: E8b-E1 +1.67 BACC [-1.98, 5.42],
> E8c-E1 -2.14 [-7.84, 3.24]. The statements below that E1 is significantly worse, and that gate/dilation structure
> is the likely driver, are **withdrawn**. The original prediction (InstanceNorm generalises better) remains
> unsupported. The text below is kept as the original record. See `E6b_five_checkpoint_external_rescore/README.md`.

## Question
How much does performance drop outside NIH?

## Prediction (written before any run; date it)
**Written by the AI assistant, 2026-10-29, before any run.** Reasoning: MalariaNet (a comparably-sized model)
reports NIH-to-MP-IDB 79.5% accuracy (SE 74.0, SP 82.7) and NIH-to-BBBC041 53.2% (SE 96.4, SP 18.0). Our own
contact-sheet comparison (2026-10-23) shows MP-IDB's Giemsa prep looks visibly more yellow-brown than NIH's
pink-lavender. E8 found LayerNorm (E8b) profits from patient leakage inside NIH without a grouped-accuracy
gain, while InstanceNorm (E1, E8c) does not -- the natural external-validity question is whether that holds up
off-domain too.
**Prediction:** all three arms (E1, E8b, E8c) drop substantially on MP-IDB, landing in the 60-80% BACC range
(roughly MalariaNet's ballpark, possibly lower since these crops come from a different, Cellpose-based
segmentation pipeline, not MP-IDB's own). Sensitivity drops more than specificity (more missed infections than
false alarms), since DANet was tuned to NIH's specific stain. **The InstanceNorm arms (E1, E8c) generalise
better (smaller drop from their own NIH grouped level) than the LayerNorm arm (E8b)**, consistent with E8's
reading that InstanceNorm discards patient/stain-specific signal that doesn't transfer across datasets.
**Falsified if:** E8b (LayerNorm) generalises as well as or better than E1/E8c on MP-IDB -- that would weaken
the InstanceNorm-protects-against-shortcuts story at the external-validity level, not just the internal one.

## Protocol
- Splits / runs: fixed crop protocol (`protocols/external_crop_protocol.md`); score each test set once
- `analysis.py`: scores E1/E8b/E8c's grouped-repeat-0 fold-0 checkpoints (none ever trained on MP-IDB) on all
  14,468 MP-IDB crops once each. Compares BACC against each arm's own NIH grouped-r0 level, with an
  image-level cluster bootstrap (MP-IDB's nearest available unit to "patient"). BBBC041 (2.3GB) moved to a
  later phase, not scored here.
- Config: `analysis.py` (local, no JSON config; reuses E5's `to_tensor` and E2's `ckpt_path`)
- Primary metric: BACC, SE, SP on MP-IDB Pf
- Estimated T4 GPU-h: 0 (local)

## Runs

| Date | What | Notes |
|---|---|---|
| 2026-10-23 | (local) `extract_mpidb.py` | 963 Parasitized + 13,505 Uninfected crops; leukocyte rule revised once (see protocol) |
| 2026-10-29 | (local) `analysis.py` | ~40 min, all 3 arms scored once each on all 14,468 crops |

## Result
**BACC (image-cluster bootstrap 95% CI):**

| Arm | BACC | SE | SP | AUC | NIH grouped-r0 BACC | Drop |
|---|---|---|---|---|---|---|
| E1 | 74.12 [70.76, 77.34] | 52.54 | 95.69 | 0.867 | 96.78 | **+22.66** |
| E8b | 78.83 [75.86, 81.86] | 60.23 | 97.43 | 0.891 | 96.71 | **+17.88** |
| E8c | 79.08 [75.65, 82.45] | 65.94 | 92.23 | 0.880 | 96.78 | **+17.70** |

All three land within the predicted 60-80% BACC range, and sensitivity drops far more than specificity in
every arm, as predicted. **But the core directional prediction is wrong:** E1 (InstanceNorm) has the
*largest* drop, not the smallest.

**Paired bootstrap on identical MP-IDB crops** (image-clustered, B=5000) -- the right comparison, since all
three arms are scored on exactly the same cells:

| Comparison | Diff (BACC) | 95% CI | Significant? |
|---|---|---|---|
| E8b - E1 | +4.71 | [+2.47, +7.07] | **Yes** |
| E8c - E1 | +4.97 | [+2.71, +7.27] | **Yes** |
| E8b - E8c | -0.25 | [-2.53, +2.04] | No |

## Verdict and what we learned
**Falsified by its own registered condition.** E8b (LayerNorm) generalises to MP-IDB as well as E8c and
significantly *better* than E1 (both InstanceNorm). Since E8b and E8c differ only in their norm layer and are
statistically indistinguishable here, **the norm layer does not explain cross-domain robustness** -- whatever
separates E1 from {E8b, E8c} must be their shared gate (sigmoid vs softmax) and/or dilation structure
(parallel vs cascade), not normalisation.

**This genuinely complicates the project's running thesis, and should be reported as such, not reframed to
fit.** The norm layer (InstanceNorm vs LayerNorm) explains two things well: E8's internal leakage-profit gap
(+0.63 vs +0.02 BACC) and E2's patient-ID probe decodability (39.8% vs ~25-28%). It does **not** explain
external-domain robustness, where the gate/dilation structure appears to matter more. A model can resist
exploiting in-domain patient leakage (InstanceNorm) while still generalising worse out-of-domain than an
architecture that profits more from in-domain leakage (LayerNorm) -- these are not the same property, and
conflating them would overstate what E8/E2 showed.

**For the paper:** present this as three separate, honestly scoped claims rather than one unified "InstanceNorm
fixes everything" story: (1) InstanceNorm limits exploitable patient-identity leakage in-domain (E2, E3, E8);
(2) InstanceNorm does not, by itself, improve out-of-domain robustness (E6); (3) the gate/dilation structure
is a more promising lead for out-of-domain robustness, not yet isolated (would need an E8-style ablation that
holds norm fixed and varies gate/dilation alone -- not yet run, a natural Phase 2 candidate).

## Notes
RedDino's own training data includes MP-IDB, so a RedDino external-generalisation probe on MP-IDB (if later
added, extending E9's backbones to this test set) would be contaminated -- use BBBC041 for that comparison
instead, or disclose the overlap prominently if MP-IDB is used anyway.
