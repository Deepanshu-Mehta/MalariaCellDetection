# E9: Foundation-model probes

**Status:** done  ·  **Outcome:** partially confirmed (ranking right, magnitude far too low in the prediction)

> **Correction, 2026-11-02:** "DANet beats all three" holds on repeat 0 only. DANet's margin over RedDino is not robust:
> its CI includes 0 on repeats 1 and 2 (`exploratory/2026-11-02_paper_readiness/verify_corrections.json`). DANet does
> beat DINOv2 and DinoBloom. The text below is kept as the original record.

## Question
Do frozen DINOv2/v3 / DinoBloom / RedDino features match DANet?

## Prediction (written before any run; date it)
**Written by the AI assistant, 2026-10-29, before any run.** Reasoning: no published linear probe of any
foundation model on NIH Malaria exists (confirmed gap, Phase-0 research round). RedDino is RBC-specific
(1.25M images) though not malaria-specific and NOT trained on NIH (clean probe here), while its own training
set does include MP-IDB/IML-Malaria/M5, so any RedDino result ON THOSE sets (not NIH) would be contaminated --
flag this explicitly if RedDino is later probed on MP-IDB in E6. DinoBloom is WBC-focused (minimal RBC data).
DINOv2/v3 are natural-image models. E2's colour-only classifier gets 64% BACC on this task from 5 numbers
alone, which is the floor any frozen-feature probe should clear easily.
**Prediction:** on NIH's patient-grouped folds, a frozen linear probe's BACC ranks RedDino > DinoBloom >
DINOv2/v3-generic > the E2 colour-only floor (64%) > chance. RedDino's probe lands around 85-93% BACC -- well
above the colour floor and below DANet/ResNet-18's ~96.7% fine-tuned ceiling, since a linear probe on frozen
features is a weaker model than an end-to-end fine-tuned 2.3M-parameter CNN. DinoBloom and DINOv2/v3 land lower,
around 75-85%. At 1% of labels, RedDino's probe stays within 5-10 points of its full-label performance, while
a from-scratch DANet trained on 1% of labels collapses toward chance -- a sample-efficiency gap that is itself
a usable result even where full-label accuracy favours DANet.
**Falsified if:** any frozen linear probe matches or beats DANet's ~96.7% fine-tuned grouped BACC -- a
stronger and more surprising finding (DANet's training adds nothing beyond off-the-shelf frozen features) that
should be reported prominently if it happens, not treated as a failed prediction to bury.

## Protocol
- **Backbones:** `backbones.py` loads DINOv2-S, DinoBloom-S, RedDino-S (all via Hugging Face / timm, frozen,
  384-dim) and DINOv3-S (gated, unavailable). Availability is checked live (`available_backbones`), not
  assumed, and failures are reported per backbone rather than blocking the run.
- **Extraction + probe:** `extract_and_probe.py` extracts one frozen embedding per cell per backbone (cached
  to disk, independent of split), then runs logistic-regression + 15-NN probes on grouped repeat 0 at label
  fractions {1%, 10%, 100%}, pooled into out-of-fold predictions across the 5 folds.
- **Statistical comparison:** `compare_to_danet.py` recomputes the 100%-label probe locally from the cached
  features (no extra GPU time) and runs a paired patient-cluster bootstrap against DANet's own grouped-r0 OOF
  predictions (same cells, same folds) -- point estimates alone are not treated as sufficient.
- Config: `backbones.py`, `extract_and_probe.py`, `compare_to_danet.py` (local, no JSON config)
- Primary metric: BACC, ROC-AUC; label fractions 1/10/100%
- Estimated T4 GPU-h: ~1.5 (feature extraction only; probes are CPU)

## Runs

| Date | Where | Notes |
|---|---|---|
| 2026-10-29 | Local (failed) | DinoBloom-S loaded fine; RedDino-S and DINOv2 (both via torch.hub / HF) hung indefinitely on this machine's network path (confirmed via file-size monitoring: RedDino crawled to 40MB in 8 min, genuinely stalled, not slow). Moved to Kaggle. |
| 2026-10-29 | Kaggle (`e9-foundation-probes`) | All three open backbones loaded and extracted successfully -- confirms the earlier hang was a local network issue, not a model/code issue. DINOv3-S failed as expected (gated). |
| 2026-10-29 | Local | `compare_to_danet.py` run on the downloaded cached features (no GPU needed) |

## Result
**Backbone availability:** DINOv2-S, DinoBloom-S, RedDino-S all available (384-dim); DINOv3-S gated/unavailable.

**Probe accuracy (logistic regression, grouped repeat 0, pooled OOF):**

| Backbone | 1% labels | 10% labels | 100% labels |
|---|---|---|---|
| DINOv2-S (generic) | 94.24 | 94.49 | 96.03 |
| DinoBloom-S (WBC-focused) | 94.69 | 95.06 | 96.31 |
| RedDino-S (RBC-specific) | 94.96 | 95.59 | 96.45 |

Label-fraction curves are remarkably flat: every backbone is already within ~2 points of its own full-label
score using just 1% of the training labels. The E2 colour-only floor (63.98% BACC) is cleared by >30 points.

**Paired comparison vs DANet (fine-tuned, grouped-r0 BACC 96.78), patient-cluster bootstrap:**

| Comparison | Diff | 95% CI |
|---|---|---|
| DANet - DINOv2-S | +0.74 | [+0.44, +1.07] |
| DANet - DinoBloom-S | +0.47 | [+0.18, +0.78] |
| DANet - RedDino-S | **+0.32** | **[+0.03, +0.63]** |

All three differences are statistically significant (every CI excludes 0): **no frozen probe matches or
beats DANet**, so the headline falsification condition did not trigger. But the margin shrinks monotonically
with domain relevance (generic -> WBC-focused -> RBC-specific), and RedDino's margin sits barely above the
0.29pp MDE -- the smallest fine-tuned/frozen gap found anywhere in this project's comparisons.

**A real analysis bug caught and fixed before reporting:** the first version of `compare_to_danet.py` had the
arguments to the paired-difference function in the wrong order, which silently flipped every sign -- it
initially appeared DANet was *worse* than the frozen probes, the opposite of the true result. Caught by
cross-checking the raw point estimates (DANet 96.78 vs. probes 96.0-96.5, which cannot produce a negative
"DANet minus probe") before the number was written anywhere permanent. Fixed and reran; see the corrected
numbers above.

## Verdict and what we learned
**Partially confirmed.** The predicted *ranking* (RedDino > DinoBloom > DINOv2 > colour floor, all below
DANet) matched exactly. The predicted *magnitude* (75-93% BACC) was far too low -- actual results are
96.0-96.5%, within a point of DANet's fine-tuned ceiling using nothing but a frozen encoder and a logistic
regression. This is a genuinely graded, non-binary result rather than a clean confirm/falsify:

- It is **not** the "DANet adds nothing" finding the falsification clause was written to catch -- DANet does
  measurably better, every time.
- But it **does** show that almost all of DANet's apparent advantage over a generic, off-the-shelf,
  never-fine-tuned self-supervised encoder (DINOv2, trained on natural images, zero malaria/blood exposure)
  can be recovered with a linear probe and 1% of the labels. A bespoke architecture and full fine-tuning buy
  roughly 0.3-0.7 points of balanced accuracy over that baseline, and that margin keeps shrinking as the
  frozen backbone gets more domain-relevant.
- For the paper: this belongs next to the E1r (ResNet-18 ~ DANet) result as further evidence that DANet's
  headline number is not explained by anything unusual or hard to replicate -- it is close to what several
  very different, much cheaper or much more generic approaches already achieve on this task.

**Limits:** one grouped repeat (r0), linear probes only for the formal comparison (kNN results in
`results.json` are consistently a little lower and not formally tested), RedDino's own training data
includes MP-IDB/IML-Malaria/M5 (not NIH), so this NIH probe is clean but a future RedDino probe on MP-IDB
(E6) would be contaminated and must be disclosed as such.

## Notes
`backbones.py`'s loaders are reusable for E6 (external-set scoring with the same frozen features) without
re-extracting NIH features -- only the external-set crops need a fresh extraction pass.

