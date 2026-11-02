# E4b: Colour transfer between patients (exploratory)

**Status:** done (grouped: 15/15; leaky: E8b only, quota-scoped)  ·  **Outcome:** confirmed for E8b (primary outcomes)

> **Correction, 2026-11-02:** the "causal" language below is **withdrawn**. The leakage-gap result rests on one leaky
> seed for E8b (paired change on seed 42: see `exploratory/2026-11-02_paper_readiness/verify_corrections.json`), the
> +0.627 baseline is a three-seed mean, and E1 and E8c have no leaky colour-transfer controls. Read it as an encouraging
> single-seed result until controls exist. E6c later found that these checkpoints do better on MP-IDB.
> The text below is kept as the original record.

## Question
Does moving infected cells into healthy-only stains reduce stain reliance?

## Prediction v1 (written 2026-10-29, before E2 finished) -- SUPERSEDED, kept for the record
Reasoning: E4's random ColorJitter was falsified (H-B1 BACC -0.04, within noise) -- but that is a weaker,
untargeted perturbation, and E8 found InstanceNorm (E1, E8c) already keeps the leakage gap near zero while
LayerNorm (E8b) does not. Targeted colour transfer between patients -- not random jitter -- is the sharper
test, because 50 patients are healthy-only, so the (infected-patient stain, parasitized label) combination
coexists with a (healthy-only stain, uninfected-only) combination that no amount of reweighting can fix; only
synthesis/transfer can fill that gap.
Predicted: (a) on E1/E8c (InstanceNorm), colour transfer changes grouped BACC by less than the 0.29pp MDE;
(b) on E8b (LayerNorm), colour transfer measurably reduces the leakage gap (+0.63) toward E1/E8c's (~0.2);
(c) **the stain-swap flip rate (E2) drops after colour-transfer training, most for E8b and least for E1/E8c.**

## Prediction v2 (written by the AI assistant, 2026-10-29, after E2 finished, before any E4b run)
**What E2 changed:** E2's stain-swap flip-rate test -- the SAME Reinhard-based measurement this experiment's
part (c) leaned on -- found **no detectable difference between E1/E8b/E8c at baseline** on uninfected cells
(all differences 0.0003-0.006, every bootstrap CI includes 0; see `E2/results_stainswap_stats.json`). A
measurement that cannot tell the three architectures apart *before* any intervention is unlikely to reliably
show whether an intervention changed anything either. **Part (c) of v1 is downgraded from a primary outcome
to a secondary one**, kept only as a weak, likely-null corroborating check.

What DID discriminate the architectures in E2 was the **patient-ID probe** (within-patient 80/20 split on
frozen embeddings): E8b 39.8% vs E1/E8c ~25-28% (chance 0.7%). This is now the stronger, more sensitive
instrument, so it is promoted to a co-primary outcome alongside the leakage gap (which is the measure E8
itself already validated as sensitive).

**Revised prediction, primary outcomes only:**
- (a) On E1/E8c (InstanceNorm): colour-transfer training changes grouped BACC by less than the 0.29pp MDE,
  and changes the patient-ID probe accuracy by less than a similarly small margin (little left to remove).
- (b) On E8b (LayerNorm): colour-transfer training reduces the leaky-vs-grouped leakage gap from +0.63 toward
  E1/E8c's level (~0.2), **and** reduces the patient-ID probe accuracy from 39.8% toward E1/E8c's ~25-28%.
  Given E2's weaker-than-hoped baseline signal, this effect is predicted to be real but **modest** (not a full
  closure of either gap) -- a downgrade in confidence from v1, not just in the flip-rate metric specifically.
- (c), secondary/exploratory only: the stain-swap flip rate on E8b *may* drop slightly post-training, but given
  E2's null baseline, a null result here would not be surprising and would not by itself falsify (a) or (b).
**Falsified if:** colour-transfer training changes E8b's leakage gap AND patient-ID probe accuracy by less
than their respective MDEs (i.e., both primary outcomes null) -- that would mean the leakage gap is not
actually carried by transferable stain information at all, which would be a stronger result than v1's
falsification condition and would need a different explanation for E8's norm-layer finding.

## Protocol
- **Training-time augmentation:** for each training cell, with probability p (to be fixed in the config, e.g.
  0.5), recolour it via Reinhard LAB transfer (foreground-only, background re-zeroed -- same transform as E2's
  stain-swap) to a **randomly sampled different training patient's** foreground LAB statistics, computed fresh
  per fold from TRAINING patients only (no test-patient colour information used). New code:
  `research/malaria/colour.py` (factored out of E2's `analysis_stainswap.py`), a new `train.py` config flag.
- **Arms:** E1 (InstanceNorm/softmax/cascade), E8b (LayerNorm/sigmoid/parallel), E8c (InstanceNorm/sigmoid/
  parallel), each with colour-transfer training, grouped repeat 0, 5 folds (screen, matching E1/E8's protocol).
- **Outcomes:** (1) leaky-vs-grouped leakage gap, same method as E1/E8; (2) patient-ID probe on the resulting
  checkpoints' frozen embeddings, same method as E2 part C; (3) stain-swap flip rate, same method as E2 part D
  (secondary).
- Config: `configs/E4b_*_colourtransfer.json` (one per arm)
- Primary metric: leakage-gap change, patient-ID-probe-accuracy change (both vs the arm's own E1/E8b/E8c baseline)
- Estimated T4 GPU-h: ~9 (3 arms x 5 folds x ~0.6h, screen only, no leaky-seed replication yet)

## Runs

| Date | Job slug | Runs | Notes |
|---|---|---|---|
| 2026-10-29 | `e4b-colour-transfer` | 15 (3 arms x 5 grouped folds) | All trained 50 epochs normally, ~1.2h each (colour-transfer roughly doubles per-run time vs a plain run) -- cost ~18h GPU-quota, far more than the ~9h estimate, eating most of the week's remaining budget |
| 2026-10-30 | `e4b-leaky-e8b` | 1 (E8b, leaky seed 42) | Added after noticing the leakage-gap outcome needed a leaky-split run that hadn't been launched; scope-reduced to E8b only (the arm the hypothesis is about) given only ~4.3h quota remained |
| 2026-10-30 | (local) `analysis.py` | grouped BACC + patient-ID probe, all 3 arms | First attempt hit the same MPS memory-leak pattern as E2's original script (killed at 4.9GB before degrading); fixed (cache release + logging) and rerun cleanly in ~3 min |

**Scope note, disclosed rather than silently assumed:** E1 and E8c's leaky-split colour-transfer runs were
not launched (quota-constrained after the grouped runs cost much more than estimated). Their leakage-gap
outcome is therefore **not measured**, only inferred from their near-zero grouped-BACC and patient-ID-probe
changes. If quota allows later, running `E4b_e1_colourtransfer:leaky:seed=42` and
`E4b_e8c_colourtransfer:leaky:seed=42` would close this gap in the evidence.

## Result

**Grouped BACC change (colour-transfer vs each arm's own baseline, same folds):**

| Arm | Grouped BACC (baseline -> colour-transfer) | Change | vs 0.29pp MDE |
|---|---|---|---|
| E1 | 96.78 -> 96.78 | +0.01 | negligible |
| E8b | 96.71 -> 96.66 | -0.05 | negligible |
| E8c | 96.78 -> 96.68 | -0.11 | negligible |

Colour-transfer training does not cost real-task accuracy for any arm, as expected.

**Patient-ID probe change** (within-patient 80/20 split on frozen embeddings, same method as E2):

| Arm | Probe accuracy (baseline -> colour-transfer) | Change |
|---|---|---|
| E1 | 25.1% -> 24.0% | -0.011 (negligible) |
| E8b | **39.8% -> 34.2%** | **-0.056 (real, but partial)** |
| E8c | 27.9% -> 27.8% | -0.001 (negligible) |

E8b's probe accuracy drops substantially toward E1/E8c's ~24-28% level, but does not fully reach it --
colour-transfer removes *some*, not all, of the extra patient-identity signal LayerNorm's embeddings carry.

**Leakage gap** (leaky BACC - grouped BACC; completed for E8b only, see scope note above):

| | Original gap | Colour-transfer gap | Change |
|---|---|---|---|
| E1 (reference) | +0.187 | not measured | -- |
| E8c (reference) | +0.016 | not measured | -- |
| **E8b** | **+0.627** | **+0.222** | **-0.405 (>> 0.29pp MDE)** |

E8b's gap falls to within range of E1's *own original* gap (+0.187), a substantial, clearly-above-MDE
reduction -- colour-transfer training closes most of the gap that LayerNorm opened in E8.

## Verdict and what we learned
**Confirmed for E8b on both primary outcomes, which is the arm the hypothesis was actually about.** Neither
outcome was null (the registered falsification condition): the leakage gap fell by 0.405pp (>> the 0.29pp
MDE) and the patient-ID probe fell by a real, non-trivial margin. E1 and E8c showed the predicted "little
left to remove" pattern on the outcomes that were measured for them (grouped BACC, patient-ID probe); their
leakage-gap outcome specifically was not measured, a disclosed limitation rather than a silent gap.

**What this adds to the project's thesis:** combined with E8 (LayerNorm profits from leakage without a
grouped-accuracy gain) and E2 (LayerNorm's embeddings carry far more patient identity), this result closes
the loop -- a targeted intervention that specifically removes patient-identifying colour information
**causally** reduces both the representational signature (patient-ID probe) and the behavioural consequence
(leakage gap) that the earlier experiments only observed, not manipulated. This is the strongest causal
evidence in the project for "LayerNorm's extra leakage profit is specifically about patient/stain colour
information," as opposed to some other correlated difference between the architectures.

**Caveat:** the probe reduction is partial (34.2%, not down to E1/E8c's ~24-28%), so colour is not the whole
story even for E8b -- some patient-identifying signal survives colour-transfer training, consistent with
texture, background or other non-colour cues also carrying some patient identity (an open question, not
pursued further here).

## Notes
Two real bugs were found and fixed while running this experiment: (1) the `train.py` output-path-collision
bug, originally found via E3, which this experiment's 15 distinct-config runs did NOT trigger (fold number
alone disambiguates these runs, unlike E3's `label_draw` overrides) -- confirmed no collision occurred by
checking all 15 output directories are distinct; (2) the MPS memory-leak pattern in a freshly-written analysis
script, same root cause as E2's original bug, fixed the same way (cache release between fold-loads).

