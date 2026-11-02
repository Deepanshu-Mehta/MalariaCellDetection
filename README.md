# Automated Microscopic Diagnosis of Malaria

A computer-vision research-reproduction project on the **NIH Malaria cell dataset**
(Semester 7 · Computer Vision).
**Team:** Agrapujya Lashkari, Deepanshu Mehta

The workflow follows the course's six steps. We survey the literature, pick one base paper, reproduce it, propose **one**
change with a prediction made *before* coding, test it, then analyse and write it up in IEEE format.

| Milestone | What | Status |
|---|---|---|
| M1 | Literature survey (11 papers, incl. DANet) + slides | ✅ done |
| M2 | Reproduce the base paper (DANet) | ✅ done: 97.02 % accuracy |
| M3 | Hypothesis + prediction (before coding) + Introduction | ✅ submitted 2026-10-05 |
| M4 | Implement the change, run ablations | ✅ done (M3's colour-jitter hypothesis was falsified, see [Research audit](#research-audit)) |
| M5 | Failure analysis, IEEE paper, defence | ⏳ |

**Beyond the course scope:** M2's reproduction grew into an independent, pre-registered audit of DANet and of what the NIH
benchmark measures. The target is *Computers in Biology and Medicine*. See [Research audit](#research-audit) for current
results, `research/ROADMAP.md` for the plan, and `research/registry.csv` for every prediction, result and correction.

---

## Repository layout

```
.
├── milestone1-literaturereview/
│   ├── CV2.pdf                                 # M1 survey report (IEEE format, 11 papers; DANet = Sec. VIII)
│   └── malaria-literature-survey-slides.pdf    # M1 slides (15; DANet = paper 11/11, slide 13)
├── milestone2-baseline/
│   ├── danet_baseline.ipynb                    # reproduction notebook (clean, re-runnable)
│   ├── paper/Ali2025_DANet_SciRep.pdf          # base paper
│   └── outputs/
│       ├── danet_baseline_executed.ipynb       # executed notebook, all results ← M2 submission
│       ├── results.json                        # test metrics, config, params, FLOPs
│       ├── splits.csv                          # the exact 70/10/20 split + patient IDs
│       ├── test_predictions.csv                # per-cell test predictions
│       ├── ckpt/best.pth, ckpt/train_log.csv   # best checkpoint (epoch 47) and per-epoch log
│       └── figures/                            # curves, confusion matrix/ROC/PR, Grad-CAM, augmentations
└── milestone3-hypothesis/
    ├── HYPOTHESIS.md                           # hypothesis + predictions ← M3 submission
    ├── malaria-slides-M1-M3.pdf                # full deck: M1 (15, incl. DANet) + M2/M3 (6) = 21 slides
    └── report/report.tex, report.pdf           # survey report + DANet section + the new M3 Introduction
research/
├── ROADMAP.md, registry.csv                    # living plan; one row per experiment: prediction, result, verdict
├── train.py, configs/, malaria/, splits/       # shared training code, experiment configs, frozen leakage-proof splits
├── experiments/<ID>_<name>/                    # one write-up per experiment, plus its analysis scripts and results
├── exploratory/                                # unregistered checks, labelled as such
├── protocols/, scripts/, kaggle/, results/     # external-data protocol, utilities, Kaggle launcher, master tables
└── runs_kaggle/                                # per-run configs, logs and predictions (checkpoints are not tracked)
```

Datasets, checkpoints, cached arrays, `.txt` notes, `reports/` and `research_notes/` are kept local and are not tracked.

---

## Dataset

[NIH Malaria cell images](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html):
27,558 segmented red-blood-cell crops from Giemsa-stained thin smears (Chittagong Medical College Hospital),
**13,779 parasitized / 13,779 uninfected**, from 201 patient IDs. Images are small (≈ 134 × 134 px).
NIH also publishes patient-ID ↔ cell mapping CSVs. We use them to measure patient leakage and to build patient-level splits.

The notebook downloads everything automatically into `milestone2-baseline/data/` (not tracked by git):
`cell_images.zip`, `patientid_cellmapping_parasitized.csv` and `patientid_cellmapping_uninfected.csv`.

---

## M2: Reproducing DANet

**Base paper:** A. Ali *et al.*, "DANet: a lightweight dilated attention network for malaria parasite detection",
*Scientific Reports* 15:36689 (2025). Code: <https://github.com/asfakali/DANet>

DANet-S has 9 Dilated Attention Blocks and 2.31 M parameters (0.83 GFLOPs per 224² image). Each block chains 3×3 convs with
dilation 1 → 2 → 3, then applies avg ‖ max pooling, a channel-softmax attention map multiplied with the input, and a 1×1 projection.

### Reproducibility notes
- **The released code does not run as shipped.** It raises a `NameError` in `super()`, and its output layer has 32 classes for a 2-class task.
  Both are fixed in our port, which is numerically identical to the repo's model given the same weights.
- **The paper and the code disagree.** The text describes parallel dilations, a sigmoid gate and LayerNorm; the code uses cascaded dilations, a channel-softmax gate and InstanceNorm.
  We follow the **code**, which matches the 2.3 M parameter count.
- **The paper's own learning rate does not train.** Table 2 gives lr 0.01, the text gives 0.001, and a default `OneCycleLR` call silently adds
  0.85–0.95 momentum to RMSprop. A pre-run diagnostic (4k images, 8 epochs) gave:

  | Optimiser setting | Val acc, epoch 1 → 2 → 8 |
  |---|---|
  | RMSprop + OneCycle(0.01), default momentum cycling (paper Table 2) | 0.500 → 0.500 → 0.515 |
  | RMSprop + OneCycle(0.001), default momentum cycling | 0.500 → 0.500 → 0.500 |
  | RMSprop + OneCycle(0.01), momentum 0 | 0.500 → 0.500 → 0.500 |
  | **RMSprop + OneCycle(0.001), momentum 0** ← used | **0.736 → 0.919 → 0.964** |
  | Adam 1e-3, constant (repo `train.py`) | 0.547 → 0.577 → 0.961 |

### Frozen baseline configuration
Stratified 70/10/20 split (19,290 / 2,756 / 5,512, identical to paper Table 1, seed 42) · 224×224 inputs scaled to [0,1] ·
rotation ±15°, H/V flip with p = 0.2 · RMSprop (momentum 0) + OneCycleLR(max_lr 0.001) · NLL loss · batch 64 · 75 epochs ·
no regularisation · best-validation-loss checkpoint, test set evaluated once · **Parasitized = positive class**.

### Results (test set, 5,512 cells)

| Metric | Paper (single split) | Paper (5-fold CV) | **Ours** | Δ vs. paper |
|---|---|---|---|---|
| Accuracy | 97.95 | 97.25 (96.97–97.42) | **97.02** (95 % CI 96.58–97.47) | −0.93 |
| Precision | 97.76 | 97.21 | **97.72** | −0.04 |
| Sensitivity | 98.07 | 97.20 | **96.30** | −1.77 |
| Specificity | 97.87 | — | **97.75** | −0.12 |
| F1 | 97.86 | 97.20 | **97.00** | −0.86 |
| ROC-AUC | 0.98 | — | **0.9945** | +0.01 |
| PR-AUC | 0.98 | — | **0.9950** | +0.01 |
| Params | 2.3 M | — | 2.31 M | ≈ 0 |

**Verdict:** the result reproduces at the level of the paper's *own* cross-validation, but not its headline 97.95 %. That headline is
higher than all five of the paper's CV folds. The gap is concentrated in **sensitivity** (102 missed infections vs. 62 false alarms).

### Findings from the baseline
1. **Patient leakage.** All 201 test patients also contribute training cells, so the paper's protocol measures *within-patient* accuracy.
2. **Stain fingerprint.** Patient ID explains 74–90 % of the variance in a cell's mean colour. Slides differ mainly in saturation (±48 %) and hue.
3. **Overfitting after epoch ≈ 45.** Train loss falls to 0.03 while validation loss rises from 0.087 to 0.11. The paper reports no overfitting.
4. **Grad-CAM is off-target.** The heatmap is cold on visible parasites and hot at the cell rim, which does not support the paper's Fig. 9.
5. **Likely label noise.** Several high-confidence errors look mislabelled, consistent with known NIH label errors.

---

## M3: Hypothesis (submitted before any M4 code)

> **Adding stain-colour augmentation to DANet's training will raise sensitivity on *unseen patients*, because each slide's stain
> is a near-unique fingerprint the network can use in place of parasite morphology.**

- **The one change:** `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.5, hue=0.1)` on training images. The ranges are calibrated to the measured
  between-patient colour spread. Everything else stays frozen.
- **Evaluation:** a patient-level 70/10/20 split (40 test patients), repeated over 3 split seeds. Primary metrics are sensitivity,
  balanced accuracy and ROC-AUC.

| Row | Config | Split | Change | Prediction |
|---|---|---|---|---|
| B0 | Baseline | cell | — | 97.02 / SE 96.30 (measured) |
| B1 | Baseline | patient | protocol | balanced acc −1 to −2.5 pts, SE −1.5 to −3 pts |
| **H** | + colour aug | patient | colour aug | **vs. B1: SE +0.5 to +1.5, balanced acc +0.3 to +1.0** |
| C | + colour aug | cell | control | vs. B0: within ±0.3 pts |

**Falsified if** H ≤ B1 on sensitivity and balanced accuracy, averaged over the 3 splits.
**Why we might be wrong:** InstanceNorm in every block already strips global colour, and mean colour alone predicts the label at only 61 %.
Full reasoning: [`milestone3-hypothesis/HYPOTHESIS.md`](milestone3-hypothesis/HYPOTHESIS.md).

---

## Research audit

All predictions are written and dated before a run. Falsified results stay on the record, and corrections are added above
the original text rather than replacing it. **Exploratory** marks anything not pre-registered.

| Question | Current evidence | Experiments | Status |
|---|---|---|---|
| Does patient leakage inflate the score? | Barely under patient-grouped CV: gap +0.26 BACC, far below M3's prediction. Imaging sessions still cross folds | E1, E10 | Patient level done; session level pending |
| Is DANet better than simple references? | About equal to ResNet-18; frozen foundation-model probes reach 96.0-96.5 BACC vs 96.8 | E1r, E9 | Done |
| Does ColorJitter help on unseen patients? | No: M3's hypothesis is falsified by its own rule | E4 | Done |
| Does the norm layer change leakage use? | LayerNorm gap +0.63 vs InstanceNorm +0.02; new seeds LN-IN +.15 [-.05, .36] | E8, E8r | **Not replicated** under the registered rule; stays exploratory |
| Is any architecture more robust on MP-IDB? | No robust ranking across 5 checkpoints per arm; BACC SD 5.3-8.4 between folds | E6, E6b | Done; E6's original ranking withdrawn |
| Does colour invariance help externally? | Training-time colour transfer: BACC +7.05, AUC +0.059. Test-time Reinhard normalisation harmed all arms | E6c | Done |
| Are explanations trustworthy? | They land on annotated parasites, but so does a darkest-pixel rule | E5 | Narrowed; NIH question open |
| Are the labels noisy? | 312 cells (1.13%) flagged; some sit in healthy-only patients | E7 | Needs expert review |
| How far does one hand-made feature go? | NIH AUC .981; MP-IDB AUC .987. Worse than DANet on NIH healthy patients (false positives about 3.6% vs 2.0%) | exploratory | Needs BBBC041 |

**Dataset caveats found so far:** cell count alone separates infected from healthy-only patients (AUC .992), because every
patient has about 68 uninfected cells. Infection status is also heavily confounded with imaging date: only 7 of 50
healthy-only patients share a date with any infected patient. Both are exploratory findings.

---

## How to run

**Environment:** Python ≥ 3.10, `torch` ≥ 2.0, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, `jupyter`
(tested with Python 3.13, torch 2.9, torchvision 0.24 on Apple M2 / MPS).

```bash
cd milestone2-baseline
jupyter notebook danet_baseline.ipynb            # or run all cells in VS Code / Kaggle / Colab

# Headless, full run (75 epochs, ≈ 3 h on a T4 or an M2; resumes from outputs/ckpt/last.pth if interrupted)
jupyter nbconvert --to notebook --execute danet_baseline.ipynb \
  --output-dir outputs --output danet_baseline_executed.ipynb --ExecutePreprocessor.timeout=-1

# Smoke test: 10 % of the data, 2 epochs, writes to outputs_smoke/ (checks the pipeline, not accuracy)
DANET_SMOKE=1 jupyter nbconvert --to notebook --execute danet_baseline.ipynb --output-dir outputs_smoke
```

- **Kaggle:** attach `iarunava/cell-images-for-detecting-malaria`, enable GPU (and internet, for the patient-ID CSVs).
- **Apple Silicon:** if `import torch` fails with an architecture error, the shell is running under Rosetta. Use `arch -arm64 python3 …`.
- **Report:** `cd milestone3-hypothesis/report && tectonic report.tex`.

---

## Limitations

NIH is one public dataset from one hospital. The M2 numbers come from one seed on a leaky cell-level split, and the M2
findings above are the original notes: `research/` revises several of them, including findings 4 and 5. MP-IDB has been
reused across experiments, so BBBC041 is reserved as the clean external test. This is a research reproduction:
**no clinical claims** are made.

## References
- A. Ali *et al.*, "DANet: a lightweight dilated attention network for malaria parasite detection," *Sci. Rep.*, 15:36689, 2025.
- S. Rajaraman *et al.*, "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images," *PeerJ*, 6:e4568, 2018.
- D. Tellez *et al.*, "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology," *Med. Image Anal.*, 58:101544, 2019.
- Full survey bibliography: `milestone3-hypothesis/report/report.pdf`.
