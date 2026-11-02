# What does the NIH malaria cell benchmark measure? An audit of DANet with a one-feature baseline

**Working draft for *Computers in Biology and Medicine* (Full Length Article). Not for submission as written.**

Deepanshu Mehta, Agrapujya Lashkari. Affiliation and corresponding author to be added.

> **Note to authors.** The AI assistant drafted this text. Journal policy requires the manuscript to reflect your own analysis, interpretation, insights and ideas, so read every claim against
> `research/registry.csv` and rewrite it in your own words before submission. `[PENDING]` marks results that do not exist yet. No BBBC041 result exists in this draft.

## Abstract
`[PENDING: at most 250 words, written after E11. Keep one claim per sentence, and state the one-shot outcome first.]`

**Keywords (1-7):** malaria; blood smear; external validation; shortcut learning; underspecification; reproducibility

## 1. Introduction

Automated classification of thin-smear blood cells is a widely studied route to faster malaria diagnosis, and the NIH cell dataset is its usual benchmark [1]. Recent lightweight networks report accuracy near 98%
on it. DANet, for example, reports 97.95% with 2.3 million parameters [2]. Scores this high leave little room to separate methods, and they do not by themselves show that a model will work on another
laboratory's slides.

Four features of how the benchmark is used make such a score hard to interpret. Random cell-level splits place cells from one patient in both training and test sets. Images taken on one day share stain and
acquisition conditions, so a patient-level split can still leak a session. The crops are curated, with parasites as the main dark objects, so hard negatives such as leukocytes are absent. A single trained
checkpoint can also hide the fact that models with equal test scores may behave differently elsewhere, a problem known as underspecification [3]. Reliance on easily learned cues that fail outside the training
distribution is the related problem of shortcut learning [4].

Two recent reports bear directly on these questions. MalariaNet, evaluated without slide leakage, reported accuracy 79.5% on MP-IDB and 53.2% on BBBC041, with sensitivity 96.4% and specificity 18.0% on the
latter [5]. EMFE showed that a five-feature model built on green-channel spots reached AUC .9765 on NIH under patient-grouped validation, without an external test [6]. Together they suggest that NIH scores
overstate transfer, and that much of the NIH task may be solvable by very simple means.

We reproduced DANet and audited what its score means, writing dated predictions before most experiments, labelling the rest exploratory, and keeping failed predictions on record. We make four contributions: (i) we measure
leakage at patient and session level; (ii) we show that a single hand-made measurement, with no training, comes close to trained models on NIH; (iii) we separate ranking from thresholding on external data, across
retraining and across crop masks; and (iv) we test all of this once, with frozen candidates, on a dataset that played no part in fitting or choosing any model or feature.

## 2. Materials and methods

### 2.1 Datasets

| Dataset | Species | Use | Size used | Labels |
|---|---|---|---|---|
| NIH cell images [1] | *P. falciparum* | Training and internal testing | 27,558 crops, 200 patients (150 infected, 50 healthy) | Cell level |
| MP-IDB, Falciparum [7] | *P. falciparum* | External development | 104 images, 14,468 crops (963 Parasitized, 13,505 Uninfected) | Expert parasite masks |
| BBBC041 [8] | *P. vivax* | One-shot external test | 1,328 images, 79,722 crops (2,425 Parasitized, 77,194 Uninfected, 103 leukocytes) | Expert boxes and stages |

One NIH patient with two microscope entries was merged, giving 200 patients. MP-IDB annotates parasites only, so cells were segmented with Cellpose [9] and processed as in the protocol: crops touching the border (3,013),
size outliers (510), ambiguous overlaps (387) and leukocyte-like cells (20) were excluded. Background pixels were set to black to match NIH. In an AI-assisted audit of 100 crops, 10 were partial-cell fragments, which we report as a limitation.

For BBBC041, ring, trophozoite, schizont and gametocyte boxes were labelled Parasitized and red blood cell boxes Uninfected. We excluded boxes touching the image border (1,523), `difficult` boxes (446) and red cell boxes intersecting a parasite box
(4,344). No label-based subsampling was used, so prevalence stayed natural at 3.05%. Two acquisition groups can be identified from format and image size (1,208 PNG, 120 JPG).

### 2.2 Splits

All patient-level splits were built before training and hashed. **Grouped:** five folds over patients, three repeats, balanced across four strata. **Leaky:** the original random cell-level 70/10/20 split. **Session:** patients sharing any imaging date
(from file names) were linked into blocks by connected components, giving 50 blocks. Whole blocks were assigned to five folds, so training, validation and test sets share no date. Test folds held 8 to 13 healthy patients, with test prevalence SD .10
percentage points. Date is a proxy for session.

### 2.3 Models

**DANet.** We used DANet-S with the authors' published code (softmax gate, cascaded dilations, InstanceNorm; arm E1) and a paper-faithful variant (sigmoid gate, parallel dilations) with LayerNorm (E8b) or InstanceNorm (E8c). Training used RMSprop with a
one-cycle schedule, maximum learning rate .001, batch size 64, 224 px and 50 epochs. **ResNet-18.** ImageNet weights, AdamW, 128 px, 15 epochs. **Colour transfer.** During training only, with probability .5, a cell was recoloured to another training patient's
statistics by Reinhard transfer [10] in LAB space, foreground only. **Frozen probes.** Logistic regression on DINOv2 [11], DinoBloom [12] and RedDino [13] embeddings. RedDino was not used externally, because it was trained on MP-IDB and its training list could not be
confirmed free of BBBC041.

### 2.4 Dark-spot baseline

For a crop with background set to zero, resized to 128 px, the score is (median − min) / median of the Gaussian-smoothed green channel (sigma 1.5) inside the foreground eroded by 4 px. A cell is called Parasitized when the score is at least .3047, the NIH maximiser of balanced accuracy,
rounded to four places. The rule has one threshold and no other fitted parameter. **The feature was chosen after inspecting explanation results on MP-IDB, so its MP-IDB performance is exploratory.** Only BBBC041 tests it.

### 2.5 External evaluation

No model, threshold or normalisation was fitted on external data. On MP-IDB, E6b scored 15 checkpoints across three arms; E6c added colour-transfer checkpoints. E12 fixed the training data and varied the seed (`[PENDING]`).

**BBBC041, one-shot.** Four candidates were frozen, with hashes recorded before scoring: (1) the dark-spot rule; (2) an E1 five-fold ensemble; (3) a colour-transfer ensemble of 15 DANets and 5 ResNet-18s; (4) DINOv2 and DinoBloom probes, as descriptive comparators. Ensembles use the mean logit, with
decision threshold 0. BBBC041 offers boxes, not masks, so crops were made with two rules: an ellipse inscribed in the box, and Otsu thresholding of the green channel. Delta is candidate 3 minus candidate 1. AUC and balanced accuracy were co-primary, each with a 97.5% image-clustered percentile bootstrap CI (5,000
draws). The five-way outcome rule is given in Table 1. It was applied separately to each crop rule, and if they disagreed, the outcome was recorded as "depends on crop rule". This agreement requirement was added before any BBBC041 score existed, after MP-IDB rehearsal showed the crop rule alone could decide the outcome.
Scores were saved and hashed before any metric was computed. A rescore was allowed only for a failed automated integrity check.

**Table 1. Outcome rule, delta = candidate 3 minus candidate 1. The first condition that holds decides.**

| Order | Outcome | Condition |
|---|---|---|
| 1 | No transfer | AUC below .80 for both candidates |
| 2 | Feature holds | AUC upper bound at most +.01 and BACC upper bound at most +2 |
| 3 | Trained model wins | AUC lower bound above +.01 and BACC lower bound above +2 |
| 4 | Ranking and threshold disagree | The AUC and BACC decisions point opposite ways |
| 5 | Inconclusive | Anything else |

### 2.6 Statistics

Primary metrics were balanced accuracy (BACC), sensitivity (SE), specificity (SP) and AUC. Cells from one patient or image are not independent, so NIH intervals resample patients and external intervals resample images. Where several checkpoints are compared, the bootstrap also resamples folds. E1's minimum detectable effect of .29 BACC points
was used as the threshold for meaningful change on NIH.

### 2.7 Predictions and corrections

Predictions, with decision rules and falsification conditions, were written before running each registered experiment, and are listed in `registry.csv`. Experiments without one, such as E8 and E1r, are reported as exploratory. An AI assistant wrote many of these predictions and proposed parts of the experimental design, and each record says so. **These dates are self-reported.** Only the Milestone 3 submission (5 October 2026) is externally dated. The repository's first research commit is 2 November 2026, and later registrations are
local commits, not externally timestamped. Where a later check overturned a claim, the correction was added above the original text. Exploratory analyses are labelled as such.

## 3. Results

### 3.1 The structure of the NIH dataset

Infection status is confounded with imaging date. Of 56 dates, only 4 hold both healthy and infected patients, covering 7 of 50 healthy and 27 of 150 infected patients, and 31 dates span several test folds of the grouped split. Colour follows date: among 13,665 uninfected cells, date explained 86.5% of the colour variance that patient explained, against a predicted
minimum of 70%. Cell count alone separates infected from healthy patients with AUC .992 (exploratory), because about 68 uninfected cells were sampled per patient in both groups, so infected patients simply have more cells. Any patient-level result on natural bags must be read with care.

On uninfected cells alone, frozen RedDino features separated healthy from infected patients with AUC .933. Restricted to shared dates, this fell to .767 [.519, .958], narrowly missing our prediction of below .75. With 7 healthy patients the interval is too wide to say how much of this signal is batch.

### 3.2 Leakage and internal accuracy

Patient leakage barely changed DANet's score: leaky BACC 96.96 ± .20 against grouped 96.71 ± .08, a gap of +.26, below the MDE and against our Milestone 3 prediction of a 1 to 2.5 point drop. Session leakage: `[PENDING E10 part B]`.

DANet matched a generic ResNet-18 on unseen patients (96.58 ± .05; paired difference +.13 [−.05, .30]), and frozen probes came within a point (DINOv2 96.03, DinoBloom 96.31, RedDino 96.45). DANet's margin over RedDino was above zero on repeat 0 only.

An early finding did not hold up. LayerNorm gained more from leakage than InstanceNorm on seeds 42-44 (leaky BACC 97.34 against 96.80, exploratory). A registered replication on new seeds gave a mean paired difference of +.15 [−.05, .36], and the claim was not replicated under our rule. We make no claim that the norm layer controls leakage use.

### 3.3 A single measurement on NIH

The dark-spot rule reached AUC .981 and out-of-fold BACC 95.62 on NIH (exploratory). DANet remained reliably better: BACC +1.16 [.90, 1.45], AUC +.011 [.008, .015], 888 against 1,208 errors, and false positives on healthy patients of 1.96% against 3.63%. So DANet's training adds about one point over one measurement, and that point is real.

### 3.4 What explanations show

A rule with no model, pointing to the darkest smoothed pixel, hit annotated parasites more often than Grad-CAM [14] on all 15 checkpoints. Darkest pixel minus Grad-CAM was +27.3 [15.8, 38.0] for E1, +4.4 [2.1, 6.9] for E8b and +55.6 [22.1, 88.6] for E8c. Grad-CAM was also unstable: E8c hit about 90% on two folds and 8 to 16% on the other three, so our original
single-checkpoint result was not representative. These hits show dependence on a dark object. They give no evidence of learned morphology.

### 3.5 External development on MP-IDB

No architecture ranking survived five checkpoints per arm (E8b − E1: +1.67 [−1.98, 5.42]; E8c − E1: −2.14 [−7.84, 3.24]), and we withdrew our earlier claim that gate and dilation drive robustness. Between-checkpoint spread was large: E1 BACC 72.75 ± 8.41 on repeat 0, and 71.56 ± 6.53 on 10 further checkpoints. ResNet-18 scored 67.41 ± 5.81 across 15 checkpoints.
Seed-only spread: `[PENDING E12]`.

Training-time colour transfer raised BACC by +7.05 [4.35, 9.65] and AUC by +.059 [.036, .088], against our prediction of little change. Test-time Reinhard normalisation lowered BACC by 18.75 [15.03, 22.73].

The dark-spot rule reached AUC .987 and BACC 93.0 on MP-IDB (exploratory; development data).

### 3.6 Ranking is stable where the threshold is not

Candidate 3 had similar AUC across crop masks, and similar BACC when given a best-case threshold, yet its BACC at the fixed threshold ranged from 71.7 to 91.2 (Table 2; 15 of 20 members, ResNet-18s pending). The feature's BACC moved by under 4 points. ResNet-18 showed the same split: it had higher AUC than E1 (.902 against .874) but lower BACC, with SE 35.6 and SP 99.2. A model's external operating point depends on how the crop was prepared, which NIH cannot reveal.

**Table 2. MP-IDB by crop mask.**

| Crops | Feature AUC / BACC | Candidate 3 AUC / BACC | Candidate 3 SE | Best-threshold BACC (upper bound) |
|---|---|---|---|---|
| Cellpose | .987 / 93.0 | .973 / 82.0 | 64.7 | 92.6 |
| Inscribed ellipse | .982 / 93.9 | .966 / 71.7 | 43.9 | 91.7 |
| Otsu | .970 / 90.3 | .973 / 91.2 | 85.0 | 92.7 |

### 3.7 One-shot test on BBBC041

`[PENDING E11. Report in this order: integrity checks; candidate table for both crop rules; delta with 97.5% CIs; the outcome; by stage and group; leukocytes; PPV at 3.05% prevalence; probes.]`

### 3.8 Prediction record

`[Table of every prediction and its outcome, from registry.csv. It must include E1, E4, E6, E6c and E10 part A's misses, E5b's missed centre range, and all corrections.]`

## 4. Discussion: one outline per E11 outcome

Use exactly one. The outcome is set by the rule, and the outline must not be swapped afterwards.

**Common ground, whatever the outcome.** NIH accuracy near 97% mostly reflects a task with a strong, simple cue; DANet adds a real but small margin; patient leakage is small; a single checkpoint is a poor unit of evidence externally; ranking and threshold fail differently.

| E11 outcome | Headline | Main argument | What it does not show |
|---|---|---|---|
| **Feature holds** (both crop rules) | One NIH-fitted measurement matches trained ensembles on a new species | NIH training adds little that transfers. Baselines of this kind should be mandatory | Clinical readiness, or that models cannot do better with target data |
| **Trained model wins** (both) | Trained ensembles keep something a simple cue lacks | On curated NIH the gap is small, but it widens under shift. Colour transfer matters | That the models learned morphology: E5b still stands |
| **Ranking and threshold disagree** (both) | Good ranking does not give a usable operating point | External validation must report threshold-free and fixed-threshold results together. Local calibration is needed | Which candidate is better overall |
| **Depends on crop rule** | The verdict turns on preprocessing | The result is itself the finding: preprocessing, invisible on NIH, can decide an external comparison, so papers must report it | Any ranking of candidates. We make no such claim |
| **No transfer** | Neither approach survives species and stain shift | NIH-trained classifiers and NIH-fitted cues do not carry over. It matches MalariaNet's result | That the methods fail on *P. falciparum* elsewhere |
| **Inconclusive** | The test could not separate them | State the CI widths plainly, and report what they exclude. Do not lean on point estimates | Equivalence |

**Limitations to state in every case.** Prediction dates are self-reported. Date is a proxy for session. E7's flagged cells were not reviewed by an expert. MP-IDB was reused for development, so its feature results are exploratory. The crop audit was done by an AI assistant, not a microscopist. BBBC041's boxes are not masks. Species, stage mix and stain all change at once, so
transfer failures cannot be assigned to one cause. Curated crops are not a field workflow, and no result here supports clinical use.

## Declarations

**Ethics statement.** `[Authors to confirm.]` This study used only public, de-identified datasets and involved no new human participants or samples, so we believe no ethics approval was required.

**Data and code statement.** `[Authors to confirm the repository plan.]` NIH, MP-IDB and BBBC041 are publicly available. BBBC041 is CC BY-NC-SA 3.0, so its images are not redistributed. Code, split files, predictions and scores will be released at `[URL]`.

**Declaration of competing interest.** `[Authors to confirm.]`

**Funding.** `[Authors to confirm.]`

**CRediT author statement.** `[Authors to complete.]`

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** `[Authors to finalise.]` During the preparation of this work the authors used Claude Code (Anthropic) to propose experiments and write dated predictions, write analysis code, analyse results, audit crops, and draft text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

Status: **V** = checked online on 2026-11-02 for this draft, **B** = from the project's base-paper records, **C** = from memory, must be checked against the source. Reformat to the journal's style at submission.

1. **V** S. Rajaraman et al., Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images, *PeerJ* 6 (2018) e4568. doi:10.7717/peerj.4568
2. **B** A. Ali et al., DANet: a lightweight dilated attention network for malaria parasite detection, *Sci. Rep.* 15 (2025) 36689. doi:10.1038/s41598-025-20402-1
3. **V** A. D'Amour et al., Underspecification presents challenges for credibility in modern machine learning, *J. Mach. Learn. Res.* 23(226) (2022) 1-61.
4. **C** R. Geirhos et al., Shortcut learning in deep neural networks, *Nat. Mach. Intell.* 2 (2020) 665-673.
5. **V** M. Hou et al., MalariaNet: a microcontroller-deployable malaria-microscopy detector for point-of-care biosensing under leakage-free evaluation, *Biosensors* 16(7) (2026) 358. doi:10.3390/bios16070358
6. **V** M.A. Al Kafi et al., EMFE: a lightweight, explainable machine learning framework for malaria cell classification, arXiv:2608.24793 (2026).
7. **V** A. Loddo et al., MP-IDB: the malaria parasite image database for image processing and analysis, *LNCS* 11379 (2019) 57-65. doi:10.1007/978-3-030-13835-6_7 (sources give 2018 or 2019: check the Springer page)
8. **V** J. Hung, A. Carpenter, Applying Faster R-CNN for object detection on malaria images, CVPR Workshops (2017). doi:10.1109/CVPRW.2017.112 (page numbers conflict between sources). Add **C** V. Ljosa et al., Annotated high-throughput microscopy image sets for validation, *Nat. Methods* 9 (2012) 637.
9. **C** C. Stringer et al., Cellpose: a generic algorithm for cellular segmentation, *Nat. Methods* 18 (2021) 100-106.
10. **C** E. Reinhard et al., Color transfer between images, *IEEE Comput. Graph. Appl.* 21(5) (2001) 34-41.
11. **C** M. Oquab et al., DINOv2: learning robust visual features without supervision, arXiv:2304.07193 (2023).
12. **V** V. Koch et al., DinoBloom: a foundation model for generalizable cell embeddings in hematology, MICCAI (2024). arXiv:2404.05022
13. **V** L. Zedda et al., RedDino: a foundation model for red blood cell analysis, MICCAI (2025). arXiv:2508.08180
14. **C** R.R. Selvaraju et al., Grad-CAM: visual explanations from deep networks via gradient-based localization, ICCV (2017) 618-626.

**Submission checklist.** Abstract at most 250 words. 3-5 highlights of at most 85 characters each, in a separate file. 1-7 keywords. Declaration of AI use before the references. Ethics statement or the reason none is needed. STARD reviewed for the external test. Section chosen at submission (Medical Imaging, or Foundations of AI in Medicine). Graphical abstract and data statement encouraged.
