# What does the NIH malaria cell benchmark measure? An audit of DANet with a one-feature baseline

**Working draft for *Computers in Biology and Medicine* (Full Length Article). Not for submission as written.**

Deepanshu Mehta, Agrapujya Lashkari. Affiliation and corresponding author to be added.

> **Note to authors.** The AI assistant drafted this text. Journal policy requires the manuscript to reflect your own analysis, interpretation, insights and ideas, so read every claim against
> `research/registry.csv` and rewrite it in your own words before submission. `[PENDING]` marks results that do not exist yet. The BBBC041 result was calculated once on 2026-11-03 and is final.

## Abstract
Malaria cell classifiers report near 98% accuracy on the NIH thin-smear benchmark, but such scores may not show what transfers. We reproduced DANet and audited its score with dated predictions, patient- and session-grouped splits, retraining with new seeds, and external development on MP-IDB. We then froze four candidates, a one-feature dark-spot rule fitted on NIH, two trained ensembles and two foundation-model probes, and tested them once on BBBC041 (*P. vivax*; 79,619 cells; 3.05% prevalence) under a pre-specified outcome rule applied to two crop rules. A colour-transfer ensemble beat the feature under both crop rules, with AUC .977 against .909 and balanced accuracy 92.1 against 61.3 on ellipse crops. The differences were +.068 [.053, .084] and +30.8 [29.8, 31.9] points. Our leading prediction, that the feature would hold, was wrong. The feature kept its ranking but not its NIH-fitted threshold, calling 82% of uninfected cells in the larger acquisition group. On NIH, the feature came within about one point of DANet, patient leakage changed balanced accuracy by +.26 points, and the session effect (+.24 [−.06, .59]) stayed below our detection limit. On MP-IDB, retraining with only a new seed moved balanced accuracy with SD 6.63, against .18 on NIH. NIH accuracy mostly reflects a strong, simple cue, and neither NIH nor a reused development set predicted behaviour on a new dataset. External tests should use frozen candidates and ensembles, and report threshold-free and fixed-threshold results together.

**Highlights (each at most 85 characters):**
- A one-feature rule nearly matched DANet on NIH but lost its threshold on BBBC041
- Colour-transfer ensembles won a frozen one-shot test on a new dataset
- Seed alone moved external balanced accuracy by SD 6.6, against 0.2 on NIH
- Patient leakage changed NIH balanced accuracy by only 0.26 points
- Every prediction was dated before its run, and misses stay on record

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

All patient-level splits were built before training and hashed. **Grouped:** five folds over patients, three repeats, balanced across four strata. **Leaky:** a random cell-level 70/10/20 split (seed 42), our reconstruction of DANet's unseeded split. **Session:** patients sharing any imaging date
(from file names) were linked into blocks by connected components, giving 50 blocks. Whole blocks were assigned to five folds, so training, validation and test sets share no date. Test folds held 8 to 13 healthy patients, with test prevalence SD .10
percentage points. Date is a proxy for session.

### 2.3 Models

**DANet.** We used DANet-S with the authors' published code (softmax gate, cascaded dilations, InstanceNorm; arm E1) and a paper-faithful variant (sigmoid gate, parallel dilations) with LayerNorm (E8b) or InstanceNorm (E8c). Training used RMSprop with a
one-cycle schedule, maximum learning rate .001, batch size 64, 224 px and 50 epochs (the M2 reproduction used the paper's 75), negative log-likelihood loss, rotation (±15°) and flip (p = .2) augmentation, keeping the checkpoint with the lowest validation loss. The paper's listed learning rate (.01) did not train. **Ablations (E13).** E13a removes the attention path of every Dilated Attention Block, leaving its 1×1 projection (42,778 parameters); E13b replaces the average-pool head with a max pool. **ResNet-18.** ImageNet weights, AdamW, 128 px, 15 epochs. **Colour transfer.** During training only, with probability .5, a cell was recoloured to another training patient's
statistics by Reinhard transfer [10] in LAB space, foreground only. **Frozen probes.** Logistic regression on DINOv2 [11], DinoBloom [12] and RedDino [13] embeddings. RedDino was not used externally, because it was trained on MP-IDB and its training list could not be
confirmed free of BBBC041.

### 2.4 Dark-spot baseline

For a crop with background set to zero, resized to 128 px, the score is (median − min) / median of the Gaussian-smoothed green channel (sigma 1.5) inside the foreground eroded by 4 px. A cell is called Parasitized when the score is at least .3047, the NIH maximiser of balanced accuracy,
rounded to four places. The rule has one threshold and no other fitted parameter. **The feature was chosen after inspecting explanation results on MP-IDB, so its MP-IDB performance is exploratory.** Only BBBC041 tests it.

### 2.5 External evaluation

No model, threshold or normalisation was fitted on external data. On MP-IDB, E6b scored 15 checkpoints across three arms; E6c added colour-transfer checkpoints. E12 fixed the training data and varied only the seed: E1 configuration, grouped repeat 0, two randomly drawn folds and seeds 1 to 4, scored on MP-IDB at threshold .5 and summarised as a pooled within-fold SD with a chi-square CI.

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

Primary metrics were balanced accuracy (BACC), sensitivity (SE), specificity (SP) and AUC. Cells from one patient or image are not independent, so NIH intervals resample patients and external intervals resample images. Where several checkpoints are compared, the bootstrap also resamples folds. An A/A test between runs differing only in seed gave a noise floor of .29 BACC points (2.8 times the paired SE), which we call the minimum detectable effect (MDE) and use as a practical-relevance threshold on NIH. It is not a power calculation: an effect below it can have a CI that excludes zero, and we report such effects as real but too small to matter.

### 2.7 Predictions and corrections

Predictions, with decision rules and falsification conditions, were written before running each registered experiment, and are listed in `registry.csv`. Experiments without one, such as E8 and E1r, are reported as exploratory. An AI assistant wrote many of these predictions and proposed parts of the experimental design, and each record says so. **These dates are self-reported.** Only the Milestone 3 submission (5 October 2026) is externally dated. The repository's first research commit is 2 November 2026, and later registrations are
local commits, not externally timestamped. Where a later check overturned a claim, the correction was added above the original text. Exploratory analyses are labelled as such.

## 3. Results

### 3.1 The structure of the NIH dataset

Infection status is confounded with imaging date. Of 56 dates, only 4 hold both healthy and infected patients, covering 7 of 50 healthy and 27 of 150 infected patients, and 31 dates span several test folds of the grouped split. Colour follows date: among 13,665 uninfected cells, date explained 86.5% of the colour variance that patient explained, against a predicted
minimum of 70%. Cell count alone separates infected from healthy patients with AUC .992 (exploratory), because about 68 uninfected cells were sampled per patient in both groups, so infected patients simply have more cells. Any patient-level result on natural bags must be read with care.

On uninfected cells alone, frozen RedDino features separated healthy from infected patients with AUC .933. Restricted to shared dates, this fell to .767 [.519, .958], narrowly missing our prediction of below .75. With 7 healthy patients the interval is too wide to say how much of this signal is batch.

### 3.2 Leakage and internal accuracy

Patient leakage barely changed DANet's score: leaky BACC 96.96 ± .20 against grouped 96.71 ± .08, a gap of +.26, below the MDE and against our Milestone 3 prediction of a 1 to 2.5 point drop. Moving to session-held-out folds, with train, validation and test sharing no date, changed pooled BACC from 96.78 to 96.54, a gap of +.24 [−.06, .59] on one repeat. This is below the MDE and below our prediction of about +.35, so the rule does not call session leakage real, and AUC did not move (−.0001 [−.0009, .0007]). The gap sits close to the MDE and our protocol asks for a second repeat before any claim, so we do not claim that session leakage is absent either.

DANet matched a generic ResNet-18 on unseen patients (96.58 ± .05; paired difference +.13 [−.05, .30]), and frozen probes came within a point (DINOv2 96.03, DinoBloom 96.31, RedDino 96.45). DANet's margin over RedDino was above zero on repeat 0 only.

**Architecture ablation (E13).** Removing attention cost −.74 BACC [−1.08, −.43], lower on all five folds and almost all through sensitivity (−1.47; SP −.01), so a network with 1.9% of DANet's parameters still reached 96.04. A max-pool head cost −.19 [−.37, −.01], a CI that excludes zero but an effect below the MDE. One seed per arm.

An early finding did not hold up. LayerNorm gained more from leakage than InstanceNorm on seeds 42-44 (leaky BACC 97.34 against 96.80, exploratory). A registered replication on new seeds gave a mean paired difference of +.15 [−.05, .36], and the claim was not replicated under our rule. We make no claim that the norm layer controls leakage use.

### 3.3 A single measurement on NIH

The dark-spot rule reached AUC .981 and out-of-fold BACC 95.62 on NIH (exploratory). DANet remained reliably better: BACC +1.16 [.90, 1.45], AUC +.011 [.008, .015], 888 against 1,208 errors, and false positives on healthy patients of 1.96% against 3.63%. So DANet's training adds about one point over one measurement, and that point is real.

### 3.4 What explanations show

A rule with no model, pointing to the darkest smoothed pixel, hit annotated parasites more often than Grad-CAM [14] on all 15 checkpoints. Darkest pixel minus Grad-CAM was +27.3 [15.8, 38.0] for E1, +4.4 [2.1, 6.9] for E8b and +55.6 [22.1, 88.6] for E8c. Grad-CAM was also unstable: E8c hit about 90% on two folds and 8 to 16% on the other three, so our original
single-checkpoint result was not representative. These hits show dependence on a dark object. They give no evidence of learned morphology.

### 3.5 External development on MP-IDB

No architecture ranking survived five checkpoints per arm (E8b − E1: +1.67 [−1.98, 5.42]; E8c − E1: −2.14 [−7.84, 3.24]), and we withdrew our earlier claim that gate and dilation drive robustness. Between-checkpoint spread was large: E1 BACC 72.75 ± 8.41 on repeat 0, and 71.56 ± 6.53 on 10 further checkpoints. ResNet-18 scored 67.41 ± 5.81 across 15 checkpoints.
With the training patients fixed, the seed alone gave a pooled BACC SD of 6.63 [4.48, 12.71] on MP-IDB, above our predicted 3 to 6, and an AUC SD of .027 [.018, .052]. The same checkpoints had an SD of .18 [.12, .35] on NIH. On one fold, two seeds differed by .15 on NIH and by 13.7 on MP-IDB, and E1's best value of 85.7 fell to a mean of 74.1 on new seeds. Sensitivity drove the spread (19.9 to 74.4), while specificity stayed between 94.5 and 99.3.

Training-time colour transfer raised BACC by +7.05 [4.35, 9.65] and AUC by +.059 [.036, .088], against our prediction of little change. Test-time Reinhard normalisation lowered BACC by 18.75 [15.03, 22.73]. The registered protocol paired a field-level source std with a cell-level target std; an unregistered rerun with matched definitions was still −15.37 [−18.73, −12.29], so the flaw explains only part of the harm.

The dark-spot rule reached AUC .987 and BACC 93.0 on MP-IDB (exploratory; development data).

### 3.6 Ranking is stable where the threshold is not

Candidate 3 had almost identical AUC across crop masks (.981 to .982), and similar BACC when given a best-case threshold (94.2 to 95.3), yet its BACC at the fixed threshold ranged from 78.8 to 91.1 (Table 2). The feature's BACC moved by under 4 points. ResNet-18 showed the same split: it had higher AUC than E1 (.902 against .874) but lower BACC, with SE 35.6 and SP 99.2. A model's external operating point depends on how the crop was prepared, which NIH cannot reveal.

**Table 2. MP-IDB by crop mask.**

| Crops | Feature AUC / BACC | Candidate 3 AUC / BACC | Candidate 3 SE | Best-threshold BACC (upper bound) |
|---|---|---|---|---|
| Cellpose | .987 / 93.0 | .982 / 83.8 | 68.0 | 94.5 |
| Inscribed ellipse | .982 / 93.9 | .982 / 78.8 | 58.2 | 95.3 |
| Otsu | .969 / 90.3 | .981 / 91.1 | 83.4 | 94.2 |

### 3.7 One-shot test on BBBC041

Both crop rules were scored in full: 79,722 crops each, crop order and file hashes verified, no missing values and no rescore. Both score manifests were committed before any metric was calculated, and each crop rule was calculated once. The primary set held 79,619 cells from 1,328 images, with 2,425 parasitized (3.05%).

**Table 3. BBBC041, one-shot.** Thresholds frozen on NIH. PPV at the natural 3.05% prevalence.

| Candidate | Ellipse AUC / BACC | SE / SP | PPV | Otsu AUC / BACC | SE / SP | PPV |
|---|---|---|---|---|---|---|
| 1. Dark-spot feature | .909 / 61.3 | 99.7 / 22.9 | 3.9 | .906 / 64.4 | 99.3 / 29.6 | 4.2 |
| 2. E1 ensemble | .966 / 87.9 | 98.4 / 77.4 | 12.0 | .975 / 81.8 | 99.8 / 63.8 | 8.0 |
| 3. Colour-transfer ensemble | .977 / 92.1 | 92.3 / 91.9 | 26.4 | .980 / 89.2 | 98.9 / 79.5 | 13.2 |
| 4. DINOv2 probe | .936 / 77.3 | 97.2 / 57.5 | 6.7 | not run | | |
| 4. DinoBloom probe | .694 / 63.6 | 81.3 / 45.9 | 4.5 | not run | | |

Candidate 3 minus candidate 1 was +.068 [.053, .084] AUC and +30.8 [29.8, 31.9] BACC on ellipse crops, and +.074 [.058, .090] and +24.8 [23.7, 25.7] on Otsu crops (97.5% CIs). Both lower bounds cleared their margins under both crop rules, so the registered outcome is **Trained model wins**. We had given this outcome 15% (5% in the addendum) and expected the feature to hold.

The feature failed through its threshold. On the JPG group (120 images) it reached AUC .990 and BACC 96.4, close to candidate 3 (.999, 98.4). On the PNG group (1,208 images, 93% of cells) its AUC was .957 but its BACC was 58.9, because the NIH-fitted threshold called 82% of uninfected cells (exploratory). A threshold chosen on BBBC041 itself would have raised it only to 84.8, since the two groups needed different thresholds (exploratory). Candidate 3's fixed threshold was close to its best case (92.1 against 92.3).

Sensitivity by stage was 98.5 to 100 for the feature at every stage. For candidate 3 it was lowest for rings on ellipse crops (75.1; Otsu 95.8) and at least 96.5 otherwise. Of 103 leukocytes, the feature flagged 99.0% and candidate 2 98.1%, while candidate 3 flagged 23.3% (Otsu: 98.1, 87.4 and 24.3%). Both probes ranked below candidate 3, and DinoBloom ranked poorly, as on MP-IDB.

### 3.8 Prediction record

Table 4 lists every dated prediction and its outcome, including misses and later corrections. Only E1 and E4 were dated externally (Milestone 3, 2026-10-05). All other dates are self-reported.

**Table 4. Prediction record.**

| Experiment | Written | Prediction | Outcome |
|---|---|---|---|
| E1 | 2026-10-05 | Patient leakage inflates BACC by 1 to 2.5 points | **Missed:** gap +.26 |
| E4 | 2026-10-05 | ColorJitter adds BACC +.3 to +1.0 and SE +.5 to +1.5 on unseen patients | **Falsified** by its registered rule |
| E3 | 2026-10-29 | DANet can memorise random patient labels | **Withdrawn:** the runs underfit |
| E5 | 2026-10-29 | Grad-CAM localisation on MP-IDB | Directionally met, far above prediction. **Reading corrected** by E5b: dependence on a dark object, not learned morphology |
| E6 | 2026-10-29 | External behaviour by architecture arm | **Falsified** by its registered condition, and the ranking was **withdrawn** after E6b |
| E7 | 2026-10-29 | Share of genuine label errors among flagged cells | **Reading withdrawn:** cells are flagged, not adjudicated |
| E9 | 2026-10-29 | Probe BACC 75 to 93, in a stated order | Order **met**, magnitude **missed:** 96.0 to 96.5 |
| E4b | 2026-10-29 | Colour transfer reduces leakage use in E8b | Primary outcomes met on one seed. **Causal reading withdrawn** |
| E6b | 2026-11-02 | No robust architecture ranking across five checkpoints | **Supported;** the predicted range for E8c − E1 was missed |
| E6c | 2026-11-02 | Training-time colour transfer changes little; test-time normalisation helps | **Falsified** on both: +7.05 and −18.75 BACC |
| E5b | 2026-11-02 | Darkest pixel beats Grad-CAM; a centre rule stays in a stated range | Primary **supported** (15 of 15). Centre range **missed** (63.2) |
| E8r | 2026-11-02 | LayerNorm's leakage gain replicates, about +.3 | **Not replicated:** +.15 [−.05, .36] |
| E10 A | 2026-11-02 | Date explains at least 70% of patient colour variance; RedDino shared-date AUC below .75 | First **supported** (86.5%). Second **missed narrowly** (.767) |
| E10 B | 2026-11-02 | Session gap about +.35, falsified below .29 | **Not supported** by its rule: +.24 [−.06, .59]. The CI holds both 0 and +.35, so unresolved; one repeat |
| E12 | 2026-11-02 | Seed SD 3 to 6 BACC on MP-IDB; NIH SD under .5; fold 2 regresses by at least 3; unused checkpoints agree | SD range **missed high** (6.63). The other three **supported** |
| E13 | 2026-11-03 | No attention: BACC −.5 to −2.5, CI below 0, SE falls more than SP; max-pool head: |ΔBACC| < .29, CI includes 0 | No attention **supported** (−.74). Max-pool **missed:** −.19, CI excludes 0 |
| E11 | 2026-11-02 | Feature holds most likely (45%); feature BACC 78 to 90; candidate 3 BACC 65 to 85 | **Wrong:** Trained model wins (15%). Feature BACC **missed** (61.3), candidate 3 **missed high** (92.1). Probe, leukocyte, PPV and group predictions **supported** |

E8 carried no prediction, was exploratory, and did not replicate in E8r.

## 4. Discussion

The outcome was fixed by a rule written before scoring, and so was the Discussion it selects: six outlines were drafted in advance, and this is the one for "Trained model wins".

**On a dataset that played no part in development, trained ensembles beat a single NIH-fitted measurement, on ranking and on fixed-threshold accuracy, under both crop rules.** Our leading prediction was that the feature would hold. It did not, and the margins were wide.

On curated NIH crops the gap between a trained model and the feature is about one point. Under shift to BBBC041 it grew to 25 to 31 points of balanced accuracy. Most of that came from the operating point, not the ranking: the feature's AUC stayed above .90, and on the smaller acquisition group it nearly matched the ensembles. On the larger group its NIH threshold called most uninfected cells, the same high-sensitivity, low-specificity pattern MalariaNet reported on BBBC041 (sensitivity 96.4%, specificity 18.0%) [5], although its 53.2% accuracy implies a far higher parasitized share than our natural 3.05%, so the figures are not directly comparable. A fixed cue can keep its ranking and still lose its threshold, and neither NIH nor MP-IDB revealed this.

Colour transfer may have mattered, but this test does not isolate it. Of the two trained ensembles, the colour-transfer one had the higher AUC under both crop rules (+.011 [.007, .017] ellipse) and kept a usable threshold, and it was the only candidate that rejected most leukocytes. But candidate 3 differs from candidate 2 in colour transfer, ensemble size (20 against 5) and architecture mix (three DANet variants and ResNet-18) at once. The direction fits E6c, where colour transfer alone helped on MP-IDB. But E12 showed that a new seed alone moves external balanced accuracy with an SD of 6.6, so we tested ensembles, and a single checkpoint might not repeat this result.

Development data misled us. Rehearsal on MP-IDB, reused for many decisions including the feature's discovery, favoured the feature on ellipse crops. BBBC041 reversed that. External claims should rest on data untouched by development, with candidates frozen and hashed before scoring.

**Common ground.** NIH accuracy near 97% mostly reflects a task with a strong, simple cue. DANet adds a real but small margin on NIH. Patient leakage is small, and the session effect is unresolved at one repeat. A single checkpoint is a poor unit of evidence externally. Ranking and threshold fail differently, for trained models on MP-IDB and for the feature on BBBC041.

**What this does not show.** It does not show that the models learned parasite morphology: E5b found that a no-model rule pointing to the darkest pixel hit parasites more often than Grad-CAM. It cannot assign the shift to species, stain or acquisition, which change together. It is one dataset, and PPV of 13 to 26% at 3% prevalence rules out any claim of clinical use.

**Limitations.** Prediction dates are self-reported. Date is a proxy for session. E7's flagged cells were not reviewed by an expert. MP-IDB was reused for development, so its feature results are exploratory. The crop audit was done by an AI assistant, not a microscopist. BBBC041's boxes are not masks. Red cell boxes overlapping a parasite box (4,344) were excluded, removing some hard negatives, so specificity may be optimistic. Candidates 2 and 3 differ in more than colour transfer. E13 used one seed per arm. Species, stage mix and stain all change at once, so
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
4. **V** R. Geirhos et al., Shortcut learning in deep neural networks, *Nat. Mach. Intell.* 2 (2020) 665-673.
5. **V** M. Hou et al., MalariaNet: a microcontroller-deployable malaria-microscopy detector for point-of-care biosensing under leakage-free evaluation, *Biosensors* 16(7) (2026) 358. doi:10.3390/bios16070358
6. **V** M.A. Al Kafi et al., EMFE: a lightweight, explainable machine learning framework for malaria cell classification, arXiv:2608.24793 (2026).
7. **V** A. Loddo et al., MP-IDB: the malaria parasite image database for image processing and analysis, *LNCS* 11379 (2019) 57-65. doi:10.1007/978-3-030-13835-6_7 (sources give 2018 or 2019: check the Springer page)
8. **V** J. Hung, A. Carpenter, Applying Faster R-CNN for object detection on malaria images, CVPR Workshops (2017). doi:10.1109/CVPRW.2017.112 (page numbers conflict between sources). Add **V** V. Ljosa et al., Annotated high-throughput microscopy image sets for validation, *Nat. Methods* 9 (2012) 637.
9. **V** C. Stringer et al., Cellpose: a generalist algorithm for cellular segmentation, *Nat. Methods* 18 (2021) 100-106.
10. **V** E. Reinhard et al., Color transfer between images, *IEEE Comput. Graph. Appl.* 21(5) (2001) 34-41. doi:10.1109/38.946629 (Crossref gives issue 4, common citations give issue 5: check IEEE Xplore).
11. **V** M. Oquab et al., DINOv2: learning robust visual features without supervision, arXiv:2304.07193 (2023).
12. **V** V. Koch et al., DinoBloom: a foundation model for generalizable cell embeddings in hematology, MICCAI (2024). arXiv:2404.05022
13. **V** L. Zedda et al., RedDino: a foundation model for red blood cell analysis, MICCAI (2025). arXiv:2508.08180
14. **V** R.R. Selvaraju et al., Grad-CAM: visual explanations from deep networks via gradient-based localization, ICCV (2017) 618-626.

**Submission checklist.** Abstract at most 250 words. 3-5 highlights of at most 85 characters each, in a separate file. 1-7 keywords. Declaration of AI use before the references. Ethics statement or the reason none is needed. STARD reviewed for the external test. Section chosen at submission (Medical Imaging, or Foundations of AI in Medicine). Graphical abstract and data statement encouraged.
