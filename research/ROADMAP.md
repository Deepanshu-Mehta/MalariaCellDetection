# Research roadmap: DANet shortcut audit (NIH Malaria)

Working plan for the paper beyond the course milestones. The background reports are
`reports/Malaria DANet paper experiments.md` (first plan) and `reports/Malaria DANet review and phased plan.md`
(review + corrections). This file supersedes their venue and timeline sections.

## 1. Venues (revised 2026-11-02)

**Rule: submit only where LLM use for ideas, design and analysis is allowed.** MIDL 2027 and TMLR are dropped. MIDL
forbids using an LLM to generate "research ideas, study designs, analyses, results, or substantive text". TMLR
allows writing help but expects "the ideas, claims and results found in submissions are human-sourced, not produced
by such tools" ([FAQ](https://jmlr.org/tmlr/faq.html)).

| Venue | Fit | AI-use policy (checked 2026-11-02) | Timing | Cost |
|---|---|---|---|---|
| **Elsevier: *Computers in Biology and Medicine*** (primary) | Applied medical-imaging venue | Publisher policy allows AI for ideas and in research design or methods, described in Methods, with a separate AI declaration. AI must not replace the authors' judgement, and the manuscript must reflect their own analysis and conclusions ([Elsevier](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)). Journal-specific instructions not yet checked | Rolling; target submission **mid-Feb 2027** | Hybrid journal: subscription route may be free (verify) |
| **IEEE: *IEEE Access*** (backup), ISBI 2027 1-page abstract | Fast journal; poster visibility | Disclose AI-generated content in Acknowledgments with system, sections and level of use; the page sets no limit on ideas or analysis ([IEEE](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/)) | ISBI abstract due **1 Feb 2027** (non-archival poster, Lausanne, in person) | IEEE Access APC ≈ $2,160 |
| ReScience C | Pure reproduction of DANet | Not found in a 2026-11-02 search; check before targeting | Rolling | Free |

**Excluded or not recommended under the rule:**
- **Springer Nature (*Scientific Reports*, *Malaria Journal*):** guidance says AI "should not replace original
  scientific thinking or be used to write entire manuscripts or interpret data"
  ([Springer Nature](https://group.springernature.com/gp/group/ai/ai-guidance-for-our-researchers-and-communities)).
- **MICCAI workshops:** MICCAI 2026 allowed LLMs "as a general-purpose writing assistance tool" only
  ([guidelines](https://conferences.miccai.org/2026/en/PAPER-SUBMISSION-GUIDELINES.html)). Recheck only if a 2027 CFP
  changes this.
- **MELBA:** a third-party copy of its submission template limits LLMs to polishing and bullet-point outlines. Not
  confirmed on the journal's own site.

**Route:** arXiv preprint → *Computers in Biology and Medicine* → *IEEE Access* if rejected. Never have the same
content under review at two places at once.

**AI use (revised 2026-11-02):** disclosure is required everywhere, and no venue lets AI replace the authors' own
judgement. Authors must check and own every interpretation and conclusion. Pre-registration discipline does not
change: a prediction is written and dated *before* a run happens, regardless of who or what writes it, because that
is what makes a negative result credible.

## 2. Phases (revised: no 4 Dec crunch)

GPU-hours are estimates scaled from the M2 run and must be re-timed (see Phase 0 to-do).

| Phase | Dates | Goal | Experiments | T4 GPU-h |
|---|---|---|---|---|
| 0 | 22 Oct – 1 Nov 2026 | Leakage-proof data layer, frozen splits, timing, registry | E0 | ≈6 |
| 1 | 2 Nov – 30 Oct 2026 (ran faster than planned) | Core audit (also covers course M4: rows B0/B1/H/C) | E1, E1r, E2, E3, E4, E5, E6, E7, E8, E9, E4b | **✅ DONE 2026-10-30** (actual: ~70h GPU-quota across the week, more than the ~40h estimate -- colour-transfer and foundation-model jobs cost more than expected) |
| 2 | Nov 2026 – 14 Feb 2027 | Remedies + writing; **journal submission mid-Feb** | See §6 (revised priorities below) | TBD |
| 3 | Mar – Jul 2027 | Journal revisions; follow-up paper | diffusion counterfactuals, SAE analysis, optional reader study | ≈60–100 |

**Phase 1 finished on 2026-10-30, and E6b and E6c followed on 2026-11-02.** All experiments except E10 (E0, E1, E1r, E2,
E3, E4, E4b, E5, E6, E6b, E6c, E7, E8, E9) have a result and verdict in `registry.csv` and their own `experiments/<ID>/README.md`.** See §6
for the full results summary and §7 for the revised Phase 2 plan based on what was actually found (several
results came back differently than predicted, which changes what Phase 2 should prioritise).

## 3. Phase 0 status

| Task | Status |
|---|---|
| Patient IDs from NIH CSVs, merged on `C<number>`: 200 patients (150 infected + 50 healthy-only) | ✅ C47P8 is the only merge |
| Healthy-only share of uninfected cells / largest patient | ✅ 24.8 % / C68, 702 cells (2.5 %) |
| Patient-balanced folds, 5 folds × 3 repeats, leakage assertions | ✅ rank-blocked strata; test prevalence 48–51 %, SD 1.09 pp (random folds: 22–67 %, SD 6.3 pp) |
| Leaky split seed 42 identical to the M2 baseline split | ✅ |
| M3 patient splits (3 seeds) | ✅ 30 + 10 test, 15 + 5 val patients (was 151/50; now 150/50 after the merge, a documented deviation) |
| Split files hashed | ✅ `splits/SHA256SUMS` |
| Stain fingerprint recomputed on corrected IDs (foreground pixels) | ✅ patient explains R 0.89, G 0.74, B 0.90, saturation 0.90, hue 0.94 of variance (uninfected cells only: 0.75–0.93) |
| Shared training code + per-experiment configs, smoke-tested | ✅ `train.py`, `configs/` (E1, E1r, E3, E4, E8) |
| Experiment registry | ✅ `registry.csv`. The "prediction" column is filled before a run |
| Time one full DANet run and one ResNet-18 run on a Kaggle T4 | ✅ 2026-10-23 (2× T4): DANet 78 s/epoch → 1.1 h per 50-epoch run; ResNet-18 24 s/epoch → 6 min. Two jobs run in parallel per session, so a DANet run costs ≈ 0.6 quota-h. Image loading cost 19 min/job from the extracted dataset → now read from one packed file (`scripts/pack_images.py`) |
| External-data crop protocol (MP-IDB *P. falciparum*, BBBC041) | ✅ protocol in `protocols/`; MP-IDB extracted (963 Parasitized + 13,505 Uninfected crops). ✅ 100-crop audit done (AI visual review, 2026-10-27): 0 confirmed label errors, 3 uncertain, 10 partial-cell shapes (see protocol doc). BBBC041 (2.3 GB) moved to Phase 1 |
| Decide the epoch cap (50 vs the frozen 75) from the timing run | ✅ 50 for the grouped/leaky grids (best epochs 42, 34; val loss flat or rising after ~40). M3 registered rows keep 75 |

## 4. Running on Kaggle

Jobs are launched from this machine with the Kaggle CLI (token in `~/.kaggle/access_token`); no git push needed.

```bash
PY="arch -arm64 .venv/bin/python"
$PY -m research.kaggle.launch code -m "what changed"          # re-upload code + splits (private dataset)
$PY -m research.kaggle.launch push --slug <job-name> \
    --job E1_danet:grouped:0:0 --job E1r_resnet18:grouped:0:0   # config:split[:repeat:fold][:seed=N][:key=value]
$PY -m research.kaggle.launch status --slug <job-name>
$PY -m research.kaggle.launch fetch  --slug <job-name>          # -> research/runs_kaggle/<job-name>/
```

Each job runs on a T4 (`--accelerator NvidiaTeslaT4`) and writes `runs/<config>/<split>_t<seed>/`
(`metrics.json` has `sec_per_epoch` and `train_hours`; `gpu.txt` has `nvidia-smi`). One push = one Kaggle
session (max 12 h), so batch folds to fit. Check `kaggle quota` before pushing.
Locally on Apple Silicon use `arch -arm64 .venv/bin/python -m research.train ...`; InstanceNorm is very slow on
MPS (≈10× slower than LayerNorm), so do full runs on Kaggle.

## 5. Experiment tracking (how failures are kept)

| Layer | Where | What it holds |
|---|---|---|
| Index | `registry.csv` | One row per experiment: prediction (before running), status, outcome = positive / negative / invalid |
| Experiment log | `experiments/<ID>_<name>/README.md` | Question, dated prediction, protocol, Kaggle job slugs, result, verdict; analysis notebooks for that experiment live in the same folder |
| Config | `configs/<ID>_<name>.json` | The only thing that differs between experiments; all share `train.py` and `malaria/` |
| Raw runs | `runs_kaggle/<job>/runs/<config>/<split>/` | config.json, train_log.csv, test_predictions.csv, metrics.json (or a crash log in `logs/`) |
| Master table | `results/all_runs.csv`, `results/experiment_summary.csv` | Built by `scripts/collect_results.py`: every run, finished / partial / crashed |

Rule: never delete a run or an experiment folder. A negative result gets outcome `negative`; a run broken by a bug
gets `invalid` with the reason, and the corrected rerun is a new job slug.

## 6. Phase 1 results summary (2026-10-30)

Full detail is in `registry.csv` and each `experiments/<ID>_<name>/README.md`; this is the condensed version
for planning the write-up.

**What held up:**
- E0: data layer is sound (200 patients, leakage-proof folds, 1.09pp prevalence SD).
- E7: 312 cells (1.13%) flagged by cleanlab. Reading them as label errors is **not** established: 21 of 114 flagged
  Uninfected cells sit in healthy-only patients, near the base rate. Call them flagged until an expert reviews them.
- E8: LayerNorm profits from patient leakage (+0.63 BACC gap) where InstanceNorm doesn't (+0.02), with equal
  grouped accuracy -- E8a (LN + softmax gate) is invalid, doesn't train.
- E4b: colour transfer narrowed E8b's leakage gap (+0.627 to +0.222) and lowered its patient-ID probe (39.8% to 34.2%).
  Encouraging but not causal evidence yet: one leaky seed, no IN leaky controls. E6c then found these checkpoints do
  better on MP-IDB, BACC 72.6 to 79.6.

**What was falsified or came back very differently than predicted:**
- E1: the M3-predicted leakage drop (-1 to -2.5 BACC) did NOT happen -- actual gap is +0.26, essentially
  nothing. This is good news for DANet's reproducibility but bad news for the M3 hypothesis.
- E4 (M3's own colour-jitter test): **falsified** by its own registered rule.
- E6 and E6b: E6's single-checkpoint ranking (E1 worst) **did not survive E6b's five-checkpoint rescore**. No arm is
  reliably better on MP-IDB, and "gate/dilation drives external robustness" is withdrawn. There is still no evidence
  that InstanceNorm generalises better than LayerNorm. Exploratory: retraining on different NIH patients moves MP-IDB
  BACC far more (fold-mean SD 6.12) than changing architecture does (arm-mean SD 1.91), which looks like underspecification.
- E9: frozen foundation-model probes (DINOv2/DinoBloom/RedDino) land at 96.0-96.5% BACC, far above the
  predicted 75-93%, within 0.3-0.7pp of fine-tuned DANet.
- E5: Grad-CAM lands on annotated parasites on MP-IDB, but a no-model darkest-pixel rule does better (about 98%), so this
  shows dependence on the dark object, not learned parasite morphology. Single checkpoints, and NIH remains open.
- E2's stain-swap behavioural test found no difference between norm types, even though the patient-ID probe
  (representational) did -- a real dissociation between what a model's features *encode* and what a simple
  counterfactual shows it *uses*, not fully resolved.
- E3: inconclusive on capacity. The runs underfit (final train_acc at most 0.60 in 6 of 10) and there is no LN arm.

**The honest framing for the paper:** DANet's headline accuracy is not explained by patient leakage under patient-grouped
splits (E1), though session leakage is untested (E10). DANet matches a generic ResNet-18 (E1r) and sits within a point of
frozen features (E9). The norm layer changes how much in-domain leakage it exploits (E8), on exploratory evidence. No
architecture ranking is shown on MP-IDB (E6b), while retraining on different patients moves results widely, and
training-time colour transfer helps (E6c). A one-feature baseline is competitive on curated crops (exploratory, needs
BBBC041). Report these as separately scoped claims, keeping the falsified ones on record.

## 7. Phase 2: revised priorities (not yet started)

Given what Phase 1 actually found, in priority order:

1. **Revised 2026-11-02 after E6b and E6c.** E6c: training-time colour transfer raises MP-IDB AUC by +0.059 and BACC by
   +7.05, so a colour remedy exists. Test-time Reinhard harmed all arms, even after a sensitivity rerun fixed a protocol flaw.
   Registered next: E10 (leave-session-out split, needs about 3h of Kaggle quota). The gate/dilation ablation that was here lost its top
   priority under E6b's registered rule, because the E6 ranking that motivated it did not survive. New candidates: a
   registered underspecification test with independent retrains, and a five-checkpoint rerun of E5, which also used
   fold-0 checkpoints only.
2. **Resolve E2's representation/behaviour dissociation**: a channel-level or concept-based analysis (CRP/CAV,
   proposed and partly scoped in earlier research but not built) on the existing E1/E8b/E8c checkpoints and
   embeddings -- no new training needed, reuses E2's infrastructure.
3. **NIH parasite pseudo-masks** (SAM-family, 200-cell audit) to finally answer whether E5's strong MP-IDB
   localisation result holds on NIH itself, resolving the explicitly-flagged open question in E5's writeup.
4. **BBBC041 scoring** (external test, P. vivax, different species): deferred from E6 and kept untouched as the final
   holdout, because MP-IDB has now been reused several times.
5. Lower priority, from the earlier research survey (not reprioritised by Phase 1's findings): LP-FT ladder
   on frozen foundation-model features, calibration/conformal referral, diffusion counterfactuals, SAE
   analysis -- revisit after 1-4 above inform what the paper actually needs.

Re-check Kaggle quota before launching (it runs out faster than estimated in this project -- E4b's 15 runs
cost ~18h against an ~9h estimate, and the week's reset is every Saturday).
