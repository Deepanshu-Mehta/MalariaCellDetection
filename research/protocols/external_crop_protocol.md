# External test sets: cell-crop protocol (draft, Phase 0)

Purpose: turn external smear images into NIH-style single-cell crops so that an NIH-trained model can be scored
on them **once**, with no tuning on the external data. Fix this protocol before any model is evaluated on it.

## Target format (matches NIH)

- One red blood cell per crop, tight bounding box around the cell with a 4 px margin.
- Pixels outside the cell mask set to pure black (NIH crops are black-padded).
- Native resolution, saved as PNG; the model's own eval transform (resize to 224) is applied at test time.
- Binary label: `Parasitized` = cell contains any parasite stage; `Uninfected` = red cell with no parasite.
- Leukocytes, platelets, debris and clumps are **excluded**, not labelled.

## MP-IDB *P. falciparum* (primary external set)

Source: [MP-IDB](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis)
(MIT licence), `Falciparum/`: 104 images (2592×1944, Giemsa thin smears) with expert pixel masks (`gt/`) and
a box list (`mp-idb-falciparum.csv`: x, y = box centre, then width, height) for 1,297 parasites
(1,230 ring, 42 trophozoite, 18 schizont, 7 gametocyte). Red cells are ≈110–130 px across, close to NIH crops.

MP-IDB annotates **parasites only**, so red cells must be segmented:

1. **Cell segmentation.** Cellpose `cyto3` (Cellpose < 4), diameter 120 px, default thresholds, on the RGB image.
2. **Keep single, whole red cells.** Drop masks touching the image border; drop masks whose area is outside
   0.5–1.8 × the image's median cell area (clumps, fragments).
3. **Parasite mask.** `gt/` is JPEG-compressed: binarise at > 127.
4. **Label** (in this order).
   - Parasitized: ≥ 50 % of a connected parasite component lies inside the cell mask.
   - Excluded (ambiguous): any other parasite-mask overlap, or within 10 px of a parasite.
   - Excluded (leukocyte-like): > 25 % of the cell darker than 0.5 × the image's median cell brightness
     (min RGB channel). Applied only to cells without parasite overlap. *Revision 2026-10-23: a fixed cut-off
     of 100 excluded 4,356 ordinary red cells in darker images; the relative rule excludes 20 (≤ 2 per image).*
   - Uninfected: everything else.
5. **Crop**, black out the background, save `{Parasitized|Uninfected}/{image}_{cell}.png`, and log every kept and
   excluded cell with its reason in `mpidb_cells.csv`.
6. **Audit.** Check 100 random crops (50 per class) by eye and record the error rate. Report it.
   Files: `mpidb_pf_cells/audit_sheet.png` (labels hidden) and `audit_sample.csv`/`audit_ai_prescreen.csv`.
   **Done 2026-10-27 (AI visual review of all 100 crops):** 0 confirmed label errors (0/50 Parasitized crops
   with no visible parasite; 0/50 Uninfected crops with a visible parasite); 3 crops judged ambiguous (only
   faint colour at the crop edge, parasite may belong to a neighbouring cell -- ids 12, 19, 52); 10 crops are
   partial-cell fragments rather than whole cells, from overlapping-cell segmentation (ids 13, 15, 21, 27, 29,
   66, 70, 75, 76, 96) -- their labels still look correct. **Estimated pipeline label error: 0-6% for
   Parasitized, ~0% for Uninfected.** A convexity filter was tried to drop the fragments automatically; it
   caught only 6/10 and wrongly dropped 3 good crops, so it was not adopted -- report the 10% fragment rate
   as a known limitation instead.

**Result (2026-10-23, `scripts/extract_mpidb.py`):** 104 images → 963 Parasitized + 13,505 Uninfected crops;
1,166 of 1,297 annotated parasites (90 %) fall in a kept cell; 87 kept cells hold > 1 parasite. Excluded:
3,013 touching the border, 510 size outliers, 387 ambiguous overlaps, 20 leukocyte-like. Segmentation masks are
cached in `mpidb_pf_cells/_masks/`, so rule changes re-run in seconds.

Known biases to report: clumped cells (where crowded infected cells may sit) are excluded; MP-IDB is mostly
rings; uninfected cells vastly outnumber infected ones, so report balanced accuracy, SE, SP and AUC, never
plain accuracy. **RedDino was pre-trained on MP-IDB**, so any RedDino result on MP-IDB is flagged as contaminated.

## BBBC041 *P. vivax* (secondary external set)

Source: [BBBC041](https://bbbc.broadinstitute.org/BBBC041) (CC BY-NC-SA 3.0, 2.26 GB), 1,364 Giemsa images,
≈80k cells with boxes and classes: red blood cell, leukocyte, ring, trophozoite, schizont, gametocyte, difficult.

1. **Label mapping.** ring / trophozoite / schizont / gametocyte → Parasitized; red blood cell → Uninfected;
   leukocyte and difficult → excluded.
2. **Crop** each box with a 4 px margin. Build the cell mask inside the box (Otsu threshold on the darkness /
   saturation channel, largest connected component, holes filled) and black out the rest.
3. **Subsample uninfected cells** to at most 5 per parasitized cell per image (seeded), because >95 % of cells
   are uninfected; report the subsampling.
4. Same 100-crop audit as MP-IDB.

Known differences: different species (*P. vivax*), enlarged infected cells, different labs and cameras.
This is a harder, species-shifted test; do not compare its numbers directly with MP-IDB.

## Evaluation rules

- Models are trained on NIH only; no fine-tuning, threshold tuning or normalisation fitted on external data
  (few-shot adaptation is a separate Phase 2 arm with its own split).
- Each frozen model is scored once per external set; report the drop against its own NIH grouped-CV level, and
  treat a drop smaller than the seed-to-seed spread as "no detectable change" (MIRASOL 2026 protocol).
- Report BACC, SE, SP, ROC-AUC with patient/image-level bootstrap CIs (resample images, not cells).
