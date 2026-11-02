"""E2 part D: stain-swap counterfactual flip rate.

    arch -arm64 .venv/bin/python research/experiments/E2_shortcut_evidence/analysis_stainswap.py

For each arm's OOF checkpoints (repeat 0, same 5 folds as E1/E8b/E8c), recolour each test cell to another
patient's stain statistics (Reinhard transfer in LAB space, computed on FOREGROUND pixels only, background
re-zeroed afterwards) and re-score it with the SAME frozen model. The flip rate is how often the predicted
class changes.

Two swap conditions per cell, both targeting a patient of the SAME broad type as the source (same-type
control) and of the OPPOSITE type (the shortcut-relevant swap):
  - uninfected cell, healthy-only patient   -> target: another healthy-only patient (control)
                                             -> target: an infected patient (cross-type)
  - uninfected cell, infected patient       -> target: another infected patient (control)
                                             -> target: a healthy-only patient (cross-type)
  - parasitized cell (always an infected patient) -> target: another infected patient (control only;
    there is no healthy-only equivalent since healthy-only patients contribute no parasitized cells)

Reported separately per arm and per condition, since E8's hypothesis is that LayerNorm (E8b) flips more than
InstanceNorm (E1, E8c), especially for uninfected cells and especially on the cross-type swap.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.malaria import data, models  # noqa: E402
from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
N_PER_PATIENT = 20          # cells sampled per patient per condition (cost control)
SEED = 0


def reinhard_transfer(img, src_stats, tgt_stats, fg):
    """img: HxWx3 uint8. src_stats/tgt_stats: (mean, std) of LAB on foreground pixels. fg: HxW bool."""
    import cv2
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    sm, ss = src_stats
    tm, ts = tgt_stats
    out = (lab - sm) / np.maximum(ss, 1e-3) * ts + tm
    out = np.clip(out, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    rgb[~fg] = 0
    return rgb


def lab_stats(img, fg):
    import cv2
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[fg]
    return px.mean(0), px.std(0)


def to_tensor(img):
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return torch.nn.functional.interpolate(t[None], size=(224, 224), mode="bilinear",
                                           align_corners=False, antialias=True)[0]


def run_arm(arm, cfg, cells, folds, root, rng):
    para_count = cells.groupby("patient").label.sum()
    healthy = set(para_count.index[para_count == 0])
    infected = set(para_count.index) - healthy

    rows = []
    for fold in range(5):
        test_patients = set(folds[folds.fold == fold].patient)
        sub = cells[cells.patient.isin(test_patients)]
        model = models.build_model(cfg["model"]).to(DEVICE).eval()
        model.load_state_dict(torch.load(ckpt_path(cfg, fold), map_location=DEVICE))
        for p in model.parameters():
            p.requires_grad_(False)

        sample_idx = np.concatenate([g.sample(min(len(g), N_PER_PATIENT), random_state=0).index.values
                                     for _, g in sub.groupby(["patient", "label"])])
        sample = sub.loc[sample_idx]
        imgs, fgs, stats = {}, {}, {}
        for i in sample.index:
            im = data.read_rgb(root / cells.rel_path[i])
            fg = data.foreground_mask(im)
            imgs[i], fgs[i], stats[i] = im, fg, lab_stats(im, fg)

        test_pats = list(test_patients)
        for i, r in sample.iterrows():
            same_pool = [p for p in (healthy if r.patient in healthy else infected)
                        if p in test_pats and p != r.patient]
            cross_pool = [p for p in (infected if r.patient in healthy else healthy) if p in test_pats]
            targets = {"control": rng.choice(same_pool) if same_pool else None,
                      "cross_type": rng.choice(cross_pool) if (r.label == 0 and cross_pool) else None}
            with torch.no_grad():
                orig_p = float(torch.exp(model(to_tensor(imgs[i])[None].to(DEVICE)))[0, 1])
            for cond, tgt_patient in targets.items():
                if tgt_patient is None:
                    continue
                tgt_cell = sub[sub.patient == tgt_patient].sample(1, random_state=rng.integers(1e6)).iloc[0].name
                tgt_stats = stats.get(tgt_cell) or lab_stats(data.read_rgb(root / cells.rel_path[tgt_cell]),
                                                             data.foreground_mask(data.read_rgb(root / cells.rel_path[tgt_cell])))
                swapped = reinhard_transfer(imgs[i], stats[i], tgt_stats, fgs[i])
                with torch.no_grad():
                    new_p = float(torch.exp(model(to_tensor(swapped)[None].to(DEVICE)))[0, 1])
                rows.append(dict(arm=arm, fold=fold, idx=i, patient=r.patient,
                                 patient_type="healthy" if r.patient in healthy else "infected",
                                 label=int(r.label), condition=cond, orig_p=orig_p, new_p=new_p,
                                 flipped=int((orig_p >= 0.5) != (new_p >= 0.5))))
        print(f"{arm} fold {fold}: {len(sample)} source cells", flush=True)
    return pd.DataFrame(rows)


def main():
    cells = pd.read_csv(ROOT / "research" / "splits" / "cells.csv", index_col="idx")
    folds = pd.read_csv(ROOT / "research" / "splits" / "grouped_folds.csv")
    folds = folds[(folds.repeat == 0) & (folds.role == "test")]
    root = data.image_root()
    out = []
    for arm, cfg in ARMS.items():
        f = HERE / f"stainswap_{arm}.csv"
        if f.exists():
            out.append(pd.read_csv(f)); print(f"{arm}: already computed"); continue
        d = run_arm(arm, cfg, cells, folds, root, np.random.default_rng(SEED))
        d.to_csv(f, index=False)
        out.append(d)
    all_d = pd.concat(out, ignore_index=True)
    summary = all_d.groupby(["arm", "patient_type", "label", "condition"]).flipped.agg(["mean", "size"]).round(3)
    summary.to_csv(HERE / "results_stainswap_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
