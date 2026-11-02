"""Phase 0: data integrity, frozen splits and the corrected stain-fingerprint numbers.

Writes research/splits/{cells.csv, grouped_folds.csv, leaky_splits.csv, m3_splits.csv, SHA256SUMS}
and research/results/phase0/phase0_report.json. Run from the repo root:
    arch -arm64 .venv/bin/python -m research.scripts.phase0_build
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.malaria import data, splits

ROOT = Path(__file__).resolve().parents[1]
SPLITS, RES = ROOT / "splits", ROOT / "results" / "phase0"


def eta_squared(values, groups):
    """Share of variance explained by group membership (one-way ANOVA R^2)."""
    v = pd.Series(values)
    g = pd.Series(groups)
    grand = v.mean()
    ss_tot = ((v - grand) ** 2).sum()
    means = v.groupby(g.values).transform("mean")
    return float(((means - grand) ** 2).sum() / ss_tot)


def colour_features(imgs):
    """Foreground-only mean R, G, B and mean HSV saturation / hue (circular mean) per cell."""
    rgb, sat, hue = [], [], []
    for im in imgs:
        fg = data.foreground_mask(im)
        px = im[fg].astype(np.float32) / 255.0
        rgb.append(px.mean(0))
        mx, mn = px.max(1), px.min(1)
        sat.append(np.mean(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)))
        r, g, b = px.T
        h = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)                  # hue angle
        hue.append(np.angle(np.exp(1j * h).mean()))
    rgb = np.array(rgb)
    return pd.DataFrame({"R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2], "S": sat, "H": hue})


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. index + patient IDs
    df = data.build_index()
    pt = data.patient_table(df)
    report["n_cells"] = len(df)
    report["n_nih_ids"] = int(df.nih_id.nunique())
    report["n_patients_merged"] = int(len(pt))
    report["n_infected"] = int((pt.ptype == "infected").sum())
    report["n_healthy_only"] = int((pt.ptype == "healthy").sum())
    report["merged_ids"] = {p: sorted(g.nih_id.unique()) for p, g in df.groupby("patient") if g.nih_id.nunique() > 1}
    healthy_uninf = int(pt[pt.ptype == "healthy"].n_uninf.sum())
    report["healthy_only_share_of_uninfected"] = healthy_uninf / int((df.label == 0).sum())
    report["largest_patient"] = {"patient": pt.n_cells.idxmax(), "cells": int(pt.n_cells.max()),
                                 "share": float(pt.n_cells.max() / len(df))}
    inf = pt[pt.ptype == "infected"]
    report["infected_para_cells"] = {"min": int(inf.n_para.min()), "median": float(inf.n_para.median()),
                                     "max": int(inf.n_para.max())}
    assert report["n_patients_merged"] == 200 and report["n_infected"] == 150 and report["n_healthy_only"] == 50

    # 2. frozen splits
    cells = df[["rel_path", "cls", "label", "nih_id", "patient"]]
    cells.to_csv(SPLITS / "cells.csv", index_label="idx")
    folds = splits.grouped_folds(df)
    splits.check_grouped(folds, df)
    folds.to_csv(SPLITS / "grouped_folds.csv", index=False)
    leaky = splits.leaky_splits(df)
    leaky.to_csv(SPLITS / "leaky_splits.csv", index_label="idx")
    m3 = splits.m3_patient_splits(df)
    m3.to_csv(SPLITS / "m3_splits.csv", index=False)

    # M2 reproduction check: leaky seed 42 must equal the split the baseline was trained on
    m2 = pd.read_csv(ROOT.parent / "milestone2-baseline" / "outputs" / "splits.csv", index_col="idx")
    assert (m2.rel_path.values == df.rel_path.values).all()
    report["leaky_seed42_matches_M2"] = bool((m2.split.values == leaky.seed42.values).all())

    # 3. fold balance: test-fold prevalence of parasitized cells
    para = df.groupby("patient").label.sum()
    ncell = df.groupby("patient").size()
    test = folds[folds.role == "test"]
    prev = test.groupby(["repeat", "fold"]).patient.apply(lambda p: para[p].sum() / ncell[p].sum())
    report["grouped_test_prevalence"] = {"min": float(prev.min()), "max": float(prev.max()),
                                         "sd_pp": float(100 * prev.std())}
    rng = np.random.default_rng(0)
    inf_p, hea_p = pt[pt.ptype == "infected"].index.values, pt[pt.ptype == "healthy"].index.values
    sims = []
    for _ in range(2000):
        p = np.r_[rng.choice(inf_p, 30, replace=False), rng.choice(hea_p, 10, replace=False)]
        sims.append(para[p].sum() / ncell[p].sum())
    sims = np.array(sims)
    report["random_30plus10_prevalence"] = {"min": float(sims.min()), "max": float(sims.max()),
                                            "sd_pp": float(100 * sims.std())}

    # 4. stain fingerprint on corrected IDs (foreground pixels only)
    t0 = time.time()
    feats = colour_features(data.load_images(df))
    feats.insert(0, "idx", df.index)
    feats.to_csv(RES / "cell_colour_features.csv", index=False)
    report["colour_seconds"] = round(time.time() - t0, 1)
    fp = {}
    for subset, mask in [("all_cells", np.ones(len(df), bool)), ("uninfected_only", df.label.values == 0)]:
        fp[subset] = {c: round(eta_squared(feats.loc[mask, c].values, df.patient.values[mask]), 3)
                      for c in ["R", "G", "B", "S", "H"]}
    fp["all_cells_by_nih_id"] = {c: round(eta_squared(feats[c].values, df.nih_id.values), 3)
                                 for c in ["R", "G", "B", "S", "H"]}
    report["colour_variance_explained_by_patient"] = fp

    json.dump(report, open(RES / "phase0_report.json", "w"), indent=1, default=str)
    with open(SPLITS / "SHA256SUMS", "w") as f:
        for name in ["cells.csv", "grouped_folds.csv", "leaky_splits.csv", "m3_splits.csv"]:
            f.write(f"{splits.sha256(SPLITS / name)}  {name}\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
