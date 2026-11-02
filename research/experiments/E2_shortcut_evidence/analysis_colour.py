"""E2 part A/B: how much does colour alone predict the label or the patient, under grouped CV?

Reuses the per-cell foreground colour features from Phase 0 (research/results/phase0/cell_colour_features.csv:
mean R, G, B, foreground HSV saturation S, circular-mean hue H). No parasite mask exists for NIH cells, so for
Parasitized cells these means include the parasite's own chromatin colour -- that is disclosed, not hidden.

    arch -arm64 .venv/bin/python research/experiments/E2_shortcut_evidence/analysis_colour.py

A. Colour-only label classifier (logistic regression, 5 colour features) on the same patient-grouped folds as
   E1 repeat 0. Compares against E1's own BACC on identical folds.
B. Healthy-only vs infected-patient probe, UNINFECTED CELLS ONLY (no parasite colour confound by construction):
   can colour tell a healthy-only patient's red cell from an infected patient's red cell?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FEATS = ["R", "G", "B", "S", "H"]


def metrics(y, p):
    pred = p >= 0.5
    se = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    sp = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    return dict(BACC=100 * (se + sp) / 2, SE=100 * se, SP=100 * sp)


def main():
    cells = pd.read_csv(ROOT / "research" / "splits" / "cells.csv", index_col="idx")
    feats = pd.read_csv(ROOT / "research" / "results" / "phase0" / "cell_colour_features.csv", index_col="idx")
    folds = pd.read_csv(ROOT / "research" / "splits" / "grouped_folds.csv")
    folds = folds[folds.repeat == 0]
    d = cells.join(feats)
    res = {}

    # A. colour-only label classifier, pooled OOF over the 5 folds (identical folds to E1)
    oof = np.full(len(d), np.nan)
    for fold in range(5):
        role = folds[folds.fold == fold].set_index("patient").role
        tr_mask = d.patient.map(role).eq("train").fillna(False) | d.patient.map(role).eq("val").fillna(False)
        te_mask = d.patient.map(role).eq("test").fillna(False)
        sc = StandardScaler().fit(d.loc[tr_mask, FEATS])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(d.loc[tr_mask, FEATS]), d.loc[tr_mask, "label"])
        oof[te_mask.values] = clf.predict_proba(sc.transform(d.loc[te_mask, FEATS]))[:, 1]
    assert not np.isnan(oof).any()
    res["A_colour_only_label_classifier"] = dict(
        all_cells=metrics(d.label.values, oof),
        note="RGB/S/H means include parasite chromatin colour for Parasitized cells (no NIH parasite mask).")

    # B. healthy-only vs infected-patient probe, uninfected cells only
    u = d[d.label == 0].copy()
    para_count = d.groupby("patient").label.sum()
    u["is_healthy_only_patient"] = (u.patient.map(para_count) == 0).astype(int)
    oof_b = np.full(len(u), np.nan)
    for fold in range(5):
        role = folds[folds.fold == fold].set_index("patient").role
        tr_mask = u.patient.map(role).isin(["train", "val"]).values
        te_mask = u.patient.map(role).eq("test").values
        sc = StandardScaler().fit(u.loc[tr_mask, FEATS])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(u.loc[tr_mask, FEATS]), u.loc[tr_mask, "is_healthy_only_patient"])
        oof_b[te_mask] = clf.predict_proba(sc.transform(u.loc[te_mask, FEATS]))[:, 1]
    assert not np.isnan(oof_b).any()
    y_b = u.is_healthy_only_patient.values
    res["B_healthy_only_vs_infected_patient_colour_probe"] = dict(
        n_uninfected_cells=len(u), base_rate_healthy_only=float(y_b.mean()),
        **metrics(y_b, oof_b),
        note="Uninfected cells only, so no parasite-colour confound. BACC > 50 means a red cell's colour alone "
             "can place its patient in the healthy-only vs infected-patient group.")

    json.dump(res, open(HERE / "results_colour.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
