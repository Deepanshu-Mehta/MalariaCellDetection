"""E9 follow-up: does any frozen foundation-model probe match DANet's fine-tuned grouped accuracy?

    arch -arm64 .venv/bin/python research/experiments/E9_foundation_model_probes/compare_to_danet.py

Recomputes the full-label-fraction (100%) logistic-regression probe from the cached features (downloaded
from the Kaggle run, no extra GPU time), saves per-cell OOF predictions, and runs the same paired
patient-cluster bootstrap used throughout (E1/E8/E2) against DANet's own grouped-repeat-0 OOF predictions on
the identical cells. Point estimates alone are not enough per this project's own standard.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                      # research/
sys.path.insert(0, str(ROOT.parent))
from research.malaria.stats import load_preds, paired_diff  # noqa: E402

RUN = ROOT / "runs_kaggle" / "e9-foundation-probes" / "E9_outputs"


def oof_probe(feats, cells, folds):
    oof = np.full(len(cells), np.nan)
    for fold in range(5):
        role = folds[folds.fold == fold].set_index("patient").role
        mask = cells.patient.map(role)
        tr = mask.isin(["train", "val"]).values
        te = mask.eq("test").values
        sc = StandardScaler().fit(feats[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(feats[tr]), cells.label.values[tr])
        oof[te] = lr.predict_proba(sc.transform(feats[te]))[:, 1]
    assert not np.isnan(oof).any()
    return oof


def main():
    cells = pd.read_csv(ROOT / "splits" / "cells.csv", index_col="idx")
    folds = pd.read_csv(ROOT / "splits" / "grouped_folds.csv")
    folds = folds[folds.repeat == 0]

    danet = load_preds("E1_danet", "grouped")
    danet = danet[danet.tag.str.startswith("grouped_r0")].sort_values("idx").reset_index(drop=True)
    assert len(danet) == len(cells)

    res = {}
    for name in ["dinov2_s", "dinobloom_s", "reddino_s"]:
        feats = np.load(RUN / "features" / f"{name}.npy")
        p = oof_probe(feats, cells, folds)
        probe_df = pd.DataFrame(dict(idx=cells.index.values, patient=cells.patient.values,
                                     label=cells.label.values, p=p))
        probe_df.to_csv(HERE / f"oof_predictions_{name}.csv", index=False)
        # paired_diff(a, b, ...) computes metrics(a) - metrics(b); pass danet first so diff reads "DANet - probe"
        diff = paired_diff(danet[["idx", "patient", "label", "p"]], probe_df, "BACC")
        res[name] = dict(probe_BACC=float(((p >= 0.5) == cells.label.values).mean() * 100),  # sanity only
                         danet_minus_probe_BACC=diff)
        print(f"{name}: DANet - probe BACC = {diff['diff']:+.3f}  CI=[{diff['lo']:+.3f}, {diff['hi']:+.3f}]"
             f"  {'(no detectable difference)' if diff['lo'] < 0 < diff['hi'] else '(DANet significantly higher)'}")
    json.dump(res, open(HERE / "results_vs_danet.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
