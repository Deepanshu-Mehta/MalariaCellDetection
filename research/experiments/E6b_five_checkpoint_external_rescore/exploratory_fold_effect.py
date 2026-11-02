"""Exploratory, added after seeing E6b's results: NIH vs MP-IDB stability, and shared fold effects."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import fast_metrics
from .score_external import ARMS, N_FOLDS, ckpt_path

HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "results.json"))["summary"]
out = {"NIH": {}, "MPIDB": {}}
for a, c in ARMS.items():
    nih = []
    for f in range(N_FOLDS):
        d = pd.read_csv(ckpt_path(c, f).parent / "test_predictions.csv")
        nih.append(fast_metrics(d.label.values, d.p.values))
    out["NIH"][a] = {m: dict(per_fold=[round(float(x[m]), 3) for x in nih], sd=float(np.std([x[m] for x in nih], ddof=1)))
                     for m in ("BACC", "SE", "AUC")}
    out["MPIDB"][a] = {m: dict(sd=r[a][m]["sd"], range=[min(r[a][m]["per_fold"]), max(r[a][m]["per_fold"])])
                       for m in ("BACC", "SE", "AUC")}
for m in ("BACC", "AUC"):
    t = np.array([r[a][m]["per_fold"] for a in ARMS])
    out[f"{m}_sd_of_fold_means"] = float(t.mean(0).std(ddof=1))
    out[f"{m}_sd_of_arm_means"] = float(t.mean(1).std(ddof=1))
    out[f"{m}_between_arm_fold_corr"] = np.round(np.corrcoef(t), 2).tolist()
json.dump(out, open(HERE / "exploratory_fold_effect.json", "w"), indent=1)
print(json.dumps(out, indent=0)[:2600])
