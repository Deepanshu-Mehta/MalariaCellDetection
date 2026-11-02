"""E3: random-patient-label permutation null. Does DANet have the capacity to memorise patients when the
label carries zero parasite information?

    arch -arm64 .venv/bin/python research/experiments/E3_random_patient_label_permutation_null/analysis.py

5 independent label draws (one random label per patient, cell-balanced), each trained on the leaky cell
split (seed 42) and on one patient-grouped split (repeat 0, fold 0), 30 epochs. BACC is computed against
each draw's OWN random labels (the only labels that run ever saw), so chance is always 50%.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "runs_kaggle" / "e3-random-patient-label-v2" / "runs" / "E3_random_patient"


def bacc(y, p, thr=0.5):
    pred = p >= thr
    se = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    sp = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    return 100 * (se + sp) / 2


def main():
    rows = []
    for d in sorted(RUN.glob("*")):
        cfg = json.load(open(d / "config.json"))
        pred = pd.read_csv(d / "test_predictions.csv")
        rows.append(dict(split=cfg["split"], draw=cfg["cfg"]["label_draw"],
                         bacc=bacc(pred.label.values, pred.p.values), n=len(pred)))
    t = pd.DataFrame(rows)
    res = {}
    for split, g in t.groupby("split"):
        res[split] = dict(per_draw=g.set_index("draw").bacc.round(2).to_dict(),
                          mean=round(g.bacc.mean(), 2), sd=round(g.bacc.std(), 2))
    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    print(t.to_string(index=False))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
