"""Metrics and the outcome rule. Run once on BBBC041, after scores/<name>/manifest.json exists and is committed.

    arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.calculate --name <run name> --labels <labels.csv>

`--labels` columns: crop_id, image, y (1 Parasitized, 0 Uninfected, empty for secondary-only rows), group, stage.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from research.malaria import dark_spot

HERE = Path(__file__).resolve().parent
B, SEED = 5000, 20261102
AUC_MARGIN, BACC_MARGIN = .01, 2.0


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, s):
    n1, n0 = y.sum(), (~y).sum()
    return (st.rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rates(y, s, t):
    q = s >= t
    tp, fp = q[y].sum(), q[~y].sum()
    se, sp = 100 * q[y].mean(), 100 * (~q[~y]).mean()
    return dict(BACC=(se + sp) / 2, SE=se, SP=sp, PPV=100 * tp / max(tp + fp, 1), called=int(tp + fp))


def outcome(auc1, auc3, d_auc, d_bacc):
    """auc1, auc3: candidate point AUCs. d_auc, d_bacc: (lower, upper) of candidate 3 minus candidate 1."""
    if auc1 < .80 and auc3 < .80:
        return "No transfer"
    feature = dict(AUC=d_auc[1] <= AUC_MARGIN, BACC=d_bacc[1] <= BACC_MARGIN)
    trained = dict(AUC=d_auc[0] > AUC_MARGIN, BACC=d_bacc[0] > BACC_MARGIN)
    if feature["AUC"] and feature["BACC"]:
        return "Feature holds"
    if trained["AUC"] and trained["BACC"]:
        return "Trained model wins"
    if (feature["AUC"] and trained["BACC"]) or (trained["AUC"] and feature["BACC"]):
        return "Ranking and threshold disagree"
    return "Inconclusive"


def load(name):
    out = HERE / "scores" / name
    man = json.load(open(out / "manifest.json"))
    s = {f.stem: np.load(f) for f in out.glob("*.npy")}
    assert all(len(v) == man["n"] for v in s.values())
    return s, man


def candidates(s):
    mean_logit = lambda pre: np.mean([logit(v) for k, v in s.items() if k.startswith(pre)], axis=0)
    c = {"1_feature": (np.nan_to_num(s["feature"]), dark_spot.THRESHOLD), "2_E1_ensemble": (mean_logit("cand2_"), 0.0),
         "3_colour_ensemble": (mean_logit("cand3_"), 0.0)}
    for k in s:
        if k.startswith("probe_"):
            c[f"4_{k[6:]}"] = (logit(s[k]), 0.0)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    s, man = load(a.name)
    lab = pd.read_csv(a.labels)
    assert len(lab) == man["n"]
    assert hashlib.sha256(",".join(lab.crop_id).encode()).hexdigest() == man["crop_ids_sha256"], "labels not in crop order"
    c = candidates(s)
    n3 = sum(k.startswith("cand3_") for k in s)
    prim = lab.y.notna().values
    y = lab.y.values[prim] == 1
    img = lab.image.values[prim]
    S = {k: (v[0][prim], v[1]) for k, v in c.items()}
    O = dict(name=a.name, n_primary=int(prim.sum()), parasitized=int(y.sum()), images=len(set(img)), n_cand3_members=n3,
             prevalence=100 * float(y.mean()),
             candidates={k: dict(AUC=auc(y, v[0]), **rates(y, v[0], v[1])) for k, v in S.items()})

    rng = np.random.default_rng(SEED)
    groups = [np.where(img == k)[0] for k in pd.unique(img)]
    keys = ["1_feature", "3_colour_ensemble", "2_E1_ensemble"]
    d = {k: [] for k in ("AUC_3m1", "BACC_3m1", "AUC_2m1", "BACC_2m1", "AUC_3m2")}
    pts = {k: S[k] for k in keys}
    for _ in range(B):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        yb = y[i]
        A = {k: auc(yb, pts[k][0][i]) for k in keys}
        R = {k: rates(yb, pts[k][0][i], pts[k][1])["BACC"] for k in keys}
        d["AUC_3m1"].append(A[keys[1]] - A[keys[0]])
        d["BACC_3m1"].append(R[keys[1]] - R[keys[0]])
        d["AUC_2m1"].append(A[keys[2]] - A[keys[0]])
        d["BACC_2m1"].append(R[keys[2]] - R[keys[0]])
        d["AUC_3m2"].append(A[keys[1]] - A[keys[2]])
    point = dict(AUC_3m1=O["candidates"][keys[1]]["AUC"] - O["candidates"][keys[0]]["AUC"],
                 BACC_3m1=O["candidates"][keys[1]]["BACC"] - O["candidates"][keys[0]]["BACC"],
                 AUC_2m1=O["candidates"][keys[2]]["AUC"] - O["candidates"][keys[0]]["AUC"],
                 BACC_2m1=O["candidates"][keys[2]]["BACC"] - O["candidates"][keys[0]]["BACC"],
                 AUC_3m2=O["candidates"][keys[1]]["AUC"] - O["candidates"][keys[2]]["AUC"])
    O["deltas"] = {k: dict(point=point[k], ci_975=np.percentile(v, [1.25, 98.75]).tolist()) for k, v in d.items()}
    O["outcome"] = outcome(O["candidates"][keys[0]]["AUC"], O["candidates"][keys[1]]["AUC"],
                           O["deltas"]["AUC_3m1"]["ci_975"], O["deltas"]["BACC_3m1"]["ci_975"])

    O["by_group"] = {}
    if "group" in lab:
        for g, sub in lab[prim].groupby("group"):
            j = lab.index[prim].get_indexer(sub.index)
            O["by_group"][str(g)] = {k: dict(AUC=auc(y[j], v[0][j]), **rates(y[j], v[0][j], v[1])) for k, v in S.items()}
    if "stage" in lab:
        O["sensitivity_by_stage"] = {}
        for st_, sub in lab[prim & (lab.y == 1).values].groupby("stage"):
            pos = sub.index.values
            O["sensitivity_by_stage"][st_] = {k: 100 * float((v[0][pos] >= v[1]).mean()) for k, v in c.items()} | dict(n=len(pos))
        hard = lab.stage.eq("leukocyte").values
        if hard.any():
            O["leukocyte_flag_pct"] = {k: 100 * float((v[0][hard] >= v[1]).mean()) for k, v in c.items()} | dict(n=int(hard.sum()))
    json.dump(O, open(HERE / f"results_{a.name}.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    main()
