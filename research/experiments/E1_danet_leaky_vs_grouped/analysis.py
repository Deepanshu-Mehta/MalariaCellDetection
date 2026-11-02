"""E1 analysis: DANet under the paper's leaky cell split vs patient-grouped 5-fold CV.

    arch -arm64 .venv/bin/python research/experiments/E1_danet_leaky_vs_grouped/analysis.py

grouped: each repeat's 5 test folds are pooled into out-of-fold (OOF) predictions covering all 27,558 cells
         once; metrics per repeat, then mean +/- sd over the 3 repeats.
leaky:   3 random 70/10/20 cell splits (seed 42 = M2 split), 5,512 test cells each.
CIs: patient-cluster bootstrap (resample patients with replacement, B = 5000).
MDE: A/A test = paired patient bootstrap of the BACC difference between two grouped repeats on the same cells;
     MDE = 2.8 x SE (80 % power, alpha 0.05). Written to results.json for E4.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs_kaggle"
B = 5000


def load_preds(split):
    out = []
    for f in RUNS.glob(f"*/runs/E1_danet/{split}_*/test_predictions.csv"):
        if "smoke" in str(f):
            continue
        d = pd.read_csv(f)
        d["tag"] = f.parent.name
        out.append(d)
    return pd.concat(out, ignore_index=True)


def metrics(y, p):
    pred = p >= 0.5
    se = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    sp = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return dict(BACC=100 * (se + sp) / 2, SE=100 * se, SP=100 * sp, AUC=auc)


def by_patient(d):
    return {p: g for p, g in d.groupby("patient")}


def bootstrap(d, rng, stat=lambda g: metrics(g.label.values, g.p.values)["BACC"]):
    groups = by_patient(d)
    keys = np.array(list(groups))
    vals = []
    for _ in range(B):
        g = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        vals.append(stat(g))
    return np.percentile(vals, [2.5, 97.5])


def patient_specificity(d):
    t = d.assign(pred=d.p >= 0.5).groupby("patient").agg(para=("label", "sum"))
    healthy = set(t.index[t.para == 0])
    u = d[d.label == 0]
    sp = lambda x: 100 * (x.p < 0.5).mean()
    return dict(SP_healthy_only=sp(u[u.patient.isin(healthy)]), SP_infected_patients=sp(u[~u.patient.isin(healthy)]))


def main():
    rng = np.random.default_rng(0)
    res = {}
    g = load_preds("grouped")
    g["repeat"] = g.tag.str.extract(r"grouped_r(\d)").astype(int)
    per_rep = {}
    for r, d in g.groupby("repeat"):
        assert d.idx.is_unique and len(d) == 27558, f"repeat {r} OOF incomplete ({len(d)} cells)"
        per_rep[r] = d.sort_values("idx").reset_index(drop=True)
    rep_m = pd.DataFrame({r: {**metrics(d.label.values, d.p.values), **patient_specificity(d)}
                          for r, d in per_rep.items()}).T
    res["grouped_per_repeat"] = rep_m.round(3).to_dict(orient="index")
    res["grouped_mean"] = rep_m.mean().round(3).to_dict()
    res["grouped_sd"] = rep_m.std().round(3).to_dict()
    res["grouped_bacc_ci_repeat0"] = bootstrap(per_rep[0], rng).round(2).tolist()

    lk = load_preds("leaky")
    lk_m = pd.DataFrame({t: {**metrics(d.label.values, d.p.values), **patient_specificity(d)}
                         for t, d in lk.groupby("tag")}).T
    res["leaky_per_seed"] = lk_m.round(3).to_dict(orient="index")
    res["leaky_mean"] = lk_m.mean().round(3).to_dict()
    res["leaky_sd"] = lk_m.std().round(3).to_dict()
    res["leaky_bacc_ci_seed42"] = bootstrap(lk[lk.tag.str.contains("s42")], rng).round(2).tolist()
    res["gap_leaky_minus_grouped"] = (lk_m.mean() - rep_m.mean()).round(3).to_dict()

    # A/A test: same cells, two grouped repeats, paired patient bootstrap of the BACC difference
    a, b = per_rep[0], per_rep[1]
    assert (a.idx.values == b.idx.values).all()
    pair = a[["patient", "label"]].assign(pa=a.p.values, pb=b.p.values)
    groups = by_patient(pair)
    keys = np.array(list(groups))
    diffs = []
    for _ in range(B):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        diffs.append(metrics(s.label.values, s.pa.values)["BACC"] - metrics(s.label.values, s.pb.values)["BACC"])
    se = float(np.std(diffs))
    res["aa_test"] = dict(observed_diff=round(metrics(pair.label.values, pair.pa.values)["BACC"]
                                              - metrics(pair.label.values, pair.pb.values)["BACC"], 3),
                          se=round(se, 3), mde_bacc_pp=round(2.8 * se, 2))

    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
