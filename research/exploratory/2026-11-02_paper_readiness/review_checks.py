"""Checks for the second review. MP-IDB used as a development set only; BBBC041 untouched."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "research"
SCR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
O = {}
rng = np.random.default_rng(0)


def auc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return (st.rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bacc(y, s, t):
    y = np.asarray(y).astype(bool)
    q = np.asarray(s) >= t
    return 50 * (q[y].mean() + (~q[~y]).mean())


c = pd.read_pickle(SCR / "nih_f.pkl")
m = pd.read_pickle(SCR / "mp_f.pkl")

# 1. Exact NIH threshold, no grid
f, y = c.f.values, c.label.values == 1
order = np.argsort(f)
fs, ys = f[order], y[order]
n1, n0 = y.sum(), (~y).sum()
tp = n1 - np.cumsum(ys)              # positives strictly above position i
tn = np.cumsum(~ys)
b = 50 * (tp / n1 + tn / n0)
i = int(np.argmax(b))
exact = float((fs[i] + fs[min(i + 1, len(fs) - 1)]) / 2)
near = b >= b[i] - 0.05
O["threshold"] = dict(exact=exact, nih_bacc=float(b[i]), plateau_within_005=[float(fs[near].min()), float(fs[near].max())])
for t in (exact, .2946, .303, .316):
    O["threshold"][f"t={t:.4f}"] = dict(NIH=round(bacc(c.label, f, t), 2), MP=round(bacc(m.y, m.f, t), 2))

# 2. Outcome-rule precision and fusion, on MP-IDB as development data
S = R / "experiments/E6b_five_checkpoint_external_rescore/scores"
arms = ("E1", "E8b", "E8c")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ct = np.mean([logit(np.load(S / f"B_{a}_f{k}.npy")) for a in arms for k in range(5)], axis=0)
base = np.mean([logit(np.load(S / f"A_E1_f{k}.npy")) for k in range(5)], axis=0)
oof = []
for a, tag in zip(arms, ("e1", "e8b", "e8c")):
    root = R / f"runs_kaggle/e4b-colour-transfer/runs/E4b_{tag}_colourtransfer"
    d = pd.concat([pd.read_csv(next(root.glob(f"grouped_r0f{k}_t0*")) / "test_predictions.csv") for k in range(5)])
    oof.append(d.set_index("idx").p.rename(a))
oof = pd.concat(oof, axis=1)
oofL = logit(oof.values).mean(1)
fN = c.set_index("idx").f.reindex(oof.index).values
yN = c.set_index("idx").label.reindex(oof.index).values
lr = LogisticRegression(max_iter=1000).fit(np.c_[oofL, fN], yN)
fus = lr.decision_function(np.c_[ct, m.f.values])
O["fusion"] = dict(coef_logit=float(lr.coef_[0][0]), coef_feature=float(lr.coef_[0][1]), intercept=float(lr.intercept_[0]),
                   NIH_oof_AUC_model=auc(yN, oofL), NIH_oof_AUC_fusion=auc(yN, lr.decision_function(np.c_[oofL, fN])),
                   NIH_BACC_model=bacc(yN, oofL, 0), NIH_BACC_fusion=bacc(yN, lr.decision_function(np.c_[oofL, fN]), 0),
                   MP_AUC_feature=auc(m.y, m.f), MP_AUC_ct=auc(m.y, ct), MP_AUC_fusion=auc(m.y, fus),
                   MP_BACC_feature=bacc(m.y, m.f, exact), MP_BACC_ct=bacc(m.y, ct, 0), MP_BACC_fusion=bacc(m.y, fus, 0))
yy = m.y.values
G = [np.where(m.image.values == k)[0] for k in m.image.unique()]
dA, dB, d21 = [], [], []
for _ in range(2000):
    i = np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])
    dA.append(auc(yy[i], ct[i]) - auc(yy[i], m.f.values[i]))
    dB.append(bacc(yy[i], ct[i], 0) - bacc(yy[i], m.f.values[i], exact))
    d21.append(auc(yy[i], base[i]) - auc(yy[i], m.f.values[i]))
pct = lambda v: [round(float(np.mean(v)), 4), *np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()]
O["rule_precision_MPIDB_104_images"] = dict(AUC_3minus1=pct(dA), BACC_3minus1=pct(dB), AUC_2minus1=pct(d21),
                                            AUC_CI_width=float(np.subtract(*np.percentile(dA, [97.5, 2.5]))),
                                            implied_width_at_1364_images=float(np.subtract(*np.percentile(dA, [97.5, 2.5])) * np.sqrt(104 / 1364)))

# 3. Image-median recalibration, development on MP-IDB
L = pd.Series(ct)
med = L.groupby(m.image.values).transform("median")
O["recalibration_dev"] = dict(BACC_raw=bacc(yy, ct, 0), BACC_image_median_centred=bacc(yy, (L - med).values, 0),
                              AUC_raw=auc(yy, ct), AUC_centred=auc(yy, (L - med).values),
                              median_logit_of_image_medians=float(med.groupby(m.image.values).first().median()))

# 4. Date-linked blocks and fold balance
cells = pd.read_csv(R / "splits/cells.csv")
cells["date"] = cells.rel_path.str.extract(r"IMG_(\d{8})")[0]
pats = cells.patient.unique().tolist()
parent = {p: p for p in pats}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for _, g in cells.groupby("date"):
    ps = g.patient.unique()
    for b_ in ps[1:]:
        parent[find(b_)] = find(ps[0])
t = cells.groupby("patient").label.agg(n="size", pos="sum")
t["healthy"] = t.pos == 0
t["block"] = [find(p) for p in t.index]
blk = t.groupby("block").agg(pats=("n", "size"), healthy=("healthy", "sum"), n=("n", "sum"), pos=("pos", "sum")).sort_values("n", ascending=False)
O["blocks"] = dict(n_blocks=len(blk), top5=blk.head(5).reset_index(drop=True).to_dict("records"),
                   healthy_in_top3=int(blk.healthy.head(3).sum()), healthy_total=int(blk.healthy.sum()),
                   share_cells_largest_block=float(blk.n.iloc[0] / blk.n.sum()))
best = None
arr = blk[["healthy", "n", "pos"]].values.astype(float)
for seed in range(200):
    r = np.random.default_rng(seed)
    a = r.integers(0, 5, len(arr))
    def cost(a):
        s = np.array([arr[a == k].sum(0) for k in range(5)])
        prev = 100 * s[:, 2] / np.maximum(s[:, 1], 1)
        return s[:, 0].std() + s[:, 1].std() / 500 + 2 * prev.std()
    c0 = cost(a)
    for _ in range(4000):
        i, k = r.integers(len(arr)), r.integers(5)
        old = a[i]
        a[i] = k
        c1 = cost(a)
        if c1 <= c0:
            c0 = c1
        else:
            a[i] = old
    if best is None or c0 < best[0]:
        best = (c0, a.copy())
a = best[1]
s = np.array([arr[a == k].sum(0) for k in range(5)])
O["fold_search"] = dict(healthy=s[:, 0].astype(int).tolist(), cells=s[:, 1].astype(int).tolist(),
                        prevalence=np.round(100 * s[:, 2] / s[:, 1], 2).tolist(), prevalence_sd=float((100 * s[:, 2] / s[:, 1]).std(ddof=1)))

# 5. Repo facts
ls = pd.read_csv(R / "splits/leaky_splits.csv", nrows=2)
O["leaky_seed_columns"] = [x for x in ls.columns if x.startswith("seed")]
txt = open(R / "experiments/E5_explanation_validity/README.md").read()
O["E5_runtime_lines"] = [l[:150] for l in txt.splitlines() if re.search(r"\bmin\b|minutes|runtime", l)][:4]
print(json.dumps(O, indent=1, default=float))
