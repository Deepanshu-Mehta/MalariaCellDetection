"""Exploratory, unregistered no-GPU checks behind E10, 2026-11-02. Touches no new MP-IDB scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.malaria import data, models  # noqa: E402
from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path  # noqa: E402

R = ROOT / "research"
OUT = {}
torch.set_num_threads(6)
rng = np.random.default_rng(0)

cells = pd.read_csv(R / "splits" / "cells.csv", index_col="idx")
folds = pd.read_csv(R / "splits" / "grouped_folds.csv")
folds = folds[(folds.repeat == 0) & (folds.role == "test")].set_index("patient").fold
flagged = set(pd.read_csv(R / "experiments/E7_label_noise_audit/flagged_cells.csv").iloc[:, 0])

# ---- A. Patient composition
g = cells.groupby("patient").label.agg(n="size", pos="sum")
g["infected"] = g.pos > 0
g["frac"] = g.pos / g.n
inf = g[g.infected]
OUT["A_composition"] = {
    "patients": len(g), "infected": int(g.infected.sum()), "healthy_only": int((~g.infected).sum()),
    "cells_per_patient_quartiles": g.n.quantile([0, .25, .5, .75, 1]).tolist(),
    "infected_cells_per_patient_quartiles": inf.n.quantile([0, .25, .5, .75, 1]).tolist(),
    "healthy_cells_per_patient_quartiles": g[~g.infected].n.quantile([0, .25, .5, .75, 1]).tolist(),
    "parasitised_frac_in_infected_quartiles": inf.frac.quantile([0, .25, .5, .75, 1]).round(3).tolist(),
    "uninfected_cells_in_infected_patients": int(((cells.label == 0) & cells.patient.isin(inf.index)).sum()),
}


def boot_auc(y, s, B=5000):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(B):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        v.append(roc_auc_score(y[i], s[i]))
    return [round(float(roc_auc_score(y, s)), 4), *np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()]


# ---- B. Patient-level ceiling, natural bags, E1 out-of-fold predictions
pred = pd.concat([pd.read_csv(ckpt_path(ARMS["E1"], f).parent / "test_predictions.csv") for f in range(5)])
assert pred.idx.is_unique and len(pred) == len(cells)
p = pred.groupby("patient").p.agg(mean_p="mean", frac_pos=lambda x: (x >= .5).mean())
y = g.infected.astype(int).reindex(p.index)
healthy = pred[pred.patient.isin(g.index[~g.infected])]
unin_inf = pred[pred.patient.isin(inf.index) & (pred.label == 0)]
hfp = healthy.assign(fp=healthy.p >= .5).groupby("patient").fp.mean()
OUT["B_natural_bag_E1"] = {
    "AUC_mean_p": boot_auc(y, p.mean_p), "AUC_frac_pos": boot_auc(y, p.frac_pos),
    "lowest_infected_frac_pos": float(p.frac_pos[y == 1].min()),
    "highest_healthy_frac_pos": float(p.frac_pos[y == 0].max()),
    "cell_FPR_healthy_only_pct": round(100 * float((healthy.p >= .5).mean()), 3),
    "cell_FPR_uninfected_in_infected_pct": round(100 * float((unin_inf.p >= .5).mean()), 3),
    "healthy_patient_FPR_quartiles_pct": (100 * hfp.quantile([0, .5, .9, 1])).round(2).tolist(),
}

# ---- C. Parasite-free patient-level task: uninfected-labelled cells only
un = cells[(cells.label == 0) & ~cells.index.isin(flagged)]
root = data.image_root()
rows = []
for i, r in un.iterrows():
    im = data.read_rgb(root / r.rel_path)
    m = data.foreground_mask(im)
    rows.append(im[m].mean(0) if m.any() else np.full(3, np.nan))
colour = pd.DataFrame(rows, index=un.index, columns=list("RGB")).dropna()


def patient_cv(pm):
    """LogReg on one mean feature vector per patient, patient-level 5-fold (the frozen r0 folds)."""
    ys, ss = [], []
    yy = g.infected.astype(int).reindex(pm.index)
    fo = folds.reindex(pm.index)
    for f in range(5):
        tr, te = fo != f, fo == f
        sc = StandardScaler().fit(pm[tr])
        c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(pm[tr]), yy[tr])
        ss.append(c.decision_function(sc.transform(pm[te])))
        ys.append(yy[te].values)
    return boot_auc(np.concatenate(ys), np.concatenate(ss))


OUT["C_parasite_free"] = {
    "n_uninfected_cells_after_E7_filter": len(un),
    "colour_RGB_means": patient_cv(colour.join(cells.patient).groupby("patient")[list("RGB")].mean()),
}
for name in ["dinov2_s", "reddino_s"]:
    arr = np.load(R / f"runs_kaggle/e9-foundation-probes/E9_outputs/features/{name}.npy", mmap_mode="r")
    assert arr.shape[0] == len(cells)
    d = pd.DataFrame(np.asarray(arr)[un.index.values], index=un.index)
    OUT["C_parasite_free"][name] = patient_cv(d.join(cells.patient).groupby("patient").mean())
pu = pred[pred.idx.isin(un.index)].groupby("patient").p.mean()
OUT["C_parasite_free"]["E1_mean_p"] = boot_auc(g.infected.astype(int).reindex(pu.index), pu)

# ---- D. E6 threshold-free comparison, bag size and acquisition date
import re  # noqa: E402
mp = pd.read_csv(R / "experiments/E6_external_test/mpidb_predictions.csv")
OUT["E6_rows"] = {"n": len(mp), "label_values": mp.groupby(["label", "binary_label"]).size().to_dict().__str__(),
                  "excluded": mp.excluded.value_counts().to_dict().__str__()}
y = mp.binary_label.values
pc = {a: mp[f"p_{a}"].values for a in ["E1", "E8b", "E8c"]}
assert abs(roc_auc_score(y, pc["E1"]) - 0.8672955633908137) < 1e-6, "must reproduce E6 results.json"
groups = [np.where(mp.image.values == k)[0] for k in mp.image.unique()]
d = {"E8b-E1": [], "E8c-E1": [], "E8b-E8c": []}
for _ in range(2000):
    i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
    a = {k: roc_auc_score(y[i], v[i]) for k, v in pc.items()}
    for k in d:
        l, r = k.split("-")
        d[k].append(a[l] - a[r])
OUT["E6_AUC"] = {"n_images": len(groups), "auc": {k: round(roc_auc_score(y, v), 4) for k, v in pc.items()},
                 "paired": {k: [round(float(np.mean(v)), 4), *np.round(np.percentile(v, [2.5, 97.5]), 4).tolist()]
                            for k, v in d.items()}}

g = cells.groupby("patient").label.agg(n="size", pos="sum")
g["infected"] = (g.pos > 0).astype(int)
OUT["bag_size"] = {
    "healthy_n_quartiles": g.n[g.infected == 0].quantile([0, .25, .5, .75, 1]).tolist(),
    "infected_n_quartiles": g.n[g.infected == 1].quantile([0, .25, .5, .75, 1]).tolist(),
    "AUC_cell_count_alone": round(float(roc_auc_score(g.infected, g.n)), 4),
    "infected_patients_under_100_cells": int(((g.infected == 1) & (g.n < 100)).sum()),
    "uninfected_cells_only_healthy_quartiles": g.n[g.infected == 0].quantile([0, .5, 1]).tolist(),
}
un = cells[cells.label == 0].groupby("patient").size().reindex(g.index).fillna(0)
OUT["bag_size"]["AUC_uninfected_cell_count_alone"] = round(float(roc_auc_score(g.infected, un)), 4)
OUT["bag_size"]["uninfected_count_infected_quartiles"] = un[g.infected == 1].quantile([0, .25, .5, .75, 1]).tolist()


def date(p):
    m = re.search(r"IMG_(\d{8})", p)
    return m.group(1) if m else None


cells["date"] = cells.rel_path.map(date)
OUT["date"] = {"cells_without_date": int(cells.date.isna().sum()), "unique_dates": int(cells.date.nunique())}
pdate = cells.groupby("patient").date.agg(lambda s: tuple(sorted(set(s.dropna()))))
OUT["date"]["patients_with_multiple_dates"] = int(pdate.map(len).gt(1).sum())
h = set(x for k in g.index[g.infected == 0] for x in pdate[k])
i = set(x for k in g.index[g.infected == 1] for x in pdate[k])
OUT["date"].update({"healthy_dates": len(h), "infected_dates": len(i), "shared_dates": len(h & i),
                    "healthy_range": [min(h), max(h)] if h else None,
                    "infected_range": [min(i), max(i)] if i else None})
sh = h & i
OUT["date"]["healthy_patients_on_shared_dates"] = int(sum(any(x in sh for x in pdate[k]) for k in g.index[g.infected == 0]))
OUT["date"]["infected_patients_on_shared_dates"] = int(sum(any(x in sh for x in pdate[k]) for k in g.index[g.infected == 1]))
OUT["date"]["patients_per_date_quartiles"] = cells.groupby("date").patient.nunique().quantile([0, .5, .9, 1]).tolist()
pre = cells.groupby("patient").nih_id.first().map(lambda s: re.sub(r"\d+", "#", s))
OUT["id_pattern_by_group"] = pd.crosstab(pre, g.infected).to_dict().__str__()

cells["fold"] = cells.patient.map(folds)
sp = cells.dropna(subset=["date"]).groupby("date").fold.nunique()
OUT["date"]["dates_spanning_multiple_test_folds"] = f"{int((sp > 1).sum())}/{len(sp)}"

# ---- E. Effective receptive field of trained checkpoints (fold-held-out cells, CPU)
def tensor(j):
    t = torch.from_numpy(data.read_rgb(root / cells.rel_path[j])).permute(2, 0, 1).float() / 255.0
    return torch.nn.functional.interpolate(t[None], size=(224, 224), mode="bilinear",
                                           align_corners=False, antialias=True)[0]


yy, xx = np.mgrid[0:224, 0:224]
rad = np.hypot(yy - 111.5, xx - 111.5).ravel()
order = np.argsort(rad)
res = {}
for arm, cfg in ARMS.items():
    r50, r90 = [], []
    for f in range(5):
        m = models.build_model(cfg["model"]).eval()
        m.load_state_dict(torch.load(ckpt_path(cfg, f), map_location="cpu"))
        te = folds.index[folds == f]
        sub = cells[cells.patient.isin(te)]
        ids = np.r_[rng.choice(sub.index[sub.label == 1], 30, False), rng.choice(sub.index[sub.label == 0], 30, False)]
        x = torch.stack([tensor(j) for j in ids]).requires_grad_(True)
        h = m.features(m.stem(x))
        c = h.shape[-1] // 2
        h[:, :, c, c].abs().sum().backward()
        e = x.grad.abs().sum(1).detach().numpy()
        e = (e / e.sum((1, 2), keepdims=True)).mean(0).ravel()
        cum = np.cumsum(e[order])
        r50.append(float(rad[order][np.searchsorted(cum, .5)]))
        r90.append(float(rad[order][np.searchsorted(cum, .9)]))
    res[arm] = dict(r50_mean=round(np.mean(r50), 1), r50_sd=round(np.std(r50, ddof=1), 1),
                    r90_mean=round(np.mean(r90), 1), r90_sd=round(np.std(r90, ddof=1), 1))
OUT["E_ERF_px_at_224"] = res

json.dump(OUT, open(Path(__file__).parent / "exploratory_results.json", "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
