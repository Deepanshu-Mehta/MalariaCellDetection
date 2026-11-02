"""E10 part A, no GPU. Colour variance by date vs patient, and parasite-free patient AUC within shared dates.

    arch -arm64 .venv/bin/python -m research.experiments.E10_leave_session_out.part_a

Fixed before running. Cells: Uninfected, E7 flags removed, so a parasite's own colour cannot drive the result.
Colour primary: eta-squared of the three foreground channel means together (summed between-group over summed total
sums of squares), date divided by patient. Epsilon-squared repeats it, because patient has 200 groups to date's 56.
AUC primary: the out-of-fold patient scores from exploratory_checks.py, restricted to patients on a shared date. Secondary:
AUC over healthy-infected pairs on the same date only, since a pooled AUC can still compare across the 4 dates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.malaria import data

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
B = 5000
CH = list("RGB")


def eta(frame, group):
    """(eta-squared, epsilon-squared) of `frame`'s columns jointly, on one grouping."""
    total = ((frame - frame.mean()) ** 2).values.sum()
    within = ((frame - frame.groupby(group).transform("mean")) ** 2).values.sum()
    n, k = len(frame), group.nunique()
    return 1 - within / total, 1 - (within / (n - k)) / (total / (n - 1))


def oof_scores(pm, y, folds):
    out = pd.Series(index=pm.index, dtype=float)
    fo = folds.reindex(pm.index)
    for f in range(5):
        tr, te = fo != f, fo == f
        sc = StandardScaler().fit(pm[tr])
        c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(pm[tr]), y[pm.index][tr])
        out[te] = c.decision_function(sc.transform(pm[te]))
    return out


def within_date_auc(s, y, dates):
    """Share of healthy-infected pairs from one date ranked correctly, ties counted half."""
    win = tot = 0.0
    for _, ps in dates.groupby(dates):
        a, b = s[ps.index][y[ps.index] == 1].values, s[ps.index][y[ps.index] == 0].values
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            win, tot = win + (d > 0).sum() + .5 * (d == 0).sum(), tot + d.size
    return win / tot if tot else np.nan


def main():
    O = {}
    rng = np.random.default_rng(0)
    cells = pd.read_csv(R / "splits" / "cells.csv", index_col="idx")
    cells["date"] = cells.rel_path.str.extract(r"IMG_(\d{8})")[0]
    folds = pd.read_csv(R / "splits" / "grouped_folds.csv")
    folds = folds[(folds.repeat == 0) & (folds.role == "test")].set_index("patient").fold
    flagged = set(pd.read_csv(R / "experiments/E7_label_noise_audit/flagged_cells.csv").iloc[:, 0])
    g = cells.groupby("patient").label.agg(n="size", pos="sum")
    y = (g.pos > 0).astype(int)

    un = cells[(cells.label == 0) & ~cells.index.isin(flagged)]
    root = data.image_root()
    rows = {}
    for i, r in un.iterrows():
        im = data.read_rgb(root / r.rel_path)
        m = data.foreground_mask(im)
        if m.any():
            rows[i] = im[m].mean(0)
    col = pd.DataFrame.from_dict(rows, orient="index", columns=CH)
    meta = un.loc[col.index]

    e_d, e_p = eta(col, meta.date), eta(col, meta.patient)
    O["colour"] = dict(n_cells=len(col), eta2_date=e_d[0], eta2_patient=e_p[0], ratio_primary=e_d[0] / e_p[0],
                       eps2_date=e_d[1], eps2_patient=e_p[1], ratio_adjusted=e_d[1] / e_p[1],
                       per_channel={c: eta(col[[c]], meta.date)[0] / eta(col[[c]], meta.patient)[0] for c in CH})
    pat_mean = col.groupby(meta.patient).mean()
    first = cells.groupby("patient").date.agg(lambda d: d.value_counts().index[0])
    O["colour"]["patient_means_eta2_date"] = eta(pat_mean, first[pat_mean.index])[0]

    # Shared dates: dates holding at least one healthy-only and one infected patient
    pd_ = cells[["patient", "date"]].drop_duplicates()
    pd_["infected"] = pd_.patient.map(y)
    shared = [d for d, h in pd_.groupby("date") if h.infected.nunique() == 2]
    on = pd_[pd_.date.isin(shared)]
    assert on.patient.is_unique, "a patient spans two shared dates"
    dates = on.set_index("patient").date
    sy = y[dates.index]
    O["shared"] = dict(dates=shared, healthy=int((sy == 0).sum()), infected=int((sy == 1).sum()))

    feats = dict(colour_RGB_means=pat_mean)
    for name in ("dinov2_s", "reddino_s"):
        arr = np.load(R / f"runs_kaggle/e9-foundation-probes/E9_outputs/features/{name}.npy", mmap_mode="r")
        assert arr.shape[0] == len(cells)
        d = pd.DataFrame(np.asarray(arr)[un.index.values], index=un.index)
        feats[name] = d.join(cells.patient).groupby("patient").mean()

    O["auc"] = {}
    for name, pm in feats.items():
        s = oof_scores(pm, y, folds)
        ok = s.index.intersection(dates.index)
        yy, ss, dd = y[ok], s[ok], dates[ok]
        pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
        boot, within = [], []
        for _ in range(B):
            i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            boot.append(roc_auc_score(yy.values[i], ss.values[i]))
        for _ in range(B):
            keep = []
            for h in dd.unique():                          # resample patients inside each date, keeping dates fixed
                ids = np.where(dd.values == h)[0]
                keep.append(rng.choice(ids, len(ids)))
            i = np.concatenate(keep)
            t = pd.DataFrame(dict(y=yy.values[i], s=ss.values[i], d=dd.values[i]))
            t.index = np.arange(len(t))
            within.append(within_date_auc(t.s, t.y, t.d))
        O["auc"][name] = dict(all_patients=roc_auc_score(y[s.index], s), shared_dates=roc_auc_score(yy, ss),
                              shared_ci=np.percentile(boot, [2.5, 97.5]).tolist(), n=len(ok),
                              within_date_pairs=within_date_auc(ss, yy, dd),
                              within_ci=np.nanpercentile(within, [2.5, 97.5]).tolist())
    json.dump(O, open(HERE / "part_a_results.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    main()
