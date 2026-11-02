"""Split definitions. All grouped splits are over the 200 merged patients (see data.py).

grouped  5 folds x 3 repeats over *patients*, rank-blocked within 4 strata:
         healthy-only (50) and infected tertiles by parasitized-cell count (50 / 50 / 50),
         so every test fold holds exactly 10 patients from each stratum.
         Inner validation (early stopping / checkpoint selection) = 4 training patients per stratum.
leaky    the paper's random cell-level 70/10/20 split, stratified by label (seed 42 = M2 baseline).
m3       the M3 pre-registered patient-level 70/10/20 split, stratified by patient type, 3 seeds.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import patient_table

N_FOLDS, N_REPEATS, GROUPED_SEED = 5, 3, 2026
INNER_VAL_PER_STRATUM = 4
LEAKY_SEEDS = (42, 43, 44)
M3_SEEDS = (0, 1, 2)


def patient_strata(df):
    t = patient_table(df)
    inf = t[t.ptype == "infected"]
    tert = pd.qcut(inf.n_para.rank(method="first"), 3, labels=["I1", "I2", "I3"])
    t["stratum"] = "H"
    t.loc[inf.index, "stratum"] = tert.astype(str)
    return t


def blocked_fold_ids(t, rng):
    """Rank-blocked assignment: within each stratum, sort patients (infected by parasitized-cell count,
    healthy-only by cell count), cut into consecutive blocks of N_FOLDS and deal each block across the
    folds in random order. Every fold gets 10 patients per stratum and near-equal parasitized/total
    cell counts, which keeps test-fold prevalence stable."""
    fold = pd.Series(-1, index=t.index)
    for s, g in t.groupby("stratum"):
        key = g.n_cells if s == "H" else g.n_para
        order = key.sort_values(kind="stable").index.values
        for b in range(0, len(order), N_FOLDS):
            block = order[b:b + N_FOLDS]
            fold[block] = rng.permutation(N_FOLDS)[:len(block)]
    return fold


def grouped_folds(df, seed=GROUPED_SEED):
    """Long table: one row per (repeat, fold, patient) with role 'test' / 'val' / 'train'."""
    t = patient_strata(df)
    rows = []
    for r in range(N_REPEATS):
        fold_of = blocked_fold_ids(t, np.random.default_rng([seed, r]))
        for k in range(N_FOLDS):
            rng = np.random.default_rng([seed, r, k])
            tr_t = t[fold_of != k]
            val = set()
            for _, g in tr_t.groupby("stratum"):
                val |= set(rng.choice(g.index.values, INNER_VAL_PER_STRATUM, replace=False))
            for p in t.index[fold_of == k]:
                rows.append((r, k, p, "test"))
            for p in tr_t.index:
                rows.append((r, k, p, "val" if p in val else "train"))
    out = pd.DataFrame(rows, columns=["repeat", "fold", "patient", "role"])
    out["stratum"] = out.patient.map(t.stratum)
    return out


def stratified_cell_split(frame, fractions, seed):
    """The notebook's M2 split: random, stratified by label (reproduces paper Table 1 at seed 42)."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=frame.index)
    for _, g in frame.groupby("label"):
        idx = rng.permutation(g.index.values)
        n_test, n_val = round(fractions["test"] * len(g)), round(fractions["val"] * len(g))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def leaky_splits(df):
    fr = dict(train=0.7, val=0.1, test=0.2)
    return pd.DataFrame({f"seed{s}": stratified_cell_split(df, fr, s) for s in LEAKY_SEEDS})


def m3_patient_splits(df):
    """M3 protocol: 70/10/20 by patient, stratified by patient type -> 30+10 test, 15+5 val patients."""
    t = patient_table(df)
    rows = []
    for s in M3_SEEDS:
        rng = np.random.default_rng(s)
        for _, g in t.groupby("ptype"):
            p = rng.permutation(g.index.values)
            n_test, n_val = round(0.2 * len(p)), round(0.1 * len(p))
            rows += [(s, x, "test") for x in p[:n_test]]
            rows += [(s, x, "val") for x in p[n_test:n_test + n_val]]
            rows += [(s, x, "train") for x in p[n_test + n_val:]]
    return pd.DataFrame(rows, columns=["seed", "patient", "role"])


def check_grouped(folds, df):
    """Leakage and balance assertions; raises on any violation."""
    n_pat = df.patient.nunique()
    for (r, k), g in folds.groupby(["repeat", "fold"]):
        roles = g.groupby("role").patient.apply(set)
        assert not (roles["test"] & roles["train"]) and not (roles["test"] & roles["val"]) \
            and not (roles["val"] & roles["train"]), f"patient overlap in repeat {r} fold {k}"
        assert sum(len(v) for v in roles) == n_pat, f"patients missing in repeat {r} fold {k}"
        assert (g[g.role == "test"].stratum.value_counts() == 10).all(), f"unbalanced test fold {r}/{k}"
    for r, g in folds[folds.role == "test"].groupby("repeat"):
        assert g.patient.is_unique and g.patient.nunique() == n_pat, f"repeat {r} test folds don't partition"


def assert_disjoint(train_patients, test_patients):
    """Runtime guard used by the training script."""
    overlap = set(train_patients) & set(test_patients)
    assert not overlap, f"patient leakage: {sorted(overlap)[:5]}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
