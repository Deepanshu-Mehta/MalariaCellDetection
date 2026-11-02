"""Build the E10 session folds and the E8r leaky seeds. Frozen split files are read, never written.

    arch -arm64 .venv/bin/python -m research.scripts.build_session_splits

session_folds.csv: patients linked by any shared imaging date form one block, and whole blocks go to one role, so
train, val and test never share a date. Only patient metadata is used: no model score enters this file.
leaky_splits_replication.csv: E0's stratified_cell_split on new seeds 45-47.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.malaria import splits

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
SEED, RESTARTS, STEPS, N_VAL_DRAWS = 20261102, 200, 4000, 20000
VAL_PATIENTS, VAL_HEALTHY = 16, 4            # E1's inner validation: 4 patients from each of 4 strata
LEAKY_REPLICATION_SEEDS = (45, 46, 47)


def date_of(cells):
    d = cells.rel_path.str.extract(r"IMG_(\d{8})")[0]
    assert d.notna().all(), "cell without an imaging date"
    return d


def patient_blocks(cells):
    """Connected components of the patient-date graph. Returns a table with one row per block."""
    d = cells.assign(date=date_of(cells))
    parent = {p: p for p in d.patient.unique()}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, g in d.groupby("date"):
        ps = g.patient.unique()
        for b in ps[1:]:
            parent[find(b)] = find(ps[0])
    t = d.groupby("patient").label.agg(n="size", pos="sum")
    t["healthy"] = (t.pos == 0).astype(int)
    root = {p: find(p) for p in t.index}
    ids = {r: i for i, r in enumerate(sorted(set(root.values())))}
    t["block"] = [ids[root[p]] for p in t.index]
    return t


def fold_cost(sums):
    prev = 100 * sums[:, 2] / np.maximum(sums[:, 1], 1)
    return sums[:, 0].std() + sums[:, 1].std() / 500 + 2 * prev.std()


def assign_folds(blocks):
    """Local search over block-to-fold assignments, balancing healthy patients, cells and prevalence."""
    arr = blocks[["healthy", "n", "pos"]].values.astype(float)
    best = None
    for restart in range(RESTARTS):
        r = np.random.default_rng([SEED, restart])
        a = r.integers(0, 5, len(arr))
        sums = np.array([arr[a == k].sum(0) for k in range(5)])
        c0 = fold_cost(sums)
        for _ in range(STEPS):
            i, k = int(r.integers(len(arr))), int(r.integers(5))
            old = a[i]
            if k == old:
                continue
            sums[old] -= arr[i]
            sums[k] += arr[i]
            c1 = fold_cost(sums)
            if c1 <= c0:
                c0, a[i] = c1, k
            else:
                sums[old] += arr[i]
                sums[k] -= arr[i]
        if best is None or c0 < best[0]:
            best = (c0, a.copy())
    return best[1]


def pick_val(blocks, fold_of, k):
    """Blocks outside test fold k giving about 16 patients, 4 healthy, at the training prevalence."""
    pool = blocks[fold_of != k]
    target = 100 * pool.pos.sum() / pool.n.sum()
    r = np.random.default_rng([SEED, 1000, k])
    best = None
    for _ in range(N_VAL_DRAWS):
        order = r.permutation(len(pool))
        take = []
        pats = 0
        for i in order:
            if pats + pool.pats.iloc[i] <= VAL_PATIENTS + 2:
                take.append(i)
                pats += pool.pats.iloc[i]
            if pats >= VAL_PATIENTS:
                break
        s = pool.iloc[take]
        prev = 100 * s.pos.sum() / s.n.sum()
        c = abs(s.pats.sum() - VAL_PATIENTS) + 2 * abs(s.healthy.sum() - VAL_HEALTHY) + abs(prev - target) / 5
        if best is None or c < best[0]:
            best = (c, s.index.values.copy())
    return set(best[1])


def main():
    cells = pd.read_csv(SPLITS / "cells.csv", index_col="idx")
    t = patient_blocks(cells)
    blocks = t.groupby("block").agg(pats=("n", "size"), healthy=("healthy", "sum"), n=("n", "sum"), pos=("pos", "sum"))
    fold_of = assign_folds(blocks)

    rows = []
    for k in range(5):
        val = pick_val(blocks, fold_of, k)
        for p, r in t.iterrows():
            b = r.block
            role = "test" if fold_of[b] == k else "val" if b in val else "train"
            rows.append((0, k, p, role, int(b)))
    out = pd.DataFrame(rows, columns=["repeat", "fold", "patient", "role", "block"])

    # Assertions: patients partition, and no date crosses roles
    dates = cells.assign(date=date_of(cells)).groupby("patient").date.apply(set)
    report = dict(n_blocks=len(blocks), folds=[])
    assert out[out.role == "test"].patient.is_unique and out[out.role == "test"].patient.nunique() == 200
    for k, g in out.groupby("fold"):
        role = g.set_index("patient").role
        d = {x: set().union(*[dates[p] for p in role.index[role == x]]) for x in ("train", "val", "test")}
        assert not (d["train"] & d["test"]) and not (d["val"] & d["test"]) and not (d["train"] & d["val"]), f"date overlap fold {k}"
        rec = dict(fold=k)
        for x in ("train", "val", "test"):
            ps = role.index[role == x]
            rec[x] = dict(patients=len(ps), healthy=int(t.healthy[ps].sum()), cells=int(t.n[ps].sum()),
                          prevalence=round(100 * float(t.pos[ps].sum() / t.n[ps].sum()), 2), dates=len(d[x]))
            assert t.healthy[ps].sum() > 0 and (t.pos[ps] > 0).sum() > 0, f"one class only in {x} fold {k}"
        report["folds"].append(rec)
    p = [f["test"]["prevalence"] for f in report["folds"]]
    report["test_prevalence_sd"] = round(float(np.std(p, ddof=1)), 3)
    out.to_csv(SPLITS / "session_folds.csv", index=False)

    frozen = pd.read_csv(SPLITS / "leaky_splits.csv", index_col="idx")
    fr = dict(train=0.7, val=0.1, test=0.2)
    assert (splits.stratified_cell_split(cells, fr, 42).values == frozen.seed42.values).all(), "seed 42 not reproduced"
    assert (splits.stratified_cell_split(cells, fr, 44).values == frozen.seed44.values).all(), "seed 44 not reproduced"
    leaky = pd.DataFrame({f"seed{s}": splits.stratified_cell_split(cells, fr, s) for s in LEAKY_REPLICATION_SEEDS})
    assert leaky.index.equals(cells.index)
    leaky.to_csv(SPLITS / "leaky_splits_replication.csv", index_label="idx")

    names = ["session_folds.csv", "leaky_splits_replication.csv"]
    sums = (SPLITS / "SHA256SUMS").read_text().splitlines()
    kept = [l for l in sums if l.split()[-1] not in names]
    assert len(kept) == 4, "frozen hash lines changed"
    for name, want in (l.split()[::-1] for l in kept):
        assert splits.sha256(SPLITS / name) == want, f"frozen split changed: {name}"
    (SPLITS / "SHA256SUMS").write_text("\n".join(kept + [f"{splits.sha256(SPLITS / n)}  {n}" for n in names]) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
