"""Shared statistics for the experiment analyses: patient-cluster bootstraps and paired comparisons.

Cells from one patient are not independent, so every CI resamples *patients* with replacement.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs_kaggle"


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    se = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    sp = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    auc = np.nan
    if n1 and n0:
        ranks = pd.Series(p).rank().values
        auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return dict(BACC=100 * (se + sp) / 2, SE=100 * se, SP=100 * sp, AUC=auc)


def load_preds(config, split_prefix):
    """All test_predictions.csv for one config and split type (smoke jobs excluded), with a 'tag' column."""
    out = []
    for f in sorted(RUNS.glob(f"*/runs/{config}/{split_prefix}_*/test_predictions.csv")):
        if "smoke" in str(f):
            continue
        out.append(pd.read_csv(f).assign(tag=f.parent.name))
    if not out:
        raise FileNotFoundError(f"no runs for {config} / {split_prefix}")
    return pd.concat(out, ignore_index=True)


def cluster_bootstrap(d, stat, B=5000, seed=0, strata=None):
    """Percentile CI of stat(frame) resampling patients (within `strata` column if given)."""
    rng = np.random.default_rng(seed)
    if strata is None:
        d = d.assign(_s=0); strata = "_s"
    groups = {s: {p: g for p, g in sd.groupby("patient")} for s, sd in d.groupby(strata)}
    vals = []
    for _ in range(B):
        parts = []
        for s, gs in groups.items():
            keys = np.array(list(gs))
            parts += [gs[k] for k in rng.choice(keys, len(keys))]
        vals.append(stat(pd.concat(parts)))
    vals = np.array(vals)
    return dict(lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)), se=float(vals.std()))


def paired_diff(a, b, metric="BACC", B=5000, seed=0, strata=None):
    """a, b: frames with idx, patient, label, p on the SAME cells. Returns observed diff (a - b) and its
    patient-bootstrap CI; `strata` (e.g. split seed) resamples patients within each stratum."""
    m = a.merge(b[["idx", "p"] + ([strata] if strata else [])], on=["idx"] + ([strata] if strata else []),
                suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "arms are not on identical cells"
    f = lambda g: metrics(g.label, g.p_a)[metric] - metrics(g.label, g.p_b)[metric]
    ci = cluster_bootstrap(m, f, B=B, seed=seed, strata=strata)
    return dict(diff=float(f(m)), **ci)
