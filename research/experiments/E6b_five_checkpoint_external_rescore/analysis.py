"""E6b: does E6's MP-IDB architecture ranking survive across five checkpoints per arm?

    arch -arm64 .venv/bin/python -m research.experiments.E6b_five_checkpoint_external_rescore.analysis
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from .score_external import ARMS, N_FOLDS, RESEARCH, SCORES, scored_table

HERE = Path(__file__).resolve().parent
METRICS = ("BACC", "SE", "SP", "AUC")
PAIRS = [("E8b", "E1"), ("E8c", "E1"), ("E8b", "E8c")]
B = 2000


def fast_metrics(y, p):
    pred = p >= 0.5
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    se = 100 * (pred & (y == 1)).sum() / n1
    sp = 100 * (~pred & (y == 0)).sum() / n0
    auc = (st.rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return dict(BACC=(se + sp) / 2, SE=se, SP=sp, AUC=auc)


def load_scores(cond):
    return {(a, f): np.load(SCORES / f"{cond}_{a}_f{f}.npy") for a in ARMS for f in range(N_FOLDS)}


def hierarchical_boot(scores, y, images, keys, B=B, seed=0):
    """keys: labels, each mapping to {fold: scores}. The image and fold draws are shared by every key per b."""
    rng = np.random.default_rng(seed)
    groups = [np.where(images == k)[0] for k in pd.unique(images)]
    out = {k: {m: np.zeros(B) for m in METRICS} for k in keys}
    for b in range(B):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        fs = rng.integers(0, N_FOLDS, N_FOLDS)
        yb = y[idx]
        for k in keys:
            per = {f: fast_metrics(yb, scores[k][f][idx]) for f in set(fs.tolist())}
            for m in METRICS:
                out[k][m][b] = np.mean([per[f][m] for f in fs])
    return out


def interval(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def t_interval(d):
    d = np.asarray(d)
    h = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def main():
    t = scored_table()
    y, images = t.binary_label.values, t.image.values
    S = load_scores("A")

    saved = pd.read_csv(HERE.parent / "E6_external_test" / "mpidb_predictions.csv")
    assert (saved.binary_label.values == y).all() and (saved.file.values == t.file.values).all()
    e6 = json.load(open(HERE.parent / "E6_external_test" / "results.json"))
    repro = {}
    for a in ARMS:
        auc = fast_metrics(y, S[(a, 0)])["AUC"]
        repro[a] = dict(auc=auc, e6_auc=e6[a]["AUC"], auc_diff=abs(auc - e6[a]["AUC"]),
                        max_abs_p_diff=float(np.abs(S[(a, 0)] - saved[f"p_{a}"].values).max()))
    print("reproduction", json.dumps(repro, indent=1))

    per = {a: {f: fast_metrics(y, S[(a, f)]) for f in range(N_FOLDS)} for a in ARMS}
    summary = {a: {m: dict(per_fold=[per[a][f][m] for f in range(N_FOLDS)],
                           mean=float(np.mean([per[a][f][m] for f in range(N_FOLDS)])),
                           sd=float(np.std([per[a][f][m] for f in range(N_FOLDS)], ddof=1)))
                   for m in METRICS} for a in ARMS}

    boot = hierarchical_boot({a: {f: S[(a, f)] for f in range(N_FOLDS)} for a in ARMS}, y, images, list(ARMS))
    comparisons = {}
    for l, r in PAIRS:
        comparisons[f"{l}-{r}"] = {}
        for m in METRICS:
            d = [per[l][f][m] - per[r][f][m] for f in range(N_FOLDS)]
            comparisons[f"{l}-{r}"][m] = dict(mean=float(np.mean(d)), per_fold=d,
                                              hier_ci=interval(boot[l][m] - boot[r][m]), t_ci=t_interval(d))

    above = lambda pair, m: comparisons[pair][m]["hier_ci"][0] > 0
    rule = all(above(p, m) for p in ("E8b-E1", "E8c-E1") for m in ("BACC", "AUC"))
    out = dict(reproduction=repro, summary=summary, comparisons=comparisons, e1_robustly_worst_rule_holds=rule,
               nih_bacc={a: e6[a]["nih_grouped_bacc"] for a in ARMS}, B=B)
    json.dump(out, open(HERE / "results.json", "w"), indent=1, default=float)

    for a in ARMS:
        print(a, " ".join(f"{m}={summary[a][m]['mean']:.3f}+/-{summary[a][m]['sd']:.3f}" for m in METRICS))
        print("   BACC per fold", np.round(summary[a]["BACC"]["per_fold"], 2), "AUC", np.round(summary[a]["AUC"]["per_fold"], 3))
    for p, c in comparisons.items():
        for m in ("BACC", "AUC", "SE", "SP"):
            print(f"{p} {m}: {c[m]['mean']:+.3f} hier {np.round(c[m]['hier_ci'], 3)} t {np.round(c[m]['t_ci'], 3)}")
    print("E1 robustly worst rule holds:", rule)


if __name__ == "__main__":
    main()
