"""E6c: how much of the MP-IDB gap does colour invariance recover?

    arch -arm64 .venv/bin/python -m research.experiments.E6c_colour_invariance_external_test.analysis
"""
import json
from pathlib import Path

import numpy as np

from research.experiments.E6b_five_checkpoint_external_rescore.analysis import (
    METRICS, fast_metrics, hierarchical_boot, interval, load_scores, t_interval)
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import ARMS, N_FOLDS, scored_table

HERE = Path(__file__).resolve().parent
PASS_AUC = 0.02


def main():
    t = scored_table()
    y, images = t.binary_label.values, t.image.values
    S = {c: load_scores(c) for c in "ABC"}
    nih = json.load(open(HERE.parent / "E6b_five_checkpoint_external_rescore" / "results.json"))["nih_bacc"]

    per = {c: {a: {f: fast_metrics(y, S[c][(a, f)]) for f in range(N_FOLDS)} for a in ARMS} for c in "ABC"}
    keys = [(c, a) for c in "ABC" for a in ARMS]
    boot = hierarchical_boot({(c, a): {f: S[c][(a, f)] for f in range(N_FOLDS)} for c, a in keys}, y, images, keys)

    def fold_mean(c, a, m):
        return np.array([per[c][a][f][m] for f in range(N_FOLDS)])

    out = dict(levels={c: {a: {m: float(fold_mean(c, a, m).mean()) for m in METRICS} for a in ARMS} for c in "ABC"})
    out["pooled_change"], out["per_arm_change"] = {}, {}
    for c in "BC":
        name = f"{c}-A"
        out["pooled_change"][name] = {}
        out["per_arm_change"][name] = {a: {} for a in ARMS}
        for m in METRICS:
            fold_d = np.mean([fold_mean(c, a, m) - fold_mean("A", a, m) for a in ARMS], axis=0)
            draws = np.mean([boot[(c, a)][m] - boot[("A", a)][m] for a in ARMS], axis=0)
            out["pooled_change"][name][m] = dict(mean=float(fold_d.mean()), hier_ci=interval(draws), t_ci=t_interval(fold_d))
            for a in ARMS:
                d = fold_mean(c, a, m) - fold_mean("A", a, m)
                out["per_arm_change"][name][a][m] = dict(mean=float(d.mean()), hier_ci=interval(boot[(c, a)][m] - boot[("A", a)][m]))

    gaps = {}
    for c in "BC":
        closed = {a: float((fold_mean(c, a, "BACC").mean() - fold_mean("A", a, "BACC").mean())
                           / (nih[a] - fold_mean("A", a, "BACC").mean())) for a in ARMS}
        gaps[c] = dict(per_arm=closed, pooled=float(np.mean(list(closed.values()))),
                       pooled_bacc=float(np.mean([fold_mean(c, a, "BACC").mean() for a in ARMS])))
    out["gap_closed_fraction"] = gaps

    passed = {c: bool(out["pooled_change"][f"{c}-A"]["AUC"]["mean"] >= PASS_AUC
                      and out["pooled_change"][f"{c}-A"]["AUC"]["hier_ci"][0] > 0) for c in "BC"}
    out["colour_invariance_passes"] = passed
    json.dump(out, open(HERE / "results.json", "w"), indent=1, default=float)

    print(json.dumps({k: out[k] for k in ("levels", "pooled_change", "gap_closed_fraction", "colour_invariance_passes")}, indent=1, default=float))
    for name, arms in out["per_arm_change"].items():
        for a, v in arms.items():
            print(name, a, " ".join(f"{m} {v[m]['mean']:+.3f} {np.round(v[m]['hier_ci'], 3)}" for m in METRICS))


if __name__ == "__main__":
    main()
