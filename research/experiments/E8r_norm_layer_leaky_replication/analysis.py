"""E8r: mean paired LN-IN leaky BACC over seeds 45-47. Patients are resampled once and shared by every seed.

    arch -arm64 .venv/bin/python -m research.experiments.E8r_norm_layer_leaky_replication.analysis
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.malaria.stats import load_preds, metrics

HERE = Path(__file__).resolve().parent
SEEDS, B, MDE = (45, 46, 47), 2000, .29
ARMS = dict(LN="E8_paper_faithful", IN="E8c_paper_instancenorm")


def counts(d, patients):
    """patients x 4 array of TP, FN, TN, FP."""
    q = d.p.values >= .5
    y = d.label.values == 1
    t = pd.DataFrame(dict(patient=d.patient.values, tp=q & y, fn=~q & y, tn=~q & ~y, fp=q & ~y)).groupby("patient").sum()
    return t.reindex(patients, fill_value=0).values.astype(float)


def bacc(c):
    tp, fn, tn, fp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 50 * (tp / (tp + fn) + tn / (tn + fp))


def main():
    P = {a: {s: load_preds(c, f"leaky_s{s}").drop_duplicates("idx") for s in SEEDS} for a, c in ARMS.items()}
    O = dict(per_seed={})
    for s in SEEDS:
        assert (P["LN"][s].sort_values("idx").idx.values == P["IN"][s].sort_values("idx").idx.values).all()
        O["per_seed"][s] = {a: metrics(P[a][s].label, P[a][s].p) for a in ARMS}
        O["per_seed"][s]["LN_minus_IN"] = O["per_seed"][s]["LN"]["BACC"] - O["per_seed"][s]["IN"]["BACC"]
    diffs = [O["per_seed"][s]["LN_minus_IN"] for s in SEEDS]
    O["mean_LN_minus_IN"] = float(np.mean(diffs))
    patients = sorted(set().union(*[set(P[a][s].patient) for a in ARMS for s in SEEDS]))
    C = {a: np.stack([counts(P[a][s], patients) for s in SEEDS]) for a in ARMS}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))[None, :, None]
        draws.append((bacc((C["LN"] * w).sum(1)) - bacc((C["IN"] * w).sum(1))).mean())
    O["ci_95"] = np.percentile(draws, [2.5, 97.5]).tolist()
    O["seeds_positive"] = int(sum(d > 0 for d in diffs))
    lo = O["ci_95"][0]
    O["rule"] = "Replicated" if lo > 0 and O["mean_LN_minus_IN"] >= MDE else "Direction only" if lo > 0 else "Not replicated"
    O["gap_difference_secondary"] = O["mean_LN_minus_IN"] - (96.71238841715655 - 96.78496262428334)
    json.dump(O, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    main()
