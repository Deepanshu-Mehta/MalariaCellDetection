"""E2 part D, stats: patient-cluster bootstrap CIs on the stain-swap flip-rate differences between arms.

    arch -arm64 .venv/bin/python research/experiments/E2_shortcut_evidence/analysis_stainswap_stats.py

The raw summary (results_stainswap_summary.csv) gives point estimates only. This adds a paired patient-level
bootstrap (B=5000) for E8b-E1 and E8b-E8c differences in each (patient_type, label, condition) cell, so "look
similar" is backed by a number instead of eyeballing two decimal places.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["E1", "E8b", "E8c"]
B = 5000


def load():
    return pd.concat([pd.read_csv(HERE / f"stainswap_{a}.csv") for a in ARMS], ignore_index=True)


def boot_mean_diff(a, b, rng):
    """a, b: flipped-indicator arrays with a `patient` column each, same (patient_type,label,condition) cell."""
    pa = {p: g.flipped.values for p, g in a.groupby("patient")}
    pb = {p: g.flipped.values for p, g in b.groupby("patient")}
    common = np.array(sorted(set(pa) & set(pb)))
    if len(common) < 5:
        return None
    diffs = []
    for _ in range(B):
        keys = rng.choice(common, len(common))
        da = np.concatenate([pa[k] for k in keys]).mean()
        db = np.concatenate([pb[k] for k in keys]).mean()
        diffs.append(da - db)
    diffs = np.array(diffs)
    return dict(mean_diff=float(a.flipped.mean() - b.flipped.mean()),
               ci=np.percentile(diffs, [2.5, 97.5]).tolist())


def main():
    d = load()
    rng = np.random.default_rng(0)
    res = {}
    for (pt, label, cond), _ in d.groupby(["patient_type", "label", "condition"]):
        cell = f"{pt}_label{label}_{cond}"
        sub = {a: d[(d.arm == a) & (d.patient_type == pt) & (d.label == label) & (d.condition == cond)]
              for a in ARMS}
        res[cell] = {}
        for a, base in [("E8b", "E1"), ("E8b", "E8c"), ("E8c", "E1")]:
            if len(sub[a]) and len(sub[base]):
                r = boot_mean_diff(sub[a], sub[base], rng)
                if r:
                    res[cell][f"{a}-{base}"] = r
    json.dump(res, open(HERE / "results_stainswap_stats.json", "w"), indent=1, default=float)
    for cell, v in res.items():
        print(cell)
        for k, r in v.items():
            sig = "*" if r["ci"][0] * r["ci"][1] > 0 else " "
            print(f"  {k}: diff={r['mean_diff']:+.4f}  CI=[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] {sig}")


if __name__ == "__main__":
    main()
