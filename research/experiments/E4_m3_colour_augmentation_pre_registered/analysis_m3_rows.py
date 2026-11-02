"""M3 pre-registered rows (course M4): B0 / B1 / H / C, exactly as registered on 2026-10-05.

    arch -arm64 .venv/bin/python research/experiments/E4_m3_colour_augmentation_pre_registered/analysis_m3_rows.py

B0  baseline, leaky split seed 42, 75 epochs   = the M2 run (milestone2-baseline/outputs/test_predictions.csv)
B1  baseline, M3 patient splits seeds 0-2, 75 epochs              (E1_danet, m3_s*)
H   + ColorJitter(0.1, 0.1, 0.5, 0.1), M3 patient splits, 75 ep   (E4_colourjitter, m3_s*)
C   + ColorJitter, leaky split seed 42, 75 epochs                  (E4_colourjitter, leaky_s42)
Registered decision rule: hypothesis falsified if H <= B1 on SE and BACC averaged over the 3 splits.
Paired tests: same test cells per split seed, patients resampled within each seed (B = 5000).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from research.malaria.stats import load_preds, metrics, paired_diff  # noqa: E402

PRED = dict(SE=(0.5, 1.5), BACC=(0.3, 1.0), AUC=(0.001, 0.004))   # registered H - B1 ranges


def per_seed(d):
    return pd.DataFrame({t: metrics(g.label, g.p) for t, g in d.groupby("tag")}).T


def main():
    res = {}
    cells = pd.read_csv(HERE.parents[1] / "splits" / "cells.csv", index_col="idx")
    b1 = load_preds("E1_danet", "m3").assign(seed=lambda d: d.tag.str.extract(r"m3_s(\d)").astype(int))
    h = load_preds("E4_colourjitter", "m3").assign(seed=lambda d: d.tag.str.extract(r"m3_s(\d)").astype(int))
    c = load_preds("E4_colourjitter", "leaky")
    c = c[c.tag.str.startswith("leaky_s42")]
    m2 = pd.read_csv(HERE.parents[2] / "milestone2-baseline" / "outputs" / "test_predictions.csv")
    b0 = pd.DataFrame(dict(idx=m2.idx, patient=cells.patient[m2.idx].values, label=m2.label, p=m2.p_parasitized))

    rows = {"B0": metrics(b0.label, b0.p), "C": metrics(c.label, c.p)}
    for name, d in [("B1", b1), ("H", h)]:
        ps = per_seed(d)
        res[f"{name}_per_seed"] = ps.round(3).to_dict(orient="index")
        rows[name] = ps.mean().to_dict()
        rows[f"{name}_sd"] = ps.std().to_dict()
    res["table"] = pd.DataFrame(rows).T.round(3).to_dict(orient="index")

    for metric in ["SE", "BACC", "AUC"]:
        d = paired_diff(h, b1, metric, strata="seed")
        d["predicted_range"] = PRED[metric]
        res[f"H_minus_B1_{metric}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    for metric in ["BACC", "SE", "SP"]:
        res[f"C_minus_B0_{metric}"] = {k: round(v, 4) for k, v in paired_diff(c, b0, metric).items()}
        res[f"B1_minus_B0_{metric}_unpaired"] = round(rows["B1"][metric] - rows["B0"][metric], 3)

    h_mean, b1_mean = rows["H"], rows["B1"]
    res["decision"] = dict(
        falsified=bool(h_mean["SE"] <= b1_mean["SE"] and h_mean["BACC"] <= b1_mean["BACC"]),
        rule="falsified if H <= B1 on SE and BACC averaged over the 3 splits (M3, 2026-10-05)",
        between_split_sd_shrinks=bool(rows["H_sd"]["BACC"] < rows["B1_sd"]["BACC"]),
        control_within_0p3=bool(abs(res["C_minus_B0_BACC"]["diff"]) <= 0.3))
    json.dump(res, open(HERE / "results_m3_rows.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
