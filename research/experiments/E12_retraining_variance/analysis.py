"""E12: seed variance on MP-IDB and NIH, and the unused checkpoints.

    arch -arm64 .venv/bin/python -m research.experiments.E12_retraining_variance.analysis
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats as st

from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.experiments.E6b_five_checkpoint_external_rescore.analysis import fast_metrics
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import SCORES as E6B, scored_table

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs_kaggle"
SCORES = HERE / "scores"
FOLDS, SEEDS = (2, 3), (1, 2, 3, 4)
E1_TOTAL_SD = 8.407


def summary(vals):
    v = np.array(vals)
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)), lo=float(v.min()), hi=float(v.max()))


def pooled(by_fold):
    """Pooled within-fold SD with a chi-square 95% CI."""
    df = sum(len(v) - 1 for v in by_fold.values())
    ss = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in by_fold.values())
    return dict(sd=float(np.sqrt(ss / df)), df=df, ci=[float(np.sqrt(ss / st.chi2.ppf(.975, df))), float(np.sqrt(ss / st.chi2.ppf(.025, df)))])


def unused(y):
    O = {}
    for name, keys in (("E1_r1r2", [f"E1_r{r}f{f}_t0" for r in (1, 2) for f in range(5)]),
                       ("ResNet18_r0r2", [f"E1r_r{r}f{f}_t0" for r in range(3) for f in range(5)])):
        m = [fast_metrics(y, np.load(SCORES / f"{k}.npy")) for k in keys]
        O[name] = {k: summary([x[k] for x in m]) for k in ("BACC", "AUC", "SE", "SP")}
    b, a = O["E1_r1r2"]["BACC"], O["ResNet18_r0r2"]["BACC"]
    O["prediction4"] = dict(E1_mean_70_76=bool(70 <= b["mean"] <= 76), E1_sd_at_least_5=bool(b["sd"] >= 5),
                            ResNet_sd_at_least_3=bool(a["sd"] >= 3), ResNet_mean_below_80=bool(a["mean"] < 80))
    return O


def seeds(y):
    root = RUNS / "e10-e11-e12-batch" / "runs" / "E1_danet"
    ext, nih = {"BACC": {}, "AUC": {}}, {}
    per = {}
    for f in FOLDS:
        rows = [("t0", fast_metrics(y, np.load(E6B / f"A_E1_f{f}.npy")), json.load(open(ckpt_path(ARMS["E1"], f).parent / "metrics.json"))["cell"])]
        for s in SEEDS:
            hit = next(root.glob(f"grouped_r0f{f}_t{s}*/metrics.json"))
            rows.append((f"t{s}", fast_metrics(y, np.load(SCORES / f"E1_r0f{f}_t{s}.npy")), json.load(open(hit))["cell"]))
        per[f] = [dict(seed=k, MP=m, NIH_BACC=c["BACC"]) for k, m, c in rows]
        for k in ext:
            ext[k][f] = [m[k] for _, m, _ in rows]
        nih[f] = [c["BACC"] for _, _, c in rows]
    O = dict(per_seed=per, pooled_MP_BACC=pooled(ext["BACC"]), pooled_MP_AUC=pooled(ext["AUC"]), pooled_NIH_BACC=pooled(nih))
    sd = O["pooled_MP_BACC"]["sd"]
    O["ratio_NIH_to_MP"] = O["pooled_NIH_BACC"]["sd"] / sd
    O["seed_share_of_E1_total_variance"] = sd ** 2 / E1_TOTAL_SD ** 2
    O["fold2_seed0_minus_mean_seeds1to4"] = ext["BACC"][2][0] - float(np.mean(ext["BACC"][2][1:]))
    O["prediction1_BACC_sd_3_to_6"] = bool(3 <= sd <= 6)
    O["prediction2"] = bool(O["pooled_NIH_BACC"]["sd"] < .5 and O["ratio_NIH_to_MP"] <= .2)
    O["prediction3"] = bool(O["fold2_seed0_minus_mean_seeds1to4"] >= 3)
    return O


def main():
    y = scored_table().binary_label.values
    O = dict(unused=unused(y))
    if all((SCORES / f"E1_r0f{f}_t{s}.npy").exists() for f in FOLDS for s in SEEDS):
        O["seeds"] = seeds(y)
    json.dump(O, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    main()
