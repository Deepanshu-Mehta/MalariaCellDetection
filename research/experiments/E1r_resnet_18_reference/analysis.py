"""E1r analysis: ImageNet ResNet-18 vs DANet on identical folds, and ResNet-18's own leaky-vs-grouped gap.

    arch -arm64 .venv/bin/python research/experiments/E1r_resnet_18_reference/analysis.py

grouped: pooled out-of-fold predictions per repeat (all 27,558 cells once), mean +/- sd over 3 repeats.
leaky:   3 random cell splits (seeds 42-44). Leaky seeds were added after E1 showed a small DANet gap.
Paired DANet - ResNet difference: same cells, patients resampled within each repeat (B = 5000).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from research.malaria.stats import load_preds, metrics, paired_diff  # noqa: E402


def grouped_oof(config):
    d = load_preds(config, "grouped")
    d["repeat"] = d.tag.str.extract(r"grouped_r(\d)").astype(int)
    for r, g in d.groupby("repeat"):
        assert g.idx.is_unique and len(g) == 27558, f"{config} repeat {r}: {len(g)} cells"
    return d


def summary(d, key):
    t = pd.DataFrame({k: metrics(g.label, g.p) for k, g in d.groupby(key)}).T
    return dict(mean=t.mean().round(3).to_dict(), sd=t.std().round(3).to_dict(), per=t.round(3).to_dict(orient="index"))


def main():
    res = {}
    for name, cfg in [("resnet18", "E1r_resnet18"), ("danet", "E1_danet")]:
        g = grouped_oof(cfg)
        lk = load_preds(cfg, "leaky")
        lk = lk[lk.tag.str.match(r"leaky_s4[234]_t0")]
        if name == "danet":
            lk = lk[~lk.tag.isin([])]
        res[name] = dict(grouped=summary(g, "repeat"), leaky=summary(lk, "tag"))
        res[name]["gap_leaky_minus_grouped"] = {m: round(res[name]["leaky"]["mean"][m] - res[name]["grouped"]["mean"][m], 3)
                                                for m in ["BACC", "SE", "SP", "AUC"]}
        res[name]["_grouped_frame"] = g
    for m in ["BACC", "SE", "SP", "AUC"]:
        res[f"danet_minus_resnet_grouped_{m}"] = {k: round(v, 4) for k, v in paired_diff(
            res["danet"]["_grouped_frame"], res["resnet18"]["_grouped_frame"], m, strata="repeat").items()}
    for name in ["resnet18", "danet"]:
        del res[name]["_grouped_frame"]
    json.dump(res, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
