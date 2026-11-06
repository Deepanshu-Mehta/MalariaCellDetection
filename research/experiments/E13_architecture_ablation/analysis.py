"""E13: DANet without attention (E13a) and with a max-pool head (E13b), against E1 on the same NIH folds.

    arch -arm64 .venv/bin/python -m research.experiments.E13_architecture_ablation.analysis

NIH only (amended 2026-11-03, before any result). Primary: pooled out-of-fold BACC over all 27,558 cells of grouped
repeat 0, paired difference vs E1 r0 with the patient-cluster bootstrap (B = 5000), as in E1r. SE, SP, AUC alongside.
"""
import json
from pathlib import Path

from research.malaria.stats import load_preds, metrics, paired_diff

HERE = Path(__file__).resolve().parent
ARMS = {"E13a": "E13a_danet_noattention", "E13b": "E13b_danet_maxpool"}
EXPECT = {"E13a": dict(attention="none", head_pool="avg"), "E13b": dict(attention="dilated", head_pool="max")}


def r0(config):
    d = load_preds(config, "grouped")
    d = d[d.tag.str.match(r"grouped_r0f\d_t0($|_)")]
    assert d.idx.is_unique and len(d) == 27558, (config, len(d))
    return d


def check_configs():
    runs = HERE.parents[1] / "runs_kaggle" / "e13-architecture-ablation" / "runs"
    for arm, config in ARMS.items():
        dirs = sorted((runs / config).glob("grouped_r0f*_t0*"))
        assert len(dirs) == 5, (arm, dirs)
        for d in dirs:
            cfg = json.load(open(d / "config.json"))["cfg"]
            assert cfg["train_seed"] == 0 and cfg["epochs"] == 50, d
            for k, v in EXPECT[arm].items():
                assert cfg.get(k, {"attention": "dilated", "head_pool": "avg"}[k]) == v, (d, k)


def main():
    check_configs()
    base = r0("E1_danet")
    res = {"E1": metrics(base.label, base.p)}
    for arm, config in ARMS.items():
        d = r0(config)
        res[arm] = metrics(d.label, d.p)
        res[f"{arm}_minus_E1"] = {m: paired_diff(d, base, m) for m in ("BACC", "SE", "SP", "AUC")}
    a, b = res["E13a_minus_E1"], res["E13b_minus_E1"]
    res["predictions"] = {
        "P1_E13a_BACC_drop_0.5_to_2.5_CI_below_0": bool(-2.5 <= a["BACC"]["diff"] <= -0.5 and a["BACC"]["hi"] < 0),
        "P2_E13a_SE_falls_more_than_SP": bool(a["SE"]["diff"] < a["SP"]["diff"]),
        "P4_E13b_BACC_abs_below_0.29_CI_includes_0": bool(abs(b["BACC"]["diff"]) < 0.29 and b["BACC"]["lo"] <= 0 <= b["BACC"]["hi"]),
        "P5prime_E13b_SE_up_SP_down_each_under_1": bool(0 < b["SE"]["diff"] < 1 and -1 < b["SP"]["diff"] < 0),
        "P5prime_signs_only": bool(b["SE"]["diff"] > 0 and b["SP"]["diff"] < 0),
    }
    json.dump(res, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
