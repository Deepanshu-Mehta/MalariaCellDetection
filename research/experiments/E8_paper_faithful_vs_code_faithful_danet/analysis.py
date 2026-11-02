"""E8 analysis: does the norm layer (and the paper's other block details) change DANet's leakage gap?

    arch -arm64 .venv/bin/python research/experiments/E8_paper_faithful_vs_code_faithful_danet/analysis.py

Arms (all 50 epochs):
  E1  code-faithful DANet, InstanceNorm              (E1_danet)
  E8a code-faithful DANet, LayerNorm only            (E8a_layernorm)
  E8b paper-faithful: sigmoid gate, parallel, LN     (E8_paper_faithful)
grouped: repeat 0 folds 0-4 pooled into OOF predictions (all 27,558 cells once), identical folds for every arm.
leaky:   seeds 42-44. Gap = mean leaky BACC - grouped BACC.
Paired arm - E1 differences on identical grouped cells: patient-cluster bootstrap (B = 5000).
Gap differences use an unpaired bootstrap over leaky seeds and grouped patients (approximate).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from research.malaria.stats import load_preds, metrics, paired_diff  # noqa: E402

import os
ARMS = {"E1 (InstanceNorm)": "E1_danet", "E8b (paper-faithful)": "E8_paper_faithful"}
# E8a (LayerNorm only) is invalid: 7/8 runs never left chance (val loss = ln 2), see README.
# E8c (paper-faithful with InstanceNorm) isolates the norm layer against E8b; added when its runs exist.
if os.environ.get("E8C"):
    ARMS["E8c (paper-faithful, InstanceNorm)"] = "E8c_paper_instancenorm"


def grouped_r0(cfg):
    d = load_preds(cfg, "grouped")
    d = d[d.tag.str.startswith("grouped_r0")]
    assert d.idx.is_unique and len(d) == 27558, f"{cfg}: repeat-0 OOF has {len(d)} cells"
    return d


def gap_bootstrap(lk, g, B=2000, seed=0):
    """Approximate CI of (mean leaky BACC - grouped BACC): resample leaky seeds' test patients and grouped
    patients independently."""
    rng = np.random.default_rng(seed)
    gpat = {p: x for p, x in g.groupby("patient")}
    lpat = {t: {p: x for p, x in d.groupby("patient")} for t, d in lk.groupby("tag")}
    vals = []
    for _ in range(B):
        gk = np.array(list(gpat))
        gs = pd.concat([gpat[k] for k in rng.choice(gk, len(gk))])
        ls = []
        for t, pats in lpat.items():
            lk_keys = np.array(list(pats))
            s = pd.concat([pats[k] for k in rng.choice(lk_keys, len(lk_keys))])
            ls.append(metrics(s.label, s.p)["BACC"])
        vals.append(np.mean(ls) - metrics(gs.label, gs.p)["BACC"])
    return np.array(vals)


def main():
    res, boots, grouped = {}, {}, {}
    for arm, cfg in ARMS.items():
        g = grouped_r0(cfg)
        lk = load_preds(cfg, "leaky")
        lk = lk[lk.tag.str.match(r"leaky_s4[234]_t0")]
        assert lk.tag.nunique() == 3, f"{cfg}: {lk.tag.nunique()} leaky seeds"
        gm = metrics(g.label, g.p)
        lm = pd.DataFrame({t: metrics(x.label, x.p) for t, x in lk.groupby("tag")}).T
        boots[arm] = gap_bootstrap(lk, g)
        res[arm] = dict(grouped_r0=gm, leaky_mean=lm.mean().to_dict(), leaky_sd=lm.std().to_dict(),
                        gap_BACC=float(lm.BACC.mean() - gm["BACC"]),
                        gap_ci=np.percentile(boots[arm], [2.5, 97.5]).tolist())
        grouped[arm] = g
    base = "E1 (InstanceNorm)"
    for arm in list(ARMS)[1:]:
        res[f"{arm} - E1, grouped (paired)"] = {m: paired_diff(grouped[arm], grouped[base], m) for m in ["BACC", "SE", "SP"]}
        diff = boots[arm] - boots[base]
        res[f"{arm} - E1, gap difference"] = dict(diff=res[arm]["gap_BACC"] - res[base]["gap_BACC"],
                                                  ci=np.percentile(diff, [2.5, 97.5]).tolist())
    ln, inorm = "E8b (paper-faithful)", "E8c (paper-faithful, InstanceNorm)"
    if inorm in ARMS:                                  # the clean norm-layer contrast: LayerNorm - InstanceNorm
        res["E8b - E8c (LN - IN), grouped (paired)"] = {m: paired_diff(grouped[ln], grouped[inorm], m)
                                                        for m in ["BACC", "SE", "SP"]}
        diff = boots[ln] - boots[inorm]
        res["E8b - E8c (LN - IN), gap difference"] = dict(diff=res[ln]["gap_BACC"] - res[inorm]["gap_BACC"],
                                                          ci=np.percentile(diff, [2.5, 97.5]).tolist())
    json.dump(res, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
