"""E6: how much does DANet's performance drop on MP-IDB (external, out-of-domain)?

    arch -arm64 .venv/bin/python research/experiments/E6_external_test/analysis.py

Scores E1/E8b/E8c (grouped repeat-0 fold-0 checkpoints -- none of them ever trained on MP-IDB) on all 14,468
MP-IDB crops (963 Parasitized + 13,505 Uninfected) ONCE each, per the registered protocol ("score each test
set once"). Compares the drop against each arm's own NIH grouped-repeat-0 level (E1's analysis.py / E8's
results.json), with a patient-- here, image-level cluster bootstrap CI (MP-IDB crops are grouped by source
image, the nearest available unit to "patient" for this external set).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
PKG_ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG_ROOT))
from research.malaria import models  # noqa: E402
from research.malaria.stats import metrics  # noqa: E402
from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path  # noqa: E402
from research.experiments.E5_explanation_validity.analysis import to_tensor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CELLS_DIR = RESEARCH / "data_external" / "mpidb_pf_cells"
B = 5000


def load_model(cfg):
    m = models.build_model(cfg["model"]).to(DEVICE).eval()
    m.load_state_dict(torch.load(ckpt_path(cfg, fold=0), map_location=DEVICE))
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def image_cluster_bootstrap(df, stat, seed=0):
    rng = np.random.default_rng(seed)
    groups = {img: g for img, g in df.groupby("image")}
    keys = np.array(list(groups))
    vals = [stat(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5]).tolist()


def main():
    t = pd.read_csv(CELLS_DIR / "mpidb_cells.csv")
    scored = t[t.label.notna()].reset_index(drop=True)
    scored["binary_label"] = (scored.label == "Parasitized").astype(int)
    print(f"scoring {len(scored)} MP-IDB crops ({scored.binary_label.sum()} Parasitized, "
         f"{(1 - scored.binary_label).sum()} Uninfected)")

    # NIH grouped-repeat-0 levels, for the "drop from own NIH level" comparison (already measured, not rerun)
    nih_bacc = dict(E1=96.778,
                    E8b=json.load(open(RESEARCH / "experiments" / "E8_paper_faithful_vs_code_faithful_danet"
                                      / "results.json"))["E8b (paper-faithful)"]["grouped_r0"]["BACC"],
                    E8c=json.load(open(RESEARCH / "experiments" / "E8_paper_faithful_vs_code_faithful_danet"
                                      / "results.json"))["E8c (paper-faithful, InstanceNorm)"]["grouped_r0"]["BACC"])

    results = {}
    for arm, cfg in ARMS.items():
        model = load_model(cfg)
        ps = []
        with torch.no_grad():
            for i in range(0, len(scored), 128):
                batch = scored.iloc[i:i + 128]
                imgs = torch.stack([to_tensor(np.array(Image.open(CELLS_DIR / r.file).convert("RGB")))
                                    for r in batch.itertuples()])
                ps.append(torch.exp(model(imgs.to(DEVICE)))[:, 1].cpu().numpy())
        scored[f"p_{arm}"] = np.concatenate(ps)
        y, p = scored.binary_label.values, scored[f"p_{arm}"].values
        m = metrics(y, p)
        ci_bacc = image_cluster_bootstrap(scored.assign(label=y, p=p), lambda g: metrics(g.label, g.p)["BACC"])
        drop = nih_bacc[arm] - m["BACC"]
        results[arm] = dict(n=len(scored), **m, bacc_ci=ci_bacc,
                            nih_grouped_bacc=nih_bacc[arm], drop_from_nih=drop)
        print(f"{arm}: BACC={m['BACC']:.2f} [{ci_bacc[0]:.2f},{ci_bacc[1]:.2f}] SE={m['SE']:.2f} SP={m['SP']:.2f} "
             f"AUC={m['AUC']:.3f} | NIH grouped {nih_bacc[arm]:.2f} -> drop {drop:+.2f}")

    scored.to_csv(HERE / "mpidb_predictions.csv", index=False)
    json.dump(results, open(HERE / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
