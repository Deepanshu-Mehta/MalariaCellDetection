"""Candidate 3 with all 20 members on MP-IDB Cellpose crops, the reference row of the rehearsal table. Development data only.

    arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.rehearsal_cellpose --table
    E6_CACHE=<dir> caffeinate -i arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.score \
        --table research/experiments/E11_bbbc041_holdout/rehearsal_cellpose.csv \
        --crops research/data_external/<CELLS_DIR> --name rehearsal_cellpose --members resnet
    arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.rehearsal_cellpose
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.E6b_five_checkpoint_external_rescore.score_external import CELLS_DIR, scored_table
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import SCORES as E6B
from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS

from .calculate import auc, logit, rates

HERE = Path(__file__).resolve().parent
TABLE = HERE / "rehearsal_cellpose.csv"


def write_table():
    t = scored_table()
    pd.DataFrame(dict(crop_id=t.file.str.removesuffix(".png"), y=t.binary_label)).to_csv(TABLE, index=False)
    print(len(t), CELLS_DIR)


def main():
    t = pd.read_csv(TABLE)
    y = t.y.values == 1
    danet = [logit(np.load(E6B / f"B_{a}_f{k}.npy")) for a in ARMS for k in range(5)]
    resnet = [logit(np.load(HERE / "scores" / "rehearsal_cellpose" / f"cand3_resnet_f{k}.npy")) for k in range(5)]
    O = {}
    for name, members in (("15_members", danet), ("20_members", danet + resnet), ("resnet_only", resnet)):
        s = np.mean(members, axis=0)
        ts = np.quantile(s, np.linspace(0, 1, 2001))
        O[name] = dict(AUC=auc(y, s), **rates(y, s, 0.0), best_threshold_BACC=max(rates(y, s, x)["BACC"] for x in ts))
    json.dump(O, open(HERE / "results_rehearsal_cellpose.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    write_table() if "--table" in sys.argv else main()
