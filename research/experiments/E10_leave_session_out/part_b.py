"""E10 part B: pooled out-of-fold BACC, patient-grouped minus session-grouped, paired by cell, patient-cluster bootstrap.

    arch -arm64 .venv/bin/python -m research.experiments.E10_leave_session_out.part_b
"""
import json
from pathlib import Path

import pandas as pd

from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.malaria.stats import load_preds, metrics, paired_diff

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
MDE = 0.29


def main():
    grouped = pd.concat([pd.read_csv(ckpt_path(ARMS["E1"], f).parent / "test_predictions.csv") for f in range(5)])
    session = load_preds("E1_danet", "session_r0f*")
    assert session.tag.nunique() == 5 and session.idx.is_unique and len(session) == len(grouped) == 27558
    folds = pd.read_csv(R / "splits/session_folds.csv")
    pos = pd.read_csv(R / "splits/cells.csv", index_col="idx").groupby("patient").label.sum()
    healthy = set(pos.index[pos == 0])
    O = dict(grouped=metrics(grouped.label, grouped.p), session=metrics(session.label, session.p),
             per_fold={t: metrics(g.label, g.p) for t, g in session.groupby("tag")})
    O["gap"] = {m: paired_diff(grouped, session, m, B=2000) for m in ("BACC", "SE", "SP", "AUC")}
    O["healthy_only_fp_pct"] = {n: 100 * float((d[d.patient.isin(healthy)].p >= .5).mean()) for n, d in (("grouped", grouped), ("session", session))}
    g = O["gap"]["BACC"]
    O["rule_session_leakage_real"] = bool(g["diff"] >= MDE and g["lo"] > 0)
    O["prediction3_range_0_to_0.8"] = bool(0 <= g["diff"] <= .8)
    assert (folds.groupby("fold").role.value_counts().unstack().sum(axis=1) == 200).all()
    json.dump(O, open(HERE / "part_b_results.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    main()
