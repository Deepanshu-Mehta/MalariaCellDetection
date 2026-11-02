"""E2 part C: patient-ID probe on frozen embeddings, within-patient split.

    arch -arm64 .venv/bin/python research/experiments/E2_shortcut_evidence/analysis_embeddings.py

For each arm (E1 = InstanceNorm/softmax/cascade, E8b = LayerNorm/sigmoid/parallel, E8c = InstanceNorm
with E8b's gate+dilations), split each patient's OOF-embedded cells 80/20, train a multinomial logistic
regression (on the 150 infected patients; healthy-only patients are a separate binary target in part B) to
predict patient ID from the 80%, evaluate on the held-out 20%. Chance is 1/150 = 0.67%.

This tests the E8 hypothesis directly: if InstanceNorm removes the stain fingerprint, E1 and E8c embeddings
should carry much less patient identity than E8b's.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ARMS = ["E1", "E8b", "E8c"]


def within_patient_probe(patient, emb, seed=0):
    """80/20 split of each patient's own cells; multinomial LR + 5-NN; returns held-out accuracy."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(patient, return_counts=True)
    keep = uniq[counts >= 5]                      # need enough cells per patient to split meaningfully
    mask = np.isin(patient, keep)
    patient, emb = patient[mask], emb[mask]
    tr_idx, te_idx = [], []
    for p in keep:
        idx = np.where(patient == p)[0]
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(idx)))
        tr_idx += list(idx[:cut]); te_idx += list(idx[cut:])
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    sc = StandardScaler().fit(emb[tr_idx])
    xt, xv = sc.transform(emb[tr_idx]), sc.transform(emb[te_idx])
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(xt, patient[tr_idx])
    knn = KNeighborsClassifier(5).fit(xt, patient[tr_idx])
    return dict(n_patients=len(keep), n_train=len(tr_idx), n_test=len(te_idx), chance=1 / len(keep),
               lr_acc=float(lr.score(xv, patient[te_idx])), knn_acc=float(knn.score(xv, patient[te_idx])))


def main():
    res = {}
    for arm in ARMS:
        d = np.load(HERE / "embeddings" / f"{arm}.npz", allow_pickle=True)  # patient id is a string/object array
        label, patient, emb = d["label"], d["patient"], d["emb"]
        para_count = {p: label[patient == p].sum() for p in np.unique(patient)}
        infected_mask = np.array([para_count[p] > 0 for p in patient])
        res[arm] = dict(
            patient_id_probe_infected_patients=within_patient_probe(patient[infected_mask], emb[infected_mask]),
            patient_id_probe_all_patients=within_patient_probe(patient, emb))
    json.dump(res, open(HERE / "results_embeddings.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
