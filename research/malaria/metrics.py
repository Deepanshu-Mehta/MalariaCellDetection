"""Cell-pooled and patient-macro metrics. Parasitized = positive class (label 1)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y, p, bins=15):
    """Expected calibration error of the predicted class probability."""
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def binary_metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    out = dict(ACC=100 * (tp + tn) / len(y), BACC=100 * (se + sp) / 2, SE=100 * se, SP=100 * sp,
               PR=100 * tp / max(tp + fp, 1), ECE=ece(y, p), n=len(y), TP=tp, TN=tn, FP=fp, FN=fn)
    if len(np.unique(y)) == 2:
        out.update(ROC_AUC=float(roc_auc_score(y, p)), PR_AUC=float(average_precision_score(y, p)))
    return out


def patient_macro(pred_df, thr=0.5):
    """Per-patient SE / SP, then averaged over patients that have that class.
    Healthy-only patients contribute to SP only."""
    rows = []
    for pid, g in pred_df.groupby("patient"):
        pred = (g.p >= thr).astype(int)
        pos, neg = g.label == 1, g.label == 0
        rows.append(dict(patient=pid,
                         se=(pred[pos] == 1).mean() if pos.any() else np.nan,
                         sp=(pred[neg] == 0).mean() if neg.any() else np.nan,
                         healthy_only=not pos.any()))
    t = pd.DataFrame(rows)
    return dict(SE_patient_macro=100 * t.se.mean(), SP_patient_macro=100 * t.sp.mean(),
                SP_healthy_only=100 * t[t.healthy_only].sp.mean(),
                SP_infected_patients=100 * t[~t.healthy_only].sp.mean(),
                n_patients=len(t))
