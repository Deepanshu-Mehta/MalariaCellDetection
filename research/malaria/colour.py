"""Reinhard LAB colour transfer between patients (foreground-only, background re-zeroed).

Factored out of E2's analysis_stainswap.py so E4b's training-time augmentation and E2's test-time
counterfactual use the exact same transform.
"""
import cv2
import numpy as np


def lab_stats(img, fg):
    """img: HxWx3 uint8 RGB. fg: HxW bool foreground mask. Returns (mean, std) of LAB over foreground pixels."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[fg]
    return px.mean(0), px.std(0)


def reinhard_transfer(img, src_stats, tgt_stats, fg):
    """Recolour `img`'s foreground from src_stats to tgt_stats in LAB space; background forced to 0."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    sm, ss = src_stats
    tm, ts = tgt_stats
    out = (lab - sm) / np.maximum(ss, 1e-3) * ts + tm
    out = np.clip(out, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    rgb[~fg] = 0
    return rgb


def patient_lab_stats(cells, root, read_rgb, foreground_mask, patients=None):
    """Mean foreground LAB stats PER PATIENT (averaged over a sample of that patient's cells), used as the
    colour-transfer target distribution. `patients`: restrict to this subset (e.g. training patients only,
    to avoid leaking test-patient colour information)."""
    stats = {}
    pats = patients if patients is not None else cells.patient.unique()
    for p in pats:
        sub = cells[cells.patient == p]
        sample = sub.sample(min(len(sub), 10), random_state=0)
        means, stds = [], []
        for rel in sample.rel_path:
            img = read_rgb(root / rel)
            fg = foreground_mask(img)
            if fg.sum() < 10:
                continue
            m, s = lab_stats(img, fg)
            means.append(m); stds.append(s)
        if means:
            stats[p] = (np.mean(means, axis=0), np.mean(stds, axis=0))
    return stats
