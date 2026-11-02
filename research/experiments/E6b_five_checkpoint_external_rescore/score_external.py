"""Score MP-IDB crops for E6b and E6c. Resume-safe: one .npy per condition/arm/fold, finished ones are skipped.

    caffeinate -i arch -arm64 .venv/bin/python -m research.experiments.E6b_five_checkpoint_external_rescore.score_external --cond A

Conditions: A baseline checkpoints on raw crops; B E4b checkpoints on raw crops; C baseline checkpoints on
crops Reinhard-normalised to that fold's non-test NIH patients; D sensitivity rerun of C using within-cell
source statistics.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.experiments.E5_explanation_validity.analysis import to_tensor
from research.malaria import colour, data, models

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
CELLS_DIR = RESEARCH / "data_external" / "mpidb_pf_cells"
SCORES = HERE / "scores"
CACHE = Path(os.environ.get("E6_CACHE", HERE / "_cache"))
DEVICE = torch.device(os.environ.get("E6_DEVICE", "mps"))
SCALE = 65535.0  # uint16 keeps the resized float pixels to 1/131070, so E6's scores reproduce
E4B = {"E1": "e1", "E8b": "e8b", "E8c": "e8c"}
N_FOLDS = 5


def scored_table():
    t = pd.read_csv(CELLS_DIR / "mpidb_cells.csv")
    s = t[t.label.notna()].reset_index(drop=True)
    s["binary_label"] = (s.label == "Parasitized").astype(int)
    return s


def checkpoint(cond, arm, fold):
    if cond in ("A", "C", "D"):
        return ckpt_path(ARMS[arm], fold)
    name = f"E4b_{E4B[arm]}_colourtransfer"
    hits = list((RESEARCH / "runs_kaggle" / "e4b-colour-transfer" / "runs" / name).glob(f"grouped_r0f{fold}_t0*/best.pth"))
    assert len(hits) == 1, (name, fold, hits)
    run_cfg = json.load(open(hits[0].parent / "config.json"))["cfg"]
    assert run_cfg.get("colour_transfer") is True
    for k in ("gate", "dilation_mode", "norm"):
        assert run_cfg[k] == ARMS[arm]["model"][k], (name, k)
    return hits[0]


def write_images(table, path, transform=None):
    mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(table), 3, 224, 224))
    for i, r in enumerate(table.itertuples()):
        arr = np.array(Image.open(CELLS_DIR / r.file).convert("RGB"))
        if transform is not None:
            arr = transform(arr, r.image)
        mm[i] = np.round(to_tensor(arr).numpy() * SCALE).astype(np.uint16)
    mm.flush()
    del mm


def fast_norms(model):
    """InstanceNorm2d(affine) equals GroupNorm(C, C) but runs about 9x faster on MPS. Inference only."""
    for parent in list(model.modules()):
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.InstanceNorm2d):
                gn = torch.nn.GroupNorm(child.num_features, child.num_features, eps=child.eps)
                gn.load_state_dict(child.state_dict())
                setattr(parent, name, gn)
    return model


def score(ckpt, cfg, path, n):
    m = models.build_model(cfg).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = fast_norms(m).to(DEVICE)
    mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(n, 3, 224, 224))
    out = []
    with torch.inference_mode():
        for i in range(0, n, 64):
            x = torch.from_numpy(mm[i:i + 64].astype(np.float32) / SCALE).to(DEVICE)
            out.append(torch.exp(m(x))[:, 1].cpu().numpy())
    del m, mm
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(out)


def field_stats(table):
    """Foreground LAB mean/std pooled over all scored crops from each MP-IDB field image, ignoring labels."""
    px = {}
    for r in table.itertuples():
        arr = np.array(Image.open(CELLS_DIR / r.file).convert("RGB"))
        fg = data.foreground_mask(arr)
        if fg.sum() >= 10:
            lab = colour.cv2.cvtColor(arr, colour.cv2.COLOR_RGB2LAB).astype(np.float32)[fg]
            px.setdefault(r.image, []).append(lab[::4])
    return {k: (np.concatenate(v).mean(0), np.concatenate(v).std(0)) for k, v in px.items()}


def field_stats_within(table):
    """Sensitivity for E6c: mean of per-crop LAB stats, the same definition as the NIH targets."""
    st = {}
    for r in table.itertuples():
        arr = np.array(Image.open(CELLS_DIR / r.file).convert("RGB"))
        fg = data.foreground_mask(arr)
        if fg.sum() >= 10:
            st.setdefault(r.image, []).append(colour.lab_stats(arr, fg))
    return {k: (np.mean([a for a, _ in v], axis=0), np.mean([b for _, b in v], axis=0)) for k, v in st.items()}


def fold_targets():
    cells = pd.read_csv(RESEARCH / "splits" / "cells.csv", index_col="idx")
    folds = pd.read_csv(RESEARCH / "splits" / "grouped_folds.csv")
    folds = folds[(folds.repeat == 0) & (folds.role == "test")].set_index("patient").fold
    stats = colour.patient_lab_stats(cells, data.image_root(), data.read_rgb, data.foreground_mask)
    out = {}
    for f in range(N_FOLDS):
        keep = [p for p in stats if folds[p] != f]
        out[f] = tuple(np.mean([stats[p][i] for p in keep], axis=0) for i in (0, 1))
    return out


def run(cond):
    SCORES.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    table = scored_table()
    n = len(table)
    assert n == 14468
    todo = [(a, f) for a in ARMS for f in range(N_FOLDS) if not (SCORES / f"{cond}_{a}_f{f}.npy").exists()]
    if not todo:
        print(cond, "already complete", flush=True)
        return

    raw = CACHE / "raw.u16"
    if cond in ("A", "B") and not raw.exists():
        t0 = time.time()
        tmp = raw.with_suffix(".tmp")
        write_images(table, tmp)
        tmp.rename(raw)
        print(f"cached raw crops in {time.time() - t0:.0f}s", flush=True)

    src = tgt = None
    if cond in "CD":
        src, tgt = (field_stats if cond == "C" else field_stats_within)(table), fold_targets()
        print(f"colour stats ready: {len(src)} fields", flush=True)

    for f in sorted({f for _, f in todo}):
        path = raw
        if cond in "CD":
            path = CACHE / "normalised.u16"

            def norm(arr, image, fold=f):
                fg = data.foreground_mask(arr)
                return colour.reinhard_transfer(arr, src[image], tgt[fold], fg) if fg.sum() >= 10 else arr

            t0 = time.time()
            write_images(table, path, norm)
            print(f"fold {f}: normalised crops in {time.time() - t0:.0f}s", flush=True)
        for a in [a for a, g in todo if g == f]:
            t0 = time.time()
            p = score(checkpoint(cond, a, f), ARMS[a]["model"], path, n)
            tmp = SCORES / f"{cond}_{a}_f{f}.tmp.npy"
            np.save(tmp, p.astype(np.float32))
            tmp.rename(SCORES / f"{cond}_{a}_f{f}.npy")
            print(f"{cond} {a} fold {f}: {time.time() - t0:.0f}s", flush=True)
    print(cond, "done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list("ABCD"))
    run(ap.parse_args().cond)
