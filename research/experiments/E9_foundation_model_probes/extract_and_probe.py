"""E9: frozen foundation-model probes on NIH's patient-grouped folds.

    arch -arm64 .venv/bin/python research/experiments/E9_foundation_model_probes/extract_and_probe.py

For each available backbone (see backbones.py): extract a frozen embedding for every cell once (cached to
disk, independent of split -- these are frozen features, not OOF-per-fold like E1/E8's trained models), then
for grouped repeat 0 run a logistic-regression + kNN probe per fold at label fractions {0.01, 0.1, 1.0} of
the training patients' cells, pooled into out-of-fold predictions.

Chance / floor comparators (already measured, not recomputed here): colour-only classifier 63.98% BACC (E2A),
DANet fine-tuned 96.71% BACC, ResNet-18 fine-tuned 96.58% BACC (E1/E1r, grouped repeat 0).
"""
import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.malaria import data, metrics  # noqa: E402
from research.experiments.E9_foundation_model_probes.backbones import available_backbones  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
FRACTIONS = [0.01, 0.1, 1.0]
CACHE = HERE / "features"


def _resize(arr):
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return torch.nn.functional.interpolate(t[None], size=(224, 224), mode="bilinear",
                                           align_corners=False, antialias=True)[0]


def load_packed_images():
    """Same packed-pickle format train.py reads (research/scripts/pack_images.py): {rel_path: [...], png: [...]}."""
    packed = os.environ.get("MALARIA_PACKED") or (next(Path("/kaggle/input").rglob("cells_png.pkl"), None)
                                                  if Path("/kaggle/input").exists() else None)
    if not packed:
        return None
    blob = pickle.load(open(packed, "rb"))
    return dict(zip(blob["rel_path"], blob["png"]))


def to_batch(paths_root, rel_paths, packed=None):
    imgs = []
    for p in rel_paths:
        if packed is not None:
            arr = np.asarray(Image.open(io.BytesIO(packed[p])).convert("RGB"))
        else:
            arr = data.read_rgb(paths_root / p)
        imgs.append(_resize(arr))
    return torch.stack(imgs)


def extract(name, embed_fn, cells, root, packed=None):
    cache_file = CACHE / f"{name}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    CACHE.mkdir(parents=True, exist_ok=True)
    feats = []
    for i in range(0, len(cells), 128):
        batch = to_batch(root, cells.rel_path.iloc[i:i + 128], packed)
        feats.append(embed_fn(batch))
        if i % (128 * 20) == 0:
            print(f"  {name}: {i}/{len(cells)}", flush=True)
    feats = np.concatenate(feats)
    np.save(cache_file, feats)
    return feats


def probe_fold(feats, cells, train_idx, test_idx, frac, seed):
    rng = np.random.default_rng(seed)
    if frac < 1.0:
        keep = cells.loc[train_idx].groupby("patient", group_keys=False) \
                                   .apply(lambda g: g.sample(max(1, int(round(frac * len(g)))), random_state=seed))
        train_idx = keep.index.values
    sc = StandardScaler().fit(feats[np.searchsorted(cells.index.values, train_idx)])
    xt = sc.transform(feats[np.searchsorted(cells.index.values, train_idx)])
    xv = sc.transform(feats[np.searchsorted(cells.index.values, test_idx)])
    yt = cells.loc[train_idx, "label"].values
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(xt, yt)
    knn = KNeighborsClassifier(15).fit(xt, yt)
    p_lr = lr.predict_proba(xv)[:, 1]
    p_knn = knn.predict_proba(xv)[:, 1] if knn.classes_.shape[0] == 2 else (knn.predict(xv) == 1).astype(float)
    return p_lr, p_knn, len(train_idx)


def main():
    cells = pd.read_csv(ROOT / "research" / "splits" / "cells.csv", index_col="idx")
    folds = pd.read_csv(ROOT / "research" / "splits" / "grouped_folds.csv")
    folds = folds[folds.repeat == 0]
    packed = load_packed_images()
    root = None if packed is not None else data.image_root()
    print("image source:", "packed pickle" if packed is not None else root)

    backbones = available_backbones(DEVICE)
    json.dump({k: v[1] for k, v in backbones.items()}, open(HERE / "available_backbones.json", "w"))
    if not backbones:
        print("no backbones available; nothing to probe"); return

    results = {}
    for name, (embed_fn, dim) in backbones.items():
        print(f"=== {name} (dim={dim}) ===")
        feats = extract(name, embed_fn, cells, root, packed)
        results[name] = {}
        for frac in FRACTIONS:
            oof_lr, oof_knn = np.full(len(cells), np.nan), np.full(len(cells), np.nan)
            n_train_total = 0
            for fold in range(5):
                role = folds[folds.fold == fold].set_index("patient").role
                mask = cells.patient.map(role)
                train_idx = cells.index[mask.isin(["train", "val"])].values
                test_idx = cells.index[mask.eq("test")].values
                p_lr, p_knn, n_tr = probe_fold(feats, cells, train_idx, test_idx, frac, seed=fold)
                pos = np.searchsorted(cells.index.values, test_idx)
                oof_lr[pos], oof_knn[pos] = p_lr, p_knn
                n_train_total += n_tr
            assert not np.isnan(oof_lr).any()
            results[name][f"frac_{frac}"] = dict(
                n_train_cells_total=n_train_total,
                lr=metrics.binary_metrics(cells.label.values, oof_lr),
                knn=metrics.binary_metrics(cells.label.values, oof_knn))
            print(f"  frac={frac}: LR BACC={results[name][f'frac_{frac}']['lr']['BACC']:.2f}  "
                 f"kNN BACC={results[name][f'frac_{frac}']['knn']['BACC']:.2f}")
    json.dump(results, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
