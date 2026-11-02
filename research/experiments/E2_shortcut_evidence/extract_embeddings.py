"""E2 step 1: extract frozen penultimate-layer embeddings for every cell, using each fold's own
checkpoint on its own out-of-fold test cells (repeat 0 only, so E1/E8b/E8c are on identical folds).

    arch -arm64 .venv/bin/python research/experiments/E2_shortcut_evidence/extract_embeddings.py

For each arm, writes research/experiments/E2_shortcut_evidence/embeddings/<arm>.npz:
    idx (N,), patient (N,), label (N,), emb (N, 128)  -- DANet's pool output (16*n = 128 for n=8)
Embeddings come from the OOF model for that cell (as in E1's analysis), so a cell's embedding never
came from a model that trained on it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.malaria import data, models  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
ARMS = {
    "E1": dict(job="e1-danet-grid", fallback="danet-phase0-timing", config="E1_danet",
              model=dict(model="danet", gate="softmax", dilation_mode="cascade", norm="instance")),
    "E8b": dict(job="e8-norm-layer", config="E8_paper_faithful",
               model=dict(model="danet", gate="sigmoid", dilation_mode="parallel", norm="layer")),
    "E8c": dict(job="e8c-paper-instancenorm", config="E8c_paper_instancenorm",
               model=dict(model="danet", gate="sigmoid", dilation_mode="parallel", norm="instance")),
}
OUT = HERE / "embeddings"


def ckpt_path(arm_cfg, fold):
    for job in [arm_cfg["job"], arm_cfg.get("fallback")]:
        if job is None:
            continue
        p = ROOT / "research" / "runs_kaggle" / job / "runs" / arm_cfg["config"] / f"grouped_r0f{fold}_t0"
        if (p / "best.pth").exists():
            return p / "best.pth"
    raise FileNotFoundError(f"no checkpoint for {arm_cfg['config']} fold {fold}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "research" / "splits" / "cells.csv", index_col="idx")
    folds = pd.read_csv(ROOT / "research" / "splits" / "grouped_folds.csv")
    folds = folds[(folds.repeat == 0) & (folds.role == "test")]
    root = data.image_root()

    def to_tensor(j):
        t = torch.from_numpy(data.read_rgb(root / cells.rel_path[j])).permute(2, 0, 1).float() / 255.0
        return torch.nn.functional.interpolate(t[None], size=(224, 224), mode="bilinear",
                                               align_corners=False, antialias=True)[0]

    def release(model):
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    for arm, cfg in ARMS.items():
        out_file = OUT / f"{arm}.npz"
        if out_file.exists():
            print(f"{arm}: already extracted, skipping"); continue
        fold_dir = OUT / f"_{arm}_folds"
        fold_dir.mkdir(exist_ok=True)
        for fold in range(5):
            fold_file = fold_dir / f"fold{fold}.npz"
            if fold_file.exists():
                print(f"{arm} fold {fold}: already done, skipping"); continue
            test_patients = set(folds[folds.fold == fold].patient)
            sub = cells[cells.patient.isin(test_patients)]
            model = models.build_model(cfg["model"]).to(DEVICE).eval()
            model.load_state_dict(torch.load(ckpt_path(cfg, fold), map_location=DEVICE))
            embs = []
            with torch.no_grad():
                for i in range(0, len(sub), 256):
                    batch_idx = sub.index[i:i + 256]
                    imgs = torch.stack([to_tensor(j) for j in batch_idx])
                    embs.append(model.embed(imgs.to(DEVICE)).cpu().numpy())
            np.savez_compressed(fold_file, idx=sub.index.values, emb=np.concatenate(embs))
            release(model)
            print(f"{arm} fold {fold}: {len(sub)} cells", flush=True)
        parts = [np.load(fold_dir / f"fold{f}.npz") for f in range(5)]
        idx = np.concatenate([p["idx"] for p in parts])
        emb = np.concatenate([p["emb"] for p in parts])
        order = np.argsort(idx)
        idx, emb = idx[order], emb[order]
        assert (idx == cells.index.values).all(), "embeddings do not cover every cell exactly once"
        np.savez_compressed(out_file, idx=idx, patient=cells.patient.values, label=cells.label.values, emb=emb)
        for p in parts:
            p.close()
        import shutil
        shutil.rmtree(fold_dir)
        print(f"{arm}: saved {emb.shape} -> {out_file}")


if __name__ == "__main__":
    main()
