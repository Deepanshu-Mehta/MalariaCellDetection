"""Score E12's checkpoints on MP-IDB. Resume-safe: one .npy per checkpoint, finished ones are skipped.

    E6_CACHE=<dir> caffeinate -i arch -arm64 .venv/bin/python -m research.experiments.E12_retraining_variance.score_checkpoints --set unused

Sets: `unused` E1 repeats 1-2 and ResNet-18 repeats 0-2; `seeds` the E12 retrains. E6b's path: same crops, threshold-free.
The E1 r0f0 checkpoint is rescored first and must reproduce E6b's saved score.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import CELLS_DIR, SCALE, fast_norms, scored_table
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import SCORES as E6B_SCORES
from research.malaria import models

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
RUNS = RESEARCH / "runs_kaggle"
SCORES = HERE / "scores"
CACHE = Path(os.environ["E6_CACHE"])
DEVICE = torch.device(os.environ.get("E6_DEVICE", "mps"))
N = 14468
E1_ARCH = ARMS["E1"]["model"]
RESNET = dict(model="resnet18", pretrained=False)


def cached(size):
    path = CACHE / f"raw_{size}.u16"
    if path.exists():
        return path
    t0 = time.time()
    table = scored_table()
    assert len(table) == N
    tmp = path.with_suffix(".tmp")
    mm = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(N, 3, size, size))
    for i, r in enumerate(table.itertuples()):
        arr = np.array(Image.open(CELLS_DIR / r.file).convert("RGB"))
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
        mm[i] = np.round(t.numpy() * SCALE).astype(np.uint16)
    mm.flush()
    del mm
    tmp.rename(path)
    print(f"cached {size}px crops in {time.time() - t0:.0f}s", flush=True)
    return path


def score(ckpt, arch, size):
    m = models.build_model(arch).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = fast_norms(m).to(DEVICE)
    mm = np.memmap(cached(size), dtype=np.uint16, mode="r", shape=(N, 3, size, size))
    out = []
    with torch.inference_mode():
        for i in range(0, N, 64):
            x = torch.from_numpy(mm[i:i + 64].astype(np.float32) / SCALE).to(DEVICE)
            out.append(torch.exp(m(x))[:, 1].cpu().numpy())
    del m, mm
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(out)


def resnet_ckpt(r, f):
    for job in ("m3-rows-and-resnet", "danet-phase0-timing"):
        p = RUNS / job / "runs" / "E1r_resnet18" / f"grouped_r{r}f{f}_t0" / "best.pth"
        if p.exists():
            cfg = json.load(open(p.parent / "config.json"))["cfg"]
            assert cfg["epochs"] == 15 and cfg["img_size"] == 128, (p, cfg["epochs"])
            return p
    raise FileNotFoundError((r, f))


def unused():
    for r in (1, 2):
        for f in range(5):
            yield f"E1_r{r}f{f}_t0", RUNS / "e1-danet-grid" / "runs" / "E1_danet" / f"grouped_r{r}f{f}_t0" / "best.pth", E1_ARCH, 224
    for r in range(3):
        for f in range(5):
            yield f"E1r_r{r}f{f}_t0", resnet_ckpt(r, f), RESNET, 128


def seeds():
    root = RUNS / "e10-e11-e12-batch" / "runs" / "E1_danet"
    for f in (2, 3):
        for s in (1, 2, 3, 4):
            hits = list(root.glob(f"grouped_r0f{f}_t{s}*/best.pth"))
            assert len(hits) == 1, (f, s, hits)
            assert json.load(open(hits[0].parent / "config.json"))["cfg"]["train_seed"] == s
            yield f"E1_r0f{f}_t{s}", hits[0], E1_ARCH, 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["unused", "seeds"])
    a = ap.parse_args()
    SCORES.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    check = score(ckpt_path(ARMS["E1"], 0), E1_ARCH, 224)
    saved = np.load(E6B_SCORES / "A_E1_f0.npy")
    gap = float(np.abs(check - saved).max())
    print(f"reproduction: max abs p diff from E6b A_E1_f0 = {gap:.2e}", flush=True)
    assert gap < 1e-3, gap

    for name, ckpt, arch, size in (unused() if a.set == "unused" else seeds()):
        path = SCORES / f"{name}.npy"
        if path.exists():
            continue
        t0 = time.time()
        p = score(ckpt, arch, size)
        assert p.shape == (N,) and np.isfinite(p).all()
        tmp = SCORES / f"{name}.tmp.npy"
        np.save(tmp, p.astype(np.float32))
        tmp.rename(path)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
