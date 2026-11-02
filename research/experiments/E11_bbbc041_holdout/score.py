"""Score crops with every frozen candidate. Reads no labels. Resume-safe: one .npy per member.

    E6_CACHE=<dir> caffeinate -i arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.score \
        --table <crops.csv> --crops <dir> --name <run name> [--members feature danet resnet probes]

`--table` needs crop_id only. Output: scores/<name>/<member>.npy and scores/<name>/manifest.json with every hash.
"""
import argparse
import gc
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import checkpoint, fast_norms
from research.experiments.E9_foundation_model_probes.backbones import REGISTRY
from research.malaria import dark_spot, models

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
DEVICE = torch.device("mps")
CACHE = Path(os.environ.get("E6_CACHE", HERE / "_cache"))
RESNET_RUNS = RESEARCH / "runs_kaggle" / "e10-e11-e12-batch" / "runs" / "E11_resnet18_colourtransfer"
MANIFEST = json.load(open(HERE / "frozen_manifest.json"))
PROBES = ("dinov2_s", "dinobloom_s")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_one(p):
    return dark_spot.score(np.array(Image.open(p).convert("RGB")))


def cached(paths, size, name):
    """uint8 CHW memmap, resized as in E5/E6. 8 bits, because 80,000 crops at 16 bits would not fit on disk."""
    path = CACHE / f"{name}_{size}.u8"
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = path.with_suffix(".tmp")
    mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(len(paths), 3, size, size))

    def one(i):
        t = torch.from_numpy(np.array(Image.open(paths[i]).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
        mm[i] = np.round(t.numpy() * 255).astype(np.uint8)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, range(len(paths))))
    mm.flush()
    del mm
    tmp.rename(path)
    print(f"cached {len(paths)} crops at {size}px in {time.time() - t0:.0f}s", flush=True)
    return path


def batches(path, n, size, bs=64):
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, 3, size, size))
    for i in range(0, n, bs):
        yield torch.from_numpy(mm[i:i + bs].astype(np.float32) / 255.0)


def release():
    gc.collect()
    torch.mps.empty_cache()


def model_scores(ckpt, arch, path, n, size):
    m = models.build_model(arch).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = fast_norms(m).to(DEVICE)
    out = []
    with torch.inference_mode():
        for x in batches(path, n, size):
            out.append(torch.exp(m(x.to(DEVICE)))[:, 1].cpu().numpy())
    del m
    release()
    return np.concatenate(out)


def danet_members():
    for f in range(5):
        yield f"cand2_E1_f{f}", ckpt_path(ARMS["E1"], f), ARMS["E1"]["model"], MANIFEST["candidate_2"][f"E1_f{f}"]
    for a in ARMS:
        for f in range(5):
            yield f"cand3_{a}_f{f}", checkpoint("B", a, f), ARMS[a]["model"], MANIFEST["candidate_3"][f"{a}_f{f}"]


def resnet_members():
    for f in range(5):
        hits = list(RESNET_RUNS.glob(f"grouped_r0f{f}_t0*/best.pth"))
        assert len(hits) == 1, (f, hits)
        yield f"cand3_resnet_f{f}", hits[0], dict(model="resnet18", pretrained=False), MANIFEST["candidate_3_resnet18"][f"f{f}"]


def fit_probe(name):
    path = HERE / "probes" / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    path.parent.mkdir(exist_ok=True)
    cells = pd.read_csv(RESEARCH / "splits" / "cells.csv", index_col="idx")
    x = np.load(RESEARCH / f"runs_kaggle/e9-foundation-probes/E9_outputs/features/{name}.npy")
    assert len(x) == len(cells)
    sc = StandardScaler().fit(x)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(x), cells.label.values)
    pickle.dump((sc, lr), open(path, "wb"))
    return sc, lr


def probe_scores(name, path, n):
    sc, lr = fit_probe(name)
    embed, _ = REGISTRY[name](DEVICE)
    feats = [embed(x) for x in batches(path, n, 224)]
    del embed
    release()
    return lr.predict_proba(sc.transform(np.concatenate(feats)))[:, 1]


def save(out, name, p, n):
    p = np.asarray(p, dtype=np.float32)
    assert p.shape == (n,)
    tmp = out / f"{name}.tmp.npy"
    np.save(tmp, p)
    tmp.rename(out / f"{name}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="+", default=["feature", "danet", "resnet", "probes"])
    a = ap.parse_args()
    t = pd.read_csv(a.table, usecols=["crop_id"])
    assert t.crop_id.is_unique
    paths = [str(Path(a.crops) / f"{c}.png") for c in t.crop_id]
    n = len(paths)
    out = HERE / "scores" / a.name
    out.mkdir(parents=True, exist_ok=True)
    assert sha256(RESEARCH / "malaria" / "dark_spot.py") == MANIFEST["dark_spot_py"], "feature code changed"

    if "feature" in a.members and not (out / "feature.npy").exists():
        with ProcessPoolExecutor(4) as ex:
            save(out, "feature", list(ex.map(feature_one, paths, chunksize=200)), n)
        print("feature done", flush=True)

    for group, make, size in (("danet", danet_members, 224), ("resnet", resnet_members, 128)):
        if group not in a.members:
            continue
        for name, ckpt, arch, want in make():
            if (out / f"{name}.npy").exists():
                continue
            assert sha256(ckpt) == want, f"checkpoint changed: {name}"
            t0 = time.time()
            save(out, name, model_scores(ckpt, arch, cached(paths, size, a.name), n, size), n)
            print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    if "probes" in a.members:
        for name in PROBES:
            if not (out / f"probe_{name}.npy").exists():
                t0 = time.time()
                save(out, f"probe_{name}", probe_scores(name, cached(paths, 224, a.name), n), n)
                print(f"probe {name}: {time.time() - t0:.0f}s", flush=True)

    files = sorted(out.glob("*.npy"))
    nan = {f.stem: float(np.isnan(np.load(f)).mean()) for f in files}
    ids = hashlib.sha256(",".join(t.crop_id).encode()).hexdigest()
    json.dump(dict(table_sha256=sha256(a.table), crop_ids_sha256=ids, n=n, files={f.stem: sha256(f) for f in files}, nan_rate=nan),
              open(out / "manifest.json", "w"), indent=1)
    assert max(nan.values()) <= .005, "NaN rate above .5%"
    print(f"{len(files)} score files hashed, max NaN rate {max(nan.values()):.4f}", flush=True)


if __name__ == "__main__":
    main()
