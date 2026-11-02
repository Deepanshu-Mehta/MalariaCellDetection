"""Rehearse the box-to-crop rules on MP-IDB, where Cellpose masks give a reference. No BBBC041 data is read.

    arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.rehearse_mpidb --rule ellipse

Writes data_external/mpidb_<rule>_rehearsal/<crop id>.png, one per scored MP-IDB cell, plus rehearsal_<rule>.csv here.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats as st

from research.malaria import dark_spot

from .crops import crop

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
MP = R / "data_external/mpidb_pf_cells"
SRC = R / "data_external/MP-IDB/Falciparum/img"
MARGIN = 4


def out_dir(rule):
    return R / f"data_external/mpidb_{rule}_rehearsal"


def one_image(item):
    (name, g), rule = item
    img = np.array(Image.open(SRC / name).convert("RGB"))
    masks = np.load(MP / "_masks" / f"{Path(name).stem}.npz")["masks"]
    objs = ndi.find_objects(masks)
    rows = []
    for r in g.itertuples():
        sl = objs[r.cell - 1]
        r0, r1 = max(sl[0].start - MARGIN, 0), min(sl[0].stop + MARGIN, img.shape[0])
        c0, c1 = max(sl[1].start - MARGIN, 0), min(sl[1].stop + MARGIN, img.shape[1])
        out, mask, failed = crop(img, r0, c0, r1, c1, rule)
        ref = masks[r0:r1, c0:c1] == r.cell
        iou = (mask & ref).sum() / max((mask | ref).sum(), 1)
        Image.fromarray(out.astype(np.uint8)).save(out_dir(rule) / f"{r.crop_id}.png")
        rows.append(dict(crop_id=r.crop_id, iou=iou, failed=failed, mask_frac=mask.mean(), f_otsu=dark_spot.score(out.astype(np.uint8))))
    return rows


def auc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return (st.rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rates(y, s, t=dark_spot.THRESHOLD):
    y, q = np.asarray(y).astype(bool), np.asarray(s) >= t
    se, sp = 100 * q[y].mean(), 100 * (~q[~y]).mean()
    return dict(BACC=(se + sp) / 2, SE=se, SP=sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, choices=["ellipse", "otsu"])
    rule = ap.parse_args().rule
    out_dir(rule).mkdir(exist_ok=True)
    t = pd.read_csv(MP / "mpidb_cells.csv")
    t = t[t.label.notna()].reset_index(drop=True)
    t["crop_id"] = [f"c{i:05d}" for i in range(len(t))]
    t["y"] = (t.label == "Parasitized").astype(int)
    with ProcessPoolExecutor(4) as ex:
        rows = [x for b in ex.map(one_image, [(g, rule) for g in t.groupby("image")], chunksize=4) for x in b]
    d = t.merge(pd.DataFrame(rows), on="crop_id")
    assert len(d) == len(t)
    d["f_cellpose"] = [dark_spot.score(np.array(Image.open(MP / f).convert("RGB"))) for f in d.file]
    d[["crop_id", "image", "cell", "file", "y", "iou", "failed", "mask_frac", "f_otsu", "f_cellpose"]].to_csv(HERE / f"rehearsal_{rule}.csv", index=False)
    fo, fc = np.nan_to_num(d.f_otsu), np.nan_to_num(d.f_cellpose)
    print(json.dumps(dict(n=len(d), nan=int(d.f_otsu.isna().sum()), failed_pct=100 * d.failed.mean(),
                          iou_quartiles=d.iou.quantile([.05, .25, .5, .75]).round(3).tolist(),
                          iou_by_class=d.groupby("y").iou.median().round(3).to_dict(),
                          cellpose=dict(AUC=auc(d.y, fc), **rates(d.y, fc)), otsu=dict(AUC=auc(d.y, fo), **rates(d.y, fo)),
                          score_corr=float(np.corrcoef(fo, fc)[0, 1])), indent=1, default=float))


if __name__ == "__main__":
    main()
