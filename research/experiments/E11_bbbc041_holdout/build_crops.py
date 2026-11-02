"""Build BBBC041 crops and tables from the annotation JSON. Runs no model and no feature.

    arch -arm64 .venv/bin/python -m research.experiments.E11_bbbc041_holdout.build_crops

Writes crops.csv (no labels), labels.csv and build_report.json here, and PNG crops under data_external/bbbc041_crops/<rule>/.
Exclusions follow the registered protocol and are applied before any score exists.
"""
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .crops import crop

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
SRC = R / "data_external/BBBC041_download/extracted/malaria"
OUT = R / "data_external/bbbc041_crops"
STAGES = ("ring", "trophozoite", "schizont", "gametocyte")
BORDER = 1


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def table():
    rows, images = [], []
    for jf, group in (("training.json", "png"), ("test.json", "jpg")):
        for e in json.load(open(SRC / jf)):
            path = e["image"]["pathname"].lstrip("/")
            H, W = e["image"]["shape"]["r"], e["image"]["shape"]["c"]
            images.append(dict(image=path, checksum=e["image"]["checksum"], H=H, W=W))
            boxes = [(o["bounding_box"]["minimum"]["r"], o["bounding_box"]["minimum"]["c"],
                      o["bounding_box"]["maximum"]["r"], o["bounding_box"]["maximum"]["c"], o["category"]) for o in e["objects"]]
            para = [b[:4] for b in boxes if b[4] in STAGES]
            for r0, c0, r1, c1, cat in boxes:
                if cat == "difficult":
                    why = "difficult"
                elif r0 <= BORDER or c0 <= BORDER or r1 >= H - BORDER or c1 >= W - BORDER:
                    why = "border"
                elif cat == "red blood cell" and any(intersects((r0, c0, r1, c1), p) for p in para):
                    why = "rbc_overlaps_parasite"
                else:
                    why = ""
                rows.append(dict(image=path, group=group, stage=cat, r0=r0, c0=c0, r1=r1, c1=c1, excluded=why))
    return pd.DataFrame(rows), pd.DataFrame(images)


def one_image(item):
    image, g, checksum = item
    raw = (SRC / image).read_bytes()
    img = np.array(Image.open(SRC / image).convert("RGB"))
    ok = hashlib.md5(raw).hexdigest() == checksum or hashlib.sha256(raw).hexdigest() == checksum
    failed = 0
    for r in g.itertuples():
        for rule in ("ellipse", "otsu"):
            out, _, bad = crop(img, r.r0, r.c0, r.r1, r.c1, rule)
            failed += bad
            Image.fromarray(out.astype(np.uint8)).save(OUT / rule / f"{r.crop_id}.png")
    return image, ok, failed


def main():
    for rule in ("ellipse", "otsu"):
        (OUT / rule).mkdir(parents=True, exist_ok=True)
    d, images = table()
    keep = d[d.excluded == ""].reset_index(drop=True)
    keep["crop_id"] = [f"b{i:06d}" for i in range(len(keep))]
    keep["y"] = np.where(keep.stage.isin(STAGES), 1.0, np.where(keep.stage == "red blood cell", 0.0, np.nan))
    assert set(keep.stage[keep.y.isna()]) == {"leukocyte"}
    checks = dict(zip(images.image, images.checksum))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one_image, [(k, g, checks[k]) for k, g in keep.groupby("image")], chunksize=8))
    keep[["crop_id", "image", "group", "r0", "c0", "r1", "c1"]].to_csv(HERE / "crops.csv", index=False)
    keep[["crop_id", "image", "y", "group", "stage"]].to_csv(HERE / "labels.csv", index=False)
    report = dict(annotated_boxes=len(d), kept=len(keep), images=int(keep.image.nunique()),
                  exclusions=dict(Counter(d.excluded[d.excluded != ""])),
                  primary_parasitized=int((keep.y == 1).sum()), primary_uninfected=int((keep.y == 0).sum()),
                  leukocytes=int(keep.y.isna().sum()), prevalence_pct=round(100 * float((keep.y == 1).sum() / keep.y.notna().sum()), 3),
                  by_group=keep.groupby("group").y.agg(["size", "sum"]).to_dict("index"),
                  checksum_mismatch=int(sum(not r[1] for r in res)), otsu_mask_failed=int(sum(r[2] for r in res)),
                  png_files={r: len(list((OUT / r).glob("*.png"))) for r in ("ellipse", "otsu")})
    json.dump(report, open(HERE / "build_report.json", "w"), indent=1, default=float)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
