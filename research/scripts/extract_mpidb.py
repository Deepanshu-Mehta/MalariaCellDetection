"""MP-IDB P. falciparum -> NIH-style cell crops (research/protocols/external_crop_protocol.md).

    arch -arm64 .venv/bin/python -m research.scripts.extract_mpidb

Writes research/data_external/mpidb_pf_cells/{Parasitized,Uninfected}/*.png and mpidb_cells.csv
(one row per segmented cell, kept or excluded, with the reason). Cellpose cyto3 weights are expected in
~/.cellpose/models/cyto3 (the cellpose.org server was down; mirror: huggingface.co/spaces/mouseland/cellpose).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_external" / "MP-IDB" / "Falciparum"
OUT = ROOT / "data_external" / "mpidb_pf_cells"
DIAMETER, MARGIN = 120, 4
AREA_RANGE = (0.5, 1.8)          # x median cell area of the image
PARASITE_INSIDE = 0.5            # share of a parasite component that must lie inside the cell
CLEAR_PX = 10                    # uninfected cells must be this far from any parasite
DARK_REL, DARK_FRAC = 0.5, 0.25  # leukocyte rule (cells without parasite overlap only): >25 % of the cell darker
                                 # than 0.5 x the image's median cell brightness (min RGB channel)


def segment(model, img):
    masks, _, _ = model.eval(img, diameter=DIAMETER, channels=[0, 0])
    return masks


def main():
    import shutil
    from cellpose import models
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = models.CellposeModel(pretrained_model=str(Path.home() / ".cellpose/models/cyto3"),
                                 gpu=dev.type != "cpu", device=dev)
    for cls in ["Parasitized", "Uninfected"]:
        shutil.rmtree(OUT / cls, ignore_errors=True)
        (OUT / cls).mkdir(parents=True)
    cache = OUT / "_masks"                            # cellpose output, so relabelling never re-segments
    cache.mkdir(parents=True, exist_ok=True)
    norm = lambda name: name.replace("-", "_")       # one gt file differs only by '-' vs '_' (…0013-R-T.jpg)
    gt_files = {norm(g.name): g for g in (SRC / "gt").glob("*.jpg")}
    rows = []
    files = sorted((SRC / "img").glob("*.jpg"))
    for n, f in enumerate(files):
        img = np.array(Image.open(f).convert("RGB"))
        gt = np.array(Image.open(gt_files[norm(f.name)]).convert("L")) > 127
        par_lab, n_par = ndi.label(gt)
        par_area = np.bincount(par_lab.ravel(), minlength=n_par + 1)
        near_par = ndi.binary_dilation(gt, iterations=CLEAR_PX)
        cf = cache / f"{f.stem}.npz"
        if cf.exists():
            masks = np.load(cf)["masks"]
        else:
            masks = segment(model, img)
            np.savez_compressed(cf, masks=masks)
        areas = np.bincount(masks.ravel())[1:]
        med = np.median(areas[areas > 0]) if areas.any() else 0
        mn = img.min(axis=2)
        dark_level = DARK_REL * np.median(mn[masks > 0]) if (masks > 0).any() else 0
        h, w = masks.shape
        for cid, sl in enumerate(ndi.find_objects(masks), start=1):
            if sl is None:
                continue
            cell = masks[sl] == cid
            area = int(cell.sum())
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            inside = np.bincount(par_lab[sl][cell], minlength=n_par + 1)[1:]
            frac = inside / np.maximum(par_area[1:], 1)
            dark = float((mn[sl][cell] < dark_level).mean())
            reason, label = None, None
            if y0 == 0 or x0 == 0 or y1 == h or x1 == w:
                reason = "touches_border"
            elif not AREA_RANGE[0] * med <= area <= AREA_RANGE[1] * med:
                reason = "area_outlier"
            elif (frac >= PARASITE_INSIDE).any():
                label = "Parasitized"
            elif inside.sum() > 0 or near_par[sl][cell].any():
                reason = "ambiguous_overlap"
            elif dark > DARK_FRAC:
                reason = "leukocyte_like"
            else:
                label = "Uninfected"
            name = f"{f.stem}_cell{cid}.png"
            if label:
                ya, yb = max(y0 - MARGIN, 0), min(y1 + MARGIN, h)
                xa, xb = max(x0 - MARGIN, 0), min(x1 + MARGIN, w)
                keep = masks[ya:yb, xa:xb] == cid
                crop = img[ya:yb, xa:xb] * keep[..., None]
                Image.fromarray(crop.astype(np.uint8)).save(OUT / label / name)
            rows.append(dict(image=f.name, cell=cid, area=area, dark_frac=round(dark, 3),
                             n_parasites=int((frac >= PARASITE_INSIDE).sum()), label=label, excluded=reason,
                             file=f"{label}/{name}" if label else None))
        print(f"{n + 1}/{len(files)} {f.name}: {len(areas)} cells", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "mpidb_cells.csv", index=False)
    covered = t.n_parasites.sum()
    print(t.label.value_counts(dropna=False).to_string())
    print(t.excluded.value_counts().to_string())
    print(f"parasite components assigned to a kept cell: {covered}")


if __name__ == "__main__":
    main()
